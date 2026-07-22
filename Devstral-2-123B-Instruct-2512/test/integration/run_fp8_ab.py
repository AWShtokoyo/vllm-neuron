# SPDX-License-Identifier: Apache-2.0
"""On-device A/B harness for FP8-native vs BF16-dequant dense projections.

Usage (from a trn2 host, venv active, NEURON_PLATFORM_TARGET_OVERRIDE=trn2):

    python run_fp8_ab.py --quant bf16            # baseline (flag off)
    python run_fp8_ab.py --quant fp8:qkv         # per-family bring-up
    python run_fp8_ab.py --quant fp8:qkv,o_proj
    python run_fp8_ab.py --quant fp8             # all families

Writes greedy continuations + per-prompt logprobs to a JSON file so successive
runs can be diffed (greedy top-1 match + perplexity delta) without re-launching.
"""

import argparse
import json
import math
import os

from vllm import LLM, SamplingParams

# Fixed prompt set (matches run.py: counting / factual / creative / code).
PROMPTS = [
    "I am gonna keep counting forever, 1 2 3 4 5 ",
    "The capital of France is ",
    "Once upon a time, there was a ",
    "def fibonacci(n):",
]

# Larger diverse bank for first-token (cascade-free) faithfulness measurement.
# Greedy decoding cascades after the first divergence, so cumulative top-1 match
# across a long continuation under-reports faithfulness; the first generated
# token on each independent prompt is the unbiased per-step agreement signal.
PROMPT_BANK = [
    "The capital of France is",
    "The capital of Japan is",
    "Water is made of hydrogen and",
    "The opposite of hot is",
    "2 + 2 =",
    "The first president of the United States was",
    "The largest planet in the solar system is",
    "The chemical symbol for gold is",
    "Roses are red, violets are",
    "To be or not to be, that is the",
    "The speed of light is approximately",
    "The author of Romeo and Juliet is",
    "The square root of 16 is",
    "The currency of Japan is the",
    "An apple a day keeps the",
    "The Great Wall is located in",
    "Photosynthesis occurs in the",
    "The freezing point of water is",
    "The tallest mountain on Earth is",
    "A group of lions is called a",
    "The programming language Python was created by",
    "The sun rises in the",
    "DNA stands for",
    "The capital of Italy is",
    "Three plus five equals",
    "The Mona Lisa was painted by",
    "The human body has 206",
    "The largest ocean on Earth is the",
    "E equals m c",
    "The boiling point of water in Celsius is",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="/opt/nvme/devstral-ckpt",
        help="Local checkpoint dir (HF format) or HF model ID.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--prefill-bucket",
        type=int,
        default=2048,
        help="num_batched_tokens bucket (== max_num_batched_tokens). Lower it to "
        "shrink per-tile SBUF for the FP8 STATIC qkv kernel during bring-up.",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="bf16",
        help="bf16 (off) | fp8 | fp8:qkv | fp8:qkv,o_proj | qkv,mlp | ...",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default derived from --quant).",
    )
    parser.add_argument(
        "--bank",
        action="store_true",
        help="Use the large PROMPT_BANK and generate 1 token each — cascade-free "
        "first-token agreement (the unbiased per-step faithfulness metric).",
    )
    args = parser.parse_args()

    out_path = args.out or f"/opt/nvme/fp8_ab_{args.quant.replace(':', '_').replace(',', '+')}.json"

    bucket = args.prefill_bucket
    # max_num_seqs is env-overridable (FP8_MAX_NUM_SEQS) to free SBUF headroom for the
    # bf16-hidden #1-bisection knob, which doubles the MLP hidden-tile footprint.
    n_seqs = int(os.environ.get("FP8_MAX_NUM_SEQS", "2"))
    neuron_config = {
        "on_device_sampling_config": {"all_greedy": "true"},
        "num_batched_tokens_buckets": [bucket],
        "num_seqs_buckets": [n_seqs],
    }
    # The quant knob maps to dense_fp8_static inside Ministral3Config.from_configs.
    # "bf16" / unset leaves the BF16-dequant path unchanged (flag off).
    if args.quant and args.quant != "bf16":
        neuron_config["quantization"] = args.quant

    llm = LLM(
        model=args.model_checkpoint,
        max_model_len=args.max_model_len,
        config_format="hf",
        # The HF tokenizer_config.json declares the transformers-5.x
        # ``TokenizersBackend`` class, absent in transformers 4.57.x. Force the
        # native Mistral tokenizer (tekken.json ships in the checkpoint).
        tokenizer_mode="mistral",
        load_format="safetensors",
        max_num_seqs=n_seqs,
        max_num_batched_tokens=bucket,
        enable_prefix_caching=False,
        tensor_parallel_size=args.tensor_parallel_size,
        hf_overrides={"quantization_config": {}},
        additional_config={"neuron_config": neuron_config},
    )

    prompts = PROMPT_BANK if args.bank else PROMPTS
    max_tokens = 1 if args.bank else args.max_tokens
    # Request logprob of the sampled token so we can compute a perplexity proxy.
    sampling_params = SamplingParams(
        max_tokens=max_tokens, temperature=0.0, top_p=1.0, logprobs=0
    )
    outputs = llm.generate(prompts, sampling_params)

    records = []
    for o in outputs:
        comp = o.outputs[0]
        token_ids = list(comp.token_ids)
        # Mean NLL over generated tokens (proxy perplexity), if logprobs present.
        nll = None
        if comp.logprobs:
            lps = []
            for tid, lp_dict in zip(token_ids, comp.logprobs):
                if lp_dict and tid in lp_dict:
                    lps.append(lp_dict[tid].logprob)
            if lps:
                nll = -sum(lps) / len(lps)
        records.append(
            {
                "prompt": o.prompt,
                "text": comp.text,
                "token_ids": token_ids,
                "mean_nll": nll,
                "ppl": (math.exp(nll) if nll is not None else None),
            }
        )

    result = {"quant": args.quant, "tp": args.tensor_parallel_size, "records": records}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n===== quant={args.quant} =====")
    for r in records:
        print(f"\nPROMPT: {r['prompt']!r}")
        print(f"  TEXT: {r['text']!r}")
        print(f"  first8 tok: {r['token_ids'][:8]}")
        print(f"  ppl: {r['ppl']}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
