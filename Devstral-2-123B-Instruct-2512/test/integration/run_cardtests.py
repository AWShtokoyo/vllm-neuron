# SPDX-License-Identifier: Apache-2.0
"""Card-faithful validation: run the HF model-card's non-tool Test prompts under FULL FP8
(qkv,o_proj,mlp via the DMA-transpose+scale knob) and check the model produces the card's
expected CONTENT (Mistral's bar is qualitative correctness for an FP8 model, NOT bf16-token-match).

Uses the checkpoint's real CHAT_SYSTEM_PROMPT.txt + the Mistral chat template (llm.chat).
Non-tool tests only (3 self-identify, 4 web-server, a factual recall set) so judging is by
content, not tool-call parsing. Run the SAME prompts under bf16 and full-fp8 to compare.

Usage:
  python run_cardtests.py --quant bf16            --prefill-bucket 2048 --out results/card_bf16.json
  python run_cardtests.py --quant fp8:qkv,o_proj,mlp --prefill-bucket 2048 --out results/card_fp8.json
"""
import os, json, argparse, datetime

CKPT = "/opt/nvme/devstral-ckpt"

def load_system_prompt():
    with open(f"{CKPT}/CHAT_SYSTEM_PROMPT.txt") as f:
        sp = f.read()
    today = "2025-12-09"; yesterday = "2025-12-08"
    return sp.replace("{today}", today).replace("{yesterday}", yesterday)

# Card non-tool tests + a factual-recall probe (where fp8 errors showed up before).
USER_PROMPTS = [
    # Test 5 (small talk / self-identify)
    "Who are you ? Who made you and what day is it ?",
    # Test 4 (tech chatting)
    "How would you develop a web server if you couldn't use JS and your team doesn't like PHP.",
    # Test 3 spirit (factual recall + code): the model should answer correctly
    "Write a Python function that returns the name of the capital of Japan as a string.",
    # Direct factual recall (these are where per-tensor fp8 flipped greedy argmax)
    "Answer in one word. What is the capital of Japan?",
    "Answer in one word. What is the chemical symbol for gold?",
    "Answer in one word. Who painted the Mona Lisa?",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-checkpoint", default=CKPT)
    ap.add_argument("--tensor-parallel-size", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--prefill-bucket", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--quant", default="bf16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or f"/opt/nvme/card_{args.quant.replace(':','_').replace(',','+')}.json"

    from vllm import LLM, SamplingParams
    bucket = args.prefill_bucket
    n_seqs = int(os.environ.get("FP8_MAX_NUM_SEQS", "2"))
    neuron_config = {
        # greedy on-device sampling -> deterministic, reproducible (TEMP=0.15 is near-greedy)
        "on_device_sampling_config": {"all_greedy": "true"},
        "num_batched_tokens_buckets": [bucket],
        "num_seqs_buckets": [n_seqs],
    }
    if args.quant and args.quant != "bf16":
        neuron_config["quantization"] = args.quant

    llm = LLM(
        model=args.model_checkpoint, max_model_len=args.max_model_len,
        config_format="hf", tokenizer_mode="mistral", load_format="safetensors",
        max_num_seqs=n_seqs, max_num_batched_tokens=bucket, enable_prefix_caching=False,
        tensor_parallel_size=args.tensor_parallel_size,
        hf_overrides={"quantization_config": {}},
        additional_config={"neuron_config": neuron_config},
    )
    sp = load_system_prompt()
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)

    convs = [[{"role": "system", "content": sp}, {"role": "user", "content": u}] for u in USER_PROMPTS]
    outs = llm.chat(convs, sampling)

    records = []
    for u, o in zip(USER_PROMPTS, outs):
        txt = o.outputs[0].text
        records.append({"prompt": u, "text": txt})
        print(f"\n===== PROMPT: {u!r}\n{txt[:600]}")
    with open(out_path, "w") as f:
        json.dump({"quant": args.quant, "records": records}, f, indent=2)
    print(f"\nwrote {out_path}")

if __name__ == "__main__":
    main()
