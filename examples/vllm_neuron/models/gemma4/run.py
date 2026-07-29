# SPDX-License-Identifier: Apache-2.0
"""Offline text-generation example for Gemma 4 31B IT on Neuron.

Serves Google's Gemma 4 31B IT text-only under the transformers-5.x native TEXT
architecture ``Gemma4ForCausalLM`` (``model_type`` ``gemma4_text``) with the
vLLM offline API. The checkpoint declares the multimodal arch
``Gemma4ForConditionalGeneration``; ``vllm_neuron`` rewrites it to the text arch
at config registration. The model has
heterogeneous attention layers — sliding-window (head_dim=256) and global
(head_dim=512) — plus QK/V normalization, partial RoPE, GeGLU MLP, and logit
softcapping; these are all handled by the ``vllm_neuron.model.gemma4`` package.

This example defaults to the configuration verified on a single ``trn2.3xlarge``
(4 NeuronCores, TP=4): ``max_model_len=2048``, ``max_num_seqs=4``, and a capped
KV budget (set ``VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.08`` in the
environment) so the ~15 GiB/core sharded weights plus KV fit the 24 GiB/core
limit. Larger tensor-parallel sizes on ``trn2.48xlarge`` (e.g. ``--tensor-parallel-size 32``
with ``--max-model-len 4096``) are architecturally supported but were not
verified on this hardware. See ``docs/tutorials/tutorial-gemma4-31b.md``.

Gemma requires a leading ``<bos>`` (id 2) attention anchor. The checkpoint's
fast tokenizer does not add it for raw completion prompts, so this example
prepends it explicitly via ``TokensPrompt``; without it, greedy decoding
degenerates into a repetition loop.

Usage:
    python examples/vllm_neuron/models/gemma4/run.py \
        --model-checkpoint <path-to-checkpoint>/gemma-4-31B-it

    # Single-bucket (flat TTFT regardless of input size)
    python examples/vllm_neuron/models/gemma4/run.py \
        --model-checkpoint <path-to-checkpoint>/gemma-4-31B-it \
        --single-bucket
"""

import argparse
import os

os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "1200"
os.environ["VLLM_NEURON_COMPILATION_TIMEOUT"] = "1200"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

BOS_TOKEN_ID = 2  # Gemma <bos> attention anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="google/gemma-4-31B-it",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="TP degree. Gemma4 has 32 attention heads; valid values: "
        "1, 2, 4, 8, 16, 32. Default 4 (verified on trn2.3xlarge).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Max sequence length. Default 2048 (verified TP=4 config).",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help="Decode batch size. Default 4 (verified TP=4 config).",
    )
    parser.add_argument(
        "--single-bucket",
        action="store_true",
        help="Use a single prefill bucket equal to max-model-len instead of the "
        "multi-bucket schedule. Prefer this only when prompt length is fixed.",
    )
    args = parser.parse_args()

    # Multi-bucket lets each request use its smallest fitting NEFF; the last
    # bucket must equal max_num_batched_tokens.
    fixed = [b for b in [512, 1024, 2048, 4096] if b < args.max_model_len]
    token_buckets = [args.max_model_len] if args.single_bucket else fixed + [args.max_model_len]

    llm = LLM(
        model=args.model_checkpoint,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": token_buckets,
                "num_seqs_buckets": [args.max_num_seqs],
                "on_device_sampling_config": {
                    "all_greedy": True,
                },
            },
        },
    )

    tok = AutoTokenizer.from_pretrained(args.model_checkpoint)
    raw_prompts = [
        "The capital of France is",
        "I am gonna keep counting forever, 1 2 3 4 5 ",
        "Once upon a time, there was a ",
        "def fibonacci(n):",
    ]
    # Prepend <bos> explicitly (Gemma anchor) — the fast tokenizer strips it.
    prompts = [
        TokensPrompt(
            prompt_token_ids=[BOS_TOKEN_ID]
            + tok.encode(p, add_special_tokens=False)
        )
        for p in raw_prompts
    ]
    sampling_params = SamplingParams(max_tokens=50, temperature=0.0)

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(raw_prompts, outputs):
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print()


if __name__ == "__main__":
    main()
