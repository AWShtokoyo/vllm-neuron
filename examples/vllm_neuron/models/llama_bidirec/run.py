# SPDX-License-Identifier: Apache-2.0
"""Offline embedding inference for nvidia/llama-embed-nemotron-8b.

This is a pooling (embedding) model: prefill-only, no decode. It is loaded with
runner="pooling" and produces one 4096-dim L2-normalized embedding per input via
llm.embed().

At TP=1, export VLLM_NEURON_FORCE_LNC1=1 before running. Drop it for TP>=2.
"""
import argparse

from vllm import LLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="nvidia/llama-embed-nemotron-8b",
        help="Path to the model checkpoint (or HF id)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size (1, 2, or 4). At TP=1 set VLLM_NEURON_FORCE_LNC1=1.",
    )
    args = parser.parse_args()

    # runner="pooling" selects the embedding path; num_batched_tokens_buckets
    # entries MUST be > 128. max_num_seqs > 1 packs multiple sequences per prefill.
    llm = LLM(
        model=args.model_checkpoint,
        runner="pooling",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=512,
        max_num_seqs=4,
        enable_prefix_caching=False,
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [256, 512],
            }
        },
    )

    prompts = [
        "The capital of France is Paris.",
        "Trainium is an AWS machine learning accelerator.",
        "Sentence embeddings power semantic search and retrieval.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    ]

    outputs = llm.embed(prompts)

    for prompt, out in zip(prompts, outputs):
        emb = out.outputs.embedding
        print(f"len={len(emb)}  first4={emb[:4]}  | {prompt[:48]!r}")


if __name__ == "__main__":
    main()
