# SPDX-License-Identifier: Apache-2.0
"""Offline text inference example for Devstral-2-123B (Ministral3) on Neuron.

Devstral-2-123B-Instruct-2512 is a dense GQA causal LM with a per-tensor static
FP8 (E4M3) checkpoint. This example runs text-only generation via the vLLM
offline API. The default path dequantizes the FP8 weights to BF16 at load; the
optional FP8-native path keeps the projections in FP8 (see ``--quantization``).

Note the model-specific flags: the checkpoint repository ships both HF and
Mistral-native formats, so ``config_format="hf"`` is required to avoid vLLM's
built-in ``MistralForCausalLM``; vLLM rejects the ``fp8`` quant method, so the
checkpoint's ``quantization_config`` is emptied via ``hf_overrides`` (the model's
loader auto-detects FP8 from the checkpoint in both paths).

Usage:
    python examples/vllm_neuron/models/ministral3/run.py \
        --model-checkpoint <path-to-checkpoint>/Devstral-2-123B-Instruct-2512

    # FP8-native (needs integration_nkilib.patch applied to nkilib; full FP8 also
    # needs NKILIB_MLP_BF16_XPOSE_SRC=1 in the environment)
    python examples/vllm_neuron/models/ministral3/run.py \
        --model-checkpoint <path>/Devstral-2-123B-Instruct-2512 \
        --quantization fp8:qkv,o_proj,mlp
"""

import argparse
import os

os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "1200"
os.environ["VLLM_NEURON_COMPILATION_TIMEOUT"] = "1200"

import vllm_neuron  # noqa: F401  (registers ministral3 + platform + AutoConfig)
from vllm import LLM, SamplingParams

TEXT_PROMPTS = [
    "I am gonna keep counting forever, 1 2 3 4 5 ",
    "The capital of France is ",
    "Once upon a time, there was a ",
    "def fibonacci(n):",
]

# Prefill bucket == max_model_len keeps prefill single-shot (the validated config).
MAX_MODEL_LEN = 2048


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="mistralai/Devstral-2-123B-Instruct-2512",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=8,
        help="Tensor-parallel degree; must divide Q=96 (8, 16, or 32).",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="bf16",
        help=(
            "Dense-projection precision: 'bf16' (default, FP8 dequantized at "
            "load), 'fp8:qkv,o_proj' (BF16-token-faithful), or "
            "'fp8:qkv,o_proj,mlp' (full FP8; also set NKILIB_MLP_BF16_XPOSE_SRC=1). "
            "FP8-native paths require integration_nkilib.patch applied to nkilib."
        ),
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="auto",
        choices=["auto", "fp8_e4m3"],
        help="KV cache dtype. fp8_e4m3 runs at scale=1.0 (bucket == max_model_len only).",
    )
    args = parser.parse_args()

    llm = LLM(
        model=args.model_checkpoint,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=4,
        max_num_batched_tokens=MAX_MODEL_LEN,  # must equal the prefill bucket
        config_format="hf",                    # avoid MistralForCausalLM override
        load_format="safetensors",
        enable_prefix_caching=False,           # stack has no APC
        kv_cache_dtype=args.kv_cache_dtype,
        hf_overrides={"quantization_config": {}},  # bypass vLLM fp8 rejection
        additional_config={
            "neuron_config": {
                "quantization": args.quantization,
                "num_batched_tokens_buckets": [MAX_MODEL_LEN],
                "num_seqs_buckets": [4],
                "on_device_sampling_config": {"all_greedy": True},
            }
        },
    )

    sampling_params = SamplingParams(max_tokens=32, temperature=0.0, top_p=1.0)
    outputs = llm.generate(TEXT_PROMPTS, sampling_params)

    for prompt, output in zip(TEXT_PROMPTS, outputs):
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print()


if __name__ == "__main__":
    main()
