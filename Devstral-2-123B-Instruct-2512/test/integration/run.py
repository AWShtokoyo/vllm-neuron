# SPDX-License-Identifier: Apache-2.0
"""Sanity-check run script for Ministral3 (Devstral-2-123B-Instruct-2512).

This is a large (~123B) FP8 model. The FP8 checkpoint is dequantized to BF16 at
load time, so the served model is BF16 (~246GB). Use a TP degree that both fits
the weights and evenly divides the head counts (Q=96, KV=8) — e.g. TP=32 on a
trn2.48xlarge.

``hf_overrides={"quantization_config": {}}`` IS required: vLLM detects the HF
`quant_method="fp8"` and rejects it (`fp8 quantization is currently not
supported in neuron`). Emptying the config bypasses that check. The native
weight loaders still dequantize FP8→BF16 because they detect FP8 from the actual
checkpoint tensors (`*.weight_scale_inv`), not from the config.
"""

import argparse

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="mistralai/Devstral-2-123B-Instruct-2512",
        help="Path to the model checkpoint or HF model ID",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=32,
        help="TP degree. Must evenly divide Q heads (96) and fit weights.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=10240,
    )
    args = parser.parse_args()

    llm = LLM(
        model=args.model_checkpoint,
        max_model_len=args.max_model_len,
        # The repo ships BOTH HF format (config.json: Ministral3ForCausalLM) and
        # Mistral native format (params.json / consolidated-*.safetensors). Force
        # the HF parser, else vLLM auto-detects `params.json` and overrides the
        # architecture to `MistralForCausalLM` (which lacks our from_configs).
        config_format="hf",
        load_format="safetensors",
        max_num_seqs=2,
        # Last num_batched_tokens bucket must equal max_num_batched_tokens.
        max_num_batched_tokens=2048,
        # APC (automatic prefix caching) is not supported in this stack and
        # causes silent failures — must be disabled.
        enable_prefix_caching=False,
        tensor_parallel_size=args.tensor_parallel_size,
        # Empty out the FP8 quantization_config so vLLM does not reject the
        # unsupported `fp8` method; the native loaders still dequantize.
        hf_overrides={"quantization_config": {}},
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {
                    "all_greedy": "true",
                },
                "num_batched_tokens_buckets": [2048],
                "num_seqs_buckets": [2],
            }
        },
    )

    # Diverse prompts: counting, factual, creative, code.
    token_prompts = [
        "I am gonna keep counting forever, 1 2 3 4 5 ",
        "The capital of France is ",
        "Once upon a time, there was a ",
        "def fibonacci(n):",
    ]
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0, top_p=1.0)

    outputs = llm.generate(token_prompts, sampling_params)

    print(outputs)


if __name__ == "__main__":
    main()
