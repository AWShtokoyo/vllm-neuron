# SPDX-License-Identifier: Apache-2.0
"""Offline text inference example for Devstral-2-123B (Ministral3) on Neuron.

Devstral-2-123B-Instruct-2512 is a 123B dense-GQA causal LM shipped as a
per-tensor static FP8 (E4M3) checkpoint. This example runs the **FP8-native**
path at TP=8, which is the configuration this port validates: the dense
projections stay in FP8 and run FP8xFP8 static matmuls on Trainium2.

Two environment knobs are REQUIRED FOR CORRECTNESS, not for tuning. Omitting
either produces output that passes every automated guard the stack offers
(no asserts, no exceptions, plausible throughput) and is wrong, so this example
sets both via ``setdefault`` rather than relying on the caller:

* ``MINISTRAL3_FP8_LAYERS="mlp:0-1,3-87"`` — keeps layer 2's MLP in BF16. That
  layer's ``activation_scale`` is a 197x discontinuity against its neighbours,
  and Trainium2's e4m3 saturates at +/-240 while the checkpoint is calibrated
  for OCP +/-448.
* ``FORCE_MLP_KERNEL=cte`` — nkilib routes the MLP on ``batch x seq_len <= 96``;
  in decode ``seq_len == 1``, so ``num_seqs`` itself is compared against 96 and
  the TKG MLP is chosen, which has no static-FP8 path on this stack.

Verify both landed by grepping the log for the per-layer census:
``mlp=87/88 [0-1,3-87]``. ``mlp=88/88 all`` means the knob never reached the
workers -- discard that run.

Other model-specific flags: the checkpoint repository ships both HF and
Mistral-native formats, so ``config_format="hf"`` is required to avoid vLLM's
built-in ``MistralForCausalLM``; vLLM rejects the ``fp8`` quant method, so the
checkpoint's ``quantization_config`` is emptied via ``hf_overrides`` (the model's
loader auto-detects FP8 from the checkpoint itself).

See ``Devstral-2-123B-Instruct-2512/README.md`` for the full port documentation.

Usage:
    python examples/vllm_neuron/models/ministral3/run.py \
        --model-checkpoint <path-to-checkpoint>/Devstral-2-123B-Instruct-2512
"""

import argparse
import os

# Required for correctness on this model -- see the module docstring.
os.environ.setdefault("MINISTRAL3_FP8_LAYERS", "mlp:0-1,3-87")
os.environ.setdefault("FORCE_MLP_KERNEL", "cte")
# A single cold graph has been measured at ~1,800 s; give compilation room.
os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "1200")

import vllm_neuron  # noqa: F401  (registers ministral3 + platform + AutoConfig)
from vllm import LLM, SamplingParams

TEXT_PROMPTS = [
    "I am gonna keep counting forever, 1 2 3 4 5 ",
    "The capital of France is ",
    "Once upon a time, there was a ",
    "def fibonacci(n):",
]

MAX_MODEL_LEN = 2048
# The last entry becomes max_num_batched_tokens and must stay BELOW
# max_model_len, which is what keeps segmented prefill -- and therefore
# automatic prefix caching -- available.
PREFILL_BUCKETS = [128, 256, 512, 1024]
NUM_SEQS = 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Local path to the Devstral-2-123B-Instruct-2512 checkpoint.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=8,
        help="TP degree. 8 is the validated value (it divides Q=96).",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="fp8:qkv,o_proj,mlp",
        help="FP8-native families. The validated value keeps all three in FP8.",
    )
    args = parser.parse_args()

    llm = LLM(
        model=args.model_checkpoint,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=NUM_SEQS,
        max_num_batched_tokens=PREFILL_BUCKETS[-1],  # == largest prefill bucket
        config_format="hf",              # avoid the MistralForCausalLM override
        load_format="safetensors",
        tokenizer_mode="mistral",
        enable_prefix_caching=True,      # works on 2.32 with segmented prefill
        kv_cache_dtype="fp8_e4m3",
        hf_overrides={"quantization_config": {}},  # bypass vLLM fp8 rejection
        additional_config={
            "neuron_config": {
                "quantization": args.quantization,
                "num_batched_tokens_buckets": PREFILL_BUCKETS,
                "kv_segment_size_buckets": [512, 1024],
                "num_seqs_buckets": [NUM_SEQS],
                "fp8_packed_kv": True,
                "on_device_sampling_config": {"all_greedy": True},
            }
        },
    )

    sampling_params = SamplingParams(max_tokens=32, temperature=0.0, top_p=1.0)
    outputs = llm.generate(TEXT_PROMPTS, sampling_params)

    for prompt, output in zip(TEXT_PROMPTS, outputs):
        print(f"Prompt: {prompt!r}")
        print(f"Output: {output.outputs[0].text!r}\n")


if __name__ == "__main__":
    main()
