# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.3 Offline Inference Example
==================================

Runs offline inference with the GLM-5.3 MoE model on Neuron (trn2.48xlarge),
in the configuration this port was actually brought up and verified in. See
``GLM-5.3/README.md`` for the architecture, the FP8-only rationale and the
verification scope.

Three things here are requirements, not preferences, and the model will not start
without them:

* **An FP8 checkpoint.** ``zai-org/GLM-5.3``, not ``zai-org/GLM-5.3-BF16``. BF16
  weights do not fit a trn2.48xlarge -- see "Why FP8 only" in the README.
* **``quantization: "fp8_per_channel"``.** Reading an FP8 checkpoint is not the same as
  keeping FP8 weights on HBM; a mode that dequantizes at load time has the BF16
  footprint and does not fit either.
* **``enable_prefix_caching=False``.** vLLM 0.24 turns automatic prefix caching on
  by default, and on this platform APC requires segmented prefill. At this
  ``max_model_len`` there is no segmentation, so leaving APC on fails at startup.

This example does NOT set ``kv_segment_size_buckets``, so it takes the single-shot
prefill path. That path has no DSA implementation, and ``GlmMoeDsaAttention`` refuses to run
rather than silently serving full attention while the config asks for sparse — so
``VLLM_GLM_DSA=1`` left over in the environment would make this example raise
``NotImplementedError``. It is unset below for exactly that reason: the flag is opt-in
per run, and an example is not the place to inherit it from a shell.

Usage:
    python run.py                       # max_model_len=128, the minimal bring-up
    python run.py --max-model-len 4096   # larger bucket, much longer cold compile
"""

import argparse
import os

from vllm import LLM, SamplingParams

os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")

# 🔴 Unset, not `setdefault`. This example uses single-shot prefill (no
# kv_segment_size_buckets), which has no DSA path, so a VLLM_GLM_DSA=1 inherited from the
# shell makes the forward raise. The README tells readers to export that flag to try DSA;
# without this line, following the README and then running the example fails.
os.environ.pop("VLLM_GLM_DSA", None)

# Ensure neuronx-cc is on PATH for child worker processes
import sys
venv_bin = os.path.dirname(sys.executable)
if venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")

# 🔴 Do NOT raise VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION or drop
# VLLM_NEURON_MIN_KV_BUDGET_GIB here. This example used to set them to "1.0" and "0",
# which removes the plugin's interim KV safety cap (defaults 0.30 and 1.0 GiB, see
# vllm_neuron/envs.py). The KV cache is sized from free HBM rather than from
# max_model_len, so without the cap it takes everything left after the weights --
# measured 128,992 tokens instead of 79,136 -- and the runtime then cannot allocate its
# own 141.7 MB. Every rank fails identically in initialize_kv_cache with
# `nrt_tensor_allocate status=4`, i.e. the example does not start at all. The capped
# default is also the condition every configuration in the README was verified under.
os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1800")
os.environ.setdefault("NEURON_CC_FLAGS", "--hbm-scratchpad-page-size=512")
os.environ.setdefault("NEURON_SCRATCHPAD_PAGE_SIZE", "512")

# On Neuron 2.32 the compile stack moved into the external libtorch-neuronx-lite
# package and these names changed. The old VLLM_NEURON_* spellings of the two
# below still work through a documented fallback, but the trace-worker variable's
# old name does NOT and is silently ignored, so use the 2.32 names throughout.
os.environ.setdefault("NEURON_LIBTORCH_COMPILATION_TIMEOUT", "3600")
# Serial trace. Parallel trace forks a child per graph, and after fork the
# parent's Neuron runtime enters NRT_STATE_CHILD, so each child has to swap the
# model's modules to the meta device -- memory-heavy enough to OOM the host on
# graphs this size.
os.environ.setdefault("NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS", "1")
# 🔴 While rank 0 compiles, the other 63 ranks wait in a barrier. If this expires
# first they exit, and the run dies at the finish line even though every graph
# compiled and cached successfully. The 3600 s default is not enough for this
# model; it has been observed firing on a compile that needed 4.63 h.
os.environ.setdefault("VLLM_NEURON_BARRIER_TIMEOUT", "86400")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="zai-org/GLM-5.3",
        help="Path to (or HF id of) an FP8 checkpoint. BF16 does not fit.",
    )
    parser.add_argument(
        "--ep-degree",
        type=int,
        default=16,
        help="Expert parallelism degree",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=128,
        help=(
            "Context length. 128 is the minimal bring-up configuration and the "
            "fastest first run. Larger values compile a larger prefill bucket and "
            "cost substantially more cold-compile time."
        ),
    )
    args = parser.parse_args()

    # GLM-5.3: 64 attention heads, 256 routed experts
    # EP config: world_size=64, ep_degree=16, tp_sub=4
    # Attention/Dense: sharded across full 64 ranks (1 head/rank)
    # MoE experts: 16 local, intermediate sharded by tp_sub=4 (512/rank)
    llm = LLM(
        model=args.model_checkpoint,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_model_len,
        tensor_parallel_size=64,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=False,
        additional_config={
            "neuron_config": {
                "ep_degree": args.ep_degree,
                "quantization": "fp8_per_channel",
                "num_batched_tokens_buckets": [args.max_model_len],
                "num_seqs_buckets": [1],
                # Greedy sampling on device. Matches the verified configuration;
                # SamplingParams(temperature=0) alone does not select it.
                "on_device_sampling_config": {"all_greedy": True},
            }
        },
    )

    sampling_params = SamplingParams(max_tokens=16, temperature=0.0, top_p=1.0)

    prompts = [
        "The capital of France is",
        "1 2 3 4 5 6 7 8",
        "def fibonacci(n):",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text!r}")
        print()


if __name__ == "__main__":
    main()
