# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.2 Offline Inference Example
==================================

Runs offline inference with the GLM-5.2 MoE model on Neuron (trn2.48xlarge).

Usage:
    NEURON_SKIP_EFA_AFFINITY=1 NXDI_SWITCH_CC=1 python run.py
"""

import argparse
import os

from vllm import LLM, SamplingParams

os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
os.environ.setdefault("NXDI_SWITCH_CC", "1")
# Ensure neuronx-cc is on PATH for child worker processes
import sys
venv_bin = os.path.dirname(sys.executable)
if venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")
os.environ.setdefault("VLLM_NEURON_MIN_KV_BUDGET_GIB", "0")
os.environ.setdefault("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "1.0")
os.environ["VLLM_NEURON_COMPILATION_TIMEOUT"] = "3600"
os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1800")
os.environ.setdefault("NEURON_CC_FLAGS", "--hbm-scratchpad-page-size=512")
os.environ.setdefault("NEURON_SCRATCHPAD_PAGE_SIZE", "512")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="zai-org/GLM-5.2",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--ep-degree",
        type=int,
        default=16,
        help="Expert parallelism degree",
    )
    args = parser.parse_args()

    # GLM-5.2: 64 attention heads, 256 routed experts
    # EP config: world_size=64, ep_degree=16, tp_sub=4
    # Attention/Dense: sharded across full 64 ranks (1 head/rank)
    # MoE experts: 16 local, intermediate sharded by tp_sub=4 (512/rank)
    llm = LLM(
        model=args.model_checkpoint,
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        tensor_parallel_size=64,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.92,
        additional_config={
            "neuron_config": {
                "ep_degree": args.ep_degree,
                "num_batched_tokens_buckets": [128],
                "num_seqs_buckets": [1],
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
