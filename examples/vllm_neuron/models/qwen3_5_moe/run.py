# SPDX-License-Identifier: Apache-2.0
"""Offline inference example for Qwen3.6-35B-A3B (qwen3_5_moe).

Hybrid MoE: 30 GatedDeltaNet linear-attention layers + 10 full-attention layers,
256-expert top-8 + shared expert, keeping the IsHybrid / HasInnerState /
HybridKVSpec / MambaSpec unified-cache attention stack.

EP config: tensor_parallel_size = 8 (= world on a single node),
enable_expert_parallel, ep_degree = 8 (-> tp_sub = world/ep = 1, pure EP,
32 experts/rank). The model has 16 attention heads, so tensor_parallel_size
MUST divide 16 (max valid = 8); EP subdivides the TP group for MoE.

      python examples/vllm_neuron/models/qwen3_5_moe/run.py \
      --model-checkpoint /path/to/Qwen3.6-35B-A3B

Runs OUT OF THE BOX: the validated recipe flags are applied via os.environ.setdefault()
below, so no `source env.sh` / manual exports are required. An explicit export still
overrides (setdefault only fills unset vars). For seq >= 512 the bounded-graph GDN prefill
(VLLM_GDN_SEQ_NKI=1) is REQUIRED — the unsegmented monolithic prefill trips a scatter/gather
OOB -> NaN. Full recipe: the Qwen3.6-35B-A3B bundle README (single source of truth).
"""
import argparse
import os

# Validated recipe flags (default-OFF in the model, REQUIRED for the correct path). Baked in via
# setdefault so `python run.py` works with no env setup; an explicit export still wins. Must be set
# BEFORE importing vllm (read at model/platform import time).
os.environ.setdefault("VLLM_GDN_SEQ_NKI", "1")          # bounded-graph GDN prefill (seq>=512)
os.environ.setdefault("VLLM_UNIFIED_KV_GATHER", "1")    # block-indexed .ap KV gather (else torch fallback)
os.environ.setdefault("VLLM_MOE_TKG_ROUTER_FP32", "1")  # fp32 decode router (correct top-8)

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="Qwen/Qwen3.6-35B-A3B",
        help="Path to the model checkpoint (pre-staged on FSx / local nvme).",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--ep-degree", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=1,
                        help="Max concurrent sequences. bs=1 is the safe/validated "
                             "path; >1 exercises batch>1 decode (slot-bounds unverified).")
    parser.add_argument("--prompts", type=str, default=None,
                        help="'@@'-separated prompt list overriding the default 4.")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    # Text-first: set all multimodal per-prompt limits to 0 so the runner does
    # not require a vision_neuron_config (vision tower not built). The arch
    # alias resolves to the text MoE decoder.
    llm = LLM(
        model=args.model_checkpoint,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        # MoE text-only arch alias (registered to qwen3_5_moe); NOT the dense
        # Qwen3_5ForCausalLM, which would load dense MLP weights.
        hf_overrides={"architectures": ["Qwen3_5MoeForCausalLM"]},
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "ep_degree": args.ep_degree,
                "num_batched_tokens_buckets": [args.max_model_len],
                "num_seqs_buckets": [args.max_num_seqs],
                "on_device_sampling_config": {"all_greedy": "true"},
            }
        },
    )

    try:
        print("MM_DETECT is_multimodal_model=",
              llm.llm_engine.model_config.is_multimodal_model)
    except Exception as e:  # pragma: no cover
        print("MM_DETECT probe failed:", e)

    if args.prompts:
        prompts = args.prompts.split("@@")
    else:
        prompts = [
            "The capital of France is ",
            "I am gonna keep counting forever, 1 2 3 4 5 ",
            "Once upon a time, there was a ",
            "def fibonacci(n):",
        ]
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)

    outputs = llm.generate(prompts, sampling_params)
    for o in outputs:
        print(repr(o.prompt), "->", repr(o.outputs[0].text))


if __name__ == "__main__":
    main()
