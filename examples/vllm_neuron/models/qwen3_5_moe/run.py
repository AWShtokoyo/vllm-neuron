# SPDX-License-Identifier: Apache-2.0
"""Offline inference example for Qwen3.6-35B-A3B (qwen3_5_moe).

Hybrid MoE: 30 GatedDeltaNet linear-attention layers + 10 full-attention layers
(full_attention_interval=4) over 40 layers, 256-expert top-8 routing + a shared
expert, keeping the IsHybrid / HasInnerState / HybridKVSpec / MambaSpec
unified-cache attention stack. Unlike the sibling qwen3_5_dense port, the FFN is
sparse, which is why this one also configures expert parallelism.

      python examples/vllm_neuron/models/qwen3_5_moe/run.py \
      --model-checkpoint /path/to/Qwen3.6-35B-A3B

Sharding: the model has 16 attention heads, so tensor_parallel_size must divide
16. Expert parallelism subdivides the TP group for the MoE only; ep_degree ==
tensor_parallel_size means pure EP for the FFN (tp_sub = world / ep = 1).

Expert parallelism does NOT change how many bytes of expert weights a rank holds:
at ep_degree 4 a rank holds 64 experts at the full intermediate width, and at
ep_degree 1 it holds all 256 with the intermediate sharded across the four ranks
instead, which is the same product. What it changes is decode: without EP the
kernel loads only the experts a small batch actually selected, which measures 2.49x
the output throughput at one request in flight and 1.63x at four, while EP=4 wins
first-token latency at every concurrency and overtakes on total throughput at 16
concurrent requests (by 5.6%). Both sides of each of those ratios were measured at the
same MoE prefill block_size, so they are not confounded by that knob.

The default below is therefore TP=4 with **no expert parallelism** — the recipe the
bundle README calls the default. Pass --ep-degree 4 for the high-concurrency recipe.

Runs OUT OF THE BOX: the validated recipe flags are applied via os.environ.setdefault()
below, so no `source env.sh` / manual exports are required. An explicit export still
overrides (setdefault only fills unset vars). The GatedDeltaNet prefill kernel defaults
to the chunk-group form (`VLLM_GDN_PREFILL` unset == `grouped`), the fastest of the
three; `parallel` and `sequential` are the earlier kernels, kept for comparison.
Full recipe: the Qwen3.6-35B-A3B bundle README (single source of truth).
"""
import argparse
import os

# Validated recipe flags (default-OFF in the model, REQUIRED for the correct path). Baked in via
# setdefault so `python run.py` works with no env setup; an explicit export still wins. Must be set
# BEFORE importing vllm (read at model/platform import time).
os.environ.setdefault("VLLM_UNIFIED_KV_GATHER", "1")    # block-indexed .ap KV gather (else torch fallback)
# Serial compile/trace workers: parallel workers x this model's graphs x TP4 can
# host-OOM on trn2.3xlarge (seen as a kernel panic, and as a killed neuronx-cc).
# Remove on a larger-RAM host if compile is slow.
os.environ.setdefault("VLLM_NEURON_PARALLEL_COMPILE_WORKERS", "1")
os.environ.setdefault("VLLM_NEURON_PARALLEL_TRACE_WORKERS", "1")
# EFA affinity is a CPU perf optimization, not a correctness requirement; the
# smaller trn2 instances (e.g. 3xlarge) have no EFA device, so skip it there.
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
# REQUIRED for BF16, not a tuning knob: 18.12 GiB/core of BF16 weights leaves ~5.9 GiB,
# and the default KV budget cap (0.30 of the GMU-scaled total) takes more than that for
# the KV pool, so the NEFF scratchpad does not fit and load fails with NRT_RESOURCE --
# at max-model-len 256 exactly as at 2048. Without this line "python run.py" does not
# start. Omit it when serving the FP8 checkpoint: FP8 frees ~7.5 GiB/core and the
# default cap is what makes max-model-len 8192 reachable.
os.environ.setdefault("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "0.08")

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="Qwen/Qwen3.6-35B-A3B",
        help="Path to the model checkpoint (or an HF model id).",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--ep-degree", type=int, default=1,
                        help="Expert-parallel degree; must divide tensor_parallel_size. "
                             "The default 1 is the documented default recipe: without EP the "
                             "decode kernel loads only the experts a small batch selected, "
                             "which is 2.49x the output throughput at one request in flight. "
                             "Pass --ep-degree 4 for the high-concurrency recipe (ep == tp is "
                             "pure EP for the FFN, tp_sub = 1).")
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
        # Follow --ep-degree rather than forcing EP on: the documented default recipe has
        # no expert parallelism, and hardcoding True here made this example run a different
        # recipe from the one the README calls the default.
        enable_expert_parallel=args.ep_degree > 1,
        limit_mm_per_prompt={"image": 0, "video": 0},
        # Pass this explicitly to match the documented serve recipe. The port pins the GDN
        # recurrent state to bf16 in get_mamba_state_dtype() regardless of this value, so it
        # cannot change behaviour — it only makes the resolved configuration unambiguous.
        # The checkpoint's text_config declares mamba_ssm_dtype float32, which is what makes
        # stating it here worth the line.
        mamba_ssm_cache_dtype="bfloat16",
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
