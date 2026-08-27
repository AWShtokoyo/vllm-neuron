# SPDX-License-Identifier: Apache-2.0
"""Offline inference example for Qwen3.6-27B (qwen3_5_dense).

Hybrid dense: 48 GatedDeltaNet linear-attention layers + 16 full-attention
layers (full_attention_interval=4) over 64 layers, keeping the IsHybrid /
HasInnerState / HybridKVSpec / MambaSpec unified-cache attention stack. Unlike
the sibling qwen3_5_moe port, the FFN is a plain SwiGLU MLP (no experts, no EP).

      python examples/vllm_neuron/models/qwen3_5_dense/run.py \
      --model-checkpoint /path/to/Qwen3.6-27B

Runs OUT OF THE BOX: the validated recipe flags are applied via os.environ.setdefault()
below, so no `source env.sh` / manual exports are required. An explicit export still
overrides (setdefault only fills unset vars). The GatedDeltaNet prefill kernel defaults to
the numerically faithful sequential one; set VLLM_GDN_PREFILL=parallel for markedly lower
prefill latency at the cost of higher logit error.
"""
import argparse
import os

# Validated recipe flags (default-OFF in the model, REQUIRED for the correct path). Baked in via
# setdefault so `python run.py` works with no env setup; an explicit export still wins. Must be set
# BEFORE importing vllm (read at model/platform import time).
os.environ.setdefault("VLLM_UNIFIED_KV_GATHER", "1")    # block-indexed .ap KV gather (else torch fallback)
# TP=4 fused-QKV SBUF workaround: the gated-QKV megakernel (per-rank width 3584 at
# TP=4) overruns SBUF by ~8%; take the equivalent matmul path on trn2.3xlarge. At
# TP>=8 (48xlarge) the kernel fits — unset this there to keep the fused kernel.
os.environ.setdefault("VLLM_QWEN_QKV_MATMUL", "1")
# Serial compile/trace workers: TP4 x 9 graphs of parallel compile can host-OOM
# on trn2.3xlarge; force serial. Remove on a larger-RAM host if compile is slow.
os.environ.setdefault("VLLM_NEURON_PARALLEL_COMPILE_WORKERS", "1")
os.environ.setdefault("VLLM_NEURON_PARALLEL_TRACE_WORKERS", "1")
# EFA affinity is a CPU perf optimization, not a correctness requirement; the
# smaller trn2 instances (e.g. 3xlarge) have no EFA device, so skip it there.
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint. Required: there is no safe default, since which "
             "checkpoint is loaded also decides whether the fp8 or the bf16 path is taken.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=1,
                        help="Max concurrent sequences. bs=1 is the safe/validated "
                             "path; >1 exercises batch>1 decode (slot-bounds unverified).")
    parser.add_argument("--prompts", type=str, default=None,
                        help="'@@'-separated prompt list overriding the default 4.")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    # SBUF note (TP=4): the fused-QKV prefill megakernel (NF.qkv_proj) keeps the
    # per-rank weight [hidden=5120, qkv=3584] (gated Q 3072 + k 256 + v 256 at TP=4)
    # SBUF-resident and overruns the budget by ~8% (weight-bound, independent of
    # prefill length). VLLM_QWEN_QKV_MATMUL=1 (set above) routes QKV through an
    # equivalent matmul with no such ceiling. At TP=8 (48xlarge) the per-rank width
    # halves to 1792 and the fused kernel fits — unset the flag there.

    # Text-first: set all multimodal per-prompt limits to 0 so the runner does
    # not require a vision_neuron_config (vision tower not built). The arch
    # alias resolves to the text dense decoder.
    llm = LLM(
        model=args.model_checkpoint,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"image": 0, "video": 0},
        # Pass this explicitly to match the documented serve recipe. The port pins the GDN
        # recurrent state to bf16 in get_mamba_state_dtype() regardless of this value, so it
        # cannot change behaviour — it only makes the resolved configuration unambiguous.
        mamba_ssm_cache_dtype="bfloat16",
        # Dense text-only arch alias (registered to qwen3_5_dense); NOT the MoE
        # Qwen3_5MoeForCausalLM, which would load 256-expert weights.
        hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]},
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
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
