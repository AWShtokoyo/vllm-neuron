# SPDX-License-Identifier: Apache-2.0
"""Hybrid-pool scheduler livelock regression test for Qwen3.6-35B-A3B.

Reproduces the configuration that livelocked before the pool-aware admission
gate: a tight ``max_model_len`` with batch size > 1. At ``max_model_len=512``
each request reserves its full worst-case KV allocation, and the GatedDeltaNet
mamba page dominates (~800 blocks/request on device), so the hybrid unified
block pool saturates at only ~3 concurrent decodes. Historically a further
request could neither be admitted nor preempt — ``schedule()`` hid the running
decodes to make room for the waiting prefill, then dead-ended into empty
batches (KV pinned at ~88.8%, 0 tok/s), an unbounded livelock.

The fix (``NeuronScheduler._pool_admission_ok`` in ``can_schedule``) predicts
each prefill's worst-case block footprint via the KV manager's own coordinator
and defers admission when it would not fit the free pool, instead of stalling.
With the gate on (default) every request completes; the test asserts forward
progress (all prompts return non-empty output within the timeout). A livelock
is observable only as a timeout, so to see the pre-fix behaviour set
``VLLM_NEURON_POOL_ADMISSION_GATE=0`` and expect this test to time out.

Device root cause + fix A/B are recorded in the model recipe
(``docs/model-recipes/qwen3-6-moe.md``) and
``vllm_neuron/model/qwen3_5_moe/doc/RUN_CONFIGS.md``.

Requires a Trainium2 host (trn2.48xlarge) with 8 Neuron devices and the
checkpoint staged locally. Point QWEN3_5_MOE_CHECKPOINT at it:

    QWEN3_5_MOE_CHECKPOINT=/path/to/Qwen3.6-35B-A3B \
    python3 -m pytest test/integration/test_scheduler_livelock.py -v -s
"""

import logging
import os
import time

import pytest

logger = logging.getLogger(__name__)

# Point this at a locally-staged checkpoint; falls back to the HF id.
MODEL = os.environ.get("QWEN3_5_MOE_CHECKPOINT", "Qwen/Qwen3.6-35B-A3B")
TP = int(os.environ.get("LIVELOCK_HYBRID_TP", "8"))
EP = int(os.environ.get("LIVELOCK_HYBRID_EP", "8"))
# Tight max_model_len=512 + bs>1 is the pool-saturating (pre-fix failing) recipe.
# Segment/batched bucket <= 512 keeps the bounded-graph GDN prefill.
MAX_MODEL_LEN = int(os.environ.get("LIVELOCK_HYBRID_MAX_MODEL_LEN", "512"))
SEG = int(os.environ.get("LIVELOCK_HYBRID_SEG", "256"))
MAX_NUM_SEQS = int(os.environ.get("LIVELOCK_HYBRID_MAX_NUM_SEQS", "4"))
# Enough prompts to exceed pool capacity (~3 concurrent) so requests queue and
# the Step-3.2 "hide running to admit a waiting prefill" path is exercised.
NUM_PROMPTS = int(os.environ.get("LIVELOCK_HYBRID_NUM_PROMPTS", "24"))
# ~256 input tokens + 128 output tokens => a decode crosses the 256-token
# attention block boundary (matches the device repro in256/out128).
INPUT_WORDS = int(os.environ.get("LIVELOCK_HYBRID_INPUT_WORDS", "220"))
MAX_TOKENS = int(os.environ.get("LIVELOCK_HYBRID_MAX_TOKENS", "128"))
TIMEOUT = int(os.environ.get("LIVELOCK_HYBRID_TIMEOUT", "1800"))

# Text-only MoE: force the MoE arch (not the dense Qwen3_5ForCausalLM) and zero
# the multimodal limits so the vision tower is not built.
HF_OVERRIDES = {"architectures": ["Qwen3_5MoeForCausalLM"]}
LIMIT_MM = {"image": 0, "video": 0}


@pytest.mark.timeout(TIMEOUT)
def test_livelock_hybrid_pool_saturation():
    """Forward progress under hybrid unified-pool saturation (mml512 + bs>1)."""
    from vllm import LLM, SamplingParams

    gate = os.environ.get("VLLM_NEURON_POOL_ADMISSION_GATE", "1")
    logger.info(
        "Hybrid livelock: model=%s tp=%d ep=%d mml=%d seg=%d bs=%d prompts=%d gate=%s",
        MODEL, TP, EP, MAX_MODEL_LEN, SEG, MAX_NUM_SEQS, NUM_PROMPTS, gate,
    )

    # Unique ~256-token prompts (no shared prefix; APC off) so each request
    # reserves its own worst-case KV allocation and the pool saturates.
    prompts = [
        f"Request {i}: " + " ".join(f"w{i}_{j}" for j in range(INPUT_WORDS))
        + f" Please describe idea number {i * 7 + 3} in detail."
        for i in range(NUM_PROMPTS)
    ]
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS
    )

    llm = LLM(
        model=MODEL,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        tensor_parallel_size=TP,
        enable_expert_parallel=True,
        limit_mm_per_prompt=LIMIT_MM,
        hf_overrides=HF_OVERRIDES,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "ep_degree": EP,
                "num_batched_tokens_buckets": [SEG],
                "kv_segment_size_buckets": [SEG],
                "num_seqs_buckets": [MAX_NUM_SEQS],
                "on_device_sampling_config": {"all_greedy": "true"},
            }
        },
    )

    start = time.monotonic()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.monotonic() - start

    logger.info("All %d requests completed in %.1fs", len(outputs), elapsed)
    assert len(outputs) == NUM_PROMPTS, (
        f"Expected {NUM_PROMPTS} outputs, got {len(outputs)} (livelock? gate={gate})"
    )
    for i, output in enumerate(outputs):
        assert output.outputs[0].text, f"Request {i} produced empty output"
