# Contributed Model: Qwen3.6-35B-A3B (Qwen3.5-MoE hybrid)

vllm-neuron implementation of
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), a **hybrid
Mixture-of-Experts** decoder served **text-only**. This README is the **single
source of truth** for the port: architecture, setup, serving, verification, and
measured performance. The `docs/` recipe and tutorial and the model-package README
are thin pointers to this file.

## Introduction

Qwen3.6-35B-A3B is a hybrid Mixture-of-Experts (MoE) language model from the Qwen
team. Its 40 decoder layers follow a 3:1 pattern — **30 GatedDeltaNet
(linear-attention) layers + 10 full-attention layers** — and every token is routed
through a 256-expert top-8 MoE plus a shared expert (35B total parameters, ~3B
active per token). BF16 weights.

This is a **text-only** deployment. The native checkpoint ships a
`Qwen3_5MoeForConditionalGeneration` (multimodal) architecture; the Neuron path
loads its text decoder via the `Qwen3_5MoeForCausalLM` alias and skips the vision
tower.

> **Qwen3.5 and Qwen3.6 share this architecture** (weights-only difference; HF loads
> both under `qwen3_5_moe`). Validated on device with Qwen3.6-35B-A3B weights; serves
> Qwen3.5-35B-A3B unchanged.

**Compatible checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Qwen3.6-35B-A3B | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | Trn2 | BF16 |
| Qwen3.5-35B-A3B | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | Trn2 | BF16 (same arch) |

## Verification scope

Validated on `trn2.48xlarge` (Trainium2, 64 logical NeuronCores at LNC2 / 16 chips)
at **TP8 / EP8**, BF16, greedy on-device sampling, on the Neuron 2.31 stack
(vLLM 0.21 / vllm-neuron 0.21.0.1.0.0): **GSM8K-CoT exact-match 93.0%** at batch
size 1 (100 questions, 4-shot), plus offline batch-1 and online concurrency 1 / 4 / 8
throughput on the `max_model_len=1024` + `kv_segment_size=512` recipe. Prefix caching
(APC) was exercised functionally with an 817-token shared prefix at batch size 4.

**Not verified:**

- Tensor/expert-parallel degrees other than TP8 / EP8, and quantization other than
  BF16.
- Sequence lengths above `max_model_len=1024`; the segmented GatedDeltaNet prefill
  path is bounded to ≤ 512-token segments, so longer contexts need a new bucket
  recipe (and a cold recompile).
- APC beyond that functional check — no accuracy sweep or throughput measurement was
  taken with prefix caching enabled.
- Vision input — the vision tower is skipped and this is a text-only deployment.
- Chunked GatedDeltaNet prefill — the alternative prefill kernel is present in the
  tree but **not enabled on this stack**; the segmented path is the only supported
  GatedDeltaNet prefill for this port.

**Gate this model with GSM8K, not with the generic logit-validation script's stock
thresholds.** Those thresholds sit below this model's BF16 noise floor — the CPU BF16
baseline's own L-inf error against FP32 is 0.03–0.11, above the script's static 0.011
top-5 tolerance — so `run_logit_validation_offline.py` reports FAILED even though every
prompt's greedy-argmax divergence stays within the script's 1-ULP acceptance threshold.

Every number in this document is this port's own measurement at TP8 / EP8 on the
current tree.

## Model Architecture

The checkpoint uses `model_type` `qwen3_5_moe`; the Neuron port serves it text-only
under the architecture alias `Qwen3_5MoeForCausalLM`.

| Parameter | Value |
|-----------|-------|
| Hidden size | 2048 |
| Num hidden layers | 40 (30 GatedDeltaNet + 10 full-attention) |
| Full-attn layer indices | 3, 7, 11, … , 39 |
| Num attention heads (full attn) | 16 |
| Num KV heads (full attn) | 2 (GQA) |
| Head dim (full attn) | 256 |
| Partial rotary factor | 0.25 (64 of 256 dims rotated) |
| Linear num key heads | 16 |
| Linear num value heads | 32 (K repeated 2× to V) |
| Linear key/value head dim | 128 |
| Linear conv kernel dim | 4 |
| Num experts | 256 |
| Experts per token (top-k) | 8 (+ shared expert) |
| MoE intermediate size | 512 |
| Vocab size | 248320 |
| SSM (mamba) dtype | float32 |

**Hybrid state.** The 10 full-attention layers use a paged KV cache; the 30
GatedDeltaNet layers keep recurrent + conv state in the unified paged pool, addressed
by the stable page slot (`block_table[:,0]`) and read/written by slot-indexed NKI
kernels. The port keeps the vLLM hybrid interfaces (`HasInnerState` / `IsHybrid`,
`get_mamba_state_shape_from_config()`, `get_mamba_state_dtype()`,
`bind_mamba_state()`).

**Parallelism.** TP8 / EP8 (pure expert parallelism, 32 experts per rank; all
collectives go through one unified world group — the EP-DGE constraint).
`tensor_parallel_size` must divide the 16 full-attention heads (maximum valid value
is 8). Prefill runs sequence-parallel; the GDN recurrent scan all-gathers to full-T
then reduce-scatters `out_proj` back to SP-local.

**GDN prefill (bounded-graph, required for seq ≥ 512).** The port ships the
**segmented sequential scan** GDN prefill (`VLLM_GDN_SEQ_NKI=1`), the
device-validated path. For sequences ≥ 512 tokens the bounded-graph prefill is
**required** — the unsegmented monolithic prefill traces one flat T-deep graph whose
token-position scatter trips a runtime scatter/gather out-of-bound access (→ NaN
logits). Keep `max_model_len`, `max_num_batched_tokens`, and `kv_segment_size_buckets`
so that no 1024-extent prefill graph is built (buckets ≤ 512). Short prompts (≤ 256)
run as a single segment and need none of these flags.

## Feature status

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| | Vision (image / video) | ❌ (text-only port) |
| **Quantization** | BF16 weights | ✅ |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Expert parallelism (EP) | ✅ |
| | Pipeline parallelism (PP) | ❌ |
| **Performance** | Continuous batching | ✅ |
| | Segmented GDN prefill | ✅ |
| | On-device sampling (greedy) | ✅ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (testing) | ✅ |

## Setup

### Native integration (nothing to apply)

The model is **natively integrated** into this repository — you do not need to apply
anything to use it:

- Model package: [`vllm_neuron/model/qwen3_5_moe/`](../vllm_neuron/model/qwen3_5_moe/)
  (architecture alias `Qwen3_5MoeForCausalLM`, `model_type` `qwen3_5_moe`), plus the
  GatedDeltaNet NKI kernels under `vllm_neuron/functional/` and the hybrid mamba
  prefix-cache sidecar `vllm_neuron/vllm/worker/neuron_mamba_apc.py`.
- Registration and the shared-file touch-points (registry, KV-cache `HybridKVSpec`,
  attention block-size alignment, platform hybrid guards, the pool-aware scheduler
  gate, and the runner/worker hybrid-state wiring) are committed directly on this
  branch.

### Step 1: Environment Setup

Verify Neuron devices are visible:

```bash
neuron-ls
# Expected: 16 NeuronCores listed for trn2.48xlarge
```

Set environment variables before running any inference script:

```bash
# Extend timeouts for large-model compilation
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_NEURON_COMPILATION_TIMEOUT=1200

# vLLM/NEFF + NKI compile cache — point at a large volume (not /)
export VLLM_CACHE_ROOT=/path/to/scratch/vllm_cache

# Required recipe flags for this model (default-OFF; all three are load-bearing)
export VLLM_GDN_SEQ_NKI=1          # bounded-graph GatedDeltaNet prefill (required for seq >= 512)
export VLLM_UNIFIED_KV_GATHER=1    # block-indexed unified KV gather for the full-attention layers
export VLLM_MOE_TKG_ROUTER_FP32=1  # fp32 decode router for correct top-8 expert selection
```

> **Cores:** for a tensor-parallel serve, select devices with
> `NEURON_VISIBLE_DEVICES` (e.g. `0-7`). Do **not** use `NEURON_RT_VISIBLE_CORES` for
> a multi-process TP serve — it hard-errors with "cannot be used with multi-processing
> execution".
>
> **Compiler flags:** leave `NEURON_CC_FLAGS` unset and let the framework compose
> them. Setting it *replaces* (rather than appends to) the framework's flag string and
> drops flags the model requires (e.g. `--enable-verifier=false`). Relocate caches
> with `VLLM_CACHE_ROOT` instead.

### Step 2: Download the Model

```bash
hf download \
    Qwen/Qwen3.6-35B-A3B \
    --local-dir /path/to/Qwen3.6-35B-A3B
```

> On the `huggingface_hub` version this stack pins (1.x), the older
> `huggingface-cli` entry point is removed and only prints a deprecation notice — use
> `hf download` as above. Set `HF_XET_HIGH_PERFORMANCE=1` for faster transfers (the
> older `HF_HUB_ENABLE_HF_TRANSFER` is likewise deprecated).

> **Tip:** The BF16 checkpoint is large — download it to a filesystem with room for it
> rather than the root volume.

## Serving

The `hf-overrides` and `limit-mm-per-prompt` flags select the text-only decoder path
and are **mandatory** — omitting the `Qwen3_5MoeForCausalLM` override forces the
multimodal config path (fp32 SSM cache → block size 384 → compile-time OOM).

### Online serving (OpenAI-compatible)

```bash
vllm serve /path/to/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-model-len 1024 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 1 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --hf-overrides '{"architectures":["Qwen3_5MoeForCausalLM"]}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "bf16",
            "ep_degree": 8,
            "on_device_sampling_config": {"all_greedy": "true"},
            "kv_segment_size_buckets": [512],
            "num_batched_tokens_buckets": [512],
            "num_seqs_buckets": [1]
        }
    }'
```

> **Bucket sizing:** `kv_segment_size_buckets` and `num_batched_tokens_buckets` must
> be **≤ 512** so no 1024-extent prefill graph is traced (see GDN prefill above).
> Each bucket adds compile time, and changing any bucket forces a full cold recompile
> — freeze one recipe rather than sweeping sizes.

Once the server is up, send requests using the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
resp = client.chat.completions.create(
    model="Qwen3.6-35B-A3B",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=50,
)
print(resp.choices[0].message.content)
```

For higher throughput keep `--max-model-len 1024` (with `kv_segment_size_buckets:
[512]`) and serve with a larger batch — e.g. `--max-num-seqs 8` + `num_seqs_buckets:
[8]` — then drive load with `vllm bench serve`. A tight `--max-model-len 512` with
batch size > 1 also runs correctly: the scheduler's pool-aware admission gate (see
**Measured performance**) throttles concurrency to what the hybrid block pool can
hold rather than stalling.

### Offline inference (`llm.generate()`)

The ready-to-run example
[`examples/vllm_neuron/models/qwen3_5_moe/run.py`](../examples/vllm_neuron/models/qwen3_5_moe/run.py)
sets the required recipe flags via `os.environ.setdefault`, so it works standalone.
Equivalent direct `LLM` call:

```python
import os
os.environ.setdefault("VLLM_GDN_SEQ_NKI", "1")
os.environ.setdefault("VLLM_UNIFIED_KV_GATHER", "1")
os.environ.setdefault("VLLM_MOE_TKG_ROUTER_FP32", "1")

from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Qwen3.6-35B-A3B",
    tensor_parallel_size=8,
    enable_expert_parallel=True,
    max_model_len=1024,
    max_num_batched_tokens=512,
    max_num_seqs=1,
    limit_mm_per_prompt={"image": 0, "video": 0},
    hf_overrides={"architectures": ["Qwen3_5MoeForCausalLM"]},
    additional_config={
        "neuron_config": {
            "quantization": "bf16",
            "ep_degree": 8,
            "on_device_sampling_config": {"all_greedy": "true"},
            "kv_segment_size_buckets": [512],
            "num_batched_tokens_buckets": [512],
            "num_seqs_buckets": [1],
        }
    },
)
out = llm.generate(["What is the capital of France?"],
                   SamplingParams(max_tokens=200, temperature=0.0))
print(out[0].outputs[0].text)
```

## Accuracy Evaluation

**Benchmark:** GSM8K (grade-school math word problems), exact-match on the final
answer, measured on real hardware (`trn2.48xlarge`, TP8/EP8, BF16, greedy on-device
sampling).

| Metric | Qwen3.6-35B-A3B, Neuron Trn2 BF16 |
|--------|:---------------------------------:|
| GSM8K-CoT exact-match, flexible-extract (batch size 1) | **93.0%** ± 2.6% |
| GSM8K-CoT exact-match, strict-match (batch size 1) | **92.0%** ± 2.7% |

Measured with `lm_eval --tasks gsm8k_cot` over the OpenAI-compatible endpoint of a
server started with the online recipe above: 100 questions, 4-shot
(`--apply_chat_template --fewshot_as_multiturn`), `max_tokens 448`,
`enable_thinking: false`, `max_length 1024`, `num_concurrent 1`. The ± figures are
lm_eval's reported standard error; 100 questions is a subset, not the full split.

**Reproduce:** serve the checkpoint with the online recipe above, then run the GSM8K
task from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
against the running server over its OpenAI-compatible endpoint. Keep the few-shot
prompt plus generation length within the 1024-token context window (e.g. 4-shot,
≤ 448 generated tokens) so requests are not rejected for exceeding `max_model_len`.
Disable Qwen3 thinking (`chat_template_kwargs: {"enable_thinking": false}`) so the
generation budget goes to the answer rather than a truncated `<think>` block. Pass
the value of `--served-model-name` as lm_eval's `model=` argument, not the checkpoint
path, or every request is rejected with a 404.

> **Do not gate this model on the generic logit-validation script's stock
> thresholds** — they sit below this model's BF16 noise floor and report FAILED even
> though the greedy argmax stays within the script's own 1-ULP acceptance threshold.
> See [Verification scope](#verification-scope).

## Measured performance

Measured on a `trn2.48xlarge` (TP8/EP8, BF16, greedy on-device sampling,
`max_model_len=1024` + `kv_segment_size_buckets:[512]`) with the repository's own
model-agnostic benchmark clients: `vllm bench serve` against a running server for the
online rows, `vllm bench throughput` for the offline row. Random dataset, 256-token
input / 128-token output (`--random-range-ratio 0`), `--ignore-eos`, 24 prompts per
concurrency level online and 16 prompts offline. Every run completed all requests
(0 failed).

| Configuration | Concurrency | Output throughput | Mean TPOT | Mean TTFT |
|---------------|:-----------:|:-----------------:|:---------:|:---------:|
| `max_model_len=1024`, `kv_segment_size=512`, online | 1 | 22.57 tok/s | 42.18 ms | 314 ms |
| `max_model_len=1024`, `kv_segment_size=512`, online | 4 | 74.37 tok/s | 48.10 ms | 775 ms |
| `max_model_len=1024`, `kv_segment_size=512`, online | 8 | 123.78 tok/s | 54.01 ms | 1411 ms |
| `max_model_len=1024`, offline (`bench throughput`) | 1 | 22.45 tok/s (67.36 total tok/s) | — | — |

Output throughput scales 5.5× from concurrency 1 to 8 while time-per-output-token
grows only 28% (42.2 → 54.0 ms), i.e. the added concurrency is nearly free per token.
Offline batch-1 matches online concurrency-1 (22.45 vs 22.57 tok/s), so the serving
stack adds no measurable per-token overhead at batch 1. A second pass of the same
sweep gave 22.54 / 74.39 / 123.80 tok/s, so these figures repeat to ~0.03 tok/s.

Peak KV-cache utilization stayed at or below **1.5%** across the whole sweep (0.2% at
1 running request, 0.7% at 4, 1.5% at 8); the server samples this metric every 10 s,
so these are observed peaks rather than guaranteed maxima.

> **Multi-batch serving: use `max_model_len=1024` with `kv_segment_size=512`** for
> best throughput (concurrency 4 and 8 both run cleanly, all requests completed).
>
> **Pool-aware admission gate.** A tight `max_model_len=512` makes each request
> reserve its full worst-case KV allocation up front, so the hybrid (attention +
> GatedDeltaNet) unified block pool can saturate at only a few concurrent decodes.
> The scheduler (`NeuronScheduler._pool_admission_ok`, on by default) predicts each
> incoming prefill's worst-case block footprint against the free pool and defers
> admission when the pool is full, instead of dead-ending into empty batches. Batch
> size > 1 at `max_model_len=512` therefore runs correctly: verified on device at
> batch size 4 with all 24 requests completing in 41.6 s and **0 empty batches**. It
> is a no-op when the pool has room, so it does not affect the `max_model_len=1024`
> recipe. It can be disabled with `VLLM_NEURON_POOL_ADMISSION_GATE=0` (not
> recommended for hybrid models with a tight `max_model_len`).
>
> Any change to the segmentation, sequence-length, or bucket configuration changes the
> traced graph and triggers a full cold recompile, so freeze one recipe rather than
> sweeping bucket sizes.

## Contents of this bundle

The model is integrated in-tree (committed directly on this branch), so the
top-level `Qwen3.6-35B-A3B/` bundle is just this source-of-truth README. There is
**no `src/` copy** here (that would duplicate the in-tree package).

```text
Qwen3.6-35B-A3B/
└── README.md            # This file — the source of truth for the Qwen3.6-35B-A3B port
```

The paths the port touches are listed below; that listing is the record of which shared
framework files this model changes.

### Paths this port touches across the repository

**New — model package** (`vllm_neuron/model/qwen3_5_moe/`)

```text
vllm_neuron/model/qwen3_5_moe/
├── __init__.py               # Package exports
├── README.md                 # Package-level module structure (points here)
├── config.py                 # HF → Neuron config translation (hybrid layer pattern, MoE, GDN dims)
├── factory.py                # Model builder + weight-loader wiring
├── model.py                  # Hybrid MoE decoder (30 GatedDeltaNet + 10 full-attention, 256-expert top-8 MoE)
└── weight_loaders_bf16.py    # BF16 sharded / fused weight loaders (TP8 / EP8)
```

**New — NKI kernels + hybrid sidecar**

```text
vllm_neuron/functional/gated_delta_rule_seq.py       # Bounded-graph sequential GDN prefill (VLLM_GDN_SEQ_NKI — validated path)
vllm_neuron/functional/gated_delta_rule.py           # Chunked GDN delta-rule prefill (present, not enabled here)
vllm_neuron/functional/gdn_conv_update.py            # GDN causal-conv state update (decode)
vllm_neuron/functional/gdn_conv_update_compact.py    # Compact variant of the conv-state update
vllm_neuron/functional/gdn_state_update.py           # GDN recurrent-state update (decode)
vllm_neuron/functional/gdn_state_update_compact.py   # Compact variant of the recurrent-state update
vllm_neuron/functional/paged_kv_gather.py            # Block-indexed unified KV gather (.ap page-strided)
vllm_neuron/functional/slot_indirect_probe.py        # Slot-indexed indirect-probe helper
vllm_neuron/vllm/worker/neuron_mamba_apc.py          # Hybrid mamba prefix-cache sidecar
```

**New — docs & example**

```text
docs/model-recipes/qwen3-6-moe.md               # Model recipe / card (pointer to this README)
docs/tutorials/tutorial-qwen3-6-moe.md          # Deployment tutorial (pointer to this README)
examples/vllm_neuron/models/qwen3_5_moe/run.py  # Offline runner (bakes the recipe flags)
```

**Modified — shared framework touch-points** (7 files; all hybrid code is `MambaSpec` /
`_mamba_apc_enabled()`-gated, so the non-hybrid path stays byte-identical)

```text
vllm_neuron/model/registry.py                   # Register Qwen3_5MoeForConditionalGeneration / Qwen3_5MoeForCausalLM aliases
vllm_neuron/model/kv_cache.py                   # Add HybridKVSpec (KVSpec subclass carrying the stateful/GDN layer names)
vllm_neuron/vllm/attention/attn.py              # get_supported_kernel_block_sizes → MultipleOf(128) for the decode-attn mask kernel
vllm_neuron/vllm/platform.py                    # Hybrid block-size align + mamba-mode / APC-async guards + text-only vision-gate guard
vllm_neuron/vllm/core/scheduler.py              # Pool-aware admission gate (_pool_admission_ok) — the mml=512 + bs>1 livelock fix
vllm_neuron/vllm/worker/neuron_model_runner.py  # Hybrid state alloc / bind / slot mgmt + mamba-page-padded KV spec
vllm_neuron/vllm/worker/neuron_worker.py        # Cap the unified-KV budget so the shared unified slab fits the .ap slab size
docs/model-recipes/index.md                     # Add the recipe grid card + toctree entry
docs/tutorials/index.md                         # Add the tutorial grid card + toctree entry
```
