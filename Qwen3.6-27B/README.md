# Contributed Model: Qwen3.6-27B (Qwen3.5 dense hybrid)

vllm-neuron implementation of
[`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) — a **hybrid dense**
decoder served **text-only**. This README is the **single source of truth** for the
port: architecture, setup, serving, verification, and measured performance. The
`docs/` recipe and tutorial and the model-package README are thin pointers to this
file.

## Introduction

Qwen3.6-27B is a hybrid **dense** language model from the Qwen team. Its 64 decoder
layers follow a 3:1 pattern (`full_attention_interval=4`) — **48 GatedDeltaNet
(linear-attention) layers + 16 full-attention layers** — and every token flows
through a plain **SwiGLU MLP** (no Mixture-of-Experts routing). BF16 weights.

This is a **text-only** deployment. The native checkpoint ships a
`Qwen3_5ForConditionalGeneration` (multimodal) architecture; the Neuron path loads
its text decoder via the `Qwen3_5ForCausalLM` alias and skips the vision tower.

> **Dense sibling of [`Qwen3.6-35B-A3B`](https://github.com/AWShtokoyo/vllm-neuron/tree/add-qwen36-moe/Qwen3.6-35B-A3B/README.md).** The two
> share the hybrid GatedDeltaNet + full-attention stack, QK-norm, sigmoid output
> gate, partial M-RoPE, the GatedDeltaNet NKI kernels, and the whole IsHybrid /
> MambaSpec state framework; they differ **only** in the FFN (dense SwiGLU here vs a
> 256-expert top-8 MoE there), so this port has **no expert parallelism**. All the
> shared hybrid framework touch-points are introduced by the MoE port — this port
> adds only the registry arch aliases.

> **Qwen3.5 and Qwen3.6 share this architecture** (weights-only difference; HF loads
> both under `qwen3_5`). Validated on device with Qwen3.6-27B weights; serves
> Qwen3.5-27B unchanged.

**Compatible checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Qwen3.6-27B | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | Trn2 | BF16 |
| Qwen3.5-27B | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | Trn2 | BF16 (same arch) |

## Verification scope

Validated on a `trn2.48xlarge` (Trainium2) at **TP=4 and TP=8**, BF16, greedy
on-device sampling, on the Neuron 2.31 stack (vLLM 0.21 / vllm-neuron 0.21.0.1.0.0):
**GSM8K-CoT exact match 96.0% strict / 97.0% flexible** at TP=8 (100 questions,
4-shot), 3-way logit validation on both the unsegmented and the segmented
(`kv_segment_size_buckets: [512]`) prefill recipes, and `vllm bench serve` latency and
throughput at concurrency 1 / 2 (TP=4) and 1 / 2 / 4 / 8 (TP=8). TP=4 occupies 4 of
the instance's logical NeuronCores (one Trainium2 chip), so the same recipe also fits
a `trn2.3xlarge`; the numbers here were taken on the 48xlarge.

**Not verified:**

- Parallel degrees other than TP=4 and TP=8, and quantization other than BF16.
- Sequence lengths above `max_model_len=1024`; the bounded-graph GatedDeltaNet prefill
  is kept to ≤ 512-token segments, so a longer context needs a new bucket recipe (and
  a cold recompile).
- **Prefix caching (APC).** It is off by default for this checkpoint and every recipe
  in this document serves with `--no-enable-prefix-caching`; leave it off.
- **Chunked GatedDeltaNet prefill.** The alternative prefill kernel is present in the
  tree but not enabled for this port; segmented prefill is the verified path.
- Vision input — the vision tower is skipped and this is a text-only deployment.

Every number in this document is this port's own measurement.

## Model Architecture

The checkpoint uses `model_type` `qwen3_5`; the Neuron port serves it text-only under
the architecture alias `Qwen3_5ForCausalLM` (the **dense** alias — not the MoE
`Qwen3_5MoeForCausalLM`, which loads 256-expert weights).

| Parameter | Value |
|-----------|-------|
| Hidden size | 5120 |
| Num hidden layers | 64 (48 GatedDeltaNet + 16 full-attention) |
| Full-attn layer indices | every 4th (3, 7, 11, … , 63), `full_attention_interval=4` |
| Num attention heads (full attn) | 24 |
| Num KV heads (full attn) | 4 (GQA) |
| Head dim (full attn) | 256 |
| Attention output gate | sigmoid (Q projection emits 2× per head: query + gate) |
| Partial rotary factor | 0.25 (64 of 256 dims rotated) |
| Linear num key heads | 16 |
| Linear num value heads | 48 |
| Linear key/value head dim | 128 |
| Linear conv kernel dim | 4 |
| FFN | dense SwiGLU, `intermediate_size=17408` (no experts) |
| Vocab size | 248320 |
| SSM (mamba) dtype | float32 |

**Hybrid state.** The 16 full-attention layers use a paged KV cache; the 48
GatedDeltaNet layers keep recurrent + conv state in the unified paged pool, addressed
by the stable page slot and read/written by slot-indexed NKI kernels. The port keeps
the vLLM hybrid interfaces (`HasInnerState` / `IsHybrid`,
`get_mamba_state_shape_from_config()`, `get_mamba_state_dtype()`,
`bind_mamba_state()`).

**Parallelism.** **TP=4 and TP=8**, both measured on a `trn2.48xlarge`; TP=4 uses 4
logical NeuronCores, so it also fits a `trn2.3xlarge`. `tensor_parallel_size` must
divide the 24 full-attention query heads and the 4 KV heads. Prefill runs
sequence-parallel; the GDN recurrent scan all-gathers to full-T then reduce-scatters
`out_proj` back to SP-local; decode uses `all_reduce`. There is **no EP** — the dense
SwiGLU MLP shards by TP directly (`gate`/`up` shard_dim=1, `down` shard_dim=0).

**TP=4 fused-QKV SBUF workaround.** The gated full-attention block emits 2× per Q head
(query + gate), so the per-rank fused-QKV width at TP=4 is `24/4·256·2 (Q) + 2·4/4·256
(K,V) = 3584`. The framework's fused `qkv_proj` megakernel keeps that
`[hidden=5120, qkv=3584]` weight tile SBUF-resident and overruns the ~208 KiB
per-core budget by ~8% (`[NCC_INKI016] SBUF budget exceeded`, **weight-bound** —
identical at a 128- and 256-token prefill, so reducing `max_model_len` does not help).
Set **`VLLM_QWEN_QKV_MATMUL=1`** to route QKV through an equivalent plain `torch.matmul`
(QKV is used as a pure projection — QK-norm, RoPE, and the gate split are applied
separately downstream — so it is numerically equivalent). At TP=8 on a `trn2.48xlarge`
the per-rank width halves to 1792 and the fused kernel fits — **leave the flag unset
there.**

**GDN prefill (bounded-graph, required for seq ≥ 512).** The port ships the
recurrence-in-kernel bounded-graph GatedDeltaNet prefill (`VLLM_GDN_SEQ_NKI=1`), the
device-validated path (shared with the MoE sibling). For sequences ≥ 512 tokens the
bounded-graph prefill is **required** — the unsegmented monolithic prefill traces one
flat T-deep graph whose token-position scatter trips a runtime scatter/gather
out-of-bound access (→ NaN logits). Keep `max_model_len`, `max_num_batched_tokens`,
and `kv_segment_size_buckets` so that no 1024-extent prefill graph is built (buckets
≤ 512). Short prompts (≤ 256) run as a single segment and need none of these flags.

## Feature status

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| | Vision (image / video) | ❌ (text-only port) |
| **Quantization** | BF16 weights | ✅ |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Expert parallelism (EP) | ❌ (dense FFN — not applicable) |
| | Pipeline parallelism (PP) | ❌ |
| **Performance** | Continuous batching | ✅ |
| | Bounded-graph GDN prefill | ✅ |
| | On-device sampling (greedy) | ✅ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (testing) | ✅ |

## Setup

### Native integration (nothing to apply)

The model is **natively integrated** into this repository — you do not need to apply
anything to use it:

- Model package: [`vllm_neuron/model/qwen3_5_dense/`](../vllm_neuron/model/qwen3_5_dense/)
  (architecture alias `Qwen3_5ForCausalLM`, `model_type` `qwen3_5`). The GatedDeltaNet
  NKI kernels under `vllm_neuron/functional/` and the hybrid framework machinery
  (`HybridKVSpec`, the pool-aware scheduler gate, the platform hybrid-align +
  vision-gate guards, the runner/worker hybrid-state wiring) are **shared with — and
  introduced by — the [`qwen3_5_moe` port](https://github.com/AWShtokoyo/vllm-neuron/tree/add-qwen36-moe/Qwen3.6-35B-A3B/README.md)**; the dense
  port reuses them unchanged.
- The only shared-file edit unique to this port is the registry arch aliases, committed
  directly on this branch.

### Step 1: Environment Setup

Verify Neuron devices are visible:

```bash
neuron-ls
# Expected: 4 NeuronCores listed for trn2.3xlarge
```

Set environment variables before running any inference script:

```bash
# Extend timeouts for large-model compilation
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_NEURON_COMPILATION_TIMEOUT=1200

# vLLM/NEFF + NKI compile cache — point at a large volume (not /)
export VLLM_CACHE_ROOT=/path/to/scratch/vllm_cache

# Required recipe flags for this model (default-OFF; all three are load-bearing)
export VLLM_GDN_SEQ_NKI=1         # bounded-graph GatedDeltaNet prefill (required for seq >= 512)
export VLLM_UNIFIED_KV_GATHER=1   # block-indexed unified KV gather for the full-attention layers
export VLLM_QWEN_QKV_MATMUL=1     # matmul QKV projection — fits SBUF at TP=4 (UNSET at TP>=8)

# trn2.3xlarge has no EFA device; EFA affinity is a perf optimization, not correctness
export NEURON_SKIP_EFA_AFFINITY=1

# Serial compile/trace workers — avoid a host-OOM when the model's graphs x TP4
# compile in parallel on the smaller instance
export VLLM_NEURON_PARALLEL_COMPILE_WORKERS=1
export VLLM_NEURON_PARALLEL_TRACE_WORKERS=1
```

> **QKV at TP=4 vs TP≥8:** `VLLM_QWEN_QKV_MATMUL=1` is **required at TP=4** (fused-QKV
> tile overruns SBUF by ~8%) and must be **left unset at TP≥8** (the fused kernel fits
> and is preferred). See the workaround derivation under **Model Architecture**.
>
> **Cores:** for a tensor-parallel serve, select devices with
> `NEURON_VISIBLE_DEVICES` (e.g. `0-3`). Do **not** use `NEURON_RT_VISIBLE_CORES` for
> a multi-process TP serve — it hard-errors with "cannot be used with multi-processing
> execution".
>
> **Compiler flags:** leave `NEURON_CC_FLAGS` unset and let the framework compose
> them. Setting it *replaces* (rather than appends to) the framework's flag string and
> drops flags the model requires. Relocate caches with `VLLM_CACHE_ROOT` instead.

### Step 2: Download the Model

```bash
hf download \
    Qwen/Qwen3.6-27B \
    --local-dir /path/to/Qwen3.6-27B
```

> On the `huggingface_hub` version this stack pins (1.x), the older `huggingface-cli`
> entry point is removed and only prints a deprecation notice — use `hf download` as
> above. Set `HF_XET_HIGH_PERFORMANCE=1` for faster transfers (the older
> `HF_HUB_ENABLE_HF_TRANSFER` is likewise deprecated).

> **Tip:** The BF16 checkpoint is ~56 GB — download it to a filesystem with room for it
> rather than the root volume.

## Serving

The `hf-overrides` and `limit-mm-per-prompt` flags select the text-only **dense**
decoder path and are **mandatory** — omitting the `Qwen3_5ForCausalLM` override forces
the multimodal config path (fp32 SSM cache → larger block size → compile-time OOM).
The `Qwen3_5ForCausalLM` alias resolves to the dense decoder; do not confuse it with
the MoE `Qwen3_5MoeForCausalLM`.

### Offline inference (`llm.generate()`)

The ready-to-run example
[`examples/vllm_neuron/models/qwen3_5_dense/run.py`](../examples/vllm_neuron/models/qwen3_5_dense/run.py)
sets the required recipe flags via `os.environ.setdefault`, so it works standalone:

```bash
python examples/vllm_neuron/models/qwen3_5_dense/run.py \
    --model-checkpoint /path/to/Qwen3.6-27B \
    --tensor-parallel-size 4 --max-model-len 256 --max-num-seqs 1
```

Equivalent direct `LLM` call:

```python
import os
os.environ.setdefault("VLLM_GDN_SEQ_NKI", "1")
os.environ.setdefault("VLLM_UNIFIED_KV_GATHER", "1")
os.environ.setdefault("VLLM_QWEN_QKV_MATMUL", "1")          # UNSET at TP>=8
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
os.environ.setdefault("VLLM_NEURON_PARALLEL_COMPILE_WORKERS", "1")
os.environ.setdefault("VLLM_NEURON_PARALLEL_TRACE_WORKERS", "1")

from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Qwen3.6-27B",
    tensor_parallel_size=4,
    max_model_len=256,
    max_num_seqs=1,
    limit_mm_per_prompt={"image": 0, "video": 0},
    hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]},   # dense alias
    additional_config={
        "neuron_config": {
            "quantization": "bf16",
            "on_device_sampling_config": {"all_greedy": "true"},
            "num_batched_tokens_buckets": [256],
            "num_seqs_buckets": [1],
        }
    },
)
out = llm.generate(["The capital of France is "],
                   SamplingParams(max_tokens=32, temperature=0.0))
print(out[0].outputs[0].text)
```

### Online serving (OpenAI-compatible)

Export the recipe flags from Step 1 first, then:

```bash
vllm serve /path/to/Qwen3.6-27B \
    --served-model-name Qwen3.6-27B \
    --tensor-parallel-size 4 \
    --max-model-len 256 \
    --max-num-seqs 1 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "bf16",
            "on_device_sampling_config": {"all_greedy": "true"},
            "num_batched_tokens_buckets": [256],
            "num_seqs_buckets": [1]
        }
    }'
```

For a long-seq (≥ 512) recipe, add `--max-model-len 1024 --max-num-batched-tokens 512`
and set `kv_segment_size_buckets: [512]` + `num_batched_tokens_buckets: [512]` in the
neuron config (keep every bucket ≤ 512 so no 1024-extent GDN graph is traced). That
segmented recipe is the one the GSM8K and logit-validation results below were measured
on.

> **Bucket sizing:** changing any bucket forces a full cold recompile, so freeze one
> recipe rather than sweeping sizes.

> **Leave prefix caching off.** It is off by default for this checkpoint and is not
> part of the verified recipe — see [Verification scope](#verification-scope).

> **Reproducing the latency numbers:** `num_seqs_buckets` must match the concurrency
> you drive. A bucket list of `[4]` makes a single-sequence request run the
> batch-4-padded decode graph and inflates TPOT by ~18% (53.0 vs 44.9 ms at
> concurrency 1) with TTFT unchanged. The [Measured
> performance](#measured-performance) table was taken with `num_seqs_buckets: [1, 2]`
> and `--max-num-seqs 2`.

## Accuracy Evaluation

Measured on real hardware (`trn2.48xlarge`, TP=4 and TP=8, BF16, greedy on-device
sampling): **GSM8K-CoT exact match 96.0% (strict) / 97.0% (flexible)** at TP=8, plus
the generic 3-way logit validation on the segmented and unsegmented prefill recipes.

### GSM8K-CoT

Measured on `trn2.48xlarge` at **TP=8**, BF16, greedy on-device sampling, via
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(`gsm8k_cot`) driven against the server's OpenAI-compatible endpoint:

| Metric | Value | Stderr |
|:---|:---:|:---:|
| `exact_match` (strict-match) | **96.0%** | ±1.97 |
| `exact_match` (flexible-extract) | **97.0%** | ±1.71 |

Recipe: 100 questions, 4-shot with the chat template applied
(`--apply_chat_template --fewshot_as_multiturn`), `max_tokens 448` and
`enable_thinking: false` so prompt plus generation stays inside the 1024-token
window, `max-model-len 1024`, `max-num-batched-tokens 512`,
`kv_segment_size_buckets: [512]`, `--max-num-seqs 1`, prefix caching off. The
4-shot / 448-token bound keeps this directly comparable to the
[Qwen3.6-35B-A3B](https://github.com/AWShtokoyo/vllm-neuron/tree/add-qwen36-moe/Qwen3.6-35B-A3B/README.md) MoE port's number, which was taken
with the identical recipe.

> `lm_eval`'s `model=` argument must be the server's `--served-model-name` value, not
> the checkpoint path.

### Logit validation

Generic 3-way logit validation (FP32 baseline → CPU BF16 expected → Neuron target)
from `examples/vllm_neuron/accuracy/`, greedy, 3 prompts × 16 generated tokens,
thresholds top-1 divergence ≤ 1 ULP and TopK-5 rtol ≤ 0.011. **Segmented prefill
(`kv_segment_size_buckets: [512]`) passes**, and so does single-shot prefill:

| TP | `max-model-len` | `kv_segment_size_buckets` | Result | Aggregate σ-ratio | Max top-1 divergence |
|:--:|:---:|:---:|:---:|:---:|:---:|
| 4 | 256 | absent | **PASSED** | 0.9774 | 0.125 (≤ 1 ULP) |
| 4 | 512 | absent | **PASSED** | 0.9326 | 0.125 (≤ 1 ULP) |
| 8 | 512 | `[512]` | **PASSED** | 1.0082 | 0.000 |
| 8 | 512 | absent | FAILED on TopK-5 rtol | 1.0459 | 0.000 |

The σ-ratio compares the Neuron-vs-FP32 error against the CPU-BF16-vs-FP32 reference
error, so ≤ 1.0 means Neuron is no further from FP32 than BF16 itself is. On the
segmented recipe the aggregate is 1.0082 with **zero divergent tokens** and all
per-prompt L2 / L∞ / static gates at 3/3.

The one FAILED row is a threshold artefact at this model's BF16 noise floor, not a
wrong-token result: top-1 argmax never diverges (max divergence 0.000), the per-prompt
max-L2 / max-L∞ / static gates are 3/3, and only 2 of 16 tokens exceed the static
TopK-5 tolerance, at rtol 0.0119 against a 0.011 limit. Use the segmented recipe, and
gate the model on GSM8K.

**Reproduce:** run the generic 3-way logit-validation scripts under
`examples/vllm_neuron/accuracy/` against the serving recipe above. For GSM8K, serve
the checkpoint with the online recipe and run the `gsm8k_cot` task from
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) against
the running server over its OpenAI-compatible endpoint, with the arguments listed
under [GSM8K-CoT](#gsm8k-cot).

## Measured performance

**Concurrency** in the tables below is `vllm bench serve --max-concurrency` — the
number of requests in flight at once, reported by the benchmark as `Maximum request
concurrency`. It is not data parallelism (this port runs a single instance, DP=1); the
decode batch size equals it because `num_seqs_buckets` is matched to it in every run.

Latency at TP=4 on a `trn2.48xlarge` with `vllm bench serve` (`random` dataset,
input=256 / output=128, greedy, `--ignore-eos`, warm cache, `--max-num-seqs 2` with
`num_seqs_buckets: [1, 2]`, 8 prompts per concurrency level, 0 failed requests):

| Concurrency | TTFT median (ms) | TPOT median (ms) | ITL median (ms) | Output throughput (tok/s) |
|:-----------:|:----------------:|:----------------:|:---------------:|:-------------------------:|
| 1 | 1250 | 44.91 | 44.91 | 18.4 |
| 2 | 1920 | 53.01 | 47.89 | 29.7 |

**TP=8** on the same instance, same `vllm bench serve` recipe (`random`, input=256 /
output=128, `--random-range-ratio 0`, greedy, `--ignore-eos`, warm cache,
`max-model-len 512`, `num_batched_tokens_buckets: [512]`,
`num_seqs_buckets: [1, 2, 4, 8]`, prefix caching off, 8 prompts per concurrency
level, 0 failed requests at every level):

| Concurrency | TTFT mean (ms) | TPOT mean (ms) | TPOT P99 (ms) | Output throughput (tok/s) | Total throughput (tok/s) |
|:-----------:|:--------------:|:--------------:|:-------------:|:-------------------------:|:------------------------:|
| 1 | 668 | 32.22 | 32.23 | 26.9 | 80.7 |
| 2 | 1002 | 36.62 | 39.27 | 45.3 | 135.9 |
| 4 | 1662 | 46.13 | 53.94 | 68.1 | 204.2 |
| 8 | 2999 | 59.47 | 77.56 | 97.0 | 291.1 |

TP=8 is the configuration to use for throughput: at concurrency 1 it is ~1.46× the
TP=4 output throughput (26.9 vs 18.4 tok/s) at roughly half the TTFT, and it scales to
97.0 tok/s at concurrency 8.

> **Match `num_seqs_buckets` to the concurrency** or TPOT regresses ~18% from decode
> batch padding — see the reproduction note under [Serving](#serving).

> `vllm bench serve --model` must be the server's `--served-model-name`, which the
> benchmark's local tokenizer loader cannot resolve; pass `--tokenizer <checkpoint
> path>` alongside it or every request fails with `is not a local folder`.

> **Batch size at TP=4 is bounded by HBM, not by the scheduler.** TP=4 gives the model
> 4 NeuronCores — one Trainium2 chip, 96 GB HBM — and a full BF16 copy of the 27B
> weights is resident on it. Driving 4 concurrent requests leaves too little for the
> hybrid (full-attention + GatedDeltaNet) block pool and the device raises `status=4
> (Allocation Failure)`; concurrency 1 and 2 are what fits, which is what the TP=4 table
> measures. Use **TP=8** for higher concurrency: per-rank weights halve, the pool has
> room, and the TP=8 table above completes concurrency 4 and 8 with 0 failed requests.
>
> Any change to the segmentation, sequence-length, or bucket configuration changes the
> traced graph and triggers a full cold recompile, so freeze one recipe rather than
> sweeping bucket sizes.

## Contents of this bundle

The model is integrated in-tree (committed directly on this branch), so the top-level
`Qwen3.6-27B/` bundle is just this source-of-truth README. There is **no `src/` copy**
here (that would duplicate the in-tree package).

```text
Qwen3.6-27B/
└── README.md            # This file — the source of truth for the Qwen3.6-27B port
```

The paths the port touches are listed below; that listing is the record of which shared
framework files this model changes.

### Paths this port touches across the repository

**New — model package** (`vllm_neuron/model/qwen3_5_dense/`)

```text
vllm_neuron/model/qwen3_5_dense/
├── __init__.py               # Package exports
├── README.md                 # Package-level module structure (points here)
├── config.py                 # HF → Neuron config translation (hybrid layer pattern, dense FFN, GDN dims)
├── factory.py                # Model builder + IsHybrid classmethod delegation
├── model.py                  # Hybrid dense decoder (48 GatedDeltaNet + 16 full-attention, SwiGLU MLP)
└── weight_loaders_bf16.py    # BF16 GatedDeltaNet TP head-sharding loaders (dense FFN uses the standard loader)
```

**New — NKI kernels + hybrid sidecar** (the branch carries its own copies so it builds
and serves standalone — no dependency on any other branch)

```text
vllm_neuron/functional/gated_delta_rule.py      # Chunked GDN prefill kernel (run_gdn_chunk_prefill)
vllm_neuron/functional/gated_delta_rule_seq.py  # Segmented GDN prefill kernel (run_gdn_seq_prefill)
vllm_neuron/functional/gdn_conv_update.py       # Short-conv state update (decode)
vllm_neuron/functional/gdn_state_update.py      # GDN recurrent state update (decode)
vllm_neuron/functional/paged_kv_gather.py       # Paged KV / GDN-state gather-scatter kernels
vllm_neuron/vllm/worker/neuron_mamba_apc.py     # Hybrid mamba prefix-cache sidecar
```

**New — docs & example**

```text
docs/model-recipes/qwen3-6-27b.md               # Model recipe / card (pointer to this README)
docs/tutorials/tutorial-qwen3-6-27b.md          # Deployment tutorial (pointer to this README)
examples/vllm_neuron/models/qwen3_5_dense/run.py  # Offline runner (bakes the recipe flags)
```

**Modified — shared framework touch-points** (7 framework files; this branch is cut from a
MoE-less release tree, so it carries the full hybrid machinery — 6 of the 7 are
**byte-identical to the `qwen3_5_moe` port**, `registry.py` differs only in the arch
aliases it registers)

```text
vllm_neuron/model/registry.py                   # Register Qwen3_5ForConditionalGeneration / Qwen3_5ForCausalLM aliases (dense-specific)
vllm_neuron/model/kv_cache.py                   # HybridKVSpec — shared with qwen3_5_moe, byte-identical
vllm_neuron/vllm/attention/attn.py              # decode-attn mask kernel block size — shared, byte-identical
vllm_neuron/vllm/platform.py                    # Hybrid block-size align + mamba/APC guards — shared (comment-only delta)
vllm_neuron/vllm/core/scheduler.py              # Pool-aware admission gate (livelock fix) — shared, byte-identical
vllm_neuron/vllm/worker/neuron_model_runner.py  # Hybrid state alloc / bind / slot mgmt — shared, byte-identical
vllm_neuron/vllm/worker/neuron_worker.py        # Unified-KV budget cap — shared, byte-identical
docs/model-recipes/index.md                     # Add the recipe grid card + toctree entry
docs/tutorials/index.md                         # Add the tutorial grid card + toctree entry
```

> **This branch is fully self-contained** — it carries every file it needs to build and
> serve on its own (the model package, the GatedDeltaNet NKI kernels above, and the
> hybrid framework edits), with no runtime dependency on any other branch. The GDN
> kernels and hybrid machinery are the same design as the sibling `qwen3_5_moe` port (see
> [`Qwen3.6-35B-A3B/README.md`](https://github.com/AWShtokoyo/vllm-neuron/tree/add-qwen36-moe/Qwen3.6-35B-A3B/README.md)) — 6 of the 7 framework
> touch-points are byte-identical to it; the only dense-specific framework edit is the
> `registry.py` arch aliases.
