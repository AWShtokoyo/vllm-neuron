# Run Configs for qwen3_5_moe (Qwen3.6-35B-A3B MoE hybrid)

The one authoritative reference for **how to run this model correctly** on Neuron
(trn2). These are the *tested* configs — the ones that passed device runs. Deviating
from them (especially the segmented-prefill and PATH items) has caused real failures.

## COPY-PASTE ENV BLOCK (source of truth — do NOT hand-pick a subset)

Set `<scratch>` to a large volume (not `/`) and `<devs>` to a free device range, then paste ALL of it.
Every var here is load-bearing — omitting one causes a failed run (each was learned the hard way; see the
pre-flight gotchas below). This is the exact working set; do not "clean it up" into fewer vars.

```bash
export SCRATCH=<scratch>                 # e.g. /opt/dlami/nvme/<you>  — large volume, NOT /
export WS=<path-to>/vllm-neuron          # the vllm-neuron repo root
export PATH=$SCRATCH/venv/bin:$PATH      # venv compiler FIRST (else neuronx-cc 2.23 shadow → NKI skew)
export PYTHONPATH=$WS${PYTHONPATH:+:$PYTHONPATH}
# --- caches off root (relocate; box-path only) ---
export TMPDIR=$SCRATCH/tmp XDG_CACHE_HOME=$SCRATCH/xdg_cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/torchinductor_cache TRITON_CACHE_DIR=$SCRATCH/triton_cache
export PIP_CACHE_DIR=$SCRATCH/pip_cache
# --- LOAD-BEARING (omit any → failed run) ---
export HF_HOME=$SCRATCH/hf_cache HF_HUB_CACHE=$SCRATCH/hf_cache/hub HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export VLLM_CACHE_ROOT=$SCRATCH/vllm_cache                       # relocates NEFF + NKI compile cache
export NEURON_CC_FLAGS="--cache_dir=$SCRATCH/neuron_compile_cache"
export VLLM_NEURON_GOLDEN_CACHE_DIR=$SCRATCH/vllm_neuron_goldens # else equiv/TF recomputes FP32+BF16 HF → OOM
export NXDI_CHECKPOINT_CACHE=$SCRATCH/vllm_neuron_checkpoints
export QWEN3_5_MOE_CHECKPOINT=$SCRATCH/checkpoints/Qwen3.6-35B-A3B
# --- device (TP serve): NEURON_VISIBLE_DEVICES, NEVER NEURON_RT_VISIBLE_CORES ---
export NEURON_VISIBLE_DEVICES=<devs>     # e.g. 0-7 ; pick a range not held by another job
export NEURON_SKIP_EFA_AFFINITY=1 VLLM_NEURON_SWITCH_CC=1
# --- recipe flags (default-OFF, REQUIRED for the validated path) ---
export VLLM_GDN_SEQ_NKI=1 VLLM_UNIFIED_KV_GATHER=1 VLLM_MOE_TKG_ROUTER_FP32=1
# --- always before a device run ---
unset VLLM_NEURON_CPU_MODE NKI_SIMULATOR VLLM_NEURON_CPU_COMPILE
```

## Non-negotiable environment (per-item detail)

| Item | Value | Why |
|------|-------|-----|
| **PATH** | venv `bin` FIRST: `export PATH=<venv>/bin:$PATH` | A stray `~/.local/bin/neuronx-cc` (2.23) shadows the venv's compiler → NKI binary-ID skew (`[NCC_INLA001] 21d0259`). Set PATH venv-first in your own shell. |
| **Unified-cache KV gather** | `VLLM_UNIFIED_KV_GATHER=1` | Enables the block-indexed `.ap` page-strided full-attn KV gather (`_unified_kv_gather_on()`, model.py:135). **Live flag, default-OFF** — without it the model uses the torch-fallback KV path. (The old `VLLM_GDN_SLOT_INDEXED/SLOT_KERNEL/SLAB_PACKED` flags are RETIRED — dead, no live read; the MambaSpec slot routing is unconditional in code.) |
| **MoE decode router** | `VLLM_MOE_TKG_ROUTER_FP32=1` | fp32 router for correct top-8 selection at decode. Live flag, default-OFF. |
| **GDN prefill (seq≥512)** | `VLLM_GDN_SEQ_NKI=1` | bounded-graph GDN prefill (see next section). Live flag, default-OFF. |
| **TP / EP** | `--tensor-parallel-size 8 --enable-expert-parallel`, `ep_degree: 8` | pure EP8, 32 experts/rank. |
| **Text-only** | `--limit-mm-per-prompt '{"image":0,"video":0}'` + `--hf-overrides '{"architectures":["Qwen3_5MoeForCausalLM"]}'` | skip the vision tower; avoids the vision-bucket auto-config gate. |
| **Cores (⚠ TP serving)** | serving/TP: `NEURON_VISIBLE_DEVICES=<N cores, e.g. 0-7>`. **NEVER `NEURON_RT_VISIBLE_CORES` for a multiproc TP serve** — it hard-errors `"NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution"`. Module tests (single MPExecutor) use `NEURON_RT_VISIBLE_CORES`. Do NOT mix. | TP8 = 8 logical cores. |
| **Golden + HF cache (⚠ or compiler OOM)** | `VLLM_NEURON_GOLDEN_CACHE_DIR=<scratch>/vllm_neuron_goldens` **and** `HF_HOME=<scratch>/hf_cache` | The equivalence/TF harness computes FP32 **and** BF16 HF reference logits IN-PROCESS. Without a populated golden cache it loads TWO full 35B models concurrently with the 8-rank device compile → starves the compiler → `NCC_EOOM002` / `compilation failed with 70`. Pre-cache goldens (`precompute_goldens`) and point the dir here. |
| **Compiler flags (⚠ do NOT set NEURON_CC_FLAGS)** | leave `NEURON_CC_FLAGS` UNSET — let the framework compose them | Setting `NEURON_CC_FLAGS="--cache_dir=..."` **REPLACES** (not appends) the framework's flag string, dropping the required `--internal-backend-options=--enable-verifier=false` (+ `--auto-cast=none -O1 --internal-hlo2tensorizer-options=...`) → the verifier runs → `NCC_EVRF029` → exit 70. Use `VLLM_CACHE_ROOT` for cache location instead (it relocates the NKI/NEFF cache under `neuron/compile_cache` automatically). |
| **Caches off root** | `VLLM_CACHE_ROOT=<scratch>/...` | point the cache at a large scratch volume, not `/` (near-full on these boxes). This ALSO relocates `neuron/compile_cache` + the NKI cache — no separate flag needed. |

### Pre-flight checklist (the traps that each cost a failed device run)
Every one of these was a launcher mistake, not a model bug — verify before launching a TP8 serve/equiv run:
1. `NEURON_VISIBLE_DEVICES=<cores>` (NOT `NEURON_RT_VISIBLE_CORES`) → else EngineCore-init hard-error.
2. `VLLM_NEURON_GOLDEN_CACHE_DIR` + `HF_HOME` set, goldens pre-cached → else in-process HF-model OOM → `NCC_EOOM002`.
3. `NEURON_CC_FLAGS` UNSET (framework composes flags) → else drops `--enable-verifier=false` → `NCC_EVRF029` / exit 70.
4. `unset VLLM_NEURON_CPU_MODE NKI_SIMULATOR VLLM_NEURON_CPU_COMPILE` (else a "device" run fakes a CPU pass).
5. `pgrep -af "VLLM::|EngineCore|neuronx-cc"` → pick a device range disjoint from any other job (don't collide).

## CRITICAL: bounded-graph GDN prefill (seq >= 512)

**The unsegmented seq>=1024 GDN prefill traces one flat T-deep graph whose
token-position scatter trips a runtime scatter/gather OOB (indirect vector-DGE)
→ NaN logits.** Every long-seq run MUST use a bounded-graph GDN prefill.

There are THREE prefill options in `model.py` (do not confuse them):

| Path | Env gate | Status |
|------|----------|--------|
| **SEQ-NKI** (recurrence-in-kernel) | `VLLM_GDN_SEQ_NKI=1` | **the device-validated recipe** (gsm8k exact-match 95.0%, 0 OOB). USE THIS. |
| Segmented-sequential (host loop) | `VLLM_GDN_SEGMENTED=1` + `VLLM_GDN_SEG_SIZE=512` | alternative; bit-exact in CPU-sim but NOT the run that passed gsm8k on device |
| Unsegmented monolith | (none) | seq>=1024 → scatter/gather OOB → NaN. DO NOT USE at long seq. |

The **exact device-validated seq1024 recipe**:
```
export VLLM_GDN_SEQ_NKI=1                              # bounded-graph GDN prefill
--max-model-len 1024 --max-num-batched-tokens 512      # batched <= 512
additional-config neuron_config: "kv_segment_size_buckets": [512],
                                  "num_batched_tokens_buckets": [512]
```
`kv_segment_size_buckets` / `num_batched_tokens_buckets` must be **<= 512** so no
1024-extent graph is built. Short prompts (<= 256) run as a single segment and need
none of these flags (validated seq256 offline smoke).

> NOTE: any change to the segmentation/seq-len/bucket config changes the traced graph
> → new compile-cache key → full cold recompile (~40 min). Freeze ONE recipe; do not
> sweep buckets. See "Do / Don't" below.

## Tested serve configs

### Offline / logit-validation / gsm8k (NON-APC) — PASSING recipe
```
export PATH=<venv>/bin:$PATH              # venv compiler FIRST (avoid neuronx-cc shadow)
export VLLM_GDN_SEQ_NKI=1                  # bounded-graph GDN prefill (REQUIRED for seq>=512)
export VLLM_UNIFIED_KV_GATHER=1           # unified block-indexed KV gather (else torch fallback)
export VLLM_MOE_TKG_ROUTER_FP32=1         # fp32 decode router (correct top-8 selection)
vllm serve $CKPT --tensor-parallel-size 8 --enable-expert-parallel \
  --max-model-len 1024 --max-num-batched-tokens 512 --max-num-seqs 1 \
  --no-enable-log-requests \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --hf-overrides '{"architectures":["Qwen3_5MoeForCausalLM"]}' \
  --additional-config '{"neuron_config":{"quantization":"bf16","ep_degree":8,
      "on_device_sampling_config":{"all_greedy":"true"},
      "kv_segment_size_buckets":[512],"num_batched_tokens_buckets":[512],
      "num_seqs_buckets":[1]}}'
```
gsm8k (exact-match 95.0%, bs1) + logit-validation passed on device (trn2, TP8/EP8)
with this exact recipe. For a fast seq256 offline smoke (no long-seq flags needed) use
`examples/vllm_neuron/models/qwen3_5_moe/run.py`.

### Multi-batch throughput (NON-APC) — PASSING recipe: keep max-model-len 1024
```
# same env block as above; only the batch knobs change
vllm serve $CKPT --tensor-parallel-size 8 --enable-expert-parallel \
  --max-model-len 1024 --max-num-batched-tokens 512 --max-num-seqs 8 \
  --no-async-scheduling --no-enable-log-requests \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --hf-overrides '{"architectures":["Qwen3_5MoeForCausalLM"]}' \
  --additional-config '{"neuron_config":{"quantization":"bf16","ep_degree":8,
      "on_device_sampling_config":{"all_greedy":"true"},
      "kv_segment_size_buckets":[512],"num_batched_tokens_buckets":[512],
      "num_seqs_buckets":[8]}}'
```
Verified on device (trn2, TP8/EP8, `vllm bench serve`): bs4 ~59 tok/s and bs8
~131 tok/s, 24/24 completed, KV usage ~1–6%. **Use `max-model-len 1024` (with
`kv_segment_size_buckets:[512]`) for best multi-batch throughput.**

A tight `max-model-len 512` + bs>1 makes each request reserve its full worst-case
KV allocation up front, so the hybrid (attention + GatedDeltaNet) unified block pool
can saturate at only ~3 concurrent decodes (~803 blocks/request, dominated by the
GatedDeltaNet mamba page, not the 256-token attention block). Historically a 4th
request then could neither be admitted nor preempt: `schedule()` Step 3.2 hid all
running decodes to make room for the waiting prefill, but the base scheduler could
not allocate that prefill (only ~303 of 2713 blocks free) and could not preempt the
now-hidden decodes → `scheduled_tokens=0` indefinitely (py-spy: all workers idle in
shm `acquire_read`, EngineCore in `time.sleep` at `core.py` `_process_engine_step`
with `has_unfinished_requests()` true; KV pinned at 88.8%). `_max_kv_concurrent`
does not catch this because it counts request concurrency by tokens (the
attention-only `get_max_concurrency_for_kv_cache_config` = 542x) and is blind to the
mamba/GDN page footprint.

**Fixed by a pool-aware admission gate** (`NeuronScheduler._pool_admission_ok`, in
`can_schedule`): before admitting a prefill it predicts that request's worst-case
block footprint via the KV manager's own coordinator
(`get_num_blocks_to_allocate(..., apply_admission_cap=True)`, the same math as
`allocate_slots(full_sequence_must_fit=True)`) and defers admission when
`need > free`, instead of dead-ending into empty batches. So `max-model-len 512` +
bs>1 now runs correctly — the gate throttles admitted concurrency to what the pool
can hold. Device A/B (trn2, TP8/EP8, warm cache): gate ON bs4/mml512 in256/out128 →
24/24 completed, ~68.6 tok/s, peak KV 1.2%, 0 empty-batch steps, 2 pool deferrals;
gate OFF (control) → stall, ~497k empty-batch steps; gate ON bs8/mml1024
(non-regression) → 24/24, ~90 tok/s, 0 deferrals (gate transparent when the pool has
room). Kill-switch: `VLLM_NEURON_POOL_ADMISSION_GATE=0` (default on; not recommended
for hybrid models with a tight `max-model-len`). The gate lives in the
model-agnostic baseline `NeuronScheduler`, so it applies to any hybrid model, not
just Qwen3.6. Not applicable at bs1 (single request never saturates the pool).

### APC (prefix caching) — EXPERIMENTAL: works below block size, hangs on cross-block reuse
```
vllm serve $CKPT --tensor-parallel-size 8 --enable-expert-parallel \
  --max-model-len 512 --max-num-batched-tokens 512 --max-num-seqs 2 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --hf-overrides '{"architectures":["Qwen3_5MoeForCausalLM"]}' \
  --additional-config '{"neuron_config":{"quantization":"bf16","ep_degree":8,
      "on_device_sampling_config":{"all_greedy":"true"},
      "num_batched_tokens_buckets":[512],"num_seqs_buckets":[2]}}' \
  --enable-prefix-caching
```
`--enable-prefix-caching` forces mamba `align` mode which REQUIRES chunked prefill
(+ async scheduling) — you cannot disable those. A >256-token prompt (real
cross-block reuse) currently **hangs in `sample_tokens`** on the partial-prefill
chunk. Short prompts (<256) run correctly but exercise no reuse.

## Do / Don't (learned the hard way)

- **DO** run one device-compile job at a time, or give each its own
  `VLLM_CACHE_ROOT` + `NKI_COMPILE_CACHE_URL` — concurrent fresh compiles racing the
  shared `/var/tmp/nki-intermediate-cache` cause `[NCC_EVRF059]` missing-kernel errors.
- **DON'T** `rm -rf /var/tmp/nki-intermediate-cache` while any device job is running
  (or while cached vLLM graphs reference its `.json` kernels) — it strands graphs → EVRF059.
- **DON'T** run seq>=1024 without the segmented config above.
- **NEVER** clear `/var/tmp/neuron-compile-cache` (shared system dir).

## Environment requirements (version-bound, not path-bound)

- checkpoint: a local `Qwen3.6-35B-A3B` HF checkpoint (place on a large scratch volume, not `/`).
- venv/compiler: **neuronx-cc 2.0.262175 + nki g58374325**, on PATH FIRST (see the PATH row above).
- transformers **5.12.1** provides the HF `Qwen3_5Moe` reference.
- (Machine-specific paths — venv dir, checkpoint dir, scratch root, device ids — are intentionally NOT
  pinned here; this doc is branch-bound. Supply them from your own shell/run script.)
