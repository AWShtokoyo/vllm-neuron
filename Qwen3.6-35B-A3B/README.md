# Qwen3.6-35B-A3B (Qwen3.5-MoE hybrid) — vllm-neuron bundle

vllm-neuron implementation of
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — a **hybrid
Mixture-of-Experts** decoder (native arch `Qwen3_5MoeForConditionalGeneration`,
`model_type: qwen3_5_moe`), **served text-only** as `Qwen3_5MoeForCausalLM`. 40
layers in a 3:1 pattern — **30 GatedDeltaNet linear-attention + 10 full-attention**
— and every FFN is a 256-expert top-8 MoE plus a shared expert. BF16 weights.
Validated on `trn2.48xlarge` at **TP8 / EP8** (gsm8k-CoT exact-match **95.0%**).

> **Qwen3.5 and Qwen3.6 share this architecture** (weights-only difference; HF loads
> both under `qwen3_5_moe`). Validated on device with Qwen3.6-35B-A3B weights; serves
> Qwen3.5-35B-A3B unchanged.

**The model is integrated in-tree** on this branch (`vllm_neuron/model/qwen3_5_moe/`),
so it is installed simply by installing the framework from this branch. This
top-level `Qwen3.6-35B-A3B/` directory is a **slim bundle**: the integration patch
(for applying the shared-file edits onto a *pristine* release tree) plus a
self-contained integration test. It is supplementary — there is **no `src/` copy**
here (that would duplicate the in-tree package).

> Framework: public [vLLM Neuron Plugin](https://github.com/vllm-project/vllm-neuron)
> release **`0.21.0.1.0.0`** (Neuron 2.31 = neuronx-cc 2.26 / nki 0.5.0 / torch 2.11).

## Documentation
The authoritative model docs live under `docs/` (in-tree, MyST):
- **Model recipe / card** — [`docs/model-recipes/qwen3-6-moe.md`](../docs/model-recipes/qwen3-6-moe.md)
- **E2E tutorial** — [`docs/tutorials/tutorial-qwen3-6-moe.md`](../docs/tutorials/tutorial-qwen3-6-moe.md)
- **Run configs (the one authoritative "how to run correctly" reference)** —
  [`vllm_neuron/model/qwen3_5_moe/doc/RUN_CONFIGS.md`](../vllm_neuron/model/qwen3_5_moe/doc/RUN_CONFIGS.md)
- **Offline example runner** —
  [`examples/vllm_neuron/models/qwen3_5_moe/run.py`](../examples/vllm_neuron/models/qwen3_5_moe/run.py)
  (bakes in the validated recipe flags via `os.environ.setdefault`).

Required recipe flags (all default-OFF, load-bearing — the example runner and the
bundle `conftest.py` set them for you): `VLLM_GDN_SEQ_NKI=1` (bounded-graph GDN
prefill, required for seq ≥ 512), `VLLM_UNIFIED_KV_GATHER=1` (block-indexed unified
KV gather), `VLLM_MOE_TKG_ROUTER_FP32=1` (fp32 decode router). See `RUN_CONFIGS.md`
for the full copy-paste environment and the pre-flight checklist.

## Contents of this bundle
| Path | Purpose |
|---|---|
| `integration.patch` | The 7 shared-file edits below, as a patch to apply onto a **pristine** `0.21.0.1.0.0` tree (`-p1` from the `vllm_neuron` package root). Already committed in-tree on this branch. |
| `test/integration/conftest.py` | Autouse fixture that sets the required recipe env vars before `import vllm` (restored on teardown). |
| `test/integration/test_scheduler_livelock.py` | Self-contained hybrid pool-saturation regression test (see **Testing**). |

## Installation
The model is in-tree, so the normal path is just to install the framework from this
branch (`pip install -e .` from the repo root). Use `integration.patch` only to graft
the shared-file edits onto a **separate, pristine** `0.21.0.1.0.0` checkout:

```bash
# From a pristine vllm-neuron 0.21.0.1.0.0 checkout, after copying the in-tree
# vllm_neuron/model/qwen3_5_moe/ package + the new vllm_neuron/functional/*.py
# kernels + vllm_neuron/vllm/worker/neuron_mamba_apc.py into place.
SP=$(python -c "import os, vllm_neuron; print(os.path.dirname(vllm_neuron.__file__))")
PATCH="$PWD/integration.patch"
( cd "$SP/.." && git apply --check "$PATCH" && git apply "$PATCH" )   # or: patch -p1 --fuzz=3 < "$PATCH"
```

The patch touches **7 shared files** (all hybrid code is `MambaSpec` /
`_mamba_apc_enabled()`-gated, so the non-hybrid path stays byte-identical) — see the
**Modified** table below.

## Testing
```bash
# Self-contained hybrid pool-saturation livelock regression (needs a trn2 with 8
# Neuron devices + the checkpoint staged locally). conftest sets the recipe flags.
QWEN3_5_MOE_CHECKPOINT=/path/to/Qwen3.6-35B-A3B \
  python3 -m pytest test/integration/test_scheduler_livelock.py -v -s
```
This reproduces the config that livelocked before the pool-aware admission gate
(`max_model_len=512` + batch size > 1): each request reserves its full worst-case KV
allocation, the GatedDeltaNet mamba page dominates (~800 blocks/request on device),
and the hybrid unified block pool saturates at ~3 concurrent decodes. The gate
(`NeuronScheduler._pool_admission_ok`, on by default) defers admission when a
prefill would not fit the free pool instead of dead-ending into empty batches; the
test asserts forward progress (every prompt completes). To observe the pre-fix
behaviour, set `VLLM_NEURON_POOL_ADMISSION_GATE=0` and expect a timeout.

The in-tree CPU / NKI-sim numeric suite (module + kernel + real-checkpoint 3-way
accuracy) lives under `test/vllm_neuron/model/qwen3_5_moe/` and passes 107 + 1
expected-xfail; see `RUN_CONFIGS.md` and the tutorial for the device gsm8k / logit
recipes.

## Files added / modified
The port touches the following paths across the repository (relative to the repo
root). The model is integrated in-tree, so most of these are committed directly on
this branch; the `Qwen3.6-35B-A3B/` bundle above is supplementary.

**New — model package** (`vllm_neuron/model/qwen3_5_moe/`)
| Path | Purpose |
|---|---|
| `vllm_neuron/model/qwen3_5_moe/model.py` | The hybrid MoE decoder (30 GatedDeltaNet + 10 full-attention layers, 256-expert top-8 MoE + shared expert). |
| `vllm_neuron/model/qwen3_5_moe/config.py` | HF → Neuron config translation (hybrid layer pattern, MoE, GDN/linear-attn dims). |
| `vllm_neuron/model/qwen3_5_moe/factory.py` | Model builder + weight-loader wiring. |
| `vllm_neuron/model/qwen3_5_moe/weight_loaders_bf16.py` | BF16 sharded / fused weight loaders (TP8 / EP8). |
| `vllm_neuron/model/qwen3_5_moe/__init__.py` | Package exports. |
| `vllm_neuron/model/qwen3_5_moe/README.md` | Package-level notes. |
| `vllm_neuron/model/qwen3_5_moe/doc/RUN_CONFIGS.md` | Authoritative run-recipe reference. |

**New — NKI kernels + hybrid sidecar**
| Path | Purpose |
|---|---|
| `vllm_neuron/functional/gated_delta_rule_seq.py` | Bounded-graph sequential GDN prefill kernel (`VLLM_GDN_SEQ_NKI` — the validated path). |
| `vllm_neuron/functional/gated_delta_rule.py` | Chunked GDN delta-rule prefill kernel (present, not enabled on this hybrid). |
| `vllm_neuron/functional/gdn_conv_update.py` | GDN causal-conv state update (decode). |
| `vllm_neuron/functional/gdn_conv_update_compact.py` | Compact variant of the conv-state update. |
| `vllm_neuron/functional/gdn_state_update.py` | GDN recurrent-state update (decode). |
| `vllm_neuron/functional/gdn_state_update_compact.py` | Compact variant of the recurrent-state update. |
| `vllm_neuron/functional/paged_kv_gather.py` | Block-indexed unified KV gather (`.ap` page-strided). |
| `vllm_neuron/functional/slot_indirect_probe.py` | Slot-indexed indirect-probe helper. |
| `vllm_neuron/vllm/worker/neuron_mamba_apc.py` | Hybrid mamba prefix-cache sidecar. |

**New — docs, example, bundle**
| Path | Purpose |
|---|---|
| `docs/model-recipes/qwen3-6-moe.md` | Model card / recipe (MyST). |
| `docs/tutorials/tutorial-qwen3-6-moe.md` | E2E tutorial (MyST). |
| `examples/vllm_neuron/models/qwen3_5_moe/run.py` | Offline runner (bakes recipe flags). |
| `Qwen3.6-35B-A3B/` | This slim bundle (integration.patch + self-contained test + README). |

**Modified — shared framework touch-points**
| Path | Change |
|---|---|
| `vllm_neuron/model/registry.py` | Register the `Qwen3_5MoeForConditionalGeneration` / `Qwen3_5MoeForCausalLM` arch aliases. |
| `vllm_neuron/model/kv_cache.py` | Add `HybridKVSpec` (a `KVSpec` subclass carrying the stateful/GDN layer names; non-hybrid readers of `.layers` unchanged). |
| `vllm_neuron/vllm/attention/attn.py` | Declare `get_supported_kernel_block_sizes → MultipleOf(128)` so the grown hybrid `block_size` stays a 128-multiple for the decode-attn mask kernel. |
| `vllm_neuron/vllm/platform.py` | Hybrid block-size align (pad the mamba page to the full-attn page via vLLM's own `_align_hybrid_block_size`) + mamba-mode / APC-async guards + the text-only **vision-gate guard**. |
| `vllm_neuron/vllm/core/scheduler.py` | **Pool-aware admission gate (`NeuronScheduler._pool_admission_ok`) + empty-batch diagnostic — the `max_model_len=512` + bs>1 hybrid-pool livelock fix.** |
| `vllm_neuron/vllm/worker/neuron_model_runner.py` | Hybrid state alloc / bind / slot management + mamba-page-padded KV spec (`mamba_page_size_padded`). |
| `vllm_neuron/vllm/worker/neuron_worker.py` | Cap the unified-KV budget so the shared unified slab stays within the `.ap` slab size. |
| `docs/model-recipes/index.md` | Add the recipe grid card + toctree entry. |
| `docs/tutorials/index.md` | Add the tutorial grid card + toctree entry. |

## Example Checkpoints
- `Qwen/Qwen3.6-35B-A3B` (also serves `Qwen/Qwen3.5-35B-A3B`)
