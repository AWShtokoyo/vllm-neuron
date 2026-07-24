# Devstral-2-123B-Instruct-2512 (Ministral3) — deployment artifacts

Deployment artifacts for
[`mistralai/Devstral-2-123B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512)
on `vllm-neuron` — a 123B dense-GQA causal LM (model_type `ministral3`,
architecture `Ministral3ForCausalLM`) with a per-tensor static FP8 (E4M3)
checkpoint.

**The model is already integrated into this repository** at
[`vllm_neuron/model/ministral3/`](../vllm_neuron/model/ministral3/) and registered
in `vllm_neuron/model/registry.py`, so a build of this fork serves Devstral out of
the box in the default **BF16-dequant** path. This directory holds the
supporting artifacts that do not belong in the framework package:

| Path | What it is |
|---|---|
| `test/integration/` | On-device sanity run, tiny CPU end-to-end test, `vllm bench serve` harnesses (TP=8 DP=1/DP=8, FP8 A/B, card-tests, multi-bucket TTFT A/B) and their recorded result JSONs |
| `integration.patch` | The framework-side edits that register the model (already applied in this fork; kept as a portable patch for a pristine `vllm-neuron 0.21.0.1.0.0` install) |
| `integration_nkilib.patch` | **`nkilib` kernel** patch that enables the **FP8-native** path — a *separate package* from `vllm-neuron`, so it cannot live in this tree |

## Files added / modified

The port touches the following paths across the repository (relative to the
repo root). The model is integrated in-tree, so most of these are committed
directly on this branch; the `Devstral-2-123B-Instruct-2512/` bundle above is
supplementary.

**New — model package** (`vllm_neuron/model/ministral3/`)

| Path | Purpose |
|---|---|
| `model.py` | Ministral3 (Devstral) dense-GQA causal LM |
| `config.py` | `Ministral3Config` — builds the Neuron config from the HF config |
| `factory.py` | model construction / weight-loading factory |
| `weight_loaders.py` | per-tensor static FP8 (E4M3) + BF16-dequant weight loaders |
| `__init__.py` | package exports |

**New — docs, example, bundle**

| Path | Purpose |
|---|---|
| `docs/model-recipes/devstral-2-123b.md` | model recipe / card |
| `docs/tutorials/tutorial-devstral-2-123b.md` | deployment tutorial |
| `examples/vllm_neuron/models/ministral3/run.py` | offline generation example |
| `Devstral-2-123B-Instruct-2512/` | this artifact bundle (README, `integration.patch`, `integration_nkilib.patch`, `test/integration/`) |

**Modified — shared framework touch-points**

| Path | Change |
|---|---|
| `vllm_neuron/model/registry.py` | register `Ministral3ForCausalLM` |
| `vllm_neuron/model/__init__.py` | add `ministral3` to the lazy-import allowlist |
| `vllm_neuron/__init__.py` | register the `ministral3` HuggingFace `AutoConfig` (picklable for `data_parallel_size>1`) |
| `vllm_neuron/vllm/platform.py` | pre-register Ministral3 with a pre-built `_ModelInfo` (`is_text_generation_model=True`) at plugin load |
| `README.md` | add Devstral-2-123B to the Contributed Models list |
| `docs/model-recipes/index.md` | link the model recipe |
| `docs/tutorials/index.md` | link the tutorial |

## Documentation

Full deployment guidance lives in the repository docs:

- **Model card:** [`docs/model-recipes/devstral-2-123b.md`](../docs/model-recipes/devstral-2-123b.md)
  — checkpoints, feature status, quantization modes, accuracy, performance.
- **Tutorial:** [`docs/tutorials/tutorial-devstral-2-123b.md`](../docs/tutorials/tutorial-devstral-2-123b.md)
  — environment setup, download, online serving, offline inference, FP8-native,
  whole-box (DP=8) throughput.

## FP8-native (optional)

The default path (BF16-dequant) needs nothing beyond a normal build of this fork.
The optional **FP8-native** path (up to −50% weight HBM, +19% decode) requires
`integration_nkilib.patch` applied to the installed `nkilib` (bundled inside
`neuronx-cc`), then `neuron_config.quantization = "fp8:qkv,o_proj"` (BF16-faithful)
or `"fp8:qkv,o_proj,mlp"` (full FP8, also set `NKILIB_MLP_BF16_XPOSE_SRC=1`):

```bash
NKILIB="$(python3 -c 'import os,nkilib; print(os.path.dirname(nkilib.__file__))' | tail -1)"
( cd "$(dirname "$NKILIB")" && git apply -p1 /path/to/integration_nkilib.patch )
```

See the [tutorial](../docs/tutorials/tutorial-devstral-2-123b.md) (Step 5) for the
full FP8-native walkthrough and the
[model card](../docs/model-recipes/devstral-2-123b.md) for the accuracy tradeoff.

## `integration.patch` (already applied here)

For reference, `integration.patch` touches four framework files (all additive,
model-isolated); this fork already carries them:

| File | Change |
|---|---|
| `vllm_neuron/model/registry.py` | register `("Ministral3ForCausalLM", Ministral3ForCausalLM)` |
| `vllm_neuron/model/__init__.py` | add `ministral3` to the lazy-import set |
| `vllm_neuron/__init__.py` | `_register_ministral3_hf_config()` → `AutoConfig.register("ministral3", ...)` (module-scope for DP pickling) |
| `vllm_neuron/vllm/platform.py` | `_ModelInfo` pre-registration for `Ministral3ForCausalLM` |

To apply the same integration onto a *pristine* `vllm-neuron 0.21.0.1.0.0` (e.g. a
`pip install`ed copy) instead of this fork, drop `vllm_neuron/model/ministral3/`
into the package and `git apply -p1 integration.patch` from the package root.

## Testing

```bash
# Tiny CPU end-to-end (NKI simulator): load → prefill → decode → sampling, TP=1 & TP=2.
# Needs pytest (test extra): pip install pytest
VLLM_NEURON_CPU_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python3 -m pytest test/integration/test_tiny_ministral3_e2e.py --capture=tee-sys
#   → 2 passed

# On-device sanity run (trn2, real weights)
python3 test/integration/run.py --tensor-parallel-size 8 --max-model-len 2048

# Online serving (trn2)
bash test/integration/run_devstral_serve.sh    # vllm serve on :8000
```
