# qwen3_5_moe (Qwen3.6-35B-A3B, text-only MoE hybrid)

vllm-neuron model implementation for `Qwen/Qwen3.6-35B-A3B` (architecture alias
`Qwen3_5MoeForCausalLM`, `model_type: qwen3_5_moe`) — a **hybrid Mixture-of-Experts**
decoder (30 GatedDeltaNet + 10 full-attention layers, 256-expert top-8 MoE + shared
expert), served **text-only**.

> **The single source of truth for this port** — architecture, setup, serving,
> verification, and measured performance — is the bundle README:
> [`Qwen3.6-35B-A3B/README.md`](../../../Qwen3.6-35B-A3B/README.md). This file
> documents only the module structure of the package.

## Module Structure

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports. |
| `config.py` | HF → Neuron config translation: the 3:1 hybrid layer pattern (full-attn indices), MoE (256 experts / top-8 / shared expert), and GatedDeltaNet dims. Forces the text-only decoder path off the native VLM checkpoint. |
| `factory.py` | Model builder + weight-loader wiring (selects the BF16 loaders, binds the hybrid mamba state). |
| `model.py` | The hybrid MoE decoder: 30 GatedDeltaNet (linear-attention) layers + 10 full-attention layers, 256-expert top-8 MoE + shared expert. Keeps the vLLM hybrid interfaces (`HasInnerState` / `IsHybrid`, `get_mamba_state_shape_from_config()`, `get_mamba_state_dtype()`, `bind_mamba_state()`). |
| `weight_loaders_bf16.py` | BF16 sharded / fused weight loaders for TP8 / EP8 (per-rank expert sharding). |

The GatedDeltaNet NKI kernels this package calls live under `vllm_neuron/functional/`
(`gated_delta_rule_seq.py`, `gdn_conv_update*.py`, `gdn_state_update*.py`,
`paged_kv_gather.py`, `slot_indirect_probe.py`); the hybrid mamba prefix-cache sidecar
is `vllm_neuron/vllm/worker/neuron_mamba_apc.py`. See the bundle README for how they
fit together.

## Running

Offline example:
[`examples/vllm_neuron/models/qwen3_5_moe/run.py`](../../../examples/vllm_neuron/models/qwen3_5_moe/run.py)
(sets the required recipe flags via `os.environ.setdefault`, so `python run.py` works
standalone). The full tested TP8/EP8 serving recipe, required environment flags, and
the segmented-prefill configuration are in the
[bundle README](../../../Qwen3.6-35B-A3B/README.md).
