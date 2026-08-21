# qwen3_5_dense (Qwen3.6-27B, text-only dense hybrid)

vllm-neuron model implementation for `Qwen/Qwen3.6-27B` (architecture alias
`Qwen3_5ForCausalLM`, `model_type: qwen3_5`) — a **hybrid dense** decoder (48
GatedDeltaNet + 16 full-attention layers, plain SwiGLU MLP), served **text-only**. It
is the **dense** sibling of the [`qwen3_5_moe`](../qwen3_5_moe/README.md) port
(Qwen3.6-35B-A3B): identical hybrid attention stack, differing only in the FFN (dense
SwiGLU vs 256-expert MoE), so there is **no expert parallelism**.

> **The single source of truth for this port** — architecture, setup, serving,
> verification, and measured performance — is the bundle README:
> [`Qwen3.6-27B/README.md`](../../../Qwen3.6-27B/README.md). This file documents only
> the module structure of the package.

## Module Structure

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports. |
| `config.py` | HF → Neuron config translation: the 3:1 hybrid layer pattern (`full_attention_interval=4`), the dense SwiGLU FFN, and the GatedDeltaNet / linear-attention dims. Forces the text-only dense decoder path off the native VLM checkpoint. |
| `factory.py` | Model builder + IsHybrid classmethod delegation. |
| `model.py` | The hybrid dense decoder: 48 GatedDeltaNet (linear-attention) layers + 16 full-attention layers, plain SwiGLU MLP. Keeps the vLLM hybrid interfaces (`HasInnerState` / `IsHybrid`, `get_mamba_state_shape_from_config()`, `get_mamba_state_dtype()`, `bind_mamba_state()`). |
| `weight_loaders_bf16.py` | BF16 GatedDeltaNet TP head-sharding loaders (the dense FFN uses the framework's standard loader). |

The GatedDeltaNet NKI kernels this package calls (`vllm_neuron/functional/gdn_conv_update`,
`gdn_state_update`, `gated_delta_rule*`, `paged_kv_gather`) and the hybrid framework edits
are **carried on this branch in-tree**, so the port builds and serves standalone with no
dependency on any other branch. They are the same design as the sibling `qwen3_5_moe`
port; the only dense-specific framework edit is the registry arch aliases. See the bundle
README for how they fit together.

## Running

Offline example:
[`examples/vllm_neuron/models/qwen3_5_dense/run.py`](../../../examples/vllm_neuron/models/qwen3_5_dense/run.py)
(sets the required recipe flags via `os.environ.setdefault`, so `python run.py` works
standalone). The full tested TP=4 serving recipe, required environment flags, the
QKV-matmul SBUF workaround, and the bounded-graph GDN configuration are in the
[bundle README](../../../Qwen3.6-27B/README.md).
