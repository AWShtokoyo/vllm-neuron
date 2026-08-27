# qwen3_5_dense (Qwen3.6-27B, dense hybrid, text + vision)

vllm-neuron model implementation for `Qwen/Qwen3.6-27B` (architecture aliases
`Qwen3_5ForCausalLM` for text-only and `Qwen3_5ForConditionalGeneration` for text +
vision, `model_type: qwen3_5`) — a **hybrid dense** decoder (48 GatedDeltaNet + 16
full-attention layers, plain SwiGLU MLP). The vision alias wires the byte-for-byte
Qwen3-VL ViT (`vllm_neuron/model/qwen3_vl/`) into the hybrid backbone via the
encoder-cache / EPD path; the text-only alias leaves the ViT unbuilt. It is the
**dense** sibling of the `qwen3_5_moe` port
(Qwen3.6-35B-A3B): identical hybrid attention stack, differing only in the FFN (dense
SwiGLU vs 256-expert MoE), so there is **no expert parallelism**.

> **The single source of truth for this port** — architecture, setup, serving,
> verification, and measured performance — is the bundle README:
> [`Qwen3.8-27B/README.md`](../../../Qwen3.8-27B/README.md). This file documents only
> the module structure of the package.

## Module Structure

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports. |
| `config.py` | HF → Neuron config translation: the 3:1 hybrid layer pattern (`full_attention_interval=4`), the dense SwiGLU FFN, the GatedDeltaNet / linear-attention dims, and the `vision_config` for the ViT. |
| `factory.py` | Model builder + IsHybrid classmethod delegation + EPD / max-pixels classmethods for the vision path. |
| `model.py` | The hybrid dense decoder: 48 GatedDeltaNet (linear-attention) layers + 16 full-attention layers, plain SwiGLU MLP. Keeps the vLLM hybrid interfaces (`HasInnerState` / `IsHybrid`, `get_mamba_state_shape_from_config()`, `get_mamba_state_dtype()` — which **pins the SSM state to bf16, not tunable**, `bind_mamba_state()`). Also builds + wires the Qwen3-VL ViT for the `Qwen3_5ForConditionalGeneration` alias (encoder-cache merge, image + video); the ViT is left unbuilt for `Qwen3_5ForCausalLM`. |
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
[bundle README](../../../Qwen3.8-27B/README.md).
