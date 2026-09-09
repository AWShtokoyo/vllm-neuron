# qwen3_5_moe (Qwen3.6-35B-A3B, MoE hybrid, BF16 / FP8, text + vision)

vllm-neuron model implementation for `Qwen/Qwen3.6-35B-A3B` (architecture aliases
`Qwen3_5MoeForCausalLM` for text and `Qwen3_5MoeForConditionalGeneration` for text +
vision, `model_type: qwen3_5_moe`) — a **hybrid Mixture-of-Experts** decoder (30
GatedDeltaNet + 10 full-attention layers, 256-expert top-8 MoE + shared expert), in
**BF16 or FP8**, with **image and video** input. It is the **MoE** sibling of the
`qwen3_5_dense` port: identical hybrid attention stack, differing only in the FFN
(256-expert MoE + shared expert vs plain SwiGLU), so this package additionally carries
expert parallelism.

⚠️ Expert parallelism is **off** in the default recipe: without it the decode kernel loads
only the experts a small batch selected, which is measurably faster at low concurrency. Both
layouts cost the same resident bytes. See the bundle README.

> **The single source of truth for this port** — architecture, setup, serving,
> verification, and measured performance — is the bundle README:
> [`Qwen3.6-35B-A3B/README.md`](../../../Qwen3.6-35B-A3B/README.md). This file
> documents only the module structure of the package.

## Module Structure

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports. |
| `config.py` | HF → Neuron config translation: the 3:1 hybrid layer pattern (full-attn indices), MoE (256 experts / top-8 / shared expert), and GatedDeltaNet dims. Reads `quantization_config` → sets `quant_scheme`. Resolves either architecture alias, so the same config path serves the text-only and the multimodal build. |
| `factory.py` | Model builder + weight-loader wiring + `IsHybrid` classmethod delegation (the platform's hybrid page-size alignment calls these on the resolved factory class before the model exists). |
| `model.py` | The hybrid MoE decoder: 30 GatedDeltaNet (linear-attention) layers + 10 full-attention layers, 256-expert top-8 MoE + shared expert. Also the FP8 params and ROW scale buffers, the prefill/decode MoE kernel calls, and the ViT wiring used by the multimodal alias. Keeps the vLLM hybrid interfaces (`HasInnerState` / `IsHybrid`, `get_mamba_state_shape_from_config()`, `get_mamba_state_dtype()` — which **pins the GatedDeltaNet state to bf16, not tunable** — `bind_mamba_state()`). |
| `quantization.py` | `QuantScheme` (NONE / FP8_ROW) + `quantization_config` parsing. |
| `weight_loaders_bf16.py` | BF16 sharded / fused weight loaders, including the per-rank expert sharding used with expert parallelism. |
| `weight_loaders_fp8_row.py` | DeepSeek `[128,128]` block-FP8 → per-output-channel ROW fp8 re-quant at load for the **expert** weights; block dequant → bf16 for the tensors kept in bf16. Wraps each module's existing loader so the GQA / GDN head-aware sharding is reused verbatim. |

FP8 is conditional on `quant_scheme`: with a BF16 checkpoint the package takes the
unquantized path and none of the fp8 code runs.

The GatedDeltaNet NKI kernels this package calls live under `vllm_neuron/functional/`:
the chunk-group prefill kernel and its causal-conv1d companion in
`linear_attention/`, the earlier prefill forms in `gated_delta_rule.py` /
`gated_delta_rule_seq.py`, the decode state/conv updates in `gdn_state_update.py` /
`gdn_conv_update.py`, and the paged KV gather in `paged_kv_gather.py`. The hybrid
mamba prefix-cache sidecar is `vllm_neuron/vllm/worker/neuron_mamba_apc.py`. See the
bundle README for how they fit together.

## Running

Offline example:
[`examples/vllm_neuron/models/qwen3_5_moe/run.py`](../../../examples/vllm_neuron/models/qwen3_5_moe/run.py)
(sets the required recipe flags via `os.environ.setdefault`, so `python run.py` works
standalone). The full tested serving recipe, required environment flags, and the
GatedDeltaNet prefill-kernel selection are in the
[bundle README](../../../Qwen3.6-35B-A3B/README.md).
