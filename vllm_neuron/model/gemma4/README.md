# gemma4

vllm-neuron model package for
[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it), served
text-only under the transformers-5.x native TEXT architecture
`Gemma4ForCausalLM`.

> **Deployment, configuration, verification, and performance** for this port are
> documented in the bundle README, which is the single source of truth:
> [`gemma4-31b/README.md`](../../../gemma4-31b/README.md).
> This file describes the module structure only.

## Module Structure

```text
vllm_neuron/model/gemma4/
├── __init__.py                  # Package exports (Gemma4ForCausalLM)
├── README.md                    # This file — module structure
├── config.py                    # _Gemma4Config: rewrites the checkpoint's multimodal
│                                #   arch to the text arch Gemma4ForCausalLM at registration;
│                                #   per-layer (SWA vs global) shape accessors
├── factory.py                   # Model construction / weight-loading factory
│                                #   (registered with the vLLM ModelRegistry)
├── attention_decode_kernel.py   # Optional NKI decode attention for head_dim > 128
│                                #   (opt-in via VLLM_GEMMA4_DECODE_KERNEL=1; not
│                                #   verified on device)
└── model.py                     # Full text model: heterogeneous attention
                                 #   (sliding head_dim=256 / global head_dim=512),
                                 #   QK/V norm, partial RoPE, GeGLU MLP, logit softcap,
                                 #   weight loading
```

## Attention paths

| Phase | Path | Head-dim ceiling |
|---|---|---|
| Prefill (full + segmented) | NKI CTE flash attention (`NF.flash_attention`), automatic PyTorch fallback | 512 (kernel's `_MAX_HEAD_DIM`) |
| Decode (default) | Decomposed fp32 PyTorch attention | — |
| Decode (`VLLM_GEMMA4_DECODE_KERNEL=1`) | `attention_decode_kernel.py`, automatic PyTorch fallback | d-tiled, no 128 cap |

The fused TKG decode megakernel (`attention_block_tkg`) is capped at head_dim
≤ 128, so it does not apply to either Gemma 4 layer type.
