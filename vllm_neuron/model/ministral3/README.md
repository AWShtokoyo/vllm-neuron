# ministral3

vllm-neuron model package for
[`mistralai/Devstral-2-123B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512),
a 123B dense GQA text decoder with YaRN RoPE and a per-tensor static FP8 (E4M3)
checkpoint. Served under the architecture `Ministral3ForCausalLM` (model_type
`ministral3`).

> **Deployment, configuration, verification, and performance** for this port are
> documented in the bundle README, which is the single source of truth:
> [`Devstral-2-123B-Instruct-2512/README.md`](../../../Devstral-2-123B-Instruct-2512/README.md).
> This file describes the module structure only.

## Module Structure

```text
vllm_neuron/model/ministral3/
├── __init__.py          # Package exports (Ministral3Config, Ministral3ForCausalLM,
│                        #   Ministral3Attention / MLP / RMSNorm / RotaryEmbedding)
├── README.md            # This file — module structure
├── config.py            # Ministral3Config: HF → Neuron config translation (GQA head
│                        #   counts, YaRN RoPE parameters, untied embeddings)
├── factory.py           # Ministral3ForCausalLM factory: validates the config and
│                        #   instantiates the implementation
├── model.py             # Dense GQA decoder: attention, SwiGLU MLP, RMSNorm, YaRN
│                        #   rotary embedding; BF16-dequant and FP8-native (per-family
│                        #   FP8×FP8 static matmul) compute paths
└── weight_loaders.py    # Per-tensor static FP8 (E4M3) weight loaders: shard first,
                         #   then dequantize (or keep FP8 + scalar scale for FP8-native);
                         #   checkpoint format auto-detected from the slice count
```
