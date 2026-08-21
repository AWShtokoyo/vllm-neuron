# ministral3

vllm-neuron model package for
[`mistralai/Devstral-2-123B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512),
a 123B FP8 dense-GQA causal LM served under the architecture
`Ministral3ForCausalLM` (`model_type` `ministral3`).

> **Deployment, configuration, verification, and performance** for this port are
> documented in the bundle README, which is the single source of truth:
> [`Devstral-2-123B-Instruct-2512/README.md`](../../../Devstral-2-123B-Instruct-2512/README.md).
> This file describes the module structure only.
>
> 🔴 Two environment knobs are **required for correctness** on this model —
> `MINISTRAL3_FP8_LAYERS="mlp:0-1,3-87"` and `FORCE_MLP_KERNEL=cte`. Omitting
> either produces wrong output with no kernel asserts. See the bundle README.

## Module Structure

```text
vllm_neuron/model/ministral3/
├── __init__.py          # Package exports (Ministral3ForCausalLM)
├── README.md            # This file — module structure
├── config.py            # Ministral3 config: YaRN RoPE parameters, the per-layer
│                        #   FP8/BF16 split parsed from MINISTRAL3_FP8_LAYERS, and
│                        #   neuron_config plumbing (buckets, packed FP8 KV)
├── factory.py           # Maps the HF arch name Ministral3ForCausalLM to the
│                        #   Neuron implementation (registered with the registry)
├── model.py             # Dense GQA decoder: YaRN RoPE (interleaved rotate_half),
│                        #   static-FP8 QKV / o_proj / MLP with a per-layer
│                        #   BF16 escape, CTE-vs-TKG MLP kernel selection, and
│                        #   packed FP8 KV cache
└── weight_loaders.py    # Adaptive FP8 / BF16 weight loaders: detects FP8 from the
                         #   checkpoint's slice count and *.weight_scale_inv (not
                         #   from config.json, which is emptied), fused QKV, TP shard
```
