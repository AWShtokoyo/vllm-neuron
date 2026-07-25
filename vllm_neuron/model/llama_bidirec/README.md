# llama_bidirec

vllm-neuron model package for
[`nvidia/llama-embed-nemotron-8b`](https://huggingface.co/nvidia/llama-embed-nemotron-8b),
an 8B bidirectional Llama embedding (pooling) model served under the architecture
`LlamaBidirectionalModel` (`model_type` `llama_bidirec`).

> **Deployment, configuration, verification, and performance** for this port are
> documented in the bundle README, which is the single source of truth:
> [`llama-embed-nemotron-8b/README.md`](../../../llama-embed-nemotron-8b/README.md).
> This file describes the module structure only.

## Module Structure

```text
vllm_neuron/model/llama_bidirec/
├── __init__.py    # Package exports (LlamaBidirectionalModel)
├── README.md      # This file — module structure
├── config.py      # LlamaBidirectional config (pooling / temperature fields)
├── factory.py     # Model construction / weight-loading factory
│                  #   (registered with the vLLM ModelRegistry)
└── model.py       # Bidirectional (non-causal) Llama-3.1-8B backbone with
                   #   per-query key_bounds, mask-weighted mean-pool + L2-norm
                   #   head (→ 4096-dim embedding), no KV cache; weight loading
```
</content>
