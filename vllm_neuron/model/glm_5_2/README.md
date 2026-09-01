# glm_5_2

vllm-neuron model package for
[`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8), a
Mixture-of-Experts text model with Multi-head Latent Attention (MLA). Served
text-only under the architecture
`GlmMoeDsaForCausalLM`. **FP8 weights on HBM only**: BF16 weights do not fit a
single `trn2.48xlarge`, so `quantization` must be `"fp8_per_channel"`.

> **Deployment, configuration, verification, and performance** for this port are
> documented in the bundle README, which is the single source of truth:
> [`GLM-5.2/README.md`](../../../GLM-5.2/README.md).
> This file describes the module structure only.

## Module Structure

```text
vllm_neuron/model/glm_5_2/
├── __init__.py                    # Package exports (Glm52Config, Glm52ForCausalLM)
├── README.md                      # This file — module structure
├── config.py                      # Glm52Config: HF → Neuron config translation (MLA ranks,
│                                  #   MoE routing, first_k_dense_replace, interleaved RoPE)
├── factory.py                     # Glm52ForCausalLM factory: validates the config and selects
│                                  #   the implementation from `quantization`
├── model_fp8_per_channel.py       # quantization="fp8_per_channel": per-row (per-output-channel) FP8
│                                  #   on HBM; dequant happens inside the kernel (ROW mode).
│                                  #   The supported configuration.
├── mtp_fp8.py                     # FP8 ROW variant of mtp.py — unsupported path
├── weight_loaders_fp8.py          # Block-128 FP8 dequant + shard / re-quantize weight loaders
├── model.py                       # Base implementation the FP8 classes extend: MLA attention
│                                  #   (q_lora 2048 / kv_lora 512), 256-expert top-8 sigmoid MoE
│                                  #   + shared expert, 3 dense layers. Holds the module graph
│                                  #   and BF16 forward path; not a single-node config on its
│                                  #   own — BF16 weights exceed one node's HBM.
└── mtp.py                         # Glm52MtpForCausalLM: layer-78 head that mtp_fp8.py extends
                                   #   + factory — unsupported path, not run on device
```
