# SPDX-License-Identifier: Apache-2.0
"""
LlamaBidirectional (embedding) Configuration
============================================
<-- MODEL-SPECIFIC: config for nvidia/llama-embed-nemotron-8b (model_type
`llama_bidirec`). A Llama-3 8B backbone run with BIDIRECTIONAL attention, then
mean-pooled and L2-normalized to produce sentence embeddings.

This is a pooling/embedding model — there is NO lm_head, no KV cache, no decode.
Pooling (mean + L2 normalize) is performed by the reused upstream DispatchPooler
in the NeuronModelRunner._pool path, NOT inside this model's forward.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class LlamaBidirecConfig:
    # ── Backbone (same shape as Llama-3 8B) ──────────────────────────────
    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling: dict | None = None
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig = None
    ):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            if hasattr(hf_config, "torch_dtype") and hf_config.torch_dtype is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = hf_config

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered and isinstance(filtered["torch_dtype"], str):
            filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])

        if neuron_config is not None:
            filtered["neuron_config"] = neuron_config

        return cls(**filtered)
