# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.2 Config
======================

MLA attention + MoE with DeepSeek Sparse Attention (DSA).
DSA indexer is opt-in (`GLM52_DSA=1`); the default is full attention for all tokens.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Glm52Config:
    vocab_size: int = 154880
    hidden_size: int = 6144
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    rope_theta: float = 8000000.0
    rope_interleave: bool = True
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # MLA (Multi-head Latent Attention) parameters
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 192
    v_head_dim: int = 256

    # MoE parameters
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.5
    n_group: int = 1
    topk_group: int = 1
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    first_k_dense_replace: int = 3
    moe_layer_freq: int = 1

    # DSA (DeepSeek Sparse Attention) indexer. Carried through so the sparse path can
    # be built; whether it RUNS is a separate decision (see Glm52Attention).
    #
    # index_topk        rows each query attends to, regardless of context length
    # index_head_dim    indexer head width (independent of the MLA head dims)
    # index_n_heads     indexer heads (independent of num_attention_heads)
    # index_topk_freq   period of the "full" layers; layers between them are "shared"
    # indexer_types     per-layer "full" | "shared". A "shared" layer owns no indexer
    #                   weights and reuses the selection of the nearest preceding
    #                   "full" layer, so ALL layers end up sparse, not just the ones
    #                   carrying weights (21 "full" of 78 in the shipped checkpoint).
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk_freq: int = 4
    indexer_types: list[str] | None = None

    # Framework config
    neuron_config: NeuronConfig | None = None

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

        # Extract rope_theta from rope_parameters if present
        if "rope_parameters" in config_dict and isinstance(config_dict["rope_parameters"], dict):
            if "rope_theta" in config_dict["rope_parameters"]:
                config_dict["rope_theta"] = config_dict["rope_parameters"]["rope_theta"]

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        # 🔴 BF16 compute, unconditionally, whatever the checkpoint declares. The FP8
        # checkpoint says `float8_e4m3fn`, but the port dequantizes at load and every
        # forward runs in BF16, so honouring the checkpoint dtype here would build
        # parameters the forward cannot use.
        #
        # This used to be preceded by a `getattr(torch, filtered_dict["torch_dtype"])`
        # branch that parsed the checkpoint's string dtype and was then discarded by
        # this very line. It read as if the checkpoint were honoured; it never was.
        filtered_dict["torch_dtype"] = torch.bfloat16

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
