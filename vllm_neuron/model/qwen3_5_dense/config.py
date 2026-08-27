# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 Dense Config
====================

Multimodal hybrid model with vision encoder + text decoder — DENSE FFN variant
(Qwen3.6-27B). Text decoder mixes linear attention (Gated DeltaNet) and full
attention layers. Architecture: Qwen3_5ForConditionalGeneration.

Differs from the sibling ``qwen3_5_moe`` port only in the FFN: every layer here
is a plain SwiGLU MLP (single ``intermediate_size``) instead of a 256-expert
top-8 MoE + shared expert. The hybrid attention (GatedDeltaNet + full attention),
RoPE, and vision fields are identical.

Key features:
  - Hybrid layers: 48 linear attention + 16 full attention (pattern: 3 linear + 1 full)
  - Full attention: GQA with QK-norm, output gate (sigmoid), partial RoPE
  - Linear attention: Gated DeltaNet with conv1d, recurrent state
  - Dense FFN: SwiGLU MLP (intermediate_size=17408), no experts
  - M-RoPE with partial_rotary_factor=0.25
  - Vision encoder: ViT with spatial merger
"""

import json
from dataclasses import dataclass, field, fields

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig
from vllm_neuron.model.qwen3_5_dense.quantization import QuantScheme, parse_quant_config


def _from_hf_sub_config(cls, hf_sub_config, neuron_config=None):
    """Build a sub-config from an HF config sub-object."""
    if isinstance(hf_sub_config, PretrainedConfig):
        config_dict = hf_sub_config.to_dict()
        if hasattr(hf_sub_config, "torch_dtype") and hf_sub_config.torch_dtype is not None:
            config_dict["torch_dtype"] = hf_sub_config.torch_dtype
    elif isinstance(hf_sub_config, dict):
        config_dict = hf_sub_config
    else:
        raise TypeError(f"Unsupported config type: {type(hf_sub_config)}")

    field_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in config_dict.items() if k in field_names}

    if "torch_dtype" not in filtered and "dtype" in config_dict and "torch_dtype" in field_names:
        filtered["torch_dtype"] = config_dict["dtype"]

    if "torch_dtype" in filtered and isinstance(filtered["torch_dtype"], str):
        filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])

    if neuron_config is not None:
        filtered["neuron_config"] = neuron_config

    return cls(**filtered)


@dataclass
class Qwen3_5TextConfig:
    """Text decoder config for Qwen3.5 hybrid model."""

    attention_bias: bool = False
    attn_output_gate: bool = True
    head_dim: int = 256
    hidden_act: str = "silu"
    hidden_size: int = 5120
    max_position_embeddings: int = 262144
    num_attention_heads: int = 24
    num_hidden_layers: int = 64
    num_key_value_heads: int = 4
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    vocab_size: int = 248320

    # Dense FFN parameter (Qwen3.6-27B: plain SwiGLU MLP, no experts).
    intermediate_size: int = 17408

    # Linear attention (Gated DeltaNet) parameters
    full_attention_interval: int = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_value_head_dim: int = 128

    # RoPE parameters
    rope_parameters: dict = field(default_factory=lambda: {
        "rope_type": "default",
        "rope_theta": 10000000,
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
    })

    # Layer type pattern
    layer_types: list[str] = field(default_factory=list)

    # Resident-weight quantization scheme (set by Qwen3_5Config.from_configs from
    # the HF top-level ``quantization_config``). Default NONE = bf16 weights,
    # byte-identical to the un-quantized port.
    quant_scheme: QuantScheme = QuantScheme.NONE

    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if not self.layer_types:
            # Generate from full_attention_interval: every Nth layer is full_attention
            self.layer_types = []
            for i in range(self.num_hidden_layers):
                if (i + 1) % self.full_attention_interval == 0:
                    self.layer_types.append("full_attention")
                else:
                    self.layer_types.append("linear_attention")

    @property
    def rope_theta(self) -> float:
        return self.rope_parameters.get("rope_theta", 10000000.0)

    @property
    def mrope_section(self) -> list[int]:
        return self.rope_parameters.get("mrope_section", [11, 11, 10])

    @property
    def partial_rotary_factor(self) -> float:
        return self.rope_parameters.get("partial_rotary_factor", 0.25)

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_hf_config(cls, hf_text_config, neuron_config: NeuronConfig = None):
        return _from_hf_sub_config(cls, hf_text_config, neuron_config)


@dataclass
class Qwen3_5VisionConfig:
    """Vision encoder config for Qwen3.5."""

    depth: int = 27
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_size: int = 1152
    in_channels: int = 3
    intermediate_size: int = 4304
    num_heads: int = 16
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2048
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    deepstack_visual_indexes: list[int] | None = None

    neuron_config: VisionNeuronConfig | None = None

    def __post_init__(self):
        if self.deepstack_visual_indexes is None:
            self.deepstack_visual_indexes = []

    @classmethod
    def from_hf_config(cls, hf_vision_config, neuron_config: VisionNeuronConfig = None):
        return _from_hf_sub_config(cls, hf_vision_config, neuron_config)


@dataclass
class Qwen3_5Config:
    """Top-level multimodal config composing text and vision sub-configs."""

    text_config: Qwen3_5TextConfig | None = None
    vision_config: Qwen3_5VisionConfig | None = None

    image_token_id: int = 248056
    tie_word_embeddings: bool = False
    video_token_id: int = 248057
    vision_end_token_id: int = 248054
    vision_start_token_id: int = 248053

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict | str,
        text_neuron_config: NeuronConfig = None,
        vision_neuron_config: VisionNeuronConfig = None,
        neuron_config: NeuronConfig = None,
    ):
        # Compat alias: the equivalence framework adapter (and standard non-hybrid
        # ports) call from_configs(hf_config, neuron_config=...). This vision-hybrid
        # port names it text_neuron_config; accept `neuron_config` as an alias for it.
        if neuron_config is not None and text_neuron_config is None:
            text_neuron_config = neuron_config
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
            hf_text = config_dict["text_config"]
            hf_vision = config_dict["vision_config"]
            top_level = config_dict
        elif isinstance(hf_config, PretrainedConfig):
            hf_text = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config.to_dict().get("text_config", {})
            hf_vision = hf_config.vision_config if hasattr(hf_config, "vision_config") else hf_config.to_dict().get("vision_config", {})
            top_level = hf_config.to_dict()
        elif isinstance(hf_config, dict):
            hf_text = hf_config["text_config"]
            hf_vision = hf_config["vision_config"]
            top_level = hf_config
        else:
            raise TypeError(f"Unsupported hf_config type: {type(hf_config)}")

        text_config = Qwen3_5TextConfig.from_hf_config(hf_text, text_neuron_config)
        vision_config = Qwen3_5VisionConfig.from_hf_config(hf_vision, vision_neuron_config)

        tie_word_embeddings = top_level.get("tie_word_embeddings", False)
        text_config.tie_word_embeddings = tie_word_embeddings

        # Resident-weight FP8: the ``quantization_config`` is top-level in this
        # checkpoint (a DeepSeek block-FP8 model). NONE when absent → bf16 path.
        text_config.quant_scheme = parse_quant_config(top_level.get("quantization_config"))

        return cls(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=top_level.get("image_token_id", 248056),
            tie_word_embeddings=tie_word_embeddings,
            video_token_id=top_level.get("video_token_id", 248057),
            vision_end_token_id=top_level.get("vision_end_token_id", 248054),
            vision_start_token_id=top_level.get("vision_start_token_id", 248053),
        )
