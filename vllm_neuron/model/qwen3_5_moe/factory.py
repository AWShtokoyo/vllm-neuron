# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3.5 model selection."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.interfaces import SupportsSpatialMerge
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig


class Qwen3_5MoeForConditionalGeneration(nn.Module, SupportsSpatialMerge):
    """Factory for Qwen3.5-MoE (Qwen3.6-35B-A3B).

    Rebased onto the dense qwen3_5 hybrid base (IsHybrid/MambaSpec framework-
    managed state); the FFN is the 256-expert MoE + shared expert + EP.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> nn.Module:
        return cls._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
        vision_neuron_config: VisionNeuronConfig | None,
    ) -> nn.Module:
        from .model import Qwen3_5ForConditionalGeneration as Model

        return Model.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )

    @classmethod
    def get_vision_token_merge_factor(cls, hf_config: PretrainedConfig) -> int:
        return hf_config.vision_config.spatial_merge_size ** 2

    # ── IsHybrid delegation ──────────────────────────────────────────────
    # ModelRegistry.resolve_model_cls returns THIS factory class, and the
    # platform's hybrid page-size alignment (update_block_size_for_backend ->
    # _align_hybrid_block_size) calls these classmethods on the resolved class
    # BEFORE the model is instantiated. The real implementations live on the
    # routed model class, so delegate to it here.
    @classmethod
    def get_mamba_state_shape_from_config(cls, *args, **kwargs):
        from .model import Qwen3_5ForConditionalGeneration as Model
        return Model.get_mamba_state_shape_from_config(*args, **kwargs)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, *args, **kwargs):
        from .model import Qwen3_5ForConditionalGeneration as Model
        return Model.get_mamba_state_dtype_from_config(*args, **kwargs)

    @classmethod
    def get_mamba_state_dtype(cls, *args, **kwargs):
        from .model import Qwen3_5ForConditionalGeneration as Model
        return Model.get_mamba_state_dtype(*args, **kwargs)
