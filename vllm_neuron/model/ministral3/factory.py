# SPDX-License-Identifier: Apache-2.0
"""Factory for Ministral3 model selection based on platform and configuration."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Ministral3ForCausalLM(nn.Module):
    """Factory that validates config and selects the Ministral3 implementation.

    This class extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    The factory stores the selected implementation and delegates forward() calls.
    """

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the selected implementation."""
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Create model from configs. Returns the selected implementation directly."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Select and instantiate the appropriate implementation based on config."""
        cls._validate_config(hf_config, neuron_config)

        # A single implementation handles both compute modes. By default FP8
        # checkpoints are dequantized to BF16 at load time (see model.load_weights).
        # When ``dense_fp8_static`` is set on the config, the listed projection
        # families instead keep their FP8 weights and run FP8×FP8 STATIC matmuls
        # on TRN2 (see model.py + weight_loaders.py); families not listed stay on
        # the BF16-dequant path. Selection is per-family for bring-up isolation.
        from .model import Ministral3ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate that the configuration is supported."""
        # TODO: Add validation rules as the model matures.
        pass
