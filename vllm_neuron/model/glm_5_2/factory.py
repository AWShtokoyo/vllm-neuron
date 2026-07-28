# SPDX-License-Identifier: Apache-2.0
"""Factory for GLM-5.2 model selection based on platform and configuration."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Glm52ForCausalLM(nn.Module):
    """Factory that validates config and selects the appropriate GLM-5.2 implementation."""

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)
        quantization = neuron_config.quantization if neuron_config else None

        if quantization == "fp8":
            from .model_fp8 import Glm52ForCausalLM as Model
        elif quantization == "fp8_native":
            from .model_fp8_native import Glm52ForCausalLM as Model
        elif quantization == "fp8_fwd":
            from .model_fp8_fwd_dequant import Glm52ForCausalLM as Model
        else:
            from .model import Glm52ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        quantization = neuron_config.quantization if neuron_config else None

        if quantization and quantization not in ("fp8", "fp8_native", "fp8_fwd", "bf16"):
            raise ValueError(
                f"quantization='{quantization}' is not supported for GLM-5.2. "
                "Supported: 'fp8', 'fp8_native', 'fp8_fwd', or None/bf16."
            )
