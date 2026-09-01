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

        if quantization == "fp8_per_channel":
            from .model_fp8_per_channel import Glm52ForCausalLM as Model
        else:
            # BF16. Kept because model.py is the base class the FP8 implementation
            # extends, not because BF16 is deployable: its weights do not fit one
            # trn2.48xlarge (see "Why FP8 only" in GLM-5.2/README.md).
            from .model import Glm52ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        quantization = neuron_config.quantization if neuron_config else None

        if quantization and quantization not in ("fp8_per_channel", "bf16"):
            raise ValueError(
                f"quantization='{quantization}' is not supported for GLM-5.2. "
                "Supported: 'fp8_per_channel', or None/bf16. Note that BF16 weights "
                "do not fit one trn2.48xlarge, so 'fp8_per_channel' is the only "
                "deployable value."
            )
