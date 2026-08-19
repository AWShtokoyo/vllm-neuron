# SPDX-License-Identifier: Apache-2.0
"""Factory for LlamaBidirectional embedding model selection.

Mirrors the Qwen3 factory pattern: the HF architecture name
`LlamaBidirectionalModel` maps here, and the factory selects the Neuron
implementation. This model is pooling-only (nvidia/llama-embed-nemotron-8b is a
native sentence-transformers embedding checkpoint), so there is a single
implementation, `LlamaBidirecForEmbedding`.
"""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class LlamaBidirectionalModel(nn.Module):
    """Factory mapping the HF architecture name `LlamaBidirectionalModel` to the
    Neuron embedding implementation. Extends nn.Module for vLLM's ModelRegistry.
    """

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
        from .model import LlamaBidirecForEmbedding as Model

        return Model.from_configs(hf_config, neuron_config)
