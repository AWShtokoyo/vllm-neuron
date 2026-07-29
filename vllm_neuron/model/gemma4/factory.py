# SPDX-License-Identifier: Apache-2.0
"""Factory for Gemma4 model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Gemma4ForCausalLM(nn.Module):
    """Factory that validates config and selects the appropriate Gemma4 implementation.

    Registered as ``Gemma4ForCausalLM`` — the transformers-5.x native TEXT
    architecture name (``model_type: gemma4_text``). We serve Gemma4 text-only,
    which is exactly what upstream separates into ``Gemma4ForCausalLM`` (vs the
    multimodal ``Gemma4ForConditionalGeneration`` / ``gemma4_mm``). vLLM-core
    maps ``Gemma4ForCausalLM`` to a non-multimodal text model, so presenting the
    model under this name routes it down the text path end-to-end and avoids the
    ``gemma4_mm`` multimodal renderer entirely. The gemma-4-31B checkpoint
    declares the multimodal arch name; ``vllm_neuron`` rewrites it to this text
    arch at config-registration time (see ``_register_gemma4_hf_config``).

    Implements the VllmModel protocol stubs so that vLLM's ModelRegistry
    recognizes this as a text generation model during architecture validation.
    The actual model implementation lives in model.py; the factory delegates
    to it via from_configs().
    """

    def __init__(
        self,
        hf_config: PretrainedConfig = None,
        neuron_config: NeuronConfig | None = None,
        *,
        vllm_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hf_config is not None:
            self._model = self._select_implementation(hf_config, neuron_config)
        else:
            self._model = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *args,
        **kwargs,
    ):
        return self._model(input_ids, positions, *args, **kwargs)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Protocol stub for VllmModel interface."""
        raise NotImplementedError("Use from_configs() to create the model.")

    def compute_logits(self, hidden_states):
        """Protocol stub for VllmModelForTextGeneration interface."""
        raise NotImplementedError("Use from_configs() to create the model.")

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
    ) -> nn.Module:
        # Registered under the text arch ``Gemma4ForCausalLM`` and served with a
        # text-only hf_config (no ``vision_config``), so the Neuron runner uses
        # the standard text ``from_configs(hf_config, neuron_config)`` convention.
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)

        from .model import Gemma4ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        pass
