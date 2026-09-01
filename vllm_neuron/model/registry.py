# SPDX-License-Identifier: Apache-2.0
import os

from .llama3 import LlamaForCausalLM
from .gpt_oss import GptOssForCausalLM
from .llama3 import Eagle3LlamaForCausalLM
from .qwen3 import Qwen3ForCausalLM
from .qwen3_vl import Qwen3VLForConditionalGeneration
from .glm_5_2 import Glm52ForCausalLM
from .glm_5_2.mtp import Glm52MtpForCausalLMFactory


def get_models() -> list[tuple[str, type]]:
    """Return a list of available model classes.

    Returns:
        list[tuple[str, type]]: A list of tuples containing model names and their corresponding classes.
            Each tuple contains (model_name, model_class) where:
            - model_name (str): The string identifier for the model, compatible with Hugging Face transformers architecture
            - model_class (type): The actual model class implementation
    """
    models = [
        ("LlamaForCausalLM", LlamaForCausalLM),
        ("GptOssForCausalLM", GptOssForCausalLM),
        ("Eagle3LlamaForCausalLM", Eagle3LlamaForCausalLM),
        ("Qwen3ForCausalLM", Qwen3ForCausalLM),
        ("Qwen3VLForConditionalGeneration", Qwen3VLForConditionalGeneration),
        ("GlmMoeDsaForCausalLM", Glm52ForCausalLM),
        # GLM-5.2 layer-78 MTP self-speculative draft head. MtpProposer hard-sets
        # the draft arch to this name (vLLM's hf_config_override rewrites the raw
        # glm_moe_dsa draft arch to DeepSeekMTPModel). Speculative decoding is not
        # a supported configuration for this port yet -- see GLM-5.2/README.md.
        ("Glm52MtpForCausalLM", Glm52MtpForCausalLMFactory),
    ]

    # SyntheticNeuronModel is a testing-only model that replaces real neural
    # network computation with deterministic KV cache fill/verify. Useful for
    # validating infrastructure (KV transfer, sharding, block management)
    # without requiring model weights or compilation.
    # Not for production inference — gated to avoid exposing to customers.
    if os.environ.get("VLLM_NEURON_SYNTHETIC_MODEL") == "1":
        from .synthetic import SyntheticNeuronModel

        models.append(("SyntheticNeuronModel", SyntheticNeuronModel))

    return models
