# SPDX-License-Identifier: Apache-2.0
"""Functional operators for gated-DeltaNet (linear-attention) layers.

These back the Qwen3.5 (``qwen3_5``) hybrid decoder's ``linear_attention``
layers. Each op dispatches to a NKI kernel on device and a numerically
faithful PyTorch fallback on CPU (``VLLM_NEURON_CPU_MODE=1``), mirroring the
convention used by the other ``vllm_neuron.functional`` ops.

The PyTorch fallbacks are ports of the HuggingFace ``transformers``
``qwen3_next`` reference implementation (``causal_conv1d_fn`` /
``causal_conv1d_update`` / ``torch_chunk_gated_delta_rule`` /
``torch_recurrent_gated_delta_rule``) and are the equivalence reference.
"""

from .causal_conv1d import causal_conv1d_prefill, causal_conv1d_update
from .gated_delta_rule import (
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)

__all__ = [
    "causal_conv1d_prefill",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "recurrent_gated_delta_rule",
]
