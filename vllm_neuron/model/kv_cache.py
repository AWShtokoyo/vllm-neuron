# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import torch


@dataclass
class LayerSpec:
    """
    Defines the KV cache specification for a single transformer layer.

    Used to specify the memory requirements and configuration for storing
    key-value pairs in the attention mechanism of a transformer layer.
    """

    name: str
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window_size: int | None = None
    chunk_size: int | None = None


@dataclass
class KVSpec:
    """
    Defines the KV cache needs of a model by specifying all layer configurations.

    Contains a list of LayerSpec objects that collectively define the complete
    KV cache requirements for an entire transformer model.
    """

    layers: list[LayerSpec]


@dataclass
class HybridKVSpec(KVSpec):
    """Extended KVSpec for hybrid models with stateful (non-attention) layers.

    Subclasses KVSpec so:
    - isinstance(spec, KVSpec) is True (backward compat for all existing code)
    - Non-hybrid code reading spec.layers works unchanged
    - Hybrid-aware code checks hasattr(spec, 'stateful_layer_names')

    The model runner uses stateful_layer_names to emit MambaSpec entries
    for the scheduler/cache manager. The model provides shapes/dtypes
    via the IsHybrid interface (get_mamba_state_shape_from_config).
    """

    stateful_layer_names: list[str] = field(default_factory=list)
