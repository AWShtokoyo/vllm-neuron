# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

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
    # Multi-head Latent Attention: a single compressed latent buffer serves as
    # both K and V (no separate value cache). When True, the runner emits an
    # MLAAttentionSpec (half the page size of FullAttentionSpec, which budgets
    # for K+V) and allocates one buffer that k_cache/v_cache alias, halving the
    # per-layer KV HBM footprint.
    is_mla: bool = False


@dataclass
class KVSpec:
    """
    Defines the KV cache needs of a model by specifying all layer configurations.

    Contains a list of LayerSpec objects that collectively define the complete
    KV cache requirements for an entire transformer model.
    """

    layers: list[LayerSpec]
