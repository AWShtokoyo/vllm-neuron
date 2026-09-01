# SPDX-License-Identifier: Apache-2.0
"""
GLM BF16 Implementation
====================================

GLM with Multi-head Latent Attention (MLA) and Mixture of Experts (MoE).

Key architectural features:
- MLA: compressed KV via low-rank projections + weight absorption
- MoE: 256 routed experts (top-8) + 1 shared expert, with 3 dense layers
- Standard RoPE (interleaved layout) on qk_rope_head_dim=64
- DSA (DeepSeek Sparse Attention) indexer: implemented and opt-in via
  `VLLM_GLM_DSA=1`. Default OFF, so the default forward path is full attention.
  Enabling it widens a `full` layer's KV cache row, so it needs its own compile
- BF16 checkpoint (no FP8 dequantization needed)

Supported parallelism: TP, SP, EP.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.parallel.neuron_parallel_state import (
    get_neuron_ep_degree,
    get_neuron_ep_rank,
    get_neuron_ep_tp_group,
)

import vllm_neuron.functional as NF
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.vllm.spec_decode.decorator import async_speculative_decoding

import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

from .config import GlmMoeDsaConfig

logger = logging.getLogger(__name__)

# Query-tile height for the tiled MLA prefill path. 128 is `nl.tile_size.pmax`, the
# partition limit the NKI inner block asserts on; the torch fallback is indifferent to
# the value, so this is sized for the kernel.
_Q_TILE = 128


def dsa_enabled() -> bool:
    """Whether the DSA sparse-attention indexer participates in the forward path.

    Default OFF, opt-in via `VLLM_GLM_DSA=1`, same convention (and same reason) as
    `VLLM_GLM_MLA_BLOCK_KERNEL`: validated on CPU against the HuggingFace reference but
    not yet run on device, so it must not sit in front of requests. Read from the
    environment rather than `vllm_neuron/envs.py` to keep the shared-framework diff
    minimal, per ADD_MODEL_TO_FORK_INSTRUCTIONS.md §2c.

    🔴 This changes the KV cache LAYOUT (a `full` layer's cache row grows by
    `index_head_dim`), so it is read once at construction and must not be toggled
    against an already-compiled graph or a populated cache.
    """
    import os

    return os.environ.get("VLLM_GLM_DSA", "") in ("1", "true", "True")


def _dsa_segment_mask(
    dsa_topk: torch.Tensor, ks: int, ke: int, dtype: torch.dtype
) -> torch.Tensor:
    """DSA term for one key segment, shaped to broadcast over heads: [B, 1, Sq, seg].

    The selection is per QUERY and shared by every attention head: the indexer has its
    own 32 heads, but they are collapsed into a single score by the head-weighted sum
    before the top-k. So a head axis of 1 broadcasts against the [B, H, Sq, seg] mask
    rather than materialising H identical copies.
    """
    from .dsa_indexer import segment_sparse_mask

    return segment_sparse_mask(dsa_topk, ks, ke, dtype).unsqueeze(1)


# =============================================================================
# Section 1: RMS Normalization
# =============================================================================


class GlmMoeDsaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding (Standard, Interleaved)
# GLM uses default RoPE with rope_theta=8M, rope_interleave=true,
# on qk_rope_head_dim=64 dimensions.
# =============================================================================


class GlmMoeDsaRotaryEmbedding(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.qk_rope_head_dim = config.qk_rope_head_dim

        dim = self.qk_rope_head_dim
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float64, device="cpu") / dim)
        )
        self._inv_freq_cpu = inv_freq.float()
        self._inv_freq_tensor: torch.Tensor | None = None

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self._inv_freq_tensor is None:
            self._inv_freq_tensor = self._inv_freq_cpu.clone()
        self._inv_freq_tensor = fn(self._inv_freq_tensor)
        return self

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._inv_freq_tensor
        if inv_freq is None:
            inv_freq = self._inv_freq_cpu.to(position_ids.device)
            self._inv_freq_tensor = inv_freq
        inv_freq_expanded = inv_freq[:, None].float()  # [Rd/2, 1]
        positions_expanded = position_ids[None, :].float()  # [1, T]
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, Rd/2]
        # For interleaved RoPE, cos/sin are [T, Rd/2] (NOT doubled)
        return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)


def _apply_rotary_emb_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Interleaved RoPE: rotate pairs (x[..., 2i], x[..., 2i+1])."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)


# =============================================================================
# Section 3: Multi-head Latent Attention (MLA)
# Same approach as DeepSeek-V3 MLA but with different dimensions:
# q_lora_rank=2048, kv_lora_rank=512, qk_rope_head_dim=64,
# qk_nope_head_dim=192, v_head_dim=256, num_heads=64
# DSA indexer is opt-in (VLLM_GLM_DSA=1); default is full attention for all tokens.
# =============================================================================


class GlmMoeDsaAttention(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        self.tp_group = get_tp_group()
        self.sp_group = self.tp_group
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.num_heads_per_rank = self.num_attention_heads // self.world_size
        self.sp_world_size = self.sp_group.world_size
        self.sp_rank = self.sp_group.rank_in_group
        self.hidden_size_per_sp = self.hidden_size // self.sp_world_size

        # MLA dimensions
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 256

        # Softmax scaling
        self.softmax_scale = self.q_head_dim ** (-0.5)

        # ---- DSA (sparse attention) indexer ----------------------------------
        # `indexer_types[i]` is "full" or "shared". A "full" layer owns indexer
        # weights and computes a top-k selection; a "shared" layer owns none and
        # reuses the nearest preceding "full" layer's selection. The shipped
        # checkpoint has 21 full and 57 shared, so all 78 layers end up sparse while
        # only 21 carry weights -- and only those 21 need the indexer key cached.
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        types = config.indexer_types
        if not types:
            # 🔴 Refuse rather than silently disable. `GlmMoeDsaConfig` defaults
            # `indexer_types` to None, and `from_configs` only fills it if the
            # checkpoint's config.json carries the key — so a variant checkpoint that
            # omits it would leave DSA requested-but-off, giving full attention with
            # VLLM_GLM_DSA=1 set and nothing in the log to say so. That is the same
            # silently-inert failure mode as the NKI kernel whose gate never opened.
            #
            # index_topk / index_head_dim / index_n_heads have the same exposure: they
            # default to the shipped checkpoint's values, so a variant with different
            # ones would be scored at the wrong width with no error. Those are only
            # meaningful when indexer_types is present, which this check requires.
            if dsa_enabled():
                raise ValueError(
                    "VLLM_GLM_DSA=1 but the model config carries no `indexer_types`, so "
                    "there is no way to tell which layers own an indexer. Either the "
                    "checkpoint's config.json omits the key, or GlmMoeDsaConfig.from_configs "
                    "filtered it out. Refusing to run full attention while DSA is asked "
                    "for."
                )
            self.indexer_type = None
        elif layer_idx < len(types):
            self.indexer_type = types[layer_idx]
        else:
            # 🔴 The MTP draft layer is built at layer_idx = num_hidden_layers (78),
            # one PAST the end of `indexer_types`, which has exactly 78 entries. An
            # unguarded `types[layer_idx]` raises IndexError there and breaks MTP even
            # with DSA off, because this line runs unconditionally.
            #
            # "full" is not a guess: the checkpoint ships
            # `model.layers.78.self_attn.indexer.wk.weight`, so 22 layers carry indexer
            # weights against the config's 21 "full" entries, and layer 78 is the extra
            # one. Verified against the reference checkpoint's weight map.
            self.indexer_type = "full"
        self.dsa_enabled = dsa_enabled() and self.indexer_type is not None
        self.dsa_is_full = self.dsa_enabled and self.indexer_type == "full"

        # KV cache: stores [k_pe | compressed_kv] per token, and with DSA on the indexer
        # key rides along in the same row: [k_pe | compressed_kv | idx_k]. Sharing the
        # paged buffer means the indexer key inherits slot_mapping, the block table and
        # the eviction policy for free instead of needing a second cache with its own
        # allocation and binding.
        #
        # 🔴 THE ROW WIDTH MUST BE UNIFORM ACROSS EVERY LAYER, so `shared` layers widen
        # too even though they never write an indexer key. This is forced by the
        # framework, not chosen: `MLAAttentionSpec.merge` (vllm/v1/kv_cache_interface.py)
        # builds the group spec from `specs[0]` and does NOT check `head_size` for
        # uniformity. Giving `full` layers 704 and `shared` layers 576 made merge adopt
        # layer 0's 704 for the whole group, so every layer got a 704-wide cache while a
        # `shared` layer's module still believed 576 — measured on device as
        #   RuntimeError: shape '[1, -1, 576]' is invalid for input of size 1441792
        # during parallel trace (1441792 = 64 blocks x 32 block_size x 704).
        #
        # Cost, corrected: a uniform row is 128/576 = **22.2% of KV**. An earlier version
        # of this comment claimed 6.0% (21/78 x 128/576) on the assumption that `shared`
        # layers could keep the narrow row. They cannot.
        self.kv_cache_head_dim = self.qk_rope_head_dim + self.kv_lora_rank  # 576
        if self.dsa_enabled:
            self.kv_cache_head_dim += self.index_head_dim                  # 704
        self.num_kv_cache_heads = 1

        self.indexer = None
        if self.dsa_is_full:
            from .dsa_indexer import GlmMoeDsaDsaIndexer

            self.indexer = GlmMoeDsaDsaIndexer(config, layer_idx)

        self.k_cache = None
        self.v_cache = None

        # Q path: hidden -> q_a_proj -> LayerNorm -> q_b_proj
        self.q_a_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size_per_sp, self.q_lora_rank, dtype=self.dtype)
        )
        self.q_a_layernorm = GlmMoeDsaRMSNorm(
            self.q_lora_rank, config.rms_norm_eps, self.dtype
        )
        q_b_out_per_rank = self.num_heads_per_rank * self.q_head_dim
        self.q_b_proj_weight = nn.Parameter(
            torch.empty(self.q_lora_rank, q_b_out_per_rank, dtype=self.dtype)
        )

        # KV path: hidden -> kv_a_proj_with_mqa -> [compressed_kv, k_pe]
        kv_a_out = self.kv_lora_rank + self.qk_rope_head_dim
        self.kv_a_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size_per_sp, kv_a_out, dtype=self.dtype)
        )
        self.kv_a_layernorm = GlmMoeDsaRMSNorm(
            self.kv_lora_rank, config.rms_norm_eps, self.dtype
        )

        # kv_b_proj: [kv_lora_rank, num_heads_per_rank * (qk_nope_head_dim + v_head_dim)]
        kv_b_out_per_rank = self.num_heads_per_rank * (self.qk_nope_head_dim + self.v_head_dim)
        self.kv_b_proj_weight = nn.Parameter(
            torch.empty(self.kv_lora_rank, kv_b_out_per_rank, dtype=self.dtype)
        )

        # Output projection
        o_proj_in_per_rank = self.num_heads_per_rank * self.v_head_dim
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_per_rank, self.hidden_size, dtype=self.dtype)
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        set_weight_loader(
            self.q_a_proj_weight,
            _row_parallel_weight_loader(
                shard_size=self.hidden_size_per_sp,
                num_shards=self.sp_world_size,
            ),
        )
        set_weight_loader(
            self.q_b_proj_weight,
            _sharding_weight_loader(
                shard_dim=0,
                shard_size=self.num_heads_per_rank * self.q_head_dim,
                num_shards=self.world_size,
            ),
        )
        set_weight_loader(
            self.kv_a_proj_weight,
            _row_parallel_weight_loader(
                shard_size=self.hidden_size_per_sp,
                num_shards=self.sp_world_size,
            ),
        )
        set_weight_loader(
            self.kv_b_proj_weight,
            _sharding_weight_loader(
                shard_dim=0,
                shard_size=self.num_heads_per_rank * (self.qk_nope_head_dim + self.v_head_dim),
                num_shards=self.world_size,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            _sharding_weight_loader(
                shard_dim=1,
                shard_size=self.num_heads_per_rank * self.v_head_dim,
                num_shards=self.world_size,
            ),
        )
        if self.indexer is not None:
            # Replicated, transposed at load. k_norm's weight and bias are 1-D and need
            # neither, so they take the bare loader.
            t_loader = _replicated_transposed_weight_loader()
            set_weight_loader(self.indexer.wq_b_weight, t_loader)
            set_weight_loader(self.indexer.wk_weight, t_loader)
            set_weight_loader(self.indexer.weights_proj_weight, t_loader)
            set_weight_loader(self.indexer.k_norm_weight, SafetensorsWeightLoader())
            set_weight_loader(self.indexer.k_norm_bias, SafetensorsWeightLoader())

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        dsa_state: dict | None = None,
    ):
        layer_name = f"layers.{self.layer_idx}.self_attn"
        meta = attn_metadata[layer_name]
        max_query_len = meta["max_query_len"]
        decode_token_threshold = meta["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata,
                dsa_state=dsa_state,
            )

        if self.sp_group.world_size > 1:
            hidden_states = self.sp_group.all_gather(hidden_states, dim=0)

        # Segmented (chunked) prefill: a chunk after the first has a non-empty
        # prior, which forward_prefill cannot see — it attends only inside the
        # current chunk, so chunk 2 would silently ignore chunk 1. Route those
        # through the tiled prior-KV path. kv_segment_size is 0 (or the key is
        # absent) when the runner did not enable segmented prefill, in which case
        # the single-shot path stays in charge and behaviour is unchanged.
        kv_segment_size = meta.get("kv_segment_size", 0) or 0
        if kv_segment_size:
            return self.forward_prefill_segmented(
                hidden_states, positions, position_embeddings, attn_metadata,
                dsa_state=dsa_state,
            )
        # 🔴 Refuse rather than silently skip. `forward_prefill` is the single-shot path
        # and does not build its scores through `_mla_attend_tiled`, so the DSA mask has
        # nowhere to attach; running it anyway would give FULL attention while the
        # config says sparse — a wrong-model result with no error. index_topk=2048 is
        # below the 16 KiB single-shot ceiling, so the selection is not a no-op there
        # and the gap cannot be waved away as harmless. `kv_segment_size` is a Python
        # int from the metadata, so this resolves at trace time, not per request.
        if self.dsa_enabled:
            raise NotImplementedError(
                f"layers.{self.layer_idx}: VLLM_GLM_DSA=1 requires segmented prefill "
                "(kv_segment_size > 0); the single-shot prefill path has no DSA "
                "implementation. Set kv_segment_size_buckets, or unset VLLM_GLM_DSA."
            )
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata,
        )

    def _compute_mla_qkv(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        """Compute MLA Q/KV projections with weight absorption.

        Returns:
            q_nope_absorbed: [B, num_heads_per_rank, T, kv_lora_rank]
            q_pe: [B, num_heads_per_rank, T, qk_rope_head_dim]
            compressed_kv: [B, T, kv_lora_rank]
            k_pe: [B, 1, T, qk_rope_head_dim]
            dsa: None, or (idx_k [B, T, index_head_dim], q [B,T,H,D], weights [B,T,H])
                 on a `full` DSA layer -- the indexer key for these tokens plus the
                 query-side halves of the selection score. Returned from here because
                 the indexer consumes the SAME two intermediates this method already
                 computes (the unsharded hidden states and the post-layernorm
                 q_compressed), and recomputing them at the call site would double a
                 6144-wide matmul per layer.
        """
        bsz, q_len = hidden_states.shape[0], hidden_states.shape[1] if hidden_states.dim() == 3 else 1
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            bsz = 1
            q_len = hidden_states.shape[1]

        h_start = self.sp_rank * self.hidden_size_per_sp
        h_end = h_start + self.hidden_size_per_sp
        hidden_shard = hidden_states[..., h_start:h_end]

        # Q and KV down-projections share the same input shard and each need an
        # all-reduce over the SP group. All-reduce is linear, so concatenate the two
        # projection outputs and reduce once, then split — one collective instead of two.
        q_compressed = torch.matmul(hidden_shard, self.q_a_proj_weight)
        kv_a = torch.matmul(hidden_shard, self.kv_a_proj_weight)
        qkv_a = torch.cat([q_compressed, kv_a], dim=-1)
        self.sp_group.all_reduce(qkv_a)
        q_compressed = qkv_a[..., :self.q_lora_rank]
        kv_a = qkv_a[..., self.q_lora_rank:]

        # Q path
        q_compressed = self.q_a_layernorm(q_compressed)
        q = torch.matmul(q_compressed, self.q_b_proj_weight)
        q = q.view(bsz, q_len, self.num_heads_per_rank, self.q_head_dim).transpose(1, 2)

        q_nope = q[..., :self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim:]

        # KV path
        compressed_kv = kv_a[..., :self.kv_lora_rank]
        k_pe = kv_a[..., self.kv_lora_rank:]
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # ---- DSA indexer, on the same intermediates -----------------------------
        # Computed before cos/sin are reshaped for the MLA rotation below, because the
        # indexer takes them in the [B, T, Rd/2] layout. 🔴 The indexer's own RoPE
        # helper uses the reference's `cat` output layout, NOT the `stack().flatten()`
        # interleave used just below -- see _apply_rope_interleaved_pairs. The two are
        # not interchangeable for an elementwise comparison against the reference.
        dsa = None
        if self.dsa_is_full:
            cos_idx = cos.view(bsz, q_len, -1)
            sin_idx = sin.view(bsz, q_len, -1)
            idx_k = self.indexer.compute_keys(hidden_states, cos_idx, sin_idx)
            idx_q, idx_w = self.indexer.index_query(
                hidden_states, q_compressed, cos_idx, sin_idx
            )
            dsa = (idx_k, idx_q, idx_w)

        # Apply interleaved RoPE to q_pe and k_pe
        # cos/sin: [T, Rd/2] -> [B, 1, T, Rd/2]
        cos = cos.view(bsz, q_len, -1).unsqueeze(1)
        sin = sin.view(bsz, q_len, -1).unsqueeze(1)
        q_pe = _apply_rotary_emb_interleaved(q_pe, cos, sin)
        k_pe = _apply_rotary_emb_interleaved(k_pe, cos, sin)

        # Weight absorption: absorb kv_b_proj into q_nope
        wkv_b = self.kv_b_proj_weight.view(
            self.kv_lora_rank, self.num_heads_per_rank, self.qk_nope_head_dim + self.v_head_dim
        )
        q_absorb = wkv_b[:, :, :self.qk_nope_head_dim]
        q_nope_absorbed = torch.einsum('bhqd,chd->bhqc', q_nope, q_absorb)

        return q_nope_absorbed, q_pe, compressed_kv, k_pe, dsa

    def _cache_row(
        self,
        k_pe_flat: torch.Tensor,   # [T, qk_rope_head_dim]
        c_kv_flat: torch.Tensor,   # [T, kv_lora_rank]
        dsa: tuple | None,
    ) -> torch.Tensor:
        """One paged-cache row per token, exactly `kv_cache_head_dim` wide.

        Layout `[k_pe | compressed_kv | idx_k]`. Built here rather than at each call site
        because the width is not a per-layer choice: `MLAAttentionSpec.merge` forces one
        row width for every layer (see `__init__`), so with DSA on a `shared` layer's row
        is 704 even though it produces no indexer key.

        🔴 A `shared` layer therefore pads with ZEROS. Those columns are never read — only
        a `full` layer reads indexer keys, and only from its own cache — so the value is
        arbitrary; zeros are chosen so a bug that did read them yields an identically-zero
        score rather than plausible garbage. Writing 576 columns into a 704-wide row is
        what would otherwise happen, and `index_put_` would reject it.
        """
        row = [k_pe_flat, c_kv_flat]
        if dsa is not None:
            row.append(dsa[0].reshape(-1, self.index_head_dim))
        combined = torch.cat(row, dim=-1)

        pad = self.kv_cache_head_dim - combined.shape[-1]
        if pad > 0:
            combined = F.pad(combined, (0, pad))
        return combined

    def _dsa_consume_shared(self, dsa_state: dict | None) -> torch.Tensor:
        """The selection a `shared` layer reuses. Raises if none was published.

        🔴 Raises rather than returning None. A `shared` layer owns no indexer weights,
        so its only possible source of sparsity is the nearest preceding `full` layer's
        published selection. Falling back to "no mask" would run this layer with FULL
        attention while `indexer_types` says it is sparse — a wrong-model result with no
        error, no log line, and no shape mismatch, in a path where the shipped
        `indexer_types` puts 57 of 78 layers.

        The shipped config cannot reach this (layer 0 is `full`, the first `shared` is
        layer 3), so this is a guard against a reordered config or a refactor that stops
        threading `dsa_state` through the layer loop — not a live failure.
        """
        topk = dsa_state.get("topk") if dsa_state is not None else None
        if topk is None:
            raise RuntimeError(
                f"layers.{self.layer_idx}: indexer_types says '{self.indexer_type}', so "
                "this layer reuses the nearest preceding 'full' layer's DSA selection, "
                "but none was published. Either no 'full' layer ran before it (check the "
                "order of indexer_types) or dsa_state is not threaded through the layer "
                "loop. Running full attention here would silently contradict the config."
            )
        return topk

    def _read_cache_segment(
        self,
        ks: int,
        ke: int,
        block_table: torch.Tensor,
        block_size: int,
        B: int,
    ) -> torch.Tensor:
        """Paged-cache rows for global slot range `[ks, ke)`: [B, ke-ks, row_width].

        Factored out so the DSA selection pass and the attention pass read the cache
        through ONE implementation. They must agree exactly on which slot a given key
        index denotes -- the selection returns indices that the attention pass then
        turns into a mask -- and two copies of this block-table arithmetic would be
        free to drift. A one-slot disagreement would open the wrong key with no shape
        error and no log line to reveal it.
        """
        num_cache_blocks = self.k_cache.shape[0]
        blk_lo, blk_hi = ks // block_size, (ke - 1) // block_size + 1
        idx = block_table[:, blk_lo:blk_hi].reshape(-1).to(torch.long)
        idx = torch.clamp(idx, min=0, max=num_cache_blocks - 1)
        seg = torch.index_select(self.k_cache, 0, idx)
        seg = seg.squeeze(1).view(B, -1, self.kv_cache_head_dim)
        off = ks - blk_lo * block_size
        return seg[:, off:off + (ke - ks)]

    def _dsa_select(
        self,
        dsa: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        q_global_pos: torch.Tensor,          # [B, Sq]
        prior_len: torch.Tensor | None,      # [B]
        block_table: torch.Tensor | None,
        block_size: int,
        kv_tile: int,
        prior_bucket: int,
    ) -> torch.Tensor:
        """Top-k indexer selection over prior cache + current chunk: [B, Sq, k] int32.

        KEY INDEX SPACE. `[0, prior_bucket)` is a paged-cache slot; `prior_bucket + j`
        is position j of the current chunk. The attention pass masks with the same
        convention, which is the only thing that makes the returned indices meaningful
        there. The current chunk's tokens were already written into the cache before
        this runs, so they exist in BOTH halves of the space -- the `kpos < prior_len`
        invalidation below is what stops them being scored twice.

        MEMORY. Scores are folded segment by segment into a running top-k rather than
        concatenated, so peak is `[B, Sq, k + kv_tile]` (~42 MB at Sq=4096, k=2048,
        kv_tile=512 in fp32) instead of `[B, Sq, T]` (~1 GB at T=64K). See
        `running_topk_update` for why streaming is exact and not an approximation.

        COST, stated plainly: this is a SECOND sweep of the prior cache per `full`
        layer, on top of the attention sweep. Nothing here makes attention cheaper --
        the attention pass still visits every key and merely masks the unselected ones.
        DSA is implemented for fidelity to the trained model, which was trained with
        the selection in place; turning the selection into a saved read requires
        gathering only the chosen keys, which is a kernel-shaped problem and is not
        attempted here.
        """
        from .dsa_indexer import init_running_topk, running_topk_update

        idx_k, idx_q, idx_w = dsa
        B, Sq = idx_q.shape[0], idx_q.shape[1]
        dev = idx_q.device
        chunk_len = idx_k.shape[1]
        neg_inf = float("-inf")

        # Static: both bounds are Python ints fixed by the compiled bucket.
        k = min(self.index_topk, prior_bucket + chunk_len)
        vals, sel = init_running_topk(B, Sq, k, dev)

        if prior_len is not None and block_table is not None and prior_bucket > 0:
            off = self.qk_rope_head_dim + self.kv_lora_rank
            for ks in range(0, prior_bucket, kv_tile):
                ke = min(ks + kv_tile, prior_bucket)
                seg = self._read_cache_segment(ks, ke, block_table, block_size, B)
                scores = self.indexer.index_scores_segment(
                    idx_q, idx_w, seg[..., off:off + self.index_head_dim]
                )
                # Slots at or past a sequence's own prior_len are padding, or hold the
                # current chunk (written above) which the chunk half scores instead.
                kpos = torch.arange(ks, ke, device=dev).view(1, 1, -1)
                scores = torch.where(
                    kpos < prior_len.view(B, 1, 1),
                    scores,
                    torch.full((), neg_inf, device=dev, dtype=scores.dtype),
                )
                vals, sel = running_topk_update(vals, sel, scores, ks)

        base = prior_len.view(B, 1, 1) if prior_len is not None else 0
        for ks in range(0, chunk_len, kv_tile):
            ke = min(ks + kv_tile, chunk_len)
            scores = self.indexer.index_scores_segment(
                idx_q, idx_w, idx_k[:, ks:ke]
            )
            # Causal, applied BEFORE the top-k -- the reference masks then selects, so
            # selecting then masking would let a future key consume a top-k slot and
            # shrink the effective selection.
            kpos = torch.arange(ks, ke, device=dev).view(1, 1, -1) + base
            scores = torch.where(
                kpos <= q_global_pos.view(B, Sq, 1),
                scores,
                torch.full((), neg_inf, device=dev, dtype=scores.dtype),
            )
            vals, sel = running_topk_update(vals, sel, scores, prior_bucket + ks)

        return sel.to(torch.int32)

    def _mla_attend_tiled(
        self,
        q_lift: torch.Tensor,      # [B, H, Sq, L]  absorbed Q latent
        q_pe: torch.Tensor,        # [B, H, Sq, R]  rotated Q RoPE
        c_kv_cur: torch.Tensor,    # [B, Sq, L]     this chunk's latent
        k_pe_cur: torch.Tensor,    # [B, 1, Sq, R]  this chunk's rotated K RoPE
        q_global_pos: torch.Tensor,  # [B, Sq] global position of each query
        prior_len: torch.Tensor | None,  # [B] cached tokens before this chunk
        block_table: torch.Tensor | None,  # [B, max_blocks_per_seq]
        block_size: int,
        q_tile: int,
        kv_tile: int,
        prior_bucket: int = 0,
        dsa_topk: torch.Tensor | None = None,   # [B, Sq_all, k] int32, or None
    ) -> torch.Tensor:
        """Prior-KV-aware MLA attention, tiled over queries and keys.

        `prior_bucket` is the STATIC number of prior tokens to sweep, in tokens
        (normally `max_blocks_per_seq * block_size`). It must be a Python int, not
        derived from a tensor — see the comment at the prior loop.

        One code path for prefill and decode, and the piece Phase 5a-ii replaces
        with a NKI kernel. See design_5a_tiled_mla_prefill.md.

        Why this exists: ``forward_prefill`` attends only inside the current chunk
        (no prior-KV term), so a second chunk would silently ignore the first, and
        it materialises the whole ``[B, H, T, T]`` score matrix (~25 GB in bf16 at
        T=64K with 1 head/rank). Both block segmented prefill, APC and any context
        above vllm-neuron 0.24's 16 KiB single-shot ceiling.

        The MLA structure makes this cheaper than ordinary flash attention: the
        latent IS the value, so a key tile's ``c_kv`` serves both the score
        (contraction over L) and the output accumulation. Peak score memory is
        ``q_tile x kv_tile`` per head, independent of context length.

        Online softmax is exact — rescaling the accumulator by
        ``exp(m_old - m_new)`` is algebraically identical to one softmax over the
        concatenated scores — so the only expected deviation from the untiled path
        is fp32 accumulation order, not an approximation.
        """
        B, H, Sq_all, L = q_lift.shape
        R = q_pe.shape[-1]

        # ---- query tiling -----------------------------------------------------
        # 🔴 Without this the peak score block is Sq_all x kv_tile, i.e. 4096x4096
        # at mml=16384/seg=4096 — the `q_tile x kv_tile` bound this method
        # advertises would not hold, and the NKI inner block (which needs
        # Sq <= 128 partitions) would be unreachable at real segment sizes, so the
        # opt-in kernel would silently never run.
        q_tile = min(q_tile, Sq_all) if q_tile and q_tile > 0 else Sq_all
        if q_tile < Sq_all:
            outs = [
                self._mla_attend_tiled(
                    q_lift[:, :, qs:qs + q_tile],
                    q_pe[:, :, qs:qs + q_tile],
                    c_kv_cur,
                    k_pe_cur,
                    q_global_pos[:, qs:qs + q_tile],
                    prior_len,
                    block_table,
                    block_size,
                    q_tile=q_tile,
                    kv_tile=kv_tile,
                    prior_bucket=prior_bucket,
                    # 🔴 Slice the selection with the tile. It is indexed by QUERY, so
                    # handing the whole tensor to a tile would align tile-local query 0
                    # with global query 0 and mask every tile after the first against
                    # the wrong rows' selections.
                    dsa_topk=(
                        None if dsa_topk is None else dsa_topk[:, qs:qs + q_tile]
                    ),
                )
                for qs in range(0, Sq_all, q_tile)
            ]
            return torch.cat(outs, dim=2)

        Sq = Sq_all
        dev, acc_dtype = q_lift.device, torch.float32

        # Phase 5a-ii: an opt-in NKI implementation of the inner (query tile x key
        # segment) step. Default OFF — validated on the CPU simulator only, so it
        # must not sit in front of requests until it has run on device. See
        # mla_block.py for the measured agreement figures.
        from .mla_block import MASK_NEG, can_use_block_kernel

        use_kernel = can_use_block_kernel(q_lift, Sq, L, R)

        m = torch.full((B, H, Sq, 1), float("-inf"), device=dev, dtype=acc_dtype)
        l = torch.zeros((B, H, Sq, 1), device=dev, dtype=acc_dtype)
        acc = torch.zeros((B, H, Sq, L), device=dev, dtype=acc_dtype)

        def kernel_step(c_kv_seg2d, k_pe_seg2d, mask_add):
            """One online-softmax step through the NKI kernel, per (batch, head).

            The kernel takes latent-/rope-major operands and a [Sq, Sk] additive
            mask, and returns the updated (m, l, acc); the loop over B and H is
            here because the kernel puts Sq on the partition axis (see
            mla_block_kernel.py for why heads cannot go there for GLM at TP=64).
            """
            nonlocal m, l, acc
            from .mla_block import MASK_NEG, mla_block

            m_f = torch.where(torch.isneginf(m), torch.full_like(m, MASK_NEG), m)
            for b in range(B):
                for h in range(H):
                    mo, lo, ao = mla_block(
                        q_lift[b, h].transpose(0, 1).contiguous(),
                        c_kv_seg2d[b].transpose(0, 1).contiguous(),
                        q_pe[b, h].transpose(0, 1).contiguous(),
                        k_pe_seg2d[b].transpose(0, 1).contiguous(),
                        mask_add[b, h],
                        m_f[b, h],
                        l[b, h],
                        acc[b, h],
                        float(self.softmax_scale),
                    )
                    m[b, h], l[b, h], acc[b, h] = mo, lo, ao

        def absorb_step(scores, c_kv_seg):
            """One online-softmax update against a [B,H,Sq,Sk] score block."""
            nonlocal m, l, acc
            s = scores.to(acc_dtype)
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            # A fully-masked block leaves m_new at -inf; exp() would give NaN.
            m_safe = torch.where(torch.isneginf(m_new), torch.zeros_like(m_new), m_new)
            p = torch.exp(s - m_safe)
            alpha = torch.exp(torch.where(torch.isneginf(m), m_safe * 0 - 1e30, m) - m_safe)
            alpha = torch.where(torch.isneginf(m), torch.zeros_like(alpha), alpha)
            l = l * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + torch.matmul(p.to(c_kv_seg.dtype), c_kv_seg).to(acc_dtype)
            m = m_new

        # ---- prior segments, read from the paged cache -------------------------
        # 🔴 The loop bound MUST be a Python int known at trace time.
        # An earlier version used `int(prior_len.max().item())`, which works in
        # eager/CPU but makes the bound data-dependent under torch.compile and dies
        # during parallel trace on device with:
        #   Could not extract specialized integer from data-dependent expression u0
        #   Caused by: for ks in range(0, max_prior, kv_tile)
        # (measured 2026-08-26 JST, mml=16384/seg=4096; every rank failed). The
        # autoport guide lists the same trap: no `.item()` in the forward path.
        #
        # So iterate over the STATIC bucket (`prior_bucket`, derived from
        # max_blocks_per_seq * block_size) and let the validity mask below drop
        # keys past each sequence's real `prior_len` — the mask is a tensor
        # comparison, which traces fine.
        #
        # Consequence, stated plainly: the prior read is bucket-sized again, so it
        # does not shrink with real context. The way to read less is to shrink the
        # BUCKET, i.e. set `decode_context_length_buckets` — which is a static value
        # and therefore traceable. That is the config-only win the context ladder
        # identified (TPOT flat at 374 ms over 32..3682 tokens because the bucket,
        # not the context, sets the work).
        if prior_len is not None and block_table is not None and prior_bucket > 0:
            for ks in range(0, prior_bucket, kv_tile):
                ke = min(ks + kv_tile, prior_bucket)
                seg = self._read_cache_segment(ks, ke, block_table, block_size, B)
                k_pe_seg = seg[..., :R].unsqueeze(1)                      # [B,1,Sk,R]
                # 🔴 Bound the latent slice explicitly. `seg[..., R:]` took "everything
                # after the rope part", which is only the latent while the cache row is
                # exactly [k_pe | c_kv]. The DSA indexer key rides in the same row on
                # `full` layers, so an open-ended slice would silently feed 128 extra
                # columns into the score contraction. Correct either way, so it is not
                # conditional on DSA being on.
                c_kv_seg = seg[..., R:R + L]                               # [B,Sk,L]

                # Prior keys are all strictly before this chunk, so every one is
                # visible to every query -- except padding beyond a sequence's own
                # prior_len when sequences in the batch differ.
                kpos = torch.arange(ks, ke, device=dev).view(1, 1, 1, -1)
                valid = (kpos < prior_len.view(B, 1, 1, 1)).expand(B, H, Sq, ke - ks)
                mask_add = torch.where(
                    valid,
                    torch.zeros((), device=dev, dtype=acc_dtype),
                    torch.full((), MASK_NEG, device=dev, dtype=acc_dtype),
                )
                if dsa_topk is not None:
                    mask_add = mask_add + _dsa_segment_mask(
                        dsa_topk, ks, ke, acc_dtype
                    )
                if use_kernel:
                    kernel_step(c_kv_seg, k_pe_seg.squeeze(1), mask_add)
                else:
                    scores = (
                        torch.matmul(q_pe, k_pe_seg.transpose(-1, -2))
                        + torch.matmul(q_lift, c_kv_seg.unsqueeze(1).transpose(-1, -2))
                    ) * self.softmax_scale
                    absorb_step(
                        scores.to(acc_dtype) + mask_add,
                        c_kv_seg.unsqueeze(1).expand(B, H, ke - ks, L),
                    )

        # ---- current chunk, causally masked ------------------------------------
        # 🔴 Sweep the WHOLE chunk, not `Sq`. Under query tiling `Sq` is the tile
        # length while `c_kv_cur` still holds every key of the chunk, so bounding
        # this loop by Sq would drop keys past the first tile. The causal mask below
        # discards keys after each query's own position, so covering the full chunk
        # is correct (a later tile legitimately attends to earlier chunk tokens).
        chunk_len = c_kv_cur.shape[1]
        base = prior_len.view(B, 1, 1, 1) if prior_len is not None else 0
        for ks in range(0, chunk_len, kv_tile):
            ke = min(ks + kv_tile, chunk_len)
            c_kv_seg = c_kv_cur[:, ks:ke]                                 # [B,Sk,L]
            k_pe_seg = k_pe_cur[:, :, ks:ke]                              # [B,1,Sk,R]
            kpos = (torch.arange(ks, ke, device=dev).view(1, 1, 1, -1) + base)
            visible = (kpos <= q_global_pos.view(B, 1, Sq, 1)).expand(
                B, H, Sq, ke - ks
            )
            mask_add = torch.where(
                visible,
                torch.zeros((), device=dev, dtype=acc_dtype),
                torch.full((), MASK_NEG, device=dev, dtype=acc_dtype),
            )
            if dsa_topk is not None:
                # Current-chunk keys live at `prior_bucket + local` in the selection's
                # index space -- the convention `_dsa_select` documents.
                mask_add = mask_add + _dsa_segment_mask(
                    dsa_topk, prior_bucket + ks, prior_bucket + ke, acc_dtype
                )
            if use_kernel:
                kernel_step(c_kv_seg, k_pe_seg.squeeze(1), mask_add)
            else:
                scores = (
                    torch.matmul(q_pe, k_pe_seg.transpose(-1, -2))
                    + torch.matmul(q_lift, c_kv_seg.unsqueeze(1).transpose(-1, -2))
                ) * self.softmax_scale
                absorb_step(
                    scores.to(acc_dtype) + mask_add,
                    c_kv_seg.unsqueeze(1).expand(B, H, ke - ks, L),
                )

        out_latent = (acc / l.clamp_min(1e-30)).to(self.dtype)            # [B,H,Sq,L]

        wkv_b = self.kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads_per_rank,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        out_absorb = wkv_b[:, :, self.qk_nope_head_dim:]
        return torch.einsum("bhqc,chd->bhqd", out_latent, out_absorb)

    def forward_prefill_segmented(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
        dsa_state: dict | None = None,
    ) -> torch.Tensor:
        """Prefill for a chunk that may have a non-empty prior (segmented prefill).

        Same QKV projection, cache write and SP collectives as ``forward_prefill``;
        only the attention itself differs, delegating to ``_mla_attend_tiled`` so
        the prior read from the paged cache is included and neither the score
        matrix nor the prior read scales with the full context.

        This is what makes segmented prefill — and therefore APC, and any context
        above vllm-neuron 0.24's 16 KiB single-shot ceiling — possible for this
        model. See design_5a_tiled_mla_prefill.md.
        """
        meta = attn_metadata[f"layers.{self.layer_idx}.self_attn"]
        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]
        block_table = meta["block_table_tensor"]
        kv_segment_size = int(meta.get("kv_segment_size", 0) or 0)
        # cached_seq_len is the batch-wide MAX of already-computed tokens, emitted
        # as an int32 tensor of shape [1, 1] (neuron_model_runner.py:4182).
        cached = meta.get("cached_seq_len")

        hidden_states = hidden_states.to(self.dtype)
        tokens = hidden_states.shape[0]
        B = block_table.shape[0]
        Sq = tokens // B

        cos, sin = position_embeddings
        q_lift, q_pe, c_kv, k_pe, dsa = self._compute_mla_qkv(
            hidden_states.view(B, Sq, hidden_states.shape[-1]), cos, sin
        )

        # ---- write this chunk's latent into the paged cache --------------------
        num_cache_blocks = self.k_cache.shape[0]
        block_indices = torch.clamp(
            slot_mapping // block_size, min=0, max=num_cache_blocks - 1
        )
        position_indices = slot_mapping % block_size
        combined = self._cache_row(
            k_pe.squeeze(1).reshape(-1, self.qk_rope_head_dim),
            c_kv.reshape(-1, self.kv_lora_rank),
            dsa,
        )
        head_idx = torch.zeros(
            slot_mapping.shape[0], dtype=torch.long, device=hidden_states.device
        )
        self.k_cache.index_put_((block_indices, head_idx, position_indices), combined)
        # MLA: v_cache aliases k_cache, so this single write populates both.

        # ---- attention over prior (cache) + this chunk -------------------------
        prior_len = (
            cached.reshape(-1)[:1].to(torch.long).expand(B)
            if cached is not None
            else torch.zeros(B, dtype=torch.long, device=hidden_states.device)
        )
        q_global_pos = positions.reshape(B, Sq).to(torch.long)
        prior_bucket = int(meta["max_blocks_per_seq"]) * block_size

        # ---- DSA selection -----------------------------------------------------
        # A `full` layer computes the selection and publishes it; a `shared` layer owns
        # no indexer weights and consumes the nearest preceding `full` layer's. The
        # holder is threaded through the layer loop rather than stored on self, so the
        # dataflow is visible in the trace instead of hiding in module state.
        dsa_topk = None
        if self.dsa_enabled:
            if dsa is not None:
                dsa_topk = self._dsa_select(
                    dsa, q_global_pos, prior_len, block_table, block_size,
                    kv_segment_size or Sq, prior_bucket,
                )
                if dsa_state is not None:
                    dsa_state["topk"] = dsa_topk
            else:
                dsa_topk = self._dsa_consume_shared(dsa_state)

        attn_output = self._mla_attend_tiled(
            q_lift,
            q_pe,
            c_kv,
            k_pe,
            q_global_pos,
            prior_len,
            block_table,
            block_size,
            # 🔴 Must be <= 128, not Sq. The NKI inner block puts the query axis on
            # partitions and `_shape_ok` rejects Sq > 128, so passing Sq here left
            # `can_use_block_kernel` returning False at every real segment size —
            # the opt-in kernel was unreachable even with VLLM_GLM_MLA_BLOCK_KERNEL=1.
            # Measured with the gate instrumented: q_tile=Sq consulted it once at
            # Sq=256 and opened 0 times; q_tile=128 opened 2 of 2. Query tiling is
            # exact (each row's softmax is independent of the others), so this
            # changes cost, not results.
            q_tile=_Q_TILE,
            # The runner already sized the segment for the attention kernel; reuse
            # it so the key tiling matches the compiled bucket rather than
            # introducing a second, unrelated tile size.
            kv_tile=kv_segment_size or Sq,
            # STATIC prior sweep length. max_blocks_per_seq is
            # `blk_table_tensor.shape[1]` on the runner side, i.e. a Python int
            # fixed by the compiled bucket — safe as a loop bound, unlike anything
            # read out of cached_seq_len.
            prior_bucket=prior_bucket,
            dsa_topk=dsa_topk,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            tokens, self.num_heads_per_rank * self.v_head_dim
        )
        attn_output = torch.matmul(attn_output, self.o_proj_weight)

        if self.sp_group.world_size > 1:
            attn_output = self.sp_group.reduce_scatter(attn_output, dim=0)
        return attn_output.contiguous()

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens = hidden_states.shape[0]

        cos, sin = position_embeddings
        q_nope_absorbed, q_pe, compressed_kv, k_pe, _dsa = self._compute_mla_qkv(
            hidden_states, cos, sin
        )

        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]

        block_indices = slot_mapping // block_size
        num_cache_blocks = self.k_cache.shape[0]
        block_indices = torch.clamp(block_indices, min=0, max=num_cache_blocks - 1)
        position_indices = slot_mapping % block_size

        k_pe_flat = k_pe.squeeze(0).squeeze(0)
        ckv_flat = compressed_kv.squeeze(0)
        combined_kv = torch.cat([k_pe_flat, ckv_flat], dim=-1)

        head_indices_for_put = torch.zeros(
            slot_mapping.shape[0], dtype=torch.long, device=hidden_states.device
        )
        self.k_cache.index_put_(
            (block_indices, head_indices_for_put, position_indices),
            combined_kv,
        )
        # MLA: v_cache aliases k_cache (single latent buffer); the write above
        # already populates it. No separate V write needed.

        # Attention scores
        ckv_4d = compressed_kv.unsqueeze(1)
        active_scores = (
            torch.matmul(q_pe, k_pe.transpose(2, 3))
            + torch.matmul(q_nope_absorbed, ckv_4d.transpose(2, 3))
        )
        active_scores *= self.softmax_scale

        T = tokens
        causal_mask = torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=hidden_states.device)
        )
        active_scores = torch.where(
            causal_mask.unsqueeze(0).unsqueeze(0),
            active_scores,
            torch.finfo(active_scores.dtype).min,
        )

        attn_weights = F.softmax(active_scores, dim=-1, dtype=torch.float32).to(self.dtype)

        # Output with V absorption
        wkv_b = self.kv_b_proj_weight.view(
            self.kv_lora_rank, self.num_heads_per_rank, self.qk_nope_head_dim + self.v_head_dim
        )
        out_absorb = wkv_b[:, :, self.qk_nope_head_dim:]

        x = torch.matmul(attn_weights, ckv_4d)
        attn_output = torch.einsum('bhqc,chd->bhqd', x, out_absorb)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(tokens, self.num_heads_per_rank * self.v_head_dim)

        attn_output = torch.matmul(attn_output, self.o_proj_weight)

        if self.sp_group.world_size > 1:
            attn_output = self.sp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
        dsa_state: dict | None = None,
    ) -> torch.Tensor:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        S_ctx = max_blocks_per_seq * block_size

        hidden_states = hidden_states.to(self.dtype)

        cos, sin = position_embeddings
        q_nope_absorbed, q_pe, compressed_kv_active, k_pe_active, dsa = self._compute_mla_qkv(
            hidden_states.view(B, S_decode, hidden), cos, sin
        )

        flat_indices = block_table.reshape(-1).to(torch.long)
        num_cache_blocks = self.k_cache.shape[0]
        flat_indices = torch.clamp(flat_indices, min=0, max=num_cache_blocks - 1)
        k_blocks = torch.index_select(self.k_cache, 0, flat_indices)
        k_blocks = k_blocks.squeeze(1).view(B, max_blocks_per_seq, block_size, self.kv_cache_head_dim)
        prior_kv = k_blocks.reshape(B, S_ctx, self.kv_cache_head_dim)

        k_pe_prior = prior_kv[..., :self.qk_rope_head_dim]
        # 🔴 Bounded, not open-ended — see the same fix in `_mla_attend_tiled`. On a
        # `full` DSA layer the cache row carries the indexer key after the latent, and
        # `[..., qk_rope:]` would pull it into `compressed_kv`.
        compressed_kv_prior = prior_kv[
            ..., self.qk_rope_head_dim:self.qk_rope_head_dim + self.kv_lora_rank
        ]
        k_pe_prior = k_pe_prior.unsqueeze(1)

        # Update cache
        block_indices = slot_mapping // block_size
        block_indices = torch.clamp(block_indices, min=0, max=num_cache_blocks - 1)
        position_indices = slot_mapping % block_size

        combined_active = self._cache_row(
            k_pe_active.squeeze(1).reshape(-1, self.qk_rope_head_dim),
            compressed_kv_active.reshape(-1, self.kv_lora_rank),
            dsa,
        )

        head_indices_for_put = torch.zeros(
            slot_mapping.shape[0], dtype=torch.long, device=hidden_states.device
        )
        self.k_cache.index_put_(
            (block_indices, head_indices_for_put, position_indices),
            combined_active,
        )
        # MLA: v_cache aliases k_cache (single latent buffer); the write above
        # already populates it. No separate V write needed.

        # Attention scores
        ckv_prior_4d = compressed_kv_prior.unsqueeze(1)
        prior_scores = (
            torch.matmul(q_pe, k_pe_prior.transpose(2, 3))
            + torch.matmul(q_nope_absorbed, ckv_prior_4d.transpose(2, 3))
        )
        prior_scores *= self.softmax_scale

        ckv_active_4d = compressed_kv_active.unsqueeze(1)
        active_scores = (
            torch.matmul(q_pe, k_pe_active.transpose(2, 3))
            + torch.matmul(q_nope_absorbed, ckv_active_4d.transpose(2, 3))
        )
        active_scores *= self.softmax_scale

        # Causal mask for prior tokens.
        #
        # 🔴 BOUNDED BY THE FIRST ACTIVE POSITION, NOT BY EACH QUERY'S OWN POSITION.
        # `prior_kv` was gathered above, BEFORE `index_put_` wrote this step's active
        # tokens, so the cache rows for the active positions hold whatever was there when
        # the step began. With a per-query bound (`ctx_pos < query_pos`) query j > 0 admits
        # ctx_pos == p0 .. p0+j-1 — its own block's earlier tokens — and reads those rows
        # from the pre-write gather. At S_decode == 1 that cannot happen, which is why
        # non-speculative decode is unaffected; at the speculative verify shape
        # S_decode = 1 + gamma it happens on every step, with two flavours of damage:
        #   * the row had been written before -> the key is counted TWICE, once from prior
        #     and once from the active block, perturbing the softmax deterministically;
        #   * it had not (positions shift after a rejection) -> the read is stale or
        #     uninitialised, so the SAME prompt answers differently every time.
        # Measured before this fix: MTP repeatable on 2 of 8 greedy prompts and identical to
        # the non-speculative baseline on 0 of 8, the same prompt giving 6 different answers
        # in 6 tries; the non-speculative twin on the same build was 8/8 repeatable.
        # Proof and regression: a sentinel test writes a marker value into
        # the active token's own row and shows it reaching the output at S_decode=2 only.
        # ⚠️ The active-block mask below already anticipated "query i attends to its own
        # future" and closed it INSIDE the block; the same hole was open on the prior side.
        # `_mla_attend_tiled` was always correct here: it bounds by `prior_len`, the chunk
        # start, not per query.
        ctx_pos = torch.arange(S_ctx, device=hidden_states.device, dtype=torch.float32)
        query_pos = positions.float().view(B, S_decode).unsqueeze(1).unsqueeze(-1)
        active_base = positions.float().view(B, S_decode)[:, :1].view(B, 1, 1, 1)
        causal_mask = ctx_pos.view(1, 1, 1, S_ctx) < active_base
        prior_scores = torch.where(
            causal_mask,
            prior_scores,
            torch.finfo(prior_scores.dtype).min,
        )

        # Causal mask WITHIN the active block. At S_decode == 1 this is a no-op --
        # the single active token is the query itself and attending to it is
        # correct -- which is why the single-token decode path is unaffected. At
        # S_decode > 1 it is required: without it query i attends to active tokens
        # j > i, i.e. to its own future. That is exactly the shape of the
        # speculative-decoding verify step (gamma draft tokens checked in one
        # forward), so the bug would surface as MTP silently accepting drafts it
        # scored with information it should not have had.
        active_pos = positions.float().view(B, 1, 1, S_decode)
        active_scores = torch.where(
            active_pos <= query_pos,
            active_scores,
            torch.finfo(active_scores.dtype).min,
        )

        # ---- DSA selection, decode ---------------------------------------------
        # Same key index space as the prefill path: [0, S_ctx) is a cache slot and
        # S_ctx + j is active token j. S_ctx here IS max_blocks_per_seq * block_size,
        # the same quantity prefill calls `prior_bucket`, so a selection made in one
        # path means the same thing in the other. No running fold is needed: decode
        # already holds the whole prior in `prior_kv`, and S_decode is 1 (or 1 + gamma
        # under MTP), so the score block is [B, S_decode, S_ctx] and small.
        dsa_topk = None
        if self.dsa_enabled:
            if dsa is not None:
                idx_k, idx_q, idx_w = dsa
                off = self.qk_rope_head_dim + self.kv_lora_rank
                s_prior = self.indexer.index_scores_segment(
                    idx_q, idx_w, prior_kv[..., off:off + self.index_head_dim]
                )
                s_active = self.indexer.index_scores_segment(idx_q, idx_w, idx_k)
                neg = torch.full(
                    (), float("-inf"), device=hidden_states.device, dtype=s_prior.dtype
                )
                qpos_l = positions.view(B, S_decode, 1).to(torch.long)
                # The selection masks must match the two SCORE masks below exactly. A
                # mismatch would let the selection spend a slot on a key attention then
                # masks, silently shrinking the selection.
                #
                # 🔴 So the prior bound is the FIRST ACTIVE position, not each query's own
                # position — the same correction the score mask needed, and for the same
                # reason: `prior_kv` predates this step's cache write, so `ctx_pos` in
                # [p0, p0+j) is the pre-write content of the active rows. This instance was
                # missed on the first pass because it spells the bound `qpos_l` rather than
                # `query_pos`, so a grep for the score mask's spelling did not find it.
                # ⚠️ Searching for one spelling of a duplicated invariant is not searching
                # for the invariant.
                abase_l = positions.view(B, S_decode, 1)[:, :1, :].to(torch.long)
                cpos = torch.arange(
                    S_ctx, device=hidden_states.device, dtype=torch.long
                ).view(1, 1, S_ctx)
                s_prior = torch.where(cpos < abase_l, s_prior, neg)
                apos_l = positions.view(B, 1, S_decode).to(torch.long)
                s_active = torch.where(apos_l <= qpos_l, s_active, neg)

                k = min(self.index_topk, S_ctx + S_decode)
                scores_cat = torch.cat([s_prior, s_active], dim=-1)
                dsa_topk = torch.topk(scores_cat, k, dim=-1).indices.to(torch.int32)
                if dsa_state is not None:
                    dsa_state["topk"] = dsa_topk
            else:
                dsa_topk = self._dsa_consume_shared(dsa_state)

        if dsa_topk is not None:
            prior_scores = prior_scores + _dsa_segment_mask(
                dsa_topk, 0, S_ctx, prior_scores.dtype
            )
            active_scores = active_scores + _dsa_segment_mask(
                dsa_topk, S_ctx, S_ctx + S_decode, active_scores.dtype
            )

        all_scores = torch.cat([prior_scores, active_scores], dim=-1)
        all_scores = all_scores.to(torch.float32)
        attn_weights = F.softmax(all_scores, dim=-1).to(self.dtype)

        prior_weights = attn_weights[..., :S_ctx]
        active_weights = attn_weights[..., S_ctx:]

        wkv_b = self.kv_b_proj_weight.view(
            self.kv_lora_rank, self.num_heads_per_rank, self.qk_nope_head_dim + self.v_head_dim
        )
        out_absorb = wkv_b[:, :, self.qk_nope_head_dim:]

        x_prior = torch.matmul(prior_weights, ckv_prior_4d)
        attn_prior = torch.einsum('bhqc,chd->bhqd', x_prior, out_absorb)

        x_active = torch.matmul(active_weights, ckv_active_4d)
        attn_active = torch.einsum('bhqc,chd->bhqd', x_active, out_absorb)

        attn_output = attn_prior + attn_active

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B * S_decode, self.num_heads_per_rank * self.v_head_dim)

        attn_output = torch.matmul(attn_output, self.o_proj_weight)

        if self.sp_group.world_size > 1:
            self.sp_group.all_reduce(attn_output)

        return attn_output


# =============================================================================
# Section 4a: Dense MLP (for first_k_dense_replace=3 layers)
# =============================================================================


class GlmMoeDsaDenseMLP(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()

        self.tp_group = get_tp_group()
        self.sp_group = self.tp_group
        self.world_size = self.tp_group.world_size

        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size

        self.gate_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.intermediate_size_per_rank, dtype=config.torch_dtype)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.intermediate_size_per_rank, dtype=config.torch_dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_size_per_rank, config.hidden_size, dtype=config.torch_dtype)
        )

        self._setup_weight_loaders(config)

    def _setup_weight_loaders(self, config):
        gate_up_loader = _sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
        )
        down_loader = _sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
        )
        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill and self.sp_group.world_size > 1:
            hidden_states = self.sp_group.all_gather(hidden_states, dim=0)

        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        if self.sp_group.world_size > 1:
            if is_prefill:
                output = self.sp_group.reduce_scatter(output, dim=0)
            else:
                self.sp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4b: Shared Expert MLP
# =============================================================================


class GlmMoeDsaSharedExpertMLP(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.hidden_size = config.hidden_size
        shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
        self.intermediate_size_per_rank = shared_intermediate // self.world_size

        self.gate_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.intermediate_size_per_rank, dtype=config.torch_dtype)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.intermediate_size_per_rank, dtype=config.torch_dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_size_per_rank, config.hidden_size, dtype=config.torch_dtype)
        )

        self._setup_weight_loaders(config)

    def _setup_weight_loaders(self, config):
        gate_up_loader = _sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
        )
        down_loader = _sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
        )
        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )


# =============================================================================
# Section 4c: MoE Layer (256 routed experts + 1 shared expert)
# GLM uses sigmoid scoring with n_group=1 (simplified: no group-limited
# selection needed), e_score_correction_bias, and routed_scaling_factor.
# =============================================================================


def _moe_expert_weight_loader(
    num_experts: int,
    shard_dim: int,
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    def transform(slices, rank):
        tp_rank = rank % num_shards
        start_idx = tp_rank * shard_size
        end_idx = start_idx + shard_size
        expert_tensors = []
        assert len(slices) == num_experts, (
            f"Expected {num_experts} expert slices, got {len(slices)}"
        )
        for slice_obj in slices:
            sl = [slice(None)] * len(slice_obj.get_shape())
            sl[shard_dim] = slice(start_idx, end_idx)
            expert_tensors.append(slice_obj[tuple(sl)].T)
        return torch.stack(expert_tensors, dim=0)

    return SafetensorsWeightLoader(transform=transform)



class GlmMoeDsaMoE(nn.Module):
    """Mixture of Experts: 256 routed + 1 shared.

    With n_group=1, routing simplifies to standard sigmoid + top-k.
    """

    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()

        self.ep_degree = get_neuron_ep_degree()
        self.ep_enabled = self.ep_degree > 1

        if self.ep_enabled:
            self.ep_rank = get_neuron_ep_rank()
            self.ep_tp_group = get_neuron_ep_tp_group()
            self.moe_group = get_tp_group()
        else:
            self.ep_rank = 0
            self.ep_tp_group = get_tp_group()
            self.moe_group = get_tp_group()

        self.tp_degree = self.ep_tp_group.world_size
        self.rank = self.ep_tp_group.rank_in_group

        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.num_local_experts = config.n_routed_experts // self.ep_degree
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.dtype = config.torch_dtype
        self.routed_scaling_factor = config.routed_scaling_factor

        self.moe_intermediate_size = config.moe_intermediate_size
        self.intermediate_size_per_rank = config.moe_intermediate_size // self.tp_degree

        # Router: replicated on all ranks
        self.gate_weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=self.dtype)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32)
        )

        # Expert weights: LOCAL experts only, TP-sharded intermediate
        self.gate_proj_weights = nn.Parameter(
            torch.empty(self.num_local_experts, self.hidden_size, self.intermediate_size_per_rank, dtype=self.dtype)
        )
        self.up_proj_weights = nn.Parameter(
            torch.empty(self.num_local_experts, self.hidden_size, self.intermediate_size_per_rank, dtype=self.dtype)
        )
        self.down_proj_weights = nn.Parameter(
            torch.empty(self.num_local_experts, self.intermediate_size_per_rank, self.hidden_size, dtype=self.dtype)
        )

        self.shared_expert = GlmMoeDsaSharedExpertMLP(config)

        self._setup_weight_loaders(config)

    def _setup_weight_loaders(self, config):
        set_weight_loader(self.gate_weight, SafetensorsWeightLoader())
        set_weight_loader(self.e_score_correction_bias, SafetensorsWeightLoader())

        set_weight_loader(
            self.gate_proj_weights,
            _moe_expert_weight_loader(
                num_experts=self.num_local_experts,
                shard_dim=0,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.tp_degree,
            ),
        )
        set_weight_loader(
            self.up_proj_weights,
            _moe_expert_weight_loader(
                num_experts=self.num_local_experts,
                shard_dim=0,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.tp_degree,
            ),
        )
        set_weight_loader(
            self.down_proj_weights,
            _moe_expert_weight_loader(
                num_experts=self.num_local_experts,
                shard_dim=1,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.tp_degree,
            ),
        )

    def _compute_routing(self, hidden_states: torch.Tensor):
        """Compute sigmoid top-k routing (simplified: n_group=1, no group selection)."""
        router_logits = torch.matmul(
            hidden_states.to(torch.float32),
            self.gate_weight.to(torch.float32).T,
        )

        scores = torch.sigmoid(router_logits)
        scores_for_selection = scores + self.e_score_correction_bias.unsqueeze(0)

        # With n_group=1, group-limited selection is just standard top-k
        topk_weights, topk_indices = torch.topk(
            scores_for_selection, self.num_experts_per_tok, dim=-1
        )

        # Use original scores (without correction bias) for the weights
        topk_weights = scores.gather(1, topk_indices)

        # L1 normalize and scale
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor

        # Returned in FP32 on purpose. Everything above is computed in FP32 (HF's
        # GlmMoeDsaTopkRouter does the same and also returns FP32), and every
        # consumer on the supported fp8_per_channel path immediately wants FP32:
        # `full_affinities` is allocated as torch.float32 in both _forward_prefill
        # and _forward_decode, and NF.moe_tkg / NF.moe_cte take it as such. Casting
        # to self.dtype here and back to FP32 there discarded ~0.2% of every
        # routing coefficient for nothing: equivalence Stage 2 measured
        # router_topk_weights R=10.0123 with the cast and R=1.0000 without it,
        # against an unchanged R=1.0000 for router_logits and an exactly identical
        # expert selection. Consumers that genuinely want the model dtype now cast
        # explicitly at their own allocation site.
        return topk_weights, topk_indices

    def _fuse_decode_weights(self):
        """Pre-fuse gate+up weights for batched einsum decode path."""
        self._decode_gate_up_w = nn.Parameter(
            torch.cat([self.gate_proj_weights, self.up_proj_weights], dim=-1).contiguous(),
            requires_grad=False,
        )

    def _forward_decode(self, hidden_states_2d: torch.Tensor) -> torch.Tensor:
        T = hidden_states_2d.shape[0]
        topk_weights, topk_indices = self._compute_routing(hidden_states_2d)

        # _compute_routing now returns FP32; this einsum path feeds the affinities
        # straight into a bf16 einsum, so keep the previous dtype explicitly rather
        # than inheriting it (an FP32 affinity would promote the einsum output).
        topk_weights = topk_weights.to(self.dtype)
        local_affinities = torch.zeros(
            T, self.num_local_experts,
            dtype=topk_weights.dtype, device=hidden_states_2d.device
        )
        for e_local in range(self.num_local_experts):
            e_global = self.local_expert_start + e_local
            mask = (topk_indices == e_global)
            local_affinities[:, e_local] = (topk_weights * mask.to(topk_weights.dtype)).sum(dim=-1)

        gate_up = torch.einsum("th,ehi->eti", hidden_states_2d, self._decode_gate_up_w)
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = F.silu(gate) * up
        down = torch.einsum("eti,eih->eth", intermediate, self.down_proj_weights)
        output = torch.einsum("eth,te->th", down, local_affinities)

        return output

    def _forward_prefill(self, hidden_states_2d: torch.Tensor) -> torch.Tensor:
        """Prefill MoE: blockwise CTE kernel for large token counts."""
        from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
        from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
        import nki.language as nl

        num_tokens = hidden_states_2d.shape[0]
        topk_weights, topk_indices = self._compute_routing(hidden_states_2d)

        full_affinities = torch.zeros(
            num_tokens, self.num_experts, dtype=torch.float32, device=hidden_states_2d.device
        )
        full_affinities.scatter_(1, topk_indices, topk_weights.to(torch.float32))

        local_expert_indices = torch.arange(
            self.local_expert_start,
            self.local_expert_start + self.num_local_experts,
            device=hidden_states_2d.device,
            dtype=torch.int64,
        )
        local_affinities = NF.get_local_expert_affinities(full_affinities, local_expert_indices)

        block_size = 128
        affinities_masked, token_pos_to_id, block_to_expert, conditions = NF.build_blockwise_mapping(
            expert_affinities=local_affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.num_experts_per_tok,
            block_size=block_size,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
        )

        gate_up_weight = torch.stack(
            [self.gate_proj_weights, self.up_proj_weights], dim=2
        )

        output = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=hidden_states_2d,
            expert_affinities_masked=affinities_masked,
            gate_up_proj_weight=gate_up_weight,
            down_proj_weight=self.down_proj_weights,
            activation_function=ActFnType.SiLU,
            block_size=block_size,
            token_position_to_id=token_pos_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        )

        return output

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        # 🔴 Gate each collective on the world size of the group it ACTUALLY runs
        # on. `self.tp_degree` is ep_tp_group's size, which is not moe_group's size
        # once EP is on: at tensor_parallel_size=64 / ep_degree=16 they are 4 and
        # 64. The previous code gated a moe_group all_gather on `tp_degree > 1`
        # while leaving the matching moe_group reduce_scatter unguarded, so the two
        # disagreed exactly when ep_degree == tensor_parallel_size (tp_sub == 1):
        # the gather was skipped, the MoE ran on an ungathered dim-0 shard, and the
        # output was then scattered a second time. Unreachable at ep_degree=16,
        # which is why it never showed up on device -- but it is the configuration
        # a reader would try next.
        comm = self.moe_group if self.ep_enabled else self.ep_tp_group
        sharded = comm.world_size > 1

        if is_prefill and sharded:
            hidden_states = comm.all_gather(hidden_states, dim=0)

        hidden_states_2d = hidden_states.view(-1, self.hidden_size)

        if is_prefill:
            output = self._forward_prefill(hidden_states_2d)
        else:
            output = self._forward_decode(hidden_states_2d)

        # The shared expert consumes the same (gathered) activations as the routed
        # experts, so it must stay on this side of the reduction.
        output = output + self.shared_expert(hidden_states_2d)

        if sharded:
            if is_prefill:
                output = comm.reduce_scatter(output, dim=0)
            else:
                comm.all_reduce(output)

        return output


# =============================================================================
# Section 5: Decoder Layer
# =============================================================================


class GlmMoeDsaDecoderLayer(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_dense_layer = layer_idx < config.first_k_dense_replace

        self.input_layernorm = GlmMoeDsaRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = GlmMoeDsaRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = GlmMoeDsaAttention(config, layer_idx=layer_idx)

        if self.is_dense_layer:
            self.mlp = GlmMoeDsaDenseMLP(config)
        else:
            self.mlp = GlmMoeDsaMoE(config)

    def _is_decode(self, attn_metadata) -> bool:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len <= decode_token_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        dsa_state: dict | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
            dsa_state=dsa_state,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# =============================================================================


class GlmMoeDsaModel(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()
        self.config = config

        self.tp_group = get_tp_group()
        self.sp_group = self.tp_group

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.sp_group.device_group,
        )

        self.layers = nn.ModuleList(
            [
                GlmMoeDsaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = GlmMoeDsaRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = GlmMoeDsaRotaryEmbedding(config)

        # MTP: when True, forward also returns the pre-`model.norm` hidden
        # (the layer-77 residual output) for the layer-78 draft head (D3 default).
        self.capture_prenorm = False

        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.sp_group.world_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name]["decode_token_threshold"]
        is_prefill = max_query_len > decode_token_threshold

        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )

        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        # Carries the current `full` layer's DSA selection forward to the `shared`
        # layers that reuse it. A plain dict threaded through the loop rather than
        # state on a module: the layer order that defines "nearest preceding full
        # layer" is this loop's order, so the dependency belongs here where it is
        # visible, and nothing survives past one forward.
        dsa_state: dict = {}

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                dsa_state=dsa_state,
            )

        # MTP capture (D3 default): the pre-`model.norm` residual is the
        # layer-78 draft head's target hidden. Grab it before the final norm.
        prenorm_hidden = hidden_states if self.capture_prenorm else None

        hidden_states = self.norm(hidden_states)

        if is_prefill and self.sp_group.world_size > 1:
            hidden_states = self.sp_group.all_gather(hidden_states, dim=0)
            if prenorm_hidden is not None:
                prenorm_hidden = self.sp_group.all_gather(prenorm_hidden, dim=0)

        if self.capture_prenorm:
            return hidden_states, prenorm_hidden
        return hidden_states


# =============================================================================
# Section 7: Language Model Head
# =============================================================================


@async_speculative_decoding
class GlmMoeDsaForCausalLM(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()
        self.config = config
        self.model = GlmMoeDsaModel(config)

        # MTP self-speculation: when True, the spec-decode forward captures the
        # single pre-lm_head hidden state and returns it (as aux_hidden_states)
        # for the layer-78 draft head. Set by the runner. Off = the base
        # non-spec / Eagle3 path is byte-for-byte unchanged.
        self.capture_hidden_for_mtp_spec = False

        self.ep_degree = get_neuron_ep_degree()
        self.tp_group = get_tp_group()
        self.sp_group = self.tp_group
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        debug_logits_enabled = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0
        ) or debug_logits_enabled

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.sp_group.device_group,
        )
        if not config.tie_word_embeddings:
            set_weight_loader(
                self.lm_head.weight,
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=config.vocab_size // self.sp_world_size,
                    num_shards=self.sp_world_size,
                    is_storage_transposed=False,
                    pad_shard=True,
                ),
            )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.sp_group.device_group,
            )

    def set_mtp_hidden_state_capture(self, enabled: bool) -> None:
        """Enable/disable single pre-lm_head hidden capture for the MTP draft.

        When enabled, the spec-decode forward also returns the (pre-`model.norm`,
        D3 default) hidden gathered at the sampling positions, so the layer-78 MTP
        head can consume it as its target hidden state. The backbone captures the
        pre-norm residual (mirrors Eagle3 llama's ``hidden_prenorm``)."""
        self.capture_hidden_for_mtp_spec = enabled
        self.model.capture_prenorm = enabled

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise ValueError("Input Embedding as Inputs is Not Supported Yet.")

        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name]["decode_token_threshold"]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]

        if is_prefill and ((T <= self.sp_world_size) or (T % self.sp_world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > sp_world_size ({self.sp_world_size}) for SP."
            )

        model_out = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )
        # MTP capture: backbone returns (hidden, prenorm_hidden) when
        # capture is enabled; otherwise a bare hidden tensor (unchanged path).
        if self.capture_hidden_for_mtp_spec:
            hidden_states, prenorm_hidden = model_out
            # The layer-78 draft head's target hidden is the pre-`model.norm`
            # residual (D3 default). Its shape must match how the draft consumes
            # it, which differs by phase (mirrors upstream vLLM's
            # llm_base_proposer, where target_hidden_states is [num_tokens,hidden]
            # at prefill and [batch,hidden] at decode):
            #  * PREFILL: the draft prefills over the FULL prompt to seed its
            #    layer-78 KV, so it needs the full-T [T, hidden] hidden (it fuses
            #    per-token with the T-token embedding and samples at
            #    sampling_positions itself). prenorm_hidden is already full-T here
            #    (the backbone all-gathers it at model.py:1042).
            #  * DECODE: the draft consumes one hidden per request, so gather at
            #    the sampling positions → [bs, hidden].
            if is_prefill:
                mtp_hidden = prenorm_hidden
            else:
                mtp_hidden = torch.index_select(
                    prenorm_hidden, dim=0, index=sampling_positions
                )
        else:
            hidden_states = model_out

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )

        logits = self.lm_head(hidden_states_for_logits)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        gathered_logits = None
        if self._gather_logits:
            if self.sp_group is not None:
                gathered_logits = self.sp_group.all_gather(logits, dim=1)
            else:
                gathered_logits = logits

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            rejection_sampled_tokens = rejection_sampler(
                spec_decode_metadata, sampled_tokens
            )
            if self.capture_hidden_for_mtp_spec:
                # 3-tuple; the @async_speculative_decoding epilogue appends the
                # 4th element (last_accepted_token) → matches the runner's
                # 4-tuple unpack (neuron_model_runner.py).
                return rejection_sampled_tokens, mtp_hidden, gathered_logits
            return rejection_sampled_tokens

        if self.capture_hidden_for_mtp_spec:
            # Non-spec step with MTP active: emit the 3-tuple the runner unpacks
            # (sampled_tokens, aux_hidden_states, gathered_logits).
            return sampled_tokens, mtp_hidden, gathered_logits

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = GlmMoeDsaConfig.from_configs(hf_config, neuron_config)
        return cls(config)

    # -- KV Cache Management --

    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            attn = layer.self_attn
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=attn.num_kv_cache_heads,
                    head_size=attn.kv_cache_head_dim,
                    dtype=attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                    # MLA: the [k_pe | compressed_kv] latent is the whole cache;
                    # value is reconstructed via the absorbed projection, so
                    # there is no separate V buffer. Request single-buffer KV.
                    is_mla=True,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading --

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load BF16 weights from checkpoint.

        DSA indexer weights are mapped only on `full` layers and only when DSA is
        enabled, because that is when `GlmMoeDsaAttention` constructs the module. With DSA
        off the indexer tensors are present in the checkpoint and deliberately
        unmapped, which the loader reports as unexpected keys.
        """
        tp_rank = self.sp_group.rank_in_group
        tp_size = self.sp_world_size

        if self.ep_degree > 1:
            ep_rank = get_neuron_ep_rank()
            num_local_experts = self.config.n_routed_experts // self.ep_degree
            local_expert_start = ep_rank * num_local_experts
        else:
            num_local_experts = self.config.n_routed_experts
            local_expert_start = 0

        mappings = dict()

        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"
            is_dense = layer_id < self.config.first_k_dense_replace

            # MLA Attention
            mappings[f"{prefix}.self_attn.q_a_proj_weight"] = (
                f"{prefix}.self_attn.q_a_proj.weight"
            )
            mappings[f"{prefix}.self_attn.q_a_layernorm.weight"] = (
                f"{prefix}.self_attn.q_a_layernorm.weight"
            )
            mappings[f"{prefix}.self_attn.q_b_proj_weight"] = (
                f"{prefix}.self_attn.q_b_proj.weight"
            )
            mappings[f"{prefix}.self_attn.kv_a_proj_weight"] = (
                f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"
            )
            mappings[f"{prefix}.self_attn.kv_a_layernorm.weight"] = (
                f"{prefix}.self_attn.kv_a_layernorm.weight"
            )
            mappings[f"{prefix}.self_attn.kv_b_proj_weight"] = (
                f"{prefix}.self_attn.kv_b_proj.weight"
            )
            mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                f"{prefix}.self_attn.o_proj.weight"
            )

            # DSA indexer, only on `full` layers and only when DSA is on. `shared`
            # layers carry no indexer tensors in the checkpoint at all, so mapping them
            # would fail the load rather than load zeros.
            if self.model.layers[layer_id].self_attn.dsa_is_full:
                ix = f"{prefix}.self_attn.indexer"
                for ours, theirs in (
                    ("wq_b_weight", "wq_b.weight"),
                    ("wk_weight", "wk.weight"),
                    ("weights_proj_weight", "weights_proj.weight"),
                    ("k_norm_weight", "k_norm.weight"),
                    ("k_norm_bias", "k_norm.bias"),
                ):
                    mappings[f"{ix}.{ours}"] = f"{ix}.{theirs}"

            # Layer norms
            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )
            mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )

            if is_dense:
                # Dense MLP
                mappings[f"{prefix}.mlp.gate_proj_weight"] = (
                    f"{prefix}.mlp.gate_proj.weight"
                )
                mappings[f"{prefix}.mlp.up_proj_weight"] = (
                    f"{prefix}.mlp.up_proj.weight"
                )
                mappings[f"{prefix}.mlp.down_proj_weight"] = (
                    f"{prefix}.mlp.down_proj.weight"
                )
            else:
                # MoE: Router (replicated)
                mappings[f"{prefix}.mlp.gate_weight"] = (
                    f"{prefix}.mlp.gate.weight"
                )
                mappings[f"{prefix}.mlp.e_score_correction_bias"] = (
                    f"{prefix}.mlp.gate.e_score_correction_bias"
                )

                # MoE: Routed experts (LOCAL only, EP-filtered)
                expert_range = range(local_expert_start, local_expert_start + num_local_experts)
                mappings[f"{prefix}.mlp.gate_proj_weights"] = [
                    f"{prefix}.mlp.experts.{j}.gate_proj.weight"
                    for j in expert_range
                ]
                mappings[f"{prefix}.mlp.up_proj_weights"] = [
                    f"{prefix}.mlp.experts.{j}.up_proj.weight"
                    for j in expert_range
                ]
                mappings[f"{prefix}.mlp.down_proj_weights"] = [
                    f"{prefix}.mlp.experts.{j}.down_proj.weight"
                    for j in expert_range
                ]

                # MoE: Shared expert
                mappings[f"{prefix}.mlp.shared_expert.gate_proj_weight"] = (
                    f"{prefix}.mlp.shared_experts.gate_proj.weight"
                )
                mappings[f"{prefix}.mlp.shared_expert.up_proj_weight"] = (
                    f"{prefix}.mlp.shared_experts.up_proj.weight"
                )
                mappings[f"{prefix}.mlp.shared_expert.down_proj_weight"] = (
                    f"{prefix}.mlp.shared_experts.down_proj.weight"
                )

        # Backbone
        mappings["model.norm.weight"] = "model.norm.weight"
        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["lm_head.weight"] = "lm_head.weight"

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)

        load_result = checkpoint.load_sharded(
            tp_rank, tp_size, self, mappings, device, strict=False,
        )
        rank_sharded = load_result.state_dict

        # Ensure correct dtypes
        target_dtype = self.config.torch_dtype
        param_dtypes = {n: p.dtype for n, p in self.named_parameters()}
        for name, tensor in rank_sharded.items():
            expected = param_dtypes.get(name, target_dtype)
            if tensor.dtype != expected:
                rank_sharded[name] = tensor.to(expected)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
        if load_result.missing_keys:
            logger.error("MISSING weights (%d): %s", len(load_result.missing_keys), load_result.missing_keys[:20])
        if load_result.unexpected_keys:
            logger.error("UNEXPECTED weights (%d): %s", len(load_result.unexpected_keys), load_result.unexpected_keys[:20])
        loaded_keys = set(rank_sharded.keys())
        all_params = set(n for n, _ in self.named_parameters())
        not_loaded = all_params - loaded_keys
        if not_loaded:
            logger.error("PARAMS NOT LOADED (%d): %s", len(not_loaded), sorted(not_loaded)[:20])
        logger.info("Weight loading: %d params loaded, %d total params", len(loaded_keys), len(all_params))

        # NOTE: No EP scaling needed — attention and dense MLP are sharded across
        # the full world group (64 ranks), so all_reduce sums partial results correctly.

        layer0 = self.model.layers[0]
        logger.info(
            "DIAG shapes: embed=%s q_a=%s q_b=%s kv_a=%s kv_b=%s o=%s lm_head=%s",
            list(self.model.embed_tokens.weight.shape),
            list(layer0.self_attn.q_a_proj_weight.shape),
            list(layer0.self_attn.q_b_proj_weight.shape),
            list(layer0.self_attn.kv_a_proj_weight.shape),
            list(layer0.self_attn.kv_b_proj_weight.shape),
            list(layer0.self_attn.o_proj_weight.shape),
            list(self.lm_head.weight.shape),
        )

        # Pre-fuse decode weights before model moves to device/compiles
        for layer in self.model.layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, '_fuse_decode_weights'):
                layer.mlp._fuse_decode_weights()
        logger.info("Pre-fused decode MoE weights (BF16 einsum path)")


# =============================================================================
# Weight Loaders (BF16, no FP8 dequantization)
# =============================================================================


def _row_parallel_weight_loader(
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Row-parallel: shard on in_features (dim=1 in checkpoint [out, in]), transpose."""

    def transform(slices, rank):
        tp_rank = rank % num_shards
        start_idx = tp_rank * shard_size
        end_idx = start_idx + shard_size

        assert len(slices) == 1
        weight = slices[0][:]
        weight_shard = weight[:, start_idx:end_idx]
        return weight_shard.T

    return SafetensorsWeightLoader(transform=transform)


def _replicated_transposed_weight_loader() -> SafetensorsWeightLoader:
    """Replicate a 2-D weight on every rank, transposing [out, in] -> [in, out].

    For the DSA indexer, whose ~9.6 M parameters are not worth sharding: replication
    costs 0.08% of the ~11.8 GB each rank already holds and avoids a collective on the
    selection path. Distinct from the bare `SafetensorsWeightLoader()` used for the MoE
    router, which keeps the checkpoint layout and transposes on every forward instead --
    the indexer stores the transposed form so the `matmul(x, W)` in its forward needs no
    per-call transpose.
    """

    def transform(slices, rank):
        assert len(slices) == 1
        return slices[0][:].T

    return SafetensorsWeightLoader(transform=transform)


def _sharding_weight_loader(
    shard_dim: int,
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """TP-shard a weight, then transpose (checkpoint is [out, in])."""

    def transform(slices, rank):
        tp_rank = rank % num_shards
        start_idx = tp_rank * shard_size
        end_idx = start_idx + shard_size

        assert len(slices) == 1
        weight = slices[0][:]
        sl = [slice(None)] * weight.ndim
        sl[shard_dim] = slice(start_idx, end_idx)
        weight_shard = weight[tuple(sl)]
        return weight_shard.T

    return SafetensorsWeightLoader(transform=transform)
