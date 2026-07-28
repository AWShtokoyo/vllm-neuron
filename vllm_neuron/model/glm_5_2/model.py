# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.2 BF16 Implementation
====================================

GLM-5.2 with Multi-head Latent Attention (MLA) and Mixture of Experts (MoE).

Key architectural features:
- MLA: compressed KV via low-rank projections + weight absorption
- MoE: 256 routed experts (top-8) + 1 shared expert, with 3 dense layers
- Standard RoPE (interleaved layout) on qk_rope_head_dim=64
- DSA (DeepSeek Sparse Attention) indexer is present in checkpoint but
  omitted from the forward path in this port (full attention used)
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

from .config import Glm52Config

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: RMS Normalization
# =============================================================================


class Glm52RMSNorm(nn.Module):
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
# GLM-5.2 uses default RoPE with rope_theta=8M, rope_interleave=true,
# on qk_rope_head_dim=64 dimensions.
# =============================================================================


class Glm52RotaryEmbedding(nn.Module):
    def __init__(self, config: Glm52Config):
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
# DSA indexer is skipped — full attention for all tokens.
# =============================================================================


class Glm52Attention(nn.Module):
    def __init__(self, config: Glm52Config, layer_idx: int):
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

        # KV cache: stores [k_pe | compressed_kv] per token
        self.kv_cache_head_dim = self.qk_rope_head_dim + self.kv_lora_rank  # 576
        self.num_kv_cache_heads = 1

        self.k_cache = None
        self.v_cache = None

        # Q path: hidden -> q_a_proj -> LayerNorm -> q_b_proj
        self.q_a_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size_per_sp, self.q_lora_rank, dtype=self.dtype)
        )
        self.q_a_layernorm = Glm52RMSNorm(
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
        self.kv_a_layernorm = Glm52RMSNorm(
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata,
            )
        else:
            if self.sp_group.world_size > 1:
                hidden_states = self.sp_group.all_gather(hidden_states, dim=0)
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

        return q_nope_absorbed, q_pe, compressed_kv, k_pe

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
        q_nope_absorbed, q_pe, compressed_kv, k_pe = self._compute_mla_qkv(
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
        q_nope_absorbed, q_pe, compressed_kv_active, k_pe_active = self._compute_mla_qkv(
            hidden_states.view(B, S_decode, hidden), cos, sin
        )

        flat_indices = block_table.reshape(-1).to(torch.long)
        num_cache_blocks = self.k_cache.shape[0]
        flat_indices = torch.clamp(flat_indices, min=0, max=num_cache_blocks - 1)
        k_blocks = torch.index_select(self.k_cache, 0, flat_indices)
        k_blocks = k_blocks.squeeze(1).view(B, max_blocks_per_seq, block_size, self.kv_cache_head_dim)
        prior_kv = k_blocks.reshape(B, S_ctx, self.kv_cache_head_dim)

        k_pe_prior = prior_kv[..., :self.qk_rope_head_dim]
        compressed_kv_prior = prior_kv[..., self.qk_rope_head_dim:]
        k_pe_prior = k_pe_prior.unsqueeze(1)

        # Update cache
        block_indices = slot_mapping // block_size
        block_indices = torch.clamp(block_indices, min=0, max=num_cache_blocks - 1)
        position_indices = slot_mapping % block_size

        k_pe_flat = k_pe_active.squeeze(1).reshape(-1, self.qk_rope_head_dim)
        ckv_flat = compressed_kv_active.reshape(-1, self.kv_lora_rank)
        combined_active = torch.cat([k_pe_flat, ckv_flat], dim=-1)

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

        # Causal mask for prior tokens
        ctx_pos = torch.arange(S_ctx, device=hidden_states.device, dtype=torch.float32)
        query_pos = positions.float().view(B, S_decode).unsqueeze(1).unsqueeze(-1)
        causal_mask = ctx_pos.view(1, 1, 1, S_ctx) < query_pos
        prior_scores = torch.where(
            causal_mask,
            prior_scores,
            torch.finfo(prior_scores.dtype).min,
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


class Glm52DenseMLP(nn.Module):
    def __init__(self, config: Glm52Config):
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


class Glm52SharedExpertMLP(nn.Module):
    def __init__(self, config: Glm52Config):
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
# GLM-5.2 uses sigmoid scoring with n_group=1 (simplified: no group-limited
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



class Glm52MoE(nn.Module):
    """Mixture of Experts: 256 routed + 1 shared.

    With n_group=1, routing simplifies to standard sigmoid + top-k.
    """

    def __init__(self, config: Glm52Config):
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

        self.shared_expert = Glm52SharedExpertMLP(config)

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
        topk_weights = topk_weights.to(self.dtype)

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
        if is_prefill and self.tp_degree > 1:
            if self.ep_enabled:
                hidden_states = self.moe_group.all_gather(hidden_states, dim=0)
            else:
                hidden_states = self.ep_tp_group.all_gather(hidden_states, dim=0)

        hidden_states_2d = hidden_states.view(-1, self.hidden_size)

        if is_prefill:
            output = self._forward_prefill(hidden_states_2d)
        else:
            output = self._forward_decode(hidden_states_2d)

        shared_output = self.shared_expert(hidden_states_2d)

        if not self.ep_enabled:
            output = output + shared_output
        elif self.ep_enabled:
            output = output + shared_output
            if is_prefill:
                output = self.moe_group.reduce_scatter(output, dim=0)
            else:
                self.moe_group.all_reduce(output)

        if not self.ep_enabled and self.tp_degree > 1:
            if is_prefill:
                output = self.ep_tp_group.reduce_scatter(output, dim=0)
            else:
                self.ep_tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: Decoder Layer
# =============================================================================


class Glm52DecoderLayer(nn.Module):
    def __init__(self, config: Glm52Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_dense_layer = layer_idx < config.first_k_dense_replace

        self.input_layernorm = Glm52RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Glm52RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = Glm52Attention(config, layer_idx=layer_idx)

        if self.is_dense_layer:
            self.mlp = Glm52DenseMLP(config)
        else:
            self.mlp = Glm52MoE(config)

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
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
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


class Glm52Model(nn.Module):
    def __init__(self, config: Glm52Config):
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
                Glm52DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Glm52RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Glm52RotaryEmbedding(config)

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

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
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
class Glm52ForCausalLM(nn.Module):
    def __init__(self, config: Glm52Config):
        super().__init__()
        self.config = config
        self.model = Glm52Model(config)

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
        config = Glm52Config.from_configs(hf_config, neuron_config)
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

        DSA indexer weights are intentionally skipped — they are present in the
        checkpoint but not used in this port (full attention is used instead).
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
