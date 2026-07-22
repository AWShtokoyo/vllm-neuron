# SPDX-License-Identifier: Apache-2.0
"""
LlamaBidirectional (embedding) Implementation
=============================================

Neuron-native implementation of nvidia/llama-embed-nemotron-8b (model_type
`llama_bidirec`). A Llama-3 8B backbone run with BIDIRECTIONAL attention, whose
flattened ``[T, H]`` post-norm hidden states are consumed by the
NeuronModelRunner ``_pool`` path, which runs the reused upstream pooler
(MEAN-token gather + L2 normalize) via ``DispatchPooler``.

This is a POOLING/EMBEDDING model:
  - bidirectional attention (NF.flash_attention(causal_mask=False))
  - prefill-only: NO decode path, NO KV cache writes, NO lm_head, NO sampler
  - forward returns the raw ``[T, H]`` backbone hidden states (NOT [B, H]).

Design (mirrors qwen3/model_embedding.py):
    Generative:  backbone -> [T, H] -> index_select + lm_head -> [B, vocab]
    Embedding:   backbone -> [T, H]                            (returned as-is)

The pooling itself (MEAN + L2 normalize) is NOT done here — it is performed by
the reused upstream ``DispatchPooler`` in ``NeuronModelRunner._pool``. Returning
``[T, H]`` (not pre-gathered ``[B, H]``) is required so the upstream pooling
cursor can gather the right rows; see design/vllm/pooling-models.rst.

Ported from the canonical Llama dense model (llama3/model.py). Only the prefill
attention path is kept; decode/megakernel/KV machinery is removed.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    llama_bidirec-specific.

Phase-1 verified (real weights, cos=0.99997) that the bidirectional backbone +
mean-pool + L2 normalize matches the official HF LlamaBidirectionalModel +
sentence-transformers reference.
"""

import logging
import math

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

from .config import LlamaBidirecConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: RMS Normalization (same as Llama)
# =============================================================================
class RMSNorm(nn.Module):
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
# Section 2: Rotary Position Embedding (Llama3 scaling — same as llama3)
# =============================================================================
def _apply_rotary_emb(x, cos, sin):
    """Interleaved RoPE (rotate_half style), matching the Llama checkpoint."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    return _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)


class LlamaRotaryEmbedding(nn.Module):
    """Llama3-style RoPE scaling (computed fresh on-device in forward)."""

    def __init__(self, config: LlamaBidirecConfig):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim
        self.rope_scaling = config.rope_scaling
        if self.rope_scaling is not None:
            rt = self.rope_scaling.get("rope_type", self.rope_scaling.get("type"))
            if rt != "llama3":
                raise NotImplementedError(
                    f"rope_type '{rt}' not supported, only 'llama3'."
                )
            self.scaling_factor = self.rope_scaling.get("factor", 1.0)
            self.low_freq_factor = self.rope_scaling.get("low_freq_factor", 1.0)
            self.high_freq_factor = self.rope_scaling.get("high_freq_factor", 1.0)
            self.orig_max_position = self.rope_scaling.get(
                "original_max_position_embeddings", 8192
            )

    def _compute_inv_freq(self, device):
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )
        if self.rope_scaling is not None:
            inv_freq = self._apply_llama3_scaling(inv_freq)
        return inv_freq

    def _apply_llama3_scaling(self, inv_freq):
        low_wl = self.orig_max_position / self.low_freq_factor
        high_wl = self.orig_max_position / self.high_freq_factor
        wave_len = 2 * math.pi / inv_freq
        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
            )
        else:
            smooth = torch.zeros_like(wave_len)
        return torch.where(
            wave_len < high_wl,
            inv_freq,
            torch.where(
                wave_len > low_wl,
                inv_freq / self.scaling_factor,
                (1 - smooth) * inv_freq / self.scaling_factor + smooth * inv_freq,
            ),
        )

    def forward(self, position_ids, device, dtype):
        inv_freq = self._compute_inv_freq(device)
        inv_freq_expanded = inv_freq[:, None].float()
        positions_expanded = position_ids[None, :].float()
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Section 3: Bidirectional Attention (prefill-only)
# <-- MODEL-SPECIFIC: bidirectional (causal_mask=False), no KV cache, no decode
# =============================================================================
class LlamaBidirecAttention(nn.Module):
    """Multi-head GQA attention with TP head sharding, BIDIRECTIONAL, prefill-only.

    >>> PARALLELISM: TP head sharding + SP all-gather/reduce-scatter <<<
    <-- MODEL-SPECIFIC: causal_mask=False; no KV cache writes; no decode path.
    """

    def __init__(self, config: LlamaBidirecConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5

        # >>> PARALLELISM: TP group <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: head sharding (TP only; no attention DP for embedding) <<<
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in = (self.num_attention_heads * self.head_dim) // self.world_size

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in, self.hidden_size, dtype=self.dtype)
        )
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=(self.num_attention_heads * self.head_dim)
                // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, hidden_states, position_embeddings, key_bounds=None):
        """Bidirectional full-sequence attention (prefill only).

        hidden_states arrives gathered to full sequence (caller all-gathers from SP).

        key_bounds: optional (bound_min, bound_max) tensors of shape [1, T, 1]
            restricting each query to valid (non-padding) keys [0, valid_len).
            Required so bidirectional attention does not let padded tokens leak
            into the real tokens' representations.
        """
        hidden_states = hidden_states.to(self.dtype)
        tokens, _ = hidden_states.shape

        # ── QKV projection (TP-sharded heads) ──
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
        ).squeeze(0)
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(0, 1)
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)

        # ── RoPE (interleaved) ──
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ── GQA expand + BIDIRECTIONAL flash attention ──
        k = k.repeat_interleave(self.num_key_value_groups, dim=0)
        v = v.repeat_interleave(self.num_key_value_groups, dim=0)
        q_flash = q.transpose(1, 2)  # [Nh, Dh, T]
        k_flash = k.transpose(1, 2)  # [Nh, Dh, T]
        v_flash = v  # [Nh, T, Dh]
        bound_min, bound_max = (key_bounds if key_bounds is not None else (None, None))
        # flash_attention treats heads as the batch dim, so bound_min/bound_max
        # must be [Nh, T, 1] (built as [1, T, 1] upstream; broadcast per head).
        if bound_min is not None:
            nh = q_flash.shape[0]
            # .contiguous() so the kernel gets a materialized [Nh, T, 1] tensor
            # (expand alone yields a stride-0 broadcast view, which segfaults XLA).
            bound_min = bound_min.expand(nh, -1, -1).contiguous()
            bound_max = bound_max.expand(nh, -1, -1).contiguous()
        attn_output = NF.flash_attention(
            q_flash,
            k_flash,
            v_flash,
            scale=self.scaling,
            causal_mask=False,  # <-- MODEL-SPECIFIC: bidirectional
            bound_min=bound_min,
            bound_max=bound_max,
            tp_q=False,
            tp_out=True,
        )  # [Nh, Dh, T]

        # ── Output projection + SP reduce-scatter ──
        attn_output = attn_output.unsqueeze(0)
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None).squeeze(0)
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)
        return attn_output.contiguous()


# =============================================================================
# Section 4: MLP (SwiGLU, same as llama3)
# =============================================================================
class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaBidirecConfig):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
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
        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        gate_up = sharding_weight_loader(
            shard_dim=1, shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size, is_storage_transposed=True,
        )
        down = sharding_weight_loader(
            shard_dim=0, shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size, is_storage_transposed=True,
        )
        set_weight_loader(self.gate_proj_weight, gate_up)
        set_weight_loader(self.up_proj_weight, gate_up)
        set_weight_loader(self.down_proj_weight, down)

    def forward(self, hidden_states):
        # Prefill: input is already full-sequence (gathered). Compute then reduce-scatter.
        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )
        if self.world_size > 1:
            output = self.tp_group.reduce_scatter(output, dim=0)
        return output


# =============================================================================
# Section 5: Decoder Layer (prefill-only)
# =============================================================================
class LlamaBidirecLayer(nn.Module):
    def __init__(self, config: LlamaBidirecConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)
        self.self_attn = LlamaBidirecAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

    def forward(self, hidden_states, position_embeddings, key_bounds=None):
        # hidden_states is in SP layout (T/world_size per rank).
        # ── Attention (all-gather to full seq, attend, reduce-scatter back) ──
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        if self.world_size > 1:
            x = self.tp_group.all_gather(x, dim=0)
        x = self.self_attn(x, position_embeddings, key_bounds=key_bounds)
        hidden_states = residual + x
        # ── MLP (all-gather, compute, reduce-scatter) ──
        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        if self.world_size > 1:
            x = self.tp_group.all_gather(x, dim=0)
        x = self.mlp(x)
        hidden_states = residual + x
        return hidden_states


# =============================================================================
# Section 6: Backbone
# =============================================================================
class LlamaBidirecModel(nn.Module):
    def __init__(self, config: LlamaBidirecConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList(
            [LlamaBidirecLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)
        self.rotary_emb = LlamaRotaryEmbedding(config)

        emb_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.embed_tokens.vocab_size_per_rank,
            num_shards=self.embed_tokens.tp_size,
            is_storage_transposed=False,
        )
        set_weight_loader(self.embed_tokens.weight, emb_loader)

    def forward(self, input_ids, positions, key_bounds=None):
        # Prefill: embedding scatters tokens to SP layout.
        hidden_states = self.embed_tokens(input_ids, scatter_tokens=self.world_size > 1)
        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, key_bounds=key_bounds)
        hidden_states = self.norm(hidden_states)
        # Gather full sequence (SP → full) so the [T, H] contract holds for the
        # downstream pooler (which indexes rows over the flattened buffer).
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states  # [T, hidden]


# =============================================================================
# Section 7: Embedding model (returns [T, H]; pooling done by upstream pooler)
# =============================================================================
class LlamaBidirecForEmbedding(nn.Module):
    """LlamaBidirectional embedding model.

    forward() returns the flattened ``[T, H]`` post-norm backbone hidden states.
    The reused upstream ``DispatchPooler`` (MEAN-token gather + L2 normalize) is
    run by ``NeuronModelRunner._pool``, NOT here — see the module docstring and
    design/vllm/pooling-models.rst.

    >>> PARALLELISM: reuses LlamaBidirecModel's SP/TP backbone unchanged.
    <-- MODEL-SPECIFIC: bidirectional attention; drops lm_head; attaches the
    upstream DispatchPooler.
    """

    # Marks this as a pooling model for vLLM's is_pooling_model() check.
    is_pooling_model = True

    def __init__(self, config: LlamaBidirecConfig):
        super().__init__()
        self.config = config
        self.model = LlamaBidirecModel(config)
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.pooler import DispatchPooler

        vllm_config = get_current_vllm_config()

        # Embedding is prefill-only: it never loads the FP8 k_scale/v_scale
        # params the quantized attention path requires, so an fp8 KV cache would
        # crash deep in attention. Reject it up front with a clear message.
        cache_dtype = getattr(vllm_config.cache_config, "cache_dtype", None)
        if (cache_dtype or "").startswith("fp8"):
            raise ValueError(
                f"kv_cache_dtype={cache_dtype!r} is not supported for the "
                "LlamaBidirectional pooling/embedding model. It is prefill-only "
                "and does not implement FP8 KV cache; use the default (auto/bf16)."
            )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None, (
            "pooler_config is None — LlamaBidirecForEmbedding requires the "
            "pooling runner (launch with --runner pooling, or a checkpoint whose "
            "sentence-transformers modules.json resolves to a pooling model)."
        )
        # MEAN-token pooling + L2 normalize (resolved from the checkpoint's
        # sentence-transformers 1_Pooling config: pooling_mode_mean_tokens).
        self.pooler = DispatchPooler.for_embedding(pooler_config)

        # ── On-device [B, H] gather (perf optimization; ON by default) ──
        # forward() pools (MEAN + L2) on-device and returns a FIXED-shape
        # [max_num_seqs, H] embedding tensor, instead of the [T, H] hidden states.
        # This avoids copying the full [T (bucket), H] buffer off device and
        # running the upstream pooler eagerly per step — the dominant cost at
        # short-prompt / high-concurrency (e.g. 128/c4). The runner's _pool
        # detects the [B, H] shape and skips the upstream cursor/pooler.
        # Set VLLM_NEURON_POOLING_ONDEVICE_GATHER=0 to fall back to the mainline
        # [T, H] + upstream DispatchPooler path.
        import os as _os

        self._ondevice_gather = (
            _os.environ.get("VLLM_NEURON_POOLING_ONDEVICE_GATHER", "1") != "0"
        )
        # Fixed row count for the returned [B, H] so the traced NEFF is shape-
        # stable across batches of 1..max_num_seqs (trailing rows are zero and
        # dropped by the runner). Mirrors the custom impl's fixed [max_num_reqs].
        self._max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
        self.normalize = True

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        pooling_seq_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return the flattened ``[T, H]`` post-norm backbone hidden states.

        The buffer is a dense pack of one or more prefill sequences followed by
        trailing padding:  ``[seq0_real, seq1_real, ..., pad ...]`` (a single
        sequence when packing is off, several when ``VLLM_NEURON_POOLING_PACK`` is
        on). Bidirectional attention must restrict every real query to keys WITHIN
        ITS OWN sequence, so padded keys do not leak in and sequences do not
        cross-attend. We derive each token's key range ``[start, end)`` from
        ``positions`` alone (no seq_lens plumbing):

          * Each real sequence starts at position 0 and increments by 1.
          * The runner pads by REPEATING the last real position, so the padding
            breaks the +1 run.

        Thus a "run" = a maximal block whose positions increase by exactly 1;
        each real sequence is one run, and trailing padding forms separate runs
        (its output rows are ignored by the upstream pooling cursor anyway). Each
        token attends only within its own run. For a single unpacked sequence
        this reduces exactly to ``[0, valid_len)``.
        """
        positions = positions.to(torch.int32)
        T = input_ids.shape[0]

        # ── Build per-sequence key_bounds from positions (STATIC-shape device ops
        #    only — no .item()/.tolist(); a data-dependent scalar baked into the
        #    torch.compile trace SIGSEGVs the NEFF backend at runtime).
        #    Escape hatch to isolate the bounds path during bring-up.
        import os as _os

        _disable_bounds = _os.environ.get("LLAMA_BIDIREC_NO_BOUNDS") == "1"
        key_bounds = None
        if not _disable_bounds:
            device = input_ids.device
            idx = torch.arange(T, device=device, dtype=torch.int32)  # [T]
            # is_start[i] = token i begins a new run (i==0, or positions[i] is not
            # positions[i-1]+1). run_id via cumsum of is_start.
            not_contig = positions[1:] != positions[:-1] + 1  # [T-1] bool
            is_start = torch.cat(
                [torch.ones(1, dtype=torch.bool, device=device), not_contig]
            )  # [T]
            run_id = torch.cumsum(is_start.to(torch.int32), dim=0)  # [T], 1-indexed
            # same_run[i, j] = tokens i and j share a run. Static [T, T] (T<=bucket,
            # e.g. 512 -> 512x512 int32 = 1MB; same style as the pooling mask).
            same_run = run_id.unsqueeze(1) == run_id.unsqueeze(0)  # [T, T]
            idx_row = idx.unsqueeze(0).expand(T, T)  # [T, T]
            # start = min column index in the run; end (exclusive) = max+1.
            bmin = torch.where(same_run, idx_row, torch.full_like(idx_row, T)).min(dim=1).values
            bmax = torch.where(same_run, idx_row + 1, torch.zeros_like(idx_row)).max(dim=1).values
            bound_min = bmin.to(torch.int32).view(1, T, 1).contiguous()
            bound_max = bmax.to(torch.int32).view(1, T, 1).contiguous()
            key_bounds = (bound_min, bound_max)

        hidden_states = self.model(input_ids, positions, key_bounds=key_bounds)  # [T, H]

        if not self._ondevice_gather:
            return hidden_states  # [T, hidden] — upstream DispatchPooler path

        # ── On-device MEAN-pool + L2 into a fixed [max_num_seqs, H] ──
        # Avoids copying the full [T (bucket), H] buffer off device + eager
        # upstream pooler per step. Uses pooling_seq_lens (fixed shape
        # [max_num_seqs], zero-padded, on device) as the AUTHORITATIVE sequence
        # boundaries — positions alone cannot tell a real length-1 tail sequence
        # from position-0 padding, so exact counts are required here. All ops are
        # static-shape (no .item()) so the traced NEFF is reused across batches.
        device = hidden_states.device
        H = hidden_states.shape[1]
        if pooling_seq_lens is None:
            # Fallback: no lengths supplied → treat whole buffer as one sequence.
            pooled = hidden_states.mean(dim=0, keepdim=True)  # [1, H]
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
            out = torch.zeros(self._max_num_seqs, H, dtype=pooled.dtype, device=device)
            out = out + 0.0
            out[:1] = pooled
            return out

        seq_lens_i = pooling_seq_lens.to(device=device, dtype=torch.int32)  # [B]
        B = seq_lens_i.shape[0]
        tok_idx = torch.arange(T, device=device, dtype=torch.int32)  # [T]
        ends = torch.cumsum(seq_lens_i, dim=0)  # [B]
        starts = ends - seq_lens_i  # [B]
        # member[b, t] = starts[b] <= t < ends[b]  ([B, T]); real tokens only,
        # in req order. Trailing zero-length seqs (start==end) select no tokens.
        member = (tok_idx.unsqueeze(0) >= starts.unsqueeze(1)) & (
            tok_idx.unsqueeze(0) < ends.unsqueeze(1)
        )  # [B, T]
        memberf = member.to(hidden_states.dtype)  # [B, T]
        summed = memberf @ hidden_states  # [B, H]
        counts = memberf.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
        pooled = summed / counts  # [B, H]
        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
        return pooled  # [max_num_seqs, H]

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = LlamaBidirecConfig.from_configs(hf_config, neuron_config)
        return cls(config)

    # ── KV Cache ──────────────────────────────────────────────────────────
    # Embedding is prefill-only, so there is no KV cache. Provide an empty spec;
    # the runner calls get_kv_spec / bind_kv_cache unconditionally.
    def get_kv_spec(self):
        from vllm_neuron.model.kv_cache import KVSpec

        return KVSpec(layers=[])

    def bind_kv_cache(self, kv_caches):
        # Embedding model uses no KV cache.
        return

    # ── Weight loading ──────────────────────────────────────────────────────
    def load_weights(self, checkpoint_path, device, cache_dir):
        tp_rank = self.rank
        tp_size = self.world_size
        # <-- MODEL-SPECIFIC: LlamaBidirectionalModel IS a LlamaModel, so the
        # checkpoint keys have NO `model.` prefix (e.g. `embed_tokens.weight`,
        # `layers.0...`, `norm.weight`). Our parameters live under `self.model.*`,
        # so map param name -> checkpoint key by stripping the leading `model.`.
        mappings = dict()
        # embedding + final norm (NO lm_head — embedding model has no LM head).
        mappings["model.embed_tokens.weight"] = "embed_tokens.weight"
        mappings["model.norm.weight"] = "norm.weight"
        for i in range(len(self.model.layers)):
            ck = f"layers.{i}"  # checkpoint prefix (no `model.`)
            mappings[f"model.layers.{i}.self_attn.qkv_proj_weight"] = [
                f"{ck}.self_attn.q_proj.weight",
                f"{ck}.self_attn.k_proj.weight",
                f"{ck}.self_attn.v_proj.weight",
            ]
            mappings[f"model.layers.{i}.self_attn.o_proj_weight"] = f"{ck}.self_attn.o_proj.weight"
            mappings[f"model.layers.{i}.input_layernorm.weight"] = f"{ck}.input_layernorm.weight"
            mappings[f"model.layers.{i}.post_attention_layernorm.weight"] = (
                f"{ck}.post_attention_layernorm.weight"
            )
            mappings[f"model.layers.{i}.mlp.gate_proj_weight"] = f"{ck}.mlp.gate_proj.weight"
            mappings[f"model.layers.{i}.mlp.up_proj_weight"] = f"{ck}.mlp.up_proj.weight"
            mappings[f"model.layers.{i}.mlp.down_proj_weight"] = f"{ck}.mlp.down_proj.weight"

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device
        ).state_dict
        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)
        self.load_state_dict(rank_sharded, strict=False, assign=True)

    def load_weights_lite(self, checkpoint_path, device, cache_dir):
        # No KV scales to load for embedding model.
        return
