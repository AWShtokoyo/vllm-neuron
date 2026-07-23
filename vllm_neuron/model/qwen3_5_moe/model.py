# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 BF16 Implementation
============================

Hybrid multimodal model with vision encoder + text decoder.
Text decoder mixes Gated DeltaNet (linear attention) and standard
GQA full attention layers in a 3:1 ratio (30 linear + 10 full).

Key architectural features:
  - Full attention: GQA + QK-norm + output gate (sigmoid) + partial M-RoPE
  - Linear attention: Gated DeltaNet with causal conv1d + recurrent state
  - Vision: ViT encoder with spatial merger (shared with Qwen3-VL)
  - M-RoPE with partial_rotary_factor=0.25 (only 64 of 256 dims rotated)

Framework changes for hybrid support:
  - get_kv_spec() only returns specs for full attention layers
  - bind_kv_cache() only binds to full attention layers
  - Linear attention layers manage their own recurrent state
  - RecurrentStateSpec added to KVSpec for memory accounting
"""

from __future__ import annotations

import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from transformers import PretrainedConfig


def _repeat_interleave_heads(x, repeats, dim):
    """Index-free equivalent of ``x.repeat_interleave(repeats, dim=dim)``.

    ``torch.repeat_interleave`` lowers to an indirect (vector-DGE) gather on
    Neuron/XLA: the per-output index ``arange(out)//repeats`` is a runtime AP,
    so neuronx-cc cannot prove it in-bounds and emits an OOBMode.ERROR DGE that
    trips nrta-1006 at warmup_prefill@1024. The DGE-ENUM capture (graph
    1bd2820a, T_eaaac5af) named the ONLY two surviving error-mode indirect DGEs
    as ``_gather.2004``/``_gather.2159`` (dynamic_load by ``select.9`` over the
    GDN query/key head-expand tensor) == THESE call sites (model.py:975-976 GDN
    prefill q/k head expand + decode + full-attn). Equivalent broadcast form uses
    only unsqueeze/expand/reshape (NO indirect addressing) and is bit-identical:
    each slice along ``dim`` is repeated ``repeats`` times CONSECUTIVELY, which is
    exactly repeat_interleave's interleave order (CPU-verified bit-exact in the
    sibling qwen3_5_moe/model_bf16.py:33). Gated by VLLM_GDN_REPEAT_BROADCAST
    (default-on); =0 restores repeat_interleave. Campaign law: index-free rewrite,
    NOT a value clamp (clamps never remove an OOBMode.ERROR DGE).
    """
    if os.environ.get("VLLM_GDN_REPEAT_BROADCAST", "1") != "1":
        return x.repeat_interleave(repeats, dim=dim)
    shape = list(x.shape)
    exp = shape[:dim + 1] + [repeats] + shape[dim + 1:]
    out = shape[:dim] + [shape[dim] * repeats] + shape[dim + 1:]
    return x.unsqueeze(dim + 1).expand(*exp).reshape(*out)


import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.nki.nki_hop import wrap_nki

# Slot-indexed GDN decode kernels (unified-cache path). Wrapped EAGERLY at
# import — a lazy `if _X is None: _X = wrap_nki(...)` branch INSIDE the compiled forward triggers a
# graph change on the first decode step, which vLLM rejects under torch.compile stance
# 'fail_on_recompile' (EngineDeadError). Module-level wrapping keeps the traced forward branch-free.
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX as _FP8_CLAMP_MAX
from vllm_neuron.functional.gdn_conv_update import gdn_conv_update as _gdn_conv_update
from vllm_neuron.functional.gdn_state_update import gdn_state_update as _gdn_state_update
_SLOT_CONV_KERNEL = wrap_nki(_gdn_conv_update)
_SLOT_STATE_KERNEL = wrap_nki(_gdn_state_update)
# Unified-cache page-stride full-attn KV gather (VLLM_UNIFIED_KV_GATHER). Replaces the torch
# fancy-index k_cache[flat_idx] (rejected by neuronx-cc on a page-strided shared-slab view) with an
# .ap indirect DMA that lowers. Module-level wrap (branch-free traced forward). Opt-in; default off
# keeps the current standalone-alloc path byte-identical.
from vllm_neuron.functional.paged_kv_gather import paged_kv_gather as _paged_kv_gather
from vllm_neuron.functional.paged_kv_gather import paged_kv_gather_raw as _paged_kv_gather_raw
from vllm_neuron.functional.paged_kv_gather import paged_kv_write as _paged_kv_write
from vllm_neuron.functional.paged_kv_gather import paged_state_gather as _paged_state_gather
from vllm_neuron.functional.paged_kv_gather import paged_state_scatter as _paged_state_scatter
_PAGED_KV_GATHER = wrap_nki(_paged_kv_gather)
_PAGED_KV_GATHER_RAW = wrap_nki(_paged_kv_gather_raw)
_PAGED_KV_WRITE = wrap_nki(_paged_kv_write)


def _paged_kv_gather_raw_kernel():
    return _PAGED_KV_GATHER_RAW
_PAGED_STATE_GATHER = wrap_nki(_paged_state_gather)
_PAGED_STATE_SCATTER = wrap_nki(_paged_state_scatter)


def _paged_kv_gather_kernel():
    return _PAGED_KV_GATHER


def _paged_kv_write_kernel():
    return _PAGED_KV_WRITE


def _paged_state_gather_kernel():
    return _PAGED_STATE_GATHER


def _paged_state_scatter_kernel():
    return _PAGED_STATE_SCATTER


def _unified_kv_gather_on():
    import os as _os
    return (_os.environ.get("VLLM_UNIFIED_KV_GATHER") == "1"
            or _os.environ.get("VLLM_KV_GATHER_KERNEL") == "1")


def _anchor_idx(idx, anchor):
    """Tie an index tensor to a forward-input-derived `anchor` so torch.compile cannot
    CONSTANT-FOLD it to a CPU tensor. The .ap gather/scatter HOP rejects a non-XLA index
    operand ("dispatched to CPU" / "not an XLA tensor"); in PREFILL WARMUP the slot indices
    (state_indices / seed_state_indices) derive purely from arange constants and fold to CPU
    consts. Adding 0*<int slice of an XLA input> is value-preserving but pins the index into
    the XLA graph. Device-proven: const_idx probe FAILS, const_idx_fixed (anchored) PASSES.
    `idx` and `anchor` are tensors; returns idx + (0 * anchor_scalar) broadcast to idx's shape."""
    _z = (anchor.reshape(-1)[:1].to(idx.dtype) * 0).reshape(*([1] * idx.dim()))
    return idx + _z


def _slot_conv_kernel():
    return _SLOT_CONV_KERNEL


def _slot_state_kernel():
    return _SLOT_STATE_KERNEL
from vllm_neuron.model.interfaces import SupportsMRoPE, SupportsSpatialMerge
from vllm_neuron.model.kv_cache import KVSpec, HybridKVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    sharding_weight_loader_with_padding,
)

import nki.language as nl
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    NormType,
    RouterActFnType,
)
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from vllm_neuron.functional.moe.router import RouterComputationOrder
from vllm_neuron.utils.weight_loader import (
    # Public 2.31 base renamed expert_parallel_weight_loader ->
    # expert_parallel_tensor_dim_loader (identical signature). See INTERNAL §3.3.
    expert_parallel_tensor_dim_loader as expert_parallel_weight_loader,
)

from .weight_loaders_bf16 import (
    expert_gate_up_weight_loader,
    expert_down_weight_loader,
)

from .config import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Vendored attention_tkg (d_head=256) flash DECODE kernel wiring (opt-in).
# =============================================================================
# Replaces ONLY the eager scores->softmax->AV block in Qwen3_5FullAttention.
# forward_decode. Gated on VLLM_QWEN_FLASH_ATTN=1 (default OFF => the eager block
# runs byte-identical). The d_head=256 kernel additionally requires the vendored
# (512-cap) nkilib tree — VLLM_NKILIB_LATEST=1; if that tree is NOT active
# (_MAX_D_HEAD < head_dim), the flash branch self-disables and falls back to
# eager. The wrapper below is a faithful copy of the harness attn_tkg_wrapper
# (fuse_rope=False, use_pos_id=True, qk_in_sb / k_out_in_sb / out_in_sb = True)
# that the component test proved bit-matches the model eager SDPA (cos=1.0,
# flat + block KV). Module-level wrap (branch-free traced forward), mirroring the
# GDN / paged-KV kernels above. Import is guarded so a missing symbol never
# breaks module import; the flag stays inert on the default stack.
import nki  # noqa: E402
import nki.isa as _nisa  # noqa: E402

try:
    from nkilib.core.attention.attention_tkg import (
        AttnTKGConfig as _AttnTKGConfig,
        TileConstants as _AttnTileConstants,
        _compute_tile_params as _attn_compute_tile_params,
        attention_tkg as _attention_tkg,
        _MAX_D_HEAD as _ATTN_MAX_D_HEAD,
    )
    from nkilib.core.utils.allocator import SbufManager as _AttnSbufManager
    from nkilib.core.utils.logging import Logger as _AttnLogger

    _ATTN_TKG_AVAILABLE = True
except Exception as _e:  # pragma: no cover - defensive import guard
    logger.warning("attention_tkg flash decode kernel unavailable: %s", _e)
    _ATTN_TKG_AVAILABLE = False
    _ATTN_MAX_D_HEAD = 0

_ATTN_P_MAX = 128


if _ATTN_TKG_AVAILABLE:

    @nki.jit
    def _attn_tkg_flash_decode_wrapper(
        q,
        k_active,
        v_active,
        k_prior,
        v_prior,
        mask,
        rope_pos_ids,
        bs,
        q_head,
        s_active,
        s_ctx,
        d_head,
    ):
        """Faithful copy of attn_tkg_sim_harness.attn_tkg_wrapper (target config).

        fuse_rope=False so NO internal 1/sqrt(d_head) scaling is applied — the
        caller MUST pre-scale q by head_dim**-0.5. inv_freqs / start_pos_ids /
        sink = None; k_out (fuse_rope output key) = None.

        Only tensors + plain ints cross the torch.compile HOP boundary (dynamo
        cannot proxy an _AttnTKGConfig object, a dtype, or None — see the paged-KV
        kernels, which likewise pass ints only). The config object, output shapes,
        dtype, and the flat-KV active_blocks_table=None are rebuilt HERE, inside the
        nki.jit body, which runs at kernel-compile time on concrete ints.
        """
        dtype = nl.bfloat16
        active_blocks_table = None
        cfg = _AttnTKGConfig(
            bs, q_head, s_active, s_ctx, s_ctx, d_head, 0,
            tp_k_prior=True, strided_mm1=False, use_pos_id=True,
            qk_in_sb=True, k_out_in_sb=True, out_in_sb=True,
        )
        attn_out_shape = (d_head, bs * q_head * s_active)
        _d_tile = min(d_head, _ATTN_P_MAX)
        _n_d_tiles = math.ceil(d_head / _ATTN_P_MAX)
        attn_out_sb_shape = (_d_tile, _n_d_tiles * bs * q_head * s_active)
        _attn_compute_tile_params(
            cfg,
            _AttnTileConstants.get_tile_constants(),
            q,
            k_prior,
            v_prior,
            k_active,
            v_active,
            active_blocks_table,
        )

        sbm = _AttnSbufManager(
            0,
            nl.tile_size.total_available_sbuf_size - 16 * 1024,
            _AttnLogger("SBM"),
        )
        sbm.open_scope()

        d_tile_size = min(cfg.d_head, _ATTN_P_MAX)
        n_d_tiles = math.ceil(cfg.d_head / _ATTN_P_MAX)

        out = sbm.alloc_stack(attn_out_sb_shape, dtype=dtype, buffer=nl.sbuf)
        k_out = None  # fuse_rope False

        q_free = cfg.bs * cfg.q_head * cfg.s_active
        q_input = sbm.alloc_stack(
            (d_tile_size, n_d_tiles * q_free), dtype=q.dtype, buffer=nl.sbuf
        )
        for i_d in range(n_d_tiles):
            d_start = i_d * d_tile_size
            _nisa.dma_copy(
                q_input[:, i_d * q_free : (i_d + 1) * q_free],
                q[d_start : d_start + d_tile_size, :],
            )
        k_free = cfg.bs * cfg.s_active
        k_active_input = sbm.alloc_stack(
            (d_tile_size, n_d_tiles * k_free), dtype=k_active.dtype, buffer=nl.sbuf
        )
        for i_d in range(n_d_tiles):
            d_start = i_d * d_tile_size
            _nisa.dma_copy(
                k_active_input[:, i_d * k_free : (i_d + 1) * k_free],
                k_active[d_start : d_start + d_tile_size, :],
            )

        attn_out, k_out = _attention_tkg(
            q_input,
            k_active_input,
            v_active,
            k_prior,
            v_prior,
            mask,
            out,
            cfg,
            sbm,
            None,  # inv_freqs (fuse_rope False)
            rope_pos_ids,
            None,  # start_pos_ids
            None,  # sink
            active_blocks_table,
            k_out,
            DBG_TENSORS=None,
            max_context_len=None,
        )

        out_bqh = cfg.bs * cfg.q_head * cfg.s_active
        attn_out_hbm = nl.ndarray(
            attn_out_shape, dtype=attn_out.dtype, buffer=nl.shared_hbm,
            name="attn_out_hbm",
        )
        for i_d in range(n_d_tiles):
            d_start = i_d * d_tile_size
            src_offset = i_d * out_bqh
            _nisa.dma_copy(
                dst=attn_out_hbm[d_start : d_start + d_tile_size, :],
                src=attn_out[:, src_offset : src_offset + out_bqh],
            )
        attn_out = attn_out_hbm

        sbm.close_scope()
        return (attn_out,)

    _ATTN_TKG_FLASH_DECODE = wrap_nki(_attn_tkg_flash_decode_wrapper)
else:
    _ATTN_TKG_FLASH_DECODE = None


def _attn_tkg_flash_decode_kernel():
    return _ATTN_TKG_FLASH_DECODE


def _qwen_flash_decode_on(head_dim: int) -> bool:
    """Flash decode is used only when explicitly enabled AND the vendored
    (>=head_dim capacity) nkilib tree is active. Otherwise -> eager (byte-id)."""
    return (
        os.environ.get("VLLM_QWEN_FLASH_ATTN") == "1"
        and _ATTN_TKG_AVAILABLE
        and _ATTN_MAX_D_HEAD >= head_dim
    )


# =============================================================================
# Section 1: RMS Normalization
# =============================================================================


class Qwen3_5RMSNorm(nn.Module):
    """RMS Normalization — uses (1 + weight) pattern like HF Qwen3Next."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (hidden_states * (1.0 + self.weight.float())).to(input_dtype)


class Qwen3_5RMSNormGated(nn.Module):
    """Gated RMSNorm for DeltaNet output: norm(x) * weight * silu(z).

    NOTE: HF's gated norm (Qwen3_5MoeRMSNormGated, transformers
    modeling_qwen3_5_moe.py:175-189) uses PLAIN `weight` with ONES init —
    NOT the (1+weight) zeros-init convention used by the NON-gated Qwen3_5RMSNorm.
    The loader copies HF's ones-centered gated-norm weight as-is, so applying
    (1+weight) here would compute ~2x the correct GDN output in every GDN layer
    (the prefill-degeneration root cause). Mirror HF exactly: plain weight, ones init.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # DEFAULT = the device-PROVEN all-fp32 path (kept fp32 through the gate, final
        # cast only) — this is what generated "Paris" correctly at the milestone (commit
        # 12ac7cc6e). Commit 91c277f46 changed it to a mid-gate bf16 round-trip ("match
        # HF bf16 discipline"); that was measured σ-neutral at seq128 but NOT verified to
        # preserve the working generation, and the device regressed (Paris->empty, "1000.."
        # degenerate). Use the fp32 path.
        return (self.weight.float() * hidden_states * F.silu(z.float())).to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding (M-RoPE with partial rotation)
# =============================================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply partial rotary position embedding.

    Only the first rotary_dim dimensions get RoPE; the rest pass through.
    cos/sin are [T, rotary_dim/2], doubled to [T, rotary_dim].
    q/k shape: [Nh, T, Dh]
    """
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(0)  # [1, T, rotary_dim]
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(0)

    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat((q_rot, q_pass), dim=-1)
    k_embed = torch.cat((k_rot, k_pass), dim=-1)
    return q_embed, k_embed


class Qwen3_5RotaryEmbedding(nn.Module):
    """M-RoPE with partial rotary factor for Qwen3.5.

    head_dim=256, partial_rotary_factor=0.25 → rotary_dim=64
    inv_freq has 32 entries. mrope_section=[11,11,10] (sums to 32).
    """

    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config

        dim = config.rotary_dim  # 64
        base = config.rope_theta
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    def forward(
        self,
        position_ids: torch.Tensor,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin from 3D M-RoPE position IDs.

        Args:
            position_ids: [3, T] int tensor (temporal, height, width) or [T] for text-only.
        Returns:
            (cos, sin) each [T, rotary_dim/2]
        """
        if position_ids.dim() == 1:
            # Text-only: replicate position_ids across all 3 sections
            position_ids = position_ids.unsqueeze(0).expand(3, -1)

        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)

        # Split inv_freq by mrope_section
        sections = self.mrope_section
        freq_splits = torch.split(inv_freq, sections)

        cos_parts = []
        sin_parts = []
        for i, (freq_sec, sec_size) in enumerate(zip(freq_splits, sections)):
            pos = position_ids[i].float()  # [T]
            freqs = torch.outer(pos, freq_sec)  # [T, sec_size]
            cos_parts.append(freqs.cos())
            sin_parts.append(freqs.sin())

        cos = torch.cat(cos_parts, dim=-1).to(dtype)  # [T, rotary_dim/2]
        sin = torch.cat(sin_parts, dim=-1).to(dtype)
        return cos, sin


# =============================================================================
# Section 3: Full Attention (GQA + QK-norm + Output Gate)
# =============================================================================


class Qwen3_5FullAttention(nn.Module):
    """Full attention with GQA, QK-norm, output gating, and partial RoPE.

    Key differences from Qwen3:
      - q_proj outputs 2x head_dim per head (query + gate)
      - Attention output is multiplied by sigmoid(gate)
      - Partial RoPE (only rotary_dim of head_dim gets rotated)
      - head_dim=256 (larger than typical 128)
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.dtype = config.torch_dtype

        # >>> PARALLELISM: TP sharding (with KV-head replication) <<<
        # Q heads split evenly across ranks. KV heads: when world_size >= nkv
        # (our config: nkv=2, world=8), each KV head is REPLICATED across
        # num_kv_replicas (= world // nkv) ranks so every rank holds exactly one
        # local KV head (nkv_local=1) instead of zero-width k/v from 2//8=0.
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = self.num_key_value_heads // self.world_size
            self.num_kv_replicas = 1
        # GQA groups per rank = local Q heads / local KV heads.
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        self.scaling = self.head_dim ** -0.5

        # Q proj outputs 2x (query + gate) per head
        q_out_dim = self.num_attention_heads_per_rank * self.head_dim * 2
        k_out_dim = self.num_key_value_heads_per_rank * self.head_dim
        v_out_dim = self.num_key_value_heads_per_rank * self.head_dim
        total_qkv_dim = q_out_dim + k_out_dim + v_out_dim

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, total_qkv_dim, dtype=config.torch_dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(
                self.num_attention_heads_per_rank * self.head_dim,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        )

        # QK-norm (per-head, on head_dim)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, config.torch_dtype)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, config.torch_dtype)

        # Split indices for QKV
        self.qkv_split_indices = [q_out_dim, q_out_dim + k_out_dim]

        # KV cache (bound externally)
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

        # Weight loaders. KV is replicated: kv_size is the PER-RANK kv slice
        # (num_key_value_heads_per_rank * head_dim), and the loader derives the
        # source kv head from rank via kv_rank = rank // num_kv_replicas.
        q_out = self.num_attention_heads * self.head_dim * 2
        set_weight_loader(self.qkv_proj_weight, fused_qkv_weight_loader(
            q_size=q_out // self.world_size,
            kv_size=self.num_key_value_heads_per_rank * self.head_dim,
            shard_dim=1,
            num_shards=self.world_size,
            is_storage_transposed=True,
            num_kv_replicas=self.num_kv_replicas,
        ))
        set_weight_loader(self.o_proj_weight, sharding_weight_loader(
            shard_dim=0,
            shard_size=(self.num_attention_heads * self.head_dim) // self.world_size,
            num_shards=self.world_size,
            is_storage_transposed=True,
        ))

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
                hidden_states, positions, position_embeddings, attn_metadata
            )
        else:
            if self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            return self.forward_prefill(
                hidden_states, positions, position_embeddings, attn_metadata
            )

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # QKV projection
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
        ).squeeze(0)

        q_gate, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        # Split q into query and gate
        q_gate = q_gate.view(tokens, self.num_attention_heads_per_rank, self.head_dim * 2)
        q = q_gate[..., :self.head_dim]
        gate = q_gate[..., self.head_dim:]

        # Reshape to [Nh, T, Dh]
        q = q.transpose(0, 1)
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)

        # QK-norm before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Partial RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, self.rotary_dim)

        # KV cache update
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size

        # FP8 KV write: CLAMP to the platform fp8 max before the cast. On trn2 the
        # kernel interprets fp8 as legacy e4m3 (max 240), but torch.float8_e4m3fn is
        # OCP e4m3fn (max 448) — a raw .to(fp8) of any K/V value in (240,448] stores
        # an out-of-range codepoint the trn2 runtime reads back as inf/garbage ->
        # NaN in decode attention -> 1-token collapse. Clamp to FP8_CLAMP_MAX (240 on
        # trn2, 448 on trn3) so every stored value round-trips. Mirrors llama3
        # (model.py:572-582); scale-free (no k_scale) since KV-cache fp8 needs no
        # calibrated scale. bf16/fp32 caches skip the clamp -> byte-identical.
        _kv_fp8 = self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        k_r = k.reshape(-1, self.head_dim)
        v_r = v.reshape(-1, self.head_dim)
        if _kv_fp8:
            k_r = k_r.clamp(-_FP8_CLAMP_MAX, _FP8_CLAMP_MAX)
            v_r = v_r.clamp(-_FP8_CLAMP_MAX, _FP8_CLAMP_MAX)
        k_flat = k_r.to(self.k_cache.dtype)
        v_flat = v_r.to(self.v_cache.dtype)

        head_indices_for_put = torch.arange(
            self.num_key_value_heads_per_rank,
            dtype=torch.long,
            device=hidden_states.device,
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices_for_put = block_indices.repeat(self.num_key_value_heads_per_rank)
        position_indices_for_put = position_indices.repeat(self.num_key_value_heads_per_rank)

        # DECISIVE ISOLATION (env-gated VLLM_FULLATTN_SKIP_KVWRITE=1): the seq=1024
        # prefill OOB has now survived removal of EVERY GDN indirect-DGE variant
        # (chunked-reshape, single-chunk, segmented, dense-split T_5d1338ba graph
        # 6b28d2bb) AND the MoE find_nonzero/indexed_flatten dispatch DGE
        # (TORCH_DISPATCH_MIN_T=1024, exonerated x4). The surviving vector-DGE must be
        # in a path COMMON to all those graphs. This full-attn prefill KV write is
        # exactly such a path: index_put_ is an indirect scatter whose EXTENT is
        # T*num_kv_heads (=1024*heads), present in every full-attn layer of every
        # graph. The earlier KV test (T_22d054dc) only clamped index VALUES — which,
        # per the find_nonzero lesson, cannot fix a DGE whose problem is the scatter
        # mechanism / extent rather than out-of-range values; that "exoneration" is
        # therefore unsound. Skipping the cache write during PREFILL is numerically
        # safe: at prefill the attention reads k_flash/v_flash from the LOCAL k/v
        # (k=k.repeat_interleave(...) below), NOT from a k_cache gather
        # (kv_segment_size=0, no prior context), so the prefill attention OUTPUT is
        # unchanged. The cache only matters for subsequent DECODE; warmup / prefill-
        # only e2e is unaffected. If the OOB CLEARS with this on, this KV index_put_
        # is the culprit (permanent fix = DGE-free one-hot-matmul scatter, the
        # Neuron-viable form already used for GDN conv_state). If it PERSISTS,
        # full-attn KV is finally exonerated by EXTENT and the survivor is the
        # embedding / lm_head gather. Default (unset) keeps the graph byte-identical.
        if _unified_kv_gather_on() and getattr(self, "_kv_raw_slab", None) is not None:
            # UNIFIED (round-6 audit Bug 1): write K/V into the CONTIGUOUS raw slab via the .ap
            # positional write (raw slab -> free reshape; strided k_cache index_put_ would not lower).
            # K and V share the slab at columns _kv_k_off / _kv_v_off. Contract (trap #1): the write
            # persists to the manager-visible slab (self._kv_raw_slab aliases the same storage as the
            # bound k_cache/v_cache views), so a later gather of the block sees it. dst includes the
            # K/V column. vals [T, nkh, head_dim]; k is head-major [nkh, T, hd].
            _T = position_indices.shape[0]
            _ps = int(self._kv_raw_slab.shape[1])  # page_stride = raw slab row width
            _kv = self.num_key_value_heads_per_rank
            # 32-BIT-SAFE (overflow fix): pass block_id (SMALL) and position SEPARATELY — DO NOT form
            # the flat `block*_ps` int32 offset (it overflowed int32 once num_blocks*_ps >= 2**31,
            # capping the KV cache at num_blocks<=16383). The kernel builds the head_dim-slot index
            # block_id*(_ps//head_dim)+pos and lets the DMA scale by head_dim in a wider-than-32-bit
            # address space; _kv_k_off/_kv_v_off select the K/V page column. ANCHOR to k_flat
            # (input-derived) so the indices aren't constant-folded to CPU in prefill warmup. _anchor_idx.
            _bid = _anchor_idx(block_indices.to(torch.int32).reshape(_T, 1), k_flat)
            _pos = _anchor_idx(position_indices.to(torch.int32).reshape(_T, 1), k_flat)
            _kvals = k_flat.reshape(_kv, _T, self.head_dim).permute(1, 0, 2).contiguous()  # [T, nkh, hd]
            _vvals = v_flat.reshape(_kv, _T, self.head_dim).permute(1, 0, 2).contiguous()
            # SINGLE call writes BOTH K and V (one input->one output alias; two calls would violate
            # XLA's at-most-one-alias-per-param -> neuronx-cc error 70). Persist IN PLACE (copy_):
            # the bound slab is threaded as a graph input only via in-place mutation; reassignment
            # demotes it to a CPU constant -> the HOP dispatches to _cpu_impl (see _seed_state note).
            self._kv_raw_slab.copy_(_paged_kv_write_kernel()[2](
                self._kv_raw_slab, _kvals, _vvals, _bid, _pos, _T, _kv, block_size, self.head_dim,
                _ps, self._kv_k_off, self._kv_v_off))
        elif os.environ.get("VLLM_FULLATTN_SKIP_KVWRITE") != "1":
            self.k_cache.index_put_(
                (block_indices_for_put, head_indices_for_put, position_indices_for_put),
                k_flat,
            )
            self.v_cache.index_put_(
                (block_indices_for_put, head_indices_for_put, position_indices_for_put),
                v_flat,
            )

        # APC / CHUNKED-PREFILL PREFIX-KV (audit Finding #1): on a prefix-cache hit (or any chunked
        # prefill), only the NEW tokens are fed here with cached_seq_len>0 prior tokens already in the
        # paged KV cache. The default flash path below attends ONLY the local chunk -> drops all prior
        # context -> silently wrong continuation. When kv_segment_size is set, attend over prior+local.
        #
        # NOTE (audit 2026-07-08): NF.segmented_attention is UNUSABLE here — it hard-requires
        # head_dim<=128 (kernel raises, torch-fallback silently skips the check) and this model is
        # head_dim=256. Instead we mirror the PROVEN device-safe paged path from forward_decode
        # (this same method, ~648-728): gather the prior paged window, scatter the live post-RoPE new
        # K/V at their TRUE absolute positions (read-after-write safe — the index_put_ above may not
        # be visible to a same-graph gather on neuronx-cc), then attend with an absolute-position
        # causal mask. Static shapes, no .item(), head_dim-agnostic. B=1 in prefill (Neuron).
        # The gate + o_proj below are shared.
        if kv_segment_size:
            B = block_table.shape[0]
            num_blocks_per_seq = block_table.shape[1]
            S_ctx = num_blocks_per_seq * block_size
            nkh = self.num_key_value_heads_per_rank
            nqh = self.num_attention_heads_per_rank
            S_q = tokens // B

            # Dequant fp8 cache BEFORE fancy-index (fancy-indexing fp8 is unsupported; mirror decode).
            if self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                k_src = self.k_cache.to(self.dtype)
                v_src = self.v_cache.to(self.dtype)
            else:
                k_src = self.k_cache
                v_src = self.v_cache
            flat_idx = block_table.reshape(-1)
            if _unified_kv_gather_on() and getattr(self, "_kv_raw_slab", None) is not None:
                # UNIFIED-CACHE PATH: gather K and V from the CONTIGUOUS raw slab via the .ap
                # page-stride kernel. Passing the raw slab (not the strided k_cache/v_cache views)
                # avoids the reshape((num_blocks, per_block))-of-a-strided-view that is non-contiguous
                # and device-rejected. Returns NATURAL [B,nblk,nkh,block,hd]; same permute below.
                K_g, V_g = _paged_kv_gather_raw_kernel()[2](
                    self._kv_raw_slab, _anchor_idx(block_table.to(torch.int32), hidden_states),
                    B, num_blocks_per_seq, nkh, block_size, self.head_dim,
                    self._kv_k_off, self._kv_v_off,
                )
                if K_g.dtype != self.dtype:
                    K_g = K_g.to(self.dtype)
                    V_g = V_g.to(self.dtype)
            else:
                K_blocks = k_src[flat_idx]   # [B*nblk, nkh, block_size, Dh] (prior), compute dtype
                V_blocks = v_src[flat_idx]
                K_g = K_blocks.view(B, num_blocks_per_seq, nkh, block_size, self.head_dim)
                V_g = V_blocks.view(B, num_blocks_per_seq, nkh, block_size, self.head_dim)
            # .contiguous() before reshape — same device non-contiguous-reshape hazard as forward_decode.
            K_gathered = K_g.permute(0, 2, 1, 3, 4).reshape(B, nkh, S_ctx, self.head_dim)
            V_gathered = V_g.permute(0, 2, 1, 3, 4).reshape(B, nkh, S_ctx, self.head_dim)

            # Scatter live (post-RoPE) new K/V into the window at each token's true position.
            k_bnsd = k.reshape(nkh, B, S_q, self.head_dim).permute(1, 0, 2, 3)
            v_bnsd = v.reshape(nkh, B, S_q, self.head_dim).permute(1, 0, 2, 3)
            scatter_pos = positions.long().view(B, 1, S_q, 1).expand(B, nkh, S_q, self.head_dim)
            K_gathered = K_gathered.scatter(2, scatter_pos, k_bnsd.to(K_gathered.dtype))
            V_gathered = V_gathered.scatter(2, scatter_pos, v_bnsd.to(V_gathered.dtype))

            # GQA: replicate each KV head to its query-head group.
            if self.num_key_value_groups > 1:
                K_gathered = _repeat_interleave_heads(K_gathered, self.num_key_value_groups, dim=1)
                V_gathered = _repeat_interleave_heads(V_gathered, self.num_key_value_groups, dim=1)

            # Eager FP32 attention, absolute-position causal mask: query at true position p attends
            # gathered slots 0..p (prior context + causal-within-chunk + its own just-scattered slot).
            q_bnsd = q.reshape(nqh, B, S_q, self.head_dim).permute(1, 0, 2, 3)
            query_pos = positions.float().view(B, 1, S_q, 1)
            gathered_idx = torch.arange(
                S_ctx, device=hidden_states.device, dtype=torch.float32
            ).view(1, 1, 1, S_ctx)
            causal_mask = gathered_idx <= query_pos
            scores = torch.matmul(q_bnsd.float(), K_gathered.float().transpose(-2, -1)) * self.scaling
            scores = scores.masked_fill(~causal_mask, -1e9)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V_gathered.float()).to(self.dtype)
            attn_output = attn_output.permute(0, 2, 1, 3).reshape(tokens, nqh * self.head_dim)

            gate_sigmoid = torch.sigmoid(gate.reshape(tokens, -1))
            attn_output = attn_output * gate_sigmoid
            attn_output = NF.o_proj(attn_output.unsqueeze(0), self.o_proj_weight, None).squeeze(0)
            if self.world_size > 1:
                attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)
            return attn_output.contiguous()

        # Flash attention (or SDPA fallback for head_dim > 128) — LOCAL-only path (kv_segment_size=0,
        # single-shot prefill, no prior context). Byte-identical to before this branch was added.
        k = _repeat_interleave_heads(k, self.num_key_value_groups, dim=0)
        v = _repeat_interleave_heads(v, self.num_key_value_groups, dim=0)

        q_flash = q.transpose(1, 2)  # [Nh, Dh, T]
        k_flash = k.transpose(1, 2)
        v_flash = v  # [Nh, T, Dh]

        # DIAGNOSTIC (env VLLM_FULLATTN_FP32=1): run the prefill attention in an
        # explicit FP32 path instead of NF.flash_attention. The fan-out localized the
        # σ-divergence to a PROMPT-DEPENDENT magnitude spike in THIS layer's attention
        # output at the deepest full-attn layer (L31) where the residual is largest;
        # formula/RoPE/scaling all match HF. The remaining hypothesis is that the
        # bf16 q@k score matmul loses precision on large logits before softmax (HF runs
        # fp32). This computes scores=fp32(q)@fp32(k)*scale, causal-mask, fp32 softmax,
        # @fp32(v) — matching HF's precision. If L31 cosine recovers, bf16 attention
        # logit precision is the cause. Default (unset) keeps NF.flash_attention.
        if os.environ.get("VLLM_FULLATTN_FP32") == "1":
            qf = q.float()                       # [Nh, T, Dh]
            kf = k.float()
            vf = v.float()
            scores = torch.matmul(qf, kf.transpose(-1, -2)) * self.scaling  # [Nh,T,T]
            T = scores.shape[-1]
            cmask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device), diagonal=1)
            scores = scores.masked_fill(cmask, torch.finfo(torch.float32).min)
            probs = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(probs, vf).to(self.dtype)  # [Nh, T, Dh]
            attn_output = attn_output.transpose(0, 1).reshape(tokens, -1)  # [T, Nh*Dh]
        else:
            attn_output = NF.flash_attention(
                q_flash, k_flash, v_flash,
                scale=self.scaling,
                tp_q=False, tp_out=True,
            )
            # tp_out=True → [Nh, Dh, T], reshape to [T, Nh*Dh]
            attn_output = attn_output.permute(2, 0, 1).reshape(tokens, -1)

        gate_sigmoid = torch.sigmoid(gate.reshape(tokens, -1))  # [T, Nh*Dh]
        attn_output = attn_output * gate_sigmoid

        # Output projection
        attn_output = attn_output.unsqueeze(0)
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)
        attn_output = attn_output.squeeze(0)

        # >>> PARALLELISM: Reduce-scatter to SP <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """Eager paged decode.

        The fused NF.attention_decode megakernel cannot serve qwen3.5 attention:
        it bakes in FULL RoPE (rotates the whole head_dim with interleaved
        even/odd pairing), but qwen3.5 uses PARTIAL RoPE (only rotary_dim of
        head_dim, rotate_half half-split pairing) plus a fused output gate. So
        we decode EAGERLY here — same partial-RoPE + QK-norm + gate math as
        forward_prefill, reading the PAGED kv cache (bloom-style gather) instead
        of recomputing the whole sequence.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank
        nqh = self.num_attention_heads_per_rank

        # --- QKV projection + gate split (mirror forward_prefill) ---
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
        ).squeeze(0)
        q_gate, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q_gate = q_gate.view(tokens, nqh, self.head_dim * 2)
        q = q_gate[..., :self.head_dim]            # [T, Nh, Dh]
        gate = q_gate[..., self.head_dim:]         # [T, Nh, Dh]

        # [Nh, T, Dh] for QK-norm + partial RoPE (same helpers as prefill)
        q = q.transpose(0, 1)
        k = k.view(tokens, nkh, self.head_dim).transpose(0, 1)
        v = v.view(tokens, nkh, self.head_dim).transpose(0, 1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, self.rotary_dim)

        # --- Gather the PRIOR paged window, scatter the current step's live K/V
        # into it at the TRUE position, attend, then write the cache AFTER. This
        # avoids a write-then-read-back of self.k_cache within one traced graph —
        # an in-place index_put_ followed by an indexed gather of the SAME tensor
        # is not guaranteed to be ordered under neuronx-cc/XLA, so the gather can
        # read the stale (pre-write) slot at the current token's own position.
        # On CPU eager the write is always visible (component tests pass); on
        # device the hazard corrupts the current token's KV every step, collapsing
        # greedy decode into a period-2 oscillation. Mirrors the read-safe pattern
        # in qwen3_5_moe/_paged_decode_attn (gather prior -> splice live -> write
        # after) and bloom's eager paged decode. ---
        flat_idx = block_table.reshape(-1)
        # FP8 KV cache (F1): dequant the WHOLE cache to compute dtype BEFORE the
        # [flat_idx] gather — NOT after. Fancy-indexing (self.k_cache[flat_idx]) an
        # fp8 tensor is not reliably supported (the reference attention_decode fallback
        # dequants first for exactly this reason: "PyTorch does not support fancy
        # indexing on float8 dtypes", attention_decode.py:632-640). Indexing INTO the
        # fp8 tensor gathered garbage on device -> decode read collapsed (first token,
        # from prefill, was correct; step-2 cache read broke). Dequant-then-gather
        # mirrors the reference `K_cache.to(X.dtype)[flat_idx]`. fp8 is storage-only;
        # all downstream layout + attention run exactly as the bf16 path. bf16/fp32
        # caches take the no-op else branch => byte-identical to the validated decode.
        if self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            k_src = self.k_cache.to(self.dtype)
            v_src = self.v_cache.to(self.dtype)
        else:
            k_src = self.k_cache
            v_src = self.v_cache
        num_blocks_per_seq = block_table.shape[1]
        if _unified_kv_gather_on() and getattr(self, "_kv_raw_slab", None) is not None:
            # UNIFIED-CACHE PATH (prefill twin): gather K/V from the CONTIGUOUS raw slab via the .ap
            # page-stride kernel (raw slab -> free reshape; strided-view reshape would be rejected).
            K_g, V_g = _paged_kv_gather_raw_kernel()[2](
                self._kv_raw_slab, _anchor_idx(block_table.to(torch.int32), hidden_states),
                B, num_blocks_per_seq, nkh, block_size, self.head_dim,
                self._kv_k_off, self._kv_v_off,
            )
            if K_g.dtype != self.dtype:
                K_g = K_g.to(self.dtype)
                V_g = V_g.to(self.dtype)
        else:
            K_blocks = k_src[flat_idx]   # [B*nblk, nkh, block_size, Dh] (PRIOR), compute dtype
            V_blocks = v_src[flat_idx]
            K_g = K_blocks.view(B, num_blocks_per_seq, nkh, block_size, self.head_dim)
            V_g = V_blocks.view(B, num_blocks_per_seq, nkh, block_size, self.head_dim)
        K_gathered = K_g.permute(0, 2, 1, 3, 4).reshape(B, nkh, S_ctx, self.head_dim)
        V_gathered = V_g.permute(0, 2, 1, 3, 4).reshape(B, nkh, S_ctx, self.head_dim)

        # Scatter the live (post-RoPE) K/V into the gathered window at the TRUE
        # position p (= positions). This is a write into a LOCAL tensor, not the
        # cache, so there is no aliased read-after-write. K is [Nh? no, nkh] here.
        # k/v are [nkh, T, Dh]; reshape to per-batch [B, nkh, S_decode, Dh].
        k_bnsd = k.reshape(nkh, B, S_decode, self.head_dim).permute(1, 0, 2, 3)
        v_bnsd = v.reshape(nkh, B, S_decode, self.head_dim).permute(1, 0, 2, 3)
        # scatter index: absolute position of each decode token along S_ctx.
        scatter_pos = positions.long().view(B, 1, S_decode, 1).expand(
            B, nkh, S_decode, self.head_dim
        )
        K_gathered = K_gathered.scatter(2, scatter_pos, k_bnsd.to(K_gathered.dtype))
        V_gathered = V_gathered.scatter(2, scatter_pos, v_bnsd.to(V_gathered.dtype))

        # GQA: replicate each KV head to its query-head group.
        if self.num_key_value_groups > 1:
            K_gathered = _repeat_interleave_heads(K_gathered, self.num_key_value_groups, dim=1)
            V_gathered = _repeat_interleave_heads(V_gathered, self.num_key_value_groups, dim=1)

        # --- Attention core: scores -> softmax -> AV. Default = eager (FP32).
        # Behind VLLM_QWEN_FLASH_ATTN=1 (+ vendored d256 nkilib), the SAME
        # scores->softmax->AV is served by the attention_tkg NKI flash kernel,
        # which the component test proved bit-matches this eager block (cos=1.0).
        # The flash path is gated to the plain bf16 + default-KV + S_decode==1
        # case (fp8 / unified-cache / spec-decode fall back to eager); the K/V
        # gather above is common to all, so the kernel is math-agnostic to it.
        # CRITICAL: fuse_rope=False => the kernel applies NO 1/sqrt(d_head), so
        # q is PRE-SCALED by self.scaling here (NOT inside the kernel). ---
        _use_flash = (
            _qwen_flash_decode_on(self.head_dim)
            and S_decode == 1
            and self.dtype == torch.bfloat16
            and self.k_cache.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
            and not (
                _unified_kv_gather_on()
                and getattr(self, "_kv_raw_slab", None) is not None
            )
        )
        if _use_flash:
            d = self.head_dim

            # q [nqh, T, Dh] -> [B, nqh, S, Dh], PRE-SCALE by self.scaling, then to
            # kernel layout [Dh, B*nqh*S] (== inverse of the component test's
            # reshape(d,bs,q_head,s).permute(1,2,0,3).permute(0,1,3,2)).
            q_bnsd = q.reshape(nqh, B, S_decode, d).permute(1, 0, 2, 3)
            k_flash_q = (q_bnsd * self.scaling).permute(3, 0, 1, 2).reshape(
                d, B * nqh * S_decode
            ).contiguous()

            # k_active [Dh, B*S] from live post-RoPE k [nkh=1, T, Dh]; v_active
            # [B, 1, S, Dh]. NOT scaled (only q carries the scaling).
            k_bn = k.reshape(nkh, B, S_decode, d).permute(1, 0, 2, 3)  # [B,1,S,Dh]
            k_active = k_bn[:, 0].permute(2, 0, 1).reshape(
                d, B * S_decode
            ).contiguous()
            v_active = v.reshape(nkh, B, S_decode, d).permute(1, 0, 2, 3).contiguous()

            # prior K/V: the kernel wants prior SEPARATE from the active token and
            # masks prior to slots 0..pos_id-1 (EXCLUDING the active slot p), then
            # splices k_active at p. K_gathered/V_gathered are post-scatter+post-GQA
            # [B, nqh, S_ctx, Dh]; head 0 is the (consecutively-replicated) single
            # KV head, and its scattered slot p is masked out of the prior, so
            # K_gathered[:, 0:1] is exactly the flat prior window the kernel needs
            # ([B, 1, S_ctx, Dh]). This matches the audited prior/active split.
            k_prior = K_gathered[:, 0:1, :, :].contiguous()
            v_prior = V_gathered[:, 0:1, :, :].contiguous()

            # active-only causal mask [S, B, nqh, S] uint8 (== harness transposed).
            _m = torch.tril(
                torch.ones(S_decode, S_decode, dtype=torch.float32,
                           device=hidden_states.device)
            )
            mask = _m[None, None].expand(B, nqh, S_decode, S_decode).permute(
                3, 0, 1, 2
            ).contiguous().to(torch.uint8)

            rope_pos_ids = positions.reshape(B, S_decode).to(torch.float32)

            # Pass ONLY tensors + plain ints across the torch.compile HOP boundary.
            # The _AttnTKGConfig, output shapes, dtype, and flat-KV active_blocks_table=None
            # are reconstructed inside the wrapper (they are not dynamo-proxyable).
            _res = _attn_tkg_flash_decode_kernel()[2](
                k_flash_q,
                k_active,
                v_active,
                k_prior,
                v_prior,
                mask,
                rope_pos_ids,
                B,
                nqh,
                S_decode,
                S_ctx,
                d,
            )
            kernel_out = _res[0] if isinstance(_res, (list, tuple)) else _res
            # [Dh, B*nqh*S] -> [B, nqh, S, Dh] -> [T, nqh*Dh] (== eager's layout).
            attn_bnsd = kernel_out.reshape(d, B, nqh, S_decode).permute(1, 2, 3, 0)
            attn_out = attn_bnsd.permute(0, 2, 1, 3).reshape(
                tokens, nqh * self.head_dim
            ).to(self.dtype)
        else:
            # --- Eager attention (FP32 scores). True-position causal mask: the query
            # at absolute position p attends gathered slots 0..p inclusive (its own
            # true slot included, since we wrote it above). No fixed-tail clause. ---
            q_bnsd = q.reshape(nqh, B, S_decode, self.head_dim).permute(1, 0, 2, 3)
            query_pos = positions.float().view(B, 1, S_decode, 1)
            gathered_idx = torch.arange(
                S_ctx, device=hidden_states.device, dtype=torch.float32
            ).view(1, 1, 1, S_ctx)
            causal_mask = gathered_idx <= query_pos

            q_f32 = q_bnsd.float()
            scores = torch.matmul(q_f32, K_gathered.float().transpose(-2, -1)) * self.scaling
            scores = scores.masked_fill(~causal_mask, -1e9)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_weights, V_gathered.float()).to(self.dtype)
            # [B, Nh, S_decode, Dh] -> [T, Nh*Dh]
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(tokens, nqh * self.head_dim)

        # --- Output gate (sigmoid), then o_proj ---
        gate_sigmoid = torch.sigmoid(gate.reshape(tokens, -1))
        attn_out = attn_out * gate_sigmoid

        attn_out = NF.o_proj(attn_out.unsqueeze(0), self.o_proj_weight, None).squeeze(0)

        # --- Write the current step's post-RoPE K/V into the paged cache AFTER
        # attention (no read-after-write on the cache this graph). Future decode
        # steps gather this as prior context. True-slot write via slot_mapping. ---
        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        # FP8 KV write-back: clamp to platform fp8 max before cast (see prefill
        # write comment ~L456 — trn2 legacy-e4m3 max=240 vs torch e4m3fn max=448).
        _kv_fp8 = self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        k_r = k.reshape(-1, self.head_dim)
        v_r = v.reshape(-1, self.head_dim)
        if _kv_fp8:
            k_r = k_r.clamp(-_FP8_CLAMP_MAX, _FP8_CLAMP_MAX)
            v_r = v_r.clamp(-_FP8_CLAMP_MAX, _FP8_CLAMP_MAX)
        k_flat = k_r.to(self.k_cache.dtype)
        v_flat = v_r.to(self.v_cache.dtype)
        head_indices_for_put = torch.arange(
            nkh, dtype=torch.long, device=hidden_states.device
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices_for_put = block_indices.repeat(nkh)
        position_indices_for_put = position_indices.repeat(nkh)
        if _unified_kv_gather_on() and getattr(self, "_kv_raw_slab", None) is not None:
            # UNIFIED (decode twin): .ap positional write into the CONTIGUOUS raw slab, K/V at columns
            # _kv_k_off/_kv_v_off, persisted via copy_-back (trap #1). k is head-major [nkh,T,hd].
            _T = position_indices.shape[0]
            _ps = int(self._kv_raw_slab.shape[1])
            # 32-BIT-SAFE (overflow fix, decode twin): pass block_id + position SEPARATELY, NO flat
            # `block*_ps` int32 offset (see prefill twin). ANCHOR to k_flat (input-derived).
            _bid = _anchor_idx(block_indices.to(torch.int32).reshape(_T, 1), k_flat)
            _pos = _anchor_idx(position_indices.to(torch.int32).reshape(_T, 1), k_flat)
            _kvals = k_flat.reshape(nkh, _T, self.head_dim).permute(1, 0, 2).contiguous()
            _vvals = v_flat.reshape(nkh, _T, self.head_dim).permute(1, 0, 2).contiguous()
            # SINGLE call writes BOTH K and V (one input->one output alias; two would violate XLA
            # at-most-one-alias-per-param -> neuronx-cc error 70). Persist IN PLACE (copy_) so the
            # bound slab stays threaded as a graph input (reassign -> CPU const -> _cpu_impl).
            self._kv_raw_slab.copy_(_paged_kv_write_kernel()[2](
                self._kv_raw_slab, _kvals, _vvals, _bid, _pos, _T, nkh, block_size, self.head_dim,
                _ps, self._kv_k_off, self._kv_v_off))
        else:
            self.k_cache.index_put_(
                (block_indices_for_put, head_indices_for_put, position_indices_for_put),
                k_flat,
            )
            self.v_cache.index_put_(
                (block_indices_for_put, head_indices_for_put, position_indices_for_put),
                v_flat,
            )

        # >>> PARALLELISM: TP all-reduce <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(attn_out)

        return attn_out


# =============================================================================
# Section 4: Linear Attention (Gated DeltaNet)
# =============================================================================


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize along the given dimension. Use the raw sum-of-squares form
    `x * rsqrt(sum(x^2) + eps)` to match HF (modeling:231-232: eps INSIDE the
    radicand) and the device-proven ancestor (model_bf16.py:117), NOT
    F.normalize (which clamps the norm to >= eps — a different eps semantics)."""
    return x * torch.rsqrt(x.pow(2).sum(dim=dim, keepdim=True) + eps)


class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear attention layer.

    Implements a recurrent linear attention mechanism that replaces KV cache
    with a fixed-size recurrent state [B, num_heads, key_dim, value_dim].
    Uses causal conv1d for short-term memory and chunked delta rule for
    long-range dependencies.

    For Neuron compilation, this uses the pure-PyTorch fallback paths
    (no flash_linear_attention or causal_conv1d CUDA kernels needed).
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype

        # >>> PARALLELISM: shard linear-attn heads across the full TP/world group.
        # key/value heads divide by world (16/8=2, 32/8=4); gqa=2 preserved per-rank.
        # conv_dim/key_dim/value_dim shard cleanly; each rank holds its own head
        # slice of in_proj/conv/state + a partial out_proj combined by
        # reduce_scatter (prefill) / all_reduce (decode). <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        assert self.num_k_heads % self.world_size == 0, \
            f"num_k_heads {self.num_k_heads} not divisible by world {self.world_size}"
        assert self.num_v_heads % self.world_size == 0, \
            f"num_v_heads {self.num_v_heads} not divisible by world {self.world_size}"
        self.num_k_heads_local = self.num_k_heads // self.world_size
        self.num_v_heads_local = self.num_v_heads // self.world_size
        self.key_dim_local = self.head_k_dim * self.num_k_heads_local
        self.value_dim_local = self.head_v_dim * self.num_v_heads_local
        self.conv_dim_local = self.key_dim_local * 2 + self.value_dim_local

        # Conv1d over QKV (depthwise), per-rank channels.
        self.conv_dim = self.key_dim * 2 + self.value_dim  # full (for loaders/HF layout)
        self.conv1d_weight = nn.Parameter(
            torch.empty(self.conv_dim_local, self.conv_kernel_size, dtype=config.torch_dtype)
        )

        # Input projections — stored [hidden, out_local] (in-major), out sharded.
        self.in_proj_qkv_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.conv_dim_local, dtype=config.torch_dtype)
        )
        self.in_proj_z_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.value_dim_local, dtype=config.torch_dtype)
        )
        # a and b projections (for dt and beta), per-rank value heads.
        self.in_proj_a_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.num_v_heads_local, dtype=config.torch_dtype)
        )
        self.in_proj_b_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.num_v_heads_local, dtype=config.torch_dtype)
        )

        # Discretization parameters, per-rank value heads.
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads_local, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads_local, dtype=torch.float32))

        # Output norm (gated RMSNorm)
        self.norm = Qwen3_5RMSNormGated(
            self.head_v_dim, config.rms_norm_eps, config.torch_dtype
        )

        # Output projection — row-parallel: input (value_dim) sharded per-rank,
        # output H is a partial sum (reduced in forward). Stored [value_dim_local, H].
        self.out_proj_weight = nn.Parameter(
            torch.empty(self.value_dim_local, config.hidden_size, dtype=config.torch_dtype)
        )

        # Recurrent state (bound externally via bind_mamba_state for decode)
        self.recurrent_state: torch.Tensor | None = None
        self.conv_state: torch.Tensor | None = None

        # >>> PARALLELISM: head-sharding weight loaders. The rebase stores
        # in_proj/out_proj TRANSPOSED ([hidden, out]); loaders transpose HF
        # [out, hidden] and slice the correct (post-transpose) axis. <<<
        from .weight_loaders_bf16 import (
            gated_deltanet_in_proj_qkv_loader,
            gated_deltanet_conv1d_loader,
            gated_deltanet_dim1_head_loader,
            gated_deltanet_dim0_head_loader,
            gated_deltanet_out_proj_loader,
        )
        if self.world_size > 1:
            set_weight_loader(self.in_proj_qkv_weight, gated_deltanet_in_proj_qkv_loader(
                self.key_dim, self.value_dim, self.world_size))
            set_weight_loader(self.conv1d_weight, gated_deltanet_conv1d_loader(
                self.key_dim, self.value_dim, self.world_size))
            set_weight_loader(self.in_proj_z_weight, gated_deltanet_dim1_head_loader(self.world_size))
            set_weight_loader(self.in_proj_a_weight, gated_deltanet_dim1_head_loader(self.world_size))
            set_weight_loader(self.in_proj_b_weight, gated_deltanet_dim1_head_loader(self.world_size))
            set_weight_loader(self.dt_bias, gated_deltanet_dim0_head_loader(self.world_size))
            set_weight_loader(self.A_log, gated_deltanet_dim0_head_loader(self.world_size))
            set_weight_loader(self.out_proj_weight, gated_deltanet_out_proj_loader(
                self.value_dim, self.world_size))
        else:
            # world=1: transpose in_proj/out_proj, squeeze conv1d.
            transpose_loader = SafetensorsWeightLoader(
                transform=lambda slices, rank: slices[0][:].t()
            )
            conv_loader = SafetensorsWeightLoader(
                transform=lambda slices, rank: slices[0][:].squeeze(1)
            )
            set_weight_loader(self.in_proj_qkv_weight, transpose_loader)
            set_weight_loader(self.in_proj_z_weight, transpose_loader)
            set_weight_loader(self.in_proj_a_weight, transpose_loader)
            set_weight_loader(self.in_proj_b_weight, transpose_loader)
            set_weight_loader(self.conv1d_weight, conv_loader)
            set_weight_loader(self.out_proj_weight, transpose_loader)

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool,
        positions: torch.Tensor | None = None,
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Forward pass dispatching to prefill or decode."""
        if is_prefill:
            return self.forward_prefill(hidden_states, positions=positions,
                                        attn_metadata=attn_metadata)
        else:
            return self.forward_decode(hidden_states, attn_metadata=attn_metadata)

    def _slot_index(self, attn_metadata, batch_size, device):
        """Per-request page slot for the unified paged state (block_table[:,0]).

        Returns the runner-supplied state_indices tensor [batch_size], or None when
        no metadata is present (e.g. CPU module tests) so the caller falls back to
        the contiguous row index. Shared by prefill (seed) and decode (read/write)
        so both address the SAME physical state row in the shared pool.
        """
        if attn_metadata is None:
            return None
        md = attn_metadata.get(f"layers.{self.layer_idx}.linear_attn")
        if md is None:
            md = attn_metadata.get(f"language_model.layers.{self.layer_idx}.linear_attn")
        if md is not None and md.get("state_indices") is not None:
            return md["state_indices"][:batch_size].long()
        return None

    def _apc_prefill_seed(self, attn_metadata, device):
        """T2 APC: build the initial recurrent state for a cache-hit prefill.

        Reads the saved prefix-boundary recurrent state from the paged block the
        runner threaded in (``seed_state_indices``), masked by per-seq
        ``has_initial_state`` (num_computed>0). Prefill is batch=1 on Neuron, so
        this returns a ``[1, *recurrent_state.shape[1:]]`` fp32 tensor when APC is
        active and the metadata is present, else ``None`` (caller falls back to
        the byte-identical zero/None path). The read is an index-free one-hot
        gather (Neuron-viable, matches the decode _gather_rows pattern); the
        has_init mask makes a fresh request seed exactly zeros == initial_state
        None.

        GATE: activates purely on the presence of the APC metadata keys
        (``seed_state_indices`` + ``has_initial_state``), which the runner emits
        ONLY under ``_mamba_apc_enabled()`` (enable_prefix_caching / align). So
        non-APC (keys absent) -> None -> byte-identical. VLLM_GDN_APC_SEED=0 is
        an explicit device-iteration disable escape hatch (default: active)."""
        if os.environ.get("VLLM_GDN_APC_SEED") == "0":
            return None
        if attn_metadata is None or self.recurrent_state is None:
            return None
        md = attn_metadata.get(f"layers.{self.layer_idx}.linear_attn")
        if md is None:
            md = attn_metadata.get(f"language_model.layers.{self.layer_idx}.linear_attn")
        if md is None:
            return None
        seed_idx = md.get("seed_state_indices")
        has_init = md.get("has_initial_state")
        if seed_idx is None or has_init is None:
            return None
        # GUARD (review #2): [:1] is correct ONLY for batch-1 prefill (Neuron SP path,
        # platform-enforced). Batched prefill would silently give seqs 1..B-1 seq-0's seed.
        if seed_idx.shape[0] > 1 and not bool((seed_idx == seed_idx[0]).all()):
            raise AssertionError(
                "APC recurrent seed assumes batch=1 (distinct seed_state_indices detected); "
                "batched prefill would mis-seed. See _apc_prefill_seed / review #2.")
        seed_idx = seed_idx[:1].long()          # batch=1 prefill
        has_init = has_init[:1].to(torch.float32).view(1, *([1] * (self.recurrent_state.dim() - 1)))
        # One-hot gather of the seed block row (exact 0/1, one nonzero per row),
        # identical mechanism to the decode _gather_rows matmul path.
        N = self.recurrent_state.shape[0]
        if _unified_kv_gather_on() and getattr(self, "_rec_raw_slab", None) is not None:
            # UNIFIED: gather seed row from the CONTIGUOUS raw slab (recurrent component) via .ap
            # page-stride gather — raw slab (not the strided recurrent_state view) so no rejected
            # reshape. row_elems=state size, page_stride=raw row width, state_off=recurrent column.
            _re = 1
            for _d in self.recurrent_state.shape[1:]:
                _re *= _d
            _rs = self._rec_raw_slab
            _g = _paged_state_gather_kernel()[2](
                _rs, seed_idx.to(torch.int32).reshape(1, 1), 1, _re,
                int(_rs.shape[1]), int(self._rec_raw_off),
            )
            seed = _g.reshape(1, *self.recurrent_state.shape[1:]).to(torch.float32)
        else:
            slots = torch.arange(N, device=device)
            P = (seed_idx.view(1, 1) == slots.view(1, N)).to(torch.float32)   # [1, N]
            sflat = self.recurrent_state.reshape(N, -1).to(torch.float32)
            seed = (P @ sflat).reshape(1, *self.recurrent_state.shape[1:])     # [1, *tail]
        # Mask: fresh request (has_init=0) -> zeros == initial_state None.
        out = seed * has_init
        return out

    def _apc_conv_seed(self, attn_metadata, device):
        """T2 APC (conv restore, mirrors _apc_prefill_seed for the CONV window).

        On a cache-hit prefill, return the SAVED prior conv window [1, conv_dim_local, k-1]
        from the seed block (conv_state[seed_slot]), masked by has_initial_state so a FRESH
        request gets zeros (== the zero-left-pad path). This mirrors upstream
        causal_conv1d_fn(has_initial_state=..., cache_indices=[:,0]) which restores the prior
        conv window on a cache hit; without it prefill zero-left-pads and loses the prefix's
        last k-1 conv inputs. Returns None when APC off / keys absent (byte-identical fallback).
        """
        if os.environ.get("VLLM_GDN_APC_SEED") == "0":
            return None
        if attn_metadata is None or self.conv_state is None:
            return None
        md = attn_metadata.get(f"layers.{self.layer_idx}.linear_attn")
        if md is None:
            md = attn_metadata.get(f"language_model.layers.{self.layer_idx}.linear_attn")
        if md is None:
            return None
        seed_idx = md.get("seed_state_indices")
        has_init = md.get("has_initial_state")
        if seed_idx is None or has_init is None:
            return None
        # GUARD (review #2): batch-1-only seed (see _apc_prefill_seed); fail loud if batched.
        if seed_idx.shape[0] > 1 and not bool((seed_idx == seed_idx[0]).all()):
            raise AssertionError(
                "APC conv seed assumes batch=1 (distinct seed_state_indices detected).")
        seed_idx = seed_idx[:1].long()          # batch=1 prefill
        has_init = has_init[:1].to(torch.float32).view(1, *([1] * (self.conv_state.dim() - 1)))
        # One-hot gather of the seed conv block (same index-free matmul as recurrent seed).
        N = self.conv_state.shape[0]
        k1 = self.conv_kernel_size - 1
        # Reshape the gathered slot to the SEMANTIC conv window [1, conv_dim_local, k-1] (NOT the SD
        # container layout (k-1, conv_dim)); byte-consistent with _seed_state's flatten/reshape write
        # and the layout forward_prefill's cat needs.
        if _unified_kv_gather_on() and getattr(self, "_conv_raw_slab", None) is not None:
            # UNIFIED: conv_state PAGE-STRIDED -> reshape(N,-1) rejects. Gather seed via .ap.
            _re = self.conv_dim_local * k1
            _cs = self._conv_raw_slab
            _g = _paged_state_gather_kernel()[2](
                _cs, seed_idx.to(torch.int32).reshape(1, 1), 1, _re,
                int(_cs.shape[1]), int(self._conv_raw_off),
            )
            seed = _g.reshape(1, self.conv_dim_local, k1).to(torch.float32)
        else:
            slots = torch.arange(N, device=device)
            P = (seed_idx.view(1, 1) == slots.view(1, N)).to(torch.float32)   # [1, N]
            cflat = self.conv_state.reshape(N, -1).to(torch.float32)
            seed = (P @ cflat).reshape(1, self.conv_dim_local, k1)             # [1, conv_dim_local, k-1]
        has_init = has_init.reshape(1, 1, 1)
        return (seed * has_init).to(self.conv_state.dtype)

    def _carry_forward(self, attn_metadata) -> None:
        """IN-GRAPH block->block state carry-forward (task #106): move each request's running
        GDN state prev_block -> curr_block when a step opens a new block, batched over ALL pairs
        via ONE .ap gather + ONE .ap scatter per slab (conv, recurrent) — replacing the eager
        per-row copy in preprocess_mamba's _execute_plan that overflowed the DMA descriptor at
        concurrency (dmem_copy ret=-7). The runner injects carry_src_ids/carry_dst_ids
        [max_num_reqs,1] int32 (-1 == oob_mode.skip). Absent/all-(-1) -> no-op (gather of -1 is
        zeros, scatter of -1 writes nothing). Only on the unified raw-slab path."""
        if not (_unified_kv_gather_on() and getattr(self, "_rec_raw_slab", None) is not None):
            return
        md = attn_metadata.get(f"layers.{self.layer_idx}.linear_attn") if attn_metadata else None
        if md is None and attn_metadata is not None:
            md = attn_metadata.get(f"language_model.layers.{self.layer_idx}.linear_attn")
        if md is None:
            return
        src = md.get("carry_src_ids")
        dst = md.get("carry_dst_ids")
        if src is None or dst is None:
            return
        n_rows = int(src.shape[0])
        # anchor the block-id vectors to a graph tensor so warmup doesn't const-fold them to CPU.
        _src = _anchor_idx(src.to(torch.int32).reshape(n_rows, 1), self.A_log)
        _dst = _anchor_idx(dst.to(torch.int32).reshape(n_rows, 1), self.A_log)
        for _slab, _off, _rowe in (
            (self._conv_raw_slab, int(self._conv_raw_off), self.conv_dim_local * (self.conv_kernel_size - 1)),
            (self._rec_raw_slab, int(self._rec_raw_off), self.num_v_heads_local * self.head_k_dim * self.head_v_dim),
        ):
            _g = _paged_state_gather_kernel()[2](
                _slab, _src, n_rows, int(_rowe), int(_slab.shape[1]), _off,
            )  # [n_rows, rowe] — src rows gathered (rows with src=-1 -> zeros, unused)
            _slab.copy_(_paged_state_scatter_kernel()[2](
                _slab, _g, _dst, n_rows, int(_rowe), int(_slab.shape[1]), _off,
            ))  # scatter to dst rows (-1 -> skip); untouched rows preserved (in-place .ap)

    def forward_prefill(self, hidden_states: torch.Tensor, positions: torch.Tensor | None = None,
                        attn_metadata: object | None = None) -> torch.Tensor:
        """Chunked delta rule for prefill (context encoding).

        >>> PARALLELISM (SP): the backbone runs sequence-parallel during prefill
        (each rank holds T/world tokens). The GDN scan is RECURRENT over the full
        sequence, so we all_gather hidden to full-T at entry, run the rank-local
        head slice over full T, and reduce_scatter the out_proj partial sum back
        to SP-local at exit (matching the full-attention layer's contract). <<<
        """
        batch_size = 1  # SP: single sequence during prefill
        # T2 APC (task #106): in-graph block->block carry-forward BEFORE reading state, so the
        # scan seeds from the correct (migrated) block. No-op unless carry ids are present.
        self._carry_forward(attn_metadata)
        # Slot-indexed prefill seed: write the prefill state at the request's PERSISTENT page slot
        # (state_indices) so decode, which reads the same slot, finds it after condense. Kept as a
        # TENSOR slot (no .item() — that breaks torch.compile); the write uses a one-hot scatter
        # (Neuron-viable), NOT a dynamic index assign (non-contiguous, rejected). state_idx is None
        # only on CPU module tests -> contiguous row [0].
        _seed_slot = self._slot_index(attn_metadata, batch_size, hidden_states.device)
        # T2 APC (piece 3): on a cache-hit prefill, seed the recurrent scan from
        # the SAVED prefix-boundary state (stored in the paged block the runner
        # threaded in via seed_state_indices) instead of zero. has_initial_state
        # (num_computed>0) masks it so a FRESH request seeds zeros — numerically
        # identical to the initial_state=None path. Returns a [1, *rec_tail]
        # tensor or None (APC off). Read here so all scan branches can consume it.
        _apc_init = self._apc_prefill_seed(attn_metadata, hidden_states.device)
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        seq_len = hidden_states.shape[0]

        def _seed_state(state, row, page_stride=0, raw_slab=None, raw_off=None):
            # Write the single prefill state `row` [1, ...] into `state` at _seed_slot (tensor) via a
            # one-hot scatter when slot-indexed, else into contiguous row [0]. row shape =
            # [1, *state.shape[1:]]. Persists via IN-PLACE copy_ (returns None), because — per the
            # decode Option-5 note (~L1665) — a bound tensor is threaded as a compiled-graph INPUT
            # ONLY when it is mutated IN PLACE; attribute REASSIGNMENT makes dynamo capture it as a
            # CPU CONSTANT, so the .ap HOP then receives a CPU tensor and dispatches to _cpu_impl
            # ("NKI kernel dispatched to CPU"). So the raw slab MUST be persisted with copy_, exactly
            # like recurrent_state/conv_state on the default path.
            if _seed_slot is None:
                state[0] = row[0]
                return None
            if _unified_kv_gather_on() and raw_slab is not None:
                # UNIFIED: `state` is a PAGE-STRIDED view -> reshape(N,-1) rejects. Scatter row 0 to
                # slot _seed_slot via the .ap page-stride scatter (contiguous internal, lowers), then
                # persist IN PLACE into the raw slab (raw_slab is the graph-input tensor; copy_ keeps
                # it threaded/XLA, reassign would demote it to a CPU const -> CPU dispatch).
                _re = 1
                for _d in state.shape[1:]:
                    _re *= _d
                # ANCHOR the index to a forward-input-derived tensor (`row`) so torch.compile
                # cannot CONSTANT-FOLD _si to a CPU tensor (prefill-warmup slot indices derive from
                # arange consts). See _anchor_idx.
                _si = _anchor_idx(_seed_slot.to(torch.int32).reshape(1, 1), row)
                _pool = _paged_state_scatter_kernel()[2](
                    raw_slab, row.reshape(1, _re).to(raw_slab.dtype), _si,
                    1, _re, int(raw_slab.shape[1]), int(raw_off),
                )
                raw_slab.copy_(_pool)
                return None
            N = state.shape[0]
            slots = torch.arange(N, device=state.device)
            P = (_seed_slot.view(1, 1) == slots.view(1, N)).to(torch.float32)   # [1, N]
            sflat = state.reshape(N, -1).to(torch.float32)
            rflat = row.reshape(1, -1).to(torch.float32)
            Pt = P.transpose(0, 1)                                             # [N, 1]
            state.copy_((sflat - Pt @ (P @ sflat) + Pt @ rflat).reshape(N, *state.shape[1:]).to(state.dtype))

        # PADDING MASK (VLLM_GDN_PAD_STATE, default ON): mirror HF apply_mask_to_padding_states.
        # The runner right-pads (positions stop incrementing at the last real token), so trailing
        # rows are padding. Zero them BEFORE the conv/scan so padding contributes nothing to the
        # carried conv_state/recurrent_state used by decode. Without this, decode state is polluted
        # by padding -> corrupted generation (prefill outputs stay clean since the conv is causal).
        if os.environ.get("VLLM_GDN_PAD_STATE", "1") == "1" and positions is not None:
            last_real = torch.argmax(positions.to(torch.int32))
            keep = (torch.arange(seq_len, device=hidden_states.device) <= last_real)
            hidden_states = hidden_states * keep.unsqueeze(-1).to(hidden_states.dtype)

        hidden_states_3d = hidden_states.unsqueeze(0)  # [1, T, H]

        # Project to QKV and Z
        def _ip(w):
            return hidden_states_3d @ w
        qkv = _ip(self.in_proj_qkv_weight)  # [1, T, key*2+value]
        z = _ip(self.in_proj_z_weight)  # [1, T, value_dim]
        a = _ip(self.in_proj_a_weight)  # [1, T, num_v_heads]
        b = _ip(self.in_proj_b_weight)  # [1, T, num_v_heads]

        # Causal conv1d (pure-torch fallback)
        qkv_t = qkv.transpose(1, 2)  # [1, conv_dim, T]
        # Pad and apply depthwise conv. T2 APC (conv restore): on a cache-hit prefill, left-pad
        # with the SAVED prior conv window (conv_state[seed_slot]) instead of zeros, mirroring
        # upstream causal_conv1d_fn(has_initial_state, cache_indices). _apc_conv_seed returns
        # [1, conv_dim_local, k-1] (has_initial_state-masked -> zeros for a fresh request, i.e.
        # byte-identical to the F.pad zero path) or None (APC off / keys absent).
        _apc_conv = self._apc_conv_seed(attn_metadata, hidden_states.device)
        if _apc_conv is not None:
            # prepend the prior window along the T axis (dim=-1); no zero pad.
            qkv_padded = torch.cat([_apc_conv.to(qkv_t.dtype), qkv_t], dim=-1)
        else:
            qkv_padded = F.pad(qkv_t, (self.conv_kernel_size - 1, 0))
        # Depthwise conv: weight is [conv_dim, kernel_size]
        _convw = self.conv1d_weight.unsqueeze(1)  # [conv_dim_local, 1, kernel_size]
        qkv_conv = F.conv1d(
            qkv_padded,
            _convw,
            None,
            groups=self.conv_dim_local,
        )
        qkv_conv = F.silu(qkv_conv)  # [1, conv_dim_local, T]
        qkv_conv = qkv_conv.transpose(1, 2)  # [1, T, conv_dim_local]

        # Update conv state for future decode steps (fixed-slice write, XLA-safe).
        # PADDING-AWARE (VLLM_GDN_PAD_STATE, default ON): the runner RIGHT-pads the
        # prompt (real tokens first, then padding that repeats the last position), so
        # the sequence TAIL is padding. Seeding conv_state from the tail (qkv_t[...,-(k-1):])
        # carries PADDING into decode -> corrupted generation. HF masks padding before GDN
        # (apply_mask_to_padding_states) / left-pads linear attn. Fix: seed from the window
        # ENDING at the last REAL token (argmax(positions) -> last incrementing index), via
        # a dynamic-slice gather. Default ON; VLLM_GDN_PAD_STATE=0 reverts to tail-seed.
        if self.conv_state is not None:
            _pad_state = os.environ.get("VLLM_GDN_PAD_STATE", "1") == "1"
            if _pad_state and positions is not None:
                last_real = torch.argmax(positions.to(torch.int32))  # idx of last real token
                k1 = self.conv_kernel_size - 1
                # gather columns [last_real-k1+1 .. last_real] (clamped to [0, T-1]).
                # SEQ1024 OOB FIX (named op _gather/index_select hlo94 d26be9dd): the dynamic
                # torch.index_select(qkv_t[0], dim=1, index=cols) along the T-axis lowers to an
                # indirect vector-DGE that neuronx-cc emits in OOBMode.ERROR (it cannot prove
                # the runtime `cols` index stays in [0,T-1]) -> nrta-1006 at seq=1024 warmup.
                # Index-VALUE clamps NEVER remove an OOBMode.ERROR DGE (campaign lesson). CURE =
                # an INDEX-FREE one-hot @ matmul: build P[T,k1] with P[t,j]=1 iff t==cols[j], then
                # gathered = qkv_t[0] @ P -> [conv_dim_local, k1]. Pure matmul, NO indirect
                # addressing -> no DGE. Bit-identical to index_select (one nonzero per column).
                _seq_T = qkv_t.shape[2]
                cols = torch.clamp(
                    last_real - k1 + 1 + torch.arange(k1, device=qkv_t.device),
                    min=0, max=_seq_T - 1,
                )
                if os.environ.get("VLLM_GDN_CONVSEED_ONEHOT", "1") == "1":
                    # one-hot selection matrix P: [T, k1], P[t,j] = (t == cols[j])
                    _rows = torch.arange(_seq_T, device=qkv_t.device).unsqueeze(1)  # [T,1]
                    _P = (_rows == cols.unsqueeze(0)).to(qkv_t.dtype)               # [T,k1]
                    _gathered = qkv_t[0] @ _P                                       # [conv_dim_local, k1]
                    _seed_state(self.conv_state, _gathered.unsqueeze(0), getattr(self, '_conv_page_stride', 0),
                                getattr(self, '_conv_raw_slab', None), getattr(self, '_conv_raw_off', 0))
                else:
                    _seed_state(self.conv_state, torch.index_select(qkv_t[0], dim=1, index=cols).unsqueeze(0), getattr(self, '_conv_page_stride', 0),
                                getattr(self, '_conv_raw_slab', None), getattr(self, '_conv_raw_off', 0))
            else:
                _seed_state(self.conv_state, qkv_t[0, :, -(self.conv_kernel_size - 1):].unsqueeze(0), getattr(self, '_conv_page_stride', 0),
                            getattr(self, '_conv_raw_slab', None), getattr(self, '_conv_raw_off', 0))

        # Split into Q, K, V (per-rank channels)
        query = qkv_conv[..., :self.key_dim_local]
        key = qkv_conv[..., self.key_dim_local:self.key_dim_local * 2]
        value = qkv_conv[..., self.key_dim_local * 2:]

        # Reshape to per-rank heads
        query = query.reshape(batch_size, seq_len, self.num_k_heads_local, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads_local, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads_local, self.head_v_dim)

        # Compute gate: g = -exp(A_log) * softplus(a + dt_bias)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # PADDING-AWARE RECURRENT STATE (VLLM_GDN_PAD_STATE, default ON): zeroing the
        # hidden states does NOT make padding scan steps no-ops — beta=sigmoid(0)=0.5 and
        # g=-exp(A_log)*softplus(dt_bias)!=0, so each padding step still decays the carried
        # recurrent_state by exp(g) (geometric decay ~ #padding tokens) and applies a delta.
        # The carried state is taken AFTER the full padded T, so right-padding pollutes it.
        # HF avoids this by left-padding linear attn. Fix: make padding steps TRUE no-ops by
        # forcing g=0 (=> exp(0)=1, no decay) and beta=0 (=> delta=0, no state update) on
        # padding rows. q/k/v need no masking: beta=0 kills the update term. seq=positions T.
        if os.environ.get("VLLM_GDN_PAD_STATE", "1") == "1" and positions is not None:
            _last_real = torch.argmax(positions.to(torch.int32))
            _keep = (torch.arange(seq_len, device=g.device) <= _last_real)  # [T]
            _km = _keep.view(1, seq_len, 1).to(g.dtype)
            g = g * _km            # padding -> g=0 -> exp(0)=1 -> no decay
            beta = beta * _km.to(beta.dtype)  # padding -> beta=0 -> no state update

        # Repeat K heads to match V heads if needed (GQA ratio preserved per-rank)
        if self.num_v_heads_local // self.num_k_heads_local > 1:
            rep = self.num_v_heads_local // self.num_k_heads_local
            query = _repeat_interleave_heads(query, rep, dim=2)
            key = _repeat_interleave_heads(key, rep, dim=2)

        # CHUNKED gated delta rule for prefill: the outer loop unrolls only
        # seq_len/chunk_size steps (16 @ seq=1024, chunk=64) vs the sequential
        # scan's seq_len steps (1024) — a ~64x smaller unrolled graph per GDN
        # layer. The sequential scan made the single-shot prefill HLO a monolith
        # that hung neuronx-cc (see CHUNKED_GDN_EXPLORATION.md / T_08b70eee). Both
        # XLA-lowering hazards are now removed: the outer chunk write uses
        # list+stack, and the inner triangular inverse uses log-depth doubling
        # (no variable-width in-place slice). Decode still uses the recurrent rule.
        # DIAGNOSTIC (env-gated VLLM_GDN_CHUNK_SIZE): override the chunked-scan
        # chunk_size to bisect the device prefill OOB. chunk_size==seq_len degenerates
        # the multi-chunk loop to a SINGLE chunk (num_chunks=1) — if the OOB clears at
        # chunk_size>=seq the fault is the multi-chunk loop / state-carry indexing; if
        # it persists, the fault is the per-chunk reshape/cumsum/triangular-inverse.
        # Default 64 keeps the compiled graph identical (flag unset).
        # GDN prefill scan selection. DEFAULT = single-chunk (whole-sequence-as-one-chunk
        # on native [B,H,T,D], NO seq↔chunk reshape) — this is the FIX for the prefill
        # scatter/gather OOB (nrta 1006): the OOB was the chunked rule's axis-split reshape
        # lowering to an out-of-bound vector-DGE (proven by fxgraph diff e0cfe015 vs clean
        # aec963e7; ALL gather/scatter/MoE/KV ops were byte-identical, only the reshape
        # differed). Single-chunk is bit-exact to chunked@num_chunks==1 (CPU 8.5e-7 vs HF)
        # and emits no reshape. Routed for seq <= VLLM_GDN_SEQ_MAX (default 512);
        # longer seqs fall back to the chunked rule (the T×T inverse cost grows with T).
        # Escape hatch: VLLM_GDN_FORCE_CHUNKED=1 (force old chunked path).
        # GDN prefill scan selection. DEFAULT = SEQUENTIAL recurrent scan for short seq:
        # it is the ONLY device-CLEAN prefill at seq256 (T_09d7ed4d, 0 OOB) — the chunked
        # rule AND the single-chunk rewrite both still hit the vector-DGE OOB (nrta 1006).
        # The sequential scan's only drawback is a seq_len-deep unroll that hung neuronx-cc
        # compile at seq1024 (T_08b70eee); at seq<=VLLM_GDN_SEQ_MAX (default 512) it compiles
        # fine. This makes the model WORK at seq256 now; the chunked-family OOB (the long-seq
        # perf path) is tracked separately. Flags: VLLM_GDN_FORCE_CHUNKED=1 (old chunked),
        # VLLM_GDN_FORCE_SINGLECHUNK=1 (single-chunk rewrite), VLLM_GDN_CHUNK_SIZE.
        _gdn_chunk = int(os.environ.get("VLLM_GDN_CHUNK_SIZE", "64"))
        _seq_max = int(os.environ.get("VLLM_GDN_SEQ_MAX", "512"))
        # SEGMENTED-SEQUENTIAL (VLLM_GDN_SEGMENTED=1): the seq1024 "third option".
        # Problem: sequential scan is device-CLEAN (no OOB) but its T-deep unroll HANGS
        # neuronx-cc at seq1024 (monolith); chunked compiles but OOBs at every seq (the
        # OOB survives even the no-reshape SPLIT variant -> it's the chunked DISPATCH,
        # not the math). Fix: run the SAME clean sequential recurrence in fixed-size
        # SEGMENTS (default 256), carrying last_recurrent_state across segments. The
        # inner unroll is only seg steps (compiles like seq256, which is fine), repeated
        # over a short outer python loop (T/seg=4 @ 1024) -> breaks the monolith WITHOUT
        # touching the OOB-prone chunked path. Bit-exact to the flat sequential scan.
        if os.environ.get("VLLM_GDN_SEQ_NKI") == "1":
            # SEQUENTIAL-recurrence NKI kernel (segmented recurrence INSIDE the kernel):
            # bounded traced graph at any seq_len AND sequential-scan values (no chunked
            # reformulation drift -> MoE routing identical to the known-good path).
            # APC: pass the has_init-masked cache-hit seed (_apc_init, [1,H,Dk,Dv]) so a
            # reuse prefill seeds S from the saved prefix boundary state instead of zero.
            # None on fresh/non-APC requests -> plain zero-init kernel (byte-identical).
            core_attn_out, last_state = self._segmented_gated_delta_rule_nki(
                query, key, value, g, beta,
                output_final_state=(self.recurrent_state is not None),
                initial_state=_apc_init,
            )
        elif os.environ.get("VLLM_GDN_SEGMENTED") == "1":
            _seg = int(os.environ.get("VLLM_GDN_SEG_SIZE", "256"))
            _init = None
            if _apc_init is not None:  # T2 APC cache-hit seed (has_init-masked)
                _init = _apc_init
            core_attn_out, last_state = self._segmented_recurrent_gated_delta_rule(
                query, key, value, g, beta, seg_size=_seg, initial_state=_init,
            )
        elif os.environ.get("VLLM_GDN_FORCE_SINGLECHUNK") == "1":
            core_attn_out, last_state = self._single_chunk_gated_delta_rule(
                query, key, value, g, beta,
                output_final_state=(self.recurrent_state is not None),
            )
        elif os.environ.get("VLLM_GDN_FORCE_CHUNKED") == "1" or seq_len > _seq_max:
            core_attn_out, last_state = self._chunk_gated_delta_rule(
                query, key, value, g, beta,
                chunk_size=_gdn_chunk,
                output_final_state=(self.recurrent_state is not None),
            )
        else:
            # PREFILL starts a fresh sequence -> recurrent state MUST begin at zero.
            # DEFAULT = initial_state=None (scan allocates torch.zeros), EXACTLY matching
            # the last fully-COHERENT run (commit f35d2ae7b / T_4bf79b45, which hard-coded
            # initial_state=None). A later edit changed the default to read the
            # framework-bound self.recurrent_state (OPT-FULL bind_mamba_state); that tensor
            # is not guaranteed zero at prefill per-request, and the zero-state arm
            # (T_f3b87fb3) measurably REDUCED the error — so reading it was a regression.
            # T2 APC: cache-hit prefill seeds from the saved boundary state
            # (_apc_init, already has_init-masked). None -> initial_state=None.
            _init = None
            if _apc_init is not None:
                _init = _apc_init
            core_attn_out, last_state = self._recurrent_gated_delta_rule(
                query, key, value, g, beta,
                initial_state=_init,
            )

        # Update recurrent state (optB(C3): seed at page slot when slot-indexed, else row [0]).
        if self.recurrent_state is not None and last_state is not None:
            _seed_state(self.recurrent_state, last_state[:1], getattr(self, '_rec_page_stride', 0),
                        getattr(self, '_rec_raw_slab', None), getattr(self, '_rec_raw_off', 0))

        # Gated output norm: norm(attn_out) * silu(z)
        z_reshaped = z.reshape(batch_size * seq_len, self.num_v_heads_local, self.head_v_dim)
        out_reshaped = core_attn_out.reshape(batch_size * seq_len, self.num_v_heads_local, self.head_v_dim)

        normed_parts = []
        for h in range(self.num_v_heads_local):
            normed_parts.append(
                self.norm(out_reshaped[:, h, :], z_reshaped[:, h, :])
            )
        core_attn_out = torch.stack(normed_parts, dim=1)
        core_attn_out = core_attn_out.reshape(batch_size * seq_len, self.value_dim_local)

        # Output projection (row-parallel: per-rank partial sum over value heads).
        # SP exit: reduce_scatter sums the partials AND returns to SP-local (T/world).
        output = core_attn_out @ self.out_proj_weight
        output = output.squeeze(0) if batch_size == 1 else output
        if self.world_size > 1:
            output = self.tp_group.reduce_scatter(output, dim=0)
        return output

    def _decode_num_seqs(self, attn_metadata, tokens: int) -> int:
        """Number of SEQUENCES B in a decode step (with tokens == B * S_decode).

        SPEC-DECODE B1: multi-token GDN verify. During spec decode the target
        model VERIFIES N = 1 + num_speculative_tokens draft tokens per request,
        which the runner packs FLATTENED as ``[B * S_decode, H]`` (S_decode = N).
        To reshape those tokens back into per-sequence rows we need B. Read it
        from the linear_attn metadata's ``block_table_tensor`` (its row count ==
        number of active sequences) — the SAME source full-attention decode uses
        (forward_decode ~L572, ``B = block_table.shape[0]``).

        When metadata is absent (module tests) or carries no block_table (the
        Test-4/5 decode harness passes only ``state_indices``), fall back to
        ``B == tokens`` ⇒ S_decode == 1 — byte-identical to the historical
        single-token path (the common case MUST NOT change).
        """
        if attn_metadata is None:
            return tokens
        md = attn_metadata.get(f"layers.{self.layer_idx}.linear_attn")
        if md is None:
            md = attn_metadata.get(f"language_model.layers.{self.layer_idx}.linear_attn")
        if md is None:
            return tokens
        bt = md.get("block_table_tensor")
        if bt is None:
            return tokens
        B = int(bt.shape[0])
        # Guard: only accept a B that evenly partitions the token count. Anything
        # else (padding-lane skew, unexpected packing) reverts to S_decode == 1.
        if B <= 0 or tokens % B != 0:
            return tokens
        return B

    def forward_decode(self, hidden_states: torch.Tensor,
                       attn_metadata: object | None = None) -> torch.Tensor:
        """Recurrent delta rule for decode.

        Single-token decode (S_decode == 1, the common case) AND multi-token
        spec-decode VERIFY (S_decode == 1 + num_speculative_tokens draft tokens
        per request) share this one path: the per-token gated-delta recurrence is
        looped over the S_decode positions (in ``_recurrent_gated_delta_rule``,
        the same primitive as the sequential prefill kernel), carrying conv_state
        + recurrent_state across the draft tokens and emitting S_decode outputs so
        the rejection sampler can accept/reject each. When S_decode == 1 every
        reshape/window/commit below collapses to the original single-token form,
        so that path stays byte-identical.

        STATE COMMIT: we persist the running state AFTER all S_decode draft tokens
        (full window / final recurrent state). Rolling back to the accepted prefix
        (num_accepted from the rejection sampler) is handled by the mamba-APC
        postprocess / B2 block-copy — this phase only guarantees correct
        per-draft-token outputs and a correct running state.
        """
        # T2 APC (task #106): in-graph block->block carry-forward BEFORE the recurrent read, so
        # a request whose running block just moved reads its migrated state. No-op unless carry ids.
        self._carry_forward(attn_metadata)
        # hidden_states: [B, H] (S_decode==1) or [B*S_decode, H] (spec verify) or [B, S, H].
        if hidden_states.dim() == 2:
            tokens = hidden_states.shape[0]
            batch_size = self._decode_num_seqs(attn_metadata, tokens)
            seq_len = tokens // batch_size
            hidden_states_3d = hidden_states.view(batch_size, seq_len, -1)  # [B, S_decode, H]
        else:
            hidden_states_3d = hidden_states

        batch_size, seq_len, _ = hidden_states_3d.shape
        tokens = batch_size * seq_len  # total decode tokens (== B for the S_decode==1 common case)

        # State-cache slot routing (OPT-FULL hybrid-state lifecycle). The framework binds conv/
        # recurrent state sized for max_num_seqs (N_SLOTS); a MambaSpec layer gets 1 state row per
        # sequence (block_size=1) so the per-request state slot is the runner-supplied state_indices
        # (= mamba block_table[:,0]).
        #
        # OPTION 5 — gather/scatter as a ONE-HOT MATMUL (the only Neuron-viable non-contiguous form).
        # Every prior attempt that expressed the index as a *slice* failed the full-model warmup with
        # "non-contiguous slicing for Device Tensor": `state[idx]`, index_copy_(0,idx,..) (NCC_IVRF100
        # pad), and index_select+explicit-coord index_put_ (Option 1, T_4a5ddd35). neuronx-cc rejects
        # arbitrary-ROW slicing of a device tensor regardless of the operator. A gather is, however,
        # mathematically a permutation MATMUL — and matmul lowers (cf. deepseek_v32 einsum, MoE one-hot
        # routing). So:
        #   P = one_hot(state_idx, N_SLOTS)             # [B, N] dense, exact 0/1
        #   active = P @ state.reshape(N, -1)           # gather rows  (bit-EXACT: one nonzero term/row)
        #   state' = state - Pᵀ(P·state) + Pᵀ·new       # scatter: zero selected rows, write new
        #   state.copy_(state')                         # persist via FULL-tensor in-place copy (contig)
        # With exact 0/1 entries and (for distinct idx) one nonzero term per output row, NO rounding
        # occurs → identical to the contiguous slice write. When state_indices is absent (module test)
        # or N==B we still take this path; the arange one-hot == identity on [:B]. We persist with
        # copy_ (NOT attribute reassignment) because the bound state tensor is a compiled-graph input
        # threaded by in-place mutation. See STATE_INDICES_OPTIONS.md Option 5.
        # ★ DECODE STATE-SLOT INDEX — default = CONTIGUOUS arange(B), NOT block_table[:,0]. ★
        # The runner CONDENSES active sequences into state-buffer rows 0..B-1 before every forward
        # (neuron_model_runner.py condense), and prefill seeds conv_state[0]/recurrent_state[0] by
        # Decode reads/writes state at the request's page slot (== the prefill seed slot),
        # supplied by the runner as state_indices (block_table[:,0]). When metadata is absent
        # (CPU module tests) state_idx is None -> contiguous [:batch_size] fallback.
        state_idx = self._slot_index(attn_metadata, batch_size, hidden_states.device)
        _noncontig_decode = state_idx is not None

        # Project
        qkv = hidden_states_3d @ self.in_proj_qkv_weight  # [B, 1, conv_dim]
        z = hidden_states_3d @ self.in_proj_z_weight
        a = hidden_states_3d @ self.in_proj_a_weight
        b = hidden_states_3d @ self.in_proj_b_weight

        # Build the one-hot permutation P [B, N_SLOTS] once (shared by conv + recurrent updates).
        # P[i, state_idx[i]] = 1. With state_idx = arange(B) this is just [I_B | 0] (identity on the
        # first B rows) → bit-identical to the old contiguous [:batch_size] path.
        def _gather_rows(state, P):
            # state [N, d0, d1, ...] -> active [B, d0, d1, ...] via P [B, N] @ state_flat [N, D].
            N = state.shape[0]
            tail = state.shape[1:]
            flat = state.reshape(N, -1).to(P.dtype)       # [N, D]
            active = P @ flat                             # [B, D] (exact: one 1 per row)
            return active.reshape(batch_size, *tail).to(state.dtype)

        def _scatter_rows(state, new, P):
            # Return a full [N, ...] tensor with rows state_idx replaced by `new` [B, ...]:
            #   state - Pᵀ(P·state)   zeros the selected rows
            #   + Pᵀ·new              writes the new content there
            N = state.shape[0]
            tail = state.shape[1:]
            sflat = state.reshape(N, -1).to(P.dtype)      # [N, D]
            nflat = new.reshape(batch_size, -1).to(P.dtype)  # [B, D]
            Pt = P.transpose(0, 1)                        # [N, B]
            updated = sflat - Pt @ (P @ sflat) + Pt @ nflat
            return updated.reshape(N, *tail).to(state.dtype)

        # P in float32: 0/1 entries are exact in fp32, and fp32 matmul keeps the recurrent state
        # (fp32) lossless. The helpers cast each state to P.dtype then back, so conv (bf16) round-
        # trips bf16->fp32->bf16 exactly (one nonzero term per row => no accumulation rounding).
        # Slot-indexed (optB C4) OR legacy NONCONTIG -> use the one-hot matmul scatter at state_idx.
        # Default (both unset, state_idx None) -> contiguous [:batch_size] (proven, condense-invariant).
        _noncontig = _noncontig_decode
        P = None
        if _noncontig and (self.conv_state is not None or self.recurrent_state is not None):
            ref = self.conv_state if self.conv_state is not None else self.recurrent_state
            N_SLOTS = ref.shape[0]
            if state_idx is None:
                state_idx = torch.arange(batch_size, device=ref.device)
            # Build P by broadcast equality (pure elementwise + cast — avoids F.one_hot, whose
            # scatter lowering on Neuron is unproven): P[i,j] = 1.0 iff state_idx[i] == j.
            slots = torch.arange(N_SLOTS, device=ref.device)              # [N]
            P = (state_idx.view(batch_size, 1) == slots.view(1, N_SLOTS)).to(torch.float32)  # [B, N]

        # Slot-indexed NKI kernels for conv + recurrence (the unified-cache decode path).
        # The A1/A2 kernels are single-token by construction (one [B, ...] step in/out), so
        # multi-token spec-decode verify (seq_len > 1) routes through the one-hot recurrence
        # below instead (still addressing the right slot, looping the S_decode draft tokens).
        # seq_len == 1 (the common case) takes the kernel path. On CPU module tests state_idx
        # is None -> the contiguous path.
        _slot_kernel = (
            state_idx is not None
            and self.conv_state is not None
            and self.recurrent_state is not None
            and seq_len == 1
        )
        # UNIFIED-CACHE: when the GDN state pool is a PAGE-STRIDED view of the shared slab
        # (VLLM_UNIFIED_KV_GATHER=1), the slot NKI kernels (which address slot*head_elems on a
        # CONTIGUOUS pool) would mis-address it. Route through the one-hot _gather_rows/_scatter_rows
        # path instead — it is layout-agnostic (reshape(N,-1) forces a contiguous copy, then matmul;
        # no strided slice), the port's proven DGE-free form, so it lowers on the strided pool. The
        # compact 0-based kernels (gdn_*_update_compact) are a lighter future path (unit-tested) but
        # the one-hot path needs zero extra decode wiring, so use it for the first device proof.
        if os.environ.get("VLLM_UNIFIED_KV_GATHER") == "1":
            _slot_kernel = False

        # Conv1d update. DEFAULT = contiguous [:batch_size] slice (proven, condense-invariant, cheap).
        # qkv_t is [B, conv_dim_local, S_decode] (S_decode == 1 for plain decode). The depthwise
        # conv over cat([prior_window(k-1), qkv_t], dim=-1) emits S_decode outputs — one per draft
        # token, each convolving the correct trailing window — bit-exact to S_decode sequential
        # single-token conv steps. The persisted rolling window is the LAST (k-1) columns.
        _k1 = self.conv_kernel_size - 1
        qkv_t = qkv.transpose(1, 2)  # [B, conv_dim_local, S_decode]
        if _slot_kernel:
            # A2 kernel: gather conv_state[slot], depthwise conv + silu, scatter shifted window.
            # Returns y=[B, conv_dim_local] (post-silu) and the updated pool.
            y_conv, conv_new = _slot_conv_kernel()[2](
                self.conv_state, qkv_t.squeeze(-1), self.conv1d_weight,
                state_idx.to(torch.int32).reshape(batch_size, 1),
                batch_size, self.conv_dim_local, self.conv_kernel_size,
            )
            self.conv_state.copy_(conv_new)
            # gdn_conv_update returns fp32 y; cast back to the module dtype (bf16) to MATCH the
            # torch path (F.silu(F.conv1d(...)) stays bf16). Without this the fp32 leaks through the
            # Q/K/V split into the residual stream and downstream MoE matmul (dtype mismatch).
            qkv_conv = y_conv.reshape(batch_size, 1, self.conv_dim_local).to(self.dtype)  # [B, 1, conv_dim_local]
        elif self.conv_state is not None:
            # New rolling window = the LAST (k-1) columns of [prior_window | S_decode new tokens].
            # For S_decode==1 this is conv_input[:, :, 1:] (identical to the old code); for S_decode>1
            # it is the trailing (k-1) columns spanning the final draft tokens.
            if _noncontig:
                # conv_state CONTAINER layout is (k-1, conv_dim_local) — the SD backend shape — NOT
                # (conv_dim_local, k-1). _gather_rows reshapes to state.shape[1:] = the container
                # shape, so it returns [B, k-1, conv_dim_local]; the conv cat/conv1d below need the
                # SEMANTIC [B, conv_dim_local, k-1] (transpose the last two dims). Mirrors the seed
                # path's explicit semantic reshape (see _apc_conv_seed ~L1097). Byte-consistent: the
                # scatter transposes back to the container layout before writing. (This latent
                # transpose only bites the one-hot path, which pre-unified only ran for seq_len>1
                # spec-decode — untested — so the bug surfaced when the unified flag routed plain
                # single-token decode here.)
                N_cs = self.conv_state.shape[0]
                _rowe_c = self.conv_dim_local * _k1
                if _unified_kv_gather_on() and getattr(self, "_conv_raw_slab", None) is not None:
                    # UNIFIED: conv_state is a PAGE-STRIDED view -> reshape(N,-1)/one-hot would reject.
                    # Stage via .ap gather (strided read -> CONTIGUOUS [B, rowe]) then scatter back.
                    _sidx = _anchor_idx(state_idx.to(torch.int32).reshape(batch_size, 1), qkv_t)
                    _cs = self._conv_raw_slab
                    _gc = _paged_state_gather_kernel()[2](
                        _cs, _sidx, batch_size, _rowe_c,
                        int(_cs.shape[1]), int(self._conv_raw_off),
                    )  # [B, conv_dim_local*k1] contiguous, container(k1,conv_dim)-flat order
                    conv_active = _gc.reshape(batch_size, self.conv_dim_local, _k1)  # semantic reinterpret
                    conv_input = torch.cat([conv_active, qkv_t], dim=-1)
                    _new_flat = conv_input[:, :, -_k1:].reshape(batch_size, _rowe_c).to(_cs.dtype)
                    # Persist IN PLACE (copy_): keeps the bound slab threaded as a graph input;
                    # reassign would demote it to a CPU const -> HOP dispatches to _cpu_impl.
                    self._conv_raw_slab.copy_(_paged_state_scatter_kernel()[2](
                        _cs, _new_flat, _sidx, batch_size, _rowe_c,
                        int(_cs.shape[1]), int(self._conv_raw_off),
                    ))
                else:
                    cflat = self.conv_state.reshape(N_cs, -1)            # [N, conv_dim_local*k1] flat
                    _gathered_flat = (P @ cflat.to(P.dtype)).to(self.conv_state.dtype)  # [B, conv_dim_local*k1]
                    conv_active = _gathered_flat.reshape(batch_size, self.conv_dim_local, _k1)  # semantic
                    conv_input = torch.cat([conv_active, qkv_t], dim=-1)  # [B, conv_dim_local, k-1+S_decode]
                    _new_flat = conv_input[:, :, -_k1:].reshape(batch_size, -1).to(self.conv_state.dtype)
                    Pt = P.transpose(0, 1)
                    _updated = (cflat.to(P.dtype) - Pt @ (P @ cflat.to(P.dtype)) + Pt @ _new_flat.to(P.dtype))
                    self.conv_state.copy_(_updated.reshape(self.conv_state.shape).to(self.conv_state.dtype))
            else:
                conv_active = self.conv_state[:batch_size]           # [B, conv_dim_local, k-1]
                conv_input = torch.cat([conv_active, qkv_t], dim=-1)  # [B, conv_dim_local, k-1+S_decode]
                self.conv_state[:batch_size] = conv_input[:, :, -_k1:]
            qkv_conv = F.silu(F.conv1d(
                conv_input, self.conv1d_weight.unsqueeze(1), None, groups=self.conv_dim_local,
            )).transpose(1, 2)  # [B, S_decode, conv_dim_local]
        else:
            conv_input = F.pad(qkv_t, (self.conv_kernel_size - 1, 0))
            qkv_conv = F.silu(F.conv1d(
                conv_input, self.conv1d_weight.unsqueeze(1), None, groups=self.conv_dim_local,
            )).transpose(1, 2)  # [B, S_decode, conv_dim_local]

        # Split Q, K, V (per-rank channels). qkv_conv is [B, S_decode, conv_dim_local];
        # reshape to [B, S_decode, num_heads_local, head_dim] (the -1 = seq_len == S_decode).
        query = qkv_conv[..., :self.key_dim_local].reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = qkv_conv[..., self.key_dim_local:self.key_dim_local * 2].reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = qkv_conv[..., self.key_dim_local * 2:].reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads_local // self.num_k_heads_local > 1:
            rep = self.num_v_heads_local // self.num_k_heads_local
            query = _repeat_interleave_heads(query, rep, dim=2)
            key = _repeat_interleave_heads(key, rep, dim=2)

        # Recurrent update. DEFAULT = contiguous slice (see conv note); non-contig = one-hot matmul.
        if _slot_kernel:
            # A1 kernel: gather recurrent_state[slot,h], one gated-delta step, scatter back.
            # Host prologue (the torch scan does these INSIDE _recurrent_gated_delta_rule):
            #   l2norm(q,k,eps=1e-6); q *= 1/sqrt(kd); g = exp(g_raw); beta already sigmoid'd.
            # query/key/value are [B,1,H,kd] (seq=1) -> squeeze the seq dim to [B,H,kd].
            _H = self.num_v_heads_local
            q_bh = l2norm(query, dim=-1, eps=1e-6).reshape(batch_size, _H, self.head_k_dim)
            q_bh = (q_bh.float() * (self.head_k_dim ** -0.5))
            k_bh = l2norm(key, dim=-1, eps=1e-6).reshape(batch_size, _H, self.head_k_dim).float()
            v_bh = value.reshape(batch_size, _H, self.head_v_dim).float()
            g_bh = g.reshape(batch_size, _H).float().exp()
            beta_bh = beta.reshape(batch_size, _H).float()
            y_state, rec_new = _slot_state_kernel()[2](
                self.recurrent_state, q_bh, k_bh, v_bh, g_bh, beta_bh,
                state_idx.to(torch.int32).reshape(batch_size, 1),
                batch_size, _H, self.head_k_dim, self.head_v_dim,
            )
            self.recurrent_state.copy_(rec_new)
            core_attn_out = y_state.reshape(batch_size, _H, self.head_v_dim).to(query.dtype)
        else:
            _rec_unified = _noncontig and _unified_kv_gather_on() and self.recurrent_state is not None
            _rowe_r = None
            if self.recurrent_state is None:
                rec_init = None
            elif _rec_unified:
                # UNIFIED: recurrent_state PAGE-STRIDED -> stage via .ap gather (strided read ->
                # contiguous [B, rowe]) instead of one-hot reshape (which rejects on strided view).
                _rowe_r = 1
                for _d in self.recurrent_state.shape[1:]:
                    _rowe_r *= _d
                _sidx_r = _anchor_idx(state_idx.to(torch.int32).reshape(batch_size, 1), query)
                _rs = self._rec_raw_slab
                _gr = _paged_state_gather_kernel()[2](
                    _rs, _sidx_r, batch_size, _rowe_r,
                    int(_rs.shape[1]), int(self._rec_raw_off),
                )
                rec_init = _gr.reshape(batch_size, *self.recurrent_state.shape[1:]).to(self.recurrent_state.dtype)
            elif _noncontig:
                rec_init = _gather_rows(self.recurrent_state, P)
            else:
                rec_init = self.recurrent_state[:batch_size]
            core_attn_out, last_state = self._recurrent_gated_delta_rule(
                query, key, value, g, beta,
                initial_state=rec_init,
            )

            if self.recurrent_state is not None and last_state is not None:
                if _rec_unified:
                    _sidx_r = _anchor_idx(state_idx.to(torch.int32).reshape(batch_size, 1), last_state)
                    _rs = self._rec_raw_slab
                    _new_r = last_state.to(_rs.dtype).reshape(batch_size, _rowe_r)
                    # Persist IN PLACE (copy_): keeps the bound slab threaded as a graph input;
                    # reassign would demote it to a CPU const -> HOP dispatches to _cpu_impl.
                    self._rec_raw_slab.copy_(_paged_state_scatter_kernel()[2](
                        _rs, _new_r, _sidx_r, batch_size, _rowe_r,
                        int(_rs.shape[1]), int(self._rec_raw_off),
                    ))
                elif _noncontig:
                    self.recurrent_state.copy_(
                        _scatter_rows(self.recurrent_state, last_state.to(self.recurrent_state.dtype), P))
                else:
                    self.recurrent_state[:batch_size] = last_state.to(self.recurrent_state.dtype)

        # Gated norm + output proj (per-rank heads). Flatten to [tokens, H, vd] where
        # tokens == B * S_decode, keeping the (b, s) row-major order the flat input was
        # packed in — so the S_decode per-request draft-token outputs stay adjacent and
        # aligned with the rejection sampler's per-request slice. For S_decode==1,
        # tokens == B and this is byte-identical to the original [B, H, vd] reshape.
        z_reshaped = z.reshape(tokens, self.num_v_heads_local, self.head_v_dim)
        out_reshaped = core_attn_out.reshape(tokens, self.num_v_heads_local, self.head_v_dim)

        normed_parts = []
        for h in range(self.num_v_heads_local):
            normed_parts.append(self.norm(out_reshaped[:, h, :], z_reshaped[:, h, :]))
        core_attn_out = torch.stack(normed_parts, dim=1)
        core_attn_out = core_attn_out.reshape(tokens, self.value_dim_local)

        # Output projection (row-parallel: partial sum -> all_reduce across TP)
        output = core_attn_out @ self.out_proj_weight
        if self.world_size > 1:
            self.tp_group.all_reduce(output)
        return output

    def _chunk_gated_delta_rule(
        self, query, key, value, g, beta, chunk_size=64, output_final_state=False,
    ):
        """Pure-torch chunked gated delta rule."""
        # FIX 2b (VLLM_GDN_NKI=1): route the chunked scan through a hand-written NKI kernel
        # (functional/gated_delta_rule.py). The torch chunked scan's seq>=1024 OOB is structural
        # to how neuronx-cc lowers the partition-axis chunk reshape to a vector-DGE indirect copy
        # (nrta 1006) — op-bisect proved no torch reformulation clears it. The NKI kernel uses ONLY
        # static-offset slices (no gather/DGE) so it sidesteps the OOB by construction. Default-off
        # (baseline torch graph byte-identical); flag-gated A/B + device validation ladder.
        if os.environ.get("VLLM_GDN_NKI") == "1":
            return self._chunk_gated_delta_rule_nki(
                query, key, value, g, beta, chunk_size, output_final_state
            )
        # DIAGNOSTIC (env VLLM_GDN_SPLIT=1): route through a variant that NEVER builds
        # the 5-D [B,H,nchunk,chunk,D] tensor — it splits the seq axis into a python
        # list of 4-D [B,H,chunk,D] chunk tensors (torch.split) and indexes the list,
        # eliminating both the `reshape(...,-1,chunk_size,...)` seq->（nchunk,chunk)
        # axis-split AND the 5-D `x[:, :, i]` chunk-axis select. Those are the only
        # chunked-only ops remaining after cumsum / triinv-doubling / tril-triu-
        # masked_fill / chunk_size were all device-exonerated; a reshape that splits a
        # partition-mapped axis (or a static select on a non-trailing 5-D axis) can
        # lower under neuronx-cc to a vector-DGE indirect copy. CPU-verified allclose.
        if os.environ.get("VLLM_GDN_SPLIT") == "1":
            return self._chunk_gated_delta_rule_split(
                query, key, value, g, beta, chunk_size, output_final_state
            )
        # FIX (VLLM_GDN_SINGLECHUNK=1, default-on path candidate): the chunked rule's
        # OOB is the seq↔(num_chunks,chunk_size) axis-split reshape, which splits a
        # PARTITION-MAPPED axis and lowers to a vector-DGE indirect copy whose
        # per-partition offset goes out-of-bounds (nrta 1006). Proven by fxgraph diff:
        # chunked graph e0cfe015 differs from clean sequential aec963e7 ONLY by this
        # reshape; ALL gather/scatter/MoE/KV ops are byte-identical. Even chunk_size==T
        # (num_chunks==1, an unsqueeze-style reshape) still OOB'd (T_2f75259c) — so the
        # reshape must be DELETED, not made trivial. This path treats the WHOLE sequence
        # as one chunk on the native 4-D [B,H,T,D] layout (no chunk axis), with the
        # trivial single-chunk recurrence (initial_state=0). Bit-exact to the chunked
        # rule at num_chunks==1 (CPU-verified). Use for buckets where T <= one chunk.
        if os.environ.get("VLLM_GDN_SINGLECHUNK") == "1":
            return self._single_chunk_gated_delta_rule(
                query, key, value, g, beta, output_final_state
            )
        initial_dtype = query.dtype
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)

        def _to_chunks_5d(x):  # x: [B,H,T,D] -> [B,H,nchunk,chunk,D]
            d = x.shape[-1]
            return x.reshape(x.shape[0], x.shape[1], -1, chunk_size, d)

        def _to_chunks_4d(x):  # g: [B,H,T] -> [B,H,nchunk,chunk]
            return x.reshape(x.shape[0], x.shape[1], -1, chunk_size)

        query, key, value, k_beta, v_beta = [
            _to_chunks_5d(x) for x in (query, key, value, k_beta, v_beta)
        ]
        g = _to_chunks_4d(g)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

        # Invert the unit lower-triangular (I - M) via log-depth DOUBLING instead of
        # the HF forward-substitution loop. The forward-sub writes a VARIABLE-WIDTH
        # slice in place (`attn[..., i, :i] = ...`), which does NOT lower under
        # neuronx-cc (dynamic-shape xla::unselect) — the reason prefill previously
        # fell back to the seq-length sequential scan. Here `attn` (= M) is strictly
        # lower-triangular hence NILPOTENT (M^chunk_size = 0), so
        #   (I - M)^-1 = I + M + M^2 + ... + M^(chunk_size-1)
        # is a finite sum. Accumulate it with ceil(log2(chunk_size)) full-matrix
        # matmuls (S_{2k}=S_k+M^k@S_k, P_{2k}=P_k@P_k) — all STATIC-SHAPE, no indexed
        # writes. Mathematically identical to the forward substitution; gated by the
        # three-way-vs-HF module test (test_gated_deltanet.py).
        eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        inv = eye                       # S_1 = I  (sum_{j=0}^{0} M^j)
        m_pow = attn                    # P_1 = M
        for _ in range((chunk_size - 1).bit_length()):   # ceil(log2(chunk_size))
            inv = inv + m_pow @ inv
            m_pow = m_pow @ m_pow
        attn = inv                      # (I - M)^-1 = I + M + ... + M^(chunk_size-1)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        last_recurrent_state = torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim, device=query.device, dtype=torch.float32
        )
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

        # Accumulate per-chunk outputs in a list + torch.stack (XLA-safe). An
        # in-place indexed write `core_attn_out[:, :, i] = ...` into the 5-D
        # [B, H, num_chunks, chunk, v] tensor fails neuronx-cc lowering
        # (xla::unselect + edge_padding). Matches the proven qwen3_5_moe scan.
        # FIX 2a (default-on): use the explicit last index (chunk_size-1) instead of a literal -1.
        # neuronx-cc >=2.12.54 runtime-validates indirect-gather index VALUES; a literal -1 can lower
        # to an OOB gather (no Python wrap-to-last). Bit-exact: tensor[...,-1] == tensor[...,chunk_size-1]
        # for a chunk-axis of length chunk_size.
        _last = chunk_size - 1
        chunk_outs = []
        for i in range(total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn_chunk = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill(mask, 0)
            v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            chunk_outs.append(attn_inter + attn_chunk @ v_new)
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, _last, None, None].exp()
                + (k_i * (g[:, :, i, _last, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )
        core_attn_out = torch.stack(chunk_outs, dim=2)  # [B, H, num_chunks, chunk, v]

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, core_attn_out.shape[-1])
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    def _chunk_gated_delta_rule_nki(
        self, query, key, value, g, beta, chunk_size, output_final_state,
    ):
        """FIX 2b: host wrapper around the NKI gdn_chunk_prefill_kernel (no partition-axis reshape
        -> no vector-DGE OOB at seq>=1024). Prologue (transpose/pad/scale/k_beta/v_beta + mask tiles)
        stays in torch (device-exonerated ops); the chunked scan runs in the kernel. Inputs arrive
        [B,H,T,D]; prefill B=1. Flag-gated VLLM_GDN_NKI; this is the FIRST device-validation cut."""
        import torch.nn.functional as F
        from vllm_neuron.functional.gated_delta_rule import run_gdn_chunk_prefill
        # C (chunk_size) is mapped to the NKI partition axis inside the kernel, so it
        # MUST be <= 128 (NKI-writing skill: "Partition dimension (P) <= 128"). The
        # T_b18d149e failure was C=256=seq_len reaching the kernel's
        # kernel_assert(C<=128). This happens when VLLM_GDN_CHUNK_SIZE fails to
        # propagate to forked MPExecutor workers (they inherit the default at fork
        # time and never see a post-fork monkeypatch), so the requested chunk can
        # degenerate to the full padded sequence. Clamp here so the kernel always
        # receives a partition-legal chunk; the kernel iterates N = T_pad // C chunks
        # carrying the recurrent state, so a smaller C only raises N (bounded outer
        # trip count) and never changes the traced graph shape.
        C = int(chunk_size)
        if C < 1:
            C = 1
        if C > 128:
            C = 128
        # --- prologue (mirror _chunk_gated_delta_rule lines ~1264-1283) ---
        # BUGFIX: the NKI path (VLLM_GDN_NKI early-return at ~1244) skips the torch fall-through
        # where l2norm(q)/l2norm(k) is applied (lines ~1276-1277). Without it, q/k enter the kernel
        # UN-normalized -> every q@k^T score, the k_beta@k^T Gram matrix, the (I-M)^-1 inverse, and
        # the state update scale wrong -> finite-but-garbage output -> corrupts MoE routing -> the
        # downstream out-of-bounds gather. The kernel docstring already ASSUMES q/k are l2norm'd.
        # Mirror the reference exactly: normalize BEFORE the transpose/cast.
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]
        B, Hh, T, D = key.shape
        Dv = value.shape[-1]
        pad = (C - T % C) % C
        query = F.pad(query, (0, 0, 0, pad)); key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad)); beta = F.pad(beta, (0, pad)); g = F.pad(g, (0, pad))
        Tp = T + pad
        query = query * (1.0 / (D ** 0.5))
        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        # head-major [H,Tp,*] (B==1 in prefill): drop batch, contiguous for static-offset DMA.
        sq = lambda x: x[0].contiguous()
        q_h, k_h, v_h = sq(query), sq(key), sq(value)
        kb_h, vb_h = sq(k_beta), sq(v_beta)
        g_h = g[0].contiguous()                         # [H, Tp]
        # precomputed [C,C] mask tiles (built once, host)
        dev = q_h.device
        eye = torch.eye(C, dtype=torch.float32, device=dev)
        tril_incl = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev))
        strict_lower = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev), diagonal=-1)
        out_h, state_h = run_gdn_chunk_prefill(
            q_h, k_h, v_h, kb_h, vb_h, g_h, eye, tril_incl, strict_lower,
        )
        # Step 11: slice pad, restore [B,T,H,Dv] layout + cast to input dtype (match torch path).
        initial_dtype = value.dtype if value.dtype != torch.float32 else self.dtype
        core_attn_out = out_h[:, :T, :].unsqueeze(0).transpose(1, 2).contiguous().to(initial_dtype)
        last_recurrent_state = state_h.unsqueeze(0) if output_final_state else None
        return core_attn_out, last_recurrent_state

    def _segmented_gated_delta_rule_nki(
        self, query, key, value, g, beta, output_final_state=False, initial_state=None,
    ):
        """VLLM_GDN_SEQ_NKI=1: host wrapper around the SEQUENTIAL (recurrent) GDN NKI
        prefill kernel (gated_delta_rule_seq.run_gdn_seq_prefill). Reproduces the
        device-clean sequential recurrence as ONE bounded kernel (hardware
        sequential_range over T, NO python T-deep unroll -> no neuronx-cc monolith
        hang at seq>=1024, and no chunked-reformulation value drift perturbing MoE
        routing). Prologue mirrors _chunk_gated_delta_rule_nki exactly (l2norm q/k,
        head-major fp32 [H,T,D], q pre-scaled by 1/sqrt(D)); B==1 prefill.

        APC: when initial_state is provided (the has_init-masked cache-hit seed from
        _apc_prefill_seed, shape [1,H,Dk,Dv]) the kernel seeds its recurrent state S
        from it instead of zero (root-cause fix — the SEQ_NKI kernel previously always
        memset S=0, discarding the reuse seed). Simulator-validated bit-exact to a
        recurrent scan seeded from that state (microbench_gdn_seq_init [B]/[C])."""
        from vllm_neuron.functional.gated_delta_rule_seq import run_gdn_seq_prefill
        # --- prologue (mirror _chunk_gated_delta_rule_nki verbatim, minus chunk pad) ---
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]
        B, Hh, T, D = key.shape
        query = query * (1.0 / (D ** 0.5))
        # head-major [H,T,*] (B==1 in prefill): drop batch, contiguous for static-offset DMA.
        sq = lambda x: x[0].contiguous()
        q_h, k_h, v_h = sq(query), sq(key), sq(value)
        g_h = g[0].contiguous()                         # [H, T]
        beta_h = beta[0].contiguous()                   # [H, T]
        # APC seed: [1,H,Dk,Dv] -> [H,Dk,Dv] fp32, matching the kernel's state layout
        # (== the state_h it returns below). None -> plain zero-init graph (byte-identical).
        init_state_h = None
        if initial_state is not None:
            init_state_h = initial_state[0].contiguous().to(torch.float32)
        out_h, state_h = run_gdn_seq_prefill(q_h, k_h, v_h, g_h, beta_h, init_state_h=init_state_h)
        # restore [B,T,H,Dv] layout + cast to input dtype (match _chunk_gated_delta_rule_nki).
        initial_dtype = value.dtype if value.dtype != torch.float32 else self.dtype
        core_attn_out = out_h[:, :T, :].unsqueeze(0).transpose(1, 2).contiguous().to(initial_dtype)
        last_recurrent_state = state_h.unsqueeze(0) if output_final_state else None
        return core_attn_out, last_recurrent_state

    def _single_chunk_gated_delta_rule(
        self, query, key, value, g, beta, output_final_state,
    ):
        """Whole-sequence-as-one-chunk gated delta rule on the NATIVE [B,H,T,D] layout.

        Identical math to _chunk_gated_delta_rule with num_chunks==1, but emits NO
        seq↔(num_chunks,chunk_size) axis-split reshape (the op that lowers to the
        out-of-bound vector-DGE on device). T plays the role of chunk_size; the
        inter-chunk recurrence is trivial (single chunk, initial state 0), so the
        per-chunk-loop carry terms vanish and the result is just the intra-chunk
        attention. Bit-exact to the chunked rule at num_chunks==1.
        """
        initial_dtype = query.dtype
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]
        batch_size, num_heads, T, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)            # [B,H,T,Dv]
        k_beta = key * beta.unsqueeze(-1)              # [B,H,T,Dk]

        g = g.cumsum(dim=-1)                           # [B,H,T] prefix decay
        mask_incl = torch.triu(torch.ones(T, T, dtype=torch.bool, device=query.device), diagonal=0)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()  # [B,H,T,T]
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_incl, 0)

        # (I - M)^-1 via nilpotent doubling (M strictly lower-tri, M^T==0).
        eye = torch.eye(T, dtype=attn.dtype, device=attn.device)
        inv = eye
        m_pow = attn
        for _ in range((T - 1).bit_length()):
            inv = inv + m_pow @ inv
            m_pow = m_pow @ m_pow
        attn = inv
        value = attn @ v_beta                          # u_i
        # single chunk: last_recurrent_state starts at 0, so v_prime / attn_inter terms drop.
        mask_strict = torch.triu(torch.ones(T, T, dtype=torch.bool, device=query.device), diagonal=1)
        attn_chunk = (query @ key.transpose(-1, -2) * decay_mask).masked_fill(mask_strict, 0)  # [B,H,T,T]
        core_attn_out = attn_chunk @ value             # [B,H,T,Dv]  (attn_inter=0 since state=0)

        last_recurrent_state = None
        if output_final_state:
            # carry state after the single chunk = sum_t k_t * (g_T - g_t).exp() outer v_new_t
            g_last = g[:, :, T - 1]                     # [B,H]
            decay_to_end = (g_last.unsqueeze(-1) - g).exp()  # [B,H,T]
            last_recurrent_state = (
                (key * decay_to_end[..., None]).transpose(-1, -2) @ value
            )                                          # [B,H,Dk,Dv]

        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)  # [B,T,H,Dv]
        return core_attn_out, last_recurrent_state

    def _chunk_gated_delta_rule_split(
        self, query, key, value, g, beta, chunk_size, output_final_state,
    ):
        """VLLM_GDN_SPLIT variant: list-of-4D-chunks layout (no 5-D reshape / select).

        Mathematically identical to _chunk_gated_delta_rule but never materialises the
        [B,H,nchunk,chunk,D] tensor. Each per-chunk quantity is a 4-D [B,H,chunk,D]
        element of a python list produced by torch.split; the loop indexes the list
        (a compile-time constant), so there is no static select on a 5-D non-trailing
        axis and no seq->(nchunk,chunk) axis-split reshape. CPU-verified allclose
        (max 0.0) vs the default path at chunk_size 64 and 256.
        """
        initial_dtype = query.dtype
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]
        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        num_chunks = total_sequence_length // chunk_size
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale
        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)

        # split the seq axis into lists of 4-D [B,H,chunk,D] chunk tensors
        def _split(x):
            return list(torch.split(x, chunk_size, dim=2))

        q_l, k_l, v_l, kb_l, vb_l = (
            _split(query), _split(key), _split(value), _split(k_beta), _split(v_beta),
        )
        g_l = [gc.cumsum(dim=-1) for gc in torch.split(g, chunk_size, dim=2)]

        mask0 = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=0,
        )
        mask1 = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=1,
        )
        eye = torch.eye(chunk_size, dtype=torch.float32, device=query.device)

        decay_l, val_l, kcd_l = [], [], []
        for i in range(num_chunks):
            gc = g_l[i]
            decay_mask = ((gc.unsqueeze(-1) - gc.unsqueeze(-2)).tril().exp().float()).tril()
            attn = -((kb_l[i] @ k_l[i].transpose(-1, -2)) * decay_mask).masked_fill(mask0, 0)
            inv = eye
            m_pow = attn
            for _ in range((chunk_size - 1).bit_length()):
                inv = inv + m_pow @ inv
                m_pow = m_pow @ m_pow
            attn = inv
            decay_l.append(decay_mask)
            val_l.append(attn @ vb_l[i])
            kcd_l.append(attn @ (kb_l[i] * gc.exp().unsqueeze(-1)))

        last_recurrent_state = torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim, device=query.device, dtype=torch.float32
        )
        chunk_outs = []
        for i in range(num_chunks):
            q_i, k_i, v_i = q_l[i], k_l[i], val_l[i]
            gc = g_l[i]
            attn_chunk = (q_i @ k_i.transpose(-1, -2) * decay_l[i]).masked_fill(mask1, 0)
            v_prime = kcd_l[i] @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * gc[..., None].exp()) @ last_recurrent_state
            chunk_outs.append(attn_inter + attn_chunk @ v_new)
            last_recurrent_state = (
                last_recurrent_state * gc[:, :, -1, None, None].exp()
                + (k_i * (gc[:, :, -1, None] - gc).exp()[..., None]).transpose(-1, -2) @ v_new
            )

        if not output_final_state:
            last_recurrent_state = None
        # concat chunk outputs along seq (list -> [B,H,T_pad,v]); slice off pad
        core_attn_out = torch.cat(chunk_outs, dim=2)
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    def _recurrent_gated_delta_rule(
        self, query, key, value, g, beta, initial_state=None,
    ):
        """Pure-torch recurrent gated delta rule for decode."""
        initial_dtype = query.dtype
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=torch.float32)
            if initial_state is None
            else initial_state.to(torch.float32)
        )

        # Accumulate per-step outputs in a list + torch.stack (XLA-safe; an
        # in-place core_attn_out[:, :, i] = ... write fails neuronx-cc lowering).
        outs = []
        for i in range(sequence_length):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)

            last_recurrent_state = last_recurrent_state * g_t
            kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            outs.append((last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2))

        core_attn_out = torch.stack(outs, dim=2)  # [B, H, T, v_head_dim]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    def _segmented_recurrent_gated_delta_rule(
        self, query, key, value, g, beta, seg_size=256, initial_state=None,
    ):
        """Segmented sequential scan: the seq1024 'third option' (VLLM_GDN_SEGMENTED).

        Runs the SAME clean per-step recurrence as _recurrent_gated_delta_rule, but in
        fixed-size SEGMENTS along the sequence, carrying last_recurrent_state across
        segments. The inner unroll is only seg_size steps (compiles like seq256), and
        the outer python loop is ceil(T/seg_size) (e.g. 4 @ T=1024) -> avoids the flat
        T-deep monolith that hangs neuronx-cc, WITHOUT touching the OOB-prone chunked
        dispatch. Bit-exact to the flat sequential scan (same math, same state carry).
        NOTE: stays on the SEQUENTIAL code path (proven device-clean at seq256); the OOB
        survives the chunked SPLIT variant, so it is chunked-dispatch-specific -> avoided here.
        """
        initial_dtype = query.dtype
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]
        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim,
                        device=value.device, dtype=torch.float32)
            if initial_state is None else initial_state.to(torch.float32)
        )

        # Outer loop over fixed segments; inner per-step recurrence (seg_size deep).
        num_seg = (sequence_length + seg_size - 1) // seg_size
        outs = []
        for s in range(num_seg):
            lo = s * seg_size
            hi = min(lo + seg_size, sequence_length)
            for i in range(lo, hi):
                q_t = query[:, :, i]
                k_t = key[:, :, i]
                v_t = value[:, :, i]
                g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
                beta_t = beta[:, :, i].unsqueeze(-1)
                last_recurrent_state = last_recurrent_state * g_t
                kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
                delta = (v_t - kv_mem) * beta_t
                last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
                outs.append((last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2))

        core_attn_out = torch.stack(outs, dim=2)  # [B, H, T, v_head_dim]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state


# =============================================================================
# Section 5: MLP (SwiGLU)
# =============================================================================


class Qwen3_5MoePlainRMSNorm(nn.Module):
    """Plain RMSNorm (no in-forward +1). The HF (1+weight) is folded into the
    stored weight at LOAD time via a gamma loader (see _setup_weight_loaders).
    This keeps the MoE block's norm numerically equal to the dense
    Qwen3_5RMSNorm (which folds +1 in forward) while letting NF.moe_block_tkg
    consume the same gamma. Do NOT reuse Qwen3_5RMSNorm here or +1 is applied
    twice (load-fold + forward-fold)."""

    def __init__(self, size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """256-expert top-8 sparse MoE + sigmoid-gated shared expert, static-block
    NF kernels with TP/EP. Ported self-contained from qwen3_5_moe/model_bf16.py.

    The post_attention_layernorm is FOLDED into this block (gamma for the decode
    kernel; explicit torch norm for prefill) — so the decoder layer must NOT
    pre-normalize (the MoE applies the norm exactly once). Signature is
    forward(hidden_states, positions, is_decode, rank) to match the decoder
    wiring added in this rebase.
    """

    def __init__(self, c):
        super().__init__()
        from vllm.config import get_current_vllm_config
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_ep_degree,
            get_neuron_ep_rank,
            get_neuron_ep_tp_group,
        )

        self.tp_group = get_tp_group()
        vllm_config = get_current_vllm_config()
        self.ep_enabled = vllm_config.parallel_config.enable_expert_parallel

        self.ep_degree = get_neuron_ep_degree() if self.ep_enabled else 1
        self.ep_rank = get_neuron_ep_rank() if self.ep_enabled else 0
        self.tp_degree = self.tp_group.world_size // self.ep_degree
        self.ep_tp_group = (
            get_neuron_ep_tp_group() if self.ep_enabled else self.tp_group
        )
        self.moe_group = self.tp_group

        self.total_num_experts = c.num_experts
        self.num_local_experts = c.num_experts // self.ep_degree
        self.top_k = c.num_experts_per_tok
        self.norm_topk_prob = c.norm_topk_prob
        self.hidden = c.hidden_size
        self.rms_norm_eps = c.rms_norm_eps
        self.intermediate_size_per_rank = c.moe_intermediate_size // self.tp_degree
        self.shared_inter = c.shared_expert_intermediate_size
        self.block_size = 256
        self.act = F.silu
        dt = c.torch_dtype
        self.dtype = dt  # module dtype (bf16); used by forward_decode to cast hidden_states before MoE

        # Pre-MLP RMSNorm (HF (1+weight) form; +1 folded at load via gamma loader).
        self.post_attention_layernorm = Qwen3_5MoePlainRMSNorm(
            self.hidden, self.rms_norm_eps, dt
        )

        self.router_weight = nn.Parameter(
            torch.empty(self.total_num_experts, self.hidden, dtype=dt))
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(self.num_local_experts, self.hidden,
                        self.intermediate_size_per_rank * 2, dtype=dt))
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.num_local_experts, self.intermediate_size_per_rank,
                        self.hidden, dtype=dt))
        self.shared_gate_proj = nn.Parameter(torch.empty(self.shared_inter, self.hidden, dtype=dt))
        self.shared_up_proj = nn.Parameter(torch.empty(self.shared_inter, self.hidden, dtype=dt))
        self.shared_down_proj = nn.Parameter(torch.empty(self.hidden, self.shared_inter, dtype=dt))
        self.shared_expert_gate = nn.Parameter(torch.empty(1, self.hidden, dtype=dt))

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        # Fold HF (1+weight) into gamma so the plain RMSNorm / kernel matches.
        set_weight_loader(
            self.post_attention_layernorm.weight,
            SafetensorsWeightLoader(transform=lambda s, r: s[0][:] + 1.0),
        )

        local_expert_indices = list(
            range(self.ep_rank * self.num_local_experts,
                  (self.ep_rank + 1) * self.num_local_experts))

        def _maybe_ep_wrap(loader):
            if self.ep_degree > 1:
                return expert_parallel_weight_loader(local_expert_indices, loader)
            return loader

        set_weight_loader(
            self.gate_up_proj_weight,
            _maybe_ep_wrap(
                expert_gate_up_weight_loader(
                    num_experts=self.total_num_experts,
                    shard_size=self.intermediate_size_per_rank * 2,
                    num_shards=self.tp_degree,
                )
            ),
        )
        set_weight_loader(
            self.down_proj_weight,
            _maybe_ep_wrap(
                expert_down_weight_loader(
                    num_experts=self.total_num_experts,
                    shard_size=self.intermediate_size_per_rank,
                    num_shards=self.tp_degree,
                )
            ),
        )

    def _shared_expert(self, x):
        gate = self.act(x @ self.shared_gate_proj.T) * (x @ self.shared_up_proj.T)
        shared = gate @ self.shared_down_proj.T
        return torch.sigmoid(x @ self.shared_expert_gate.T) * shared

    def forward(self, hidden_states, positions, is_decode, rank):
        if is_decode:
            return self.forward_decode(hidden_states, rank)
        return self.forward_prefill(hidden_states, positions, rank)

    def forward_decode(self, hidden_states, rank):
        # DTYPE FIX (matches gpt_oss staging model_bf16.py:822): the decode MoE kernel
        # (moe_block_tkg) feeds `inp` as the MOVING operand of the gate/up nc_matmul against
        # bf16 expert weights (stationary). If the residual-stream hidden_states arrives as fp32
        # (from an fp32 residual add), the matmul fails "stationary=bfloat16, moving=float32"
        # (NKI nc_matmul requires both operands same precision class). Cast to the block dtype
        # (bf16) so `inp` matches the bf16 weights — the shipping large-seq models all do this.
        hidden_states = hidden_states.to(self.dtype)
        if self.ep_degree > 1:
            ep_rank = (rank % (self.ep_degree * self.tp_degree)) // self.tp_degree
        else:
            ep_rank = torch.zeros_like(rank)
        rank_id = ep_rank.view(1, 1).to(torch.int32)

        normed = self.post_attention_layernorm(hidden_states)
        shared = self._shared_expert(normed)

        # fp32 router: router_topk asserts x.dtype == w.dtype (router_topk.py:279).
        # router_mm_dtype casts ONLY the router input x to fp32; the router weights w
        # are NOT cast by the kernel, so w must be supplied in fp32 too or the ROUTER
        # matmul fails "x float32 must match w bfloat16". Expert weights/input stay bf16.
        _router_fp32 = os.environ.get("VLLM_MOE_TKG_ROUTER_FP32") == "1"

        output = NF.moe_block_tkg(
            inp=hidden_states.unsqueeze(0),
            gamma=self.post_attention_layernorm.weight.unsqueeze(0).to(torch.float32),
            router_weights=(self.router_weight.T.to(torch.float32)
                            if _router_fp32 else self.router_weight.T),
            expert_gate_up_weights=self.gate_up_proj_weight.reshape(
                self.num_local_experts, self.hidden, 2,
                self.intermediate_size_per_rank),
            expert_down_weights=self.down_proj_weight,
            rank_id=rank_id,
            top_k=self.top_k,
            eps=self.rms_norm_eps,
            router_act_fn=RouterActFnType.SOFTMAX,
            router_pre_norm=True,
            norm_topk_prob=self.norm_topk_prob,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            hidden_act_fn=ActFnType.SiLU,
            # DIAGNOSTIC (env VLLM_MOE_TKG_ROUTER_FP32=1): run the DECODE MoE router
            # projection in fp32. The decode kernel routes over 256 experts in bf16
            # (router_mm_dtype default) while PREFILL routes via NF.router in fp32 —
            # a prefill-fp32 / decode-bf16 asymmetry suspected (TF_DIVERGENCE_ANALYSIS.md)
            # of producing the flat ~3-9% per-token TF logit floor + occasional top-k
            # expert-selection flips. This is the ONLY fp32 toggle the tkg kernel exposes
            # (it has no expert-matmul dtype arg). Default OFF keeps the graph unchanged.
            router_mm_dtype=(nl.float32 if _router_fp32 else nl.bfloat16),
            is_all_expert=True,
            skip_router_logits=True,
        )

        if self.moe_group.world_size > 1:
            output = self.moe_group.all_reduce(output)
        # DTYPE FIX (symmetry with forward_prefill): cast back to module (bf16) dtype so
        # the residual stream stays bf16 for the next layer's in_proj matmul.
        return (output + shared).to(self.dtype)

    def forward_prefill(self, hidden_states, positions, rank):
        hidden_states = self.post_attention_layernorm(hidden_states)
        shared = self._shared_expert(hidden_states)

        expert_affinities = NF.router(
            hidden_states=hidden_states,
            router_weights=self.router_weight.T,
            top_k=self.top_k,
            activation="softmax",
            computation_dtype=torch.float32,
            router_computation_order=RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER,
        )

        if self.tp_group.world_size > 1:
            expert_affinities = self.tp_group.all_gather(expert_affinities, dim=0)
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # Padding mask (True=real token, False=padding). The runner pads the real
        # prompt up to the bucket seq_len (e.g. 29 real tokens -> 1024). Without a
        # mask, build_blockwise_mapping routes the ~995 PADDING tokens to experts,
        # inflating token_position_to_id beyond the valid token rows -> the MoE
        # indirect gather goes OUT-OF-BOUNDS at execute (nrta 1006). Warmup uses a
        # full bucket of REAL tokens so it never trips. `positions` is full-T here
        # (the backbone scatters hidden via SP but passes positions whole, and
        # hidden is all_gathered back to full-T just above), so the mask aligns
        # directly. Mirrors the device-proven gpt_oss EP path (model_bf16.py:1321).
        # DIAGNOSTIC (env VLLM_MOE_PADDING_MASK=0 disables): the padding_mask was added
        # by commit 0be73f46d (an OOB fix). The KNOWN-GOOD coherent build f35d2ae7b
        # PREDATES it and passed padding_mask=None — and generates better on open-ended
        # prompts than current. argmax(positions) on the degenerate prompt's repeated-
        # last-position layout may mis-mask, corrupting MoE routing every layer. This
        # gate lets us A/B the mask (default ON = current; =0 reverts to the f35d2ae7b
        # None behavior) to test if the mask is the regressor.
        # ROOT FIX (seq=1024 nrta-1006 DGE OOB): DEFAULT-ON. The seq=1024 prefill DGE
        # gather survived find_nonzero chunk pinning (@512/@256), VLLM_GDN_NKI=0, AND both
        # MoE-dispatch index clamps (token_position_to_id->[-1,nt-1], block_to_expert->[0,E-1]).
        # The clamps only bound the index VALUES; they cannot shrink the dispatch LENGTH.
        # Without the padding mask, build_blockwise_mapping routes the padding tokens to
        # experts, inflating tokens_per_expert -> blocks_per_expert -> the live block count.
        # The runner's warmup/generate batch carries padded rows (positions plateau at the
        # last real index), so the dispatch over-allocates blocks whose gather over-runs the
        # statically-sized moe_cte buffers (num_static_block). Masking the affinities BEFORE
        # the expert_mask (moe_blockwise.py:_apply_padding_mask) drops those tokens from the
        # dispatch entirely, so the block count and gather extents stay within bounds. Set
        # VLLM_MOE_PADDING_MASK=0 to revert to the prior (raw, unmasked) behavior.
        padding_mask = None
        if positions is not None and os.environ.get("VLLM_MOE_PADDING_MASK", "1") == "1":
            last_real_idx = torch.argmax(positions)
            token_indices = torch.arange(
                positions.shape[0], device=positions.device)
            padding_mask = token_indices <= last_real_idx

        if self.ep_degree > 1:
            ep_rank = (rank % (self.ep_degree * self.tp_degree)) // self.tp_degree
            local_expert_indices = (
                torch.arange(self.num_local_experts,
                             device=hidden_states.device, dtype=torch.int32)
                + ep_rank.to(torch.int32) * self.num_local_experts
            )
            expert_affinities = NF.get_local_expert_affinities(
                expert_affinities, local_expert_indices)

        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = NF.build_blockwise_mapping(
            expert_affinities=expert_affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.top_k,
            block_size=self.block_size,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
            padding_mask=padding_mask,
        )

        num_tokens = hidden_states.shape[0]
        num_static_block = math.ceil(
            num_tokens * self.top_k / self.ep_degree / self.block_size)

        # DIAGNOSTIC (env-gated VLLM_MOE_TPID_CLAMP=1): clamp the MoE token-dispatch
        # gather index to the valid hidden_states row bounds. token_position_to_id
        # is the indirect (DGE) gather index that moe_cte uses to pull each block's
        # token rows out of hidden_states ([num_tokens, H]); valid values are the
        # token IDs [0, num_tokens-1], with -1 the padding sentinel (handled by
        # skip_token=True). build_blockwise_mapping derives these positions from
        # cumulative token counts + per-expert block offsets; if the chunked-GDN
        # prefill emits a token count / affinity layout that inflates a position
        # past num_tokens, this gather goes OOB at execute (nrta 1006). This is the
        # decisive test for whether the surviving prefill OOB is THIS MoE gather
        # (the prime suspect after the full-attn KV index_put_ was exonerated).
        # We clamp the UPPER bound only (max=num_tokens-1) and PRESERVE
        # the -1 padding sentinel (min=-1) so skip_token semantics are unchanged.
        # If the OOB clears with this on, the dispatch mapping is overflowing past
        # the real token rows; if it persists, this gather is exonerated too.
        # clamp() is XLA-safe (static shape, no .item()); default (flag off) keeps
        # the compiled graph byte-identical.
        # ROOT FIX (seq=1024 nrta-1006 DGE OOB): this clamp is now DEFAULT-ON. The
        # seq=1024 prefill DGE scatter/gather OOB survived find_nonzero chunk pinning
        # (@512 AND @256) AND survived VLLM_GDN_NKI=0 (GDN exonerated) — proving the OOB
        # is the MoE blockwise dispatch indirect gather reading token_position_to_id
        # values past the valid [0,num_tokens-1] hidden_states rows. This bound makes the
        # gather index legal (-1 sentinel preserved for skip_token) so the DGE descriptor
        # never addresses OOB. Set VLLM_MOE_TPID_CLAMP=0 to revert to the raw indices.
        if os.environ.get("VLLM_MOE_TPID_CLAMP", "1") == "1":
            token_position_to_id = token_position_to_id.clamp(
                min=-1, max=num_tokens - 1)

        output = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=self.gate_up_proj_weight.reshape(
                self.num_local_experts, self.hidden, 2,
                self.intermediate_size_per_rank),
            down_proj_weight=self.down_proj_weight,
            activation_function=ActFnType.SiLU,
            block_size=self.block_size,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            # skip_weight=True is REQUIRED here (not just an optimization). The
            # shard_on_block kernel over-allocates N blocks (num_blocks pads by
            # +E-1 for per-expert rounding) and INTERNALLY memsets the trailing
            # padding-block expert ids to E (=num_local_experts) — one past the
            # valid [0,E-1] weight rows (bwmm_shard_on_block.py:252). With
            # skip_weight=False the padding-block weight DMA runs oob_mode.error
            # and faults at runtime ("scatter/gather ... OUT-OF-BOUND ACCESS",
            # nrta 1006) — reached only at large seq/E_local (ours: E_local=32,
            # tp_degree=1; never tripped by qwen3_moe's smaller E_local). skip_weight
            # flips that DMA to oob_mode.skip (the intended padding-block guard,
            # companion to skip_token). Validated numerically by qwen3_moe's
            # mxfp8 three-way test (test_moe_prefill_static_mx_three_way.py).
            skip_weight=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
            num_static_block=num_static_block,
        )

        if self.moe_group.world_size > 1:
            output = self.moe_group.reduce_scatter(output, dim=0)
        # DTYPE FIX: NF.router runs fp32 (computation_dtype), so the POST_SCALE
        # affinity multiply in the moe_cte torch/meta path promotes `output` to
        # fp32. Left uncast, `residual + output` makes the next layer's RMSNorm
        # return fp32, feeding the following GDN in_proj as fp32 @ bf16-weight
        # (warmup dtype mismatch). Cast back to the module (bf16) dtype.
        return (output + shared).to(self.dtype)


# =============================================================================
# Section 6: Decoder Layers (Hybrid)
# =============================================================================


class Qwen3_5FullAttentionDecoderLayer(nn.Module):
    """Decoder layer with full attention (every 4th layer)."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = Qwen3_5FullAttention(config, layer_idx=layer_idx)
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        is_prefill: bool = True,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        hidden_states = self.mlp(
            hidden_states, positions=positions, is_decode=not is_prefill, rank=rank
        )
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3_5LinearAttentionDecoderLayer(nn.Module):
    """Decoder layer with linear attention (Gated DeltaNet)."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx=layer_idx)
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        is_prefill: bool = True,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(hidden_states, is_prefill=is_prefill, positions=positions,
                                         attn_metadata=attn_metadata)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(
            hidden_states, positions=positions, is_decode=not is_prefill, rank=rank
        )
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 7: Model Backbone
# =============================================================================


class Qwen3_5TextModel(nn.Module):
    """Qwen3.5 text transformer backbone with hybrid attention layers."""

    def __init__(self, config: Qwen3_5TextConfig):
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

        # Build hybrid layer stack
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            if config.layer_types[layer_idx] == "full_attention":
                self.layers.append(Qwen3_5FullAttentionDecoderLayer(config, layer_idx))
            else:
                self.layers.append(Qwen3_5LinearAttentionDecoderLayer(config, layer_idx))

        self.norm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3_5RotaryEmbedding(config)

        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.embed_tokens.tp_size,
                is_storage_transposed=False,
            ),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        rotary_position_ids: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Determine prefill vs decode from first full attention layer's metadata
        first_full_layer_idx = next(
            i for i, lt in enumerate(self.config.layer_types) if lt == "full_attention"
        )
        first_layer_name = f"layers.{first_full_layer_idx}.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name]["decode_token_threshold"]
        is_prefill = max_query_len > decode_token_threshold

        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )

        if is_prefill and inputs_embeds is not None and is_token_ids is not None:
            local_len = hidden_states.shape[0]
            start = self.rank * local_len
            inputs_embeds = inputs_embeds[start : start + local_len]
            is_token_ids = is_token_ids[start : start + local_len]

        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        position_embeddings = self.rotary_emb(
            rotary_position_ids, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for _layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                is_prefill=is_prefill,
                rank=rank,
            )

        hidden_states = self.norm(hidden_states)

        if is_prefill:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 8: Top-Level Model (LM Head + KV Cache + Weight Loading)
# =============================================================================

HF_TEXT_PREFIX = "model.language_model"


class Qwen3_5ForConditionalGeneration(nn.Module, HasInnerState, IsHybrid, SupportsMRoPE, SupportsSpatialMerge):
    """Qwen3.5 multimodal model with hybrid attention and LM head.

    Framework changes for hybrid model support:
    - get_kv_spec(): Only returns specs for full attention layers (16 of 64)
    - bind_kv_cache(): Only binds caches to full attention layers
    - Recurrent state for linear attention layers is managed separately
    - RecurrentStateSpec provides memory accounting for linear attention state
    """

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.text_config = config.text_config

        # Text backbone
        self.language_model = Qwen3_5TextModel(config.text_config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        nc = config.text_config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        debug_logits_enabled = nc is not None and nc.debug_logits_dir is not None
        self._gather_logits = (
            nc is not None and nc.max_logprobs != 0
        ) or debug_logits_enabled

        # LM head
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
            dtype=config.text_config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=config.text_config.vocab_size // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        rotary_position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(torch.int32)

        hidden_states = self.language_model(
            input_ids,
            positions,
            rotary_position_ids,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )

        # --- nrta-1006 OOB FIX (lm_head/sampler last-token gather, seq=1024 prefill) ---
        # `torch.index_select(hidden_states, 0, sampling_positions)` lowers to an indirect
        # DGE (dynamic_load off the post-transformer hidden_states, indexed by the runtime
        # `sampling_positions` tensor) emitted in OOBMode.ERROR: neuronx-cc cannot statically
        # prove sampling_positions stays in [0, hidden_states.shape[0]-1], so at seq=1024
        # prefill warmup every tile of this gather trips nrta-1006 ("scatter/gather indirect
        # DGE out-of-bound"). After the embedding gather was clamped (graph 68b4dfec) this
        # is the surviving error-mode indirect gather in the prefill graph. Clamping the
        # index to the valid row range is numerically safe: sampling_positions are always
        # in-range last-token positions (the runner builds them from real seq lengths), so
        # the clamp cannot change which row is read for any valid request; it only gives the
        # compiler a provably in-bounds index and removes the OOBMode.ERROR DGE. Default-on;
        # disable with VLLM_SAMPLE_INDEX_CLAMP=0 for A/B isolation.
        if os.environ.get("VLLM_SAMPLE_INDEX_CLAMP", "1") == "1":
            n_rows = hidden_states.shape[0]
            sampling_positions = sampling_positions.clamp(0, n_rows - 1)
        # SEQ1024 OOB CURE (named op _gather.73304 hlo_id 7 graph d26be9dd, the EARLIEST
        # surviving error-mode indirect DGE after embed-onehot): the index-VALUE clamp above
        # does NOT remove the OOBMode.ERROR gather (campaign lesson: clamps bound the index
        # but neuronx-cc still emits the indirect dynamic_load in ERROR mode). torch.index_select
        # over the n_rows (=seq) axis lowers to dynamic_load bf16<1x2048> off all_gather.34
        # (1024,4,512) indexed by sampling_positions -> nrta-1006 at seq=1024 warmup. CURE =
        # an INDEX-FREE one-hot @ matmul (same cure proven for conv-seed line 936 and embedding):
        # P[n_sample, n_rows], P[i,j] = (sampling_positions[i] == j); P @ hidden_states selects
        # the same rows via pure matmul with NO indirect addressing -> no DGE. Bit-identical to
        # index_select (exactly one nonzero per output row). Default ON; VLLM_SAMPLE_GATHER_ONEHOT=0
        # reverts to index_select for A/B isolation.
        if os.environ.get("VLLM_SAMPLE_GATHER_ONEHOT", "1") == "1":
            n_rows = hidden_states.shape[0]
            _cols = torch.arange(n_rows, device=hidden_states.device)  # [n_rows]
            # [n_sample, n_rows] one-hot selection matrix (one nonzero per row)
            _P = (sampling_positions.to(torch.int32).unsqueeze(1) == _cols.unsqueeze(0)).to(
                hidden_states.dtype
            )
            hidden_states_for_logits = _P @ hidden_states  # [n_sample, hidden]
        else:
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
            if self.tp_group is not None:
                gathered_logits = self.tp_group.all_gather(logits, dim=1)
            else:
                gathered_logits = logits

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler
            return rejection_sampler(spec_decode_metadata, sampled_tokens)

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig = None,
        vision_neuron_config: VisionNeuronConfig = None,
    ):
        config = Qwen3_5Config.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )
        return cls(config)

    @classmethod
    def get_vision_token_merge_factor(cls, hf_config: PretrainedConfig) -> int:
        return hf_config.vision_config.spatial_merge_size ** 2

    def get_mrope_input_positions(self, input_tokens, mm_features):
        # Delegate to the M-RoPE utility. compute_mrope_positions reads
        # config.video_token_id / vision_start_token_id / vision_end_token_id /
        # vision_config.spatial_merge_size, so it needs the TOP-LEVEL Qwen3_5Config
        # (self.config) — NOT text_config (lacks those) and NOT the bare int. For
        # text-only (mm_features empty) the vision fields are inert. self.config
        # always has a populated vision_config (built from the HF checkpoint).
        from vllm_neuron.model.qwen3_vl.utils.mrope import compute_mrope_positions
        return compute_mrope_positions(input_tokens, mm_features, self.config)

    # ── KV Cache (only full attention layers) ────────────────────────────

    def get_kv_spec(self):
        """Return HybridKVSpec with attention layers + stateful layer names.

        The model runner uses stateful_layer_names to emit MambaSpec entries
        for the scheduler's unified allocator.
        """
        layers = []
        stateful_names = []
        for i, layer in enumerate(self.language_model.layers):
            if self.text_config.layer_types[i] == "full_attention":
                layer_name = f"layers.{i}.self_attn"
                layers.append(
                    LayerSpec(
                        name=layer_name,
                        num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                        head_size=layer.self_attn.head_dim,
                        dtype=layer.self_attn.dtype,
                        sliding_window_size=None,
                        chunk_size=None,
                    )
                )
            else:
                stateful_names.append(f"layers.{i}.linear_attn")

        return HybridKVSpec(layers=layers, stateful_layer_names=stateful_names)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        """Bind KV caches to full attention layers only."""
        for i, layer in enumerate(self.language_model.layers):
            if self.text_config.layer_types[i] == "full_attention":
                layer_name = f"layers.{i}.self_attn"
                if layer_name not in kv_caches:
                    raise KeyError(f"KV cache for layer {layer_name} not initialized")
                _entry = kv_caches[layer_name]
                kc = _entry[0]
                vc = _entry[1]
                layer.self_attn.k_cache = kc
                layer.self_attn.v_cache = vc
                # UNIFIED: runner also passes [.., raw_slab, k_off, v_off] — the CONTIGUOUS raw slab
                # [num_blocks, page_stride] + K/V column offsets for the .ap gather/write kernels
                # (which must NOT reshape the strided k_cache/v_cache views). Stash them; absent in
                # the default (2-element) path -> None -> the kernels aren't used.
                if len(_entry) >= 5:
                    layer.self_attn._kv_raw_slab = _entry[2]
                    layer.self_attn._kv_k_off = _entry[3]
                    layer.self_attn._kv_v_off = _entry[4]
                else:
                    layer.self_attn._kv_raw_slab = None
                # UNIFIED-CACHE: when the runner binds a PAGE-STRIDED view of the shared slab
                # (block k at page k), the per-block stride is encoded in the tensor's own dim-0
                # stride (in ELEMENTS). The .ap gather kernel needs it as a literal. A contiguous
                # (non-shared) cache has dim0 stride == nkh*block_size*head_dim, so page_stride=0
                # (the kernel's "contiguous" sentinel) is equivalent — we pass 0 in that case so
                # the default path stays byte-identical. Detect: page-strided iff dim0 stride >
                # the packed per-block size.
                try:
                    _packed = kc.shape[1] * kc.shape[2] * kc.shape[3]  # nkh*block_size*head_dim
                    _s0 = kc.stride(0)
                    layer.self_attn._kv_page_stride = _s0 if _s0 > _packed else 0
                except Exception:
                    layer.self_attn._kv_page_stride = 0

    # ── IsHybrid interface ───────────────────────────────────────────────

    @classmethod
    def get_mamba_state_dtype(cls, vllm_config) -> tuple[torch.dtype, torch.dtype]:
        """Return dtypes for (conv_state, ssm_state)."""
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype if hasattr(vllm_config, 'model_config') else torch.bfloat16,
            getattr(vllm_config, 'cache_config', None) and vllm_config.cache_config.mamba_cache_dtype or None,
            getattr(vllm_config, 'cache_config', None) and vllm_config.cache_config.mamba_ssm_cache_dtype or None,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config) -> tuple[torch.dtype, torch.dtype]:
        """Alias for the name the platform alignment / base path expects."""
        return cls.get_mamba_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config=None, **kwargs
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Calculate shapes for conv and temporal state caches.

        Returns (conv_state_shape, temporal_state_shape) per the IsHybrid protocol.
        """
        if vllm_config is not None:
            hf_config = vllm_config.model_config.hf_text_config
            tp_size = vllm_config.parallel_config.tensor_parallel_size
            num_spec = (
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config else 0
            )
        else:
            # Fallback for direct instantiation without vllm_config
            hf_config = kwargs.get('hf_config')
            tp_size = kwargs.get('tp_size', 1)
            num_spec = 0

        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, ...]:
        """Return copy functions for prefix caching."""
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    # ── Recurrent State Management (Neuron-specific) ─────────────────────

    def bind_mamba_state(self, kv_caches: dict[str, list[torch.Tensor]]):
        """Bind framework-allocated recurrent state to linear-attention layers
        (OPT-FULL lifecycle). The Neuron runner allocates per-rank state tensors
        from this model's get_mamba_state_shape_from_config (already TP-divided)
        keyed by ``layers.{i}.linear_attn`` and hands them here. Order matches
        the IsHybrid shape tuple: [0]=conv_state, [1]=temporal/recurrent state.

        Replaces the legacy bind_recurrent_state, which allocated FULL-width
        module-internal tensors (ignoring the per-rank framework shapes) — the
        source of the unsharded-state XLA crash.
        """
        for i, layer in enumerate(self.language_model.layers):
            if self.text_config.layer_types[i] == "linear_attention":
                layer_name = f"layers.{i}.linear_attn"
                if layer_name not in kv_caches:
                    raise KeyError(
                        f"Mamba state for layer {layer_name} not initialized"
                    )
                state = kv_caches[layer_name]
                lin = layer.linear_attn
                lin.conv_state = state[0]
                lin.recurrent_state = state[1]
                # UNIFIED-CACHE: when the runner binds PAGE-STRIDED views of the shared slab, the
                # per-block stride is the tensor's own dim-0 stride (elements). The staging gather/
                # scatter need it as a literal. Contiguous (default) pool has dim0 stride == packed
                # row size, so page_stride=0 (contiguous sentinel) — byte-identical default path.
                for _st, _attr in ((state[0], "_conv_page_stride"), (state[1], "_rec_page_stride")):
                    try:
                        _packed = 1
                        for _d in _st.shape[1:]:
                            _packed *= _d
                        _s0 = _st.stride(0)
                        setattr(lin, _attr, _s0 if _s0 > _packed else 0)
                    except Exception:
                        setattr(lin, _attr, 0)
                # UNIFIED-CACHE: stash per-component CONTIGUOUS raw slabs + elem offsets for the .ap
                # gather/scatter kernels (they must NOT reshape the strided conv/recurrent views).
                # _gdn_raw_meta[layer] = [(raw_slab_conv, conv_off, conv_elems),
                #                         (raw_slab_rec,  rec_off,  rec_elems)]. Absent (default) -> None.
                _meta = getattr(self, "_gdn_raw_meta", {}).get(layer_name)
                if _meta is not None and len(_meta) >= 2:
                    lin._conv_raw_slab, lin._conv_raw_off, _ = _meta[0]
                    lin._rec_raw_slab, lin._rec_raw_off, _ = _meta[1]
                else:
                    lin._conv_raw_slab = None
                    lin._rec_raw_slab = None

    def reset_mamba_state(self, seq_indices: torch.Tensor | None = None):
        """Reset recurrent state for completed/evicted sequences (generic
        lifecycle hook called by the runner on finished_req_ids)."""
        return self.reset_recurrent_state(seq_indices)

    def reset_recurrent_state(self, seq_indices: torch.Tensor | None = None):
        """Reset recurrent state for completed/evicted sequences."""
        for i, layer in enumerate(self.language_model.layers):
            if self.text_config.layer_types[i] == "linear_attention":
                lin = layer.linear_attn
                if seq_indices is None:
                    lin.recurrent_state.zero_()
                    lin.conv_state.zero_()
                else:
                    lin.recurrent_state[seq_indices] = 0
                    lin.conv_state[seq_indices] = 0

    # ── Weight Loading ───────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from Qwen3.5 HF checkpoint.

        HF key prefix: model.language_model.layers.{i}.*
        Linear attention layers: model.language_model.layers.{i}.linear_attn.*
        Full attention layers: model.language_model.layers.{i}.self_attn.*
        """
        tp_rank = self.rank
        tp_size = self.world_size
        tc = self.text_config

        mappings = {}

        # Embedding and LM head
        mappings["language_model.embed_tokens.weight"] = f"{HF_TEXT_PREFIX}.embed_tokens.weight"
        mappings["lm_head.weight"] = "lm_head.weight"
        mappings["language_model.norm.weight"] = f"{HF_TEXT_PREFIX}.norm.weight"

        for layer_id in range(tc.num_hidden_layers):
            hf_prefix = f"{HF_TEXT_PREFIX}.layers.{layer_id}"
            model_prefix = f"language_model.layers.{layer_id}"

            # Norms: input_layernorm at layer level; post_attention_layernorm
            # folded INTO the MoE block (mlp.post_attention_layernorm).
            mappings[f"{model_prefix}.input_layernorm.weight"] = f"{hf_prefix}.input_layernorm.weight"
            mappings[f"{model_prefix}.mlp.post_attention_layernorm.weight"] = f"{hf_prefix}.post_attention_layernorm.weight"

            # MoE block (router + fused experts + shared expert) — every layer.
            mappings[f"{model_prefix}.mlp.router_weight"] = f"{hf_prefix}.mlp.gate.weight"
            mappings[f"{model_prefix}.mlp.gate_up_proj_weight"] = f"{hf_prefix}.mlp.experts.gate_up_proj"
            mappings[f"{model_prefix}.mlp.down_proj_weight"] = f"{hf_prefix}.mlp.experts.down_proj"
            mappings[f"{model_prefix}.mlp.shared_gate_proj"] = f"{hf_prefix}.mlp.shared_expert.gate_proj.weight"
            mappings[f"{model_prefix}.mlp.shared_up_proj"] = f"{hf_prefix}.mlp.shared_expert.up_proj.weight"
            mappings[f"{model_prefix}.mlp.shared_down_proj"] = f"{hf_prefix}.mlp.shared_expert.down_proj.weight"
            mappings[f"{model_prefix}.mlp.shared_expert_gate"] = f"{hf_prefix}.mlp.shared_expert_gate.weight"

            if tc.layer_types[layer_id] == "full_attention":
                # Fused QKV (separate Q, K, V in checkpoint → fused in model)
                mappings[f"{model_prefix}.self_attn.qkv_proj_weight"] = [
                    f"{hf_prefix}.self_attn.q_proj.weight",
                    f"{hf_prefix}.self_attn.k_proj.weight",
                    f"{hf_prefix}.self_attn.v_proj.weight",
                ]
                mappings[f"{model_prefix}.self_attn.o_proj_weight"] = f"{hf_prefix}.self_attn.o_proj.weight"
                mappings[f"{model_prefix}.self_attn.q_norm.weight"] = f"{hf_prefix}.self_attn.q_norm.weight"
                mappings[f"{model_prefix}.self_attn.k_norm.weight"] = f"{hf_prefix}.self_attn.k_norm.weight"
            else:
                # Linear attention
                mappings[f"{model_prefix}.linear_attn.in_proj_qkv_weight"] = f"{hf_prefix}.linear_attn.in_proj_qkv.weight"
                mappings[f"{model_prefix}.linear_attn.in_proj_z_weight"] = f"{hf_prefix}.linear_attn.in_proj_z.weight"
                mappings[f"{model_prefix}.linear_attn.in_proj_a_weight"] = f"{hf_prefix}.linear_attn.in_proj_a.weight"
                mappings[f"{model_prefix}.linear_attn.in_proj_b_weight"] = f"{hf_prefix}.linear_attn.in_proj_b.weight"
                mappings[f"{model_prefix}.linear_attn.conv1d_weight"] = f"{hf_prefix}.linear_attn.conv1d.weight"
                mappings[f"{model_prefix}.linear_attn.out_proj_weight"] = f"{hf_prefix}.linear_attn.out_proj.weight"
                mappings[f"{model_prefix}.linear_attn.norm.weight"] = f"{hf_prefix}.linear_attn.norm.weight"
                mappings[f"{model_prefix}.linear_attn.dt_bias"] = f"{hf_prefix}.linear_attn.dt_bias"
                mappings[f"{model_prefix}.linear_attn.A_log"] = f"{hf_prefix}.linear_attn.A_log"

        # Load checkpoint
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device
        ).state_dict

        # Cast each loaded tensor to its TARGET PARAMETER dtype. Most params are
        # bf16, but the GatedDeltaNet keeps A_log / dt_bias / the gated norm.weight
        # in fp32 (their nn.Parameters are declared fp32). assign=True copies the
        # tensor in as-is, so a dtype mismatch (e.g. bf16 tensor into an fp32 param)
        # raises "Expected self.dtype() == dst.dtype()". Drive the cast off the
        # actual param dtype rather than name-matching to be exhaustive.
        param_dtypes = {n: p.dtype for n, p in self.named_parameters()}
        target_dtype = tc.torch_dtype
        for name, tensor in rank_sharded.items():
            want = param_dtypes.get(name, target_dtype)
            if tensor.dtype != want:
                rank_sharded[name] = tensor.to(want)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
