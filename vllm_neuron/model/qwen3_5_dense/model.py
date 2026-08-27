# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 Dense BF16 Implementation (Qwen3.6-27B)
===============================================

Hybrid multimodal model with vision encoder + text decoder — DENSE FFN variant.
Text decoder mixes Gated DeltaNet (linear attention) and standard GQA full
attention layers in a 3:1 ratio (64 layers = 48 linear + 16 full).

Sibling of the ``qwen3_5_moe`` port; identical hybrid attention / RoPE / vision,
differing only in the FFN: every layer is a plain SwiGLU MLP (Qwen3_5DenseMLP)
instead of a 256-expert top-8 MoE + shared expert.

Key architectural features:
  - Full attention: GQA + QK-norm + output gate (sigmoid) + partial M-RoPE
  - Linear attention: Gated DeltaNet with causal conv1d + recurrent state
  - Dense FFN: SwiGLU MLP (intermediate_size=17408), TP intermediate sharding
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
    trips nrta-1006 at warmup_prefill@1024. A DGE enumeration named the only two
    surviving error-mode indirect DGEs as ``_gather.2004``/``_gather.2159``
    (dynamic_load by ``select.9`` over the
    GDN query/key head-expand tensor) == THESE call sites (model.py:975-976 GDN
    prefill q/k head expand + decode + full-attn). Equivalent broadcast form uses
    only unsqueeze/expand/reshape (NO indirect addressing) and is bit-identical:
    each slice along ``dim`` is repeated ``repeats`` times CONSECUTIVELY, which is
    exactly repeat_interleave's interleave order (CPU-verified bit-exact in the
    sibling qwen3_5_moe/model_bf16.py:33). Unconditional: the repeat_interleave it
    replaces reintroduces the error-mode DGE, and the two are bit-identical, so there
    is nothing to trade. An index-free rewrite removes the DGE; a value clamp never does.
    """
    shape = list(x.shape)
    exp = shape[:dim + 1] + [repeats] + shape[dim + 1:]
    out = shape[:dim] + [shape[dim] * repeats] + shape[dim + 1:]
    return x.unsqueeze(dim + 1).expand(*exp).reshape(*out)


import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

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
from vllm_neuron.model.interfaces import (
    SupportsMRoPE,
    SupportsSpatialMerge,
    SupportsVisionWarmup,
)
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
)

import nki.language as nl
from nkilib.core.utils.common_types import NormType, QuantizationType
from nkilib.core.mlp.mlp_parameters import TKG_BS_SEQLEN_THRESHOLD

from .config import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from .quantization import QuantScheme
from .weight_loaders_fp8_row import (
    fp8_block_dequant_bf16_loader,
    fp8_row_down_loader,
    fp8_row_gate_up_loader,
)

# ── Vision wiring (image + video) ────────────────────────────────────────────
# Reuse the tested Qwen3-VL BF16 vision tower and its FFD packing / encoder-cache
# merge utilities verbatim. The ViT is byte-for-byte the Qwen3-VL ViT (same
# ``model.visual.*`` checkpoint keys); the only architectural difference is
# ``deepstack_visual_indexes == []`` for Qwen3.6-27B (no deepstack injection), so
# we WIRE the existing encoder into the hybrid backbone rather than fork it.
import time as _time

from vllm.v1.utils import record_function_or_nullcontext

from vllm_neuron.model.qwen3_vl.utils.merge_vision_embeds import (
    merge_vision_embeddings,
)
from vllm_neuron.model.qwen3_vl.utils.vision_block_packing import (
    compute_block_bounds,
    ffd_pack_images,
    scatter_to_blocks,
    select_vision_bucket,
)
from vllm_neuron.model.qwen3_vl.utils.vision_preprocessing import (
    compute_position_indices_and_weights,
    compute_rotary_pos_emb,
)
from vllm_neuron.model.qwen3_vl.vision_encoder_bf16 import Qwen3VLVisionModel

logger = logging.getLogger(__name__)

# DIAGNOSTIC (default OFF; the fp8 path is unaffected when unset): run an FP8_ROW
# checkpoint with the MLP gate/up/down kept in bf16 — the [weight,
# weight_scale_inv] pair is block-dequantised to bf16 at load and the plain bf16
# NF.mlp runs — while QKV/o_proj/GDN stay on their normal fp8→bf16 dequant path.
# This is how the fp8-vs-bf16 comparison is reproduced: the same checkpoint, the
# same serve config, and only the MLP precision differing, which is what isolates
# the effect of fp8 on memory, throughput and accuracy. It also splits "ROW MLP
# kernel/scales" from "shared fp8 checkpoint load" when debugging a regression.
_FP8_MLP_BF16 = os.environ.get("VLLM_QWEN38_FP8_MLP_BF16") == "1"


# =============================================================================
# Vendored attention_tkg (d_head=256) flash DECODE kernel wiring (opt-in).
# =============================================================================
# Replaces ONLY the eager scores->softmax->AV block in Qwen3_5FullAttention.
# forward_decode. Gated on VLLM_QWEN_FLASH_ATTN=1 (default OFF => the eager block
# runs byte-identical). The d_head=256 kernel additionally requires the vendored
# (512-cap) nkilib tree; if that tree is NOT active
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
        # All-fp32 through the gate, with a single cast at the end. HF rounds to bf16 BEFORE
        # the gate instead (Qwen3_5MoeRMSNormGated.forward, transformers
        # modeling_qwen3_5_moe.py:182). That difference was measured on device and is INERT
        # here: reproducing HF's rounding in all 48 GDN layers left the 3-way logit
        # validation's aggregate sigma-ratio unchanged from this path, and left
        # generation unchanged. The reason appears to be that a bf16 dtype annotation on an
        # intermediate of a fused elementwise chain does not force a rounding on this
        # backend — the chain is evaluated in the engine's fp32 and only the final store is
        # rounded. So this form is kept for clarity, not for numerics; do not re-litigate it
        # without a device measurement.
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
        qkv_loader = fused_qkv_weight_loader(
            q_size=q_out // self.world_size,
            kv_size=self.num_key_value_heads_per_rank * self.head_dim,
            shard_dim=1,
            num_shards=self.world_size,
            is_storage_transposed=True,
            num_kv_replicas=self.num_kv_replicas,
        )
        o_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=(self.num_attention_heads * self.head_dim) // self.world_size,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        # v1 keeps full-attn QKV/o_proj in bf16, but the FP8 checkpoint stores
        # them as block-FP8. Wrap the bf16 loaders so they block-dequant the fp8
        # bytes to bf16 first (the GQA head-aware sharding is reused verbatim).
        # fp8'ing QKV is a v1.1 add-on (needs NF.qkv_proj SBUF fit @TP4).
        if getattr(config, "quant_scheme", QuantScheme.NONE) == QuantScheme.FP8_ROW:
            qkv_loader = fp8_block_dequant_bf16_loader(qkv_loader)
            o_loader = fp8_block_dequant_bf16_loader(o_loader)
        set_weight_loader(self.qkv_proj_weight, qkv_loader)
        set_weight_loader(self.o_proj_weight, o_loader)

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

    def _qkv_project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fused-QKV projection: [T, H] @ [H, qkv_dim] -> [T, qkv_dim].

        Both call sites use qkv_proj as a PURE projection (no fused norm / RoPE /
        bias — those are applied separately downstream), so a plain matmul is
        numerically equivalent to the NF.qkv_proj megakernel.

        Why the matmul fallback exists: this is a GATED attention (Q emits 2x per
        head — query + gate), so the per-rank fused_qkv width is
        num_q_heads/TP * d_head * 2 + 2 * (num_kv_heads/TP * d_head). At TP=4 on a
        trn2.3xlarge (24 q-heads, d_head 256) that is 3072 + 512 = 3584; the fused
        NF.qkv_proj kernel's weight-resident SBUF tile then overruns the ~208 KiB
        budget by ~8% (INKI016, independent of prefill length). At TP=8 (48xlarge)
        the per-rank width halves to 1792 and the kernel fits. Set
        VLLM_QWEN_QKV_MATMUL=1 to take the matmul path on the smaller instance;
        default OFF keeps the fused kernel for the TP>=8 serving config.
        """
        if os.environ.get("VLLM_QWEN_QKV_MATMUL") == "1":
            return torch.matmul(hidden_states, self.qkv_proj_weight)
        return NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
        ).squeeze(0)

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
        qkv = self._qkv_project(hidden_states)

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

        # Write this step's K/V into the paged cache.
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
        else:
            self.k_cache.index_put_(
                (block_indices_for_put, head_indices_for_put, position_indices_for_put),
                k_flat,
            )
            self.v_cache.index_put_(
                (block_indices_for_put, head_indices_for_put, position_indices_for_put),
                v_flat,
            )

        # APC / CHUNKED-PREFILL PREFIX-KV: on a prefix-cache hit (or any chunked
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
            # PADDING-AWARE SCATTER (matches the GDN path's argmax(positions) guard):
            # the runner RIGHT-pads prefill with padding tokens whose `positions` REPEAT
            # the last real position (confirmed: positions=[base..L-1, L-1, L-1, ...]).
            # Scattering by the raw (clamped) `positions` makes every padding token target
            # the SAME slot as the real last token; torch.scatter is last-writer-wins on
            # duplicate indices, so padding (later in sequence order) OVERWRITES the real
            # last token's self-K/V -> only the last token gets a corrupted self-key, only
            # under segmentation. Fix: target by SEQUENCE-ORDER slot (base + arange), which
            # equals `positions` for real tokens but gives padding UNIQUE slots beyond the
            # last real position — slots the absolute-position causal mask below already
            # excludes (gathered_idx <= query_pos). No collision -> real last token intact.
            seq_slot = (
                positions[:1].long()
                + torch.arange(S_q, device=hidden_states.device, dtype=torch.long)
            ).clamp_(max=S_ctx - 1)
            scatter_pos = seq_slot.view(B, 1, S_q, 1).expand(B, nkh, S_q, self.head_dim)
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
        qkv = self._qkv_project(hidden_states)
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
        # GDN in_proj_qkv / in_proj_z / out_proj are block-FP8 on disk but kept
        # bf16 in v1: wrap their loaders to block-dequant fp8->bf16 first (the
        # head-aware sharding runs verbatim on the bf16 shim). in_proj_a/b,
        # conv1d, dt_bias, A_log are bf16-stored -> untouched.
        is_fp8 = getattr(config, "quant_scheme", QuantScheme.NONE) == QuantScheme.FP8_ROW

        def _fp8_wrap(loader):
            return fp8_block_dequant_bf16_loader(loader) if is_fp8 else loader

        if self.world_size > 1:
            set_weight_loader(self.in_proj_qkv_weight, _fp8_wrap(gated_deltanet_in_proj_qkv_loader(
                self.key_dim, self.value_dim, self.world_size)))
            set_weight_loader(self.conv1d_weight, gated_deltanet_conv1d_loader(
                self.key_dim, self.value_dim, self.world_size))
            set_weight_loader(self.in_proj_z_weight, _fp8_wrap(gated_deltanet_dim1_head_loader(self.world_size)))
            set_weight_loader(self.in_proj_a_weight, gated_deltanet_dim1_head_loader(self.world_size))
            set_weight_loader(self.in_proj_b_weight, gated_deltanet_dim1_head_loader(self.world_size))
            set_weight_loader(self.dt_bias, gated_deltanet_dim0_head_loader(self.world_size))
            set_weight_loader(self.A_log, gated_deltanet_dim0_head_loader(self.world_size))
            set_weight_loader(self.out_proj_weight, _fp8_wrap(gated_deltanet_out_proj_loader(
                self.value_dim, self.world_size)))
        else:
            # world=1: transpose in_proj/out_proj, squeeze conv1d.
            transpose_loader = SafetensorsWeightLoader(
                transform=lambda slices, rank: slices[0][:].t()
            )
            conv_loader = SafetensorsWeightLoader(
                transform=lambda slices, rank: slices[0][:].squeeze(1)
            )
            set_weight_loader(self.in_proj_qkv_weight, _fp8_wrap(transpose_loader))
            set_weight_loader(self.in_proj_z_weight, _fp8_wrap(transpose_loader))
            set_weight_loader(self.in_proj_a_weight, transpose_loader)
            set_weight_loader(self.in_proj_b_weight, transpose_loader)
            set_weight_loader(self.conv1d_weight, conv_loader)
            set_weight_loader(self.out_proj_weight, _fp8_wrap(transpose_loader))

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
        non-APC (keys absent) -> None -> byte-identical.

        The seed is ALWAYS ON when the APC metadata is present — there is no point
        NOT reading the precomputed prefix state from the cache — so there is no env
        short-circuit here; seeding activates purely on metadata presence (APC-only
        by construction), and the guards below keep non-APC byte-identical."""
        if attn_metadata is None or self.recurrent_state is None:
            return None
        md = attn_metadata.get(f"layers.{self.layer_idx}.linear_attn")
        if md is None:
            md = attn_metadata.get(f"language_model.layers.{self.layer_idx}.linear_attn")
        if md is None:
            return None
        seed_idx = md.get("seed_state_indices")
        has_init = md.get("has_initial_state")
        # NON-APC CHUNKED PREFILL. Without prefix caching there is no separate seed block, but the
        # request's OWN slot already holds the previous chunk's state: every prefill chunk ends by
        # writing last_state there via _seed_state(state_indices). Reading it back is what makes the
        # recurrence continuous across chunk boundaries. Upstream vLLM does exactly this on CUDA --
        # has_initial_states_p = (num_computed_tokens > 0), with no prefix-caching involvement
        # (vllm/v1/attention/backends/mamba_attn.py). Without it the 48 GatedDeltaNet layers restart
        # from zeros at every boundary and effectively see only the last max-num-batched-tokens
        # tokens, which shows up as a large strict-match regression on long prompts and as
        # 1-token empty completions at 6,300 tokens.
        _own_slot = seed_idx is None and md.get("state_indices") is not None
        if _own_slot:
            seed_idx = md.get("state_indices")
        if seed_idx is None or has_init is None:
            return None
        # GUARD: [:1] is correct ONLY for batch-1 prefill (Neuron SP path,
        # platform-enforced). Batched prefill would silently give seqs 1..B-1 seq-0's seed.
        # It applies to the APC seed only: there the indices are a per-row align gather, so distinct
        # values mean several sequences are being seeded and row 0 would be wrong for the others. On
        # the own-slot path the indices are legitimately distinct (one slot per request) and the
        # runner zeroes has_initial_state whenever more than one request is prefilling, so row 0 is
        # the prefilling request by construction.
        if not _own_slot and seed_idx.shape[0] > 1 and not bool((seed_idx == seed_idx[0]).all()):
            raise AssertionError(
                "APC recurrent seed assumes batch=1 (distinct seed_state_indices detected); "
                "batched prefill would mis-seed. See _apc_prefill_seed.")
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
        # Mask: fresh request (has_init=0) -> zeros == initial_state None. NaN-SAFE SELECT:
        # a genuinely-cold request gathers an UNALLOCATED seed block whose bytes may be
        # NaN/Inf; `seed * 0 = NaN` would poison the S0-fold (P0@S0) and diverge from the
        # APC-off single-slot path. torch.where SELECTS the zero branch (no arithmetic on the
        # poisoned value) -> exact zeros for a cold request, real seed for a cache-hit resume.
        out = torch.where(has_init.to(torch.bool), seed, torch.zeros_like(seed))
        return out

    def _apc_conv_seed(self, attn_metadata, device):
        """T2 APC (conv restore, mirrors _apc_prefill_seed for the CONV window).

        On a cache-hit prefill, return the SAVED prior conv window [1, conv_dim_local, k-1]
        from the seed block (conv_state[seed_slot]), masked by has_initial_state so a FRESH
        request gets zeros (== the zero-left-pad path). This mirrors upstream
        causal_conv1d_fn(has_initial_state=..., cache_indices=[:,0]) which restores the prior
        conv window on a cache hit; without it prefill zero-left-pads and loses the prefix's
        last k-1 conv inputs. Returns None when APC off / keys absent (byte-identical fallback).

        ALWAYS ON when APC metadata is present (mirrors _apc_prefill_seed): the precomputed
        conv window is always restored from cache on a hit; the metadata-presence guards
        below keep non-APC byte-identical."""
        if attn_metadata is None or self.conv_state is None:
            return None
        md = attn_metadata.get(f"layers.{self.layer_idx}.linear_attn")
        if md is None:
            md = attn_metadata.get(f"language_model.layers.{self.layer_idx}.linear_attn")
        if md is None:
            return None
        seed_idx = md.get("seed_state_indices")
        has_init = md.get("has_initial_state")
        # NON-APC CHUNKED PREFILL: mirror _apc_prefill_seed and left-pad from the request's own slot,
        # which the previous chunk wrote. Doing the recurrent state without the conv window would
        # leave the first conv_kernel_size-1 positions of each chunk convolved against zeros.
        _own_slot = seed_idx is None and md.get("state_indices") is not None
        if _own_slot:
            seed_idx = md.get("state_indices")
        if seed_idx is None or has_init is None:
            return None
        # GUARD: batch-1-only seed (see _apc_prefill_seed); fail loud if batched.
        # APC path only, for the reason given in _apc_prefill_seed.
        if not _own_slot and seed_idx.shape[0] > 1 and not bool((seed_idx == seed_idx[0]).all()):
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
        # NaN-SAFE SELECT: mirror _apc_prefill_seed — a cold request gathers an unallocated
        # conv block (garbage/NaN); `seed * 0 = NaN` would poison the conv-window cat.
        # torch.where selects exact zeros for a cold request, real window for a resume.
        out = torch.where(has_init.to(torch.bool), seed, torch.zeros_like(seed))
        return out.to(self.conv_state.dtype)

    def _carry_forward(self, attn_metadata) -> None:
        """IN-GRAPH block->block state carry-forward: move each request's running
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
        # T2 APC: in-graph block->block carry-forward BEFORE reading state, so the
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

        # PADDING MASK: mirrors HF apply_mask_to_padding_states.
        # The runner right-pads (positions stop incrementing at the last real token), so trailing
        # rows are padding. Zero them BEFORE the conv/scan so padding contributes nothing to the
        # carried conv_state/recurrent_state used by decode. Without this, decode state is polluted
        # by padding -> corrupted generation (prefill outputs stay clean since the conv is causal).
        if positions is not None:
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
        # PADDING-AWARE conv seed: the runner RIGHT-pads the
        # prompt (real tokens first, then padding that repeats the last position), so
        # the sequence TAIL is padding. Seeding conv_state from the tail (qkv_t[...,-(k-1):])
        # carries PADDING into decode -> corrupted generation. HF masks padding before GDN
        # (apply_mask_to_padding_states) / left-pads linear attn. Fix: seed from the window
        # ENDING at the last REAL token (argmax(positions) -> last incrementing index), via
        # a dynamic-slice gather. Seeding from the tail instead corrupts generation.
        if self.conv_state is not None:
            if positions is not None:
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
                # one-hot selection matrix P: [T, k1], P[t,j] = (t == cols[j])
                _rows = torch.arange(_seq_T, device=qkv_t.device).unsqueeze(1)  # [T,1]
                _P = (_rows == cols.unsqueeze(0)).to(qkv_t.dtype)               # [T,k1]
                _gathered = qkv_t[0] @ _P                                       # [conv_dim_local, k1]
                _seed_state(self.conv_state, _gathered.unsqueeze(0), getattr(self, '_conv_page_stride', 0),
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

        # PADDING-AWARE RECURRENT STATE: zeroing the
        # hidden states does NOT make padding scan steps no-ops — beta=sigmoid(0)=0.5 and
        # g=-exp(A_log)*softplus(dt_bias)!=0, so each padding step still decays the carried
        # recurrent_state by exp(g) (geometric decay ~ #padding tokens) and applies a delta.
        # The carried state is taken AFTER the full padded T, so right-padding pollutes it.
        # HF avoids this by left-padding linear attn. Fix: make padding steps TRUE no-ops by
        # forcing g=0 (=> exp(0)=1, no decay) and beta=0 (=> delta=0, no state update) on
        # padding rows. q/k/v need no masking: beta=0 kills the update term. seq=positions T.
        if positions is not None:
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

        # GDN prefill kernel selection — two supported paths, chosen by VLLM_GDN_PREFILL:
        #
        #   sequential (DEFAULT) : the exact sequential recurrence inside one bounded NKI kernel.
        #                          Numerically faithful, and the path that passes the generic
        #                          3-way logit validation.
        #   parallel             : the chunked reformulation. Much lower prefill latency (TTFT
        #                          ~3.8x on this model) but it solves T=(I-A)^-1 by a blocked
        #                          fp32 expansion the sequential form never performs, which
        #                          raises logit error relative to the sequential form. Top-1
        #                          argmax does not diverge either way and GSM8K
        #                          shows no significant difference, so this is a latency-versus-
        #                          numerical-fidelity trade, not a correctness one.
        #
        # Chunk width for the parallel path: 64 is what the kernel was validated at, and it is
        # the matmul moving-free dimension, which is why that path has no 512-token wall.
        _PARALLEL_CHUNK_SIZE = 64
        if self._gdn_prefill_mode() == "parallel":
            # THIRD-IMPL fast prefill: chunked GDN via a SELF-CONTAINED NKI kernel
            # (functional/linear_attention/gated_delta_rule.py). The earlier in-tree chunked
            # rule single-shots but corrupts state -> NaN: its seq<->(nchunk,chunk) axis-split
            # RESHAPE lowers to an out-of-bounds vector-DGE indirect copy (nrta 1006), proven
            # by fxgraph diff. This kernel does the chunking INSIDE the kernel with static-offset
            # slices only (no torch-level reshape, no DGE) -> single-shot capable (matmul
            # moving-free-dim == chunk_size 64, so no 512 wall) AND no OOB. Same raw-input
            # contract as _chunk_gated_delta_rule: the kernel applies q/k l2norm + q-scale
            # internally, so pass the un-normed repeat-interleaved q/k straight through.
            # APC cache-hit seed (_apc_init, [1,H,Dk,Dv]) passed as initial_state; None ->
            # kernel zero-inits the state (fresh/non-APC prefill).
            from vllm_neuron.functional.linear_attention import (
                chunk_gated_delta_rule as _nad_chunk_gdr,
            )
            # CONTIGUITY (root-cause fix): query/key/value arrive as NON-CONTIGUOUS strided
            # views — qkv_conv[..., a:b].reshape(...) keeps the T-stride == conv_dim_local
            # (a last-dim slice + view-reshape, no copy), and value is not repeat-interleaved.
            # The chunk public API passes these straight to the NKI kernel WITHOUT a
            # .contiguous() (unlike the SEQ_NKI path, whose prologue does
            # .transpose(1,2).contiguous()). A strided view fed to the kernel is read with the
            # wrong strides on device -> garbage state -> NaN decode, while the SEQ path (same
            # source tensors, forced contiguous) is fine. the sibling chunk kernel documents the same
            # hazard ("torch.split non-contiguous views mis-lower on Neuron -> garbage").
            # Force contiguous here to match the SEQ path.
            core_attn_out, last_state = _nad_chunk_gdr(
                query.contiguous(), key.contiguous(), value.contiguous(),
                g.contiguous(), beta.contiguous(),
                chunk_size=_PARALLEL_CHUNK_SIZE,
                initial_state=_apc_init,
                output_final_state=(self.recurrent_state is not None),
                use_qk_l2norm_in_kernel=True,
            )
            # OUTPUT-ALIASING GUARD (diagnostic → candidate fix): the chunk kernel returns
            # TWO outputs (core_attn_out #0, last_state #1) from a SINGLE NKI call. Compile
            # traces show `aliasing_output_rewrite` building a large io_map for THIS kernel
            # specifically. The very next op (_seed_state at ~L1830) WRITES last_state into the
            # framework-bound recurrent_state; if neuronx-cc aliased the two output buffers,
            # that write can clobber core_attn_out AFTER the kernel returned but BEFORE we
            # consume it -> garbage first token, coherent-standalone kernel. The SEQ_NKI path
            # is a different kernel whose two outputs the compiler does not alias, so it is
            # unaffected. Force independent buffers before the seed write. Standalone the
            # kernel is proven correct (rel 3e-3 @ real positions, LNC 1/2, H 12/48), so a
            # clone that restores coherence pins the bug to output aliasing.
            core_attn_out = core_attn_out.clone()
            if last_state is not None:
                last_state = last_state.clone()
            # NOTE: the clone fixes prefill garbage (aliasing), but chunk decode remains
            # unstable on some prompts. Root-caused (NOT the boundary state): the chunk
            # parallel form computes T=(I-A)^{-1} via an fp32 Neumann-series doubling
            # (_gated_delta_rule_kernel.py) that the exact sequential form never performs;
            # for ill-conditioned intra-chunk A this carries ~3e-3 error into core_attn_out
            # and this model's knife-edge decode collapses. Seeding decode from a bit-exact
            # sequential boundary state did NOT fix it (measured),
            # confirming the OUTPUT path, not the state, is the culprit. The stable path is
            # the sequential NKI path below.
        else:
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
        # T2 APC: in-graph block->block carry-forward BEFORE the recurrent read, so
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
        # pad), and index_select+explicit-coord index_put_ (Option 1,. neuronx-cc rejects
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
        # threaded by in-place mutation. Option 5.
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
            # PARALLEL MODE (VLLM_GDN_PREFILL=parallel): use the chunked recurrent NKI kernel for
            # decode so the recurrent-state CONVENTION is self-consistent end-to-end with the
            # chunk prefill kernel that seeded it. The torch decode path expects the
            # SEQUENTIAL-scan state convention; feeding it the chunk kernel's last_state drifts
            # to NaN over decode steps. Kernel requires
            # seq_len==1 and a non-None fp32 initial state; spec-decode (seq_len>1) or a missing
            # rec_init falls back to the stable torch scan.
            # the STABLE torch scan is used for decode even when
            # chunk prefill is active — a localization switch to separate a prefill-seeded
            # -state fault from a recurrent-decode-kernel wiring fault.
            # Decode MUST follow the prefill mode: the recurrent-state convention differs
            # between the two prefill kernels and decode reads what prefill wrote. Feeding the
            # parallel kernel's last_state into the sequential torch scan drifts to NaN over
            # decode steps. The kernel needs seq_len == 1 and
            # a non-None fp32 initial state; spec-decode or a missing rec_init falls back to the
            # sequential torch scan, which is also what "sequential" mode uses throughout.
            if (self._gdn_prefill_mode() == "parallel"
                    and seq_len == 1 and rec_init is not None):
                from vllm_neuron.functional.linear_attention import (
                    recurrent_gated_delta_rule as _nad_recur_gdr,
                )
                core_attn_out, last_state = _nad_recur_gdr(
                    query, key, value, g, beta,
                    initial_state=rec_init.to(torch.float32),
                    output_final_state=(self.recurrent_state is not None),
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out, last_state = self._recurrent_gated_delta_rule(
                    query, key, value, g, beta,
                    initial_state=rec_init,
                )

            # Decode availability guard: SANITIZE ONLY, no magnitude bound.
            #
            # The failure being guarded against is real: this model's GDN decode recurrence
            # S = S*exp(g) + k⊗delta is metastable, and when the forget gate exp(g)->1 the state
            # norm can grow unbounded over ~64 steps -> fp32 overflow -> NaN logits -> an
            # out-of-vocabulary token id (2143322048 = 0x7FC40000) -> EngineDeadError, which takes
            # down EVERY in-flight request rather than just the offending one.
            #
            # What this used to do, and why it changed. It also applied a magnitude clamp to +-4,
            # on the documented grounds that in-distribution trajectories keep |S| in 0.35-1.15 so
            # the bound "never fires (outputs byte-identical)". That was measured on the
            # sequential path only and is WRONG on the shipped configuration: with the clamp at 4
            # the 3-way logit validation regresses, and with the bound raised to 100 or 1e30 — the
            # ops still present, so they cannot engage — it recovers, identical to removing the guard
            # outright. The clamp was therefore making the model measurably LESS accurate than plain
            # bf16 on ordinary input,
            # while the bound was in fact engaging (|S| exceeds 4, and is <= 100). Neither the HF
            # reference nor upstream vLLM's CUDA path clamps this state.
            #
            # So the bound is gone and only the non-finite replacement remains. It cannot perturb
            # anything: it touches an element only if that element is already NaN or +-inf, i.e.
            # only after the trajectory has already diverged. Diverged entries are reset to 0
            # rather than to a large finite value, because a finite-but-huge state would simply
            # overflow again on the next step; zeroing lets that head's memory re-accumulate and
            # degrades one request instead of killing the engine.
            if last_state is not None:
                last_state = torch.nan_to_num(
                    last_state, nan=0.0, posinf=0.0, neginf=0.0
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



    _GDN_PREFILL_MODES = ("sequential", "parallel")

    @classmethod
    def _gdn_prefill_mode(cls) -> str:
        """Which GatedDeltaNet prefill kernel to use: ``sequential`` (default) or ``parallel``.

        Set ``VLLM_GDN_PREFILL``. An unrecognised value raises rather than silently falling
        back, so a typo cannot quietly change which kernel — and therefore which numerics — a
        deployment runs. See the dispatch comment in ``forward_prefill`` for the trade-off.
        """
        mode = os.environ.get("VLLM_GDN_PREFILL", "sequential").strip().lower()
        if mode not in cls._GDN_PREFILL_MODES:
            raise ValueError(
                f"VLLM_GDN_PREFILL={mode!r} is not recognised; "
                f"expected one of {', '.join(cls._GDN_PREFILL_MODES)}"
            )
        return mode

    def _segmented_gated_delta_rule_nki(
        self, query, key, value, g, beta, output_final_state=False, initial_state=None,
    ):
        """VLLM_GDN_PREFILL=sequential (the default): host wrapper around the SEQUENTIAL (recurrent) GDN NKI
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



# =============================================================================
# Section 5: MLP (SwiGLU)
# =============================================================================


class Qwen3_5DenseMLP(nn.Module):
    """Dense SwiGLU MLP with TP intermediate sharding — the Qwen3.6-27B FFN.

    Replaces the sibling qwen3_5_moe's Qwen3_5MoeSparseMoeBlock (256-expert
    top-8 + shared expert). Like that block, this owns the post_attention_layernorm
    (the decoder layer does NOT pre-normalize before self.mlp), so the norm is
    applied exactly once here to keep the residual wiring identical to the MoE
    variant. Mirrors the device-proven dense idiom in vllm_neuron/model/llama3
    (LlamaMLP): column-parallel gate/up, row-parallel down, SP all-gather on
    prefill / reduce_scatter after, all_reduce on decode.

    Weight orientation matches LlamaMLP (is_storage_transposed loaders):
      - gate_proj_weight / up_proj_weight: [hidden, intermediate/TP]
      - down_proj_weight:                  [intermediate/TP, hidden]

    Unlike the MoE block, the post-attention norm uses the standard
    Qwen3_5RMSNorm (folds +1 in forward) and loads the RAW HF weight (no
    load-time +1 fold) — there is no fused-gamma kernel here to satisfy.
    """

    def __init__(self, c):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.hidden = c.hidden_size
        self.rms_norm_eps = c.rms_norm_eps
        self.dtype = c.torch_dtype
        self.intermediate_size_per_rank = c.intermediate_size // self.world_size

        # Resident-weight FP8: when the checkpoint is block-FP8 we re-quantize
        # gate/up/down to per-channel ROW fp8 at load and drive NF.mlp(ROW). The
        # activation is quantized path-dependently (see forward): CTE (prefill /
        # large-batch decode) pre-quantizes via NF.rmsnorm_quant(ROW) -> packed
        # fp8 [T, H+4]; small-batch decode (TKG) feeds bf16 and lets the kernel
        # fuse RMSNorm + ROW quant online.
        # Default NONE keeps the model bit-identical to the un-quantized port.
        self.quant_scheme = getattr(c, "quant_scheme", QuantScheme.NONE)
        _is_fp8_ckpt = self.quant_scheme == QuantScheme.FP8_ROW
        # bf16-MLP diagnostic: fp8 checkpoint, but this MLP stays bf16
        # (block-dequant at load, plain bf16 forward). See _FP8_MLP_BF16.
        self.fp8_mlp_bf16_diag = _is_fp8_ckpt and _FP8_MLP_BF16
        self.is_fp8_row = _is_fp8_ckpt and not _FP8_MLP_BF16

        # Pre-MLP RMSNorm (HF (1+weight) form, folded in forward like the
        # layer-level input_layernorm; the RAW HF weight is loaded as-is).
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            self.hidden, self.rms_norm_eps, self.dtype
        )

        w_dtype = torch.float8_e4m3fn if self.is_fp8_row else self.dtype
        self.gate_proj_weight = nn.Parameter(
            torch.empty(self.hidden, self.intermediate_size_per_rank, dtype=w_dtype)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(self.hidden, self.intermediate_size_per_rank, dtype=w_dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_size_per_rank, self.hidden, dtype=w_dtype)
        )

        if self.is_fp8_row:
            # ROW weight scales: gate/up [128, I_pr], down [128, H] fp32 (128
            # broadcast partition-rows, one column per output channel). Held as
            # non-trainable nn.Parameters so they appear in named_parameters and
            # keep their fp32 dtype through load_weights' param-dtype cast.
            i_pr = self.intermediate_size_per_rank
            self.gate_proj_weight_scale = nn.Parameter(
                torch.empty(128, i_pr, dtype=torch.float32), requires_grad=False
            )
            self.up_proj_weight_scale = nn.Parameter(
                torch.empty(128, i_pr, dtype=torch.float32), requires_grad=False
            )
            self.down_proj_weight_scale = nn.Parameter(
                torch.empty(128, self.hidden, dtype=torch.float32), requires_grad=False
            )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        i_pr = self.intermediate_size_per_rank
        if self.is_fp8_row:
            # Block-FP8 -> ROW re-quant loaders (weight + fp32 scale from the same
            # [.weight, .weight_scale_inv] source pair; want_scale selects which).
            set_weight_loader(self.gate_proj_weight,
                              fp8_row_gate_up_loader(i_pr, self.world_size, want_scale=False))
            set_weight_loader(self.up_proj_weight,
                              fp8_row_gate_up_loader(i_pr, self.world_size, want_scale=False))
            set_weight_loader(self.down_proj_weight,
                              fp8_row_down_loader(i_pr, self.world_size, want_scale=False))
            set_weight_loader(self.gate_proj_weight_scale,
                              fp8_row_gate_up_loader(i_pr, self.world_size, want_scale=True))
            set_weight_loader(self.up_proj_weight_scale,
                              fp8_row_gate_up_loader(i_pr, self.world_size, want_scale=True))
            set_weight_loader(self.down_proj_weight_scale,
                              fp8_row_down_loader(i_pr, self.world_size, want_scale=True))
            return

        gate_up_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        down_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        if self.fp8_mlp_bf16_diag:
            # fp8 checkpoint bytes, bf16 resident weights: block-dequant the
            # [weight, weight_scale_inv] pair to bf16 then run the SAME sharding.
            gate_up_loader = fp8_block_dequant_bf16_loader(gate_up_loader)
            down_loader = fp8_block_dequant_bf16_loader(down_loader)
        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    def forward(self, hidden_states, positions=None, is_decode=False, rank=None):
        # positions/rank accepted for signature-compat with the MoE block's
        # decoder-layer call site; unused by the dense path.
        hidden_states = hidden_states.to(self.dtype)
        is_prefill = not is_decode

        # SP: the residual stream is token-sharded (T/world) on prefill; gather
        # the full sequence before the dense matmul, scatter the result back.
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        if self.is_fp8_row:
            # ROW fp8. gamma = (1 + weight): Qwen3_5RMSNorm folds the +1 in its
            # own forward, so we add it back when handing the raw gamma to the
            # kernel / manual path.
            #
            # Path split on the nkilib CTE/TKG boundary (B*S vs
            # TKG_BS_SEQLEN_THRESHOLD):
            #   * prefill / large-batch decode -> CTE. The nkilib NF.mlp(ROW)
            #     CTE path produces INCORRECT logits for this model's dims on
            #     trn2 (device-verified: the very first sampled token, which
            #     comes from prefill, is garbage token id -1131398458, while a
            #     manual dequant with the IDENTICAL fp8 weights + [128,out] ROW
            #     scales is accurate). So CTE uses a manual dequant-to-bf16
            #     SwiGLU. Weights stay resident as fp8 (the HBM saving that lets
            #     3xlarge hold longer context); only a transient bf16 view is
            #     materialized per projection.
            #   * small-batch decode (T <= threshold, e.g. T=1) -> TKG. This is a
            #     SEPARATE nkilib impl (mlp_tkg, not mlp_cte) that fuses RMSNorm +
            #     per-row activation quant online from a bf16 input; it never saw
            #     clean input before (prefill garbage poisoned the KV cache), so
            #     it is exercised here on the correct manual-CTE state.
            gamma = 1.0 + self.post_attention_layernorm.weight
            use_cte = is_prefill or hidden_states.shape[0] > TKG_BS_SEQLEN_THRESHOLD
            if use_cte:
                # Manual dequant (w_fp8 * per-out-channel ROW scale) -> bf16 SwiGLU.
                # Scale rows are identical broadcasts; row 0 = per-output-channel.
                normed = self.post_attention_layernorm(hidden_states).to(torch.bfloat16)
                gw = self.gate_proj_weight.to(torch.bfloat16) * self.gate_proj_weight_scale[0:1, :].to(torch.bfloat16)
                uw = self.up_proj_weight.to(torch.bfloat16) * self.up_proj_weight_scale[0:1, :].to(torch.bfloat16)
                dw = self.down_proj_weight.to(torch.bfloat16) * self.down_proj_weight_scale[0:1, :].to(torch.bfloat16)
                act = F.silu(normed @ gw) * (normed @ uw)
                output = (act @ dw).to(self.dtype)
            else:
                # TKG fuses norm + ROW quant online from the bf16 activation.
                output = NF.mlp(
                    hidden_states,
                    self.gate_proj_weight,
                    self.up_proj_weight,
                    self.down_proj_weight,
                    quantization_type=QuantizationType.ROW,
                    gate_w_scale=self.gate_proj_weight_scale,
                    up_w_scale=self.up_proj_weight_scale,
                    down_w_scale=self.down_proj_weight_scale,
                    norm_type=NormType.RMS_NORM,
                    ln_w=gamma.view(1, -1),
                    eps=self.rms_norm_eps,
                    output_dtype=self.dtype,
                )
        else:
            normed = self.post_attention_layernorm(hidden_states)
            output = NF.mlp(
                normed,
                self.gate_proj_weight,
                self.up_proj_weight,
                self.down_proj_weight,
            )

        if is_prefill:
            if self.world_size > 1:
                output = self.tp_group.reduce_scatter(output, dim=0)
        elif self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output.to(self.dtype)


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
        self.mlp = Qwen3_5DenseMLP(config)
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
        self.mlp = Qwen3_5DenseMLP(config)
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
        # On-device encoder-cache inputs (prefill-only vision merge).
        vision_embedding_blocks: tuple[torch.Tensor, ...] | None = None,
        vision_positions: torch.Tensor | None = None,
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

        # Vision embedding merge (prefill only; decode carries no vision inputs).
        # merge_vision_embeddings remaps GLOBAL vision_positions to this rank's SP
        # window using rank=self.rank (matches the embed_tokens scatter above and
        # the all_gather below). deepstack is empty for Qwen3.6-27B
        # (deepstack_visual_indexes == []), so the returned deepstack embeds are
        # None and ignored — no per-layer injection needed.
        if (
            is_prefill
            and vision_embedding_blocks is not None
            and vision_positions is not None
        ):
            hidden_states, _ = merge_vision_embeddings(
                hidden_states,
                vision_embedding_blocks,
                vision_positions,
                rank=self.rank,
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


class Qwen3_5ForConditionalGeneration(nn.Module, HasInnerState, IsHybrid, SupportsMRoPE, SupportsSpatialMerge, SupportsVisionWarmup):
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

        # ── Vision tower (image + video) ─────────────────────────────────
        # EPD construction-role flags read off the vision NeuronConfig; both
        # False = monolith (build both towers). The vision tower is only built
        # when the runner supplies a vision NeuronConfig (multimodal serving) and
        # this is not a language-only PD pool; text-only serving leaves
        # self.visual = None (zero behavior change vs the text-only port). The
        # text tower is always built — the hybrid recurrent-state plumbing
        # (get_kv_spec / bind_mamba_state / reset_recurrent_state) assumes
        # language_model is present, so full encoder-only tower-skipping is left
        # as a later EPD refinement.
        vnc = config.vision_config.neuron_config if config.vision_config else None
        self.mm_encoder_only = bool(vnc and vnc.mm_encoder_only)
        self.mm_language_model_only = bool(vnc and vnc.mm_language_model_only)
        if vnc is not None and not self.mm_language_model_only:
            self.visual = Qwen3VLVisionModel(
                config.vision_config, dtype=torch.bfloat16
            )
        else:
            self.visual = None

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
        vision_embedding_blocks: tuple[torch.Tensor, ...] | None = None,
        vision_positions: torch.Tensor | None = None,
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
            vision_embedding_blocks=vision_embedding_blocks,
            vision_positions=vision_positions,
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
        # compiler a provably in-bounds index and removes the OOBMode.ERROR DGE. Unconditional:
        # the unclamped form was the known-broken behaviour and is no longer selectable.
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
        # index_select (exactly one nonzero per output row). Unconditional: it replaces a plain
        # index_select, which this backend lowered to an indirect gather.
        _cols = torch.arange(n_rows, device=hidden_states.device)  # [n_rows]
        # [n_sample, n_rows] one-hot selection matrix (one nonzero per row)
        _P = (sampling_positions.to(torch.int32).unsqueeze(1) == _cols.unsqueeze(0)).to(
            hidden_states.dtype
        )
        hidden_states_for_logits = _P @ hidden_states  # [n_sample, hidden]
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

    # ── Vision Encoder (image + video) ───────────────────────────────────
    # Ported verbatim from qwen3_vl/model_bf16.py: the ViT and its packing/cache
    # utilities are shared, and Qwen3_5VisionConfig is field-compatible with the
    # Qwen3-VL vision config, so the only change is self.config.vision_config.

    def embed_multimodal(
        self,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        encoder_cache=None,
        mm_hashes: list[str] | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Encode images or videos and allocate+write into the on-device cache.

        Allocates cache blocks for each item based on its merged token count,
        then has the VE NEFF scatter-write directly into the cache buffer.
        No unpack step, no device→host transfer, no per-item split.

        Video reuses the image vision pipeline unchanged; only the kwarg names
        differ (pixel_values_videos / video_grid_thw). The runner supplies one
        modality pair per call, so the video pair is folded onto the image path.
        """
        from vllm_neuron.vllm.worker.encoder_cache_blocks import EncoderCacheBlocks

        # The runner groups multimodal kwargs by modality, so exactly one pair
        # is supplied per call. Enforce it: silently dropping one modality would
        # only surface as wrong output far downstream.
        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError(
                "embed_multimodal: cannot supply both pixel_values and "
                "pixel_values_videos in a single call; caller must group by "
                "modality."
            )
        # Video reuses the image path; fold it onto the image kwargs.
        is_video = pixel_values is None and pixel_values_videos is not None
        if is_video:
            pixel_values = pixel_values_videos
            image_grid_thw = video_grid_thw
        if pixel_values is None or image_grid_thw is None:
            raise ValueError(
                "embed_multimodal requires (pixel_values, image_grid_thw) or "
                "(pixel_values_videos, video_grid_thw)."
            )

        cache: EncoderCacheBlocks = encoder_cache
        vc = self.config.vision_config
        vnc = vc.neuron_config

        t_start = _time.perf_counter()

        head_dim = vc.hidden_size // vc.num_heads
        num_grid_per_side = int(vc.num_position_embeddings**0.5)
        merge_factor = vc.spatial_merge_size**2
        configured_block_size = vnc.vision_attention_block_size
        block_size = configured_block_size
        grid_rows = image_grid_thw.tolist()

        # A video is packed per FRAME, not as one T*H*W item: each [T,H,W] grid
        # row expands to T rows of [1,H,W] so its frames distribute across blocks
        # (and DP ranks). A frame must fit within one block (per_frame H*W <=
        # block_size); a block holds WHOLE frames plus a trailing pad; group_ids
        # keeps each video's frames in its own block run.
        if is_video:
            frames_per_item = image_grid_thw[:, 0]
            grid_for_ve = image_grid_thw.repeat_interleave(frames_per_item, dim=0)
            grid_for_ve[:, 0] = 1
            group_ids = torch.repeat_interleave(
                torch.arange(image_grid_thw.shape[0]), frames_per_item
            ).tolist()
            for t, h, w in grid_rows:
                if h * w > block_size:
                    raise ValueError(
                        f"vision_attention_block_size={block_size} is smaller "
                        f"than the per-frame token count {h * w} (H*W); a frame "
                        "must fit within one block so its attention is complete. "
                        "Increase the bucket/block size."
                    )
        else:
            grid_for_ve = image_grid_thw
            group_ids = None
            for t, h, w in grid_rows:
                if t * h * w > block_size:
                    raise ValueError(
                        f"vision_attention_block_size={block_size} is smaller "
                        f"than the per-image token count {t * h * w} (T*H*W); an "
                        "image must fit within one block so its attention is complete. "
                        "Increase the bucket/block size."
                    )

        tokens_per_image = grid_for_ve.prod(dim=1).tolist()
        total_tokens = sum(tokens_per_image)

        # 1. Allocate cache blocks per item (one mm_hash each). Images use
        #    allocate's dense default; a video packs WHOLE frames per block.
        cache_block_map: list[list[int]] = []
        for i, (t, h, w) in enumerate(grid_rows):
            raw_tokens = t * h * w
            num_merged = raw_tokens // merge_factor
            if is_video:
                per_frame_raw = h * w
                per_frame_merged = per_frame_raw // merge_factor
                frames_per_block = block_size // per_frame_raw
                num_item_blocks = math.ceil(t / frames_per_block)
                tokens_per_block: list[int] = []
                frames_left = t
                for _ in range(num_item_blocks):
                    k = min(frames_per_block, frames_left)
                    tokens_per_block.append(k * per_frame_merged)
                    frames_left -= k
                block_ids = cache.allocate(mm_hashes[i], tokens_per_block)
            else:
                block_ids = cache.allocate(
                    mm_hashes[i], cache.dense_tokens_per_block(num_merged, block_size)
                )
            cache_block_map.append(block_ids)

        # 2. Bucket selection (CPU). Whole-frame packing can need more blocks than
        #    ceil(total_tokens/block_size); size the bucket from the exact
        #    per-video block count so the warmed bucket has enough blocks.
        if group_ids is not None:
            required_blocks = sum(len(blocks) for blocks in cache_block_map)
            bucket_tokens = max(total_tokens, required_blocks * block_size)
        else:
            bucket_tokens = total_tokens
        _bucket, num_blocks = select_vision_bucket(
            bucket_tokens,
            vnc.num_vision_tokens_buckets,
            configured_block_size,
            dp_size=vnc.dp_size,
        )

        # 3. FFD packing with one-item-per-block constraint (CPU)
        assignment = ffd_pack_images(
            tokens_per_image,
            block_size,
            num_blocks,
            one_item_per_block=True,
            group_ids=group_ids,
        )

        # 4. CPU preprocessing
        cos, sin = compute_rotary_pos_emb(grid_for_ve, head_dim, vc.spatial_merge_size)
        pos_emb_idx, pos_emb_weight = compute_position_indices_and_weights(
            grid_for_ve, num_grid_per_side, vc.spatial_merge_size
        )
        bound_min, bound_max = compute_block_bounds(
            tokens_per_image, assignment, grid_for_ve
        )

        # 5. Scatter into block layout (CPU)
        packed_pixels = scatter_to_blocks(pixel_values, tokens_per_image, assignment)
        packed_cos = scatter_to_blocks(cos, tokens_per_image, assignment)
        packed_sin = scatter_to_blocks(sin, tokens_per_image, assignment)
        packed_idx = (
            scatter_to_blocks(pos_emb_idx.T, tokens_per_image, assignment)
            .permute(2, 0, 1)
            .contiguous()
        )
        packed_weight = (
            scatter_to_blocks(pos_emb_weight.T, tokens_per_image, assignment)
            .permute(2, 0, 1)
            .contiguous()
        )

        t_preprocess = _time.perf_counter()

        # 6. Build write_block_ids: map each VE output block to a cache block.
        flat_cache_blocks = [b for blocks in cache_block_map for b in blocks]
        write_block_ids_list = flat_cache_blocks + [cache.scratch_block_id] * (
            num_blocks - len(flat_cache_blocks)
        )
        write_block_ids = torch.tensor(write_block_ids_list, dtype=torch.int64)

        # 7. Move to device, dispatch VE in cache-write mode.
        with record_function_or_nullcontext("embed_multimodal: device_execution"):
            device = next(self.visual.parameters()).device
            self.visual(
                packed_pixels.to(device),
                packed_idx.to(device),
                packed_weight.to(device),
                packed_cos.to(device),
                packed_sin.to(device),
                bound_min.to(device),
                bound_max.to(device),
                cache.buffer,
                write_block_ids.to(device),
            )
        t_device = _time.perf_counter()

        logger.debug(
            "[embed_multimodal] %d items, %d tokens: "
            "cpu_preprocess=%.1fms, device_exec=%.1fms",
            len(tokens_per_image),
            total_tokens,
            (t_preprocess - t_start) * 1000,
            (t_device - t_preprocess) * 1000,
        )

    def build_vision_synthetic_inputs(
        self,
        bucket: int,
        vision_neuron_config: VisionNeuronConfig,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Construct shape-only tensors matching the vision encoder forward signature.

        Returns a kwargs dict that can be passed directly to model.visual(**kwargs).
        """
        vc = self.config.vision_config
        block_size = vision_neuron_config.vision_attention_block_size
        patch_dim = (
            vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
        )
        head_dim = vc.hidden_size // vc.num_heads

        # Pad num_blocks to be divisible by dp_size for the encoder's DP scatter.
        dp = vision_neuron_config.dp_size
        num_blocks = math.ceil(math.ceil(bucket / block_size) / dp) * dp

        return {
            "pixel_values": torch.zeros(
                num_blocks, block_size, patch_dim, dtype=torch.bfloat16, device=device
            ),
            "pos_emb_idx": torch.zeros(
                4, num_blocks, block_size, dtype=torch.int32, device=device
            ),
            "pos_emb_weight": torch.zeros(
                4, num_blocks, block_size, dtype=torch.bfloat16, device=device
            ),
            "cos": torch.zeros(
                num_blocks, block_size, head_dim, dtype=torch.float32, device=device
            ),
            "sin": torch.zeros(
                num_blocks, block_size, head_dim, dtype=torch.float32, device=device
            ),
            "bound_min": torch.zeros(
                num_blocks, block_size, 1, dtype=torch.int32, device=device
            ),
            "bound_max": torch.zeros(
                num_blocks, block_size, 1, dtype=torch.int32, device=device
            ),
        }

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
        """Return dtypes for (conv_state, ssm_state).

        The GDN recurrent (ssm) state is ALWAYS kept in bf16 for this port — we
        force the ssm dtype to "auto" (== model dtype, bf16) regardless of
        cache_config.mamba_ssm_cache_dtype. This is not a tunable: our custom GDN
        chunk-prefill / decode kernels are validated ONLY at a bf16 ssm state
        (which keeps the mamba page small enough that the hybrid block-size
        aligner stays at 512). An fp32 ssm state doubles the mamba page, pushes
        the attention block_size to 896, and drives the kernels outside their
        validated regime → decode diverges to out-of-vocab tokens.

        This matters because when the model is served under the
        `Qwen3_5ForConditionalGeneration` architecture (the multimodal / vision
        path), upstream vLLM's Qwen3_5ForConditionalGenerationConfig updater
        copies the checkpoint's text_config.mamba_ssm_dtype ("float32") into
        cache_config.mamba_ssm_cache_dtype. Forcing "auto" here neutralizes that
        so vision and text-only serving share one validated bf16 regime with no
        serve-time flag required. conv_state still honors mamba_cache_dtype.
        """
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype if hasattr(vllm_config, 'model_config') else torch.bfloat16,
            getattr(vllm_config, 'cache_config', None) and vllm_config.cache_config.mamba_cache_dtype or None,
            "auto",  # ssm state pinned to model dtype (bf16); see docstring — NOT tunable
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

        # FP8-ROW: modules the checkpoint stores as block-FP8 need BOTH the
        # ``.weight`` and its ``.weight_scale_inv`` as sources (the block-dequant
        # loaders consume the pair). Non-fp8 keeps the plain single-source string.
        fp8 = tc.quant_scheme == QuantScheme.FP8_ROW

        def _pair(hf_key):
            return [f"{hf_key}.weight", f"{hf_key}.weight_scale_inv"]

        # Embedding and LM head
        mappings["language_model.embed_tokens.weight"] = f"{HF_TEXT_PREFIX}.embed_tokens.weight"
        mappings["lm_head.weight"] = "lm_head.weight"
        mappings["language_model.norm.weight"] = f"{HF_TEXT_PREFIX}.norm.weight"

        for layer_id in range(tc.num_hidden_layers):
            hf_prefix = f"{HF_TEXT_PREFIX}.layers.{layer_id}"
            model_prefix = f"language_model.layers.{layer_id}"

            # Norms: input_layernorm at layer level; post_attention_layernorm
            # owned by the dense MLP block (mlp.post_attention_layernorm), so the
            # decoder layer does NOT pre-normalize before self.mlp.
            mappings[f"{model_prefix}.input_layernorm.weight"] = f"{hf_prefix}.input_layernorm.weight"
            mappings[f"{model_prefix}.mlp.post_attention_layernorm.weight"] = f"{hf_prefix}.post_attention_layernorm.weight"

            # Dense SwiGLU MLP (gate / up / down) — every layer. FP8-ROW: fp8
            # weight + fp32 ROW scale both derive from the same [weight,
            # weight_scale_inv] source pair (want_scale selects inside loader).
            if fp8:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    pair = _pair(f"{hf_prefix}.mlp.{proj}")
                    mappings[f"{model_prefix}.mlp.{proj}_weight"] = pair
                    # bf16-MLP diagnostic: bf16 weights → no ROW scale param.
                    if not _FP8_MLP_BF16:
                        mappings[f"{model_prefix}.mlp.{proj}_weight_scale"] = pair
            else:
                mappings[f"{model_prefix}.mlp.gate_proj_weight"] = f"{hf_prefix}.mlp.gate_proj.weight"
                mappings[f"{model_prefix}.mlp.up_proj_weight"] = f"{hf_prefix}.mlp.up_proj.weight"
                mappings[f"{model_prefix}.mlp.down_proj_weight"] = f"{hf_prefix}.mlp.down_proj.weight"

            if tc.layer_types[layer_id] == "full_attention":
                # Fused QKV (separate Q, K, V in checkpoint → fused in model).
                # FP8: v1 dequants QKV/o_proj to bf16 at load, so each source is
                # a [weight, weight_scale_inv] pair (q|k|v → 6 slices for qkv).
                if fp8:
                    mappings[f"{model_prefix}.self_attn.qkv_proj_weight"] = (
                        _pair(f"{hf_prefix}.self_attn.q_proj")
                        + _pair(f"{hf_prefix}.self_attn.k_proj")
                        + _pair(f"{hf_prefix}.self_attn.v_proj")
                    )
                    mappings[f"{model_prefix}.self_attn.o_proj_weight"] = _pair(
                        f"{hf_prefix}.self_attn.o_proj")
                else:
                    mappings[f"{model_prefix}.self_attn.qkv_proj_weight"] = [
                        f"{hf_prefix}.self_attn.q_proj.weight",
                        f"{hf_prefix}.self_attn.k_proj.weight",
                        f"{hf_prefix}.self_attn.v_proj.weight",
                    ]
                    mappings[f"{model_prefix}.self_attn.o_proj_weight"] = f"{hf_prefix}.self_attn.o_proj.weight"
                mappings[f"{model_prefix}.self_attn.q_norm.weight"] = f"{hf_prefix}.self_attn.q_norm.weight"
                mappings[f"{model_prefix}.self_attn.k_norm.weight"] = f"{hf_prefix}.self_attn.k_norm.weight"
            else:
                # Linear attention. FP8: in_proj_qkv / in_proj_z / out_proj are
                # block-FP8 (dequant-to-bf16 in v1) → source pair; in_proj_a/b,
                # conv1d, dt_bias, A_log, norm are bf16-stored → single source.
                if fp8:
                    mappings[f"{model_prefix}.linear_attn.in_proj_qkv_weight"] = _pair(
                        f"{hf_prefix}.linear_attn.in_proj_qkv")
                    mappings[f"{model_prefix}.linear_attn.in_proj_z_weight"] = _pair(
                        f"{hf_prefix}.linear_attn.in_proj_z")
                    mappings[f"{model_prefix}.linear_attn.out_proj_weight"] = _pair(
                        f"{hf_prefix}.linear_attn.out_proj")
                else:
                    mappings[f"{model_prefix}.linear_attn.in_proj_qkv_weight"] = f"{hf_prefix}.linear_attn.in_proj_qkv.weight"
                    mappings[f"{model_prefix}.linear_attn.in_proj_z_weight"] = f"{hf_prefix}.linear_attn.in_proj_z.weight"
                    mappings[f"{model_prefix}.linear_attn.out_proj_weight"] = f"{hf_prefix}.linear_attn.out_proj.weight"
                mappings[f"{model_prefix}.linear_attn.in_proj_a_weight"] = f"{hf_prefix}.linear_attn.in_proj_a.weight"
                mappings[f"{model_prefix}.linear_attn.in_proj_b_weight"] = f"{hf_prefix}.linear_attn.in_proj_b.weight"
                mappings[f"{model_prefix}.linear_attn.conv1d_weight"] = f"{hf_prefix}.linear_attn.conv1d.weight"
                mappings[f"{model_prefix}.linear_attn.norm.weight"] = f"{hf_prefix}.linear_attn.norm.weight"
                mappings[f"{model_prefix}.linear_attn.dt_bias"] = f"{hf_prefix}.linear_attn.dt_bias"
                mappings[f"{model_prefix}.linear_attn.A_log"] = f"{hf_prefix}.linear_attn.A_log"

        # Load checkpoint. strict=False so the text loader skips any parameter
        # without a mapping here — notably the vision tower's ``visual.*`` params
        # when a vision NeuronConfig built self.visual; those are loaded
        # separately below via self.visual.load_weights (own vision TP group).
        # For text-only serving self.visual is None, so this is a no-op change.
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device, strict=False
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

        # Vision encoder weights (loads on its own vision TP group, CPU-side).
        # Only present for multimodal serving; text-only keeps self.visual None,
        # so this branch is a no-op and the text-only path is unchanged.
        if getattr(self, "visual", None) is not None:
            self.visual.load_weights(checkpoint_path, device="cpu", cpu_mode=True)
