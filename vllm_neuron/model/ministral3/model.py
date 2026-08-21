# SPDX-License-Identifier: Apache-2.0
"""
Ministral3 BF16 Implementation
====================================

Annotated implementation of Mistral AI's Ministral3 architecture for the Neuron
backend (e.g. Devstral-2-123B-Instruct-2512). Ported from the canonical Llama
dense model.

Supported parallelism: TP, SP, DP.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Ministral3-specific. Change when porting.

Key differences vs the Llama template:
  - RoPE uses YaRN scaling (transformers `_compute_yarn_parameters`) instead of
    Llama3 piecewise scaling. The *rotation* style stays interleaved
    (rotate_half), matching the Mistral/Llama checkpoint convention.
  - Untied embeddings (separate lm_head weight).
  - FP8 (per-tensor static) checkpoint: the seven linear projections are stored
    in FP8 with a scalar `weight_scale_inv`. Weight loaders dequantize to BF16
    at load time. When the checkpoint is plain BF16 (e.g. the tiny smoke-test
    model), the loaders fall back to the generic BF16 path automatically.
"""

import logging
import math
import os

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.functional.attention.attention_decode import (
    _swizzle_packed_k,
    _unswizzle_packed_k,
)
from vllm_neuron.functional.attention.attention_decode_mask import (
    _resize_block_len as _mask_resize_block_len,
)
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    with_rank_override,
)

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler

import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

from nkilib.core.utils.common_types import QuantizationType

from .config import Ministral3Config
from .weight_loaders import (
    _ACT_SCALE_TRN2_RECAL,
    adaptive_fused_qkv_weight_loader,
    adaptive_sharding_weight_loader,
    fp8_native_fused_qkv_weight_loader,
    fp8_native_sharding_weight_loader,
)

logger = logging.getLogger(__name__)


def _mlp_quant_mode() -> str:
    """FP8 MLP の量子化モード: ``static``（既定）または ``row``（per-token 動的）。

    ``MINISTRAL3_MLP_QUANT_MODE`` で切り替えます。既定は ``static`` = 従来と byte 同一。
    ``row`` を選ぶと、nkilib が入力の per-token absmax をデバイス上で取り（``QuantizationType.ROW``、
    ``nkilib/core/mlp/mlp_torch.py:423``）、**校正済みの入力 scale を一切使いません**。
    ACCURACY_CONSTRAINTS §2.4 の「正しい scale は rank/token ごとに違う」という読みの検定用。
    """
    return (os.environ.get("MINISTRAL3_MLP_QUANT_MODE") or "static").strip().lower()

# FP8 weights are stored/computed as e4m3fn; on gen3/TRN2 the framework
# reinterprets these bytes under the native ±240 e4m3 table (weights are
# byte-saturated onto that grid at load — see weight_loaders.py).
_FP8_DTYPE = torch.float8_e4m3fn
# gen3/TRN2 native e4m3 saturates at ±240 (NOT OCP ±448). Activations quantized
# in-framework must clip to this grid to match the kernel's internal clip.
_FP8_E4M3_MAX = 240.0


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


# >>> L13: fused RMSNorm into the QKV/MLP kernels (opt-in, default OFF) <<<
#
# The standalone ``input_layernorm``/``post_attention_layernorm`` modules run as
# framework-compiled element-wise passes before ``self_attn``/``mlp``. Both
# kernels can absorb the norm internally (``NF.mlp(norm_type=RMS_NORM, ln_w=…)``
# and ``NF.attention_decode(rmsnorm_X_enabled=True, rmsnorm_X_gamma=…)``), which
# removes that separate pass.
#
# Gates (all default OFF, so an unset environment is byte-identical to baseline):
#   VLLM_NEURON_FUSED_RMSNORM              master — enables ATTN + MLP, both modes
#   VLLM_NEURON_FUSED_RMSNORM_{ATTN,MLP}   independent per-site sub-gates
#   VLLM_NEURON_FUSED_RMSNORM_DECODE_ONLY  RECOMMENDED — decode only
#
# Use ``_DECODE_ONLY``. Fusing the *prefill* norm costs ~15% TTFT: the standalone
# prefill norm sits in a between-collective gap the compiler already overlaps,
# and moving it inside the kernel adds a hard sync boundary (the preceding
# AllGather must fully drain before the kernel's internal norm can start).
# Decode has no such structure, so the fusion is a clean win there.
#
# The four derived per-mode flags below are what the call sites read.
_FUSED_RMSNORM_MASTER = _env_flag("VLLM_NEURON_FUSED_RMSNORM")
_FUSED_RMSNORM_DECODE_ONLY = _env_flag("VLLM_NEURON_FUSED_RMSNORM_DECODE_ONLY")
_FUSED_RMSNORM_ATTN_ANY = (
    _FUSED_RMSNORM_MASTER
    or _FUSED_RMSNORM_DECODE_ONLY
    or _env_flag("VLLM_NEURON_FUSED_RMSNORM_ATTN")
)
_FUSED_RMSNORM_MLP_ANY = (
    _FUSED_RMSNORM_MASTER
    or _FUSED_RMSNORM_DECODE_ONLY
    or _env_flag("VLLM_NEURON_FUSED_RMSNORM_MLP")
)
# DECODE_ONLY forces prefill off even if the master/sub-gates are also set.
_FUSED_RMSNORM_ATTN_PREFILL = _FUSED_RMSNORM_ATTN_ANY and not _FUSED_RMSNORM_DECODE_ONLY
_FUSED_RMSNORM_MLP_PREFILL = _FUSED_RMSNORM_MLP_ANY and not _FUSED_RMSNORM_DECODE_ONLY
_FUSED_RMSNORM_ATTN_DECODE = _FUSED_RMSNORM_ATTN_ANY
_FUSED_RMSNORM_MLP_DECODE = _FUSED_RMSNORM_MLP_ANY

# >>> In-kernel decode KV write (opt-in, default OFF) <<<
#
# VLLM_NEURON_DECODE_INKERNEL_KV_WRITE=1 asks the decode megakernel to write the new
# K/V into the cache itself (`update_cache=True` + `kv_cache_update_idx`) instead of
# returning the new tokens for this module to scatter afterwards.
#
# WHY: with a packed FP8 K cache the out-of-kernel scatter cannot address the swizzled
# layout, so it un-swizzles the WHOLE K cache, writes one token, and re-swizzles it back
# — every layer, every step. Device profiling (2026-08-03) measured the cost of that round
# trip at +19.96 ms of COPY per decode execution, 79% of the +25.31 ms that enabling packed
# FP8 KV costs. The kernel writes the packed layout natively, so this gate removes the round
# trip rather than optimising it. `gpt_oss/model_bf16.py` already uses this form.
#
# Off by default until the on-device A/B lands: correctness depends on the FX aliasing pass
# threading the in-kernel write back to self.k_cache/self.v_cache, which only device
# execution can confirm.
_DECODE_INKERNEL_KV_WRITE = _env_flag("VLLM_NEURON_DECODE_INKERNEL_KV_WRITE")


def _packed_fp8_viable_for_bucket(
    block_len: int, bs: int, q_head: int, s_active: int, s_prior: int
) -> bool:
    """Whether the packed FP8 decode kernel is usable for this bucket geometry.

    The decode kernel resizes block_len so blocks_per_batch is a multiple of
    (lnc * p_max). The packed layout pairs two consecutive tokens per BF16 slot,
    so it needs the resized block_len to stay >= 2; some bucket geometries
    (small SWA windows, or batch=1 which forces s_prior sharding) resize it down
    to 1. Mirror the kernel's resize math (shared replica) to decide statically,
    per compiled decode NEFF, whether to read the packed cache directly or
    un-swizzle it to the standard layout first.

    Kept model-local on purpose: upstream has since promoted this to
    ``functional/attention/attention_decode.packed_fp8_viable_for_bucket``, but on
    this fork's base it only exists inside ``gpt_oss/model_bf16.py``. Importing
    across model families (or moving the shared copy) would edit another model's
    deliverable, so this port carries its own replica of the same logic. When the
    fork rebases onto a version that ships the shared helper, delete this and
    import that one instead.
    """
    if block_len <= 0 or s_prior <= 0:
        return False
    return _mask_resize_block_len(block_len, bs, q_head, s_active, s_prior) >= 2


def _fused_norm_mlp_kwargs(
    norm_weight: torch.Tensor | None, eps: float | None
) -> dict:
    """L13 kwargs for ``NF.mlp``, or ``{}`` when the fusion is off.

    Returning an empty dict (rather than ``norm_type=NO_NORM, ln_w=None``) keeps
    the gate-OFF call byte-identical to the pre-L13 baseline.
    """
    if norm_weight is None:
        return {}
    from nkilib.core.utils.common_types import NormType

    return {
        "norm_type": NormType.RMS_NORM,
        # nkilib wants gamma as [1, H].
        "ln_w": norm_weight.unsqueeze(0),
        "eps": eps,
    }


def _fused_norm_qkv_kwargs(
    norm_weight: torch.Tensor | None, eps: float | None
) -> dict:
    """L13 kwargs for ``NF.qkv_proj`` (prefill), or ``{}`` when the fusion is off.

    ``qkv_proj`` names the gamma ``norm_weights`` (vs ``NF.mlp``'s ``ln_w``).
    """
    if norm_weight is None:
        return {}
    from nkilib.core.utils.common_types import NormType

    return {
        "norm_type": NormType.RMS_NORM,
        "norm_weights": norm_weight.unsqueeze(0),
        "eps": eps,
    }


def _row_quantize_activation_to_fp8(hidden: torch.Tensor) -> torch.Tensor:
    """Per-token (ROW) quantize a BF16 activation into nkilib's packed CTE ROW layout.

    Returns ``[B, S, H+4]`` fp8: the first ``H`` columns are the quantized values and the
    last 4 are the fp32 per-row scale **bit-cast** into 4 fp8 bytes. That layout is the
    kernel's documented interface for CTE + ``QuantizationType.ROW``
    (``mlpp_input_has_packed_scale``; the decoder is
    ``nkilib/core/mlp/mlp_torch.py::_extract_precomputed_row_scale``).

    🔴 WHY WE MUST PRE-QUANTIZE. Passing a BF16 hidden with ROW does NOT work on gen3: the
    CTE kernel fuses input-quant into the PE transpose and asserts
    ``nc_matmul (transpose mode) dst dtype must match input dtype on gen3+, got
    dst=float8_e4m3 but input=bfloat16`` (measured 2026-08-18, all 32 ranks). Only a
    pre-quantized fp8 hidden makes ``mlpp_has_quantized_input`` True, and for ROW that
    hidden must additionally carry the packed scale — hence the 4 extra bytes. The earlier
    ``NKILIB_MLP_BF16_XPOSE_SRC`` escape hatch is NOT present in this nkilib.

    The formula mirrors ``_row_quantize`` exactly: ``scale = max(amax / 240, 1e-5)`` and
    ``q = x / scale``, with 240 (not OCP's 448) because TRN2's native e4m3 saturates there.
    """
    h32 = hidden.to(torch.float32)
    amax = h32.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax / _FP8_E4M3_MAX, min=1e-5)          # [..., 1] f32
    q = (h32 / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(_FP8_DTYPE)
    # fp32 scale -> 4 raw bytes -> reinterpret as 4 fp8 values (a pure bit-cast; the kernel
    # bit-casts them back). ``view(dtype)`` needs the last dim contiguous.
    # ⚠️ Both views must be materialised. On the Neuron/XLA backend a dtype-``view`` result is
    # not itself a first-class XLA tensor, and ``torch.cat`` then rejects it with
    # "Expected all tensors in the given list to be XLA tensors ... Got: XLAFloat8_e4m3fnType"
    # (measured 2026-08-18). ``.contiguous()`` after the bit-cast forces a real tensor.
    scale_bytes = (
        scale.contiguous().view(torch.uint8).contiguous().view(_FP8_DTYPE).contiguous()
    )  # [..., 4]
    return torch.cat([q.contiguous(), scale_bytes], dim=-1)


def _quantize_activation_to_fp8(
    hidden: torch.Tensor, dequant_scale: torch.Tensor
) -> torch.Tensor:
    """Quantize a BF16 activation to Neuron e4m3 using a per-tensor DEQUANT scale.

    Mirrors the nkilib STATIC formula exactly (fp8_quantize.py): the kernel does
    ``quant = 1/dequant_scale; x_fp8 = clip(x * quant, ±MAXVAL)``. We reproduce it
    so the result is numerically identical to letting the kernel quantize the
    input internally — the only difference is *where* it happens.

    This exists to work around the gen3 MLP-CTE source-transpose dtype bug: the
    prefill kernel fuses input-quant into the PE transpose (``dst=fp8`` while the
    transposed hidden source is bf16), which gen3 rejects (``nc_matmul transpose
    dst dtype must match input dtype``). Feeding an already-fp8 hidden makes
    ``mlpp_has_quantized_input`` True → the transpose is fp8→fp8 → no mismatch.
    This is the kernel's documented "PREFILL supports quantized input" path
    (mlp.py docstring). ``dequant_scale`` is a [128,1] f32 scalar buffer; only
    element [0,0] matters (per-tensor).
    """
    quant = (1.0 / dequant_scale.flatten()[0]).to(torch.float32)
    scaled = (hidden.to(torch.float32) * quant).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return scaled.to(_FP8_DTYPE)


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: Ministral3 uses standard RMSNorm (no hidden dim padding)
# =============================================================================
class Ministral3RMSNorm(nn.Module):
    """Standard RMS Normalization."""

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
# Section 2: Rotary Position Embedding (YaRN)
# <-- MODEL-SPECIFIC: Ministral3 uses YaRN scaling for RoPE, with the standard
# interleaved (rotate_half) rotation style used by Mistral/Llama checkpoints.
# =============================================================================


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """<-- MODEL-SPECIFIC: Interleaved RoPE (rotate_half style).

    Mistral/Llama use rotate_half: split into first/second half, rotate as
    (-x2, x1). This must match the checkpoint — using the GPT-OSS split-half
    style here would produce garbage attention outputs.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: [Nh, T, Dh], k: [Nkv, T, Dh]
        cos: [T, Dh], sin: [T, Dh]   (full head_dim — duplicated halves)

    Returns:
        Rotated (q, k) tensors
    """
    cos = cos.unsqueeze(0)  # [1, T, Dh]
    sin = sin.unsqueeze(0)  # [1, T, Dh]
    return _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)


class Ministral3RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with YaRN scaling.

    <-- MODEL-SPECIFIC: This mirrors transformers' `_compute_yarn_parameters`
    (the source of truth for Ministral3). The YaRN inverse-frequency blending
    (interpolation vs extrapolation via a linear ramp) and the cos/sin
    attention_factor (mscale) are model-specific.
    """

    def __init__(self, config: Ministral3Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        rope = config.rope_scaling or {}

        self.factor = rope.get("factor", 1.0)
        self.beta_fast = rope.get("beta_fast", 32.0)
        self.beta_slow = rope.get("beta_slow", 1.0)
        self.orig_max_position = (
            rope.get("original_max_position_embeddings")
            or config.max_position_embeddings
        )
        self.truncate = rope.get("truncate", True)

        # attention_factor: explicit override, else derive from factor (+ optional
        # mscale / mscale_all_dim), exactly as transformers does.
        attention_factor = rope.get("attention_factor")
        mscale = rope.get("mscale")
        mscale_all_dim = rope.get("mscale_all_dim")
        if attention_factor is None:
            if mscale and mscale_all_dim:
                attention_factor = float(
                    self._get_mscale(self.factor, mscale)
                    / self._get_mscale(self.factor, mscale_all_dim)
                )
            else:
                attention_factor = self._get_mscale(self.factor)
        self.attention_factor = attention_factor
        # NOTE: inv_freq is computed fresh on-device in forward() (llama3 pattern),
        # not cached as a buffer — caching + .to(device) breaks graph capture.

    @staticmethod
    def _get_mscale(scale: float, mscale: float = 1.0) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        """<-- MODEL-SPECIFIC: YaRN frequency computation (transformers parity)."""
        base = self.rope_theta
        dim = self.head_dim

        def find_correction_dim(num_rotations):
            return (
                dim * math.log(self.orig_max_position / (num_rotations * 2 * math.pi))
            ) / (2 * math.log(base))

        low = find_correction_dim(self.beta_fast)
        high = find_correction_dim(self.beta_slow)
        if self.truncate:
            low = math.floor(low)
            high = math.ceil(high)
        low = max(low, 0)
        high = min(high, dim - 1)

        # Linear ramp over dim//2 (guard the singularity exactly as transformers does)
        ramp_min, ramp_max = low, high
        if ramp_min == ramp_max:
            ramp_max += 0.001
        linear = (
            torch.arange(dim // 2, dtype=torch.float32, device=device) - ramp_min
        ) / (ramp_max - ramp_min)
        ramp = torch.clamp(linear, 0, 1)
        inv_freq_extrapolation_factor = 1 - ramp

        pos_freqs = base ** (
            torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.factor * pos_freqs)

        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation * inv_freq_extrapolation_factor
        )
        return inv_freq

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin embeddings for given positions.

        Returns:
            cos, sin: both of shape [T, head_dim] (halves duplicated to match
            the interleaved rotate_half application).
        """
        # Compute inv_freq fresh on the target device (llama3 pattern). Do NOT
        # cache a CPU buffer and .to(device) here — during torch.compile graph
        # capture that triggers `_copy_from xla→neuron` (NotImplementedError).
        inv_freq = self._compute_inv_freq(device)  # [Hd/2]
        inv_freq_expanded = inv_freq[:, None].float()  # [Hd/2, 1]
        positions_expanded = position_ids[None, :].float()  # [1, T]
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, Hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, Hd]
        cos = emb.cos() * self.attention_factor
        sin = emb.sin() * self.attention_factor
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (GQA config, no bias, no sinks, no sliding window)
# =============================================================================


class Ministral3Attention(nn.Module):
    """Multi-head attention with TP head sharding.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads sharded across TP ranks; KV replicated under GQA when fewer
      than TP size
    - Prefill: all-gather input -> QKV proj -> attention -> O proj -> reduce-scatter
    - Decode: fused megakernel with TP all-reduce

    <-- MODEL-SPECIFIC:
    - GQA with separate Q (96) and KV (8) head counts
    - No attention bias, no sinks, no sliding window
    - YaRN RoPE (interleaved rotate_half application)
    - FP8 checkpoint weights (dequantized to BF16 at load)
    """

    def __init__(self, config: Ministral3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5

        # <-- MODEL-SPECIFIC: FP8-native dense projections (opt-in, per-family).
        # When enabled, q/k/v (fused) and/or o_proj keep their FP8 weights and run
        # FP8×FP8 STATIC matmuls; otherwise they are dequantized to BF16 at load.
        # 🔬 layer_idx is passed so `MINISTRAL3_FP8_LAYERS` can restrict FP8 to a
        # layer range (root-cause bisection, config.py). Unset env → all layers.
        self._fp8_qkv = config.fp8_enabled("qkv", layer_idx)
        self._fp8_o = config.fp8_enabled("o_proj", layer_idx)

        # >>> PARALLELISM: TP group setup <<<
        from vllm.distributed.parallel_state import get_dcp_group
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_dcp_kv_group,
            get_neuron_dcp_tp_group,
        )

        dcp_group = get_dcp_group()
        self.dcp_size = dcp_group.world_size
        self.dcp_group = dcp_group if self.dcp_size > 1 else None

        dcp_kv_group = get_neuron_dcp_kv_group()
        self.cp_kv_group = dcp_kv_group

        # 2.32 PORT: vllm-neuron 0.24.x removed `apply_prefill_dcp` from
        # NeuronConfig -- 2.32's own models (see llama3/model.py) no longer take
        # prefill DCP from a config flag at all, they derive DCP purely from the
        # dcp group's world_size. Read it defensively so this package works on
        # both 0.21.x (flag present) and 0.24.x (flag absent), where absent means
        # "no prefill DCP", i.e. the pre-existing default on 0.21.x too.
        self.apply_prefill_dcp = (
            getattr(config.neuron_config, "apply_prefill_dcp", False)
            if config.neuron_config
            else False
        )
        if self.apply_prefill_dcp and self.dcp_size > 1:
            dcp_tp_group = get_neuron_dcp_tp_group()
            self.tp_group = dcp_tp_group
            self.world_group = get_tp_group()
            self.world_size = dcp_tp_group.world_size
            self.rank = dcp_tp_group.rank_in_group
        else:
            self.tp_group = get_tp_group()
            self.world_size = self.tp_group.world_size
            self.rank = self.tp_group.rank_in_group

        full_tp_rank = get_tp_group().rank_in_group
        full_tp_size = get_tp_group().world_size
        tp_pair_size = full_tp_size // max(self.dcp_size, 1)
        self.cp_chunk_idx = full_tp_rank // tp_pair_size if self.dcp_size > 1 else 0

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        self.cp_kv_cache_interleave_size = getattr(
            vllm_config.parallel_config, "cp_kv_cache_interleave_size", 1
        )

        # >>> PARALLELISM: Dependent DP setup (decode-only Q/O sharding across DP) <<<
        self.attention_dp_size = (
            config.neuron_config.attention_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_dp_group,
            get_neuron_attention_dp_rank,
        )

        self.attention_dp_group = get_neuron_attention_dp_group()
        self.attention_dp_rank = get_neuron_attention_dp_rank()

        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_tp_group,
        )

        self.attn_tp_group = get_neuron_attention_tp_group()

        effective_q_shards = self.world_size * self.attention_dp_size

        # >>> PARALLELISM: Head sharding calculation <<<
        self.num_attention_heads_per_rank = (
            self.num_attention_heads // effective_q_shards
        )

        self.kv_needs_a2a = (
            self.attention_dp_size > 1
            and self.num_key_value_heads > self.world_size
            and self.num_key_value_heads % effective_q_shards == 0
        )

        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1

        num_kv_heads_for_weight = (
            self.num_key_value_heads // effective_q_shards
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )
        # The fused QKV weight's KV segment is sized by num_kv_heads_for_weight
        # (= per-rank heads, or the A2A-sharded count under attention-DP). The FP8
        # STATIC kernel uses num_kv_heads to locate the Q|K|V scale-column
        # boundaries, so it must match the *weight* layout, not num_kv_heads_per_rank
        # (which differs when kv_needs_a2a). Store it for the qkv_proj call site.
        self.num_kv_heads_for_weight = num_kv_heads_for_weight

        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // num_kv_heads_for_weight
        )

        self.num_q_heads_after_a2a = (
            self.num_attention_heads_per_rank * self.attention_dp_size
        )
        self.num_kv_heads_after_a2a = (
            num_kv_heads_for_weight * self.attention_dp_size
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )

        # >>> PARALLELISM: QKV weight shapes <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = num_kv_heads_for_weight * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in_features = (
            self.num_attention_heads * self.head_dim
        ) // effective_q_shards

        qkv_dtype = _FP8_DTYPE if self._fp8_qkv else self.dtype
        o_dtype = _FP8_DTYPE if self._fp8_o else self.dtype
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=qkv_dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=o_dtype)
        )

        # >>> FP8 STATIC scale buffers (non-persistent; filled in load_weights). <<<
        # Shapes target the decode TKG kernel ([PMAX,·] = [128,·]); the prefill CTE
        # kernel accepts the same [128,·] (it also allows [1,·]). qkv weight scale is
        # [128,3] (columns Q|K|V, matching the fused weight); input scale [128,1].
        if self._fp8_qkv:
            self.register_buffer("qkv_w_scale", None, persistent=False)
            self.register_buffer("qkv_in_scale", None, persistent=False)
        if self._fp8_o:
            self.register_buffer("o_proj_w_scale", None, persistent=False)
            self.register_buffer("o_proj_in_scale", None, persistent=False)
            # ROW (per-token dynamic) variant: the kernel wants the weight dequant scale as
            # [P_MAX, H] instead of [P_MAX, 1]. Populated in load_weights only when
            # OPROJ_QUANT_MODE=row, so the default path allocates nothing extra.
            self.register_buffer("o_proj_w_scale_row", None, persistent=False)

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        self.k_cache = None
        self.v_cache = None

        # Packed (swizzled) FP8 K cache layout. Auto-detected in bind_kv_cache()
        # from the bound cache rank — packed K carries one dim more than V. It is
        # opt-in server-side via neuron_config.fp8_packed_kv; declaring the
        # attribute here is what tells bind_kv_cache this path is packed-aware.
        #
        # Why it matters: attention_segmented_cte CANNOT read a NON-packed FP8 K
        # cache (see validate_fp8_segmented_supported), so FP8 KV + segmented
        # prefill — and therefore FP8 KV + APC — is only reachable with the
        # packed layout.
        self.fp8_packed = False

        self.register_buffer("k_scale", None, persistent=False)
        self.register_buffer("v_scale", None, persistent=False)
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach weight loaders for checkpoint -> parameter transformation.

        >>> PARALLELISM: Weight loaders handle TP sharding of checkpoint tensors.
        <-- MODEL-SPECIFIC: Ministral3 stores separate Q, K, V weights (no bias).
        When the checkpoint is FP8, the FP8 loaders dequantize each projection
        with its per-tensor scale before fusion/sharding.
        """
        ddp = self.attention_dp_size
        effective_q_rank = self.attention_dp_rank + self.rank * ddp
        o_shard_size = (self.num_attention_heads * self.head_dim) // (
            self.world_size * ddp
        )

        # FP8-native families keep their fp8 weights (fuse/shard + ±240 saturation,
        # scales loaded separately). Other families use the adaptive loaders, which
        # dequantize FP8 to BF16 when scale slices are present, else behave as the
        # generic BF16 loaders (format detected per-parameter from the checkpoint
        # slice count, not from config — robust to --hf-overrides emptying
        # quantization_config).
        if self._fp8_qkv:
            qkv_loader = fp8_native_fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
                attention_dp_rank=self.attention_dp_rank,
                attention_dp_size=ddp,
                kv_sharded_across_attention_dp=self.kv_needs_a2a,
            )
        else:
            qkv_loader = adaptive_fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
                attention_dp_rank=self.attention_dp_rank,
                attention_dp_size=ddp,
                kv_sharded_across_attention_dp=self.kv_needs_a2a,
            )

        o_base = (
            fp8_native_sharding_weight_loader if self._fp8_o
            else adaptive_sharding_weight_loader
        )
        o_loader = with_rank_override(
            o_base(
                shard_dim=0,
                shard_size=o_shard_size,
                num_shards=self.world_size * ddp,
                is_storage_transposed=True,
            ),
            rank=effective_q_rank,
        )

        set_weight_loader(self.qkv_proj_weight, qkv_loader)
        set_weight_loader(self.o_proj_weight, o_loader)

    # ── Forward dispatch ─────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        fused_norm_weight: torch.Tensor | None = None,
        fused_norm_eps: float | None = None,
    ):
        """Dispatch to prefill or decode path based on metadata.

        ``fused_norm_weight``/``fused_norm_eps`` are L13: when supplied, the
        pre-attention RMSNorm runs inside the QKV kernel and the caller must NOT
        have applied ``input_layernorm`` itself.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
                fused_norm_weight=fused_norm_weight,
                fused_norm_eps=fused_norm_eps,
            )
        else:
            # >>> PARALLELISM: All-gather from SP before attention <<<
            if self.apply_prefill_dcp and self.dcp_size > 1:
                hidden_states = self.world_group.all_gather(hidden_states, dim=0)
                I = self.cp_kv_cache_interleave_size
                W = self.dcp_size
                R = self.cp_chunk_idx
                S, H = hidden_states.shape
                hidden_states = hidden_states.view(S // (W * I), W, I, H)[
                    :, R, :, :
                ].reshape(-1, H)
                cos, sin = position_embeddings
                Dc = cos.shape[1]
                cos = cos.view(S // (W * I), W, I, Dc)[:, R, :, :].reshape(-1, Dc)
                sin = sin.view(S // (W * I), W, I, Dc)[:, R, :, :].reshape(-1, Dc)
                position_embeddings = (cos, sin)
                positions = positions.view(S // (W * I), W, I)[:, R, :].reshape(-1)
            elif self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

            return self.forward_prefill(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
                fused_norm_weight=fused_norm_weight,
                fused_norm_eps=fused_norm_eps,
            )

    # ── Prefill path ─────────────────────────────────────────────────────

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        fused_norm_weight: torch.Tensor | None = None,
        fused_norm_eps: float | None = None,
    ) -> torch.Tensor:
        """Prefill: full-sequence attention with flash attention."""
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # ── Step 1: QKV Projection ───────────────────────────────────────
        if self._fp8_qkv:
            # FP8 STATIC: the kernel needs the Q|K|V segment sizes (d_head,
            # num_q_heads, num_kv_heads) to apply the per-segment [.,3] weight
            # scale, plus the per-tensor input scale.
            qkv = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=None,
                d_head=self.head_dim,
                num_q_heads=self.num_attention_heads_per_rank,
                num_kv_heads=self.num_kv_heads_for_weight,
                quantization_type=QuantizationType.STATIC,
                qkv_w_scale=self.qkv_w_scale,
                qkv_in_scale=self.qkv_in_scale,
                **_fused_norm_qkv_kwargs(fused_norm_weight, fused_norm_eps),
            ).squeeze(0)
        else:
            qkv = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=None,
                **_fused_norm_qkv_kwargs(fused_norm_weight, fused_norm_eps),
            ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )

        # ── Step 2: Apply RoPE ───────────────────────────────────────────
        # <-- MODEL-SPECIFIC: YaRN RoPE (interleaved rotate_half, externally computed)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ── Step 3: Update KV Cache ─────────────────────────────────────
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        def _write_kv_cache(k_to_cache, v_to_cache, slot_map):
            blk_idx = slot_map // block_size
            pos_idx = slot_map % block_size
            num_slots = slot_map.shape[0]
            nkh = self.num_key_value_heads_per_rank

            from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX

            if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                k_f = (
                    (k_to_cache.reshape(-1, self.head_dim) * self.k_scale)
                    .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                    .to(self.k_cache.dtype)
                )
                v_f = (
                    (v_to_cache.reshape(-1, self.head_dim) * self.v_scale)
                    .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                    .to(self.k_cache.dtype)
                )
            else:
                k_f = k_to_cache.reshape(-1, self.head_dim).to(self.k_cache.dtype)
                v_f = v_to_cache.reshape(-1, self.head_dim).to(self.v_cache.dtype)

            h_idx = torch.arange(
                nkh, dtype=torch.long, device=k_to_cache.device
            ).repeat_interleave(num_slots)
            index = (blk_idx.repeat(nkh), h_idx, pos_idx.repeat(nkh))
            # V is never packed → scatters directly.
            self.v_cache.index_put_(index, v_f)
            if self.fp8_packed:
                # Packed K cache [blocks, Nkh, block_size // 2, Dh, 2]: the
                # swizzle interleaves adjacent token positions into the trailing
                # size-2 dim, so a per-token scatter is not expressible directly
                # against it. Un-swizzle to the standard layout, scatter, then
                # re-swizzle in place (matching the decode kernel's write).
                k_unpacked = _unswizzle_packed_k(self.k_cache)
                k_unpacked.index_put_(index, k_f)
                self.k_cache.copy_(_swizzle_packed_k(k_unpacked))
            else:
                self.k_cache.index_put_(index, k_f)

        # ── Step 4: Write KV cache + Attention ────────────────────────────
        # <-- MODEL-SPECIFIC: No sinks, no sliding window (standard causal attention)
        if self.dcp_size > 1:
            _write_kv_cache(k, v, slot_mapping)

            local_tokens = q.shape[1]
            S_total = local_tokens * self.dcp_size
            q_gathered = self.cp_kv_group.all_gather(q.contiguous(), dim=1)

            I = self.cp_kv_cache_interleave_size
            W = self.dcp_size
            p = torch.arange(S_total, device=q.device)
            unshuffle_idx = (p // I) % W * local_tokens + (p // (W * I)) * I + (p % I)
            q_gathered = q_gathered[:, unshuffle_idx, :]

            local_prior = (
                cached_seq_len // self.dcp_size if cached_seq_len is not None else None
            )
            # segmented_attention_cp has no packed-FP8 read path; reject the
            # combination rather than silently misreading the swizzled cache.
            assert not self.fp8_packed, (
                "packed FP8 KV cache is not supported with DCP prefill "
                "(segmented_attention_cp cannot read the swizzled K layout)"
            )
            attn_output = NF.segmented_attention_cp(
                q=q_gathered,
                k_local=k,
                v_local=v,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=block_table,
                prior_tokens=local_prior,
                block_size=block_size,
                cp_rank=self.cp_chunk_idx,
                cp_world_size=self.dcp_size,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                cp_group=self.cp_kv_group,
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
            )
        elif kv_segment_size:
            # attention_segmented_cte cannot read a NON-packed FP8 K cache, so
            # fail fast on that combo instead of hitting a cryptic kernel abort.
            # Packed FP8 (fp8_packed_kv=True) and bf16 KV both pass.
            from vllm_neuron.utils.dtype_utils import (
                validate_fp8_segmented_supported,
            )

            kv_is_fp8 = self.k_cache.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            )
            validate_fp8_segmented_supported(kv_is_fp8, self.fp8_packed)

            _write_kv_cache(k, v, slot_mapping)

            # FP8 KV dequant for the segmented read: _write_kv_cache stored
            # k*k_scale / v*v_scale, so fold the K-dequant into the softmax scale
            # and divide the V-dequant out of the output — the same convention
            # forward_decode uses (softmax_scale=scaling/k_scale_float and the
            # o_proj weight absorbing v_scale). Exact no-ops when the checkpoint
            # ships no k/v scales (our case: k_scale_float=v_scale_float=1.0).
            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=block_table,
                prior_tokens=cached_seq_len,
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling / self.k_scale_float,
                tp_q=True,
                tp_out=True,
                fp8_packed=self.fp8_packed,
            )
            if self.v_scale_float != 1.0:
                attn_output = attn_output / self.v_scale_float
        else:
            _write_kv_cache(k, v, slot_mapping)

            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)

            q_flash = q.transpose(1, 2)  # [Nh, Dh, T]
            k_flash = k.transpose(1, 2)  # [Nh, Dh, T]
            v_flash = v  # [Nh, T, Dh]

            attn_output = NF.flash_attention(
                q_flash,
                k_flash,
                v_flash,
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )

        # ── Step 5: Output Projection ────────────────────────────────────
        attn_output = attn_output.unsqueeze(0)  # [1, Nh, Dh, T]
        # DIAGNOSTIC (2026-08-10 01:10 JST) — REMOVED, and here is why, so nobody re-adds it.
        # Two attempts to measure the o_proj input ON DEVICE both failed for structural reasons:
        #   1. print/logger  -> `torch.compile` expands the format string at TRACE time, so the
        #      log shows `%.6g` placeholders and SymNodeVariable objects instead of values.
        #   2. capture_tensor -> the tensor-capture side channel needs FIXED output shapes; a
        #      scalar reduction reshaped to [1] fails NEFF verification with
        #      "total_element_count == Multiply(output_sizes) (196608 vs. 1)".
        # The question it was meant to answer (does the activation saturate at scale*240?) has
        # since been answered TWICE by other means, both saying NO:
        #   * `harness/measure_oproj_activations.py` (CPU BF16 reconstruction): saturation rate
        #     0.0000% on layers 0/43/87; amax reaches only 22-52% of the clip threshold.
        #   * the full 26-prompt A/B of the 448/240 recalibration: 5 prompts improve and 5
        #     regress, i.e. NET ZERO -- the signature of a range/resolution trade-off, not of
        #     clipping being removed.
        # ⇒ If this needs revisiting, use `debug_logits_dir` (which already has a working
        #   fixed-shape write path) rather than reintroducing either approach above.
        # 🎯 ROW (per-token DYNAMIC) quantization for o_proj — opt-in via OPROJ_QUANT_MODE=row.
        # Motivation (README §1.6c): a static per-tensor input scale was measured to be a pure
        # trade-off. Recalibrating it by the theoretically-correct 448/240 factor improves 5 of 24
        # prompts and regresses 5 — net zero — because the activation never saturates (0.0000%
        # measured) and only reaches 22-52% of the clip threshold. A single scalar cannot suit
        # every token, so the fix that removes the calibration problem instead of relocating it is
        # to let the kernel take a per-token absmax (QuantizationType.ROW,
        # nkilib/.../output_projection_cte_quantization.py::perform_row_quantized_projection).
        # ⚠️ ROW wants the activation as [B, S, N, D] (STATIC wants [B, N, D, S]) and a per-row
        # weight scale of shape [P_MAX, H]; ``o_proj_w_scale_row`` is built at load time.
        _oproj_mode = (os.environ.get("OPROJ_QUANT_MODE") or "static").strip().lower()
        if self._fp8_o and _oproj_mode == "row" and self.o_proj_w_scale_row is not None:
            # [1, Nh, Dh, T] -> [1, T, Nh, Dh]
            attn_row = attn_output.permute(0, 3, 1, 2).contiguous()
            attn_output = NF.o_proj(
                attn_row,
                self.o_proj_weight,
                None,
                quantization_type=QuantizationType.ROW,
                input_scales=None,          # unused: the kernel derives it per token
                weight_scales=self.o_proj_w_scale_row,
            )  # [1, T, H]
        elif self._fp8_o:
            attn_output = NF.o_proj(
                attn_output,
                self.o_proj_weight,
                None,
                quantization_type=QuantizationType.STATIC,
                input_scales=self.o_proj_in_scale,
                weight_scales=self.o_proj_w_scale,
            )  # [1, T, H]
        else:
            attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)  # [1, T, H]
        attn_output = attn_output.squeeze(0)  # [T, H]

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # ── Decode path ──────────────────────────────────────────────────────

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
        fused_norm_weight: torch.Tensor | None = None,
        fused_norm_eps: float | None = None,
    ):
        """Decode: fused megakernel for single-token generation."""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B_local = block_table.shape[0]
        B = B_local * self.attention_dp_size
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        X = hidden_states.view(B, S_decode, hidden)

        # Prepare RoPE for megakernel format: [T, Hd] → [Hd/2, B, S]
        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_kernel = (
            cos[:, :half_d]
            .view(B_local, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin[:, :half_d]
            .view(B_local, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )

        # <-- MODEL-SPECIFIC: Standard causal mask, no sliding window
        dcp_decode_active = self.dcp_size > 1 and not self.apply_prefill_dcp
        mask_q_heads = self.num_q_heads_after_a2a * (
            self.dcp_size if dcp_decode_active else 1
        )
        local_filled = None
        dcp_active_mask = None
        if dcp_decode_active:
            cached_seq_len = attn_metadata[layer_name]["cached_seq_len"]
            n = cached_seq_len.to(torch.float32)
            I = self.cp_kv_cache_interleave_size
            W = self.dcp_size
            R = self.dcp_group.rank_in_group
            vblock = block_size * W
            stride = I * W

            full_vblocks = torch.div(n, vblock, rounding_mode="trunc")
            remaining = n - full_vblocks * vblock
            full_strides = torch.div(remaining, stride, rounding_mode="trunc")
            leftover = remaining - full_strides * stride

            local_filled = (
                full_vblocks * block_size
                + full_strides * I
                + torch.clamp(leftover - R * I, min=0, max=I)
            )

            owner = torch.div(remaining, I, rounding_mode="trunc") % W
            dcp_active_mask = (owner == R).float()

        pos_ids = positions.view(1, B_local * S_decode)
        attention_mask = NF.gen_attention_decode_mask(
            pos_ids=pos_ids.to(torch.float32),
            bs=B_local,
            q_head=mask_q_heads,
            s_active=S_decode,
            s_prior=S_ctx,
            start_pos=None,
            block_len=block_size,
            local_filled_slots=local_filled,
            dcp_active_mask=dcp_active_mask if dcp_decode_active else None,
        )

        # >>> Packed FP8 K cache (fp8_packed): the decode kernel can read the
        # swizzled layout directly, but only when its internal block_len resize
        # keeps the paired axis >= 2. Some bucket geometries (batch=1, which
        # forces s_prior sharding) resize it to 1, and then the packed cache must
        # be un-swizzled to the standard layout first. The flag is a trace-time
        # constant, so each compiled decode NEFF takes exactly one branch.
        #
        # Note the rank bookkeeping: the kernel's __internal_squeeze_head_dim
        # accepts the head axis on K and V only as a MATCHED PAIR -- either
        # (K 5-D, V 4-D) for packed / (K 4-D, V 3-D) for plain, or both already
        # squeezed. Squeezing V while leaving packed K at 5-D fails validation
        # with "Expecting V_cache to have 4 dims when K_cache has 5, got 3".
        # So V is squeezed only when K is also being squeezed here: on the packed
        # read path we hand both tensors over with their head axis intact and let
        # the kernel drop it. On the un-swizzled fallback K is back to 4-D and
        # both are squeezed exactly as on the bf16 path.
        _k_cache_src = self.k_cache
        _read_packed = False
        if self.fp8_packed:
            _use_packed_kernel = _packed_fp8_viable_for_bucket(
                block_len=block_size,
                bs=B_local,
                q_head=self.num_q_heads_after_a2a,
                s_active=S_decode,
                s_prior=S_ctx,
            )
            if _use_packed_kernel:
                _read_packed = True
            else:
                _k_cache_src = _unswizzle_packed_k(self.k_cache)

        if _read_packed:
            # Packed: leave the head axis on both so the pair stays consistent.
            k_cache = _k_cache_src
            v_cache = self.v_cache
        else:
            k_cache = (
                _k_cache_src.squeeze(1)
                if _k_cache_src.dim() == 4 and nkh
                else _k_cache_src
            )
            v_cache = (
                self.v_cache.squeeze(1)
                if self.v_cache.dim() == 4 and nkh
                else self.v_cache
            )

        active_blocks_table = block_table

        # ── FP8-native decode wiring ──────────────────────────────────────
        # QKV: STATIC quant with [128,3] weight scale (Q|K|V) + [128,1] input scale.
        # o_proj: the kernel scales attention output by v_scale; the caller must
        # absorb v_scale. For BF16 o_proj that means W_out = W_out / v_scale_float;
        # for FP8 o_proj W_out must stay fp8, so instead fold v_scale into the
        # output dequant scale: weight_dequant_scale_out = o_proj_w_scale /
        # v_scale_float (kernel-documented, attention_block_tkg.py:262-264). When
        # the KV cache is BF16, v_scale_float == 1.0 so this is a no-op.
        qkv_quant = QuantizationType.STATIC if self._fp8_qkv else QuantizationType.NONE
        out_quant = QuantizationType.STATIC if self._fp8_o else QuantizationType.NONE
        if self._fp8_o:
            W_out = self.o_proj_weight
            weight_dequant_scale_out = self.o_proj_w_scale / self.v_scale_float
            input_dequant_scale_out = self.o_proj_in_scale
        else:
            W_out = self.o_proj_weight / self.v_scale_float
            weight_dequant_scale_out = None
            input_dequant_scale_out = None

        # In-kernel KV write (opt-in): hand the kernel the write positions and let it
        # store K/V itself. `kv_cache_update_idx` is [B, S_tkg] uint32 of slot indices;
        # the kernel's no-overlap requirement holds because a decode step only ever writes
        # the next, previously-unread position. With update_cache=True the API returns just
        # the attention output, so the unpacking below is conditional.
        _inkernel_kv = _DECODE_INKERNEL_KV_WRITE
        _kv_update_idx = (
            slot_mapping.view(B_local, S_decode).to(torch.uint32) if _inkernel_kv else None
        )

        # >>> PARALLELISM: Fused megakernel with TP-sharded weights <<<
        _decode_ret = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            # >>> L13: fuse the pre-attention RMSNorm into the QKV step of the
            # decode megakernel. The kernel normalizes X and only then applies
            # the FP8 input scale (attention_block_tkg passes fused_norm_type +
            # qkv_in_scale to the same qkv() call), so the scale stays
            # calibrated on normalized activations.
            rmsnorm_X_enabled=fused_norm_weight is not None,
            rmsnorm_X_eps=fused_norm_eps,
            rmsnorm_X_gamma=(
                fused_norm_weight.unsqueeze(0) if fused_norm_weight is not None else None
            ),
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            quantization_type_qkv=qkv_quant,
            weight_dequant_scale_qkv=self.qkv_w_scale if self._fp8_qkv else None,
            input_dequant_scale_qkv=self.qkv_in_scale if self._fp8_qkv else None,
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache,
            V_cache=v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling / self.k_scale_float,
            sink=None,
            update_cache=_inkernel_kv,
            kv_cache_update_idx=_kv_update_idx,
            W_out=W_out,
            bias_out=None,
            quantization_type_out=out_quant,
            weight_dequant_scale_out=weight_dequant_scale_out,
            input_dequant_scale_out=input_dequant_scale_out,
            transposed_out=False,
            out_in_sb=False,
            k_scale=self.k_scale
            if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
            else None,
            v_scale=self.v_scale
            if self.v_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
            else None,
            # Tell the decode megakernel whether the K cache it just received is
            # swizzled. This must describe `k_cache`, NOT self.fp8_packed: on the
            # non-viable-bucket fallback above we hand it an un-swizzled copy, so
            # claiming packed there would make the kernel misread a standard cache.
            fp8_packed=self.fp8_packed and _k_cache_src is self.k_cache,
            attention_dp=self.attention_dp_size,
            attention_dp_group=self.attention_dp_group.device_group
            if self.attention_dp_group
            else None,
            attention_dp_rank=self.attention_dp_rank,
            kv_needs_a2a=self.kv_needs_a2a,
            dcp_size=self.dcp_size if dcp_decode_active else 1,
            dcp_group=self.dcp_group if dcp_decode_active else None,
        )

        # ── KV cache update ─────────────────────────────────────────────
        if _inkernel_kv:
            # The kernel already stored K/V (in the packed layout when it read packed),
            # so there is nothing to scatter and no swizzle round trip to pay.
            output = _decode_ret
            if self.fp8_packed and not _read_packed:
                # Non-viable packed bucket: the kernel wrote into the temporary
                # standard-layout copy, so fold it back into the packed cache. This is
                # the only place the round trip survives, and our buckets never take it.
                self.k_cache.copy_(_swizzle_packed_k(_k_cache_src))
            # >>> PARALLELISM: Sum O-proj partials across TP * attn_dp supergroup <<<
            self.attn_tp_group.all_reduce(output)
            return output

        output, K_new, V_new = _decode_ret

        # ── Manual KV cache update (out-of-kernel; the default path) ────
        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        num_tokens = slot_mapping.shape[0]

        k_new = (
            K_new.permute(1, 2, 0)
            .reshape(B_local, nkh, S_decode, self.head_dim)
            .transpose(0, 1)
            .reshape(nkh, B_local * S_decode, self.head_dim)
        )
        k_new_flat = k_new.reshape(-1, self.head_dim)

        v_new_flat = V_new.transpose(0, 1).reshape(-1, self.head_dim)

        head_indices_for_put = torch.arange(
            nkh, dtype=torch.long, device=hidden_states.device
        ).repeat_interleave(num_tokens)
        block_indices_for_put = block_indices.repeat(nkh)
        position_indices_for_put = position_indices.repeat(nkh)

        _put_index = (
            block_indices_for_put,
            head_indices_for_put,
            position_indices_for_put,
        )
        # V is never packed → scatters directly.
        self.v_cache.index_put_(_put_index, v_new_flat.to(self.v_cache.dtype))
        if self.fp8_packed:
            # Same reason as the prefill scatter: the swizzle interleaves adjacent
            # token positions into the trailing size-2 dim, so un-swizzle, scatter,
            # re-swizzle in place.
            _k_unpacked = _unswizzle_packed_k(self.k_cache)
            _k_unpacked.index_put_(_put_index, k_new_flat.to(self.k_cache.dtype))
            self.k_cache.copy_(_swizzle_packed_k(_k_unpacked))
        else:
            self.k_cache.index_put_(_put_index, k_new_flat.to(self.k_cache.dtype))

        # >>> PARALLELISM: Sum O-proj partials across TP * attn_dp supergroup <<<
        self.attn_tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4: MLP (Dense)
# Mixed: PARALLELISM (TP intermediate sharding, SP collectives) +
#        MODEL-SPECIFIC (SiLU activation, gate/up/down structure)
# =============================================================================


class Ministral3MLP(nn.Module):
    """Dense SwiGLU MLP with TP intermediate sharding.

    <-- MODEL-SPECIFIC:
    - SiLU-gated MLP (down(silu(gate(x)) * up(x)))
    - No bias on any projection
    - FP8 checkpoint weights (dequantized to BF16 at load)
    """

    def __init__(self, config: Ministral3Config, layer_idx: int | None = None):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: MLP TP group (TP * mlp_dp_size) + DP column <<<
        self.mlp_dp_size = (
            config.neuron_config.mlp_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_mlp_tp_group,
            get_neuron_mlp_dp_group,
        )

        mlp_tp_group = get_neuron_mlp_tp_group()
        self.mlp_tp_group = mlp_tp_group
        self.mlp_tp_size = mlp_tp_group.world_size
        self.mlp_tp_rank = mlp_tp_group.rank_in_group
        self.mlp_dp_group = get_neuron_mlp_dp_group()

        self.hidden_size = config.hidden_size
        self.dtype = config.torch_dtype
        self.intermediate_size_per_rank = config.intermediate_size // self.mlp_tp_size

        # <-- MODEL-SPECIFIC: FP8-native MLP (opt-in). When enabled, gate/up/down
        # keep FP8 weights and run FP8×FP8 STATIC matmuls; else dequant to BF16.
        # 🔬 layer_idx enables `MINISTRAL3_FP8_LAYERS` bisection (config.py).
        self._fp8_mlp = config.fp8_enabled("mlp", layer_idx)
        mlp_dtype = _FP8_DTYPE if self._fp8_mlp else config.torch_dtype

        # <-- MODEL-SPECIFIC: Separate gate and up projections (SwiGLU pattern)
        self.gate_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=mlp_dtype,
            )
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=mlp_dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank,
                config.hidden_size,
                dtype=mlp_dtype,
            )
        )

        # >>> FP8 STATIC scale buffers (non-persistent; filled in load_weights). <<<
        # All five are [128,1] f32 (the mlp kernel's per-tensor scale shape, which
        # both CTE and TKG use). gate/up share a single input scale (gate_up_in_scale).
        if self._fp8_mlp:
            self.register_buffer("gate_w_scale", None, persistent=False)
            self.register_buffer("up_w_scale", None, persistent=False)
            self.register_buffer("down_w_scale", None, persistent=False)
            self.register_buffer("gate_up_in_scale", None, persistent=False)
            self.register_buffer("down_in_scale", None, persistent=False)
            # 🎯 ROW (per-token DYNAMIC) variant — opt-in via MINISTRAL3_MLP_QUANT_MODE=row.
            # nkilib asserts these shapes for QuantizationType.ROW
            # (nkilib/core/mlp/mlp_parameters.py:222): gate/up [P_MAX, I], down [P_MAX, H],
            # where I/H are the per-rank shard widths. Populated in load_weights only when the
            # knob is set, so the default path allocates nothing extra (mirrors o_proj above).
            self.register_buffer("gate_w_scale_row", None, persistent=False)
            self.register_buffer("up_w_scale_row", None, persistent=False)
            self.register_buffer("down_w_scale_row", None, persistent=False)

        self._setup_weight_loaders(config)

    def _setup_weight_loaders(self, config):
        """>>> PARALLELISM: TP (and MLP DP) sharding of MLP weights. <<<

        When fp8-native is enabled the loaders keep fp8 weights (shard + ±240
        saturation, scales loaded separately); otherwise the adaptive loaders
        dequantize FP8 to BF16 when scale slices are present, else behave as the
        generic BF16 loaders (detected per-parameter from the checkpoint slice
        count).
        """
        sharding_loader = (
            fp8_native_sharding_weight_loader if self._fp8_mlp
            else adaptive_sharding_weight_loader
        )
        gate_up_loader = sharding_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.mlp_tp_size,
            is_storage_transposed=True,
        )
        down_loader = sharding_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.mlp_tp_size,
            is_storage_transposed=True,
        )

        gate_up_loader = with_rank_override(gate_up_loader, rank=self.mlp_tp_rank)
        down_loader = with_rank_override(down_loader, rank=self.mlp_tp_rank)

        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool,
        fused_norm_weight: torch.Tensor | None = None,
        fused_norm_eps: float | None = None,
    ) -> torch.Tensor:
        """Run the MLP.

        ``fused_norm_weight``/``fused_norm_eps`` are L13: when supplied, the
        pre-MLP RMSNorm is computed *inside* the kernel and the caller must NOT
        have applied ``post_attention_layernorm`` itself. When None (default) the
        caller normalizes and this is the unmodified baseline path.
        """
        # >>> PARALLELISM: All-gather from SP for full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # <-- MODEL-SPECIFIC: SiLU gated MLP (down(silu(gate) * up))
        if self._fp8_mlp:
            mlp_hidden = hidden_states
            # gen3 MLP-CTE (prefill) fuses gate/up input-quant into the PE
            # transpose, which fails when the hidden is bf16 (dst=fp8 != src=bf16).
            # Pre-quantize the hidden to fp8 so the transpose is fp8→fp8 (the
            # kernel's "quantized input" path). Decode uses mlp_tkg, which has no
            # such transpose, so it keeps the bf16 hidden + internal input-quant.
            #
            # #1-BISECTION (NKILIB_MLP_BF16_XPOSE_SRC): the nkilib knob makes the
            # source transpose a coherent BF16 transpose, so it WANTS a bf16 hidden.
            # Skip the prefill pre-quant so we feed bf16 and the knob isolates whether
            # the fp8 step-2 source transpose is the prefill MLP fp8 break.
            _bf16_xpose = os.environ.get("NKILIB_MLP_BF16_XPOSE_SRC", "0") == "1"
            # L13 SAFETY: the external pre-quant divides by ``gate_up_in_scale``,
            # a scale calibrated on *normalized* activations. If the norm is fused
            # into the kernel, the hidden reaching this point is un-normalized, so
            # pre-quantizing here would apply that scale to the wrong tensor (and
            # would also quantize before the norm rather than after). In that case
            # keep the hidden bf16 and let the kernel do norm→quant in the right
            # order. The prefill CTE transpose then needs the bf16-xpose knob, so
            # refuse the combination that would silently miscompile instead.
            # 2.32 PORT FIX -- the pre-quant must key off which nkilib kernel will
            # actually run, NOT off is_prefill. nkilib routes mlp() to CTE vs TKG on
            # ``batch_size * sequence_len <= TKG_BS_SEQLEN_THRESHOLD`` (96, see
            # nkilib/core/mlp/mlp_parameters.py::is_mlp_tkg). A decode step with
            # batch > 96 -- e.g. num_seqs_buckets=[128], which every #5 arm uses --
            # is therefore executed by mlp_CTE, not mlp_tkg. Keying on is_prefill left
            # that graph's hidden in bf16 and the new nki (0.6.0b1) asserts on it:
            #   "nc_matmul (transpose mode) dst dtype must match input dtype on gen3+,
            #    got dst=float8_e4m3 but input=bfloat16"
            # (observed at bs=128 decode, hidden (1,128,12288) bf16). The old nki
            # 0.5.0 did not assert, which is why 231 never surfaced this.
            # Mirror the kernel's own predicate so CTE always gets an fp8 hidden.
            _tkg_bs_seqlen_threshold = 96
            _n_tokens = hidden_states.shape[0] * hidden_states.shape[1]
            _runs_on_cte = is_prefill or _n_tokens > _tkg_bs_seqlen_threshold
            # EXPERIMENT ONLY (2026-08-08): FORCE_MLP_KERNEL lets us force TKG or CTE so the
            # num_seqs sweep can be decomposed into "less padding" vs "different kernel" (the
            # threshold above means crossing 96 changes both at once). This gate MUST track
            # functional/mlp.py's force_cte_mode/mode exactly: if the kernel runs CTE while this
            # says TKG, the hidden stays bf16 and the kernel's fp8 source transpose asserts
            # ("dst=float8_e4m3 but input=bfloat16") -- the very 2.32 port bug fixed above.
            # Prefill is always CTE regardless, so the override only affects decode.
            _force_mlp_kernel = (
                os.environ.get("FORCE_MLP_KERNEL") or "auto"
            ).strip().lower()
            if _force_mlp_kernel == "cte":
                _runs_on_cte = True
            elif _force_mlp_kernel == "tkg":
                _runs_on_cte = is_prefill
            # 🎯 ROW (per-token DYNAMIC) input quantization — MINISTRAL3_MLP_QUANT_MODE=row.
            # Resolved BEFORE the fused-norm guard below because that guard does not apply to
            # ROW: it exists because the *external* pre-quant would run before a kernel-internal
            # norm and would use a scale calibrated on normalized activations. ROW does no
            # external pre-quant and derives its scale inside the kernel after the norm, so the
            # ordering problem cannot arise.
            _mlp_row = (
                self._fp8_mlp
                and _mlp_quant_mode() == "row"
                and self.down_w_scale_row is not None
            )
            if (
                fused_norm_weight is not None
                and _runs_on_cte
                and not _bf16_xpose
                and not _mlp_row
            ):
                raise RuntimeError(
                    "Fused-RMSNorm with full-FP8 MLP on the CTE path requires "
                    "NKILIB_MLP_BF16_XPOSE_SRC=1 (the external "
                    "pre-quant is calibrated on normalized activations and "
                    "cannot run before a kernel-internal norm). Use "
                    "VLLM_NEURON_FUSED_RMSNORM_DECODE_ONLY=1 instead."
                )
            # The kernel row-quantizes the hidden itself, so the external pre-quant MUST be
            # skipped: feeding an already-fp8 hidden would make nkilib read it as the
            # "pre-quantized ROW" layout, which requires 4 extra bytes carrying the per-row
            # scale ([B, S, H+4], mlp_torch.py:428) and raises on a plain [B, S, H] fp8 tensor.
            # The calibrated input scales are not passed at all — that is the whole point.
            if _mlp_row:
                output = NF.mlp(
                    _row_quantize_activation_to_fp8(hidden_states),
                    self.gate_proj_weight,
                    self.up_proj_weight,
                    self.down_proj_weight,
                    quantization_type=QuantizationType.ROW,
                    gate_w_scale=self.gate_w_scale_row,
                    up_w_scale=self.up_w_scale_row,
                    down_w_scale=self.down_w_scale_row,
                    gate_up_in_scale=None,   # unused: derived per token on device
                    down_in_scale=None,      # unused: derived per token on device
                    output_dtype=self.dtype,
                    **_fused_norm_mlp_kwargs(fused_norm_weight, fused_norm_eps),
                )
                if output.dtype != self.dtype:
                    output = output.to(self.dtype)
                return self._mlp_reduce(output, is_prefill)
            if _runs_on_cte and not _bf16_xpose and fused_norm_weight is None:
                mlp_hidden = _quantize_activation_to_fp8(
                    hidden_states, self.gate_up_in_scale
                )
            output = NF.mlp(
                mlp_hidden,
                self.gate_proj_weight,
                self.up_proj_weight,
                self.down_proj_weight,
                quantization_type=QuantizationType.STATIC,
                gate_w_scale=self.gate_w_scale,
                up_w_scale=self.up_w_scale,
                down_w_scale=self.down_w_scale,
                gate_up_in_scale=self.gate_up_in_scale,
                down_in_scale=self.down_in_scale,
                # NF.mlp defaults output_dtype to the hidden's dtype; when we feed
                # an fp8 hidden (prefill), force a bf16 output so the residual add
                # (bf16 + mlp_out) doesn't hit "Float8 promotion not supported".
                output_dtype=self.dtype,
                **_fused_norm_mlp_kwargs(fused_norm_weight, fused_norm_eps),
            )
            # The @nki.jit abstract/meta impl infers the output FakeTensor dtype
            # from the (fp8) input hidden and ignores output_dtype during FX
            # tracing, so the tracer thinks `output` is fp8 → the later residual
            # add (bf16 + fp8) raises a Float8-promotion error at compile time.
            # An explicit cast is a legal fp8→bf16 op the tracer accepts and is a
            # no-op at runtime (the kernel already emits bf16 via output_dtype).
            # 2.32 PORT FIX: gate on _runs_on_cte for the same reason as the
            # pre-quant above -- a >96-token decode is fed an fp8 hidden too, so it
            # needs the same tracer-visible cast back to bf16.
            if _runs_on_cte and output.dtype != self.dtype:
                output = output.to(self.dtype)
        else:
            output = NF.mlp(
                hidden_states,
                self.gate_proj_weight,
                self.up_proj_weight,
                self.down_proj_weight,
                **_fused_norm_mlp_kwargs(fused_norm_weight, fused_norm_eps),
            )

        return self._mlp_reduce(output, is_prefill)

    def _mlp_reduce(self, output, is_prefill):
        """>>> PARALLELISM: Combine TP (+ MLP DP) shards <<<

        Split out verbatim from ``forward`` so the ROW branch can reuse it; the STATIC and
        BF16 paths are unchanged.

        🎯 ``MINISTRAL3_MLP_REDUCE_FP32=1`` performs the cross-rank combine in FP32 instead of the
        activation dtype (default off ⇒ byte-identical to every recorded measurement).

        WHY (ACCURACY_CONSTRAINTS §2.4 items 6–7). ``down_proj`` is row-parallel, so each rank
        contributes a partial sum over ``I/TP`` products and the partials are combined here. The
        combine runs in BF16 because ``output`` is BF16. Holding the quantized values and the scale
        fixed and varying only the partition, 12 simulated trials at the real ``I = 28,672`` give a
        relative error of **3.67e-03 at TP=8 and 7.20e-03 at TP=32** for BF16 cross-rank
        accumulation, versus **1.09e-07 / 2.16e-07** in FP32. That 2.0× ratio is the same order as
        the measured 3.74× growth of |Δ| from TP=8 to TP=32, and the FP32 variant removes ~3.7e-03
        of error at **every** TP, not only at TP=32.

        ⚠️ COST: the collective moves twice the bytes, on the critical path of every MLP. Measure
        throughput alongside accuracy before considering this for the shipping configuration —
        and note that the accuracy arms run with synchronous scheduling, so they cannot be used
        for the throughput number (§7).
        """
        if os.environ.get("MINISTRAL3_MLP_REDUCE_FP32", "0") == "1":
            orig_dtype = output.dtype
            output = output.to(torch.float32)
            if is_prefill:
                if self.world_size > 1:
                    output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.mlp_tp_group.all_reduce(output)
            return output.to(orig_dtype)

        if is_prefill:
            if self.world_size > 1:
                output = self.tp_group.reduce_scatter(output, dim=0)
        else:
            self.mlp_tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: Decoder Layer
# =============================================================================


def _dp_transition(
    x: torch.Tensor,
    current_group,
    target_group,
    dim: int = 0,
) -> torch.Tensor:
    """Transition tensor between DP gathered states."""
    current_dp = current_group.world_size
    target_dp = target_group.world_size
    if current_dp == target_dp:
        return x
    if current_dp > target_dp:
        per_dp = x.shape[dim] // current_dp
        start = (current_group.rank_in_group // target_dp) * target_dp * per_dp
        return x.narrow(dim, start, target_dp * per_dp)
    per_dp = x.shape[dim] // current_dp
    x = x.narrow(dim, current_group.rank_in_group * per_dp, per_dp)
    return target_group.all_gather(x, dim=dim)


class Ministral3DecoderLayer(nn.Module):
    """Single transformer decoder layer.

    Architecture (MODEL-SPECIFIC):
        hidden_states → RMSNorm → Attention → residual → RMSNorm → MLP → residual
    """

    def __init__(self, config: Ministral3Config, layer_idx: int):
        super().__init__()
        # <-- MODEL-SPECIFIC: Pre-attention and pre-MLP RMSNorm
        self.input_layernorm = Ministral3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Ministral3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = Ministral3Attention(config, layer_idx=layer_idx)
        self.mlp = Ministral3MLP(config, layer_idx=layer_idx)
        self.layer_idx = layer_idx

        # >>> PARALLELISM: DP sizes for batch state transitions <<<
        nc = config.neuron_config
        self.attn_dp = nc.attention_dp_size if nc else 1
        self.mlp_dp = nc.mlp_dp_size if nc else 1

        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_dp_group,
            get_neuron_mlp_dp_group,
        )

        self.attn_dp_group = get_neuron_attention_dp_group()
        self.mlp_dp_group = get_neuron_mlp_dp_group()

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

        if not is_decode:
            return self._forward_prefill(
                hidden_states, positions, position_embeddings, attn_metadata
            )

        # Transition mlp_dp → attn_dp
        hidden_states = _dp_transition(
            hidden_states, self.mlp_dp_group, self.attn_dp_group
        )

        # ── Self Attention ───────────────────────────────────────────────
        # >>> L13 (decode): hand gamma to the kernel and skip the standalone norm.
        residual = hidden_states
        if not _FUSED_RMSNORM_ATTN_DECODE:
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
            fused_norm_weight=(
                self.input_layernorm.weight if _FUSED_RMSNORM_ATTN_DECODE else None
            ),
            fused_norm_eps=self.input_layernorm.variance_epsilon,
        )

        hidden_states = residual + hidden_states
        hidden_states = _dp_transition(
            hidden_states, self.attn_dp_group, self.mlp_dp_group
        )

        # ── MLP Feed-Forward ─────────────────────────────────────────────
        residual = hidden_states
        if not _FUSED_RMSNORM_MLP_DECODE:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            is_prefill=False,
            fused_norm_weight=(
                self.post_attention_layernorm.weight
                if _FUSED_RMSNORM_MLP_DECODE
                else None
            ),
            fused_norm_eps=self.post_attention_layernorm.variance_epsilon,
        )
        hidden_states = residual + hidden_states

        return hidden_states

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        # ── Self Attention ───────────────────────────────────────────────
        # >>> L13 (prefill): OFF under the recommended _DECODE_ONLY gate. Fusing
        # here costs ~15% TTFT (new sync boundary after the input AllGather).
        residual = hidden_states
        if not _FUSED_RMSNORM_ATTN_PREFILL:
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
            fused_norm_weight=(
                self.input_layernorm.weight if _FUSED_RMSNORM_ATTN_PREFILL else None
            ),
            fused_norm_eps=self.input_layernorm.variance_epsilon,
        )
        hidden_states = residual + hidden_states

        # ── MLP Feed-Forward ─────────────────────────────────────────────
        residual = hidden_states
        if not _FUSED_RMSNORM_MLP_PREFILL:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            is_prefill=True,
            fused_norm_weight=(
                self.post_attention_layernorm.weight
                if _FUSED_RMSNORM_MLP_PREFILL
                else None
            ),
            fused_norm_eps=self.post_attention_layernorm.variance_epsilon,
        )
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# =============================================================================


class Ministral3Model(nn.Module):
    """Ministral3 transformer backbone."""

    def __init__(self, config: Ministral3Config):
        super().__init__()
        self.config = config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.embedding_dp_size = (
            config.neuron_config.embedding_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_embedding_tp_group,
            get_neuron_embedding_dp_group,
            get_neuron_mlp_dp_group,
        )

        emb_tp_group = get_neuron_embedding_tp_group()
        emb_device_group = emb_tp_group.device_group
        self.embedding_dp_group = get_neuron_embedding_dp_group()
        self.embedding_tp_rank = emb_tp_group.rank_in_group
        self.mlp_dp_group = get_neuron_mlp_dp_group()

        # >>> PARALLELISM: Vocab-sharded embedding <<<
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=emb_device_group,
        )

        # <-- MODEL-SPECIFIC: Stack of decoder layers
        self.layers = nn.ModuleList(
            [
                Ministral3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # <-- MODEL-SPECIFIC: Final RMSNorm
        self.norm = Ministral3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Ministral3RotaryEmbedding(config)

        emb_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.embed_tokens.vocab_size_per_rank,
            num_shards=self.embed_tokens.tp_size,
            is_storage_transposed=False,
        )
        emb_loader = with_rank_override(emb_loader, rank=self.embedding_tp_rank)
        set_weight_loader(self.embed_tokens.weight, emb_loader)

        # Eagle3 not supported for Ministral3 yet — empty.
        self.aux_hidden_state_layers = []

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        if not is_prefill and self.embedding_dp_size > 1:
            input_ids = self.embedding_dp_group.all_gather(input_ids, dim=0)

        hidden_states = self.embed_tokens(input_ids, scatter_tokens=is_prefill)

        if not is_prefill:
            hidden_states = _dp_transition(
                hidden_states, self.embedding_dp_group, self.mlp_dp_group
            )

        if (
            is_prefill
            and self.world_size > 1
            and inputs_embeds is not None
            and is_token_ids is not None
        ):
            local_len = hidden_states.shape[0]
            start = self.rank * local_len
            inputs_embeds = inputs_embeds[start : start + local_len]
            is_token_ids = is_token_ids[start : start + local_len]

        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        aux_hidden_states = []
        for idx, decoder_layer in enumerate(self.layers):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states)
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )

        hidden_states = self.norm(hidden_states)

        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            aux_hidden_states = [
                self.tp_group.all_gather(aux, dim=0) for aux in aux_hidden_states
            ]

        return hidden_states, aux_hidden_states


# =============================================================================
# Section 7: Language Model Head
# =============================================================================


class Ministral3ForCausalLM(nn.Module):
    """Ministral3 model with language modeling head.

    <-- MODEL-SPECIFIC: Untied embeddings (separate lm_head weight, BF16 in the
    checkpoint).
    """

    def __init__(self, config: Ministral3Config):
        super().__init__()
        self.config = config
        self.model = Ministral3Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.lm_head_dp_size = (
            config.neuron_config.lm_head_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_lm_head_tp_group,
            get_neuron_lm_head_dp_group,
            get_neuron_mlp_dp_group,
        )

        lm_head_tp_group = get_neuron_lm_head_tp_group()
        self.lm_head_tp_group = lm_head_tp_group
        lm_head_device_group = lm_head_tp_group.device_group
        self.lm_head_dp_group = get_neuron_lm_head_dp_group()
        self.mlp_dp_group = get_neuron_mlp_dp_group()

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

        lm_head_tp_rank = lm_head_tp_group.rank_in_group

        # <-- MODEL-SPECIFIC: Untied lm_head (BF16 weight)
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=lm_head_device_group,
        )

        lm_head_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.lm_head.out_features_per_rank,
            num_shards=self.lm_head.tp_size,
            is_storage_transposed=False,
        )
        lm_head_loader = with_rank_override(lm_head_loader, rank=lm_head_tp_rank)
        set_weight_loader(self.lm_head.weight, lm_head_loader)

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=lm_head_device_group,
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]

        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        hidden_states, aux_hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )

        mlp_dp = self.mlp_dp_group.world_size
        if mlp_dp > 1:
            local_size = hidden_states.shape[0] // mlp_dp
            dp_rank = self.mlp_dp_group.rank_in_group
            hidden_states = hidden_states[
                dp_rank * local_size : (dp_rank + 1) * local_size
            ]

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )

        if self.lm_head_dp_size > 1:
            hidden_states_for_logits = self.lm_head_dp_group.all_gather(
                hidden_states_for_logits, dim=0
            )

        logits = self.lm_head(hidden_states_for_logits)

        gathered_logits = None
        if self._gather_logits:
            if self.lm_head.gather_output:
                gathered_logits = logits
            else:
                gathered_logits = self.lm_head_tp_group.all_gather(logits, dim=1)

        if self.lm_head_dp_size > 1:
            B_local = sampling_positions.shape[0]
            dp_rank = self.lm_head_dp_group.rank_in_group
            logits = logits[dp_rank * B_local : (dp_rank + 1) * B_local]
            if gathered_logits is not None:
                gathered_logits = gathered_logits[
                    dp_rank * B_local : (dp_rank + 1) * B_local
                ]

        # ── No on-device sampling: return logits directly ────────────────
        if self.on_device_sampling_config is None:
            return logits

        # ── On-device sampling ───────────────────────────────────────────
        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Ministral3Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # ── KV Cache Management ──────────────────────────────────────────────

    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
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
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]
            # Detect the swizzled packed FP8 K layout from the bound cache rank:
            # packed K is [num_blocks, kv_heads, block_size // 2, head_dim, 2],
            # one dim higher than V (which is never packed). A bf16 or standard
            # FP8 K cache stays 4-D, so this is False there.
            k_is_packed = (
                layer.self_attn.k_cache.dim() == layer.self_attn.v_cache.dim() + 1
            )
            if hasattr(layer.self_attn, "fp8_packed"):
                layer.self_attn.fp8_packed = k_is_packed
            elif k_is_packed:
                # The runner allocated a packed (5-D) K cache (fp8_packed_kv +
                # fp8 KV dtype) but this attention path is not packed-aware.
                # Fail loudly rather than silently reading the swizzled 5-D
                # cache as if it were a standard 4-D one.
                raise NotImplementedError(
                    f"{layer_name}: a packed FP8 K cache was allocated "
                    "(fp8_packed_kv=True) but this attention path does not "
                    "support packed FP8 KV. Unset fp8_packed_kv."
                )

    # ── Weight Loading ───────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from checkpoint.

        >>> PARALLELISM: Weight loaders handle TP sharding <<<
        <-- MODEL-SPECIFIC: HF checkpoint key → model parameter mappings. For FP8
        checkpoints, each projection mapping also lists its `weight_scale_inv`
        key so the adaptive loaders can dequantize. FP8 is detected from the
        actual checkpoint tensors (presence of `*.weight_scale_inv`), NOT from
        the HF config — robust to `--hf-overrides '{"quantization_config": {}}'`
        which vLLM needs to avoid rejecting the unsupported `fp8` method.
        """
        tp_rank = self.rank
        tp_size = self.world_size

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        # Index the checkpoint so we can probe for FP8 scale tensors.
        checkpoint._ensure_indexed()
        is_fp8 = (
            "model.layers.0.self_attn.q_proj.weight_scale_inv"
            in checkpoint._tensor_name_to_file
        )
        # FP8-native families (kept in FP8, run as STATIC FP8×FP8 matmuls). For
        # these the loaders take WEIGHT-only slices (no scale appended) — the
        # per-tensor scales are loaded separately into buffers below. Families NOT
        # listed use the BF16-dequant path (scale keys appended when is_fp8).
        # 🔬 PER-LAYER: with `MINISTRAL3_FP8_LAYERS` a family can be FP8 on some
        # layers and BF16 on others, and the WEIGHT DTYPE must match the module's
        # own decision layer by layer (a mismatch silently loads fp8 bytes into a
        # bf16 parameter, or appends a scale key the fp8 loader does not expect).
        # So the per-layer answer is taken inside the loop below; these
        # family-level values only drive the "any layer at all" validation and log.
        L = len(self.model.layers)
        fp8_qkv = any(self.config.fp8_enabled("qkv", i) for i in range(L))
        fp8_o = any(self.config.fp8_enabled("o_proj", i) for i in range(L))
        fp8_mlp = any(self.config.fp8_enabled("mlp", i) for i in range(L))
        if (fp8_qkv or fp8_o or fp8_mlp) and not is_fp8:
            raise ValueError(
                "dense_fp8_static is set but the checkpoint has no "
                "*.weight_scale_inv tensors (not an FP8 checkpoint)."
            )
        def _fp8_layer_summary(fam: str) -> str:
            """Compact per-layer FP8 census, e.g. "44/88 [0-43]" — always logged so
            an arm's identity is recoverable from the server log alone."""
            on = [i for i in range(L) if self.config.fp8_enabled(fam, i)]
            if not on:
                return f"0/{L}"
            if len(on) == L:
                return f"{L}/{L} all"
            rs, st = [], on[0]
            for a, b in zip(on, on[1:] + [None]):
                if b != (a + 1):
                    rs.append(f"{st}-{a}" if st != a else f"{st}")
                    st = b
            return f"{len(on)}/{L} [{','.join(rs)}]"

        logger.info(
            "Ministral3 FP8 checkpoint=%s; FP8-native families: qkv=%s o_proj=%s mlp=%s",
            is_fp8, fp8_qkv, fp8_o, fp8_mlp,
        )
        logger.info(
            "Ministral3 FP8 PER-LAYER census: qkv=%s o_proj=%s mlp=%s (MINISTRAL3_FP8_LAYERS=%r)",
            _fp8_layer_summary("qkv"), _fp8_layer_summary("o_proj"),
            _fp8_layer_summary("mlp"), os.environ.get("MINISTRAL3_FP8_LAYERS", ""),
        )
        # 🔴 2026-08-18 JST: log every scale-modifying knob. E2 (MINISTRAL3_ACT_SCALE_RECAL)
        # was SILENT, so "did the knob take effect?" could not be answered from the log —
        # and a silent no-op would have been read as a falsified hypothesis. Never again:
        # a knob that changes numerics must announce itself.
        logger.info(
            "Ministral3 FP8 SCALE KNOBS: ACT_SCALE_RECAL=%r (x%.4f where selected) "
            "MLP_IN_SCALE_MULT=%s (down_only=%s) OPROJ_IN_SCALE_MULT=%s FP8_RANGE_FIX=%s "
            "MLP_QUANT_MODE=%s OPROJ_QUANT_MODE=%s MLP_REDUCE_FP32=%s",
            os.environ.get("MINISTRAL3_ACT_SCALE_RECAL", ""), _ACT_SCALE_TRN2_RECAL,
            os.environ.get("MINISTRAL3_MLP_IN_SCALE_MULT", "1.0"),
            os.environ.get("MINISTRAL3_MLP_IN_SCALE_MULT_DOWN_ONLY", "0"),
            os.environ.get("OPROJ_IN_SCALE_MULT", "1.0"),
            os.environ.get("MINISTRAL3_FP8_RANGE_FIX", "0"),
            _mlp_quant_mode(), (os.environ.get("OPROJ_QUANT_MODE") or "static"),
            os.environ.get("MINISTRAL3_MLP_REDUCE_FP32", "0"),
        )

        mappings = dict()
        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"

            # <-- MODEL-SPECIFIC: Separate Q, K, V → fused QKV.
            # BF16-dequant path appends per-projection scale keys (adaptive loader
            # expects [q_w, k_w, v_w, q_scale, k_scale, v_scale]); FP8-native path
            # passes weight-only [q_w, k_w, v_w] (scales loaded as buffers).
            qkv_keys = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            o_keys = f"{prefix}.self_attn.o_proj.weight"
            gate_keys = f"{prefix}.mlp.gate_proj.weight"
            up_keys = f"{prefix}.mlp.up_proj.weight"
            down_keys = f"{prefix}.mlp.down_proj.weight"

            # 🔬 per-layer decision — MUST match the module's own gating above
            l_qkv = self.config.fp8_enabled("qkv", layer_id)
            l_o = self.config.fp8_enabled("o_proj", layer_id)
            l_mlp = self.config.fp8_enabled("mlp", layer_id)
            if is_fp8:
                if not l_qkv:
                    qkv_keys = qkv_keys + [
                        f"{prefix}.self_attn.q_proj.weight_scale_inv",
                        f"{prefix}.self_attn.k_proj.weight_scale_inv",
                        f"{prefix}.self_attn.v_proj.weight_scale_inv",
                    ]
                if not l_o:
                    o_keys = [o_keys, f"{prefix}.self_attn.o_proj.weight_scale_inv"]
                if not l_mlp:
                    gate_keys = [gate_keys, f"{prefix}.mlp.gate_proj.weight_scale_inv"]
                    up_keys = [up_keys, f"{prefix}.mlp.up_proj.weight_scale_inv"]
                    down_keys = [down_keys, f"{prefix}.mlp.down_proj.weight_scale_inv"]

            mappings[f"{prefix}.self_attn.qkv_proj_weight"] = qkv_keys
            mappings[f"{prefix}.self_attn.o_proj_weight"] = o_keys

            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )
            mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )

            mappings[f"{prefix}.mlp.gate_proj_weight"] = gate_keys
            mappings[f"{prefix}.mlp.up_proj_weight"] = up_keys
            mappings[f"{prefix}.mlp.down_proj_weight"] = down_keys

        # <-- MODEL-SPECIFIC: Untied embeddings — lm_head loaded from its own key.
        mappings["lm_head.weight"] = "lm_head.weight"

        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device
        ).state_dict

        # Cast non-FP8 params to the compute dtype. FP8-native params are kept in
        # float8_e4m3fn (their loaders already produced that dtype) — do NOT cast
        # them to BF16, that would defeat FP8 matmul and break the kernel assert.
        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype == torch.float8_e4m3fn:
                continue
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self._load_kv_cache_scales(checkpoint, device)

        self.load_state_dict(rank_sharded, strict=False, assign=True)

        # FP8-native: load the per-tensor STATIC scales into the module buffers
        # (scalars are shard-invariant — replicate; do this after load_state_dict
        # so it does not clobber the buffers).
        if fp8_qkv or fp8_o or fp8_mlp:
            self._load_fp8_static_scales(
                checkpoint, device, fp8_qkv, fp8_o, fp8_mlp
            )

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Lightweight weight loading used during CPU compile (KV scales only)."""
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        checkpoint._ensure_indexed()
        self._load_kv_cache_scales(checkpoint, device)

    def _load_kv_cache_scales(
        self, checkpoint: SafetensorsCheckpoint, device: torch.device
    ):
        """Load KV cache quantization scales from checkpoint if provided."""
        from vllm_neuron.utils.dtype_utils import QUANTIZED_KV_CACHE_DTYPES
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()

        for layer_id in range(len(self.model.layers)):
            attn = self.model.layers[layer_id].self_attn

            if vllm_config.cache_config.cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
                continue

            for scale_name in ("k_scale", "v_scale"):
                key = f"model.layers.{layer_id}.self_attn.{scale_name}"
                if key in checkpoint._tensor_name_to_file:
                    val = 1.0 / checkpoint._get_slice(key)[:].to(
                        dtype=torch.bfloat16, device=device
                    )
                else:
                    val = torch.ones(1, dtype=torch.bfloat16, device=device)
                setattr(attn, scale_name, val.reshape(1, 1))

            attn.k_scale_float = attn.k_scale.item()
            attn.v_scale_float = attn.v_scale.item()

    def _load_fp8_static_scales(
        self,
        checkpoint: SafetensorsCheckpoint,
        device: torch.device,
        fp8_qkv: bool,
        fp8_o: bool,
        fp8_mlp: bool,
    ):
        """Load per-tensor STATIC FP8 scales into module buffers.

        The HF checkpoint stores a BF16 scalar ``<proj>.weight_scale_inv`` (dequant
        multiplier: ``w_bf16 = w_fp8 * weight_scale_inv``) and a BF16 scalar
        ``<proj>.activation_scale`` (the matching input *dequant* scale) for each
        projection. The nkilib STATIC kernels consume the *dequant* scales directly
        (they compute ``quant = 1/scale`` internally), so we pass both through as-is.

        Scales are per-tensor scalars → shard-invariant (replicate, no slicing).
        Shapes target the decode TKG kernel (``[PMAX,·]`` = ``[128,·]``); the prefill
        CTE kernel accepts the same. qkv weight scale is ``[128,3]`` with columns in
        Q|K|V order (matching the fused weight); all other scales are ``[128,1]``.
        gate/up share one input scale (``gate_up_in_scale``); q/k/v share one
        ``qkv_in_scale`` (verified equal in the checkpoint).
        """
        PMAX = 128

        def scalar(key: str) -> torch.Tensor:
            # weight_scale_inv / activation_scale are 0-dim scalars (shape []);
            # index with Ellipsis (``[...]``), NOT ``[:]`` which raises on a 0-dim
            # tensor (matches the scalar-read idiom in weight_loaders.py). Cast to
            # float32 on CPU here; the device move happens as a SEPARATE step in
            # ``to_dev`` — on the Neuron device a single ``.to()`` cannot both
            # change dtype and move device (it raises "self.dtype()==dst.dtype()").
            return checkpoint._get_slice(key)[...].to(torch.float32).reshape(())

        def to_dev(t: torch.Tensor) -> torch.Tensor:
            # Pure device move (dtype unchanged) — safe on Neuron.
            return t.to(device=device)

        def col(key: str) -> torch.Tensor:  # [128,1]
            return to_dev(scalar(key).expand(PMAX, 1).contiguous())

        # 🔴 240/448 RANGE FIX (2026-08-10 00:10 JST) — see the long rationale in
        # weight_loaders.py. The fp8 WEIGHTS are downscaled by 240/448 at load so they fit
        # Neuron's native e4m3 grid; their dequant scales must be multiplied by 448/240 to
        # keep ``w_fp8 * weight_scale`` mathematically equal to the OCP-calibrated original.
        # 🔴 WEIGHT scales ONLY. ``activation_scale`` must NOT be compensated — activations
        # were never rescaled. llama3 draws the identical line via ``is_weight_scale``
        # (``llama3/model_static_fp8.py:519`` compensates, ``:523`` does not).
        # ``compensate_weight_scale`` is a no-op when the fix is disabled.
        from vllm_neuron.model.ministral3.weight_loaders import (
            compensate_weight_scale,
            recalibrate_activation_scale,
        )

        def col_w(key: str) -> torch.Tensor:  # [128,1], WEIGHT scale (compensated)
            return to_dev(
                compensate_weight_scale(scalar(key)).expand(PMAX, 1).contiguous()
            )

        # 🎯 ACTIVATION scales: recalibrate the 448-convention scale for TRN2's ±240 range
        # (``× 448/240``). Opt-in via ``MINISTRAL3_ACT_SCALE_RECAL=1``; a no-op otherwise, so
        # every previously recorded measurement stays comparable. Rationale, proof of the
        # calibration convention, and the MEASURED CEILING (this alone does not reach the
        # golden) are in weight_loaders.py next to ``recalibrate_activation_scale``.
        def col_a(key: str, family: str) -> torch.Tensor:  # [128,1], ACTIVATION scale
            return to_dev(
                recalibrate_activation_scale(scalar(key), family)
                .expand(PMAX, 1)
                .contiguous()
            )

        for layer_id in range(len(self.model.layers)):
            attn = self.model.layers[layer_id].self_attn
            mlp = self.model.layers[layer_id].mlp
            p = f"model.layers.{layer_id}"

            # 🔬 per-layer: use the SAME gate as the module and the weight loader,
            # so a layer that runs BF16 never gets FP8 scale buffers (harmless but
            # confusing) and a layer that runs FP8 never misses them (fatal).
            l_qkv = self.config.fp8_enabled("qkv", layer_id)
            l_o = self.config.fp8_enabled("o_proj", layer_id)
            l_mlp = self.config.fp8_enabled("mlp", layer_id)

            if l_qkv:
                # [128,3] weight scale, columns Q|K|V (fused-weight order).
                # WEIGHT scales -> compensated by 448/240 (see col_w rationale above).
                q_w = compensate_weight_scale(
                    scalar(f"{p}.self_attn.q_proj.weight_scale_inv")
                )
                k_w = compensate_weight_scale(
                    scalar(f"{p}.self_attn.k_proj.weight_scale_inv")
                )
                v_w = compensate_weight_scale(
                    scalar(f"{p}.self_attn.v_proj.weight_scale_inv")
                )
                qkv_w = to_dev(
                    torch.stack([q_w, k_w, v_w]).reshape(1, 3).expand(PMAX, 3).contiguous()
                )
                # Single input scale (q/k/v activation_scale are equal); use q's.
                qkv_in = col_a(f"{p}.self_attn.q_proj.activation_scale", "qkv")
                attn.qkv_w_scale = qkv_w
                attn.qkv_in_scale = qkv_in

            if l_o:
                attn.o_proj_w_scale = col_w(f"{p}.self_attn.o_proj.weight_scale_inv")
                attn.o_proj_in_scale = col_a(f"{p}.self_attn.o_proj.activation_scale", "o_proj")
                # 🎯 ROW variant of the weight scale: [P_MAX, H] instead of [P_MAX, 1].
                # ⚠️ HONEST LIMITATION: the checkpoint only ships a per-TENSOR scalar, so every
                # "row" gets the same value. That forfeits ROW's per-output-channel weight
                # precision — but the point of this arm is the INPUT side, where ROW computes a
                # per-token absmax on device and therefore needs no calibrated input scale at all.
                # ⚠️ The scale must NOT carry the 448/240 activation recalibration here: with ROW
                # the input scale is derived on device, so only the weight factor belongs in it.
                if (os.environ.get("OPROJ_QUANT_MODE") or "").strip().lower() == "row":
                    # H from the actual parameter, not from config: o_proj_weight is
                    # [N*D/shards, H] and only the first dim is TP-sharded.
                    _h = attn.o_proj_weight.shape[1]
                    _ws = compensate_weight_scale(
                        scalar(f"{p}.self_attn.o_proj.weight_scale_inv")
                    )
                    attn.o_proj_w_scale_row = to_dev(
                        _ws.expand(PMAX, _h).contiguous()
                    )
                # EXPERIMENT ONLY (2026-08-09 23:00 JST, hypothesis test — default is a NO-OP).
                # RCA (TEST_PLAN 22:51–23:05 JST) attributes the numeric-collapse defect to the
                # static FP8 on o_proj, with this mechanism: the checkpoint's activation_scale is
                # calibrated for OCP e4m3 (±448) but TRN2's native e4m3 saturates at ±240
                # (model.py:75, nkilib kernel_helpers.py:463). o_proj's scale is 15–38x smaller
                # than every other family's, so its clip threshold is scale*240 = 1.67 (vs 12–63
                # elsewhere) and it is the only family with no headroom to absorb the 46% range
                # reduction. The kernel computes clamp(x / scale, ±240), so ENLARGING the scale
                # raises the clip threshold and should remove the collapse if saturation is the
                # cause. This knob multiplies o_proj's input scale only:
                #   OPROJ_IN_SCALE_MULT=1.8667  ->  restores the GPU-equivalent 448/240 headroom
                # A falsifiable prediction: if saturation is the mechanism the collapse must
                # weaken monotonically as this rises; if it does not, the mechanism is wrong.
                _oproj_mult = float(os.environ.get("OPROJ_IN_SCALE_MULT", "1.0"))
                if _oproj_mult != 1.0:
                    attn.o_proj_in_scale = attn.o_proj_in_scale * _oproj_mult

            if l_mlp:
                mlp.gate_w_scale = col_w(f"{p}.mlp.gate_proj.weight_scale_inv")
                mlp.up_w_scale = col_w(f"{p}.mlp.up_proj.weight_scale_inv")
                mlp.down_w_scale = col_w(f"{p}.mlp.down_proj.weight_scale_inv")
                # gate/up share one input scale (activation_scale equal); use gate's.
                mlp.gate_up_in_scale = col_a(f"{p}.mlp.gate_proj.activation_scale", "mlp")
                mlp.down_in_scale = col_a(f"{p}.mlp.down_proj.activation_scale", "mlp")
                # 🎯 ROW variant of the MLP weight scales: [P_MAX, I] / [P_MAX, H].
                # WHY THIS ARM EXISTS (ACCURACY_CONSTRAINTS §2.4): the TP=32 residual is
                # MLP-side, shard-width dependent (TP=8 has Delta ~ 0, TP=32 has -0.107 at
                # k<=4 for the SAME quantization and calibration), and it is NOT a scalar
                # mis-sizing -- widening the static input scale by 448/240 is 66% worse and
                # narrowing it buys <=10%. What is left is that the correct scale differs per
                # rank/token, which no single scalar can express. ROW removes the calibrated
                # input scale entirely: the kernel takes a per-token absmax on device
                # (nkilib/core/mlp/mlp_torch.py:423 _row_quantize).
                # ⚠️ HONEST LIMITATION, same as o_proj: the checkpoint ships only per-TENSOR
                # weight scalars, so every "row" of the WEIGHT scale gets the same value and
                # ROW's per-output-channel weight precision is forfeited. The point of this arm
                # is the INPUT side.
                # ⚠️ The activation recalibration must NOT be folded in here (col_a applies it);
                # with ROW the input scale is derived on device, so only the weight factor
                # belongs in these tensors -- hence col_w, not col_a.
                # ⚠️ These go through ``compensate_weight_scale`` exactly like ``col_w``:
                # omitting it would silently break the arm whenever
                # MINISTRAL3_FP8_RANGE_FIX=1 (it is a no-op otherwise).
                # ⚠️ MEMORY: gate/up are [128, I_per_rank] but down is [128, HIDDEN] and HIDDEN
                # is NOT sharded by the MLP (gate/up are column-parallel, down is row-parallel),
                # so down alone is 128x12288xf32 = 6.3 MB per layer => ~635 MB/core over 88
                # layers at TP=32. Affordable against 24 GB/core but it does eat KV cache;
                # o_proj's ROW arm already pays the same cost, which is why this is opt-in.
                if _mlp_quant_mode() == "row":
                    def _w_row(key: str, width: int) -> torch.Tensor:
                        return to_dev(
                            compensate_weight_scale(scalar(key))
                            .expand(PMAX, width).contiguous()
                        )
                    _I = mlp.up_proj_weight.shape[-1]
                    _H = mlp.down_proj_weight.shape[-1]
                    mlp.gate_w_scale_row = _w_row(
                        f"{p}.mlp.gate_proj.weight_scale_inv", _I
                    )
                    mlp.up_w_scale_row = _w_row(f"{p}.mlp.up_proj.weight_scale_inv", _I)
                    mlp.down_w_scale_row = _w_row(
                        f"{p}.mlp.down_proj.weight_scale_inv", _H
                    )
                # EXPERIMENT ONLY (2026-08-18 JST, hypothesis test — default is a NO-OP).
                # Mirrors OPROJ_IN_SCALE_MULT above, for the MLP input scales.
                #
                # WHY: teacher-forced measurement (ACCURACY_CONSTRAINTS §2.4, E4) shows the TP=32
                # residual is MLP-side and concentrated in the first few generated tokens, and that
                # ENLARGING these scales by 448/240 (E2, MINISTRAL3_ACT_SCALE_RECAL="mlp") makes it
                # ~66% WORSE at k<=4. So these scales are not too tight -- they are too WIDE, and
                # the loss is resolution, not clipping. The proposed mechanism: the static scale is
                # calibrated on the whole-tensor amax, but TP=32 hands each rank a 4x narrower
                # activation shard (down_proj input: 3,584 elements per rank at TP=8 -> 896 at
                # TP=32) whose local amax is smaller as an order statistic, so the scale over-
                # provisions range and spends FP8 levels on values that never occur.
                #
                # A falsifiable prediction: if resolution loss on a narrower shard is the
                # mechanism, |Delta logprob| at k<=4 must fall as this multiplier goes BELOW 1.0,
                # with a minimum somewhere in (0, 1). If it only ever worsens, the mechanism is
                # wrong (or clipping binds first and the two effects cannot be separated).
                #
                # ⚠️ Applies to BOTH gate/up and down input scales. Set MINISTRAL3_MLP_IN_SCALE_MULT
                # _DOWN_ONLY=1 to restrict it to down_proj (that is where the 197x layer-2 outlier
                # lives, §1.2), which separates "the outlier layer" from "the whole family".
                _mlp_mult = float(os.environ.get("MINISTRAL3_MLP_IN_SCALE_MULT", "1.0"))
                if _mlp_mult != 1.0:
                    _down_only = os.environ.get(
                        "MINISTRAL3_MLP_IN_SCALE_MULT_DOWN_ONLY", "0"
                    ) == "1"
                    if not _down_only:
                        mlp.gate_up_in_scale = mlp.gate_up_in_scale * _mlp_mult
                    mlp.down_in_scale = mlp.down_in_scale * _mlp_mult
