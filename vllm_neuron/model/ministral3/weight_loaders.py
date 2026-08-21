# SPDX-License-Identifier: Apache-2.0
"""
Ministral3 FP8 weight loaders
=============================

<-- MODEL-SPECIFIC: The Ministral3 checkpoint (e.g. Devstral-2-123B) stores the
seven linear projections (q/k/v/o_proj, gate/up/down_proj) in FP8 (E4M3) with a
**per-tensor scalar** dequantization scale named ``<proj>.weight_scale_inv`` and
an unused static ``<proj>.activation_scale``. Dequantization is simply:

    w_bf16 = w_fp8.to(bfloat16) * weight_scale_inv

Embedding, norms, and lm_head are already BF16 in the checkpoint.

These loaders wrap the generic loaders in ``vllm_neuron/utils/weight_loader.py``
so that TP/SP sharding is unchanged — the only addition is the scalar dequant,
applied after slicing. The FP8 slice is sharded first (cheap, reads only this
rank's shard from disk), cast to BF16, then multiplied by the scalar.

**Adaptive to checkpoint format.** Each loader inspects the *number of slices*
it receives and dispatches accordingly:

  - single projection: 1 slice  -> BF16 (no dequant), 2 slices -> FP8 (+scale)
  - fused QKV:          3 slices -> BF16,              6 slices -> FP8 (+3 scales)

This means the model does NOT need to know up front whether the checkpoint is
FP8 — ``load_weights`` simply includes the ``weight_scale_inv`` keys in the
mapping when they exist in the checkpoint. This is robust even when the HF
``quantization_config`` is emptied via ``--hf-overrides`` (required so vLLM does
not reject the unsupported ``fp8`` method) because detection is driven by the
actual checkpoint tensors, not the config.
"""

import os

import torch

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    sharding_weight_loader,
)

# ── FP8-native constants ────────────────────────────────────────────────────
# Neuron gen3/TRN2 reinterprets ``torch.float8_e4m3fn`` bytes under its native
# e4m3 table whose exponent field 1111 is RESERVED, capping the finite range at
# ±240 (vs OCP e4m3fn's ±448). Devstral's HF checkpoint is OCP and *does* contain
# a handful of codes in (240, 448] (verified: layer-0 q_proj 182 codes incl. a
# literal 448.0, k_proj 20) which would decode to NaN on gen3 and poison the
# matmul. We saturate those rare codes onto the ±240 grid in raw-byte space; this
# is exact (post-clamp bytes decode identically under both the OCP and Neuron
# tables, all finite) and leaves the per-tensor scale untouched. It is NOT the
# whole-tensor ×(448/240) rescale.
_E4M3_EXP1111_LO = 0x78  # first byte with (b & 0x7F) in the reserved exp=1111 band
_E4M3_PLUS_240 = 0x77  # max finite e4m3 code (|value| == 240)

# ── ⚠️ THE 240/448 DOWNSCALE — TRIED AND REJECTED FOR THIS CHECKPOINT ──────
#            (added 2026-08-10 00:05 JST, disabled by default 00:20 JST)
#
# ❌ **Do not enable this by default.** It was implemented after noticing that
# ``llama3/weight_loaders_static_fp8.py:45-124`` does a 240/448 weight downscale plus a
# 448/240 weight-scale compensation, which this port does not. A CPU A/B against the real
# checkpoint then showed the downscale is **4 orders of magnitude WORSE here**:
#
#   family      |w|>240 fraction   relative-L1 saturate-only   relative-L1 downscale
#   o_proj      0.00000%           0.0                          2.13e-02
#   q_proj      0.00004%           1.34e-06                     2.13e-02
#   gate_proj   0.00000%           0.0                          2.12e-02
#   down_proj   0.00000%           1.43e-07                     2.13e-02
#
# Why: ``×240/448`` moves **every** element off the fp8 grid, so re-quantization rounding
# error lands on **all** of them (measured: ~933k of 1M elements). Saturation touches only
# the handful above 240 — and in **this** checkpoint that is 0.00004%, i.e. a few elements
# per tensor. llama3's downscale is right for **ModelOpt** checkpoints, whose values do
# populate ``(240, 448]``; it is wrong for Devstral's HF checkpoint, which effectively
# does not. **The original saturate-only comment above was correct.**
#
# ⇒ Kept behind ``MINISTRAL3_FP8_RANGE_FIX=1`` for A/B only. Default OFF.
# ⇒ Consequence for the RCA: the o_proj **weights** are exonerated, which leaves the
#    **activation** side as the remaining suspect (README §1.6c).
#
# On "can TRN2 not just use OCP e4m3?": no — TRN2's native e4m3 reserves exponent field
# 1111, so ±448 is not representable at all. The byte saturation below exists precisely
# because those codes would otherwise decode to NaN.
_FP8_RANGE_FIX_DEFAULT = False  # 🔴 measured to be a regression on this checkpoint
_FP8_E4M3FN_MAX = 448.0                   # OCP e4m3fn max finite (what the ckpt uses)
_FP8_E4M3_MAX_VAL = 240.0                 # Neuron gen3/TRN2 native e4m3 max finite
_FP8_WEIGHT_DOWNSCALE = _FP8_E4M3_MAX_VAL / _FP8_E4M3FN_MAX   # 240/448, on the WEIGHT
_FP8_SCALE_COMPENSATION = _FP8_E4M3FN_MAX / _FP8_E4M3_MAX_VAL  # 448/240, on the W scale


def _fp8_range_fix_enabled() -> bool:
    """True only when explicitly opted into via ``MINISTRAL3_FP8_RANGE_FIX=1``.

    🔴 **Default OFF, deliberately.** Unlike llama3's ``_needs_downscale()`` this does NOT
    auto-enable on trn2: the CPU A/B above shows the downscale is a **regression** on this
    checkpoint (relative-L1 1.3e-06 -> 2.1e-02). The env var exists so the comparison can
    be reproduced, not so it can be switched on in production.
    """
    forced = os.environ.get("MINISTRAL3_FP8_RANGE_FIX", "").strip().lower()
    if forced in ("1", "on", "true"):
        return True
    return _FP8_RANGE_FIX_DEFAULT


def _downscale_e4m3_to_neuron(w_fp8: torch.Tensor) -> torch.Tensor:
    """Rescale an OCP-calibrated fp8 weight into Neuron's +/-240 grid (llama3 pattern).

    Mirrors ``llama3/weight_loaders_static_fp8.py::_downscale_fp8_weight``: multiply by
    240/448 in float space, clamp, and cast back. The matching ``448/240`` compensation
    must be applied to the WEIGHT dequant scale (see ``_compensate_weight_scale``), or
    the projection comes out ~1.87x too small.
    """
    assert w_fp8.dtype == torch.float8_e4m3fn, (
        f"_downscale_e4m3_to_neuron expects float8_e4m3fn, got {w_fp8.dtype}"
    )
    return (
        (w_fp8.float() * _FP8_WEIGHT_DOWNSCALE)
        .clamp(-_FP8_E4M3_MAX_VAL, _FP8_E4M3_MAX_VAL)
        .to(torch.float8_e4m3fn)
    )


def _prepare_fp8_weight_for_neuron(w_fp8: torch.Tensor) -> torch.Tensor:
    """Range-correct an fp8 weight for the STATIC kernels.

    With the fix enabled: ``×240/448`` then clamp (llama3 pattern, information-preserving
    in relative terms). Disabled: the historical byte-saturate-only path, kept so the
    defect can be reproduced for A/B.
    """
    # 🔴 既定は「飽和のみ」。CPU 実測（2026-08-10 00:20 JST）で、このチェックポイントでは
    # ダウンスケールが **改悪** であると判明したため（相対L1 誤差 1.3e-06 → 2.1e-02、4桁悪化）。
    # 理由: ×240/448 は全要素を fp8 グリッド上でずらすので **全要素に再量子化の丸め誤差**が入る
    # （実測 93万/100万要素）。一方このチェックポイントで |w|>240 は **0.00004%** しかないので、
    # 飽和で失われるのは数要素だけ。llama3 の方式が正しいのは ModelOpt チェックポイント
    # （(240,448] に値が多く分布する）で、**Devstral の HF チェックポイントには当てはまらない**。
    # ⇒ MINISTRAL3_FP8_RANGE_FIX=1 は A/B 用に残すが、**既定では使わないこと**。
    if _fp8_range_fix_enabled():
        return _downscale_e4m3_to_neuron(w_fp8)
    return _saturate_e4m3_to_neuron_(w_fp8)


# ── 🎯 ACTIVATION-SCALE RECALIBRATION FOR TRN2's ±240 RANGE ─────────────────
#                        (2026-08-10 00:15 JST)
#
# THIS is the fix that follows from the measurements, and it is separate from the weight
# downscale rejected above.
#
# The calibration convention is now PROVEN, not assumed: for every family the stored fp8
# weight has ``max|w| == 448.0`` exactly, and ``dequant amax == weight_scale_inv × 448``.
# So the checkpoint was calibrated as ``scale = amax / FP8_MAX`` with **FP8_MAX = 448**
# (OCP e4m3, what an H100 implements). TRN2's native e4m3 tops out at **240**, so on this
# platform the correct per-tensor scale is ``amax / 240``, i.e.
#
#     scale_trn2 = scale_ckpt × (448 / 240) = scale_ckpt × 1.8667
#
# Applied to the ACTIVATION (input) scales, this raises the clip threshold
# (``scale × 240``) back to what the calibration intended.
#
# ⚠️ MEASURED CEILING — do not oversell this. Applying exactly this factor to o_proj (via
# the OPROJ_IN_SCALE_MULT knob, which is arithmetically the same thing) improves the output
# but does **NOT** reach the golden: powers-of-2 agrees on 13 of 37 numbers, the descending
# list on 8 of 150 and then repeats the prompt. **A correctly recalibrated scale is not
# sufficient**, so a residual cause remains (candidates: resolution loss at 8 bits,
# subnormal collapse — o_proj's scale sits near e4m3's smallest normal 2**-6 — or a kernel
# detail). Enable this as the best-known-correct scale, not as a cure.
_ACT_SCALE_TRN2_RECAL = _FP8_E4M3FN_MAX / _FP8_E4M3_MAX_VAL  # 448/240 = 1.8667


#
# 🔴 SCOPE IS PER-FAMILY, AND THAT MATTERS. The 448 convention is a property of the whole
# checkpoint, not of o_proj — so in principle every family's activation scale is off by the
# same 448/240. But the other families have ample headroom (clip threshold scale×240 is
# 12–63 for q_proj/gate_proj/down_proj vs **1.67** for o_proj) and they currently work, so
# recalibrating them could just as easily be a REGRESSION: a larger scale means a coarser
# 8-bit step, i.e. resolution traded for range that those families do not need.
# ⇒ Therefore the knob takes a FAMILY LIST, defaulting to none, so "recalibrate o_proj
#   only" can be tested separately from "recalibrate everything".
_ACT_RECAL_FAMILIES = ("qkv", "o_proj", "mlp")


def _act_recal_selection() -> frozenset:
    """Which families get the 448→240 activation-scale recalibration.

    ``MINISTRAL3_ACT_SCALE_RECAL`` accepts:
      unset / "" / "0" / "off"  -> no family (default; byte-equivalent to shipped behaviour)
      "all" / "1" / "on"        -> every family
      "o_proj" / "o_proj,mlp"   -> that comma-separated subset

    A per-family list exists because recalibration is only clearly motivated for the family
    with no headroom (``o_proj``); applying it to the others may cost resolution for range
    they do not need. Unknown names raise rather than being silently ignored.
    """
    raw = os.environ.get("MINISTRAL3_ACT_SCALE_RECAL", "").strip().lower()
    if raw in ("", "0", "off", "false", "none"):
        return frozenset()
    if raw in ("1", "on", "true", "all"):
        return frozenset(_ACT_RECAL_FAMILIES)
    fams = {f.strip() for f in raw.split(",") if f.strip()}
    unknown = fams - set(_ACT_RECAL_FAMILIES)
    if unknown:
        raise ValueError(
            f"MINISTRAL3_ACT_SCALE_RECAL: unknown family/families {sorted(unknown)}; "
            f"expected a comma-separated subset of {list(_ACT_RECAL_FAMILIES)}, "
            "'all', or unset."
        )
    return frozenset(fams)


def recalibrate_activation_scale(scale: torch.Tensor, family: str) -> torch.Tensor:
    """Rescale a checkpoint ACTIVATION scale from the 448 convention to TRN2's 240.

    ``family`` must be one of ``qkv`` / ``o_proj`` / ``mlp``; the rescale applies only if
    that family is selected by ``MINISTRAL3_ACT_SCALE_RECAL`` (default: none, a no-op).
    Weight scales are handled separately by ``compensate_weight_scale``.
    """
    assert family in _ACT_RECAL_FAMILIES, (
        f"recalibrate_activation_scale: unknown family {family!r}, "
        f"expected one of {list(_ACT_RECAL_FAMILIES)}"
    )
    if family not in _act_recal_selection():
        return scale
    return scale * _ACT_SCALE_TRN2_RECAL


def compensate_weight_scale(scale: torch.Tensor) -> torch.Tensor:
    """Apply the ``448/240`` compensation to a WEIGHT dequant scale.

    🔴 WEIGHT scales only. ACTIVATION/input scales must NOT be compensated — the
    activations were never rescaled (llama3 draws the same line via
    ``is_weight_scale``). A no-op when the range fix is disabled.
    """
    if not _fp8_range_fix_enabled():
        return scale
    return scale * _FP8_SCALE_COMPENSATION


def _saturate_e4m3_to_neuron_(w_fp8: torch.Tensor) -> torch.Tensor:
    """In-place saturate OCP e4m3fn codes >240 onto the Neuron ±240 grid.

    Operates on the raw bytes (no value conversion): any code whose magnitude
    field is in the reserved exp=1111 band ``(b & 0x7F) >= 0x78`` is clamped to
    ``±240`` (``(b & 0x80) | 0x77``), preserving sign. Codes ≤ 0x77 are bit-
    identical between the OCP and Neuron tables, so untouched weights are exact.
    Returns the same tensor (still ``float8_e4m3fn``) for chaining.
    """
    assert w_fp8.dtype == torch.float8_e4m3fn, (
        f"_saturate_e4m3_to_neuron_ expects float8_e4m3fn, got {w_fp8.dtype}"
    )
    # NOTE: ``.contiguous()`` COPIES when the input is non-contiguous (e.g. a
    # transposed shard from ``is_storage_transposed=True``). The byte-clamp below
    # mutates whatever tensor ``u`` views, so we must keep and return THAT tensor
    # — returning the original ``w_fp8`` would silently drop the saturation on a
    # non-contiguous weight (the o_proj/down_proj failure mode: OCP codes >240
    # survive → decode to NaN on gen3 → garbage output).
    w_contig = w_fp8.contiguous()
    u = w_contig.view(torch.uint8)
    hi = (u & 0x7F) >= _E4M3_EXP1111_LO
    if bool(hi.any()):
        u[hi] = (u[hi] & 0x80) | _E4M3_PLUS_240
    return w_contig


def _scale_to(scale_slice, rows: int, cols: int) -> torch.Tensor:
    """Materialize a checkpoint per-tensor scalar scale into an fp32 [rows, cols].

    The HF scales are BF16 scalars (shape []). The kernels broadcast ``[1, c]``
    to ``[128, c]``; we emit ``[128, c]`` directly so the same buffer satisfies
    both the prefill CTE path (accepts [1,·] or [128,·]) and the decode TKG path
    (wants [PMAX,·] = [128,·]). Value is replicated across all rows/cols.
    """
    val = scale_slice[...].to(torch.float32).reshape(())
    return val.expand(rows, cols).contiguous()


def _dequant(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Per-tensor FP8 -> BF16 dequant. ``scale`` is a scalar (shape [])."""
    return tensor.to(torch.bfloat16) * scale.to(torch.bfloat16)


class _EagerSlice:
    """Minimal PySafeSlice stand-in backed by an in-memory tensor.

    The generic ``fused_qkv_weight_loader`` calls ``get_shape()`` and indexes
    with ``[tuple_of_slices]``; a materialized BF16 tensor supports both. We
    need it because FP8 weights must be dequantized (materialized) before
    fusion, so the lazy safetensors slices can no longer be passed through.
    """

    def __init__(self, tensor: torch.Tensor):
        self._t = tensor

    def get_shape(self):
        return list(self._t.shape)

    def __getitem__(self, idx):
        return self._t[idx]


def adaptive_sharding_weight_loader(
    shard_dim: int,
    shard_size: int,
    num_shards: int,
    is_storage_transposed: bool = False,
) -> SafetensorsWeightLoader:
    """Sharding loader that handles both BF16 (1 slice) and FP8 (2 slices).

    BF16: ``[weight]`` -> generic sharding.
    FP8:  ``[weight_fp8, weight_scale_inv]`` -> shard then per-tensor dequant.
    """
    base = sharding_weight_loader(
        shard_dim=shard_dim,
        shard_size=shard_size,
        num_shards=num_shards,
        is_storage_transposed=is_storage_transposed,
    )

    def transform(slices, rank):
        if len(slices) == 1:
            return base.transform(slices, rank)
        assert len(slices) == 2, (
            "adaptive_sharding_weight_loader expects [weight] or "
            "[weight, weight_scale_inv]"
        )
        weight = base.transform([slices[0]], rank)
        scale = slices[1][...]  # scalar
        return _dequant(weight, scale)

    return SafetensorsWeightLoader(transform=transform)


def adaptive_fused_qkv_weight_loader(
    q_size: int,
    kv_size: int,
    shard_dim: int,
    num_shards: int,
    is_storage_transposed: bool = False,
    num_kv_replicas: int = 1,
    attention_dp_rank: int = 0,
    attention_dp_size: int = 1,
    kv_sharded_across_attention_dp: bool = False,
) -> SafetensorsWeightLoader:
    """Fused QKV loader handling both BF16 (3 slices) and FP8 (6 slices).

    BF16: ``[q_w, k_w, v_w]`` -> generic fused QKV.
    FP8:  ``[q_w, k_w, v_w, q_scale, k_scale, v_scale]`` -> dequant each
          projection with its own scalar scale, then fuse/shard via the generic
          loader. Q, K, V have independent quantization scales, so they must be
          dequantized before fusion.
    """
    base = fused_qkv_weight_loader(
        q_size=q_size,
        kv_size=kv_size,
        shard_dim=shard_dim,
        num_shards=num_shards,
        is_storage_transposed=is_storage_transposed,
        num_kv_replicas=num_kv_replicas,
        attention_dp_rank=attention_dp_rank,
        attention_dp_size=attention_dp_size,
        kv_sharded_across_attention_dp=kv_sharded_across_attention_dp,
    )

    def transform(slices, rank):
        if len(slices) == 3:
            return base.transform(slices, rank)
        assert len(slices) == 6, (
            "adaptive_fused_qkv_weight_loader expects [q,k,v] or "
            "[q,k,v,q_scale,k_scale,v_scale]"
        )
        q_w, k_w, v_w, q_s, k_s, v_s = slices
        deq = [
            _EagerSlice(_dequant(q_w[...], q_s[...])),
            _EagerSlice(_dequant(k_w[...], k_s[...])),
            _EagerSlice(_dequant(v_w[...], v_s[...])),
        ]
        return base.transform(deq, rank)

    return SafetensorsWeightLoader(transform=transform)


# =============================================================================
# FP8-native loaders (no BF16 dequant). Keep the weight in float8_e4m3fn, fuse/
# shard exactly like the generic loaders (byte-slice + .T + cat are all dtype-
# preserving for fp8), then saturate OCP codes >240 onto the Neuron ±240 grid.
# The per-tensor scales are NOT applied here — they ride alongside as separate
# buffers (loaded in model.load_weights) and are consumed by the STATIC kernels.
# These are used only for the projection families enabled via
# ``Ministral3Config.dense_fp8_static``; other families keep the adaptive
# BF16-dequant loaders above.
# =============================================================================


def fp8_native_sharding_weight_loader(
    shard_dim: int,
    shard_size: int,
    num_shards: int,
    is_storage_transposed: bool = False,
) -> SafetensorsWeightLoader:
    """Shard a single FP8 weight (keep ``float8_e4m3fn``) + ±240 saturation.

    Mirrors ``sharding_weight_loader`` but does NOT cast to BF16 and does NOT
    apply the per-tensor scale. Expects exactly one slice (the fp8 weight); the
    matching ``weight_scale_inv`` / ``activation_scale`` are loaded separately
    into scale buffers.
    """
    base = sharding_weight_loader(
        shard_dim=shard_dim,
        shard_size=shard_size,
        num_shards=num_shards,
        is_storage_transposed=is_storage_transposed,
    )

    def transform(slices, rank):
        assert len(slices) == 1, (
            "fp8_native_sharding_weight_loader expects a single [weight] slice "
            "(scales are loaded separately as buffers)"
        )
        weight = base.transform(slices, rank)
        assert weight.dtype == torch.float8_e4m3fn, (
            f"fp8_native loader expected float8_e4m3fn weight, got {weight.dtype} "
            "(is this checkpoint actually FP8?)"
        )
        return _prepare_fp8_weight_for_neuron(weight)

    return SafetensorsWeightLoader(transform=transform)


def fp8_native_fused_qkv_weight_loader(
    q_size: int,
    kv_size: int,
    shard_dim: int,
    num_shards: int,
    is_storage_transposed: bool = False,
    num_kv_replicas: int = 1,
    attention_dp_rank: int = 0,
    attention_dp_size: int = 1,
    kv_sharded_across_attention_dp: bool = False,
) -> SafetensorsWeightLoader:
    """Fuse Q/K/V FP8 weights (keep ``float8_e4m3fn``) + ±240 saturation.

    Unlike the BF16 ``adaptive_fused_qkv_weight_loader``, FP8 fusion needs NO
    pre-dequant: Q/K/V are concatenated along the fused (column) dim while still
    fp8, and the three independent per-tensor weight scales ride separately as a
    ``[128, 3]`` ``qkv_w_scale`` buffer (column order Q|K|V, matching the fused
    layout). Expects exactly three slices ``[q_w, k_w, v_w]``.
    """
    base = fused_qkv_weight_loader(
        q_size=q_size,
        kv_size=kv_size,
        shard_dim=shard_dim,
        num_shards=num_shards,
        is_storage_transposed=is_storage_transposed,
        num_kv_replicas=num_kv_replicas,
        attention_dp_rank=attention_dp_rank,
        attention_dp_size=attention_dp_size,
        kv_sharded_across_attention_dp=kv_sharded_across_attention_dp,
    )

    def transform(slices, rank):
        assert len(slices) == 3, (
            "fp8_native_fused_qkv_weight_loader expects [q_w, k_w, v_w] slices "
            "(scales are loaded separately as buffers)"
        )
        fused = base.transform(slices, rank)
        assert fused.dtype == torch.float8_e4m3fn, (
            f"fp8_native qkv loader expected float8_e4m3fn weight, got {fused.dtype}"
        )
        return _prepare_fp8_weight_for_neuron(fused)

    return SafetensorsWeightLoader(transform=transform)
