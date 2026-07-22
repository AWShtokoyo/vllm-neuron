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
        return _saturate_e4m3_to_neuron_(weight)

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
        return _saturate_e4m3_to_neuron_(fused)

    return SafetensorsWeightLoader(transform=transform)
