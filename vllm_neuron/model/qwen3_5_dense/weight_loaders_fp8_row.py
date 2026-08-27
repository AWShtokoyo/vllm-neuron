# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 -> per-channel ROW FP8 load-time re-quantization loaders.

The ``Qwen3.8-27B-FP8`` checkpoint stores each quantized ``nn.Linear`` as a
DeepSeek block-FP8 pair:

    ``.weight``            float8_e4m3fn, HF shape ``[out, in]``
    ``.weight_scale_inv``  bf16, shape ``[out/128, in/128]`` (one scalar per
                           ``[128, 128]`` block); dequant convention (vLLM
                           w8a8_block_fp8) is ``bf16 = w_fp8.float() * scale``.

Neuron's NKI GEMMs cannot consume a resident block-``[128, 128]`` scale, so at
load we DEQUANT each block back to fp32 and RE-QUANT to **per-output-channel
ROW** FP8: one scale per output channel = ``absmax(row) / fp8_max``. The kernel
then dequants ``w_fp8 * row_scale`` and the activations are quantized
dynamically per-token (``NF.rmsnorm_quant(ROW)``), matching the checkpoint's
``activation_scheme: "dynamic"``.

ROW weight-scale contract (nkilib mlp.py / output_projection): shape
``[128, out_dim]`` fp32 -- 128 broadcast partition-rows, one column per output
channel. On **trn2** the ROW NKI kernels read FP8 tiles as legacy
``nl.float8_e4m3`` (max 240, not the OCP 448); we therefore re-quant into the
240 range directly (``fp8_max = 240``), so every stored byte round-trips (values
<= 240 share the same encoding between ``float8_e4m3fn`` and legacy
``float8_e4m3``). trn3 uses the OCP 448 range.

Each quantized param needs BOTH an FP8 weight tensor and an ``[128, out]`` scale
buffer; both derive from the SAME checkpoint ``[weight, weight_scale_inv]`` pair.
The factories below take ``want_scale`` to select which piece the loader emits
(the weight loader and the scale loader run independently and each recomputes
the shared dequant/requant -- cheap, load-time, CPU-side; mirrors the
``mxfp8`` down-requant loaders).
"""

from __future__ import annotations

import torch

from vllm_neuron.model.qwen3_vl.weight_loaders_mxfp8 import qkv_shard_offsets
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

# DeepSeek square block size (both weight dims).
_BLK = 128
# Partition dimension the ROW kernels expect for the broadcast scale.
_PMAX = 128
_FP8_DTYPE = torch.float8_e4m3fn
# OCP e4m3fn max (trn3) vs legacy e4m3 max (trn2 ROW kernel SBUF dtype).
_FP8_E4M3FN_MAX = 448.0
_FP8_E4M3_MAX = 240.0


def _target_fp8_max() -> float:
    """FP8 max to re-quantize into for the current platform.

    trn2 ROW kernels read FP8 tiles as legacy ``nl.float8_e4m3`` (max 240); trn3
    uses OCP ``float8_e4m3fn`` (max 448). Queried on the loader thread. Falls
    back to the trn2 (240) range off-device so a CPU unit test is deterministic.
    """
    try:
        from libtorch_neuronx_lite.compile.platform import get_platform_target

        return _FP8_E4M3_MAX if get_platform_target() == "trn2" else _FP8_E4M3FN_MAX
    except Exception:
        return _FP8_E4M3_MAX


class _DequantedSlice:
    """Presents an already-materialized weight tensor with the tiny slice of the
    ``PySafeSlice`` protocol the bf16 loader factories touch (``[...]`` indexing
    and ``get_shape``).

    Used by :func:`fp8_block_dequant_bf16_loader` so a module whose weights we
    keep in bf16 -- but which the checkpoint stores as block-FP8 -- can be loaded
    through its EXISTING bf16 loader factory (the delicate GQA/GDN head-aware
    sharding is reused verbatim, never re-derived). The wrapped tensor is the
    full block-dequanted weight in the checkpoint's ``[out, in]`` layout, so the
    inner loader shards/transposes it identically to the bf16 path.
    """

    def __init__(self, tensor: torch.Tensor):
        self._t = tensor

    def __getitem__(self, idx):
        return self._t[idx]

    def get_shape(self):
        return list(self._t.shape)

    def get_dtype(self):
        return str(self._t.dtype)


def _dequant_block_slice(
    w_slice, s_slice, n_start: int, n_size: int, k_start: int, k_size: int
) -> torch.Tensor:
    """Dequant ``[n_start:n_start+n_size, k_start:k_start+k_size]`` of a block-FP8
    HF weight + its ``weight_scale_inv`` to fp32.

    Only the needed weight rows/cols and the ``[128, 128]`` scale blocks covering
    them are pulled from disk (same I/O footprint as the bf16 sharded loader).
    The scale is broadcast to per-element via ``repeat_interleave`` on both dims,
    then trimmed to the exact slice by the within-block offsets.
    """
    nb0 = n_start // _BLK
    nb1 = (n_start + n_size + _BLK - 1) // _BLK
    n_off = n_start % _BLK
    kb0 = k_start // _BLK
    kb1 = (k_start + k_size + _BLK - 1) // _BLK
    k_off = k_start % _BLK

    w = w_slice[n_start : n_start + n_size, k_start : k_start + k_size].to(torch.float32)
    s = s_slice[nb0:nb1, kb0:kb1].to(torch.float32)
    scale = s.repeat_interleave(_BLK, dim=0).repeat_interleave(_BLK, dim=1)
    scale = scale[n_off : n_off + n_size, k_off : k_off + k_size]
    return w * scale


def _row_quantize(w_oi_fp32: torch.Tensor, fp8_max: float):
    """Per-output-channel (ROW) requant of a math-shaped ``[out, in]`` fp32 weight.

    Returns ``(w_fp8 [out, in], row_scale [out] fp32)`` such that
    ``w_fp8.float() * row_scale[:, None] ~= w_oi_fp32``.
    """
    amax = w_oi_fp32.abs().amax(dim=1).clamp_min(1e-12)  # [out]
    scale = amax / fp8_max  # [out] fp32
    w_q = (w_oi_fp32 / scale[:, None]).clamp(-fp8_max, fp8_max).to(_FP8_DTYPE)
    return w_q, scale


def _broadcast_row_scale(scale_out: torch.Tensor) -> torch.Tensor:
    """``[out]`` fp32 row scale -> ``[128, out]`` fp32 (128 broadcast partitions)."""
    out = scale_out.numel()
    return scale_out.to(torch.float32).view(1, out).expand(_PMAX, out).contiguous()


# ---------------------------------------------------------------------------
# Fused QKV (gated-Q: q output is 2x head_dim per head). Shards the OUTPUT dim
# with GQA KV replication, exactly like the bf16 ``fused_qkv_weight_loader``.
# Maps to source ``[qW, qS, kW, kS, vW, vS]`` (6 slices).
# ---------------------------------------------------------------------------


def fp8_row_fused_qkv_loader(
    num_shards: int, num_kv_replicas: int, *, want_scale: bool
) -> SafetensorsWeightLoader:
    """ROW-requant fused-QKV loader.

    ``want_scale=False`` -> fused FP8 weight, param layout ``[H, qkv_width_pr]``.
    ``want_scale=True``  -> fused ROW scale ``[128, qkv_width_pr]`` fp32.
    """

    def transform(slices, rank):
        assert len(slices) == 6, (
            f"fp8_row_fused_qkv_loader expects [qW,qS,kW,kS,vW,vS] (6 slices); "
            f"got {len(slices)}."
        )
        fp8_max = _target_fp8_max()
        q_w, q_s, k_w, k_s, v_w, v_s = slices
        q_sl, kv_sl = qkv_shard_offsets(
            q_w.get_shape()[0], k_w.get_shape()[0], num_shards, num_kv_replicas, rank
        )
        weights, scales = [], []
        for w_slice, s_slice, sl in (
            (q_w, q_s, q_sl),
            (k_w, k_s, kv_sl),
            (v_w, v_s, kv_sl),
        ):
            hidden = w_slice.get_shape()[1]
            n0, n1 = sl.start, sl.stop
            w_deq = _dequant_block_slice(w_slice, s_slice, n0, n1 - n0, 0, hidden)
            w_q, scale = _row_quantize(w_deq, fp8_max)
            weights.append(w_q)
            scales.append(scale)
        if want_scale:
            return _broadcast_row_scale(torch.cat(scales, dim=0))
        # Fuse along the output dim (q|k|v order) then transpose to [H, qkv_width_pr].
        return torch.cat(weights, dim=0).T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


# ---------------------------------------------------------------------------
# MLP gate / up: HF weight ``[I, H]`` (out=I), shard the OUTPUT/I dim.
# ---------------------------------------------------------------------------


def fp8_row_gate_up_loader(
    intermediate_size_per_rank: int, num_shards: int, *, want_scale: bool
) -> SafetensorsWeightLoader:
    """ROW-requant gate/up loader.

    ``want_scale=False`` -> FP8 weight, param layout ``[H, I_pr]``.
    ``want_scale=True``  -> ROW scale ``[128, I_pr]`` fp32.
    """
    i_pr = intermediate_size_per_rank

    def transform(slices, rank):
        assert len(slices) == 2, (
            f"fp8_row_gate_up_loader expects [weight, weight_scale_inv] (2 "
            f"slices); got {len(slices)}."
        )
        fp8_max = _target_fp8_max()
        w_slice, s_slice = slices  # HF [I, H] fp8, [I/128, H/128]
        i_total, hidden = w_slice.get_shape()
        assert i_total == i_pr * num_shards, (
            f"gate/up weight I axis ({i_total}) must equal i_per_rank*num_shards "
            f"({i_pr}*{num_shards})."
        )
        start = (rank % num_shards) * i_pr
        # [I_pr, H] fp32 (out=I_pr rows, in=H cols).
        w_deq = _dequant_block_slice(w_slice, s_slice, start, i_pr, 0, hidden)
        w_q, scale = _row_quantize(w_deq, fp8_max)
        if want_scale:
            return _broadcast_row_scale(scale)
        return w_q.T.contiguous()  # param [H, I_pr]

    return SafetensorsWeightLoader(transform=transform)


# ---------------------------------------------------------------------------
# MLP down: HF weight ``[H, I]`` (out=H), shard the INPUT/I (contraction) dim.
# ROW scale is per-output (H) channel over this rank's LOCAL contraction slice
# -- valid because each rank dequants its own shard independently and the
# partial matmuls are all-reduced (a local ROW scale need not be global).
# ---------------------------------------------------------------------------


def fp8_row_down_loader(
    intermediate_size_per_rank: int, num_shards: int, *, want_scale: bool
) -> SafetensorsWeightLoader:
    """ROW-requant down loader.

    ``want_scale=False`` -> FP8 weight, param layout ``[I_pr, H]``.
    ``want_scale=True``  -> ROW scale ``[128, H]`` fp32.
    """
    i_pr = intermediate_size_per_rank

    def transform(slices, rank):
        assert len(slices) == 2, (
            f"fp8_row_down_loader expects [weight, weight_scale_inv] (2 slices); "
            f"got {len(slices)}."
        )
        fp8_max = _target_fp8_max()
        w_slice, s_slice = slices  # HF [H, I] fp8, [H/128, I/128]
        hidden, i_total = w_slice.get_shape()
        assert i_total == i_pr * num_shards, (
            f"down weight I axis ({i_total}) must equal i_per_rank*num_shards "
            f"({i_pr}*{num_shards})."
        )
        start = (rank % num_shards) * i_pr
        # [H, I_pr] fp32 (out=H rows, in=I_pr contraction cols).
        w_deq = _dequant_block_slice(w_slice, s_slice, 0, hidden, start, i_pr)
        w_q, scale = _row_quantize(w_deq, fp8_max)
        if want_scale:
            return _broadcast_row_scale(scale)
        return w_q.T.contiguous()  # param [I_pr, H]

    return SafetensorsWeightLoader(transform=transform)


# ---------------------------------------------------------------------------
# Dequant-to-bf16 for modules the checkpoint stores as block-FP8 but that v1
# keeps in bf16 (full-attn o_proj; GDN in_proj_qkv / in_proj_z / out_proj). The
# resident weight is bf16 -- no HBM saving here -- but the bytes on disk are FP8,
# so the plain bf16 loader would misread them. We block-dequant the WHOLE tensor
# to bf16, then hand it to the module's own bf16 loader (via _DequantedSlice) so
# the exact same sharding/transposition applies.
# ---------------------------------------------------------------------------


def fp8_block_dequant_bf16_loader(inner_loader: SafetensorsWeightLoader):
    """Wrap an existing bf16 loader so it consumes one or more block-FP8
    ``[weight, weight_scale_inv]`` source pairs instead of plain bf16
    ``.weight`` slices.

    ``inner_loader`` is the loader the bf16 path would attach to the same param
    (e.g. ``sharding_weight_loader(...)`` for o_proj, ``gated_deltanet_*_loader``
    for GDN, or ``fused_qkv_weight_loader(...)`` for the full-attn fused QKV).

    Slices arrive as consecutive ``(weight, weight_scale_inv)`` pairs -- ONE pair
    for a single-tensor loader (o_proj, GDN in_proj_z / out_proj / in_proj_qkv)
    and THREE pairs ``[qW, qS, kW, kS, vW, vS]`` for the fused-QKV loader. Each
    pair is block-dequanted to bf16 and wrapped in :class:`_DequantedSlice`; the
    resulting one-shim-per-tensor list is handed to ``inner_loader.transform`` so
    its (GQA/GDN head-aware) sharding logic runs unchanged on bf16 tensors.
    """

    def transform(slices, rank):
        assert len(slices) >= 2 and len(slices) % 2 == 0, (
            f"fp8_block_dequant_bf16_loader expects consecutive [weight, "
            f"weight_scale_inv] pairs (even count >= 2); got {len(slices)}."
        )
        shims = []
        for i in range(0, len(slices), 2):
            w_slice, s_slice = slices[i], slices[i + 1]
            out, inp = w_slice.get_shape()
            # Full block-dequant in the checkpoint's [out, in] layout, then bf16.
            w_deq = _dequant_block_slice(w_slice, s_slice, 0, out, 0, inp)
            shims.append(_DequantedSlice(w_deq.to(torch.bfloat16)))
        return inner_loader.transform(shims, rank)

    return SafetensorsWeightLoader(transform=transform)
