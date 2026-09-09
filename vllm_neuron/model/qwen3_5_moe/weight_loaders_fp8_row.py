# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 -> per-channel ROW FP8 load-time re-quantization loaders.

The ``Qwen3.6-35B-A3B-FP8`` checkpoint stores each quantized ``nn.Linear`` as a
DeepSeek block-FP8 pair:

    ``.weight``            float8_e4m3fn, HF shape ``[out, in]``
    ``.weight_scale_inv``  bf16, shape ``[out/128, in/128]`` (one scalar per
                           ``[128, 128]`` block); dequant convention (vLLM
                           w8a8_block_fp8) is ``bf16 = w_fp8.float() * scale``.

Neuron's NKI GEMMs cannot consume a resident block-``[128, 128]`` scale, so at
load we DEQUANT each block back to fp32 and RE-QUANT to **per-output-channel
ROW** FP8: one scale per output channel = ``absmax(row) / fp8_max``. The kernel
then dequants ``w_fp8 * row_scale`` and the activation is quantized dynamically
per-token inside the decode MoE kernel, matching the checkpoint's
``activation_scheme: "dynamic"``.

Two different ROW scale contracts appear here:

* the plain-MLP one (nkilib mlp.py / output_projection): ``[128, out_dim]`` fp32,
  128 broadcast partition-rows and one column per output channel. Used by the
  shared expert if it is ever moved onto the fp8 kernel; not used today.
* the MoE one (nkilib moe_tkg): ``[E_L, 2, I]`` fp32 for the fused gate/up
  weights and ``[E_L, H]`` fp32 for down. NOT broadcast over partitions.

The expert loaders below emit the MoE form. The prefill kernel
(``moe_cte``/``shard_on_i``) wants the SAME numbers flattened to ``[E, 1, 2*I]``
and ``[E, 1, H]``, so the model reshapes these tensors at the call site rather
than keeping a second copy. On **trn2** the ROW NKI kernels read FP8 tiles as legacy
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
# MoE experts. The FP8 checkpoint stores every expert SEPARATELY
# (``mlp.experts.<e>.{gate,up,down}_proj.{weight,weight_scale_inv}``), unlike the
# BF16 checkpoint which ships the experts already fused into 3-D tensors. So the
# loaders below both fuse and re-quantize, and the caller lists only THIS rank's
# local experts as sources -- there is no separate expert-parallel wrapper to
# apply, and no rank reads another rank's expert bytes.
#
# Source order is fixed by the mapping builder in ``model.py``:
#   gate_up : [gate_w, gate_s, up_w, up_s] per local expert, experts in order
#   down    : [down_w, down_s]             per local expert, experts in order
#
# Emitted layouts (matching the bf16 params they replace, and the nkilib
# ``moe_tkg`` FP8-ROW contract):
#   gate_up weight -> [E_L, H, 2 * I_pr] fp8      (reshaped to [E_L, H, 2, I_pr])
#   gate_up scale  -> [E_L, 2, I_pr]     fp32
#   down    weight -> [E_L, I_pr, H]     fp8
#   down    scale  -> [E_L, H]           fp32
# ---------------------------------------------------------------------------


def fp8_row_expert_gate_up_loader(
    num_local_experts: int, intermediate_size_per_rank: int, num_shards: int,
    *, want_scale: bool
) -> SafetensorsWeightLoader:
    """ROW-requant fused gate/up expert loader.

    ``want_scale=False`` -> fp8 weight ``[E_L, H, 2 * I_pr]``.
    ``want_scale=True``  -> fp32 scale ``[E_L, 2, I_pr]``.

    ``num_shards`` is the MoE tensor-parallel degree *within* the expert-parallel
    group (``world // ep_degree``); with pure EP it is 1 and each expert's full
    intermediate dim stays on one rank.
    """
    e_l = num_local_experts
    i_pr = intermediate_size_per_rank

    def transform(slices, rank):
        assert len(slices) == 4 * e_l, (
            f"fp8_row_expert_gate_up_loader expects 4 slices per local expert "
            f"([gate_w, gate_s, up_w, up_s] x {e_l} = {4 * e_l}); got {len(slices)}."
        )
        fp8_max = _target_fp8_max()
        start = (rank % num_shards) * i_pr if num_shards > 1 else 0
        w_out, s_out = [], []
        for e in range(e_l):
            g_w, g_s, u_w, u_s = slices[4 * e : 4 * e + 4]
            per_proj_w, per_proj_s = [], []
            for w_slice, s_slice in ((g_w, g_s), (u_w, u_s)):
                i_total, hidden = w_slice.get_shape()      # HF [I, H]
                assert i_total == i_pr * num_shards, (
                    f"expert gate/up I axis ({i_total}) must equal "
                    f"i_per_rank*num_shards ({i_pr}*{num_shards})."
                )
                w_deq = _dequant_block_slice(w_slice, s_slice, start, i_pr, 0, hidden)
                w_q, scale = _row_quantize(w_deq, fp8_max)   # [I_pr, H], [I_pr]
                per_proj_w.append(w_q.T.contiguous())        # [H, I_pr]
                per_proj_s.append(scale)                     # [I_pr]
            # [H, 2 * I_pr] with gate first, then up -- the layout the model's
            # reshape(E_L, H, 2, I_pr) expects.
            w_out.append(torch.cat(per_proj_w, dim=1))
            s_out.append(torch.stack(per_proj_s, dim=0))     # [2, I_pr]
        if want_scale:
            return torch.stack(s_out, dim=0).to(torch.float32).contiguous()
        return torch.stack(w_out, dim=0).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def fp8_row_expert_down_loader(
    num_local_experts: int, intermediate_size_per_rank: int, num_shards: int,
    *, want_scale: bool
) -> SafetensorsWeightLoader:
    """ROW-requant expert down loader.

    ``want_scale=False`` -> fp8 weight ``[E_L, I_pr, H]``.
    ``want_scale=True``  -> fp32 scale ``[E_L, H]``.

    With MoE tensor parallelism inside the EP group the contraction (I) dim is
    sharded, so the ROW scale is per-output (H) channel over this rank's LOCAL
    contraction slice. That is valid because the partial products are reduced
    across those ranks afterwards -- a local ROW scale does not have to be global.
    """
    e_l = num_local_experts
    i_pr = intermediate_size_per_rank

    def transform(slices, rank):
        assert len(slices) == 2 * e_l, (
            f"fp8_row_expert_down_loader expects 2 slices per local expert "
            f"([down_w, down_s] x {e_l} = {2 * e_l}); got {len(slices)}."
        )
        fp8_max = _target_fp8_max()
        start = (rank % num_shards) * i_pr if num_shards > 1 else 0
        w_out, s_out = [], []
        for e in range(e_l):
            w_slice, s_slice = slices[2 * e : 2 * e + 2]     # HF [H, I]
            hidden, i_total = w_slice.get_shape()
            assert i_total == i_pr * num_shards, (
                f"expert down I axis ({i_total}) must equal i_per_rank*num_shards "
                f"({i_pr}*{num_shards})."
            )
            w_deq = _dequant_block_slice(w_slice, s_slice, 0, hidden, start, i_pr)
            w_q, scale = _row_quantize(w_deq, fp8_max)        # [H, I_pr], [H]
            w_out.append(w_q.T.contiguous())                  # [I_pr, H]
            s_out.append(scale)                               # [H]
        if want_scale:
            return torch.stack(s_out, dim=0).to(torch.float32).contiguous()
        return torch.stack(w_out, dim=0).contiguous()

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
