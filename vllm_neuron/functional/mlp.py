# SPDX-License-Identifier: Apache-2.0
import math
import os

import torch
from torch import Tensor

from nkilib.core.mlp.mlp import mlp as nkilib_mlp
from nkilib.core.utils.common_types import (
    ActFnType,
    ComputationMode,
    DtypeMode,
    NormType,
    QuantizationType,
)
from vllm_neuron import envs
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

if envs.VLLM_NEURON_MLP_FORCE_TKG:
    import nki.isa as nisa
    from nkilib.core.mlp import mlp as _nkilib_mlp_module
    from nkilib.core.mlp import mlp_parameters

    _orig_is_mlp_tkg = mlp_parameters.is_mlp_tkg

    def _force_tkg_for_fp8(params):
        if (
            params.quant_params.is_quant()
            and not params.quant_params.is_dtype_mx()
            and nisa.get_nc_version() >= nisa.nc_version.gen3
        ):
            return True
        return _orig_is_mlp_tkg(params)

    mlp_parameters.is_mlp_tkg = _force_tkg_for_fp8
    _nkilib_mlp_module.is_mlp_tkg = _force_tkg_for_fp8

wrapped_mlp = wrap_nki(nkilib_mlp)

# nkilib kernel tiling constraints for grid=2 (lnc=2).
# from nkilib source code
# TODO: feature request - constants below needs to be exported by nki API
_NUM_HW_PSUM_BANKS = 8
_SRC_PROJ_INT_DIM_TILE_SIZE = 512
_TKG_BS_SEQLEN_THRESHOLD = 128
_MIN_H_FOR_I_SHARDING = 7168
_MIN_I_FOR_I_SHARDING = 1024
_MAX_T_FOR_I_SHARDING = 256

# A ROW-packed MX activation carries a per-row fp32 dequant scale in a trailing
# 4 bytes (one fp8 byte each), so the input is [T, H+4]. Not every MX caller packs
# (e.g. a bf16-in TKG path passes [T, H]); _can_use_kernel strips this tail only
# when present (H % 128 == 4).
_MX_PACKED_SCALE_BYTES = 4

# MX weight-packing tile geometry (shared MX kernel layout, NOT model-specific).
# The H_X4_* / MX_CONTIGUOUS swizzles tile the contraction dim into 512-element
# tiles of 128 partitions x 4 x4-packed fp8 bytes. The per-model MX weight loaders
# (e.g. qwen3_vl) import these so the packed nn.Parameter shapes and the loader
# swizzle share one definition.
# ── EXPERIMENT ONLY (2026-08-08): force the MLP kernel choice ────────────────────────────────
# nkilib picks TKG vs CTE on `batch_size * sequence_len <= TKG_BS_SEQLEN_THRESHOLD` (96,
# mlp_parameters.py::is_mlp_tkg). For a 1-token decode step that means num_seqs vs 96, so a
# num_seqs sweep across 96 changes the KERNEL as well as the amount of padding, and a throughput
# delta cannot be attributed to either alone. This knob forces one side so the 2x2 can be measured:
#
#   FORCE_MLP_KERNEL=cte   -> force_cte_mode=True        (CTE even at small batch)
#   FORCE_MLP_KERNEL=tkg   -> mode=ComputationMode.DECODE (TKG even at large batch; it is
#                             is_mlp_tkg's first clause and bypasses the threshold entirely)
#   unset / auto           -> stock behaviour, byte-identical to every recorded measurement
#
# 🔴 MUST be kept consistent with ministral3/model.py's `_runs_on_cte`, which recomputes the SAME
# threshold to decide whether to fp8-pre-quantize the hidden. Forcing only this side desynchronises
# them and reproduces the 2.32 port bug: a bf16 hidden reaching the kernel's fp8 source transpose
# ("nc_matmul (transpose mode) dst dtype must match input dtype ... got dst=float8_e4m3 but
# input=bfloat16"). model.py reads the same env var for exactly that reason.
_FORCE_MLP_KERNEL = (os.environ.get("FORCE_MLP_KERNEL") or "auto").strip().lower()
if _FORCE_MLP_KERNEL not in ("auto", "tkg", "cte"):
    raise ValueError(
        f"FORCE_MLP_KERNEL must be auto|tkg|cte, got {_FORCE_MLP_KERNEL!r}"
    )

_PMAX = 128  # partition dim (nl.tile_size.pmax)
_Q_WIDTH = 4  # 4 float8_e4m3fn bytes per nl.float8_e4m3fn_x4 / uint32 element
_TILE_SIZE = _PMAX * _Q_WIDTH  # 512 -- the 512-element H/I tile of the MX layout


def mlp(
    hidden: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    eps: float = 1e-6,
    ln_w: Tensor | None = None,
    skip_gate: bool = False,
    act_fn: ActFnType = ActFnType.SiLU,
    gate_b: Tensor | None = None,
    up_b: Tensor | None = None,
    down_b: Tensor | None = None,
    norm_b: Tensor | None = None,
    norm_type: NormType = NormType.NO_NORM,
    fused_add_tensor: Tensor | None = None,
    store_fused_add_result: bool = False,
    quantization_type: QuantizationType = QuantizationType.NONE,
    gate_w_scale: Tensor | None = None,
    up_w_scale: Tensor | None = None,
    down_w_scale: Tensor | None = None,
    gate_up_in_scale: Tensor | None = None,
    down_in_scale: Tensor | None = None,
    quant_clipping_bound: float = 0.0,
    output_dtype: torch.dtype | None = None,
    gate_clamp_upper_limit: float | None = None,
    gate_clamp_lower_limit: float | None = None,
    up_clamp_upper_limit: float | None = None,
    up_clamp_lower_limit: float | None = None,
    gate_up_w_layout=None,
) -> Tensor | tuple[Tensor, Tensor]:
    """MLP API that automatically selects between nkilib NKI kernel and PyTorch fallback.

    This function checks kernel constraints and dispatches to:
    - nkilib MLP kernel: Default option when running on Neuron
    - PyTorch implementation: When in CPU mode (VLLM_NEURON_CPU_MODE=1) or constraints are violated

    Args:
        hidden: Input hidden state tensor with shape [T, H] where T is total tokens.
        gate_w: Gate projection weights with shape [H, I].
        up_w: Up projection weights with shape [H, I].
        down_w: Down projection weights with shape [I, H].
        eps: Epsilon for numerical stability in normalization (default: 1e-6).
        ln_w: Normalization weights (gamma) with shape [1, H].
        skip_gate: If True, skip gate projection and apply activation to up projection.
        act_fn: Activation function type (SiLU, GELU, GELU_Tanh_Approx, Swish).
        gate_b: Optional gate projection bias with shape [1, I].
        up_b: Optional up projection bias with shape [1, I].
        down_b: Optional down projection bias with shape [1, H].
        norm_b: Optional normalization bias with shape [1, H]. Only for LayerNorm.
        norm_type: Type of normalization (NO_NORM, RMS_NORM, RMS_NORM_SKIP_GAMMA, LAYER_NORM).
        fused_add_tensor: Optional tensor for fused residual add before normalization.
        store_fused_add_result: If True and fused_add_tensor is provided, return fused add
            result as second output.
        quantization_type: Quantization type (NONE, STATIC, ROW, MX).
        gate_w_scale: FP8 dequantization scales for gate weights.
        up_w_scale: FP8 dequantization scales for up weights.
        down_w_scale: FP8 dequantization scales for down weights.
        gate_up_in_scale: FP8 dequantization scales for gate/up input (static quant).
        down_in_scale: FP8 dequantization scales for down input (static quant).
        quant_clipping_bound: Clipping bound for FP8 row quantization (default: 0.0).
        output_dtype: Output tensor data type. If None, uses hidden tensor's dtype.
        gate_clamp_upper_limit: Upper clamp bound for gate projection results.
        gate_clamp_lower_limit: Lower clamp bound for gate projection results.
        up_clamp_upper_limit: Upper clamp bound for up projection results.
        up_clamp_lower_limit: Lower clamp bound for up projection results.

    Returns:
        Tensor with shape [T, H] normally, or tuple[Tensor, Tensor] when
        fused_add_tensor is provided and store_fused_add_result is True.

    Example:
        >>> import torch
        >>> from vllm_neuron.functional.mlp import mlp
        >>>
        >>> T, H, I = 32, 256, 512
        >>> hidden = torch.randn(T, H)
        >>> gate_w = torch.randn(H, I)
        >>> up_w = torch.randn(H, I)
        >>> down_w = torch.randn(I, H)
        >>> output = mlp(hidden, gate_w, up_w, down_w)
        >>> output.shape
        torch.Size([32, 256])
    """
    T, H = hidden.shape

    if _can_use_kernel(hidden, gate_w, gate_b, up_b, down_b, quantization_type):
        # Reshape [T, H] -> [1, T, H] for kernel (kernel expects [B, S, H])
        hidden_3d = hidden.unsqueeze(0)

        # Provide default normalization weights if norm is requested but ln_w is None
        if norm_type != NormType.NO_NORM and ln_w is None:
            ln_w = torch.ones((1, H), dtype=hidden.dtype, device=hidden.device)

        # Reshape fused_add_tensor to 3D if provided
        fused_add_3d = None
        if fused_add_tensor is not None:
            fused_add_3d = fused_add_tensor.unsqueeze(0)

        # The MXFP hardware path (MX / STATIC_MX / ROW_MX) does not use column
        # tiling -- the kernel asserts ``use_tkg_*_column_tiling=False`` whenever
        # ``is_dtype_mx()`` (mlp_parameters: "MX quantization (mxfp) does not use
        # column tiling"). Compare against the enum members directly rather than
        # calling ``quantization_type.is_mx()``: a bound-method call is not
        # traceable by Dynamo ("Unexpected type in sourceless builder: method"),
        # whereas the membership check is. Covers MX and ROW_MX, not only STATIC_MX.
        _column_tiling = quantization_type not in (
            QuantizationType.MX,
            QuantizationType.STATIC_MX,
            QuantizationType.ROW_MX,
        )

        result = wrapped_mlp[2](
            hidden_tensor=hidden_3d,
            gate_proj_weights_tensor=gate_w,
            up_proj_weights_tensor=up_w,
            down_proj_weights_tensor=down_w,
            normalization_weights_tensor=ln_w,
            gate_proj_bias_tensor=gate_b,
            up_proj_bias_tensor=up_b,
            down_proj_bias_tensor=down_b,
            normalization_bias_tensor=norm_b,
            fused_add_tensor=fused_add_3d,
            store_fused_add_result=store_fused_add_result,
            activation_fn=act_fn,
            normalization_type=norm_type,
            quantization_type=quantization_type,
            gate_w_scale=gate_w_scale,
            up_w_scale=up_w_scale,
            down_w_scale=down_w_scale,
            gate_up_in_scale=gate_up_in_scale,
            down_in_scale=down_in_scale,
            quant_clipping_bound=quant_clipping_bound,
            output_dtype=output_dtype,
            store_output_in_sbuf=False,
            eps=eps,
            skip_gate_proj=skip_gate,
            use_tkg_gate_up_proj_column_tiling=_column_tiling,
            use_tkg_down_proj_column_tiling=_column_tiling,
            use_tkg_down_proj_optimized_layout=False,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            # EXPERIMENT: see _FORCE_MLP_KERNEL above. Default path is unchanged (False/AUTO).
            force_cte_mode=(_FORCE_MLP_KERNEL == "cte"),
            mode=(
                ComputationMode.DECODE
                if _FORCE_MLP_KERNEL == "tkg"
                else ComputationMode.AUTO
            ),
            dtype_mode=DtypeMode.AUTO,
            **(
                {"gate_up_w_layout": gate_up_w_layout}
                if gate_up_w_layout is not None
                else {}
            ),
        )

        # Handle return value: nkilib returns list/tuple when store_fused_add_result
        if isinstance(result, (tuple, list)):
            if store_fused_add_result and len(result) == 2:
                return result[0].squeeze(0), result[1].squeeze(0)
            return result[0].squeeze(0)
        return result.squeeze(0)
    else:
        return _torch_mlp_impl(
            hidden=hidden,
            ln_w=ln_w,
            gate_w=gate_w,
            up_w=up_w,
            down_w=down_w,
            eps=eps,
            skip_gate=skip_gate,
            act_fn=act_fn,
            gate_b=gate_b,
            up_b=up_b,
            down_b=down_b,
            norm_b=norm_b,
            norm_type=norm_type,
            fused_add_tensor=fused_add_tensor,
            store_fused_add_result=store_fused_add_result,
        )


def _can_use_kernel(
    hidden: Tensor,
    gate_w: Tensor,
    gate_b: Tensor | None = None,
    up_b: Tensor | None = None,
    down_b: Tensor | None = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
) -> bool:
    """Check if the nkilib NKI kernel can be used for the given dimensions.

    Returns False when ``VLLM_NEURON_CPU_MODE`` is enabled, the hidden tensor is on
    CPU, hidden dimension is not aligned to 128, or the model dimensions
    violate nkilib tiling constraints for grid=2 (lnc=2).

    The grid=2 constraints checked are:
    - **TKG mode** (T <= 96): Hidden dim is sharded across 2 cores, so
      ``H // 128`` must be even, i.e. ``H % 256 == 0``.
    - **CTE mode** (T > 96): Intermediate dim tiles must fit in PSUM banks.
      ``ceil(effective_I / 512) <= 8`` where ``effective_I`` is ``I // 2``
      when I-sharding is active, otherwise ``I``.

    Native-MX (``QuantizationType.MX``) is a special case, and it is path-neutral
    here (CTE *and* TKG share this gate; nkilib's ``is_mlp_tkg`` does the CTE/TKG
    split internally by ``B*S`` — this function never picks the path). ``gate_w`` is
    a pre-tiled swizzled weight whose ``shape[1]`` is NOT ``I``, and the activation
    may be EITHER ROW-packed FP8 ``[T, H+4]`` (CTE prefill: per-row dequant scale in
    a trailing 4 bytes) OR plain ``[T, H]`` (e.g. a TKG path that quantizes a bf16
    input on-device). Both break the plain 2D checks above, so for MX we:
      * strip a trailing +4 from H *only when present* (``H % 128 == 4``) before the
        128-alignment check, so a packed ``H+4`` and an unpacked ``H`` both validate;
      * bypass the ``gate_w.shape[1]``-derived PSUM-bank check entirely — the MX
        kernel pre-tiles the intermediate dim and validates its own layout/PSUM
        budget at trace time, failing loud if a config doesn't fit.

    Note: The nkilib kernel supports bfloat16 and float8 (float8_e4m3) dtypes.
    Other dtypes are not supported and will cause a validation error from nkilib.
    Callers are responsible for providing tensors with a supported dtype.

    Args:
        hidden: Input hidden state tensor with shape [T, H] (or [T, H+4] for MX).
        gate_w: Gate projection weights with shape [H, I] (or 6D for MX).
        gate_b: Optional gate projection bias with shape [1, I].
        up_b: Optional up projection bias with shape [1, I].
        down_b: Optional down projection bias with shape [1, H].
        quantization_type: Quantization type; ``MX`` takes the packed-input path.

    Returns:
        True if the NKI kernel can be used, False otherwise.

    Example:
        >>> hidden = torch.randn(32, 256, dtype=torch.bfloat16)
        >>> gate_w = torch.randn(256, 512, dtype=torch.bfloat16)
        >>> _can_use_kernel(hidden, gate_w)
        True
    """
    if not can_run_kernel(hidden):
        return False
    T, H = hidden.shape

    # Native-MX: strip a row-packed +4 tail when present (see _MX_PACKED_SCALE_BYTES),
    # then require 128-alignment. A real MX weight always has H % 128 == 0 (the
    # H_X4_INNERMOST swizzle requires it), so the H % 128 == 4 arm only ever fires on
    # a genuine ROW-packed [T, H+4] activation -- it can't silently eat 4 real
    # elements of an unpacked input. The kernel does its own I-tiling/PSUM-bank
    # validation, so skip the 2D-weight-derived bank check below.
    if quantization_type == QuantizationType.MX:
        H_eff = H - _MX_PACKED_SCALE_BYTES if H % 128 == _MX_PACKED_SCALE_BYTES else H
        return H_eff % 128 == 0

    # 🔴 2026-08-18: ROW shares MX's packed-activation layout, and this check used to reject it.
    # For CTE + QuantizationType.ROW the kernel wants a PRE-QUANTIZED fp8 hidden carrying the
    # per-row fp32 scale bit-cast into a trailing 4 bytes -- `[T, H+4]`
    # (nkilib mlpp_input_has_packed_scale; the CTE loader splits it in
    # mlp_cte/basic/mlp_cte_basic_tensor_io.py::load_packed_hidden_scales). Only the MX arm above
    # stripped that tail, so a ROW-packed hidden failed `H % 128 != 0`, this function returned
    # False, and the wrapper fell back to the PyTorch reference path -- which then died on
    # `aten.mm` with `[128, 12292] X [12288, 896]` during FX tracing. The kernel was never the
    # problem; the eligibility gate was.
    # The `H % 128 == 4` guard is the same safety argument as MX's: a real unpacked hidden always
    # has H % 128 == 0, so this arm cannot silently eat 4 genuine elements.
    # ⚠️ Compare enum members directly — do NOT call ``quantization_type.is_logical_row()``.
    # Dynamo cannot trace a bound method on a sourceless enum constant and dies with
    # "SourcelessBuilder.create does not know how to wrap <class 'method'>" (gb0116), which
    # surfaces only as a parallel-trace lane failure during compile (measured 2026-08-18).
    # Every other branch in this function uses ``==`` for the same reason.
    if (
        quantization_type == QuantizationType.ROW
        or quantization_type == QuantizationType.ROW_MX
    ) and H % 128 == _MX_PACKED_SCALE_BYTES:
        H = H - _MX_PACKED_SCALE_BYTES

    if H % 128 != 0:
        return False

    if T <= _TKG_BS_SEQLEN_THRESHOLD:
        # TKG: hidden dim sharded across 2 cores, H1 = H//128 must be even
        if H % 256 != 0:
            return False
        return True

    # CTE: PSUM bank constraint on intermediate dimension
    I = gate_w.shape[1]
    has_bias = gate_b is not None or up_b is not None or down_b is not None
    can_shard_on_i = (
        H >= _MIN_H_FOR_I_SHARDING
        and I >= _MIN_I_FOR_I_SHARDING
        and T <= _MAX_T_FOR_I_SHARDING
        and not has_bias
    )
    effective_i = I // 2 if can_shard_on_i else I
    return math.ceil(effective_i / _SRC_PROJ_INT_DIM_TILE_SIZE) <= _NUM_HW_PSUM_BANKS


def _get_activation_fn(act_fn: ActFnType):
    """Get the corresponding torch activation function.

    Args:
        act_fn: Activation function type from nkilib's ActFnType enum.

    Returns:
        A callable torch activation function.

    Raises:
        ValueError: If the activation function type is unsupported.

    Example:
        >>> fn = _get_activation_fn(ActFnType.SiLU)
        >>> fn(torch.tensor([1.0]))
        tensor([0.7311])
    """
    if act_fn == ActFnType.SiLU or act_fn == ActFnType.Swish:
        return torch.nn.functional.silu
    elif act_fn == ActFnType.GELU:
        return torch.nn.functional.gelu
    elif act_fn == ActFnType.GELU_Tanh_Approx:
        return lambda x: torch.nn.functional.gelu(x, approximate="tanh")
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


def _torch_rms_norm(x: Tensor, weight: Tensor, eps: float, skip_gamma: bool) -> Tensor:
    """RMSNorm implementation in PyTorch."""
    # x: [T, H], weight: [1, H]
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    if not skip_gamma:
        x_normed = x_normed * weight
    return x_normed


def _torch_layer_norm(
    x: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float
) -> Tensor:
    """LayerNorm implementation in PyTorch."""
    # x: [T, H], weight: [1, H], bias: [1, H] or None
    mean = x.mean(dim=-1, keepdim=True)
    variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    x_normed = (x - mean) * torch.rsqrt(variance + eps)
    if weight is not None:
        x_normed = x_normed * weight
    if bias is not None:
        x_normed = x_normed + bias
    return x_normed


def _torch_mlp_impl(
    hidden: Tensor,
    ln_w: Tensor | None,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    eps: float,
    skip_gate: bool,
    act_fn: ActFnType,
    gate_b: Tensor | None,
    up_b: Tensor | None,
    down_b: Tensor | None,
    norm_b: Tensor | None,
    norm_type: NormType | None,
    fused_add_tensor: Tensor | None = None,
    store_fused_add_result: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """PyTorch implementation of MLP with optional normalization and fused residual add.

    Computation flow:
    0. Optional fused residual add
    1. Optional normalization (RMSNorm or LayerNorm)
    2. Gate projection: x @ gate_w + gate_b (if not skip_gate)
    3. Up projection: x @ up_w + up_b
    4. Activation: act_fn(gate) * up  OR  act_fn(up) if skip_gate
    5. Down projection: result @ down_w + down_b

    Args:
        hidden: Input hidden state tensor with shape [T, H].
        ln_w: Normalization weights (gamma) with shape [1, H].
        gate_w: Gate projection weights with shape [H, I].
        up_w: Up projection weights with shape [H, I].
        down_w: Down projection weights with shape [I, H].
        eps: Epsilon for numerical stability.
        skip_gate: If True, skip gate projection.
        act_fn: Activation function type.
        gate_b: Optional gate projection bias.
        up_b: Optional up projection bias.
        down_b: Optional down projection bias.
        norm_b: Optional normalization bias.
        norm_type: Type of normalization.
        fused_add_tensor: Optional tensor for fused residual add.
        store_fused_add_result: If True, return fused add result as second output.

    Returns:
        Tensor or tuple[Tensor, Tensor]: MLP output, optionally with fused add result.

    Example:
        >>> hidden = torch.randn(32, 256)
        >>> gate_w = torch.randn(256, 512)
        >>> up_w = torch.randn(256, 512)
        >>> down_w = torch.randn(512, 256)
        >>> output = _torch_mlp_impl(hidden, None, gate_w, up_w, down_w, 1e-6,
        ...     False, ActFnType.SiLU, None, None, None, None, NormType.NO_NORM)
        >>> output.shape
        torch.Size([32, 256])
    """

    x = hidden

    # Step 0: Fused residual add
    fused_add_output = None
    if fused_add_tensor is not None:
        x = x + fused_add_tensor
        if store_fused_add_result:
            fused_add_output = x.clone()

    # Step 1: Apply normalization
    if norm_type == NormType.RMS_NORM:
        x = _torch_rms_norm(x, ln_w, eps, skip_gamma=False)
    elif norm_type == NormType.RMS_NORM_SKIP_GAMMA:
        x = _torch_rms_norm(x, ln_w, eps, skip_gamma=True)
    elif norm_type == NormType.LAYER_NORM:
        x = _torch_layer_norm(x, ln_w, norm_b, eps)
    # NormType.NO_NORM: skip normalization

    # Step 2 & 3: Gate and Up projections
    if not skip_gate:
        # Gate projection
        gate = x @ gate_w
        if gate_b is not None:
            gate = gate + gate_b

    # Up projection
    up = x @ up_w
    if up_b is not None:
        up = up + up_b

    # Step 4: Apply activation
    activation_fn = _get_activation_fn(act_fn)

    if skip_gate:
        # When skip_gate is True, apply activation directly to up projection
        activated = activation_fn(up)
    else:
        # Standard GLU-style: act(gate) * up
        activated = activation_fn(gate) * up

    # Step 5: Down projection
    output = activated @ down_w
    if down_b is not None:
        output = output + down_b

    if fused_add_output is not None:
        return output, fused_add_output
    return output
