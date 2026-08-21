# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from torch import Tensor

from nkilib.core.output_projection.output_projection_cte import (
    output_projection_cte as output_projection,
)
from nkilib.core.utils.common_types import QuantizationType

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

P_MAX = 128
MAX_VALIDATED_H_SIZE = 16384 + 4321
MAX_VALIDATED_B_TIMES_S_SIZE = 128 * 1024
MAX_VALIDATED_N_SIZE = 17


def _can_use_kernel(
    active: Tensor,
    weight: Tensor,
    quantization_type: QuantizationType = QuantizationType.NONE,
) -> bool:
    """
    Check if the kernel can be used.

    Returns False when any NKI kernel constraint is violated or the tensors
    live on the CPU.
    """
    lnc_size = 2

    if not can_run_kernel(active):
        return False

    if active.dim() != 4:
        return False

    if quantization_type == QuantizationType.ROW:
        # 🎯 ROW (2026-08-10 01:00 JST): the kernel takes [B, S, N, D] for ROW and
        # [B, N, D, S] for every other path (output_projection_cte.py:116-124). Unpack in the
        # matching order so the constraint checks below see the right dimensions.
        B, S, N, D = active.shape
    else:
        B, N, D, S = active.shape
    ND_weight, H = weight.shape

    if quantization_type == QuantizationType.MX:
        # MX packed weight: [ND//4, H] uint32 → effective ND = ND_weight * 4
        if N * D != ND_weight * 4:
            return False
    else:
        if N * D != ND_weight:
            return False

    if H > MAX_VALIDATED_H_SIZE:
        return False

    if B * S > MAX_VALIDATED_B_TIMES_S_SIZE:
        return False

    if N > MAX_VALIDATED_N_SIZE:
        return False

    if D > P_MAX:
        return False

    if H % lnc_size != 0:
        return False

    return True


def _torch_o_proj_impl(
    active: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    input_scales: Optional[Tensor] = None,
    weight_scales: Optional[Tensor] = None,
) -> Tensor:
    """
    PyTorch implementation of output projection.

    Args:
        active: Input tensor with shape [B, N, D, S] or [B, S, N*D]
        weight: Weight tensor with shape [N*D, H]
        bias: Optional bias tensor with shape [1, H]
        quantization_type: Must be QuantizationType.NONE — quantization is
            not supported in the PyTorch fallback path.
        input_scales: Must be None — not supported in the PyTorch fallback.
        weight_scales: Must be None — not supported in the PyTorch fallback.

    Returns:
        Output tensor with shape [B, S, H]

    Raises:
        AssertionError: If quantization parameters are provided.
    """
    assert quantization_type == QuantizationType.NONE, (
        f"Quantization is not supported in the PyTorch fallback path. "
        f"Got quantization_type={quantization_type}"
    )
    assert input_scales is None, (
        "input_scales must be None for the PyTorch fallback path — "
        "quantization is not supported."
    )
    assert weight_scales is None, (
        "weight_scales must be None for the PyTorch fallback path — "
        "quantization is not supported."
    )

    if active.dim() == 3:
        # [B, S, N*D] @ [N*D, H] -> [B, S, H]
        output = active @ weight
    elif active.dim() == 4:
        B, N, D, S = active.shape
        # [B, N, D, S] -> [B, N*D, S] -> [B, S, N*D]
        x = active.reshape(B, N * D, S).transpose(1, 2)
        # [B, S, N*D] @ [N*D, H] -> [B, S, H]
        output = x @ weight
    else:
        raise ValueError(f"Expected 3D or 4D input, got {active.dim()}D")

    if bias is not None:
        output = output + bias

    return output


def o_proj(
    active: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    input_scales: Optional[Tensor] = None,
    weight_scales: Optional[Tensor] = None,
) -> Tensor:
    """
    Output Projection API that automatically selects between NKI kernels and PyTorch fallback.

    This function checks kernel constraints and dispatches to:
    - output_projection: When all constraints are satisfied and running on Neuron
    - PyTorch implementation: When constraints are violated or not on Neuron

    Computes: output = active @ weight + bias

    Dimensions:
        B: Batch size
        N: Number of heads
        S: Sequence length
        H: Hidden dimension size
        D: Head dimension size

    Args:
        active: Input tensor with shape [B, N, D, S] or [B, S, N*D]
        weight: Weight tensor with shape [N*D, H]. Dtype depends on quantization:
            - NONE: bf16 / fp16
            - STATIC: ``torch.float8_e4m3fn`` in row-major layout
            - STATIC_MX (TRN3-only): ``torch.float8_e4m3fn`` pre-shuffled
              host-side via the loader's reshape-transpose (see
              ``weight_loaders_mxfp8.py:_mx_shuffle_o_proj``). The kernel
              consumes the 2D ``[N*D, H]`` tensor and reshapes it
              internally to ``[N*D//4, H, 4]``.
        bias: Optional bias tensor with shape [1, H]
        quantization_type: Type of quantization (NONE / STATIC / STATIC_MX).
            Quantization is only supported on the NKI kernel path; the PyTorch
            fallback will raise an AssertionError if a value other than NONE is
            supplied.
        input_scales: Per-tensor input scale, fp32. Required for STATIC and
            STATIC_MX. The o-proj CTE kernel asserts shape ``(128, 1)`` for
            both quantization types (broadcast scalar; see
            ``output_projection_cte_parameters.py``).
        weight_scales: Per-tensor weight dequant scale, fp32. Same shape
            contract as ``input_scales``: ``(128, 1)`` for STATIC and STATIC_MX.

    Returns:
        Output tensor with shape [B, S, H]

    Constraints (falls back to PyTorch if violated):
        - weight must be 2D tensor with shape [N*D, H]
        - Head dimension D must be <= 128
        - H must be divisible by LNC size
        - bias (if provided) must have shape [1, H]
        - FP8 static quantization requires N or D to be even

    Notes:
        LNC sharding is performed on the weight tensor's hidden dimension.
        If active is 3D [B, S, N*D], it is reshaped to [B, N, D, S] for the
        NKI kernel using D inferred from weight shape.
    """
    # Reshape 3D [B, S, N*D] to the 4D layout the kernel expects.
    # 🎯 ROW wants [B, S, N, D]; every other path wants [B, N, D, S]
    # (output_projection_cte.py:116-124). Added 2026-08-10 01:00 JST for the dynamic-quant
    # experiment on o_proj — see README §1.6c: a static per-tensor scale was measured to be a
    # pure trade-off (recalibrating it improves 5 prompts and regresses 5), so per-token dynamic
    # quantization is the option that can remove the calibration problem instead of moving it.
    if active.dim() == 3:
        B, S_len, ND = active.shape
        D = min(128, ND)  # head_dim is at most 128 (P_MAX)
        while ND % D != 0 and D > 1:
            D -= 1
        N = ND // D
        if quantization_type == QuantizationType.ROW:
            active = active.reshape(B, S_len, N, D)
        else:
            active = active.transpose(1, 2).reshape(B, N, D, S_len)
    # 🔴 A 4-D input is taken AS GIVEN — no shape sniffing. For ROW the caller must hand over
    # [B, S, N, D] itself (ministral3/model.py does this explicitly). Guessing the layout from
    # the extents would silently transpose a legitimate tensor whenever S happened to look like
    # a head count, and a wrong permute here produces plausible-looking garbage rather than an
    # error — precisely the failure mode this investigation has spent a session chasing.

    if _can_use_kernel(active, weight, quantization_type):
        wrapped_output_projection = wrap_nki(output_projection)

        return wrapped_output_projection[2](
            active,
            weight,
            bias,
            quantization_type,
            input_scales,
            weight_scales,
        )

    return _torch_o_proj_impl(
        active,
        weight,
        bias=bias,
        quantization_type=quantization_type,
        input_scales=input_scales,
        weight_scales=weight_scales,
    )
