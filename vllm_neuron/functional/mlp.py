# SPDX-License-Identifier: Apache-2.0
import math

import torch
from torch import Tensor

from nkilib.core.mlp.mlp import mlp as nkilib_mlp
from nkilib.core.utils.common_types import (
    ActFnType,
    DtypeMode,
    NormType,
    QuantizationType,
)
from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

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

    if _can_use_kernel(hidden, gate_w, gate_b, up_b, down_b):
        # Reshape [T, H] -> [1, T, H] for kernel (kernel expects [B, S, H])
        hidden_3d = hidden.unsqueeze(0)

        # Provide default normalization weights if norm is requested but ln_w is None
        if norm_type != NormType.NO_NORM and ln_w is None:
            ln_w = torch.ones((1, H), dtype=hidden.dtype, device=hidden.device)

        # Reshape fused_add_tensor to 3D if provided
        fused_add_3d = None
        if fused_add_tensor is not None:
            fused_add_3d = fused_add_tensor.unsqueeze(0)

        _column_tiling = quantization_type != QuantizationType.STATIC_MX

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
            force_cte_mode=False,
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

    Note: The nkilib kernel supports bfloat16 and float8 (float8_e4m3) dtypes.
    Other dtypes are not supported and will cause a validation error from nkilib.
    Callers are responsible for providing tensors with a supported dtype.

    Args:
        hidden: Input hidden state tensor with shape [T, H].
        gate_w: Gate projection weights with shape [H, I].
        gate_b: Optional gate projection bias with shape [1, I].
        up_b: Optional up projection bias with shape [1, I].
        down_b: Optional down projection bias with shape [1, H].

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
        # Inline the tanh approximation instead of
        # ``F.gelu(x, approximate="tanh")``: on the libtorch_neuronx_lite stack
        # that call dispatches to ``torch._C._nn.gelu``, which Dynamo cannot
        # trace ("Attempted to call function marked as skipped"). This bites the
        # torch fallback path (``_torch_mlp_impl``) used by models whose
        # intermediate_size exceeds the NKI MLP kernel tiling constraint (e.g.
        # Gemma4, I=21504). 0.7978845608028654 == sqrt(2/pi).
        #
        # Evaluate in fp32 and cast back, which is what ``F.gelu`` does
        # internally. Computing the cube and tanh in bf16/fp16 throughout
        # instead costs ~15 ULP on average and up to ~257 ULP worst case
        # (measured on CPU against an fp32 reference), because ``x**3``
        # overflows the mantissa; ``F.gelu`` stays within 1 ULP. The MLP
        # activation feeds every subsequent layer, so that error compounds —
        # and this resolver is shared by every model using GELU_Tanh_Approx,
        # not just Gemma4.
        def _gelu_tanh(x: Tensor) -> Tensor:
            x32 = x.float()
            out = (
                0.5
                * x32
                * (
                    1.0
                    + torch.tanh(
                        0.7978845608028654 * (x32 + 0.044715 * x32 * x32 * x32)
                    )
                )
            )
            return out.to(x.dtype)

        return _gelu_tanh
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
