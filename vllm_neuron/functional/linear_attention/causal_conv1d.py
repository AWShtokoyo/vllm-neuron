# SPDX-License-Identifier: Apache-2.0
"""Depthwise causal 1D convolution for gated-DeltaNet linear-attention layers.

Two entry points:

* :func:`causal_conv1d_prefill` -- full-sequence depthwise causal conv over a
  padded prompt (prefill), matching HF ``causal_conv1d_fn``.
* :func:`causal_conv1d_update` -- single-step (decode) depthwise causal conv
  that consumes and rolls a per-sequence ``conv_state``, matching HF
  ``causal_conv1d_update``.

Both take ``hidden_states`` shaped ``[batch, conv_dim, seq_len]`` and a
depthwise ``weight`` shaped ``[conv_dim, kernel_size]`` (i.e. the HF
``conv1d.weight.squeeze(1)``), ``groups == conv_dim``.

Device path: the self-contained NKI kernel in ``._causal_conv1d_kernel``.
The CPU/torch fallback (used by ``VLLM_NEURON_CPU_MODE=1``) is kept as a
reference.
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.activations import ACT2FN

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel

from ._causal_conv1d_kernel import (
    causal_conv1d_prefill as _nki_causal_conv1d_prefill,
)
from ._causal_conv1d_kernel import (
    causal_conv1d_update as _nki_causal_conv1d_update,
)

logger = logging.getLogger(__name__)

# Single-program kernels (loop over batch internally, --lnc 1); launch on 1 core.
_wrapped_prefill = wrap_nki(_nki_causal_conv1d_prefill)
_wrapped_update = wrap_nki(_nki_causal_conv1d_update)


def _torch_causal_conv1d_prefill(
    hidden_states: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    activation: str | None,
) -> Tensor:
    # Port of HF transformers qwen3_next.causal_conv1d_fn.
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def _torch_causal_conv1d_update(
    hidden_states: Tensor,
    conv_state: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    activation: str | None,
) -> Tensor:
    # Port of HF transformers qwen3_next.causal_conv1d_update.
    # conv_state is updated IN PLACE to hold the trailing (kernel_size-1) inputs.
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(
        hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size
    )
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def causal_conv1d_prefill(
    hidden_states: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    activation: str | None = None,
) -> Tensor:
    """Full-sequence depthwise causal conv1d (prefill).

    Args:
        hidden_states: ``[batch, conv_dim, seq_len]``.
        weight: depthwise conv weight ``[conv_dim, kernel_size]``.
        bias: optional ``[conv_dim]`` (``None`` for Qwen3.5).
        activation: activation name (e.g. ``"silu"``) or ``None``.

    Returns:
        ``[batch, conv_dim, seq_len]`` (activation applied).
    """
    if can_run_kernel(hidden_states):
        assert bias is None, "NKI causal_conv1d kernel does not support bias"
        return _wrapped_prefill[1](
            hidden_states=hidden_states,
            weight=weight,
            apply_silu=activation is not None,
        )
    return _torch_causal_conv1d_prefill(hidden_states, weight, bias, activation)


def causal_conv1d_update(
    hidden_states: Tensor,
    conv_state: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    activation: str | None = None,
) -> Tensor:
    """Single-step depthwise causal conv1d (decode); rolls ``conv_state`` in place.

    Args:
        hidden_states: ``[batch, conv_dim, seq_len]`` (``seq_len`` usually 1).
        conv_state: ``[batch, conv_dim, kernel_size-1]``, updated in place.
        weight: depthwise conv weight ``[conv_dim, kernel_size]``.
        bias: optional ``[conv_dim]`` (``None`` for Qwen3.5).
        activation: activation name (e.g. ``"silu"``) or ``None``.

    Returns:
        ``[batch, conv_dim, seq_len]`` (activation applied).
    """
    if can_run_kernel(hidden_states):
        assert bias is None, "NKI causal_conv1d kernel does not support bias"
        # Kernel returns the rolled state; mirror the in-place contract so
        # callers can scatter ``conv_state`` back to the paged cache.
        out, new_state = _wrapped_update[1](
            hidden_states=hidden_states,
            conv_state=conv_state,
            weight=weight,
            apply_silu=activation is not None,
        )
        conv_state.copy_(new_state)
        return out
    return _torch_causal_conv1d_update(
        hidden_states, conv_state, weight, bias, activation
    )
