# SPDX-License-Identifier: Apache-2.0
"""Gated delta-rule recurrence for gated-DeltaNet linear-attention layers.

Two entry points:

* :func:`chunk_gated_delta_rule` -- chunked parallel form for prefill
  (matches HF ``torch_chunk_gated_delta_rule``).
* :func:`recurrent_gated_delta_rule` -- single-step recurrent form for decode
  (matches HF ``torch_recurrent_gated_delta_rule``).

Tensor layout (matching the HF qwen3_next reference, post repeat-interleave so
key/value share ``num_v_heads``):

* ``query`` / ``key``: ``[batch, seq_len, num_v_heads, k_head_dim]``
* ``value``: ``[batch, seq_len, num_v_heads, v_head_dim]``
* ``g`` / ``beta``: ``[batch, seq_len, num_v_heads]``
* ``initial_state`` / ``last_recurrent_state``:
  ``[batch, num_v_heads, k_head_dim, v_head_dim]`` (fp32)

All recurrence math runs in fp32 (``mamba_ssm_dtype``), regardless of the
bf16 input dtype; the output is cast back to the input dtype.

Device path defers to the self-contained NKI kernels in
``._gated_delta_rule_kernel``; the CPU/torch form is kept as a reference
fallback for environments without a Neuron device.
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel

from ._gated_delta_rule_kernel import (
    chunk_gated_delta_rule_kernel as _nki_chunk_kernel,
)
from ._gated_delta_rule_kernel import (
    recurrent_gated_delta_rule_kernel as _nki_recurrent_kernel,
)

logger = logging.getLogger(__name__)

# Single-program kernels (loop over batch/heads internally, --lnc 1).
_wrapped_chunk = wrap_nki(_nki_chunk_kernel)
_wrapped_recurrent = wrap_nki(_nki_recurrent_kernel)


def _l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _torch_chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    chunk_size: int,
    initial_state: Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[Tensor, Tensor | None]:
    # Port of HF transformers qwen3_next.torch_chunk_gated_delta_rule.
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            k_head_dim,
            v_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _torch_recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[Tensor, Tensor | None]:
    # Port of HF transformers qwen3_next.torch_recurrent_gated_delta_rule.
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size,
        num_heads,
        sequence_length,
        v_head_dim,
        dtype=value.dtype,
        device=value.device,
    )
    last_recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            k_head_dim,
            v_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )

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
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    chunk_size: int = 64,
    initial_state: Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Chunked parallel gated delta-rule (prefill).

    Returns ``(core_attn_out, last_recurrent_state)`` where ``core_attn_out``
    is ``[batch, seq_len, num_v_heads, v_head_dim]`` in the input dtype and
    ``last_recurrent_state`` is ``[batch, num_v_heads, k_head_dim, v_head_dim]``
    in fp32 (or ``None`` if ``output_final_state`` is False).
    """
    if can_run_kernel(query):
        assert use_qk_l2norm_in_kernel, (
            "NKI chunk kernel always applies q/k l2norm; "
            "use_qk_l2norm_in_kernel=False is unsupported on device"
        )
        batch, seqlen, num_heads, k_head_dim = query.shape
        v_head_dim = value.shape[-1]
        # Kernel requires seq_len % 64 == 0 and a (non-None) fp32 initial state.
        pad = (chunk_size - seqlen % chunk_size) % chunk_size
        if pad:
            query = F.pad(query, (0, 0, 0, 0, 0, pad))
            key = F.pad(key, (0, 0, 0, 0, 0, pad))
            value = F.pad(value, (0, 0, 0, 0, 0, pad))
            g = F.pad(g, (0, 0, 0, pad))
            beta = F.pad(beta, (0, 0, 0, pad))
        if initial_state is None:
            initial_state = torch.zeros(
                batch, num_heads, k_head_dim, v_head_dim,
                dtype=torch.float32, device=query.device,
            )
        core_attn_out, last_state = _wrapped_chunk[1](
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=initial_state.to(torch.float32),
        )
        if pad:
            core_attn_out = core_attn_out[:, :seqlen]
        return core_attn_out, (last_state if output_final_state else None)
    return _torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
    )


def recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Single-step recurrent gated delta-rule (decode).

    Returns ``(core_attn_out, last_recurrent_state)`` where ``core_attn_out``
    is ``[batch, seq_len, num_v_heads, v_head_dim]`` in the input dtype and
    ``last_recurrent_state`` is ``[batch, num_v_heads, k_head_dim, v_head_dim]``
    in fp32 (or ``None`` if ``output_final_state`` is False).
    """
    if can_run_kernel(query):
        assert use_qk_l2norm_in_kernel, (
            "NKI recurrent kernel always applies q/k l2norm; "
            "use_qk_l2norm_in_kernel=False is unsupported on device"
        )
        assert initial_state is not None, (
            "NKI recurrent kernel requires a (non-None) fp32 initial state"
        )
        core_attn_out, last_state = _wrapped_recurrent[1](
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=initial_state.to(torch.float32),
        )
        return core_attn_out, (last_state if output_final_state else None)
    return _torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
    )
