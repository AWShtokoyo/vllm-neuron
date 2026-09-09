"""Depthwise causal 1D convolution NKI kernels for Qwen3.5 gated-DeltaNet.

Target: Trainium2 / trn2 (gen3), NKI 0.6.0 API.

Two entry points, matching transformers.models.qwen3_next.modeling_qwen3_next:

  * causal_conv1d_prefill  <- causal_conv1d_fn      (bias=None, activation="silu")
  * causal_conv1d_update   <- causal_conv1d_update  (bias=None, activation="silu")

Depthwise (groups == conv_dim) causal convolution semantics
-----------------------------------------------------------
For channel c, output position t (0-based):

    out[c, t] = sum_{k=0}^{K-1} weight[c, k] * x_ctx[c, t + k]

where ``x_ctx`` is the input with a length ``K-1`` left context:
  - prefill:  the context is zeros (F.conv1d padding=K-1, sliced to [:seq_len])
  - decode:   the context is ``conv_state`` (concat(conv_state, hidden))

This is a per-channel weighted sum of ``K`` shifted copies of the input, so it
maps cleanly onto tensor_scalar with a per-partition (per-channel) scalar
operand: channels tile the partition dim (<=128), the sequence tiles the free
dim.

Assumptions / constraints
-------------------------
  * bias is None (Qwen3.5 conv1d has bias=False). Bias support would be one
    extra per-partition tensor_scalar add.
  * activation is SiLU (silu(x) = x * sigmoid(x)); pass apply_silu=False to skip.
  * conv_dim tiled by 128 partitions; seq_len + (K-1) must be <= 32767 (SBUF
    free-dim limit) -- fine for seq_len up to ~2048.
  * Compute is done in fp32 for accuracy; output is cast back to the input dtype.
  * kernel_size (K) and state_len (=K-1) are compile-time constants read from
    the weight / conv_state shapes.
"""

import nki
import nki.isa as nisa
import nki.language as nl


def div_ceil(n, d):
    return (n + d - 1) // d


def kernel_assert(cond, msg):
    # NKI's tracer/parser rejects `raise` inside kernels but allows `assert`.
    assert cond, msg


_PMAX = 128


def _depthwise_conv_tile(x_ctx, weight_tile, out_len, k_size, apply_silu):
    """Compute out[:, 0:out_len] = sum_k weight[:,k] * x_ctx[:, k:k+out_len].

    Args:
        x_ctx: SBUF fp32 tile [P, K-1 + out_len]  (context columns then inputs)
        weight_tile: SBUF fp32 tile [P, K]
        out_len: number of output sequence positions
        k_size: kernel size K
        apply_silu: whether to apply SiLU to the result

    Returns:
        SBUF fp32 tile [P, out_len] holding the (activated) convolution result.
    """
    p_sz = x_ctx.shape[0]
    acc = nl.ndarray((p_sz, out_len), dtype=nl.float32, buffer=nl.sbuf)

    # k = 0: initialize accumulator
    nisa.tensor_scalar(
        dst=acc,
        data=x_ctx[0:p_sz, 0:out_len],
        op0=nl.multiply,
        operand0=weight_tile[0:p_sz, 0:1],
    )
    # k = 1 .. K-1: multiply-accumulate
    for k in range(1, k_size):
        tmp = nl.ndarray((p_sz, out_len), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=tmp,
            data=x_ctx[0:p_sz, k:k + out_len],
            op0=nl.multiply,
            operand0=weight_tile[0:p_sz, k:k + 1],
        )
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)

    if apply_silu:
        out_tile = nl.ndarray((p_sz, out_len), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=out_tile, data=acc, op=nl.silu)
        return out_tile
    return acc


@nki.jit
def causal_conv1d_prefill(hidden_states, weight, apply_silu=True):
    """Prefill depthwise causal conv1d (matches causal_conv1d_fn, bias=None).

    Args:
        hidden_states: HBM [batch, conv_dim, seq_len]
        weight:        HBM [conv_dim, kernel_size]
        apply_silu:    apply SiLU activation (default True)

    Returns:
        HBM [batch, conv_dim, seq_len]
    """
    batch, conv_dim, seq_len = hidden_states.shape
    conv_dim_w, k_size = weight.shape
    kernel_assert(conv_dim_w == conv_dim, "weight/conv_dim mismatch")
    state_len = k_size - 1
    kernel_assert(seq_len + state_len <= 32767, "seq_len+K-1 exceeds SBUF free dim")

    out = nl.ndarray(hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    n_ctiles = div_ceil(conv_dim, _PMAX)
    for b in range(batch):
        for ct in nl.affine_range(n_ctiles):
            c0 = ct * _PMAX
            p_sz = min(_PMAX, conv_dim - c0)

            # x_ctx = [zeros(K-1) | hidden]  -> [p_sz, state_len + seq_len]
            x_ctx = nl.ndarray((p_sz, state_len + seq_len), dtype=nl.float32, buffer=nl.sbuf)
            if state_len > 0:
                nisa.memset(dst=x_ctx[0:p_sz, 0:state_len], value=0.0)
            nisa.dma_copy(
                dst=x_ctx[0:p_sz, state_len:state_len + seq_len],
                src=hidden_states[b, c0:c0 + p_sz, 0:seq_len],
            )

            w_tile = nl.ndarray((p_sz, k_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_tile, src=weight[c0:c0 + p_sz, 0:k_size])

            res = _depthwise_conv_tile(x_ctx, w_tile, seq_len, k_size, apply_silu)

            out_sb = nl.ndarray((p_sz, seq_len), dtype=out.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_sb, src=res)
            nisa.dma_copy(dst=out[b, c0:c0 + p_sz, 0:seq_len], src=out_sb)

    return out


@nki.jit
def causal_conv1d_update(hidden_states, conv_state, weight, apply_silu=True):
    """Single/multi-step decode conv1d update (matches causal_conv1d_update).

    Concatenates conv_state with hidden_states along the sequence dim, runs the
    depthwise causal conv, and produces the updated conv_state (trailing
    state_len columns of the concatenation).

    Args:
        hidden_states: HBM [batch, conv_dim, seq_len]  (seq_len typically 1)
        conv_state:    HBM [batch, conv_dim, state_len]  (state_len = K-1)
        weight:        HBM [conv_dim, kernel_size]
        apply_silu:    apply SiLU activation (default True)

    Returns:
        (out, new_conv_state):
          out            HBM [batch, conv_dim, seq_len]
          new_conv_state HBM [batch, conv_dim, state_len]  (write back in place)
    """
    batch, conv_dim, seq_len = hidden_states.shape
    _, _, state_len = conv_state.shape
    conv_dim_w, k_size = weight.shape
    kernel_assert(conv_dim_w == conv_dim, "weight/conv_dim mismatch")
    kernel_assert(state_len == k_size - 1, "state_len must equal kernel_size-1")

    ctx_len = state_len + seq_len  # length of concat(conv_state, hidden)

    out = nl.ndarray(hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    new_state = nl.ndarray(conv_state.shape, dtype=conv_state.dtype, buffer=nl.shared_hbm)

    n_ctiles = div_ceil(conv_dim, _PMAX)
    for b in range(batch):
        for ct in nl.affine_range(n_ctiles):
            c0 = ct * _PMAX
            p_sz = min(_PMAX, conv_dim - c0)

            # x_ctx = concat(conv_state, hidden) -> [p_sz, state_len + seq_len]
            x_ctx = nl.ndarray((p_sz, ctx_len), dtype=nl.float32, buffer=nl.sbuf)
            if state_len > 0:
                nisa.dma_copy(
                    dst=x_ctx[0:p_sz, 0:state_len],
                    src=conv_state[b, c0:c0 + p_sz, 0:state_len],
                )
            nisa.dma_copy(
                dst=x_ctx[0:p_sz, state_len:ctx_len],
                src=hidden_states[b, c0:c0 + p_sz, 0:seq_len],
            )

            w_tile = nl.ndarray((p_sz, k_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_tile, src=weight[c0:c0 + p_sz, 0:k_size])

            res = _depthwise_conv_tile(x_ctx, w_tile, seq_len, k_size, apply_silu)

            out_sb = nl.ndarray((p_sz, seq_len), dtype=out.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_sb, src=res)
            nisa.dma_copy(dst=out[b, c0:c0 + p_sz, 0:seq_len], src=out_sb)

            # Updated state = trailing state_len columns of the concatenation.
            if state_len > 0:
                new_state_sb = nl.ndarray((p_sz, state_len), dtype=new_state.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=new_state_sb,
                    src=x_ctx[0:p_sz, ctx_len - state_len:ctx_len],
                )
                nisa.dma_copy(dst=new_state[b, c0:c0 + p_sz, 0:state_len], src=new_state_sb)

    return out, new_state
