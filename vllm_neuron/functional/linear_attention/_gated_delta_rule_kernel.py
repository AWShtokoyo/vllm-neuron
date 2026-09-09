"""NKI kernels for Qwen3.5 gated DeltaNet linear-attention recurrence (Trainium2 / gen3).

Two entry points matching HuggingFace `transformers.models.qwen3_next.modeling_qwen3_next`:

  * `recurrent_gated_delta_rule_kernel`  -> decode (seq_len == 1) single-step recurrence
      (reference: torch_recurrent_gated_delta_rule)
  * `chunk_gated_delta_rule_kernel`      -> prefill chunked parallel delta rule
      (reference: torch_chunk_gated_delta_rule)

All recurrence math is done in fp32. l2norm(eps=1e-6) is applied to q,k inside the
kernel; query is scaled by 1/sqrt(k_head_dim).

Layouts (match the reference, heads already repeat_interleaved to num_v_heads):
  query/key : [batch, seq_len, num_heads, k_head_dim]  (bf16 or fp32)
  value     : [batch, seq_len, num_heads, v_head_dim]
  g, beta   : [batch, seq_len, num_heads]
  state     : [batch, num_heads, k_head_dim, v_head_dim] fp32   (S[k, v])

Outputs:
  core_attn_out       : [batch, seq_len, num_heads, v_head_dim]  (same dtype as query)
  last_recurrent_state: [batch, num_heads, k_head_dim, v_head_dim] fp32
"""

import os

import nki
import nki.isa as nisa
import nki.language as nl


def kernel_assert(cond, msg):
    # NKI's tracer rejects `raise`; use `assert` (matches causal_conv1d kernel).
    assert cond, "[NKI gated_delta_rule] " + msg


def _transpose_pe(data, P, F):
    """Tensor-engine transpose [P, F] -> [F, P] fp32, returned in SBUF."""
    ps = nl.ndarray((F, P), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=ps, data=data, engine=nisa.engine.tensor)
    sb = nl.ndarray((F, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sb, src=ps)
    return sb


def _l2norm_scale_inplace(t, P, D, scale):
    """In-place l2norm of `t[P, D]` over the free dim D, then multiply by `scale`.

    `t` must already be an fp32 SBUF tile. Multiplying by scale=1.0 is a no-op,
    so callers pass the real scale for q and 1.0 for k (avoids a data-dependent
    branch, which the strict NKI parser prefers).
    """
    sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=t, data2=t, op=nl.multiply)
    ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=(1,))
    inv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv, op=nl.rsqrt, data=ss, bias=1e-6)
    nisa.tensor_scalar(dst=inv, data=inv, op0=nl.multiply, operand0=float(scale))
    nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=inv)


def _l2norm_scale_transpose(t_bf, P, D, scale):
    """Copy `t_bf[P, D]` to fp32, l2norm over D, multiply by `scale`, return [D, P]."""
    t = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=t, src=t_bf)
    _l2norm_scale_inplace(t, P, D, scale)
    return _transpose_pe(t, P, D)  # [D, P]


@nki.jit
def recurrent_gated_delta_rule_kernel(query, key, value, g, beta, initial_state):
    """Decode single-step gated delta rule (seq_len == 1).

    Args:
        query, key : [batch, 1, num_heads, k_head_dim]
        value      : [batch, 1, num_heads, v_head_dim]
        g, beta    : [batch, 1, num_heads]
        initial_state : [batch, num_heads, k_head_dim, v_head_dim] fp32

    Returns:
        core_attn_out        : [batch, 1, num_heads, v_head_dim] (query dtype)
        last_recurrent_state : [batch, num_heads, k_head_dim, v_head_dim] fp32
    """
    batch, seqlen, H, Dk = query.shape
    Dv = value.shape[-1]
    kernel_assert(seqlen == 1, "decode kernel requires seq_len == 1")
    kernel_assert(Dk <= 128 and Dv <= 512, "head dims out of range")
    BH = batch * H
    kernel_assert(BH <= 128, "batch*num_heads must be <= 128")
    scale = 1.0 / (Dk ** 0.5)

    out = nl.ndarray((batch, 1, H, Dv), dtype=query.dtype, buffer=nl.shared_hbm)
    state_out = nl.ndarray((batch, H, Dk, Dv), dtype=nl.float32, buffer=nl.shared_hbm)

    # --- Load q,k for all (batch,head) as [BH, Dk] ---
    q_bf = nl.ndarray((BH, Dk), dtype=query.dtype, buffer=nl.sbuf)
    k_bf = nl.ndarray((BH, Dk), dtype=key.dtype, buffer=nl.sbuf)
    for b in nl.affine_range(batch):
        nisa.dma_copy(dst=q_bf[b * H:(b + 1) * H, 0:Dk], src=query[b, 0, 0:H, 0:Dk])
        nisa.dma_copy(dst=k_bf[b * H:(b + 1) * H, 0:Dk], src=key[b, 0, 0:H, 0:Dk])

    # Transposed, l2-normed q/k as [Dk, BH] fp32 (separate single-target
    # assignments; the strict NKI parser rejects `a, b = f(...)` tuple unpack).
    qt = _l2norm_scale_transpose(q_bf, BH, Dk, scale)  # [Dk, BH] fp32
    kt = _l2norm_scale_transpose(k_bf, BH, Dk, 1.0)    # [Dk, BH] fp32

    # g,beta as [1, BH] at partition 0
    g_row = nl.ndarray((1, BH), dtype=nl.float32, buffer=nl.sbuf)
    beta_row = nl.ndarray((1, BH), dtype=nl.float32, buffer=nl.sbuf)
    g_bf = nl.ndarray((1, BH), dtype=g.dtype, buffer=nl.sbuf)
    beta_bf = nl.ndarray((1, BH), dtype=beta.dtype, buffer=nl.sbuf)
    for b in nl.affine_range(batch):
        nisa.dma_copy(dst=g_bf[0:1, b * H:(b + 1) * H], src=g[b, 0:1, 0:H])
        nisa.dma_copy(dst=beta_bf[0:1, b * H:(b + 1) * H], src=beta[b, 0:1, 0:H])
    nisa.activation(dst=g_row, op=nl.exp, data=g_bf)  # g_exp
    nisa.tensor_copy(dst=beta_row, src=beta_bf)

    # G_bc[k, bh] = g_exp[bh] broadcast over k partitions (for scaling the state)
    ones_row = nl.ndarray((1, Dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_row, value=1.0)
    G_bc_ps = nl.ndarray((Dk, BH), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=G_bc_ps, stationary=ones_row, moving=g_row)
    G_bc = nl.ndarray((Dk, BH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=G_bc, src=G_bc_ps)

    for b in nl.affine_range(batch):
        for h in nl.affine_range(H):
            bh = b * H + h
            S_old = nl.ndarray((Dk, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=S_old, src=initial_state[b, h, 0:Dk, 0:Dv])
            v_h = nl.ndarray((1, Dv), dtype=nl.float32, buffer=nl.sbuf)
            v_bf = nl.ndarray((1, Dv), dtype=value.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_bf, src=value[b, 0, h:h + 1, 0:Dv])
            nisa.tensor_copy(dst=v_h, src=v_bf)

            k_col = kt[0:Dk, bh:bh + 1]  # [Dk,1]
            q_col = qt[0:Dk, bh:bh + 1]  # [Dk,1]

            # a = k^T @ S_old, c = q^T @ S_old, qk = q.k
            a_ps = nl.ndarray((1, Dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=a_ps, stationary=k_col, moving=S_old)
            c_ps = nl.ndarray((1, Dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=c_ps, stationary=q_col, moving=S_old)
            qk_ps = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=qk_ps, stationary=q_col, moving=k_col)

            a = nl.ndarray((1, Dv), dtype=nl.float32, buffer=nl.sbuf)
            c = nl.ndarray((1, Dv), dtype=nl.float32, buffer=nl.sbuf)
            qk = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=a, src=a_ps)
            nisa.tensor_copy(dst=c, src=c_ps)
            nisa.tensor_copy(dst=qk, src=qk_ps)

            g_s = g_row[0:1, bh:bh + 1]        # [1,1]
            beta_s = beta_row[0:1, bh:bh + 1]  # [1,1]

            # delta = (v - g*a) * beta
            ga = nl.ndarray((1, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ga, data=a, op0=nl.multiply, operand0=g_s)
            delta = nl.ndarray((1, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=delta, data1=v_h, data2=ga, op=nl.subtract)
            nisa.tensor_scalar(dst=delta, data=delta, op0=nl.multiply, operand0=beta_s)

            # out = g*c + qk*delta
            gc = nl.ndarray((1, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=gc, data=c, op0=nl.multiply, operand0=g_s)
            qkd = nl.ndarray((1, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=qkd, data=delta, op0=nl.multiply, operand0=qk[0:1, 0:1])
            out_row = nl.ndarray((1, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=out_row, data1=gc, data2=qkd, op=nl.add)
            out_bf = nl.ndarray((1, Dv), dtype=query.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_bf, src=out_row)
            nisa.dma_copy(dst=out[b, 0, h:h + 1, 0:Dv], src=out_bf)

            # S_new = g*S_old + outer(k, delta)
            k_row = _transpose_pe(k_col, Dk, 1)  # [1, Dk]
            outer_ps = nl.ndarray((Dk, Dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=outer_ps, stationary=k_row, moving=delta)  # [Dk, Dv]
            gS = nl.ndarray((Dk, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=gS, data=S_old, op0=nl.multiply, operand0=G_bc[0:Dk, bh:bh + 1])
            S_new = nl.ndarray((Dk, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=S_new, data1=gS, data2=outer_ps, op=nl.add)
            nisa.dma_copy(dst=state_out[b, h, 0:Dk, 0:Dv], src=S_new)

    return out, state_out


# =====================================================================
#  Chunked prefill gated delta rule
# =====================================================================

def _matmul_AB(A, B, P, K, N):
    """Return A[P,K] @ B[K,N] = [P,N] in SBUF fp32 (contraction K on partition).

    B must already have K on its partition axis. A is transposed internally.
    """
    At = _transpose_pe(A, P, K)          # [K, P]
    ps = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=At, moving=B)   # dst[P,N] = sum_k A[P,k] B[k,N]
    out = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=ps)
    return out


def _bcast_col(scalar_11, P):
    """Broadcast a [1,1] value (partition 0) to a [P,1] column via ones matmul."""
    ones_1P = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_1P, value=1.0)
    ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=ones_1P, moving=scalar_11)  # [P,1]
    out = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=ps)
    return out


def _bmm_pt(ST, X, P, K, N):
    """Return (ST^T) @ X = [P,N], where ST is an ALREADY-TRANSPOSED stationary
    [K,P] and X is moving [K,N] (contraction K on the partition of both). No
    internal transpose (unlike _matmul_AB): the caller passes a stationary that
    is already oriented, e.g. a block of At = A^T."""
    ps = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=ST, moving=X)
    out = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=ps)
    return out


def _gather_blk(src, pr, pc, b):
    """Copy the b x b sub-block src[pr:pr+b, pc:pc+b] into a fresh partition-0
    SBUF tile. nc_matmul operands must start at partition 0, but a sub-block of
    a [C,C] tile with pr>0 starts at partition pr; DMA can move across
    partitions (a plain tensor_copy cannot). Free-dim offset (pc) is fine."""
    out = nl.ndarray((b, b), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=out, src=src[pr:pr + b, pc:pc + b])
    return out


def _horner_diag_block(AiiT, eye_b, b):
    """D = (I - A_ii)^{-1} for a strict-lower b x b block, via a b-1 step Horner
    series D = I + A_ii @ D. AiiT = (A_ii)^T is a constant stationary (reused
    every step, no per-step transpose). Bounded and stable for small b."""
    D = nl.ndarray((b, b), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=D, src=eye_b)                      # D = I
    for _s in nl.static_range(b - 1):
        p = nl.ndarray((b, b), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=p, stationary=AiiT, moving=D)    # A_ii @ D
        nisa.tensor_tensor(dst=D, data1=eye_b, data2=p, op=nl.add)  # D = I + A_ii@D
    return D


def _inverse_blocked_horner(A, At, eye, C):
    """T = (I - A)^{-1} for strict-lower-triangular A[C,C], via a 4x4 block
    (b=16) decomposition. Diagonal blocks D_i = (I-A_ii)^{-1} by a short Horner
    series; off-diagonal blocks by block forward-substitution
    T_ij = D_i @ (sum_{j<=k<i} A_ik @ T_kj).  A_ik@X reuses At[k,i] (=A_ik^T)
    directly as a pre-transposed stationary via _bmm_pt.

    Why blocked instead of a flat 63-step Horner: the flat inverse is a 63-deep
    sequential matmul chain and empirically dominates prefill TTFT (8-step vs
    63-step probe measured 200ms vs 399ms). The blocked form's critical path is
    ~(b-1) diagonal Horner steps + a few forward-sub matmuls (16x16) instead of
    63 of 64x64, recovering most of that TTFT while staying numerically stable:
    rel ~2e-7 vs fp64 (matching flat Horner) under a strict per-FMA fp32
    emulation of the PE, bounded (max|T|=1) throughout, on the correlated-key /
    weak-decay inputs where the old doubling method exploded (rel ~1e6).

    Hand-unrolled for nb=4 (the strict NKI parser rejects dict/list containers).
    """
    b = 16
    eye_b = eye[0:b, 0:b]                       # [b,b] identity (top-left of eye)
    T = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=T, value=0.0)               # block-upper triangle stays 0

    # Gather the 10 lower-triangular A blocks, ALREADY TRANSPOSED, into
    # partition-0 tiles. a_ki := At[k*b:(k+1)*b, i*b:(i+1)*b] = (A_ik)^T (a
    # matmul stationary, so it must live at partition 0 -> DMA gather). Names
    # a_ki carry (row-block k, col-block i) with k <= i.
    a00 = _gather_blk(At, 0, 0, b)
    a01 = _gather_blk(At, 0, b, b)
    a02 = _gather_blk(At, 0, 2 * b, b)
    a03 = _gather_blk(At, 0, 3 * b, b)
    a11 = _gather_blk(At, b, b, b)
    a12 = _gather_blk(At, b, 2 * b, b)
    a13 = _gather_blk(At, b, 3 * b, b)
    a22 = _gather_blk(At, 2 * b, 2 * b, b)
    a23 = _gather_blk(At, 2 * b, 3 * b, b)
    a33 = _gather_blk(At, 3 * b, 3 * b, b)

    # ---- diagonal blocks: D_i = (I - A_ii)^{-1} ----
    D0 = _horner_diag_block(a00, eye_b, b)
    D1 = _horner_diag_block(a11, eye_b, b)
    D2 = _horner_diag_block(a22, eye_b, b)
    D3 = _horner_diag_block(a33, eye_b, b)
    nisa.dma_copy(dst=T[0:b, 0:b], src=D0)
    nisa.dma_copy(dst=T[b:2 * b, b:2 * b], src=D1)
    nisa.dma_copy(dst=T[2 * b:3 * b, 2 * b:3 * b], src=D2)
    nisa.dma_copy(dst=T[3 * b:4 * b, 3 * b:4 * b], src=D3)
    # D_i^T for the final D_i @ acc left-multiply (i = 1,2,3)
    D1t = _transpose_pe(D1, b, b)
    D2t = _transpose_pe(D2, b, b)
    D3t = _transpose_pe(D3, b, b)

    # ---- off-diagonal blocks: block forward substitution (j-major) ----
    # T_ij = D_i @ (sum_{j<=k<i} A_ik @ T_kj); a_ki is A_ik^T (stationary).
    # j = 0
    T10 = _bmm_pt(D1t, _bmm_pt(a01, D0, b, b, b), b, b, b)
    nisa.dma_copy(dst=T[b:2 * b, 0:b], src=T10)
    acc20 = _bmm_pt(a02, D0, b, b, b)                 # A_20 @ D0
    acc20b = _bmm_pt(a12, T10, b, b, b)               # A_21 @ T10
    nisa.tensor_tensor(dst=acc20, data1=acc20, data2=acc20b, op=nl.add)
    T20 = _bmm_pt(D2t, acc20, b, b, b)
    nisa.dma_copy(dst=T[2 * b:3 * b, 0:b], src=T20)
    acc30 = _bmm_pt(a03, D0, b, b, b)                 # A_30 @ D0
    acc30b = _bmm_pt(a13, T10, b, b, b)               # A_31 @ T10
    acc30c = _bmm_pt(a23, T20, b, b, b)               # A_32 @ T20
    nisa.tensor_tensor(dst=acc30, data1=acc30, data2=acc30b, op=nl.add)
    nisa.tensor_tensor(dst=acc30, data1=acc30, data2=acc30c, op=nl.add)
    T30 = _bmm_pt(D3t, acc30, b, b, b)
    nisa.dma_copy(dst=T[3 * b:4 * b, 0:b], src=T30)
    # j = 1
    T21 = _bmm_pt(D2t, _bmm_pt(a12, D1, b, b, b), b, b, b)
    nisa.dma_copy(dst=T[2 * b:3 * b, b:2 * b], src=T21)
    acc31 = _bmm_pt(a13, D1, b, b, b)                 # A_31 @ D1
    acc31b = _bmm_pt(a23, T21, b, b, b)               # A_32 @ T21
    nisa.tensor_tensor(dst=acc31, data1=acc31, data2=acc31b, op=nl.add)
    T31 = _bmm_pt(D3t, acc31, b, b, b)
    nisa.dma_copy(dst=T[3 * b:4 * b, b:2 * b], src=T31)
    # j = 2
    T32 = _bmm_pt(D3t, _bmm_pt(a23, D2, b, b, b), b, b, b)
    nisa.dma_copy(dst=T[3 * b:4 * b, 2 * b:3 * b], src=T32)
    return T


@nki.jit
def chunk_gated_delta_rule_kernel(query, key, value, g, beta, initial_state):
    """Chunked parallel gated delta rule for prefill.

    Reference: torch_chunk_gated_delta_rule (chunk_size = 64, output_final_state=True,
    use_qk_l2norm_in_kernel=True). seq_len MUST be a multiple of 64 (pad on the host).

    Args:
        query, key : [batch, seq_len, num_heads, k_head_dim]
        value      : [batch, seq_len, num_heads, v_head_dim]
        g, beta    : [batch, seq_len, num_heads]
        initial_state : [batch, num_heads, k_head_dim, v_head_dim] fp32
                        (pass zeros for a fresh prompt)

    Returns:
        core_attn_out        : [batch, seq_len, num_heads, v_head_dim] (query dtype)
        last_recurrent_state : [batch, num_heads, k_head_dim, v_head_dim] fp32
    """
    C = 64
    batch, seqlen, H, Dk = query.shape
    Dv = value.shape[-1]
    kernel_assert(seqlen % C == 0, "seq_len must be a multiple of chunk_size (64)")
    kernel_assert(Dk <= 128 and Dv <= 128, "head dims must be <= 128")
    n_chunks = seqlen // C
    scale = 1.0 / (Dk ** 0.5)

    out = nl.ndarray((batch, seqlen, H, Dv), dtype=query.dtype, buffer=nl.shared_hbm)
    state_out = nl.ndarray((batch, H, Dk, Dv), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- constant [C,C] masks (built once) ----
    row_idx = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    col_idx = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=row_idx, pattern=[[0, C]], channel_multiplier=1, offset=0)  # =i
    nisa.iota(dst=col_idx, pattern=[[1, C]], channel_multiplier=0, offset=0)  # =j
    lt_incl = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)   # i>=j
    strict_lo = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)  # i>j
    ut_cumsum = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)  # [j,i]: i>=j
    eye = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)        # i==j
    nisa.tensor_tensor(dst=lt_incl, data1=row_idx, data2=col_idx, op=nl.greater_equal)
    nisa.tensor_tensor(dst=strict_lo, data1=row_idx, data2=col_idx, op=nl.greater)
    nisa.tensor_tensor(dst=ut_cumsum, data1=col_idx, data2=row_idx, op=nl.greater_equal)
    nisa.tensor_tensor(dst=eye, data1=row_idx, data2=col_idx, op=nl.equal)
    # all-ones helpers (partition dim = C = 64, well aligned) used to build the
    # decay matrix and the per-chunk g reductions without any partition-dim-1
    # transpose (see notes at each use site).
    ones_CC = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_CC, value=1.0)
    ones_C1 = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_C1, value=1.0)

    for b in nl.affine_range(batch):
        for h in nl.affine_range(H):
            # recurrent state S[Dk,Dv], starts from initial_state
            S = nl.ndarray((Dk, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=S, src=initial_state[b, h, 0:Dk, 0:Dv])

            for c in nl.sequential_range(n_chunks):
                s0 = c * C
                # ---- load chunk ----
                Qb = nl.ndarray((C, Dk), dtype=query.dtype, buffer=nl.sbuf)
                Kb = nl.ndarray((C, Dk), dtype=key.dtype, buffer=nl.sbuf)
                Vb = nl.ndarray((C, Dv), dtype=value.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=Qb, src=query[b, s0:s0 + C, h, 0:Dk])
                nisa.dma_copy(dst=Kb, src=key[b, s0:s0 + C, h, 0:Dk])
                nisa.dma_copy(dst=Vb, src=value[b, s0:s0 + C, h, 0:Dv])
                gbeta = nl.ndarray((C, 1), dtype=g.dtype, buffer=nl.sbuf)
                bbeta = nl.ndarray((C, 1), dtype=beta.dtype, buffer=nl.sbuf)
                # g,beta are [batch,seq,H] -> column [C,1]
                nisa.dma_copy(dst=gbeta, src=g[b, s0:s0 + C, h:h + 1])
                nisa.dma_copy(dst=bbeta, src=beta[b, s0:s0 + C, h:h + 1])

                Q = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.sbuf)
                K = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.sbuf)
                V = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
                g_raw = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
                bta = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=Q, src=Qb)
                nisa.tensor_copy(dst=K, src=Kb)
                nisa.tensor_copy(dst=V, src=Vb)
                nisa.tensor_copy(dst=g_raw, src=gbeta)
                nisa.tensor_copy(dst=bta, src=bbeta)

                # ---- l2norm(Q,K) over Dk, scale Q ----
                # Unrolled into two straight-line calls (no tuple-unpack for-loop:
                # the strict NKI parser rejects `for T_, sc in (...)`).
                _l2norm_scale_inplace(Q, C, Dk, scale)
                _l2norm_scale_inplace(K, C, Dk, 1.0)

                # kbeta = K*beta, vbeta = V*beta
                kbeta = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.sbuf)
                vbeta = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=kbeta, data=K, op0=nl.multiply, operand0=bta)
                nisa.tensor_scalar(dst=vbeta, data=V, op0=nl.multiply, operand0=bta)

                # ---- cumulative g within chunk: g_cum[C,1] = sum_{j<=i} g_raw[j] ----
                gcum_ps = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=gcum_ps, stationary=ut_cumsum, moving=g_raw)  # [i,0]
                g_cum = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=g_cum, src=gcum_ps)
                eg_col = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)   # exp(g_cum)
                nisa.activation(dst=eg_col, op=nl.exp, data=g_cum)

                # decay_mask[i,j] = exp(g_cum[i]-g_cum[j]) for i>=j else 0.
                # Build gj_mat[i,j] = g_cum[j] (g_cum broadcast across rows) WITHOUT
                # the old partition-dim-1 transpose of g_cum to a [1,C] row. Instead
                # mask g_raw by the cumsum pattern (grow_masked[k,j] = g_raw[k] if
                # j>=k else 0) and sum over the contraction with an all-ones
                # stationary. Both matmul operands are [C,C] (partition = C = 64),
                # so no tiny/oddly-strided transpose is emitted.
                grow_masked = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=grow_masked, data=ut_cumsum, op0=nl.multiply,
                                   operand0=g_raw)
                gj_ps = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=gj_ps, stationary=ones_CC, moving=grow_masked)  # [i,j]=g_cum[j]
                gj_mat = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=gj_mat, src=gj_ps)
                diff = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
                # diff[i,j] = g_cum[i] - g_cum[j]
                nisa.tensor_scalar(dst=diff, data=gj_mat, op0=nl.subtract, operand0=g_cum,
                                   reverse0=True)
                # Mask to the lower triangle (i>=j) BEFORE the exp, matching the
                # torch reference `(...).tril().exp().tril()`. In the strict upper
                # triangle (i<j) diff = g_cum[i]-g_cum[j] > 0 and grows with depth
                # (g is negative, g_cum decreasing); by the deeper layers it
                # reaches ~90, so exp(diff) overflows fp32 (>3.4e38) to +inf and
                # the later `decay * lt_incl` would give inf*0 = NaN. Zeroing the
                # upper triangle first makes exp(0)=1 there (finite), and it is
                # masked out again below, so real (lower-tri) values are unchanged.
                nisa.tensor_tensor(dst=diff, data1=diff, data2=lt_incl, op=nl.multiply)
                decay = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=decay, op=nl.exp, data=diff)
                decay_mask = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=decay_mask, data1=decay, data2=lt_incl, op=nl.multiply)

                # ---- attn (for the inverse): A = -(kbeta@K^T * decay_mask), strictly lower ----
                Kt = _transpose_pe(K, C, Dk)   # [Dk, C]
                kk = _matmul_AB(kbeta, Kt, C, Dk, C)   # [C,C]
                A = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=A, data1=kk, data2=decay_mask, op=nl.multiply)
                nisa.tensor_scalar(dst=A, data=A, op0=nl.multiply, operand0=-1.0)
                nisa.tensor_tensor(dst=A, data1=A, data2=strict_lo, op=nl.multiply)  # strict lower

                # ---- T = (I - A)^{-1} = sum_{k=0}^{C-1} A^k, A strict-lower ----
                # Computed by a 4x4-block (b=16) decomposition (_inverse_blocked_horner):
                # diagonal blocks by a short Horner series, off-diagonals by block
                # forward-substitution. This replaces both the old doubling method
                # (prod_j (I+A^{2^j}), which forms explicit high powers A^{2^j} ~1e18 for
                # correlated-key / weak-decay prefill and diverged to ~1e6 -> decode
                # collapse "!!!!"/NaN) and a flat 63-step Horner (numerically perfect but a
                # 63-deep matmul chain that dominated prefill TTFT). Blocked keeps the same
                # stability (rel ~2e-7 vs fp64 under
                # strict per-FMA fp32, bounded max|T|=1 throughout) with a much shorter
                # critical path, so it is what ships.
                At = _transpose_pe(A, C, C)   # [C,C] = A^T (reused by both methods)
                T = _inverse_blocked_horner(A, At, eye, C)

                # value_new = T @ vbeta ; k_cumdecay = T @ (kbeta*exp(g_cum))
                # Transpose T once and reuse it as the (already-oriented) stationary
                # operand for both matmuls, instead of letting _matmul_AB transpose
                # T twice.
                Tt = _transpose_pe(T, C, C)   # [C,C] = T^T
                value_new_ps = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=value_new_ps, stationary=Tt, moving=vbeta)
                value_new = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=value_new, src=value_new_ps)     # [C,Dv]
                kbeta_eg = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=kbeta_eg, data=kbeta, op0=nl.multiply, operand0=eg_col)
                kcd_ps = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=kcd_ps, stationary=Tt, moving=kbeta_eg)
                k_cumdecay = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=k_cumdecay, src=kcd_ps)  # [C,Dk]

                # ================= inter-chunk contribution (uses S) =================
                # attn_intra = (Q@K^T) * decay_mask
                qk = _matmul_AB(Q, Kt, C, Dk, C)   # [C,C]
                attn_intra = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=attn_intra, data1=qk, data2=decay_mask, op=nl.multiply)

                # v_prime = k_cumdecay @ S  (contraction Dk, S is [Dk,Dv])
                v_prime = _matmul_AB(k_cumdecay, S, C, Dk, Dv)   # [C,Dv]
                v_new = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=v_new, data1=value_new, data2=v_prime, op=nl.subtract)

                # attn_inter = (Q*exp(g_cum)) @ S
                q_eg = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=q_eg, data=Q, op0=nl.multiply, operand0=eg_col)
                attn_inter = _matmul_AB(q_eg, S, C, Dk, Dv)   # [C,Dv]

                # core_out = attn_inter + attn_intra @ v_new
                intra_v = _matmul_AB(attn_intra, v_new, C, C, Dv)   # [C,Dv]
                core = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=core, data1=attn_inter, data2=intra_v, op=nl.add)
                core_bf = nl.ndarray((C, Dv), dtype=query.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=core_bf, src=core)
                nisa.dma_copy(dst=out[b, s0:s0 + C, h, 0:Dv], src=core_bf)

                # ================= state update =================
                # w[i] = exp(g_last - g_cum[i]) = exp(sum_{k>i} g_raw[k]) is the
                # exclusive suffix sum of g_raw. Compute it directly with the
                # strict-lower mask as a [C,C]x[C,1] matmul -- this avoids
                # isolating g_last as a partition-dim-1 [1,1] scalar and
                # broadcasting it to a [C,1] column (both odd-shaped tiles).
                w_ps = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=w_ps, stationary=strict_lo, moving=g_raw)  # [i,0]=sum_{k>i}
                w_col = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=w_col, src=w_ps)
                nisa.activation(dst=w_col, op=nl.exp, data=w_col)
                K_w = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=K_w, data=K, op0=nl.multiply, operand0=w_col)

                # exp(g_last) broadcast to [Dk,1]; g_last = g_cum[C-1] = sum of all
                # g_raw. Reduce over the C partitions with a ones moving operand
                # ([C,1]x[C,1] -> [1,1] at partition 0), then broadcast up.
                total_ps = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=total_ps, stationary=g_raw, moving=ones_C1)  # [0,0]=sum_k
                total_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=total_sb, src=total_ps)
                eglast_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=eglast_11, op=nl.exp, data=total_sb)
                eglast_dk = _bcast_col(eglast_11, Dk)       # [Dk,1] = exp(g_last)

                # S_upd[Dk,Dv] = K_w^T @ v_new  (contraction over positions C)
                supd_ps = nl.ndarray((Dk, Dv), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=supd_ps, stationary=K_w, moving=v_new)  # [Dk,Dv]
                S_scaled = nl.ndarray((Dk, Dv), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=S_scaled, data=S, op0=nl.multiply, operand0=eglast_dk)
                S_new = nl.ndarray((Dk, Dv), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=S_new, data1=S_scaled, data2=supd_ps, op=nl.add)
                nisa.tensor_copy(dst=S, src=S_new)

            nisa.dma_copy(dst=state_out[b, h, 0:Dk, 0:Dv], src=S)

    return out, state_out
