"""Phase 5a-ii, stage 2: the full MLA inner block as a NKI kernel.

Computes one (query tile x key segment) step of `_mla_attend_tiled`:

    scores = (q_lift @ c_kv^T + q_pe @ k_pe^T) * softmax_scale     [Sq, Sk]
    scores = mask(scores)                                          causal / validity
    m_new  = max(m_in, rowmax(scores))
    p      = exp(scores - m_new)
    alpha  = exp(m_in - m_new)
    l_out  = l_in * alpha + rowsum(p)
    acc_out= acc_in * alpha + p @ c_kv                             [Sq, L]

Returns `(m_out, l_out, acc_out)` so the caller can chain segments. Online softmax
is exact, so chaining N segments must equal one softmax over the concatenation —
that is what the validation asserts, not a loose tolerance.

Stage 1 (`glm52_mla_scores.py`) validated the score matmul alone at rel_fro 1.1e-07;
this stage adds the softmax and the second matmul on top of that verified base.

🔴 Partition axis: Sq on partitions, heads looped by the caller. See the header of
glm52_mla_scores.py for why the shipped DeepSeek MLA kernels (which put H on
partitions and assert H % 16 == 0, qk_nope_head_dim == 128, per-head d_v == 128)
cannot be used for GLM-5.2 at TP=64 (1 head/rank, 192, 256).

🔴 Two hardware caps drive the tiling, both measured on this target (gen3):
  * nc_matmul MOVING free dim <= 512  -> key chunks of 512
  * PSUM free dim <= 512 (fp32)       -> same

Mask contract: the caller passes `mask_add`, a [Sq, Sk] fp32 tensor that is 0 where
a key is visible and a large negative value where it is not. Passing it as an
additive tensor rather than recomputing positions inside the kernel keeps the kernel
free of position arithmetic and lets the caller reuse the "prior segments need no
mask" shortcut from the torch path.
"""

import nki
import nki.isa as nisa
import nki.language as nl

_PMAX = nl.tile_size.pmax   # 128
_SK_CHUNK = 512             # matmul moving free-dim / PSUM free-dim cap on gen3
_NEG = -30000.0             # additive mask value; exp() of this underflows to 0


def _kernel_assert(cond, msg):
    if not cond:
        raise AssertionError(f"[GLM52 MLA block] {msg}")


@nki.jit
def glm52_mla_block_kernel(
    q_lift_t, c_kv_t, q_pe_t, k_pe_t, mask_add, m_in, l_in, acc_in, softmax_scale
):
    """One online-softmax step over a key segment.

    Args:
        q_lift_t: [L, Sq]  bf16, latent-major (transposed).
        c_kv_t:   [L, Sk]  bf16, latent-major (transposed).
        q_pe_t:   [R, Sq]  bf16, rope-major.
        k_pe_t:   [R, Sk]  bf16, rope-major.
        mask_add: [Sq, Sk] fp32, 0 = visible, large negative = masked.
        m_in:     [Sq, 1]  fp32 running max   (use -30000.0 for "empty").
        l_in:     [Sq, 1]  fp32 running sum.
        acc_in:   [Sq, L]  fp32 running output latent.
        softmax_scale: python float.

    Returns:
        (m_out [Sq,1] fp32, l_out [Sq,1] fp32, acc_out [Sq,L] fp32)

    Notes:
        `m_in = -30000.0` rather than -inf: an all-masked block would make
        `exp(m_in - m_new)` produce NaN with true infinities, and the torch path
        needs an explicit isneginf branch for the same reason. A large finite
        sentinel makes both `exp(-30000 - m)` and `exp(s - m)` underflow cleanly
        to 0 instead.
    """
    L, Sq = q_lift_t.shape
    Sk = c_kv_t.shape[1]
    R = q_pe_t.shape[0]

    _kernel_assert(Sq <= _PMAX, f"Sq must be <= {_PMAX}, got {Sq}")
    _kernel_assert(L % _PMAX == 0, f"L must be a multiple of {_PMAX}, got {L}")
    _kernel_assert(R <= _PMAX, f"R must be <= {_PMAX}, got {R}")
    _kernel_assert(c_kv_t.shape[0] == L, "c_kv_t partition dim must be L")
    _kernel_assert(k_pe_t.shape[1] == Sk, "k_pe_t free dim must be Sk")
    _kernel_assert(mask_add.shape == (Sq, Sk), f"mask_add must be [{Sq},{Sk}]")
    _kernel_assert(acc_in.shape == (Sq, L), f"acc_in must be [{Sq},{L}]")

    n_l = L // _PMAX
    n_k = (Sk + _SK_CHUNK - 1) // _SK_CHUNK

    m_out = nl.ndarray((Sq, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    l_out = nl.ndarray((Sq, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    acc_out = nl.ndarray((Sq, L), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- query-side operands and running state, loaded once ------------------
    q_sb = nl.ndarray((_PMAX, n_l, Sq), dtype=q_lift_t.dtype, buffer=nl.sbuf)
    for li in nl.affine_range(n_l):
        nisa.dma_copy(dst=q_sb[0:_PMAX, li, 0:Sq],
                      src=q_lift_t[li * _PMAX:(li + 1) * _PMAX, 0:Sq])
    qpe_sb = nl.ndarray((R, Sq), dtype=q_pe_t.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=qpe_sb, src=q_pe_t[0:R, 0:Sq])

    m_sb = nl.ndarray((Sq, 1), dtype=nl.float32, buffer=nl.sbuf)
    l_sb = nl.ndarray((Sq, 1), dtype=nl.float32, buffer=nl.sbuf)
    acc_sb = nl.ndarray((Sq, L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=m_sb, src=m_in[0:Sq, 0:1])
    nisa.dma_copy(dst=l_sb, src=l_in[0:Sq, 0:1])
    nisa.dma_copy(dst=acc_sb, src=acc_in[0:Sq, 0:L])

    # Key chunks carry a loop-carried dependency through (m, l, acc), so this
    # must be sequential_range — affine_range would let chunks reorder.
    for kc in nl.sequential_range(n_k):
        k0 = kc * _SK_CHUNK
        k_sz = min(_SK_CHUNK, Sk - k0)

        c_sb = nl.ndarray((_PMAX, n_l, k_sz), dtype=c_kv_t.dtype, buffer=nl.sbuf)
        for li in nl.affine_range(n_l):
            nisa.dma_copy(dst=c_sb[0:_PMAX, li, 0:k_sz],
                          src=c_kv_t[li * _PMAX:(li + 1) * _PMAX, k0:k0 + k_sz])
        kpe_sb = nl.ndarray((R, k_sz), dtype=k_pe_t.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=kpe_sb, src=k_pe_t[0:R, k0:k0 + k_sz])

        # --- scores ----------------------------------------------------------
        s_psum = nl.ndarray((Sq, k_sz), dtype=nl.float32, buffer=nl.psum)
        for li in nl.affine_range(n_l):
            nisa.nc_matmul(
                s_psum[0:Sq, 0:k_sz],
                q_sb[0:_PMAX, li, 0:Sq],
                c_sb[0:_PMAX, li, 0:k_sz],
                accumulate=(li > 0),
            )
        nisa.nc_matmul(
            s_psum[0:Sq, 0:k_sz], qpe_sb[0:R, 0:Sq], kpe_sb[0:R, 0:k_sz],
            accumulate=True,
        )

        # scale, then add the mask (0 / large negative)
        s_sb = nl.ndarray((Sq, k_sz), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=s_sb, data=s_psum[0:Sq, 0:k_sz],
                           op0=nl.multiply, operand0=softmax_scale)
        mk_sb = nl.ndarray((Sq, k_sz), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=mk_sb, src=mask_add[0:Sq, k0:k0 + k_sz])
        nisa.tensor_tensor(dst=s_sb, data1=s_sb, data2=mk_sb, op=nl.add)

        # --- online softmax update -------------------------------------------
        chunk_max = nl.ndarray((Sq, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=chunk_max, data=s_sb, op=nl.maximum, axis=(1,))
        m_new = nl.ndarray((Sq, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=m_new, data1=m_sb, data2=chunk_max, op=nl.maximum)

        # p = exp(s - m_new): activation's bias is added BEFORE the op, so pass
        # -m_new as the bias and let ScalarE fuse the subtract into the exp.
        neg_m = nl.ndarray((Sq, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_m, data=m_new, op0=nl.multiply, operand0=-1.0)
        p_sb = nl.ndarray((Sq, k_sz), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=p_sb, data=s_sb, op=nl.exp, bias=neg_m)

        # alpha = exp(m_in - m_new)
        d_sb = nl.ndarray((Sq, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=d_sb, data1=m_sb, data2=m_new, op=nl.subtract)
        alpha = nl.ndarray((Sq, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=alpha, data=d_sb, op=nl.exp)

        # l = l * alpha + rowsum(p)
        p_sum = nl.ndarray((Sq, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=p_sum, data=p_sb, op=nl.add, axis=(1,))
        nisa.tensor_tensor(dst=l_sb, data1=l_sb, data2=alpha, op=nl.multiply)
        nisa.tensor_tensor(dst=l_sb, data1=l_sb, data2=p_sum, op=nl.add)

        # --- acc = acc * alpha + p @ c_kv ------------------------------------
        # p @ c_kv contracts over the KEY axis, so both operands need keys on
        # partitions: p^T [k_sz, Sq] and c_kv [k_sz, L]. p is [Sq, k_sz] and
        # c_sb is latent-major, so transpose both on-chip.
        pt = nl.ndarray((k_sz, Sq), dtype=q_lift_t.dtype, buffer=nl.sbuf)
        p_bf = nl.ndarray((Sq, k_sz), dtype=q_lift_t.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=p_bf, src=p_sb)
        for kt in nl.affine_range((k_sz + _PMAX - 1) // _PMAX):
            t0 = kt * _PMAX
            t_sz = min(_PMAX, k_sz - t0)
            tp = nl.ndarray((t_sz, Sq), dtype=q_lift_t.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=p_bf[0:Sq, t0:t0 + t_sz])
            nisa.tensor_copy(dst=pt[t0:t0 + t_sz, 0:Sq], src=tp)

        # scale the running accumulator by alpha, then add this chunk's product
        nisa.tensor_tensor(dst=acc_sb, data1=acc_sb, data2=alpha, op=nl.multiply)
        for li in nl.affine_range(n_l):
            o_psum = nl.ndarray((Sq, _PMAX), dtype=nl.float32, buffer=nl.psum)
            for kt in nl.affine_range((k_sz + _PMAX - 1) // _PMAX):
                t0 = kt * _PMAX
                t_sz = min(_PMAX, k_sz - t0)
                # c_sb[:, li, :] is [L_tile=128 partitions, k_sz]; we need
                # [k_sz partitions, L_tile] -> transpose the needed slice.
                ct = nl.ndarray((t_sz, _PMAX), dtype=q_lift_t.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=ct, data=c_sb[0:_PMAX, li, t0:t0 + t_sz])
                ct_sb = nl.ndarray((t_sz, _PMAX), dtype=q_lift_t.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ct_sb, src=ct)
                nisa.nc_matmul(
                    o_psum[0:Sq, 0:_PMAX],
                    pt[t0:t0 + t_sz, 0:Sq],      # stationary [K=t_sz, M=Sq]
                    ct_sb[0:t_sz, 0:_PMAX],      # moving     [K=t_sz, N=128]
                    accumulate=(kt > 0),
                )
            o_sb = nl.ndarray((Sq, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o_sb, src=o_psum[0:Sq, 0:_PMAX])
            nisa.tensor_tensor(
                dst=acc_sb[0:Sq, li * _PMAX:(li + 1) * _PMAX],
                data1=acc_sb[0:Sq, li * _PMAX:(li + 1) * _PMAX],
                data2=o_sb,
                op=nl.add,
            )

        nisa.tensor_copy(dst=m_sb, src=m_new)

    nisa.dma_copy(dst=m_out, src=m_sb)
    nisa.dma_copy(dst=l_out, src=l_sb)
    nisa.dma_copy(dst=acc_out, src=acc_sb)
    return m_out, l_out, acc_out
