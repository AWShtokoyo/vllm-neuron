"""DRAFT NKI kernel: Chunked Gated Delta Rule (GDN) prefill.

Hand-written replacement for the torch `_chunk_gated_delta_rule`
(model.py:1219-1433). The torch path hangs / OOBs neuronx-cc at T>=1024 because
the seq -> (nchunk, chunk) reshape splits the PARTITION-mapped seq axis and
lowers to a vector-DGE indirect copy (nrta 1006). This kernel implements the
(N, C) chunk tiling with ONLY static-offset slices + nc_matmul, so there is no
indirect/DGE access on the partition axis.

STATUS: DRAFT skeleton for device iteration. Uses real NKI 0.4.0 APIs and
avoids forbidden indirect-indexing, but is NOT expected to compile/run
bit-exact on first try. See the "DEVICE-DEBUG HOTSPOTS" list at the bottom.

Host prologue (Steps 1-3, done outside this kernel by the wrapper in model.py):
  - l2norm(q), l2norm(k)  (eps=1e-6)
  - transpose to head-major fp32 [H, T, D]
  - pad seq to T_pad = N*C
  - query *= scale (1/sqrt(D))
  - k_beta = key * beta[...,None],  v_beta = value * beta[...,None]
This kernel takes the prepped tensors + precomputed [C,C] mask tiles.

Layout convention inside the kernel
------------------------------------
For chunk i, head h, we hold per-chunk tiles with the CONTRACTION axis on the
partition dim so nc_matmul (= stationary^T @ moving) lines up. The reference
chunk shapes are [C, D] and [C, C]. We keep C (<=128) and D (=128) both <=128,
so every per-chunk tile is partition-bounded.

nc_matmul reminder: nisa.nc_matmul(dst[M,N], stationary[K,M], moving[K,N])
computes stationary^T @ moving, K = contraction = partition axis of BOTH
operands. Every matmul below is annotated with its (stationary, moving)->dst
mapping and the torch line it implements.
"""
import os

import nki
import nki.isa as nisa
import nki.language as nl


def kernel_assert(condition, message=""):
    assert condition, message


# Hardware partition bound: chunk_size (C) is mapped to the SBUF partition axis
# (max 128). NKI-writing skill, "Hardware Constraints Quick Reference": Partition
# dimension (P) <= 128 for every SBUF/PSUM tile. C, D, Dv are all placed on the
# partition axis inside the kernel, so all three MUST honor <=128. When a caller
# requests a larger logical chunk we clamp to the largest divisor of the padded
# seq that is <=128, so the internal N-chunk loop (intra-chunk dense + inter-chunk
# recurrence) covers seq>=1024 with C always partition-legal.
_PARTITION_MAX = 128


def _legal_chunk_size(requested_c, t_pad):
    """Return a chunk_size that (a) is <= 128 (partition bound) and (b) divides t_pad.

    Mirrors the Nemotron-H SSD scan invariant that the partition-mapped dim
    (there head_dim, here chunk_size C) never exceeds 128. The kernel already
    iterates N = t_pad // C chunks carrying the recurrent state, so shrinking C
    only changes N (the bounded outer trip count), never the graph shape.
    """
    c = int(requested_c)
    if c < 1:
        c = 1
    if c > _PARTITION_MAX:
        c = _PARTITION_MAX
    # Ensure C divides t_pad so N = t_pad // C tiles the whole (already padded)
    # sequence with no ragged tail. Walk down from the clamped value to the
    # nearest divisor <= 128. t_pad is a power-of-two-ish padded length, so a
    # divisor <=128 always exists (worst case C=1).
    while c > 1 and (t_pad % c) != 0:
        c -= 1
    return c


def _transpose_pe(dst_sbuf, src_sbuf):
    """Transpose src_sbuf -> dst_sbuf via the PE/tensor engine (PSUM), NOT the vector engine.
    nc_transpose into an SBUF dst uses the VECTOR engine, which is limited to <=[32,32]
    (assertion 'Vector engine transpose requires shape <= [32, 32]'). Routing through a PSUM
    destination uses the PE engine (no 32x32 limit; PSUM free-dim <=512), then copy PSUM->SBUF.
    dst_sbuf shape = transpose of src_sbuf shape. Both fp32."""
    M, N = dst_sbuf.shape          # dst is [N_src_free, N_src_part] = transpose
    ps = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=ps, data=src_sbuf)   # PE-engine transpose into PSUM
    nisa.tensor_copy(dst=dst_sbuf, src=ps)     # PSUM -> SBUF


def gdn_chunk_prefill_kernel(
    q_hbm,        # [H, T_pad, D]  fp32, host-prepped (l2norm'd, *scale)
    k_hbm,        # [H, T_pad, D]  fp32, host-prepped (l2norm'd)
    v_hbm,        # [H, T_pad, Dv] fp32
    kbeta_hbm,    # [H, T_pad, D]  fp32  (= key * beta)
    vbeta_hbm,    # [H, T_pad, Dv] fp32  (= value * beta)
    g_hbm,        # [H, T_pad]     fp32  RAW log-decay (pre-cumsum), padding rows = 0
    identity_cc,  # [C, C] fp32  identity (eye)
    tril_incl,    # [C, C] fp32  (col <= row) lower-incl-diagonal float mask  (Step 6/10)
    strict_lower, # [C, C] fp32  (col <  row) strictly-lower float mask       (Step 7)
    # COMPILE-TIME INT CONSTANTS (passed by the host, NOT derived from .shape).
    # NKI traces tensor .shape dims as symbolic proxies, so Python int ops on them
    # (.bit_length(), range() trip counts) fail to resolve ("x::31.bit_length"). All
    # structural counts that drive control flow / unrolling MUST be real Python ints.
    #
    # These MUST be ordinary positional-or-keyword params (NO bare `*` keyword-only
    # separator). The NKI 0.4.0 (NKICC) tracer only constant-folds scalar params that
    # are bound as positional-or-keyword in the traced frame; params placed after `*`
    # are left as free symbolic names -> "unbound variable 'N'". This mirrors
    # functional/cumsum.py `def cumsum(x, axis=-1)` and nkilib rmsnorm_tkg(..., eps=1e-6,
    # hidden_actual=None, shard_on_h=False, ...), whose compile-time scalars fold because
    # they are plain params with no `*`.
    H=1,          # num heads (rank-local)
    N=1,          # num chunks = T_pad // C
    C=1,          # chunk_size
    D=1,          # key head dim
    Dv=1,         # value head dim
    n_double=1,   # ceil(log2 C) doubling iters for the triangular inverse
):
    T_pad = N * C
    kernel_assert(C <= 128, "chunk_size must be <= 128 (partition bound)")
    kernel_assert(D <= 128 and Dv <= 128, "head dims must be <= 128")

    # ---- outputs ----
    out_hbm = nl.ndarray((H, T_pad, Dv), dtype=nl.float32, buffer=nl.shared_hbm)
    state_hbm = nl.ndarray((H, D, Dv), dtype=nl.float32, buffer=nl.shared_hbm)

    # mask tiles are loaded once into SBUF (shared across heads/chunks)
    eye_sb = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    tril_sb = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    slow_sb = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    ones_1c = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
    # ones row of length D (state partition dim). The g_last state-decay column must
    # have D partitions (S_sb is [D,Dv]); the old [C,1] glast column was sliced [0:D],
    # which OOBs when D>C (e.g. C=64,D=128). Build the broadcast at size D directly.
    ones_1d = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eye_sb, src=identity_cc[0:C, 0:C])
    nisa.dma_copy(dst=tril_sb, src=tril_incl[0:C, 0:C])
    nisa.dma_copy(dst=slow_sb, src=strict_lower[0:C, 0:C])
    nisa.memset(dst=ones_1c, value=1.0)
    nisa.memset(dst=ones_1d, value=1.0)

    # ===== outer loop over heads: plain Python range -> fully unrolled =====
    for h in range(H):
        # cross-chunk recurrent state S[D, Dv], reset per head (Step 10 init).
        # ISSUE-7 fix: allocate the carry ONCE outside the chunk loop and update it
        # IN PLACE via tensor_copy at the loop tail. Re-allocating S_new every chunk
        # (the old `S_sb = S_new` rebind) cost N SBUF tiles and defeated the single
        # buffer the comment intended. sequential_range tracks the RAW dep through the
        # in-place writes on this one buffer.
        S_sb = nl.zeros((D, Dv), dtype=nl.float32, buffer=nl.sbuf)

        # ===== chunk loop: plain range(N) for FULL UNROLL (REQUIRED, not optional) =====
        # i MUST be a compile-time literal: nl.sequential_range does NOT unroll, so i stays a
        # runtime register and c0=i*C becomes a RUNTIME offset -> every q/k/v/g DMA and the out
        # store lower to a vector-DGE indirect copy -> nrta-1006 OOB at warmup (the long-standing
        # chunked-GDN bug). Plain range(N) fully unrolls -> c0 folds to a per-chunk constant ->
        # all DMAs are static-offset (no DGE). The loop-carried S_sb recurrent state is preserved
        # by data-flow through the in-place tensor_copy(dst=S_sb, ...) below, so correctness holds
        # without sequential_range. (N is small: seq1024/C128 = 8 unrolled bodies.)
        for i in range(N):
            c0 = i * C                  # i is a compile-time literal -> c0 constant -> static-offset DMA

            # ---- load chunk-i tiles [C, D] (seq on partition, contiguous) ----
            q_cd = nl.ndarray((C, D), dtype=nl.float32, buffer=nl.sbuf)
            k_cd = nl.ndarray((C, D), dtype=nl.float32, buffer=nl.sbuf)
            v_cd = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
            kb_cd = nl.ndarray((C, D), dtype=nl.float32, buffer=nl.sbuf)
            vb_cd = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
            g_1c = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_cd, src=q_hbm[h, c0:c0 + C, 0:D])
            nisa.dma_copy(dst=k_cd, src=k_hbm[h, c0:c0 + C, 0:D])
            nisa.dma_copy(dst=v_cd, src=v_hbm[h, c0:c0 + C, 0:Dv])
            nisa.dma_copy(dst=kb_cd, src=kbeta_hbm[h, c0:c0 + C, 0:D])
            nisa.dma_copy(dst=vb_cd, src=vbeta_hbm[h, c0:c0 + C, 0:Dv])
            # ISSUE-5 fix: index the g row with an explicit leading axis (h:h+1) so the
            # HBM view is already rank-2 [1, C]. No .reshape on a sliced 1-D HBM proxy
            # (the old g_hbm[h, c0:c0+C].reshape(1, C) could fail tracing on a
            # non-contiguous slice proxy). Partition dim is now explicit (1).
            nisa.dma_copy(dst=g_1c, src=g_hbm[h:h + 1, c0:c0 + C])

            # ================= Step 5: intra-chunk decay cumsum =================
            # g_cum[1,C] = cumsum(g) over the chunk. 1 partition, C free, no carry.
            g_cum_1c = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(
                dst=g_cum_1c, data0=ones_1c, data1=g_1c,
                initial=0.0, op0=nl.multiply, op1=nl.add,
            )
            # we need g_cum as a [C,1] column (partition = position) for the
            # per-row decay weighting in Steps 6/9/10. Transpose the 1xC row.
            g_col = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
            _transpose_pe(g_col, g_cum_1c)  # PE-engine (was nc_transpose, vector 32x32 limit)        # [1,C] -> [C,1]
            # g_last = total chunk decay = g_cum[C-1]  (positive index, NOT -1)
            g_last = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=g_last, src=g_cum_1c[0:1, C - 1:C])

            # ISSUE-1/ISSUE-3 fix: the recurrent-state decay scale is exp(g_last), not
            # the raw log g_last, and it must be available as a per-PARTITION column so
            # tensor_scalar can broadcast it over the [D,Dv] state (tensor_scalar does
            # NOT broadcast a [1,1] tile across the partition dim). Build g_last as a
            # [C,1] column (one value per partition row, all equal to g_last) by
            # broadcasting g_cum's last column over partitions via the ones-matmul trick,
            # then exp it. The STATE scale uses a [D,1] column (S_sb has D partitions),
            # so build it at length D directly (old [C,1] sliced [0:D] OOBs when D>C).
            # We also keep a [C,1] glast_col for the [C,C] decay-diff weighting below.
            g_last_1c = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=g_last_1c, data=ones_1c, op0=nl.multiply,
                               operand0=g_last)                # [1,C] row of constant g_last
            glast_col = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
            _transpose_pe(glast_col, g_last_1c)  # PE-engine   # [1,C] -> [C,1]
            # [D,1] state-decay column = exp(g_last), one row per state partition.
            g_last_1d = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=g_last_1d, data=ones_1d, op0=nl.multiply,
                               operand0=g_last)                # [1,D] row of constant g_last
            glast_exp_col = nl.ndarray((D, 1), dtype=nl.float32, buffer=nl.sbuf)
            _transpose_pe(glast_exp_col, g_last_1d)  # PE-engine  # [1,D] -> [D,1]
            nisa.activation(dst=glast_exp_col, op=nl.exp, data=glast_exp_col)  # exp(g_last)

            # exp(g_cum) per position, as a [C,1] column (Step 9/10 weighting)
            gexp_col = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=gexp_col, op=nl.exp, data=g_col)
            # also as a [C,1] for "g_last - g_cum" weighting in the state update.
            # ISSUE-2 fix: the reference weight is exp(g_last - g_cum). tensor_scalar
            # (data=g_col, op0=subtract, operand0=glast_col) computes g_cum - g_last,
            # the WRONG sign (reciprocal). Use op1=multiply by -1.0 in the same
            # tensor_scalar to negate -> (g_cum - g_last) * -1 = g_last - g_cum, then exp.
            # glast_col (the broadcasted [C,1] column) is used as operand0 so the
            # per-partition subtraction is well-defined (no [1,1] partition-broadcast).
            gdiff_col = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=gdiff_col, data=g_col,
                               op0=nl.subtract, operand0=glast_col,
                               op1=nl.multiply, operand1=-1.0)  # g_last - g_cum
            nisa.activation(dst=gdiff_col, op=nl.exp, data=gdiff_col)  # exp(g_last - g_cum)

            # =============== Step 6: pairwise within-chunk decay mask ===============
            # decay[i,j] = exp(g_i - g_j) * (j<=i).  Build diff[C,C] = g_col - g_row.
            # g_row broadcast: replicate g_cum across partition via matmul-with-ones
            # or tensor_tensor with a broadcasted [C,C]. Here: diff = g_col(bcast cols)
            #   - g_cum(bcast rows). We form it with two tensor_scalar/tensor_tensor
            #   steps against broadcast tiles.
            diff_cc = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
            # ISSUE-4 (minor) fix: memset the [C,C] scratch to 0 directly instead of
            # multiplying eye by 0.0 (cheaper, clearer; fp32 dst + float literal value).
            nisa.memset(dst=diff_cc, value=0.0)                # diff = 0 (zeroed)
            # diff += g_col (per-row constant, broadcast over columns)
            nisa.tensor_scalar(dst=diff_cc, data=diff_cc, op0=nl.add,
                               operand0=g_col)                 # + g_i  (row)
            # diff -= g_row (per-column constant). g_row = g_cum broadcast over rows;
            #   subtract a [1,C] tile broadcast over partitions via tensor_tensor.
            g_row_cc = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
            # broadcast g_cum_1c[1,C] up to [C,C]: matmul ones[1,C]^T? Simpler:
            #   replicate by nc_matmul(stationary=ones_c1[1,C], moving=g_cum_1c[1,C])
            #   -> [C,C] each row = g_cum. (ones_c1 is [1,C] of ones.)
            psum_grow = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=psum_grow, stationary=ones_1c, moving=g_cum_1c)
            nisa.tensor_copy(dst=g_row_cc, src=psum_grow)      # rows = g_cum (broadcast)
            nisa.tensor_tensor(dst=diff_cc, data1=diff_cc, data2=g_row_cc,
                               op=nl.subtract)                 # diff = g_i - g_j
            # NAN-FIX (Step 6, was the FIRST non-finite stage): the reference masks the
            # decay diff to the lower triangle BEFORE the exp:
            #     decay_mask = ((g_i - g_j).tril().exp()).tril()   (model.py:1534)
            # In the STRICT-UPPER triangle (j>i) g is a negative cumsum, so
            # diff[i,j] = g_cum_i - g_cum_j >= 0 and grows with |i-j|; over a chunk it
            # easily exceeds the fp32 exp overflow threshold (~88), so exp(diff) -> +inf.
            # The OLD code did exp() on the FULL matrix and only masked AFTERWARD, so the
            # subsequent `decay * tril` computed inf * 0 = NaN (IEEE-754), which then
            # poisoned M, the (I-M)^-1 doubling, u/w, and every downstream matmul -> the
            # observed tgt=nan / BC=0 at all seq/chunk. Mirror the reference exactly:
            # zero the strict-upper triangle of diff FIRST (so exp only sees values <= 0,
            # bounded by exp(0)=1 on the diagonal), THEN exp, THEN re-mask. Masking via a
            # tensor_tensor multiply by tril_sb (col<=row) is the elementwise equivalent
            # of torch.tril and introduces no indexed/DGE op. (NKI-writing skill: use
            # nisa.activation(op=nl.exp) for elementwise exp with dst overwrite; the input
            # to exp is now range-bounded so it cannot produce inf for finite g.)
            decay_cc = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=diff_cc, data1=diff_cc, data2=tril_sb,
                               op=nl.multiply)                 # zero STRICT-UPPER (pre-exp)
            nisa.activation(dst=decay_cc, op=nl.exp, data=diff_cc)  # exp only sees diff <= 0
            nisa.tensor_tensor(dst=decay_cc, data1=decay_cc, data2=tril_sb,
                               op=nl.multiply)                 # re-zero upper triangle (exp(0)=1)

            # =============== Step 7: M = -(k_beta @ key^T) * decay * strict_lower ===============
            # k_beta @ key^T : [C,D]@[D,C] -> [C,C].  Contraction = D.
            #   nc_matmul(stationary[D,C]=kb^T, moving[D,C]=k^T) -> [C,C] = kb @ k^T.
            # We need kb and k with D on partition. Transpose [C,D]->[D,C].
            kb_dc = nl.ndarray((D, C), dtype=nl.float32, buffer=nl.sbuf)
            k_dc = nl.ndarray((D, C), dtype=nl.float32, buffer=nl.sbuf)
            _transpose_pe(kb_dc, kb_cd)  # PE-engine (was nc_transpose, vector 32x32 limit)
            _transpose_pe(k_dc, k_cd)  # PE-engine (was nc_transpose, vector 32x32 limit)
            psum_gram = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=psum_gram, stationary=kb_dc, moving=k_dc)  # kb @ k^T
            M_cc = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=M_cc, src=psum_gram)
            nisa.tensor_tensor(dst=M_cc, data1=M_cc, data2=decay_cc, op=nl.multiply)
            nisa.tensor_scalar(dst=M_cc, data=M_cc, op0=nl.multiply, operand0=-1.0)
            nisa.tensor_tensor(dst=M_cc, data1=M_cc, data2=slow_sb, op=nl.multiply)

            # =============== Step 8: (I - M)^-1 via nilpotent doubling ===============
            #   inv = I; m_pow = M
            #   repeat n_double:  inv = inv + m_pow @ inv ;  m_pow = m_pow @ m_pow
            inv_cc = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
            mpow_cc = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=inv_cc, src=eye_sb)
            nisa.tensor_copy(dst=mpow_cc, src=M_cc)
            for _d in range(n_double):                         # static unroll
                # term = m_pow @ inv : contraction over the SHARED inner C.
                #   nc_matmul(stationary[K,M], moving[K,N]) = stat^T @ mov.
                #   want m_pow @ inv: stationary = m_pow^T, moving = inv.
                mpow_t = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
                _transpose_pe(mpow_t, mpow_cc)  # PE-engine (was nc_transpose, vector 32x32 limit)
                p_term = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p_term, stationary=mpow_t, moving=inv_cc)
                term_cc = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=term_cc, src=p_term)
                nisa.tensor_tensor(dst=inv_cc, data1=inv_cc, data2=term_cc, op=nl.add)
                # m_pow = m_pow @ m_pow
                p_sq = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p_sq, stationary=mpow_t, moving=mpow_cc)
                nisa.tensor_copy(dst=mpow_cc, src=p_sq)
            # inv_cc = (I - M)^-1 = attn

            # =============== Step 9: u = attn @ v_beta ; w = attn @ (k_beta*exp(g)) ===============
            # u[C,Dv] = attn[C,C] @ v_beta[C,Dv]. contraction = inner C.
            #   stationary = attn^T[C,C], moving = v_beta[C,Dv] -> [C,Dv].
            attn_t = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
            _transpose_pe(attn_t, inv_cc)  # PE-engine (was nc_transpose, vector 32x32 limit)
            p_u = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=p_u, stationary=attn_t, moving=vb_cd)
            u_cd = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=u_cd, src=p_u)
            # w = attn @ (k_beta * exp(g_cum)[:,None]). weight kb_cd by gexp_col first.
            kbg_cd = nl.ndarray((C, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=kbg_cd, data=kb_cd, op0=nl.multiply,
                               operand0=gexp_col)              # per-row * exp(g_i)
            p_w = nl.ndarray((C, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=p_w, stationary=attn_t, moving=kbg_cd)
            w_cd = nl.ndarray((C, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=w_cd, src=p_w)

            # =============== Step 10: per-chunk inter+intra output, then state update ===============
            # A_i = (q_i @ k_i^T) * decay * tril_incl(strict-lower-INCL, j<=i)
            #   q @ k^T: [C,D]@[D,C]->[C,C]; stationary = q^T[D,C], moving = k^T[D,C].
            q_dc = nl.ndarray((D, C), dtype=nl.float32, buffer=nl.sbuf)
            _transpose_pe(q_dc, q_cd)  # PE-engine (was nc_transpose, vector 32x32 limit)
            p_qk = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=p_qk, stationary=q_dc, moving=k_dc)      # q @ k^T
            A_cc = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=A_cc, src=p_qk)
            nisa.tensor_tensor(dst=A_cc, data1=A_cc, data2=decay_cc, op=nl.multiply)
            nisa.tensor_tensor(dst=A_cc, data1=A_cc, data2=tril_sb, op=nl.multiply)

            # v_prime = w_i @ S : [C,D]@[D,Dv] -> [C,Dv]; stationary = w^T[D,C], moving = S[D,Dv]
            w_dc = nl.ndarray((D, C), dtype=nl.float32, buffer=nl.sbuf)
            _transpose_pe(w_dc, w_cd)  # PE-engine (was nc_transpose, vector 32x32 limit)
            p_vp = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=p_vp, stationary=w_dc, moving=S_sb)
            v_new = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v_new, src=p_vp)
            nisa.tensor_tensor(dst=v_new, data1=u_cd, data2=v_new, op=nl.subtract)  # u - v_prime

            # attn_inter = (q_i * exp(g_cum)) @ S : [C,D]@[D,Dv]->[C,Dv]
            qg_cd = nl.ndarray((C, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=qg_cd, data=q_cd, op0=nl.multiply, operand0=gexp_col)
            qg_dc = nl.ndarray((D, C), dtype=nl.float32, buffer=nl.sbuf)
            _transpose_pe(qg_dc, qg_cd)  # PE-engine (was nc_transpose, vector 32x32 limit)
            p_inter = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=p_inter, stationary=qg_dc, moving=S_sb)
            out_cd = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_cd, src=p_inter)

            # out_i = attn_inter + A_i @ v_new : A[C,C]@v_new[C,Dv]->[C,Dv]
            A_t = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
            _transpose_pe(A_t, A_cc)  # PE-engine (was nc_transpose, vector 32x32 limit)
            p_intra = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=p_intra, stationary=A_t, moving=v_new)
            intra_cd = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=intra_cd, src=p_intra)
            nisa.tensor_tensor(dst=out_cd, data1=out_cd, data2=intra_cd, op=nl.add)

            # write chunk output to its static HBM slice (NO merge reshape)
            nisa.dma_copy(dst=out_hbm[h, c0:c0 + C, 0:Dv], src=out_cd)

            # ---- state update (Step 10 tail) ----
            # S = S * exp(g_last)
            #     + (k_i * exp(g_last - g_cum))^T @ v_new
            # ISSUE-1/ISSUE-3 fix: scale by exp(g_last) (NOT the raw log g_last), and use
            # the broadcasted [C,1] column glast_exp_col sliced to [D,1] so tensor_scalar's
            # per-partition operand matches S_sb's D partitions (a [1,1] g_last would NOT
            # broadcast across the partition dim). Write into a fresh S_new then copy back
            # into the persistent S_sb buffer (ISSUE-7: in-place carry).
            S_new = nl.ndarray((D, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=S_new, data=S_sb, op0=nl.multiply,
                               operand0=glast_exp_col)             # S * exp(g_last)  ([D,1] column)
            # k_decay = k_i * exp(g_last - g_cum)  (per-row weight gdiff_col)
            k_decay_cd = nl.ndarray((C, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=k_decay_cd, data=k_cd, op0=nl.multiply,
                               operand0=gdiff_col)
            # (k_decay)^T @ v_new : [D,C]@... no — we want k_decay^T @ v_new = [D,Dv].
            #   k_decay[C,D], v_new[C,Dv]; contraction over C (rows).
            #   stationary = k_decay[C,D], moving = v_new[C,Dv] -> [D,Dv].  (already
            #   contraction-on-partition since C is on partition for both.)
            p_supd = nl.ndarray((D, Dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=p_supd, stationary=k_decay_cd, moving=v_new)
            supd_dd = nl.ndarray((D, Dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=supd_dd, src=p_supd)
            nisa.tensor_tensor(dst=S_new, data1=S_new, data2=supd_dd, op=nl.add)
            # carry forward (ISSUE-7: write back into the single persistent buffer
            # in place, rather than rebinding the Python name to a new tile each chunk)
            nisa.tensor_copy(dst=S_sb, src=S_new)

        # final per-head recurrent state -> HBM
        nisa.dma_copy(dst=state_hbm[h, 0:D, 0:Dv], src=S_sb)

    return out_hbm, state_hbm


# --- Torch-invocable entry: mirror functional/cumsum.py:13,60-61 EXACTLY ---
# CRITICAL: wrap_nki(...) must be built at MODULE LOAD (not inside the call) so
# register_kernel_to_torch's @torch._dynamo.assume_constant_result folds the kernel index at
# trace time — calling wrap_nki INSIDE the forward defeats that and hits the fake-tensor wall
# (NKI-790: NKI V3 doesn't support FakeTensors). Pre-wrap here; the forward just calls _gdn_wrapped[2].
_gdn_jit = nki.jit()(gdn_chunk_prefill_kernel)
# Wrap EAGERLY at module load (NOT lazily inside run_gdn_chunk_prefill), mirroring
# gated_delta_rule_seq.py:159-169. The lazy `if _gdn_wrapped is None: _gdn_wrapped = wrap_nki(...)`
# branch inside the traced forward installs a Dynamo GUARD on this global AND mutates it mid-trace:
# warmup bakes the is-None branch, then the first real call flips the guard and Dynamo attempts a
# recompile, which vLLM rejects under torch.compile stance 'fail_on_recompile' (RuntimeError /
# EngineDeadError at warmup — the observed chunk-GDN full-model failure). Module-level wrapping keeps
# the forward branch-free (fail_on_recompile-safe). This module is only imported on the chunk-GDN
# forward path (VLLM_GDN_NKI / FORCE_CHUNKED), so the heavy/device wrap_nki import never runs on the
# default path — same rule the seq kernel follows.
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
_gdn_wrapped = wrap_nki(_gdn_jit)


def run_gdn_chunk_prefill(q, k, v, kb, vb, g, eye, tril_incl, strict_lower):
    """Invoke the NKI GDN chunked-prefill kernel through the repo's wrap_nki (LNC=2 variant).
    Mirrors cumsum.py + gated_delta_rule_seq.py: wrap once at MODULE LOAD, call wrapped[2](...) so the
    kernel folds at trace and the forward stays branch-free (no fail_on_recompile break).
    Structural counts (H,N,C,D,Dv,n_double) are computed HERE as real Python ints from the host-side
    tensor shapes and passed as compile-time constants — the kernel must NOT derive them from .shape."""
    H, T_pad, D = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    Dv = int(v.shape[-1])
    # C is the partition-mapped chunk size (NKI-writing skill: Partition dim <= 128).
    # The mask tiles (eye/tril/strict_lower) are built [C_req, C_req] by the caller,
    # so eye.shape[0] carries the REQUESTED chunk size which may exceed 128 (this is
    # exactly the observed failure: a caller passing C=256=seq drove a partition-dim OOB at
    # kernel_assert). Clamp to a partition-legal divisor of T_pad here so the kernel's
    # internal N-chunk loop always sees C<=128 regardless of the caller's request.
    C_req = int(eye.shape[0])
    C = _legal_chunk_size(C_req, T_pad)
    kernel_assert(
        C <= _PARTITION_MAX,
        f"chunk_size {C} must be <= {_PARTITION_MAX} (partition bound)",
    )
    kernel_assert(T_pad % C == 0, f"T_pad {T_pad} must be divisible by chunk_size {C}")
    # The precomputed mask tiles are [C_req, C_req]; if we shrank C below C_req they
    # are the wrong size for the kernel's [C, C] slices. Rebuild them at the legal C.
    if C != C_req:
        import torch
        dev = eye.device
        eye = torch.eye(C, dtype=torch.float32, device=dev)
        tril_incl = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev))
        strict_lower = torch.tril(
            torch.ones(C, C, dtype=torch.float32, device=dev), diagonal=-1
        )
    N = T_pad // C
    n_double = max(1, (C - 1).bit_length())   # ceil(log2 C), real Python int
    return _gdn_wrapped[2](
        q_hbm=q, k_hbm=k, v_hbm=v, kbeta_hbm=kb, vbeta_hbm=vb, g_hbm=g,
        identity_cc=eye, tril_incl=tril_incl, strict_lower=strict_lower,
        H=H, N=N, C=C, D=D, Dv=Dv, n_double=n_double,
    )
