"""0-based (compact) GatedDeltaNet decode RECURRENT-state update NKI kernel.

"Stage outside, kernel 0-based" variant of gdn_state_update.py (APC_REFACTOR_PLAN §9 host/in-graph
split; user-approved 2026-07-08). The SLOT gather/scatter is done OUTSIDE this kernel by the .ap
page-stride primitive (paged_kv_gather / paged_kv_scatter), so this kernel operates purely on a
COMPACT [B, H, kd, vd] per-active-sequence state tensor: read row (b,h), one gated-delta step,
write row (b,h). No indirect DMA, no N_SLOTS baseline copy, no slot addressing.

Recurrence MATH is VERBATIM from gdn_state_update.py steps 2-6 (S*g, k^T S, delta, rank-1 update,
q^T S). Input state [B,H,kd,vd] (row (b,h) = seq b's seed state, gathered outside). Output
[B,H,kd,vd] (final state, scattered back outside). Unconditionally contiguous -> lowers regardless
of the shared-slab layout (which lives only in the outside gather/scatter).
"""
import nki
import nki.isa as nisa
import nki.language as nl


def kernel_assert(condition, message=""):
    assert condition, message


@nki.jit
def gdn_state_update_compact(
    rec_state_hbm,   # [B, H, kd, vd] fp32 COMPACT seed state (row (b,h))
    q_hbm,           # [B, H, kd]  fp32 (l2norm'd, *scale)
    k_hbm,           # [B, H, kd]  fp32 (l2norm'd)
    v_hbm,           # [B, H, vd]  fp32
    g_hbm,           # [B, H]      fp32 decay = exp(g_t)
    beta_hbm,        # [B, H]      fp32 = sigmoid(b)
    B=1, H=1, kd=128, vd=128,
):
    """Compact decode recurrence. Returns y_hbm [B,H,vd], rec_out_hbm [B,H,kd,vd] (both compact)."""
    kernel_assert(B <= 128 and H <= 128, "B,H must be <= 128")
    kernel_assert(kd <= 128 and vd <= 512, "kd<=128 (partition), vd<=512 (psum free)")

    y_hbm = nl.ndarray((B, H, vd), dtype=nl.float32, buffer=nl.shared_hbm)
    rec_out_hbm = nl.ndarray((B, H, kd, vd), dtype=nl.float32, buffer=nl.shared_hbm)
    rs = rec_state_hbm.reshape((B * H, kd, vd))
    ro = rec_out_hbm.reshape((B * H, kd, vd))

    for b in range(B):
        for h in range(H):
            row = b * H + h
            qk_off = row * kd
            v_off = row * vd
            # GATHER S[b,h] -> [kd, vd] (contiguous slice, NO indirect DMA).
            S = nl.ndarray((kd, vd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=S, src=rs[row])

            k_kd1 = nl.ndarray((kd, 1), dtype=nl.float32, buffer=nl.sbuf)
            k_1kd = nl.ndarray((1, kd), dtype=nl.float32, buffer=nl.sbuf)
            q_kd1 = nl.ndarray((kd, 1), dtype=nl.float32, buffer=nl.sbuf)
            v_1vd = nl.ndarray((1, vd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_kd1, src=k_hbm.ap(pattern=[[1, kd], [1, 1]], offset=qk_off))
            nisa.dma_copy(dst=k_1kd, src=k_hbm.ap(pattern=[[kd, 1], [1, kd]], offset=qk_off))
            nisa.dma_copy(dst=q_kd1, src=q_hbm.ap(pattern=[[1, kd], [1, 1]], offset=qk_off))
            nisa.dma_copy(dst=v_1vd, src=v_hbm.ap(pattern=[[vd, 1], [1, vd]], offset=v_off))
            g_kd = nl.ndarray((kd, 1), dtype=nl.float32, buffer=nl.sbuf)
            beta_s = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=g_kd, src=g_hbm.ap(pattern=[[0, kd], [1, 1]], offset=row))
            nisa.dma_copy(dst=beta_s, src=beta_hbm.ap(pattern=[[1, 1], [1, 1]], offset=row))

            nisa.tensor_scalar(dst=S, data=S, op0=nl.multiply, operand0=g_kd)
            kv_ps = nl.ndarray((1, vd), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=kv_ps, stationary=k_kd1, moving=S)
            kv = nl.ndarray((1, vd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=kv, src=kv_ps)
            delta = nl.ndarray((1, vd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=delta, data1=v_1vd, data2=kv, op=nl.subtract)
            nisa.tensor_scalar(dst=delta, data=delta, op0=nl.multiply, operand0=beta_s[0:1, 0:1])
            outer_ps = nl.ndarray((kd, vd), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=outer_ps, stationary=k_1kd, moving=delta)
            outer = nl.ndarray((kd, vd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=outer, src=outer_ps)
            nisa.tensor_tensor(dst=S, data1=S, data2=outer, op=nl.add)
            y_ps = nl.ndarray((1, vd), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=y_ps, stationary=q_kd1, moving=S)
            y_sb = nl.ndarray((1, vd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=y_sb, src=y_ps)
            nisa.dma_copy(dst=y_hbm.ap(pattern=[[vd, 1], [1, vd]], offset=v_off), src=y_sb)

            # WRITE back to compact row (contiguous, NO indirect scatter).
            nisa.dma_copy(dst=ro[row], src=S)

    return y_hbm, rec_out_hbm
