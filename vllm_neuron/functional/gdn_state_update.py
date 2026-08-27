"""Slot-indexed GatedDeltaNet decode RECURRENT-state update NKI kernel.

Replaces the one-step decode recurrence (model.py:1777-1788, sequence_length==1)
for the unified-cache path, addressing the recurrent state by a RUNTIME slot index
via indirect DMA (the same idiom gdn_conv_update uses).

Per active sequence b (slot s = state_indices[b]), per head h, with state
S[kd, vd] = recurrent_state[s, h]:
    S      = S * g            # g = exp(g_t): per-(b,h) scalar decay
    kv_mem = k^T S            # [vd]   sum over kd:  (S * k[:,None]).sum(kd)
    delta  = (v - kv_mem)*b   # [vd]   b = beta (per-(b,h) scalar)
    S      = S + k[:,None] * delta[None,:]     # rank-1 update
    y      = q^T S            # [vd]   sum over kd:  (S * q[:,None]).sum(kd)
recurrent_state[s, h] = S    # scatter back

Host prologue (done by the model wrapper, matching _recurrent_gated_delta_rule):
    q = l2norm(q)*scale ; k = l2norm(k) ; g = exp(g_t) ; beta = sigmoid(b)
This kernel takes the prepped q,k,v (fp32), the per-(b,h) scalars g,beta, and the
slot indices, and does gather -> recurrence -> scatter, returning y[B,H,vd].

Layout / partition mapping
--------------------------
recurrent_state pool: [N_SLOTS, H, kd, vd] row-major; per-slot row is H*kd*vd
contiguous. kd maps to partitions (kd=128). The two reductions (k^T S, q^T S) are
contractions over kd -> nc_matmul with k/q as the [kd,1] stationary operand.
Slot access = pool.ap(pattern=..., scalar_offset=slot, indirect_dim=0), oob_mode.error.
"""
import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode


def kernel_assert(condition, message=""):
    assert condition, message


@nki.jit
def gdn_state_update(
    rec_state_hbm,   # [N_SLOTS, H, kd, vd] fp32 recurrent-state pool
    q_hbm,           # [B, H, kd]  fp32 (l2norm'd, *scale)
    k_hbm,           # [B, H, kd]  fp32 (l2norm'd)
    v_hbm,           # [B, H, vd]  fp32
    g_hbm,           # [B, H]      fp32 decay = exp(g_t)
    beta_hbm,        # [B, H]      fp32 = sigmoid(b)
    idx_hbm,         # [B, 1]      int32 slot indices = block_table[:,0]
    B=1, H=1, kd=128, vd=128,
):
    """Slot-indexed decode recurrence: gather S, one gated-delta step, scatter S; return y.

    Returns:
        y_hbm:        [B, H, vd] fp32 decode output (q^T S_after).
        rec_out_hbm:  [N_SLOTS, H, kd, vd] recurrent-state pool with rows[idx] updated.
    Notes:
        Unrolled over B and H (both small, bucketed); each (b,h) does one indirect
        gather of S[slot,h], the rank-1 recurrence via nc_matmul contractions on kd,
        and one indirect scatter back. RUNTIME slot offset -> single runtime scalar_offset
        (not a sequential_range register offset), so it lowers like the shipping
        MoE-permute / deformable-attention scatter (topk_reduce.py:278).
    """
    kernel_assert(B <= 128 and H <= 128, "B,H must be <= 128")
    kernel_assert(kd <= 128 and vd <= 512, "kd<=128 (partition), vd<=512 (psum free)")

    N_SLOTS = rec_state_hbm.shape[0]
    y_hbm = nl.ndarray((B, H, vd), dtype=nl.float32, buffer=nl.shared_hbm)
    rec_out_hbm = nl.ndarray(rec_state_hbm.shape, dtype=rec_state_hbm.dtype, buffer=nl.shared_hbm)

    # Slot indices -> SBUF, uint32 view for scalar_offset.
    idx_i32 = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_i32, src=idx_hbm[0:B, 0:1])
    idx_u32 = idx_i32.view(nl.uint32)

    # Baseline copy pool -> out (preserve untouched slots; touched rows overwritten below).
    # head_elems = H*kd*vd can be huge (e.g. 4*128*128=65536 fp32 = 256KB/partition >> SBUF ~192KB),
    # so we MUST tile the FREE axis (staging the whole row through SBUF overflows: NCC_INLA001).
    # F_TILE bounded well under the 32767-fp32 SBUF free-dim limit.
    head_elems = H * kd * vd
    F_TILE = 8192
    rs2 = rec_state_hbm.reshape((N_SLOTS, head_elems))
    ro2 = rec_out_hbm.reshape((N_SLOTS, head_elems))
    for p0 in range(0, N_SLOTS, 128):
        p_sz = min(128, N_SLOTS - p0)
        for f0 in range(0, head_elems, F_TILE):
            f_sz = min(F_TILE, head_elems - f0)
            row = nl.ndarray((p_sz, f_sz), dtype=rec_state_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=row, src=rs2[p0:p0 + p_sz, f0:f0 + f_sz])
            nisa.dma_copy(dst=ro2[p0:p0 + p_sz, f0:f0 + f_sz], src=row)

    rs_flat = rec_state_hbm.reshape((N_SLOTS, head_elems))
    ro_flat = rec_out_hbm.reshape((N_SLOTS, head_elems))

    # ===== per (b, h): gather S[slot,h] -> recurrence step -> scatter =====
    # We gather ONE slot's [kd,vd] tile per (b,h). That is a single RUNTIME base offset
    # applied to a statically-patterned tile -> use scalar_offset (NOT vector_offset).
    # vector_offset is a per-PARTITION gather and requires len(index)==dst_partition_count
    # (=kd=128 here), which is wrong for a single-slot read. scalar_offset=slot indexes
    # indirect_dim=0 of the [N_SLOTS, head_elems] pool (stride head_elems), so the tile
    # base = slot*head_elems + head_off. (shipping idiom: bwmm_bwd_dropless.py:624-630)
    for b in range(B):
        slot_b = idx_u32.ap(pattern=[[1, 1], [1, 1]], offset=b)  # [1,1] runtime slot scalar for lane b
        for h in range(H):
            head_off = h * (kd * vd)

            # 1. GATHER S[slot_b, h] -> [kd, vd] tile within the slot row (partition=kd, free=vd).
            # RISK 1 (-1 padding slot): PAD_SLOT_ID=-1 for inactive lanes. As uint32, -1 is a huge
            # index -> OUT OF BOUNDS. oob_mode.skip makes the DMA a no-op for OOB slots (reads
            # nothing), so we ZERO-INIT S first: a padded lane reads all-zeros, runs a harmless
            # recurrence step, and (below) skips the scatter -> inert no-op, matching the one-hot
            # scatter's all-zero-row semantics. For valid slots, skip == error (copy happens).
            S = nl.zeros((kd, vd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=S,
                src=rs_flat.ap(
                    pattern=[[vd, kd], [1, vd]],
                    offset=head_off,
                    scalar_offset=slot_b,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )

            # Load q,k,v via .ap (NO partition-dim reshape — NKI forbids changing dim0 size).
            # Offsets are in elements into the flat storage; (b,h) row starts at (b*H+h)*kd (or *vd).
            qk_off = (b * H + h) * kd
            v_off = (b * H + h) * vd
            # k in BOTH orientations: [kd,1] (partition=kd, for k^T S) and [1,kd] (partition=1, outer product).
            k_kd1 = nl.ndarray((kd, 1), dtype=nl.float32, buffer=nl.sbuf)
            k_1kd = nl.ndarray((1, kd), dtype=nl.float32, buffer=nl.sbuf)
            q_kd1 = nl.ndarray((kd, 1), dtype=nl.float32, buffer=nl.sbuf)
            v_1vd = nl.ndarray((1, vd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_kd1, src=k_hbm.ap(pattern=[[1, kd], [1, 1]], offset=qk_off))
            nisa.dma_copy(dst=k_1kd, src=k_hbm.ap(pattern=[[kd, 1], [1, kd]], offset=qk_off))
            nisa.dma_copy(dst=q_kd1, src=q_hbm.ap(pattern=[[1, kd], [1, 1]], offset=qk_off))
            nisa.dma_copy(dst=v_1vd, src=v_hbm.ap(pattern=[[vd, 1], [1, vd]], offset=v_off))
            # g broadcast to all kd partitions (stride-0 partition .ap); beta as a [1,1] scalar.
            g_kd = nl.ndarray((kd, 1), dtype=nl.float32, buffer=nl.sbuf)
            beta_s = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=g_kd, src=g_hbm.ap(pattern=[[0, kd], [1, 1]], offset=b * H + h))
            nisa.dma_copy(dst=beta_s, src=beta_hbm.ap(pattern=[[1, 1], [1, 1]], offset=b * H + h))

            # 2. S = S * g   (per-partition broadcast over the [kd,vd] tile).
            nisa.tensor_scalar(dst=S, data=S, op0=nl.multiply, operand0=g_kd)

            # 3. kv_mem = k^T S  -> [1, vd].  nc_matmul(stationary=k_kd1[kd,1], moving=S[kd,vd]).
            kv_ps = nl.ndarray((1, vd), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=kv_ps, stationary=k_kd1, moving=S)
            kv = nl.ndarray((1, vd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=kv, src=kv_ps)

            # 4. delta = (v - kv_mem) * beta   -> [1, vd]
            delta = nl.ndarray((1, vd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=delta, data1=v_1vd, data2=kv, op=nl.subtract)
            nisa.tensor_scalar(dst=delta, data=delta, op0=nl.multiply, operand0=beta_s[0:1, 0:1])

            # 5. S = S + k ⊗ delta  (rank-1 update): outer = k_1kd[1,kd]^T @ delta[1,vd] -> [kd,vd].
            outer_ps = nl.ndarray((kd, vd), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=outer_ps, stationary=k_1kd, moving=delta)
            outer = nl.ndarray((kd, vd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=outer, src=outer_ps)
            nisa.tensor_tensor(dst=S, data1=S, data2=outer, op=nl.add)

            # 6. y = q^T S -> [1, vd]
            y_ps = nl.ndarray((1, vd), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=y_ps, stationary=q_kd1, moving=S)
            y_sb = nl.ndarray((1, vd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=y_sb, src=y_ps)
            nisa.dma_copy(dst=y_hbm.ap(pattern=[[vd, 1], [1, vd]], offset=v_off), src=y_sb)

            # 7. SCATTER updated S back to rec_out[slot_b, h] (scalar_offset, same as gather).
            # oob_mode.skip: a -1 (padded) slot writes NOTHING -> the inert no-op (rec_out already
            # holds the baseline-copied pool, so a padded lane leaves every real slot untouched).
            nisa.dma_copy(
                dst=ro_flat.ap(
                    pattern=[[vd, kd], [1, vd]],
                    offset=head_off,
                    scalar_offset=slot_b,
                    indirect_dim=0,
                ),
                src=S,
                oob_mode=oob_mode.skip,
            )

    return y_hbm, rec_out_hbm
