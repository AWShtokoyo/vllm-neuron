"""P-KERNEL probe: does a RUNTIME-slot-indexed DMA gather+scatter lower?

This is the make-or-break isolation test for the unified-cache GDN path. It does
NO GDN math. It answers ONE question:

  Can we gather B state rows from a pool at slot indices held in a RUNTIME index
  tensor (state_indices[b] = block_table[:,0]), modify them, and scatter them
  back to the SAME slots, and have neuronx-cc lower it WITHOUT the illegal
  vector-DGE (nrta 1006) that killed the chunked-prefill kernel?

Why we expect YES (unlike the prefill kernel):
  The prefill kernel OOB'd because `nl.sequential_range` produced a runtime
  *register* offset (c0 = i*C) consumed by a DMA under the DEFAULT dge mode ->
  an unbounded vector-DGE on the partition axis. That is a different mechanism
  than an INDEX-TENSOR-driven gather. Production kernels express exactly this
  slot-gather/scatter with `.ap(pattern=..., vector_offset=<sbuf idx>, indirect_dim=0)`:
    - gather : permute_routed_tokens.py:180 (MoE token permute by expert index)
    - scatter: bwmm_bwd_dropless.py:1935, ms_deformable_attention_bwd.py:1728
  So the runtime-index gather/scatter has broad shipping precedent. This probe
  confirms it lowers in OUR toolchain/version before we invest in the full
  gdn_state_update / gdn_conv_update kernels.

Layout (mirrors the real pool, shrunk so the lowering question is isolated from
tiling): pool is [N_SLOTS, F] row-major fp32 (row stride = F). The real per-slot
state is ~34K fp32; F here is a small representative contiguous chunk. Tiling F
is mechanical and orthogonal to whether the indirect DMA lowers.
"""
import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode


def kernel_assert(condition, message=""):
    assert condition, message


@nki.jit
def slot_indirect_probe(pool_hbm, idx_hbm, B=1, N_SLOTS=64, F=512):
    """Gather-modify-scatter B pool rows at runtime slot indices.

    Args:
        pool_hbm: [N_SLOTS, F] fp32 HBM. The (shrunk) unified state pool.
        idx_hbm:  [B, 1] int32 HBM. Slot indices (block_table[:,0]); each in [0, N_SLOTS).
        B:       number of active sequences (partition dim of the gather; <=128).
        N_SLOTS: pool row count.
        F:       contiguous free width per slot row.

    Returns:
        out_hbm: [N_SLOTS, F] = pool with rows[idx[b]] += 1.0, all other rows unchanged.
                 CPU reference can then assert the round-trip landed at the right slots.

    Notes:
        The indexed access uses `.ap(vector_offset=idx_u32, indirect_dim=0)` on BOTH
        the gather (src) and the scatter (dst) — the shipping idiom, NOT a dim-0
        torch scatter and NOT a sequential_range register offset.
    """
    kernel_assert(B <= 128, "B must be <= 128 (partition bound)")
    kernel_assert(F <= 32767, "F must be <= SBUF free max")

    out_hbm = nl.ndarray((N_SLOTS, F), dtype=nl.float32, buffer=nl.shared_hbm)

    # 1. Baseline: copy the whole pool -> out so untouched rows are verifiable.
    #    Static-offset tiling over N_SLOTS (plain range -> compile-time offsets).
    for p0 in range(0, N_SLOTS, 128):
        p_sz = min(128, N_SLOTS - p0)
        tmp = nl.ndarray((p_sz, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tmp, src=pool_hbm[p0:p0 + p_sz, 0:F])
        nisa.dma_copy(dst=out_hbm[p0:p0 + p_sz, 0:F], src=tmp)

    # 2. Load the runtime slot indices into SBUF. int32 slot values (>=0, <N_SLOTS)
    #    bit-reinterpret cleanly to uint32 (the dtype the vector_offset engine wants).
    idx_i32 = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_i32, src=idx_hbm[0:B, 0:1])
    idx_u32 = idx_i32.view(nl.uint32)

    # 3. GATHER: gathered[b, :] = pool[idx[b], :]   (runtime-index gather).
    #    Idiom matches functional/moe/topk_reduce.py:278-286: vector_offset is an
    #    .ap() over the [B,1] index tensor; indirect_dim=0 selects the pool ROW.
    #    oob_mode.error = fault loudly if a descriptor extent over-runs the pool
    #    (that is exactly the nrta-1006 condition this probe is here to rule out;
    #    with idx[b] < N_SLOTS the per-row extent idx*F+F <= N_SLOTS*F is in-bounds).
    gathered = nl.ndarray((B, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=gathered[0:B, 0:F],
        src=pool_hbm.ap(
            pattern=[[F, B], [1, F]],   # dim0: row stride=F, B rows; dim1: contiguous F
            offset=0,
            vector_offset=idx_u32.ap(pattern=[[1, B], [1, 1]], offset=0),
            indirect_dim=0,
        ),
        oob_mode=oob_mode.error,
    )

    # 4. MODIFY: stand-in for the recurrence step (just +1.0 to prove read->write).
    modified = nl.ndarray((B, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=modified, data=gathered, op0=nl.add, operand0=1.0)

    # 5. SCATTER: out[idx[b], :] = modified[b, :]   (runtime-index overwrite scatter).
    #    dst-side indirect .ap(), mirroring ms_deformable_attention_bwd.py:1728 /
    #    bwmm_bwd_dropless.py scatter, but a plain overwrite (dma_copy, not add).
    nisa.dma_copy(
        dst=out_hbm.ap(
            pattern=[[F, B], [1, F]],
            offset=0,
            vector_offset=idx_u32.ap(pattern=[[1, B], [1, 1]], offset=0),
            indirect_dim=0,
        ),
        src=modified[0:B, 0:F],
        oob_mode=oob_mode.error,
    )

    return out_hbm
