"""Paged prefix-KV gather NKI kernel for the hybrid full-attention prefill/decode path.

WHY THIS EXISTS
---------------
The full-attn prefill prefix-KV gather in model.py (`K_blocks = k_src[flat_idx]`) is a torch
fancy-index. When `k_src` is a PAGE-STRIDED `as_strided` view of the shared unified KV slab
(block k at byte k*page_size, so full-attn can SHARE the slab with GDN state via disjoint block
ids), neuronx-cc's GENERIC lowering rejects it: "Detected non-contiguous slicing for requested
Device Tensor." (Empirically hit on device, 2026-07-08, prefill bucket 512.)

This kernel expresses the SAME gather as an `.ap` access-pattern indirect DMA with an EXPLICIT
page stride (`vector_offset=block_ids, indirect_dim=0`) — the form the shipping GDN slot kernels
(gdn_conv_update.py / gdn_state_update.py) already use and that LOWERS on this stack. The stride
is a pattern literal handed to the DMA engine, not a property of a strided torch tensor, so the
"non-contiguous slice" generic path is never taken.

Result: full-attn KV genuinely shares the unified slab (memory reclaimed, page table justified)
AND compiles.

LAYOUT
------
k_slab / v_slab: page-strided views, logical shape [num_blocks, nkh, block_size, head_dim].
  Physically block k starts at element k*page_stride (page_stride >= nkh*block_size*head_dim).
  Worked example at TP8: nkh=1 (KV heads replicated, num_kv_heads=2 < tp), block_size=256,
  head_dim=256 => per-block tile = 1*256*256 = 65536 elems.
block_table: [B, num_blocks_per_seq] int32 — pool block ids per sequence (real BlockPool ids).

OUTPUT
------
K_gathered, V_gathered: [B, nkh, S_ctx, head_dim] where S_ctx = num_blocks_per_seq*block_size,
matching model.py forward_prefill's `K_g.permute(0,2,1,3,4).reshape(B, nkh, S_ctx, head_dim)`.
"""
import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode


def kernel_assert(condition, message=""):
    assert condition, message


@nki.jit
def paged_kv_write(
    kv_pool_hbm,    # [num_blocks, page_stride] CONTIGUOUS raw slab (WRITE TARGET; holds BOTH K and V)
    k_vals_hbm,     # [n_tok, nkh, head_dim] post-RoPE K for the tokens
    v_vals_hbm,     # [n_tok, nkh, head_dim] post-RoPE V for the tokens
    block_id_hbm,   # [n_tok, 1] int32 per-token pool block id (SMALL: 0..num_blocks; NOT block*page)
    pos_hbm,        # [n_tok, 1] int32 per-token INTRA-page slot = position within the block (0..block_size)
    n_tok=1,
    nkh=1,
    block_size=256,
    head_dim=256,
    page_stride=0,  # per-block stride (elems) = raw slab row width; 0 => contiguous (per_block)
    k_col=0,        # column offset (elems) of K within each page (0)
    v_col=0,        # column offset (elems) of V within each page (= per_block)
):
    """POSITIONAL KV write: scatter BOTH K and V for each token into kv_pool. K and V share the SAME
    raw slab (K at page column k_col, V at v_col), so they MUST be written in ONE kernel call: XLA
    allows a given input parameter to alias at MOST ONE output per graph, and both K and V alias
    kv_pool — two separate calls would violate hlo_input_output_alias_config (neuronx-cc error 70).

    32-BIT-SAFE ADDRESSING (the load-bearing fix): the slab is BLOCK-INDEXED like paged_kv_gather_raw,
    so NO flat `block*page_stride` int32 offset is ever formed. When num_blocks*page_stride >= 2**31
    that flat offset overflowed int32 and neuronx-cc rejected the scatter ("dst_indirect_max_index
    exceeds 32-bit range"), forcing a num_blocks<=16383 cap that HALVED the usable KV cache. Instead:

      - Keep kv_pool_hbm 2-D and reshape (FREE, contiguous) to [num_blocks*n_slots, head_dim] where
        n_slots = page_stride // head_dim (a page holds n_slots head_dim-wide slots).
      - The DMA's indirect stride for indirect_dim=0 is head_dim (the reshaped row width). The
        vector_offset is a per-token SLOT index `slot = block_id*n_slots + (k/v col slot) + pos`,
        which stays int32-safe (<< 2**31: 17000*512 + 255 ~= 8.7M). The engine computes the byte
        address as `slot * head_dim` in a WIDER-than-32-bit space, so the resolved element offset
        `block_id*page_stride + col + pos*head_dim` reaches past 2**31 WITHOUT any int32 overflow.
        Device-verified 2026-07-09: a write into block 16999 (base 2.23e9 > 2**31) lands K AND V
        bytes at the correct page columns (raw slab readback of all 256 positions).
      - CRITICAL: the ENTIRE (block, pos, K/V col, head) address goes into that int32-safe slot
        index; the .ap static `offset=` is kept 0. A NONZERO static .ap offset is added in 32-bit
        address arithmetic and OVERFLOWS once the indirect base exceeds 2**31 (device-verified:
        writing V via a static offset=65536 silently dropped the store past 2**31).
      - The caller passes block_id (SMALL) and pos SEPARATELY, never `block*page_stride` — so the
        int64->int32 cast that overflowed on the host is gone.

    IN-PLACE (trap #1): scatter directly into kv_pool_hbm and RETURN it -> compiler auto-aliases
    (operand_output_aliases={0:0}), persisting to the manager-visible slab with NO fresh alloc/copy.
    dtype follows the pool; dma_copy is a byte mover. oob_mode.skip guards padding tokens (a -1 /
    OOB block id writes nothing)."""
    num_blocks = kv_pool_hbm.shape[0]
    row_w = kv_pool_hbm.shape[1]  # = page_stride (raw slab row width)
    stride = row_w if page_stride == 0 else page_stride
    kernel_assert(stride % head_dim == 0, "page_stride must be a multiple of head_dim")
    kernel_assert(k_col % head_dim == 0 and v_col % head_dim == 0,
                  "K/V column offsets must be head_dim-aligned (slot-indexed addressing)")
    _dt = kv_pool_hbm.dtype
    n_slots = stride // head_dim  # head_dim-wide slots per page (row width of the reshaped slab)
    # FREE contiguous reshape: [num_blocks, page_stride] -> [num_blocks*n_slots, head_dim]. Each row is
    # one head_dim-wide slot; row r sits at element r*head_dim in the underlying storage. The .ap below
    # scatters into row (block_id*n_slots + col_slot + pos) via vector_offset (indirect stride=head_dim).
    kv_slots = kv_pool_hbm.reshape((num_blocks * n_slots, head_dim))
    _kv3 = k_vals_hbm.reshape((n_tok, nkh, head_dim))
    _vv3 = v_vals_hbm.reshape((n_tok, nkh, head_dim))
    _bid = block_id_hbm.reshape((n_tok, 1))
    _pos = pos_hbm.reshape((n_tok, 1))
    k_col_slot = k_col // head_dim  # static K column expressed in head_dim-slots
    v_col_slot = v_col // head_dim

    for t0 in range(0, n_tok, 128):
        t_sz = min(128, n_tok - t0)
        # per-TILE block-id / position loads (partition = t_sz <= 128; prefill n_tok up to bucket 512).
        bid_i32 = nl.ndarray((t_sz, 1), dtype=nl.int32, buffer=nl.sbuf)
        pos_i32 = nl.ndarray((t_sz, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bid_i32, src=_bid[t0:t0 + t_sz, 0:1])
        nisa.dma_copy(dst=pos_i32, src=_pos[t0:t0 + t_sz, 0:1])
        # base slot = block_id * n_slots + pos  (int32-safe: << 2**31; head_dim multiply is wide/DMA).
        # gpsimd = native integer arithmetic (no fp32 round-trip, so exact even past fp32's 2**24).
        base_slot = nl.ndarray((t_sz, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=base_slot, data=bid_i32, op0=nl.multiply, operand0=n_slots,
                           engine=nisa.engine.gpsimd)
        nisa.tensor_tensor(dst=base_slot, data1=base_slot, data2=pos_i32, op=nl.add,
                           engine=nisa.engine.gpsimd)
        for h in range(nkh):
            head_slot = h * block_size  # static per-head slot offset within a page (h * block_size slots)
            # CRITICAL: fold the K/V column + head slot INTO the vector slot index (NOT the .ap static
            # `offset=`). Device-verified 2026-07-09 at num_blocks=17000: a NONZERO static .ap offset
            # is added in the 32-bit address arithmetic and OVERFLOWS once the indirect base exceeds
            # 2**31 (K@offset=0 read back correctly past 2**31; V@offset=65536 read back 0 — silently
            # dropped). Only the vector index's `slot*head_dim` multiply resolves in the wider address
            # space. So keep .ap offset=0 and put ALL of the (block, pos, col, head) address into the
            # int32-safe slot index (max ~8.7M << 2**31); the DMA scales it by head_dim in wide space.
            # --- K --- slot = base_slot + (k_col_slot + head_slot)
            k_slot = nl.ndarray((t_sz, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=k_slot, data=base_slot, op0=nl.add,
                               operand0=(k_col_slot + head_slot), engine=nisa.engine.gpsimd)
            ks = nl.ndarray((t_sz, head_dim), dtype=_dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=ks, src=_kv3[t0:t0 + t_sz, h, 0:head_dim])
            k_slot_ap = k_slot.view(nl.uint32).ap(pattern=[[1, t_sz], [1, 1]], offset=0)
            nisa.dma_copy(
                dst=kv_slots.ap(pattern=[[head_dim, t_sz], [1, head_dim]], offset=0,
                                vector_offset=k_slot_ap, indirect_dim=0),
                src=ks, oob_mode=oob_mode.skip,
            )
            # --- V (same slab, different page column via v_col) --- slot = base_slot + (v_col_slot + head_slot)
            v_slot = nl.ndarray((t_sz, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=v_slot, data=base_slot, op0=nl.add,
                               operand0=(v_col_slot + head_slot), engine=nisa.engine.gpsimd)
            vs = nl.ndarray((t_sz, head_dim), dtype=_dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=vs, src=_vv3[t0:t0 + t_sz, h, 0:head_dim])
            v_slot_ap = v_slot.view(nl.uint32).ap(pattern=[[1, t_sz], [1, 1]], offset=0)
            nisa.dma_copy(
                dst=kv_slots.ap(pattern=[[head_dim, t_sz], [1, head_dim]], offset=0,
                                vector_offset=v_slot_ap, indirect_dim=0),
                src=vs, oob_mode=oob_mode.skip,
            )
    return kv_pool_hbm  # single input -> single output alias (in-place persist)


@nki.jit
def paged_kv_gather_raw(
    raw_slab_hbm,   # [num_blocks, page_stride] CONTIGUOUS raw slab (K and V share it per page)
    block_table_hbm,  # [B, num_blocks_per_seq] int32 pool block ids
    B=1,
    num_blocks_per_seq=1,
    nkh=1,
    block_size=256,
    head_dim=256,
    k_off=0,        # column offset (elems) of K within each page (0)
    v_off=0,        # column offset (elems) of V within each page (= per_block)
):
    """Gather K and V from ONE contiguous raw slab [num_blocks, page_stride] via .ap page-stride
    indirect DMA. K at page column k_off, V at v_off. Because raw_slab is CONTIGUOUS, the internal
    reshape is free (no strided-view reshape -> no "non-contiguous slicing" reject — the root cause
    of the surviving prefill-512 failure). Output natural [B, nblk, nkh, block, hd]; caller permutes.

    This replaces paged_kv_gather's strided-view input: the runner passes the contiguous base slab
    (raw_tensor.view(dtype) reshaped [num_blocks, page_stride]) + K/V column offsets, NOT the
    as_strided k_cache/v_cache views (whose reshape((num_blocks, per_block)) is non-contiguous)."""
    per_block = nkh * block_size * head_dim
    page_stride = raw_slab_hbm.shape[1]
    kernel_assert(k_off + per_block <= page_stride and v_off + per_block <= page_stride,
                  "K/V region must fit within the page")
    # 32-BIT-SAFE ADDRESSING. The original gather formed the DMA address as idx*page_stride +
    # (k/v_off + f0): the idx*page_stride term IS computed in the wide (>32-bit) indirect-DMA address
    # space (indirect_dim=0, stride=page_stride), so the BLOCK base never overflowed — device-proven,
    # the f0=0 tile read correctly past 2**31. What overflowed was the LARGE static free-axis span:
    # the inner [1, f_sz] extent (f_sz up to 8192) plus f0 composed with the >2**31 base in 32-bit and
    # dropped the later columns (K mismatch at pos>=32). FIX: keep the block base on the wide indirect
    # axis (indirect_dim=0 selecting block_id, stride=page_stride — SMALL indirect dim = num_blocks,
    # avoiding a separate huge-indirect-dim DGE artifact seen when reshaping to [num_blocks*n_slots,
    # head_dim]) and read the per-block KV region in head_dim-wide slices whose STATIC page-column
    # offset (k/v_off + sj*head_dim) is always < page_stride (131072) — an int32-safe small constant
    # that composes with the wide base without overflow. Device-verified 2026-07-09 at num_blocks=17000.
    _dt = raw_slab_hbm.dtype
    n_kv_slots = per_block // head_dim  # head_dim-wide slots in the K (or V) region of one block
    K_out = nl.ndarray((B, num_blocks_per_seq, nkh, block_size, head_dim), dtype=_dt, buffer=nl.shared_hbm)
    V_out = nl.ndarray((B, num_blocks_per_seq, nkh, block_size, head_dim), dtype=_dt, buffer=nl.shared_hbm)
    n_rows = B * num_blocks_per_seq
    # output as [n_rows, n_kv_slots, head_dim] (row-major identity of [n_rows, per_block]) so each
    # head_dim-wide slot lands in its own contiguous column band.
    ko_slots = K_out.reshape((n_rows, n_kv_slots, head_dim))
    vo_slots = V_out.reshape((n_rows, n_kv_slots, head_dim))
    bt_flat = block_table_hbm.reshape((n_rows, 1))

    for p0 in range(0, n_rows, 128):
        p_sz = min(128, n_rows - p0)
        idx_i32 = nl.ndarray((p_sz, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_i32, src=bt_flat[p0:p0 + p_sz, 0:1])
        idx_ap = idx_i32.view(nl.uint32).ap(pattern=[[1, p_sz], [1, 1]], offset=0)
        # COALESCED gather: read SLOTS_PER_READ head_dim-wide
        # slots per indirect DMA instead of 1, collapsing the per-block read count from n_kv_slots
        # (=block_size=256, whose ~512B packets throttle the DMA engines) to
        # ceil(n_kv_slots/SLOTS_PER_READ). Correctness-preserving because the reads are the SAME
        # contiguous bytes, just fewer/larger DMAs. Two HARD constraints kept:
        #   (1) int32-safety: the STATIC page-column offset (k/v_off + grp*grp_elems) must stay
        #       < page_stride so it composes with the wide idx*page_stride base without 32-bit
        #       overflow. grp*grp_elems <= per_block <= page_stride (asserted) → safe.
        #   (2) SBUF free-dim <= 32767: grp_elems = SLOTS_PER_READ*head_dim must fit. head_dim=256 →
        #       SLOTS_PER_READ<=127; use 64 (16384 elems, safe) → 256 slots become 4 reads (64x fewer pkts).
        SLOTS_PER_READ = max(1, min(n_kv_slots, 32768 // head_dim // 2))  # 64 for head_dim=256
        grp_elems = SLOTS_PER_READ * head_dim
        for g0 in range(0, n_kv_slots, SLOTS_PER_READ):
            g_slots = min(SLOTS_PER_READ, n_kv_slots - g0)
            g_elems = g_slots * head_dim
            k_tile = nl.zeros((p_sz, g_elems), dtype=_dt, buffer=nl.sbuf)
            v_tile = nl.zeros((p_sz, g_elems), dtype=_dt, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=k_tile,
                src=raw_slab_hbm.ap(
                    pattern=[[page_stride, p_sz], [1, g_elems]],
                    offset=k_off + g0 * head_dim,
                    vector_offset=idx_ap,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )
            nisa.dma_copy(
                dst=v_tile,
                src=raw_slab_hbm.ap(
                    pattern=[[page_stride, p_sz], [1, g_elems]],
                    offset=v_off + g0 * head_dim,
                    vector_offset=idx_ap,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )
            # ko_slots[:, g0:g0+g_slots, :] is a contiguous [g_slots, head_dim] band == g_elems row-major.
            nisa.dma_copy(dst=ko_slots[p0:p0 + p_sz, g0:g0 + g_slots, 0:head_dim],
                          src=k_tile.reshape((p_sz, g_slots, head_dim)))
            nisa.dma_copy(dst=vo_slots[p0:p0 + p_sz, g0:g0 + g_slots, 0:head_dim],
                          src=v_tile.reshape((p_sz, g_slots, head_dim)))
    return K_out, V_out


@nki.jit
def paged_state_gather(
    pool_hbm,       # [num_blocks, page_row] CONTIGUOUS raw slab (GDN state region within each page)
    idx_hbm,        # [n_rows, 1] int32 pool block ids
    n_rows=1,
    row_elems=1,    # state elems per block (prod(state_tail))
    page_stride=0,  # per-block stride (elems) = raw slab row width; 0 => contiguous (row_elems)
    state_off=0,    # column offset (elems) of THIS state component within the page (conv vs recurrent)
):
    """Gather compact rows out[r] = pool[idx[r]] from a CONTIGUOUS raw slab via .ap page-stride
    indirect DMA. Output is CONTIGUOUS [n_rows, row_elems]; caller reshapes to [n_rows, *state_tail].

    Takes the raw slab (NOT the strided as_strided state view): reshape((num_blocks, row_elems)) on a
    page-strided view is non-contiguous and device-rejected; the contiguous raw slab makes addressing
    free (page stride is a pattern literal). state_off selects the component's column within the page.
    oob_mode.skip + zero-init guards OOB / -1 ids."""
    stride = row_elems if page_stride == 0 else page_stride
    kernel_assert(stride >= row_elems, "page_stride must be >= row_elems")
    row_w = pool_hbm.shape[1]  # raw slab row width (= page_stride)
    out = nl.ndarray((n_rows, row_elems), dtype=pool_hbm.dtype, buffer=nl.shared_hbm)
    o_flat = out.reshape((n_rows, row_elems))

    F_TILE = 8192
    for r0 in range(0, n_rows, 128):
        r_sz = min(128, n_rows - r0)
        # per-TILE idx load (partition <= 128)
        idx_i32 = nl.ndarray((r_sz, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_i32, src=idx_hbm.reshape((n_rows, 1))[r0:r0 + r_sz, 0:1])
        idx_ap = idx_i32.view(nl.uint32).ap(pattern=[[1, r_sz], [1, 1]], offset=0)
        for f0 in range(0, row_elems, F_TILE):
            f_sz = min(F_TILE, row_elems - f0)
            t = nl.zeros((r_sz, f_sz), dtype=pool_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=t,
                src=pool_hbm.ap(
                    pattern=[[stride, r_sz], [1, f_sz]],
                    offset=state_off + f0,
                    vector_offset=idx_ap,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )
            nisa.dma_copy(dst=o_flat[r0:r0 + r_sz, f0:f0 + f_sz], src=t)
    return out


@nki.jit
def paged_state_scatter(
    pool_hbm,       # [num_blocks, page_row] CONTIGUOUS raw slab (WRITE TARGET; GDN state region)
    new_hbm,        # [n_rows, *state_tail] COMPACT updated rows to write back
    idx_hbm,        # [n_rows, 1] int32 pool block ids
    n_rows=1,
    row_elems=1,    # state elems per block (prod(state_tail))
    page_stride=0,  # per-block stride (elems) = raw slab row width; 0 => contiguous (row_elems)
    state_off=0,    # column offset (elems) of THIS state component within the page
):
    """Scatter compact rows -> raw_slab[idx] (state region at column state_off) via .ap indirect DMA.
    Returns the FULL raw slab with rows[idx] overwritten (caller: self._raw_slab.copy_(out) — trap #1
    persist to the manager-visible slab). Takes the CONTIGUOUS raw slab (not the strided state view):
    the whole-slab baseline copy + the indirect scatter are contiguous/pattern-literal, so they lower
    where a strided-view reshape/index_copy_ would not. oob_mode.skip: a -1 (pad) idx writes nothing.
    """
    stride = row_elems if page_stride == 0 else page_stride
    kernel_assert(stride >= row_elems, "page_stride must be >= row_elems")
    n_flat = new_hbm.reshape((n_rows, row_elems))
    F_TILE = 8192
    # IN-PLACE: scatter the changed rows DIRECTLY into pool_hbm at (block idx, page column state_off)
    # and RETURN pool_hbm. Returning the input auto-aliases (operand_output_aliases={0:0}) so the
    # write persists with NO fresh full-slab pool_out and NO baseline copy — untouched bytes (incl.
    # co-resident full-attn KV and the other state component in the page) are preserved because they
    # are never written. Eliminates the ~711MB/call alloc that OOM'd the device.
    for r0 in range(0, n_rows, 128):
        r_sz = min(128, n_rows - r0)
        idx_i32 = nl.ndarray((r_sz, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_i32, src=idx_hbm.reshape((n_rows, 1))[r0:r0 + r_sz, 0:1])
        idx_ap = idx_i32.view(nl.uint32).ap(pattern=[[1, r_sz], [1, 1]], offset=0)
        for f0 in range(0, row_elems, F_TILE):
            f_sz = min(F_TILE, row_elems - f0)
            src = nl.ndarray((r_sz, f_sz), dtype=pool_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=src, src=n_flat[r0:r0 + r_sz, f0:f0 + f_sz])
            nisa.dma_copy(
                dst=pool_hbm.ap(
                    pattern=[[stride, r_sz], [1, f_sz]],
                    offset=state_off + f0,
                    vector_offset=idx_ap,
                    indirect_dim=0,
                ),
                src=src,
                oob_mode=oob_mode.skip,
            )
    return pool_hbm  # returning the input -> auto operand_output_aliases (in-place persist)


@nki.jit
def paged_kv_gather(
    k_slab_hbm,     # [num_blocks, nkh, block_size, head_dim] page-strided view (K)
    v_slab_hbm,     # [num_blocks, nkh, block_size, head_dim] page-strided view (V)
    block_table_hbm,  # [B, num_blocks_per_seq] int32 — pool block ids
    B=1,
    num_blocks_per_seq=1,
    nkh=1,
    block_size=256,
    head_dim=256,
    page_stride=0,   # per-block stride in ELEMENTS of the slab's underlying storage; 0 => contiguous (nkh*block_size*head_dim)
):
    """Gather paged K/V blocks via .ap indirect DMA with an explicit page stride.

    Returns:
        K_gathered, V_gathered: [B, nkh, S_ctx, head_dim] fp32, S_ctx = num_blocks_per_seq*block_size.
    Notes:
        The indirect gather addresses dim0 (block) of the slab by the flattened block-table ids.
        Per gathered block we copy its contiguous [nkh, block_size, head_dim] region (the slab's
        per-block bytes are contiguous WITHIN a page; only the block-to-block stride is page_stride).
        We tile the free axis so no SBUF tile exceeds limits. B*num_blocks_per_seq is the gather
        "row" count and maps to the partition axis (tiled by 128).
    """
    kernel_assert(B >= 1 and num_blocks_per_seq >= 1, "B, num_blocks_per_seq >= 1")
    per_block = nkh * block_size * head_dim
    stride = per_block if page_stride == 0 else page_stride
    kernel_assert(stride >= per_block, "page_stride must be >= per-block elems")

    # Output in the NATURAL gather order [B, num_blocks_per_seq, nkh, block_size, head_dim] — the
    # row-major identity of the torch `k_src[flat_idx].view(B, nblk, nkh, block_size, head_dim)`.
    # The CALLER does the `.permute(0,2,1,3,4).reshape(B, nkh, S_ctx, head_dim)` (same as today's
    # post-fancy-index code), so this is correct for ANY nkh (a [B,nkh,S_ctx,hd] kernel output would
    # only match for nkh==1). Keeps the kernel a pure gather; layout transform stays in torch.
    # dtype MUST match the KV slab (bf16 for this model) — dma_copy is a byte mover, so a fp32 dst
    # reading a bf16 src misreads. (Round-6 audit Bug 2: was hardcoded fp32.)
    _kv_dtype = k_slab_hbm.dtype
    K_out = nl.ndarray((B, num_blocks_per_seq, nkh, block_size, head_dim), dtype=_kv_dtype, buffer=nl.shared_hbm)
    V_out = nl.ndarray((B, num_blocks_per_seq, nkh, block_size, head_dim), dtype=_kv_dtype, buffer=nl.shared_hbm)

    n_rows = B * num_blocks_per_seq  # one gather "row" per (seq, block) pair

    # Flattened block ids [n_rows] int32 -> uint32 for the vector_offset engine (matches GDN kernels).
    idx_i32 = nl.ndarray((n_rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    bt_flat = block_table_hbm.reshape((n_rows, 1))
    nisa.dma_copy(dst=idx_i32, src=bt_flat[0:n_rows, 0:1])
    idx_u32 = idx_i32.view(nl.uint32)

    # Flatten slabs to [num_blocks, per_block] for the .ap gather; the page stride is expressed in
    # the pattern's dim0 stride (block-to-block), inner (nkh*block_size*head_dim) is contiguous.
    # NOTE: on a PAGE-STRIDED / V-OFFSET view (per_block < stride) this reshape is non-contiguous
    # and forces a device-rejected copy — the surviving prefill-512 reject. The proper fix routes the
    # RAW CONTIGUOUS slab (not the as_strided view) into this kernel; see runner plumbing task.
    num_blocks = k_slab_hbm.shape[0]
    k_flat = k_slab_hbm.reshape((num_blocks, per_block))
    v_flat = v_slab_hbm.reshape((num_blocks, per_block))
    ko_flat = K_out.reshape((n_rows, per_block))
    vo_flat = V_out.reshape((n_rows, per_block))

    # Tile the gather rows over partitions (<=128) and the per-block free axis (<=SBUF free limit).
    F_TILE = 8192  # fp32 elems per tile, well under 32767 SBUF free-dim limit
    for p0 in range(0, n_rows, 128):
        p_sz = min(128, n_rows - p0)
        # per-partition slot index for this partition tile
        idx_ap = idx_u32.ap(pattern=[[1, p_sz], [1, 1]], offset=p0)
        for f0 in range(0, per_block, F_TILE):
            f_sz = min(F_TILE, per_block - f0)
            # zero-init so an OOB / padded (-1) block id reads zeros (oob_mode.skip = no-op DMA),
            # matching the GDN slot kernels' PAD_SLOT_ID handling — inert, no garbage.
            k_tile = nl.zeros((p_sz, f_sz), dtype=_kv_dtype, buffer=nl.sbuf)
            v_tile = nl.zeros((p_sz, f_sz), dtype=_kv_dtype, buffer=nl.sbuf)
            # GATHER: row r (= partition) reads block idx[r] at element idx[r]*stride + f0,
            # then f_sz contiguous elems. stride is the PAGE stride (pattern literal), so this
            # lowers as an indirect DMA (NOT a generic non-contiguous device slice). oob_mode.skip
            # guards block ids >= num_blocks (padded rows / bounds) — the DGE OOB the GDN kernels
            # already guard this way; without it a padded/global id trips "vector DGE out-of-bound".
            nisa.dma_copy(
                dst=k_tile,
                src=k_flat.ap(
                    pattern=[[stride, p_sz], [1, f_sz]],
                    offset=f0,
                    vector_offset=idx_ap,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )
            nisa.dma_copy(
                dst=v_tile,
                src=v_flat.ap(
                    pattern=[[stride, p_sz], [1, f_sz]],
                    offset=f0,
                    vector_offset=idx_ap,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )
            nisa.dma_copy(dst=ko_flat[p0:p0 + p_sz, f0:f0 + f_sz], src=k_tile)
            nisa.dma_copy(dst=vo_flat[p0:p0 + p_sz, f0:f0 + f_sz], src=v_tile)

    # K_out/V_out are laid [B, nkh, S_ctx, head_dim] but we gathered rows as (b, block) with each
    # row = one block's [nkh, block_size, head_dim]. ko_flat row (b*nblk + blk) maps to
    # K_out[b, :, blk*block_size:(blk+1)*block_size, :] — this is exactly the row-major identity of
    # reshape((B, num_blocks_per_seq, nkh, block_size, head_dim)) THEN the caller's permute to
    # [B, nkh, S_ctx, head_dim]. We return the [B, num_blocks_per_seq*..] contiguous form and let
    # the caller reshape/permute (kept in torch, cheap, contiguous). See wrapper below.
    return K_out, V_out
