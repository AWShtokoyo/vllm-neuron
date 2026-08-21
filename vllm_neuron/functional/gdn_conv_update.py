"""A2: slot-indexed GatedDeltaNet decode CONV-state update NKI kernel.

Replaces the per-token conv1d update in the unified-cache decode path. Today that
step is done in torch either as a positional `conv_state[:B]` slice (contiguous) or
as a one-hot matmul gather/scatter (Option 3, `_gather_rows`/`_scatter_rows` in
model.py:1216-1234). This kernel does the SAME update but addresses the state by a
RUNTIME slot index via indirect DMA — the upstream-faithful mechanism.

Per active sequence b (slot s = state_indices[b]):
    window = concat(conv_state[s], x[b])          # [C_dim, k]   (old k-1 cols + new token)
    y[b]   = silu( sum_j window[:, j] * w[:, j] )  # depthwise conv, per-channel k-tap
    conv_state[s] = window[:, 1:]                  # shift: drop oldest, keep newest k-1

Equivalent torch (model.py:1254-1273):
    qkv_t = qkv.transpose(1,2)                     # [B, C_dim, 1]
    conv_active = conv_state[slot]                 # [B, C_dim, k-1]
    conv_input  = cat([conv_active, qkv_t], -1)    # [B, C_dim, k]
    y = silu(conv1d(conv_input, w, groups=C_dim))  # [B, C_dim, 1]
    conv_state[slot] = conv_input[:, :, 1:]

Layout / partition mapping
--------------------------
conv_state pool: [N_SLOTS, C_dim, k-1] row-major; per-slot row is C_dim*(k-1)
contiguous fp-elems. We map the CHANNEL axis C_dim to partitions (<=128 per tile,
tile if larger) so the k-tap reduction is along the FREE axis (contiguous, the
efficient DMA/vector layout). The slot gather/scatter is the probe idiom:
`pool.ap(pattern=..., vector_offset=idx, indirect_dim=0)` (validated by the
P-KERNEL probe; shipping precedent = topk_reduce.py:278).

Make-or-break note: the slot offset is RUNTIME (from state_indices). We express it
as an index-tensor-driven indirect DMA (NOT a sequential_range register offset,
which is what OOB'd the prefill kernel). idx[b] < N_SLOTS => per-row extent in-bounds.
"""
import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode


def kernel_assert(condition, message=""):
    assert condition, message


def _silu(dst, data, tmp):
    """dst = data * sigmoid(data), computed in SBUF. tmp is a scratch tile of data.shape."""
    nisa.activation(dst=tmp, data=data, op=nl.sigmoid)
    nisa.tensor_tensor(dst=dst, data1=data, data2=tmp, op=nl.multiply)


@nki.jit
def gdn_conv_update(
    conv_state_hbm,   # [N_SLOTS, C_dim, k1]  fp32/bf16 pool (k1 = k-1)
    x_hbm,            # [B, C_dim]            fp32/bf16 new-token channels (qkv_t squeezed)
    w_hbm,            # [C_dim, k]            fp32/bf16 depthwise conv weight
    idx_hbm,          # [B, 1]                int32 slot indices = block_table[:,0]
    B=1,              # active sequences (<=128)
    C_dim=1,          # conv channels (rank-local)
    k=4,              # conv kernel size (k1 = k-1 stored)
):
    """Slot-indexed decode conv update: gather window, depthwise conv+silu, scatter shifted window.

    Returns:
        y_hbm: [B, C_dim] fp32 = silu(depthwise-conv(window)). The updated conv_state
               is scattered back into conv_state_hbm in place (returned too for the caller).
    Notes:
        Channel axis C_dim maps to partitions; tiled by 128. k-tap reduction is on the
        free axis. One indirect gather + one indirect scatter per channel tile.
    """
    k1 = k - 1
    kernel_assert(B <= 128, "B must be <= 128 (index partition bound)")
    kernel_assert(k <= 32, "conv kernel must be small (free-axis window)")

    y_hbm = nl.ndarray((B, C_dim), dtype=nl.float32, buffer=nl.shared_hbm)
    # Pass conv_state through as an explicit output so the in-place scatter is observable.
    conv_out_hbm = nl.ndarray(conv_state_hbm.shape, dtype=conv_state_hbm.dtype, buffer=nl.shared_hbm)

    # Load slot indices once. int32 -> uint32 view for the vector_offset engine.
    idx_i32 = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_i32, src=idx_hbm[0:B, 0:1])
    idx_u32 = idx_i32.view(nl.uint32)
    idx_ap = idx_u32.ap(pattern=[[1, B], [1, 1]], offset=0)

    # Baseline copy conv_state -> conv_out (untouched rows preserved; touched rows overwritten below).
    for p0 in range(0, conv_state_hbm.shape[0], 128):
        p_sz = min(128, conv_state_hbm.shape[0] - p0)
        row = nl.ndarray((p_sz, C_dim * k1), dtype=conv_state_hbm.dtype, buffer=nl.sbuf)
        cs2 = conv_state_hbm.reshape((conv_state_hbm.shape[0], C_dim * k1))
        co2 = conv_out_hbm.reshape((conv_out_hbm.shape[0], C_dim * k1))
        nisa.dma_copy(dst=row, src=cs2[p0:p0 + p_sz, 0:C_dim * k1])
        nisa.dma_copy(dst=co2[p0:p0 + p_sz, 0:C_dim * k1], src=row)

    # Flattened [N_SLOTS, C_dim*k1] views for the indirect slot access.
    N_SLOTS = conv_state_hbm.shape[0]
    cs_flat = conv_state_hbm.reshape((N_SLOTS, C_dim * k1))
    co_flat = conv_out_hbm.reshape((N_SLOTS, C_dim * k1))
    row_stride = C_dim * k1

    # ===== per channel-tile: gather window, conv, silu, scatter shifted window =====
    for c0 in range(0, C_dim, 128):
        c_sz = min(128, C_dim - c0)

        # 1. GATHER conv_state[slot] for this channel tile -> [B, c_sz, k1].
        #    Indirect DMA on the SLOT (dim0); the channel span is a contiguous sub-block
        #    of each slot row at offset c0*k1 (row-major [C_dim, k1]).
        # RISK 1 (-1 padding slot): zero-init + oob_mode.skip so a padded lane (slot=-1, OOB as
        # uint32) reads zeros and its scatter (below) writes nothing — inert no-op per lane,
        # matching the one-hot semantics. Valid lanes copy normally.
        win = nl.zeros((B, c_sz, k1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=win,
            src=cs_flat.ap(
                pattern=[[row_stride, B], [k1, c_sz], [1, k1]],
                offset=c0 * k1,
                vector_offset=idx_ap,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip,
        )

        # 2. Load the new token x[b, c0:c0+c_sz] -> [B, c_sz].
        xnew = nl.ndarray((B, c_sz), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=xnew, src=x_hbm[0:B, c0:c0 + c_sz])

        # 3. Depthwise conv over the length-k window (k-1 old taps + new token):
        #    y[b,c] = sum_{j<k1} win[b,c,j]*w[c,j]  +  xnew[b,c]*w[c,k-1]
        #    B is on the PARTITION axis (forced by the indirect gather); channel c is on
        #    the FREE axis and the weight varies per-channel, so we CANNOT use tensor_scalar
        #    (per-partition operand). Instead build a partition-broadcast weight tile
        #    wtap[b,c] = w[c0+c, j] (same across all B partitions) via a stride-0 partition
        #    DMA, then tensor_tensor multiply. w_hbm is [C_dim, k] row-major: element
        #    (c, j) is at c*k + j; broadcast over B => partition stride 0.
        acc = nl.zeros((B, c_sz), dtype=nl.float32, buffer=nl.sbuf)
        tap = nl.ndarray((B, c_sz), dtype=nl.float32, buffer=nl.sbuf)
        wtap = nl.ndarray((B, c_sz), dtype=nl.float32, buffer=nl.sbuf)
        w_flat = w_hbm.reshape((C_dim * k,))
        for j in range(k):
            nisa.dma_copy(
                dst=wtap,
                src=w_flat.ap(pattern=[[0, B], [k, c_sz]], offset=c0 * k + j),
            )
            src_tap = win[0:B, 0:c_sz, j] if j < k1 else xnew
            nisa.tensor_tensor(dst=tap, data1=src_tap, data2=wtap, op=nl.multiply)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=tap, op=nl.add)

        # 4. SiLU -> y[b, c0:c0+c_sz]
        sig = nl.ndarray((B, c_sz), dtype=nl.float32, buffer=nl.sbuf)
        _silu(dst=acc, data=acc, tmp=sig)
        nisa.dma_copy(dst=y_hbm[0:B, c0:c0 + c_sz], src=acc)

        # 5. SHIFTED window for scatter: newwin = concat(win[:,:,1:], xnew)  -> [B, c_sz, k1]
        newwin = nl.ndarray((B, c_sz, k1), dtype=conv_state_hbm.dtype, buffer=nl.sbuf)
        if k1 > 1:
            nisa.tensor_copy(dst=newwin[0:B, 0:c_sz, 0:k1 - 1], src=win[0:B, 0:c_sz, 1:k1])
        nisa.tensor_copy(dst=newwin[0:B, 0:c_sz, k1 - 1:k1],
                         src=xnew.reshape((B, c_sz, 1)))

        # 6. SCATTER shifted window back to conv_out[slot] for this channel tile.
        # oob_mode.skip: a -1 (padded) lane writes nothing (conv_out holds the baseline-copied pool).
        nisa.dma_copy(
            dst=co_flat.ap(
                pattern=[[row_stride, B], [k1, c_sz], [1, k1]],
                offset=c0 * k1,
                vector_offset=idx_ap,
                indirect_dim=0,
            ),
            src=newwin[0:B, 0:c_sz, 0:k1],
            oob_mode=oob_mode.skip,
        )

    return y_hbm, conv_out_hbm
