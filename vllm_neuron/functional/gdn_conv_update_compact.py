"""0-based (compact) GatedDeltaNet decode CONV update NKI kernel.

"Stage outside, kernel 0-based" variant of gdn_conv_update.py (APC_REFACTOR_PLAN §9; user-approved
2026-07-08). The conv-state window is gathered OUTSIDE by the .ap page-stride primitive, so this
kernel takes a COMPACT [B, C_dim, k1] window (row b = seq b's prior window), does depthwise conv +
silu over cat(window, new_token), and returns the conv output [B, C_dim] AND the shifted window
[B, C_dim, k1] to be scattered back OUTSIDE. No indirect DMA, no N_SLOTS baseline copy, no idx.

Conv MATH is VERBATIM from gdn_conv_update.py (per-channel-tile depthwise conv, silu). Only the
slot gather/scatter is removed. Unconditionally contiguous -> lowers regardless of slab layout.
"""
import nki
import nki.isa as nisa
import nki.language as nl


def kernel_assert(condition, message=""):
    assert condition, message


def _silu(dst, data, tmp):
    """dst = data * sigmoid(data). tmp is scratch of data.shape."""
    nisa.activation(dst=tmp, data=data, op=nl.sigmoid)
    nisa.tensor_tensor(dst=dst, data1=data, data2=tmp, op=nl.multiply)


@nki.jit
def gdn_conv_update_compact(
    win_hbm,    # [B, C_dim, k1]  fp32 COMPACT prior conv window (row b = seq b), k1 = k-1
    x_hbm,      # [B, C_dim]      fp32 new-token channels
    w_hbm,      # [C_dim, k]      fp32 depthwise conv weight
    B=1,
    C_dim=1,
    k=4,
):
    """Compact decode conv update. Returns y_hbm [B, C_dim] (silu(conv)), newwin_hbm [B, C_dim, k1]
    (the shifted window to scatter back outside)."""
    k1 = k - 1
    kernel_assert(B <= 128, "B must be <= 128 (partition bound)")
    kernel_assert(k <= 32, "conv kernel must be small (free-axis window)")

    y_hbm = nl.ndarray((B, C_dim), dtype=nl.float32, buffer=nl.shared_hbm)
    newwin_hbm = nl.ndarray((B, C_dim, k1), dtype=nl.float32, buffer=nl.shared_hbm)

    w_flat = w_hbm.reshape((C_dim * k,))
    for c0 in range(0, C_dim, 128):
        c_sz = min(128, C_dim - c0)

        # GATHER the prior window for this channel tile from the COMPACT input -> [B, c_sz, k1].
        # Contiguous slice (row b, channels c0:c0+c_sz), NO indirect DMA.
        win = nl.ndarray((B, c_sz, k1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=win, src=win_hbm[0:B, c0:c0 + c_sz, 0:k1])

        # new token x[b, c0:c0+c_sz] -> [B, c_sz]
        xnew = nl.ndarray((B, c_sz), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=xnew, src=x_hbm[0:B, c0:c0 + c_sz])

        # depthwise conv over the length-k window (VERBATIM from gdn_conv_update.py):
        #   y[b,c] = sum_{j<k1} win[b,c,j]*w[c,j] + xnew[b,c]*w[c,k-1]
        acc = nl.zeros((B, c_sz), dtype=nl.float32, buffer=nl.sbuf)
        tap = nl.ndarray((B, c_sz), dtype=nl.float32, buffer=nl.sbuf)
        wtap = nl.ndarray((B, c_sz), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(k):
            nisa.dma_copy(dst=wtap, src=w_flat.ap(pattern=[[0, B], [k, c_sz]], offset=c0 * k + j))
            src_tap = win[0:B, 0:c_sz, j] if j < k1 else xnew
            nisa.tensor_tensor(dst=tap, data1=src_tap, data2=wtap, op=nl.multiply)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=tap, op=nl.add)

        # SiLU -> y[b, c0:c0+c_sz]
        sig = nl.ndarray((B, c_sz), dtype=nl.float32, buffer=nl.sbuf)
        _silu(dst=acc, data=acc, tmp=sig)
        nisa.dma_copy(dst=y_hbm[0:B, c0:c0 + c_sz], src=acc)

        # shifted window newwin = concat(win[:,:,1:], xnew) -> [B, c_sz, k1]
        newwin = nl.ndarray((B, c_sz, k1), dtype=nl.float32, buffer=nl.sbuf)
        if k1 > 1:
            nisa.tensor_copy(dst=newwin[0:B, 0:c_sz, 0:k1 - 1], src=win[0:B, 0:c_sz, 1:k1])
        nisa.tensor_copy(dst=newwin[0:B, 0:c_sz, k1 - 1:k1], src=xnew.reshape((B, c_sz, 1)))
        # WRITE shifted window to compact output (contiguous, NO indirect scatter).
        nisa.dma_copy(dst=newwin_hbm[0:B, c0:c0 + c_sz, 0:k1], src=newwin)

    return y_hbm, newwin_hbm
