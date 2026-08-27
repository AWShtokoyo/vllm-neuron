"""NKI kernel: SEQUENTIAL (recurrent) Gated Delta Rule for GDN prefill.

Reproduces the EXACT sequential recurrence of `_recurrent_gated_delta_rule`
(model.py) / `_segmented_recurrent_gated_delta_rule` — the device-CLEAN, gsm8k-
validated path — but as ONE bounded NKI kernel (Nemotron-H mamba2_ssd pattern:
the T loop is a real hardware `sequential_range`, NOT a python unroll that traces
into a giant XLA HLO and hangs neuronx-cc).

WHY sequential (not chunked): the chunked reformulation ((I-M)^-1 dense) produces
bf16 values that differ from the sequential recurrence, perturbing MoE top-8
routing and over-running the expert-dispatch buffer (nrta-1006). Matching the
sequential VALUES by construction keeps routing identical to the known-good path.

Per head h, state S = [Dk, Dv] (Dk=Dv=128). For each timestep t:
    S      = S * g_t                       # scalar decay (g_t = exp(g[t]))
    kv_mem = k_t^T @ S            -> [1,Dv] # sum_k S[k,v]*k_t[k]
    delta  = (v_t - kv_mem)*beta_t -> [1,Dv]
    S      = S + k_t (x) delta              # outer product accumulate
    out_t  = q_t^T @ S           -> [1,Dv]
Inputs arrive head-major fp32 [H, T, D] (B==1 prefill), q already l2norm'd+scaled,
k l2norm'd, g/beta raw (g exp'd inside). Matches the wrapper prologue.

nc_matmul(dst[M,N], stationary[K,M], moving[K,N]) = stationary^T @ moving,
K = contraction = partition axis of BOTH operands.
"""
import nki
import nki.isa as nisa
import nki.language as nl


def kernel_assert(condition, message=""):
    assert condition, message


# Width of the decay-broadcast matmul tiling (see SAN-SEQ-TILE-FIX below). 512 is a
# hardware limit in two places at once: nc_matmul's moving free dimension, and the fp32
# column capacity of a PSUM bank.
_MM_FREE_MAX_DEFAULT = 512


def gdn_seq_prefill_kernel(q_h, k_h, v_h, g_h, beta_h, H, T, D, init_state_h=None, HAS_INIT=False):
    """Sequential gated-delta-rule prefill over all heads.

    NOTE: NOT decorated with @nki.jit — the module wraps it via nki.jit()(...) + the
    repo's wrap_nki below (mirrors gated_delta_rule.py). Using @nki.jit directly hits a
    Dynamo trace break (setattr on func._nki_compile_cache, gb0059) / the FakeTensor wall.

    Structural dims H, T, D are passed as COMPILE-TIME Python int constants (NOT derived
    from .shape inside the kernel — NKI traces .shape as symbolic proxies, so range()/loop
    trip counts computed from them fail to resolve).

    Args:
        q_h, k_h, v_h: [H, T, D] fp32 head-major (q pre-scaled, q/k l2norm'd).
        g_h:    [H, T] fp32 raw decay (exp applied inside).
        beta_h: [H, T] fp32.
        H, T, D: ints.
    Returns:
        out_hDT: [H, D, T] fp32 per-step outputs (D-major; host transposes to [H,T,D]).
        state_h: [H, D, D] fp32 final recurrent state (S[k,v]).
    """
    kernel_assert(D <= 128, "D must be <= 128 (partition bound)")

    # Output kept [H, D, T] (D<=128 innermost-friendly). A dma_transpose of the [D,T]
    # out_col would need src innermost = T (512) > 128 (illegal); store D-major here and
    # transpose on the HOST (cheap, outside the compiled graph).
    out_hDT = nl.ndarray((H, D, T), dtype=nl.float32, buffer=nl.shared_hbm)
    state_h = nl.ndarray((H, D, D), dtype=nl.float32, buffer=nl.shared_hbm)

    kernel_assert(T <= 32767, "T must fit the SBUF free dim")

    for h in nl.affine_range(H):
        # ---- load this head's sequences into SBUF ----
        # ALL resident tiles keep D (<=128) on the PARTITION axis and T on the FREE axis
        # (T can be 512+, which is illegal on the partition axis but fine on free <=32767).
        # Column-major [D, T]: column t = [D,1] tile (contraction axis = D) for matmul.
        kT = nl.ndarray((D, T), dtype=nl.float32, buffer=nl.sbuf)   # k as [Dk, T]
        qT = nl.ndarray((D, T), dtype=nl.float32, buffer=nl.sbuf)   # q as [Dk, T]
        nisa.dma_transpose(dst=kT, src=k_h[h, 0:T, 0:D])
        nisa.dma_transpose(dst=qT, src=q_h[h, 0:T, 0:D])
        # g/beta as [1, T]
        grow = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
        brow = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=grow, src=g_h[h:h + 1, 0:T])
        nisa.dma_copy(dst=brow, src=beta_h[h:h + 1, 0:T])
        gexp = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=gexp, data=grow, op=nl.exp)
        # Broadcast exp(g) across the D PARTITION rows -> [D, T], so the per-step decay is a
        # [D,1] COLUMN (one value per partition of S). tensor_scalar does NOT broadcast a
        # [1,1] tile across partitions (MLIR: operand0 partition 1 != dst partition 128); a
        # per-partition [D,1] operand is required. ones[D,1] @ gexp[1,T] -> [D,T] (matmul
        # broadcast, contraction K=1, no DGE). Done ONCE per head, sliced per step.
        ones_D1 = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_D1, value=1.0)
        gD = nl.ndarray((D, T), dtype=nl.float32, buffer=nl.sbuf)
        # SAN-SEQ-TILE-FIX (upstream fork PR#2): two gen3/trn2 limits force tiling this decay
        # broadcast when T>512:
        #   (1) nc_matmul moving free-dim <= 512 -> "Matmul moving free dimension T exceeds 512".
        #   (2) a PSUM bank holds <= 512 fp32 columns, so a [D,T>512] PSUM tile is illegal.
        # It is a pure K=1 broadcast (ones[1,D]^T @ gexp[1,T] -> [D,T]), so computing it in
        # <=512-wide T slices through a 512-wide PSUM scratch and copying each slice into the
        # SBUF gD[D,T] (SBUF free dim up to 32767) is numerically identical. At T<=512 the loop
        # runs exactly once -> byte-identical graph for the shipped kv_segment_size=512 path.
        _MM_FREE_MAX = _MM_FREE_MAX_DEFAULT
        for _t0 in range(0, T, _MM_FREE_MAX):
            _tw = min(_MM_FREE_MAX, T - _t0)
            _gD_ps = nl.ndarray((D, _tw), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=_gD_ps, stationary=ones_D1,
                           moving=gexp[0:1, nl.ds(_t0, _tw)])
            nisa.tensor_copy(dst=gD[0:D, nl.ds(_t0, _tw)], src=_gD_ps)

        # recurrent state S [Dk, Dv] on partition=Dk. APC: seed from init_state_h[h] when provided
        # (a cache-hit prefix's saved boundary state), else zero. HAS_INIT is a compile-time flag so
        # the non-APC graph keeps the plain memset (no init input, byte-identical).
        S = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
        if HAS_INIT:
            nisa.dma_copy(dst=S, src=init_state_h[h, 0:D, 0:D])
        else:
            nisa.memset(dst=S, value=0.0)

        out_col = nl.ndarray((D, T), dtype=nl.float32, buffer=nl.sbuf)  # [Dv, T] outputs (col t)

        for t in nl.sequential_range(T):
            g_col = gD[0:D, nl.ds(t, 1)]     # [D,1] per-partition decay (broadcast-legal)
            bt = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=bt, src=brow[0:1, nl.ds(t, 1)])

            k_col = kT[0:D, nl.ds(t, 1)]      # [Dk,1]  (contraction column)
            q_col = qT[0:D, nl.ds(t, 1)]      # [Dk,1]
            # per-step [1,D] rows loaded directly from HBM (partition=1, legal) — avoids
            # both a [T,D] resident tile (T>128 illegal on partition) and a [128,1] transpose
            # (Vector-engine transpose capped at 32x32).
            k_row = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
            v_row = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_row, src=k_h[h, nl.ds(t, 1), 0:D])
            nisa.dma_copy(dst=v_row, src=v_h[h, nl.ds(t, 1), 0:D])

            # S *= g_t  (g_col is [D,1] per-partition -> broadcasts over S's free dim)
            nisa.tensor_scalar(dst=S, data=S, op0=nl.multiply, operand0=g_col)

            # kv_mem = k_t^T @ S -> [1, Dv]   (stationary=k_col[Dk,1], moving=S[Dk,Dv])
            kv_ps = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=kv_ps, stationary=k_col, moving=S)
            kv = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=kv, src=kv_ps)

            # delta = (v_row - kv_mem) * beta_t  -> [1, Dv]
            delta = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=delta, data1=v_row, data2=kv, op=nl.subtract)
            nisa.tensor_scalar(dst=delta, data=delta, op0=nl.multiply, operand0=bt)

            # S += k_t (x) delta   (outer product [Dk,Dv]; contraction K=1)
            # nc_matmul(stationary=k_row[1,Dk], moving=delta[1,Dv]) -> [Dk,Dv]
            outer_ps = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=outer_ps, stationary=k_row, moving=delta)
            outer = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=outer, src=outer_ps)
            nisa.tensor_tensor(dst=S, data1=S, data2=outer, op=nl.add)

            # out_t = S^T @ q_t -> [Dv, 1]  (stationary=S[Dk,Dv], moving=q_col[Dk,1])
            # written into out_col[:, t] as a [Dv,1] column (no [T,*] partition tile).
            out_ps = nl.ndarray((D, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=out_ps, stationary=S, moving=q_col)
            nisa.tensor_copy(dst=out_col[0:D, nl.ds(t, 1)], src=out_ps)

        # store outputs D-major [Dv,T] (no transpose) and final state S [Dk,Dv]
        nisa.dma_copy(dst=out_hDT[h, 0:D, 0:T], src=out_col)
        nisa.dma_copy(dst=state_h[h, 0:D, 0:D], src=S)

    return out_hDT, state_h


# Pre-JIT at module load (NOT @nki.jit on the def) so wrap_nki can fold it at trace time —
# calling wrap_nki INSIDE the forward hits the fake-tensor wall (NKI V3 no FakeTensors).
_gdn_seq_jit = nki.jit()(gdn_seq_prefill_kernel)

# Wrap EAGERLY at module load (NOT lazily inside run_gdn_seq_prefill). A lazy
# `if _gdn_seq_wrapped is None: _gdn_seq_wrapped = wrap_nki(...)` branch inside the
# traced forward installs a Dynamo guard on this global AND mutates it mid-trace: warmup
# bakes the is-None branch, then the first real call flips the guard and Dynamo attempts a
# recompile, which vLLM rejects under torch.compile stance 'fail_on_recompile'
# (RuntimeError / EngineDeadError). Module-level wrapping keeps the forward branch-free —
# same rule the SLOT kernels follow in model.py. This module is only imported on the
# sequential GDN prefill path (VLLM_GDN_PREFILL=sequential, the default), so the eager
# wrap_nki (heavy/device import) never runs when the parallel kernel is selected instead.
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
_gdn_seq_wrapped = wrap_nki(_gdn_seq_jit)
# APC-seeded variant (HAS_INIT=True path): separate pre-wrap so the seeded graph (with the extra
# init_state_h input + DMA-load of S) is distinct from the plain graph and neither recompiles.
_gdn_seq_wrapped_init = wrap_nki(_gdn_seq_jit)


def run_gdn_seq_prefill(q_h, k_h, v_h, g_h, beta_h, init_state_h=None):
    """Host entry: invoke the sequential GDN NKI kernel via the repo's wrap_nki (LNC=2).

    Structural dims (H, T, D) are computed HERE as real Python ints from the host tensor
    shapes and passed as compile-time constants — the kernel must NOT derive them from .shape.

    APC: when init_state_h [H,D,D] is provided (a cache-hit prefix's saved boundary recurrent
    state), the kernel seeds S from it instead of zero. Routed to a SEPARATE pre-wrapped kernel
    variant (HAS_INIT=True) so both graphs stay branch-free (fail_on_recompile-safe); the seeded
    variant takes the extra input, the plain one does not.
    """
    H, T, D = int(q_h.shape[0]), int(q_h.shape[1]), int(q_h.shape[2])
    # grid=(1,): this kernel is single-program (loops all heads internally via affine_range).
    if init_state_h is not None:
        # init_state_h MUST be keyword: the 6th positional slot is `H` (arg order
        # q_h,k_h,v_h,g_h,beta_h,H,T,D,init_state_h,HAS_INIT), so passing it positionally
        # would bind it to H and leave init_state_h at its None default (HAS_INIT=True but
        # no seed -> NoneType subscript in the kernel).
        out_hDT, state_h = _gdn_seq_wrapped_init[1](
            q_h, k_h, v_h, g_h, beta_h, H=H, T=T, D=D,
            init_state_h=init_state_h, HAS_INIT=True)
    else:
        out_hDT, state_h = _gdn_seq_wrapped[1](q_h, k_h, v_h, g_h, beta_h, H=H, T=T, D=D)
    # kernel returns [H, D, T] (D-major, to avoid an illegal in-kernel dma_transpose of a
    # 512-wide innermost); transpose to [H, T, D] on the host (cheap, outside the graph).
    out_h = out_hDT.transpose(1, 2).contiguous()
    return out_h, state_h
