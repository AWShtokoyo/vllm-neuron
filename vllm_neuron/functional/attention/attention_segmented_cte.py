# SPDX-License-Identifier: Apache-2.0
import nki
import torch

from typing import Optional
from torch import Tensor

from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki
from vllm_neuron.utils.bucket_utils import SUPPORTED_KV_SEGMENT_SIZES

# Initialize segmented attention kernel at module import time
_wrapped_attention_segmented_cte = None
# Whether the installed nkilib segmented kernel accepts the fp8_packed kwarg.
# The packed read path is on nkilib mainline but may not be in the consumed
# version set yet; gate on the live signature so passing fp8_packed never
# breaks on an older kernel.
# TODO: remove this capability gate (and forward fp8_packed unconditionally)
# once the consumed nkilib image ships the segmented-prefill fp8_packed path.
_SEGMENTED_KERNEL_HAS_FP8_PACKED = False
try:
    import inspect as _inspect

    from nkilib_src.nkilib.core.attention.attention_segmented_cte import (
        attention_segmented_cte,
    )

    _SEGMENTED_KERNEL_HAS_FP8_PACKED = (
        "fp8_packed" in _inspect.signature(attention_segmented_cte).parameters
    )
    _attention_segmented_cte_jit = nki.jit()(attention_segmented_cte)
    _wrapped_attention_segmented_cte = wrap_nki(_attention_segmented_cte_jit)
except Exception:
    pass

# Maximum head dimension supported by the NKI kernel. nkilib's
# attention_segmented_cte asserts `head_dim <= 512` and tiles the head
# dimension internally; head dims above 256 additionally require
# `prior_seg_size <= 2048`.
_MAX_HEAD_DIM = 512


def _decode_packed_to_segmented_packed(k_cache: Tensor) -> Tensor:
    """Convert a decode-packed K cache to the segmented kernel's packed layout.

    The two packed FP8 layouts differ in how heads sit relative to the
    token-paired axis:

    - decode (attention_block_tkg):
        ``[num_blocks, kv_heads, block_size // 2, head_dim, 2]``
        (the shape the model stores in ``self.k_cache``; heads are a separate
        dim and token pairs are interleaved per head)
    - segmented (attention_segmented_cte):
        ``[num_blocks, block_size // 2, kv_heads * head_dim, 2]``
        (heads flattened into the channel axis alongside head_dim)

    These are different memory orderings, not a reshape/view, so convert by
    moving the head dim next to head_dim before the size-2 token-pair axis.
    """
    num_blocks, kv_heads, half, head_dim, two = k_cache.shape
    assert two == 2, f"packed K last dim must be 2, got {two}"
    # [nb, kv_heads, half, head_dim, 2] -> [nb, half, kv_heads, head_dim, 2]
    #                                   -> [nb, half, kv_heads * head_dim, 2]
    return (
        k_cache.permute(0, 2, 1, 3, 4)
        .reshape(num_blocks, half, kv_heads * head_dim, two)
        .contiguous()
    )


def _segmented_packed_to_unpacked(k_cache: Tensor, num_kv_heads: int) -> Tensor:
    """Un-swizzle a segmented-packed K cache to the standard block layout.

    ``[num_blocks, block_size // 2, kv_heads * head_dim, 2]``  ->
    ``[num_blocks, kv_heads, block_size, head_dim]``

    Used by the CPU fallback, which gathers from a standard-layout cache.
    """
    num_blocks, half, kv_dim, two = k_cache.shape
    assert two == 2, f"packed K last dim must be 2, got {two}"
    head_dim = kv_dim // num_kv_heads
    # [nb, half, kv_dim, 2] -> [nb, half, 2, kv_dim] -> [nb, block_size, kv_dim]
    flat = k_cache.transpose(2, 3).reshape(num_blocks, half * 2, kv_dim)
    # [nb, block_size, kv_dim] -> [nb, block_size, kv_heads, head_dim]
    #                          -> [nb, kv_heads, block_size, head_dim]
    return (
        flat.reshape(num_blocks, half * 2, num_kv_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _can_use_segmented_kernel(q: Tensor) -> bool:
    """
    Check if the segmented attention NKI kernel can be used.

    Returns False if the kernel is unavailable or the device is CPU
    without the NKI simulator enabled.
    """
    if _wrapped_attention_segmented_cte is None:
        return False
    if not can_run_kernel(q):
        return False
    return True


def _torch_segmented_attention_impl(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    prior_tokens: Tensor,
    block_size: int,
    kv_segment_size: int,
    scale: float,
    tp_q: bool = True,
    tp_out: bool = False,
    sliding_window: Optional[int] = None,
    sink: Optional[Tensor] = None,
) -> Tensor:
    """
    Pure PyTorch fallback for segmented attention (CPU mode).

    Gathers K/V from the paged block cache and computes standard scaled
    dot-product attention with a causal + out-of-range mask over prior +
    active tokens.

    This implementation is dynamo-traceable under ``torch.compile(..., fullgraph=True)``:
      - ``prior_tokens`` stays a tensor (no ``.item()``); it broadcasts into
        position masks.
      - Shapes are fully static. Instead of trimming to ``prior_len + S_q``,
        the gather walks the full ``max_blocks_per_seq * block_size`` span;
        positions past ``prior_len + S_q`` are masked out of the softmax.
      - No data-dependent Python branches. ``sliding_window`` and ``sink``
        presence are Python-constant kwargs — dynamo resolves the ``if``
        branches at trace time and produces per-specialization graphs.

    Args:
        q: Query tensor [B, S_q, D] (tp_q=True) or [B, D, S_q] (tp_q=False)
        k_cache: Paged key cache [num_blocks, num_kv_heads, block_size, D]
        v_cache: Paged value cache [num_blocks, num_kv_heads, block_size, D]
        block_tables: Block table [B_kv, max_blocks_per_seq].
        prior_tokens: Number of prior cached tokens, shape [B, 1].
        block_size: Block size (Python int; static at trace time)
        kv_segment_size: Segment size (unused here; full padded gather is done)
        scale: Attention scaling factor
        tp_q: If True, q is [B, S_q, D]; if False, q is [B, D, S_q]
        tp_out: If True, output is [B, D, S_q]; if False, output is [B, S_q, D]
        sliding_window: Window size for local attention. None or 0 means full attention.
        sink: Attention sink bias tensor [B, 1]. Appended as extra column to
            attention scores before softmax, then dropped before V matmul.
    """
    # Normalize Q layout to [B, S_q, D]
    if not tp_q:
        q = q.transpose(1, 2)

    B, S_q, D = q.shape
    num_kv_heads = k_cache.shape[1]
    max_blocks_per_seq = block_tables.shape[1]
    # Full padded KV length we gather over. Static at trace time.
    padded_kv_len = max_blocks_per_seq * block_size

    # prior_tokens as a scalar tensor. Kept as tensor (no .item()) so dynamo
    # can trace the downstream mask arithmetic.
    prior_len_t = prior_tokens.reshape(-1)[0].to(torch.int64)

    # Gather K/V from the paged block cache across the full block-table span.
    # Unused slots (0 in production, -1 in tests) are clamped to index 0; the
    # in-sequence position mask below zeroes out their softmax contribution.
    bt = block_tables[0]  # [max_blocks_per_seq]
    bt_clamped = bt.clamp_min(0).to(torch.int64)

    # k_cache[bt_clamped]: [max_blocks_per_seq, num_kv_heads, block_size, D]
    k_blocks = k_cache[bt_clamped]
    v_blocks = v_cache[bt_clamped]

    # Flatten blocks into a continuous sequence: [num_kv_heads, padded_kv_len, D]
    k_seq = k_blocks.permute(1, 0, 2, 3).reshape(num_kv_heads, padded_kv_len, D)
    v_seq = v_blocks.permute(1, 0, 2, 3).reshape(num_kv_heads, padded_kv_len, D)

    # Handle GQA: expand KV heads to match Q heads. B and num_kv_heads are
    # Python-int shapes known at trace time, so this branch is safe for dynamo.
    heads_per_kv = B // num_kv_heads
    if heads_per_kv > 1:
        k_seq = k_seq.repeat_interleave(heads_per_kv, dim=0)  # [B, padded_kv_len, D]
        v_seq = v_seq.repeat_interleave(heads_per_kv, dim=0)

    # Build a single [S_q, padded_kv_len] boolean mask of *allowed* positions:
    #   1. within sequence       : k_pos < prior_len + S_q
    #   2. causal                : k_pos <= q_pos + prior_len  (q_pos in [0, S_q))
    #   3. sliding window (opt.) : k_pos > q_pos + prior_len - sliding_window
    device = q.device
    q_pos = torch.arange(S_q, device=device, dtype=torch.int64).unsqueeze(1)  # [S_q, 1]
    k_pos = torch.arange(padded_kv_len, device=device, dtype=torch.int64).unsqueeze(
        0
    )  # [1, padded_kv_len]
    total_kv_len_t = prior_len_t + S_q  # scalar tensor

    in_seq = k_pos < total_kv_len_t  # [1, padded_kv_len]
    causal = k_pos <= (q_pos + prior_len_t)  # [S_q, padded_kv_len]
    allowed = in_seq & causal  # [S_q, padded_kv_len]
    if sliding_window is not None and sliding_window > 0:
        sw = k_pos > (q_pos + prior_len_t - sliding_window)
        allowed = allowed & sw

    q_f = q.float()
    k_seq_f = k_seq.float()
    v_seq_f = v_seq.float()

    # scores: [B, S_q, padded_kv_len]
    scores = torch.bmm(q_f, k_seq_f.transpose(1, 2)) * scale

    # Mask out disallowed positions. allowed broadcasts [1, S_q, padded_kv_len].
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))

    # Sink: append learnable bias column before softmax, drop after.
    if sink is not None:
        sink_f = sink.float().reshape(B, 1, 1).expand(B, S_q, 1)
        scores = torch.cat([scores, sink_f], dim=-1)

    attn_weights = torch.nn.functional.softmax(scores, dim=-1)

    if sink is not None:
        attn_weights = attn_weights[:, :, :-1]

    # Rows where no positions are allowed (e.g. purely padded queries) softmax
    # to NaN; zero them so the matmul contributes nothing.
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    output = torch.bmm(attn_weights, v_seq_f).to(q.dtype)

    if tp_out:
        output = output.transpose(1, 2)  # [B, D, S_q]

    return output


def _torch_segmented_attention_cp_impl(
    q: Tensor,
    k_local: Tensor,
    v_local: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    prior_tokens: Tensor,
    block_size: int,
    cp_rank: int,
    cp_world_size: int,
    cp_kv_cache_interleave_size: int,
    cp_group,
    scale: float,
    tp_q: bool = True,
    tp_out: bool = False,
) -> Tensor:
    """
    Pure PyTorch fallback for Gather-Q segmented attention with context parallelism.

    Q has been AllGathered to contain all S positions. Each rank has local KV:
      - Prior: cached S_prior/DCP tokens at interleaved positions
      - Current: S/DCP tokens for this rank's owned positions

    Each rank computes full_Q × local_KV (partial), then LSE-corrects across
    DCP ranks and reduces back to this rank's Q slice.

    The causal mask maps Q global positions to local KV global positions:
      - Prior cache slot s → global pos: (s//I)*(W*I) + R*I + (s%I)
      - Current local token j → global pos: prior_global + R*(S/DCP) + j
        (contiguous block assigned to this cp_rank)
    """
    if not tp_q:
        q = q.transpose(1, 2)

    B, S_total, D = q.shape  # B=Nh_q, S_total = full gathered Q length
    num_kv_heads = k_local.shape[0]
    S_local = k_local.shape[1]  # S_total / cp_world_size
    max_blocks_per_seq = block_tables.shape[1]
    padded_kv_len = max_blocks_per_seq * block_size

    prior_local_t = prior_tokens.reshape(-1)[0].to(torch.int64)
    prior_global_t = prior_local_t * cp_world_size

    device = q.device
    I = cp_kv_cache_interleave_size
    W = cp_world_size
    R = cp_rank

    # GQA expansion
    heads_per_kv = B // num_kv_heads
    if heads_per_kv > 1:
        k_local_exp = k_local.repeat_interleave(heads_per_kv, dim=0)
        v_local_exp = v_local.repeat_interleave(heads_per_kv, dim=0)
    else:
        k_local_exp = k_local
        v_local_exp = v_local

    # ── Gather prior KV from cache (static padded shape) ──
    bt = block_tables[0].clamp_min(0).to(torch.int64)
    k_blocks = k_cache[bt]
    v_blocks = v_cache[bt]
    k_prior = k_blocks.permute(1, 0, 2, 3).reshape(num_kv_heads, padded_kv_len, D)
    v_prior = v_blocks.permute(1, 0, 2, 3).reshape(num_kv_heads, padded_kv_len, D)

    if heads_per_kv > 1:
        k_prior = k_prior.repeat_interleave(heads_per_kv, dim=0)
        v_prior = v_prior.repeat_interleave(heads_per_kv, dim=0)

    # ── Concatenate local KV: [prior_padded, current_local] ──
    k_full_local = torch.cat([k_prior, k_local_exp], dim=1)  # [B, padded+S_local, D]
    v_full_local = torch.cat([v_prior, v_local_exp], dim=1)
    total_local_kv = padded_kv_len + S_local

    # ── Build causal+validity mask [S_total, total_local_kv] ──
    # Q global positions: [prior_global, prior_global + S_total)
    q_global = prior_global_t + torch.arange(S_total, device=device, dtype=torch.int64)

    # Prior KV global positions (interleaved): slot s → (s//I)*(W*I) + R*I + (s%I)
    prior_slot = torch.arange(padded_kv_len, device=device, dtype=torch.int64)
    prior_global_pos = (prior_slot // I) * (W * I) + R * I + (prior_slot % I)

    # Current KV global positions: this rank owns interleaved positions
    # Token j on rank R has global pos: prior_global + (j//I)*(W*I) + R*I + (j%I)
    cur_j = torch.arange(S_local, device=device, dtype=torch.int64)
    cur_global_pos = prior_global_t + (cur_j // I) * (W * I) + R * I + (cur_j % I)

    # Fold validity into positions: invalid prior slots get a position larger than
    # any Q position so the <= comparison naturally returns False.
    # This avoids a broadcast boolean AND which triggers a Neuron compiler bug
    # at certain tensor shapes (e.g. [4096, 5120] & [1, 5120]).
    # TODO: triage the bug(CHRS-985)
    # INVALID_POS > q_global.max() = prior_global_t + S_total - 1
    #            <= prior_local_t * W + S_total - 1 < padded_kv_len * W + S_total
    INVALID_POS = padded_kv_len * W + S_total + 1
    prior_valid_mask = prior_slot < prior_local_t
    prior_global_pos_masked = torch.where(
        prior_valid_mask,
        prior_global_pos,
        torch.full_like(prior_global_pos, INVALID_POS),
    )
    kv_global_pos = torch.cat(
        [prior_global_pos_masked, cur_global_pos]
    )  # [total_local_kv]

    # Combined causal + validity mask in a single comparison
    allowed = kv_global_pos.unsqueeze(0) <= q_global.unsqueeze(
        1
    )  # [S_total, total_local_kv]

    # ── Compute local partial attention ──
    q_f = q.float()
    scores = torch.bmm(q_f, k_full_local.float().transpose(1, 2)) * scale
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))

    lse_local = torch.logsumexp(scores, dim=-1)  # [B, S_total]
    w = torch.nn.functional.softmax(scores, dim=-1)
    w = torch.nan_to_num(w, nan=0.0)
    out_local = torch.bmm(w, v_full_local.float())  # [B, S_total, D]

    # ── LSE correction across DCP ranks ──
    # AllGather LSE: [W*B, S_total] → [W, B, S_total]
    all_lse = cp_group.all_gather(lse_local.contiguous(), dim=0)
    all_lse = all_lse.view(W, B, S_total)
    global_lse = torch.logsumexp(all_lse, dim=0)  # [B, S_total]

    # Weight local output and ReduceScatter to get each rank's Q slice
    local_weight = torch.exp(lse_local - global_lse)  # [B, S_total]
    local_weight = torch.nan_to_num(local_weight, nan=0.0)
    weighted_out = out_local * local_weight.unsqueeze(-1)  # [B, S_total, D]

    # ReduceScatter on dim=1: splits S_total into W chunks, sums across ranks,
    # each rank receives its chunk (the Q positions it owns).
    output = cp_group.reduce_scatter(
        weighted_out.contiguous(), dim=1
    )  # [B, S_local, D]

    output = output.to(q.dtype)
    if tp_out:
        output = output.transpose(1, 2)

    return output


def _can_use_segmented_cp_kernel(q: Tensor) -> bool:
    """Check if the segmented attention CP NKI kernel can be used.

    TODO: Return True once the NKI kernel for segmented CP is implemented.
    """
    return False


def segmented_attention_cp(
    q: Tensor,
    k_local: Tensor,
    v_local: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    prior_tokens: Tensor,
    block_size: int,
    cp_rank: int,
    cp_world_size: int,
    cp_kv_cache_interleave_size: int,
    cp_group,
    scale: Optional[float] = None,
    tp_q: bool = True,
    tp_out: bool = False,
) -> Tensor:
    """
    Segmented Attention API for Context Parallelism (Gather-Q approach).

    Q has been AllGathered to contain all S positions. Each rank computes
    full_Q × local_KV (prior cache + current local), producing partial
    attention. LSE correction across DCP ranks + ReduceScatter returns
    each rank's output slice.

    Each DCP rank caches only its owned tokens (S/DCP per chunk) at interleaved
    positions. The local KV for attention is: prior from cache + current from
    this rank's projection.

    Args:
        q: AllGathered query [B, S_total, D] (tp_q=True) or [B, D, S_total].
        k_local: This rank's current chunk keys [Nh_kv, S_local, D].
        v_local: This rank's current chunk values [Nh_kv, S_local, D].
        k_cache: Paged key cache [num_blocks, num_kv_heads, block_size, D].
        v_cache: Paged value cache [num_blocks, num_kv_heads, block_size, D].
        block_tables: Block table [B_kv, max_blocks_per_seq].
        prior_tokens: Local cached token count [B, 1] (= global_prior / cp_world_size).
        block_size: Cache block size.
        cp_rank: This rank's CP index (determines owned positions).
        cp_world_size: Total CP ranks (DCP degree).
        cp_kv_cache_interleave_size: Interleave granularity for cache slot mapping.
        cp_group: Process group for DCP communication (AllGather LSE, ReduceScatter).
        scale: Scaling factor. Default: 1/sqrt(d_head).
        tp_q: Query transpose flag.
        tp_out: Output transpose flag.

    Returns:
        Output tensor [B, D, S_local] (tp_out=True) or [B, S_local, D].

    Note:
        TODO: NKI kernel integration. Currently uses PyTorch fallback only.
    """
    d_head = q.shape[2] if tp_q else q.shape[1]

    if d_head > _MAX_HEAD_DIM:
        raise ValueError(
            f"head_dim={d_head} exceeds maximum supported head dimension "
            f"({_MAX_HEAD_DIM}). Requires head_dim <= {_MAX_HEAD_DIM}."
        )

    if scale is None:
        scale = 1.0 / (d_head**0.5)

    if _can_use_segmented_cp_kernel(q):
        # TODO: Route to NKI kernel once available.
        pass

    return _torch_segmented_attention_cp_impl(
        q=q,
        k_local=k_local,
        v_local=v_local,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        prior_tokens=prior_tokens,
        block_size=block_size,
        cp_rank=cp_rank,
        cp_world_size=cp_world_size,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        cp_group=cp_group,
        scale=scale,
        tp_q=tp_q,
        tp_out=tp_out,
    )


def segmented_attention(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    prior_tokens: Tensor,
    block_size: int,
    kv_segment_size: int,
    scale: Optional[float] = None,
    tp_q: bool = True,
    tp_out: bool = False,
    sliding_window: Optional[int] = None,
    sink: Optional[Tensor] = None,
    fp8_packed: bool = False,
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
) -> Tensor:
    """
    Segmented Attention API using the attention_segmented_cte NKI kernel.

    This implements segmented prefill attention: the query attends to prior cached
    KV in fixed-size segments, enabling efficient chunked prefill with block-based
    KV cache. Computes: softmax(scale * Q @ K_cache^T + mask) @ V_cache

    Input Layouts (controlled by transpose flags):
        q: Query tensor
           - [B, S_q, D] when tp_q=True (default)
           - [B, D, S_q] when tp_q=False

    Output Layout:
        - [B, D, S_q] if tp_out=True
        - [B, S_q, D] if tp_out=False (default)

    Dimensions:
        B: Batch size (can include num_heads for multi-head attention)
        S_q: Query sequence length
        D: Head dimension (max 512)

    Args:
        q: Query tensor
        k_cache: Block KV cache for keys [num_blocks, num_kv_heads, block_size, D]
        v_cache: Block KV cache for values [num_blocks, num_kv_heads, block_size, D]
        block_tables: Block table mapping sequences to cache blocks [B_kv, max_blocks_per_seq]
        prior_tokens: Number of prior cached tokens, shape [B, 1]. Must be multiple of block_size.
        block_size: Size of each block in the KV cache
        kv_segment_size: Segment size for iterative prior KV processing
        scale: Scaling factor for attention scores. Default: 1/sqrt(d_head).
               Must be 1.0 when using sliding_window, prefix caching, or context parallel.
        tp_q: Query transpose flag. True means Q is [B, S, D]. Default: True
        tp_out: Output transpose flag. True means output is [B, D, S]. Default: False
        sliding_window: Window size for local attention. None or 0 means full attention. Default: None
        sink: Attention sink tensor [B, 1] for streaming/infinite context. Default: None
        fp8_packed: When True, k_cache uses the swizzled packed FP8 layout
            [num_blocks, block_size // 2, num_kv_heads * head_dim, 2]; the kernel
            views it as BF16 to DMA-transpose, then reinterprets back to FP8. V
            is never packed. Default: False
        k_scale: FP8 K dequant scale [PMAX, 1], applied in-kernel. Default: None
        v_scale: FP8 V dequant scale [PMAX, 1], applied in-kernel. Default: None

    Returns:
        Output tensor with attention results.

    Raises:
        ValueError: If any kernel constraint is violated:
            - head_dim > 512
            - kv_segment_size not in SUPPORTED_KV_SEGMENT_SIZES
            - kv_segment_size not divisible by block_size
            - sliding_window not divisible by block_size (when set)
            - seqlen_q != kv_segment_size (temporary constraint)
        RuntimeError: If the segmented attention NKI kernel is not available and
            there is no torch fallback implementation.

    Note:
        The segmented kernel does not yet support tp_q=False or tp_out=True natively.
        When these flags are set, transposing is handled at the boundary before/after
        the kernel call.

        fp8_packed: the caller passes the cache in the decode-packed layout
        ([num_blocks, kv_heads, block_size // 2, head_dim, 2]) — the single
        shared K cache the runner allocates. It is converted here to the
        segmented kernel's packed layout
        ([num_blocks, block_size // 2, kv_heads * head_dim, 2]).
    """
    # The K cache is stored in the decode-packed layout (the shared cache the
    # runner allocates); convert it to the segmented kernel's packed layout once
    # here so both the kernel and the CPU fallback consume a consistent shape.
    num_kv_heads = v_cache.shape[1]
    if fp8_packed:
        k_cache = _decode_packed_to_segmented_packed(k_cache)

    if not _can_use_segmented_kernel(q):
        if scale is None:
            d_head = q.shape[2] if tp_q else q.shape[1]
            scale = 1.0 / (d_head**0.5)
        # The CPU fallback gathers from a standard-layout cache, so un-swizzle
        # the (now segmented-packed) K cache to [nb, kv_heads, block_size,
        # head_dim] first. K-scale dequant is fused into the softmax scale by
        # the caller (matching the on-device path), so the fallback consumes
        # scales the same way regardless of packing.
        if fp8_packed:
            k_cache = _segmented_packed_to_unpacked(k_cache, num_kv_heads)
        return _torch_segmented_attention_impl(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            prior_tokens=prior_tokens,
            block_size=block_size,
            kv_segment_size=kv_segment_size,
            scale=scale,
            tp_q=tp_q,
            tp_out=tp_out,
            sliding_window=sliding_window,
            sink=sink,
        )

    # --- Validate kernel constraints ---
    # These mirror the kernel_assert checks in the NKI kernel

    # Extract dimensions (layout depends on tp_q)
    seqlen_q = q.shape[1] if tp_q else q.shape[2]
    d_head = q.shape[2] if tp_q else q.shape[1]

    # 1. head_dim must not exceed what the kernel tiles (see _MAX_HEAD_DIM)
    if d_head > _MAX_HEAD_DIM:
        raise ValueError(
            f"head_dim={d_head} exceeds maximum supported head dimension "
            f"({_MAX_HEAD_DIM}). The segmented attention kernel requires "
            f"head_dim <= {_MAX_HEAD_DIM}."
        )

    # 2. kv_segment_size must be a supported size
    if kv_segment_size not in SUPPORTED_KV_SEGMENT_SIZES:
        raise ValueError(
            f"kv_segment_size={kv_segment_size} is not supported. "
            f"Supported sizes: {sorted(SUPPORTED_KV_SEGMENT_SIZES)}."
        )

    # 3. kv_segment_size must be divisible by block_size
    if kv_segment_size % block_size != 0:
        raise ValueError(
            f"kv_segment_size ({kv_segment_size}) must be divisible by "
            f"block_size ({block_size})."
        )

    # 4. sliding_window must be divisible by block_size when set
    if sliding_window is not None and sliding_window > 0:
        if sliding_window % block_size != 0:
            raise ValueError(
                f"sliding_window ({sliding_window}) must be divisible by "
                f"block_size ({block_size})."
            )

    # 5. Query sequence length must equal kv_segment_size.
    #    TODO: This is a temporary constraint. The kernel can be extended to
    #    support seqlen_q != prior_seg_size (e.g., smaller Q attending to a
    #    larger KV segment) once the active-segment tiling logic is decoupled
    #    from the query length.
    if seqlen_q != kv_segment_size:
        raise ValueError(
            f"Query sequence length ({seqlen_q}) must equal "
            f"kv_segment_size ({kv_segment_size}). The segmented kernel "
            f"currently requires seqlen_q == kv_segment_size."
        )

    # Compute default scale if not provided
    if scale is None:
        scale = 1.0 / (d_head**0.5)

    # Segmented kernel does not yet support tp_q=False or tp_out=True natively,
    # so we transpose at the boundary.
    q_seg = q.transpose(1, 2) if not tp_q else q
    q_seg = q_seg * scale

    # Only forward fp8_packed when the installed kernel supports it (the packed
    # read path may not be in the consumed nkilib version set yet). A packed
    # cache with an older kernel cannot be read correctly, so fail loudly rather
    # than silently misread.
    extra_kwargs = {}
    if _SEGMENTED_KERNEL_HAS_FP8_PACKED:
        extra_kwargs["fp8_packed"] = fp8_packed
    elif fp8_packed:
        raise RuntimeError(
            "fp8_packed segmented prefill requires an nkilib "
            "attention_segmented_cte with the fp8_packed parameter, which the "
            "installed kernel does not have. Bump nkilib to a version that "
            "ships the segmented packed FP8 read path."
        )

    result = _wrapped_attention_segmented_cte[2](
        q=q_seg,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        prior_tokens=prior_tokens,
        block_size=block_size,
        prior_seg_size=kv_segment_size,
        scale=1.0,
        tp_q=True,
        tp_out=False,
        sliding_window=sliding_window if sliding_window else None,
        sink=sink,
        num_q_heads=q_seg.shape[0],
        k_scale=k_scale,
        v_scale=v_scale,
        **extra_kwargs,
    )

    if tp_out:
        result = result.transpose(1, 2)

    return result
