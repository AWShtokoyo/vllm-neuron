# SPDX-License-Identifier: Apache-2.0
import torch
from typing import Optional

from torch import Tensor

import nki
import math

from nkilib.core.attention.gen_mask_tkg import gen_mask_tkg_hbm

jitted_gen_mask = nki.jit()(gen_mask_tkg_hbm)

from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

P_MAX = 128


def gen_attention_decode_mask(
    pos_ids: Tensor,
    bs: int,
    q_head: int,
    s_active: int,
    s_prior: int,
    start_pos: Optional[Tensor] = None,
    block_len: int = 0,
    local_filled_slots: Optional[Tensor] = None,
    dcp_active_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Generate an attention mask for token-generation (TKG).

    Automatically selects between the NKI ``gen_mask_tkg_hbm`` kernel and a
    PyTorch fallback based on hardware constraints and tensor placement.

    Args:
        pos_ids:      [1, bs * s_active]  Position IDs per-batchline.
        bs:           Batch size.
        q_head:       Number of query heads.
        s_active:     Active (new-token) sequence length.
        s_prior:      Prior (KV-cache) sequence length (must be divisible by 128
                      for the NKI path).
        start_pos:    Optional [1, bs * s_active]  Per-query SWA window start.
        block_len:    Block length for block-KV cache (0 = flat cache).
        local_filled_slots: Number of filled cache slots for DCP interleaved
                      caches. When provided, generates a flat mask (slots <
                      threshold are valid) bypassing the standard block-KV mask.
        active_mask:  [s_active, bs, q_head, s_active] active-token mask.
                      **Always required**.

    Returns:
        Tensor of shape [s_prior, bs, q_head, s_active] with values in {0, 1}.

    Raises:
        ValueError: If ``active_mask is None``.
    """
    # DCP interleaved cache: flat mask based on filled slot count.
    # Prior cached tokens fill slots 0..local_filled_slots-1.
    # The active token is placed at slot s_prior-1 by stage 7b on ALL ranks.
    # dcp_active_mask gates whether this rank includes the active slot
    # (only the owning rank should, to avoid double-counting in LSE correction).
    if local_filled_slots is not None:
        slot_idx = torch.arange(s_prior, device=pos_ids.device, dtype=torch.float32)
        prior_mask = slot_idx < local_filled_slots
        active_slots = slot_idx >= (s_prior - s_active)
        if dcp_active_mask is not None:
            active_slots = active_slots * dcp_active_mask
        mask = (prior_mask.float() + active_slots.float()).clamp(max=1.0)
        mask = (
            mask.view(s_prior, 1, 1, 1)
            .expand(s_prior, bs, q_head, s_active)
            .contiguous()
        )
        return mask

    # s_prior must be divisible by P_MAX
    if s_prior % P_MAX != 0:
        raise ValueError(f"s_prior ({s_prior}) must be divisible by {P_MAX}")

    # Build causal mask over the two s_active dimensions (dim 0 and dim 3)
    causal = torch.triu(
        torch.ones(s_active, s_active, dtype=torch.float32, device=pos_ids.device)
    )

    # Broadcast to [s_active, bs, q_head, s_active]
    active_mask = (
        causal[:, None, None, :].expand(s_active, bs, q_head, s_active).contiguous()
    )

    if _can_use_kernel(pos_ids, bs, s_active, s_prior):
        wrapped = wrap_nki(jitted_gen_mask)

        pos_ids = pos_ids.to(torch.float32)

        lnc = 2
        if s_prior % (2 * P_MAX) != 0:
            lnc = 1

        return wrapped[lnc](
            pos_ids,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            s_prior=s_prior,
            start_pos_hbm=start_pos,
            block_len=block_len,
            active_mask=active_mask,
        )

    return _torch_gen_attention_decode_mask_impl(
        pos_ids,
        bs=bs,
        q_head=q_head,
        s_active=s_active,
        s_prior=s_prior,
        start_pos=start_pos,
        block_len=block_len,
        active_mask=active_mask,
    )


def _can_use_kernel(
    pos_ids: Tensor,
    bs: int,
    s_active: int,
    s_prior: int,
) -> bool:
    """
    Check if the NKI kernel can be used.

    Returns False when any NKI kernel constraint is violated or the tensors
    live on the CPU without the NKI simulator enabled.
    """
    if not can_run_kernel(pos_ids):
        return False

    return True


def _resize_block_len(
    block_len: int,
    bs: int,
    q_head: int,
    s_active: int,
    s_prior: int,
) -> int:
    """Apply the same block_len resize that the NKI kernel applies internally.

    Matches ``gen_mask_tkg_hbm``: sprior_n_prgs is LNC only when
    ``is_s_prior_sharded(bs, q_head, s_active, s_prior, P_MAX)`` is True,
    otherwise 1. s_prior sharding requires s_prior >= 2*P_MAX AND batch is
    not sharded (batch sharding takes priority; it kicks in when
    bs % 2 == 0 AND (bs*q_head*s_active > P_MAX OR s_prior <= 2*P_MAX)).

    LNC selection MUST mirror ``gen_attention_decode_mask`` above: the NKI mask
    kernel is launched with lnc=2 only when s_prior is divisible by 2*P_MAX
    (``wrapped[lnc]`` with ``lnc = 1 if s_prior % (2*P_MAX) else 2``), and the
    kernel keys its own block_len resize off that same lnc. A hardcoded lnc=2
    here computes the wrong sprior_n_prgs for any s_prior that is a multiple of
    P_MAX but not 2*P_MAX (e.g. block_size=896 = 7*128): the resize is skipped,
    block_len stays oversized, and the downstream un-shuffle reshape in
    ``_torch_attention_decode_impl`` gets an invalid (non-permutation) index
    layout and throws.

    NOTE: the is_batch_sharded threshold must match the kernel's literal. On
    nkilib 2.31 (``core/attention/attention_tkg_utils.py`` is_batch_sharded)
    that literal is ``bs*q_head*s_active > p_max`` (STRICT) with
    ``curr_sprior <= 2*p_max``; keep it strict here. If it drifts the decode
    mask breaks at 256.
    """
    if block_len <= 0:
        return block_len

    # Mirror gen_attention_decode_mask's lnc pick: lnc=2 only when s_prior is
    # divisible by 2*P_MAX, else lnc=1 (no LNC sharding at all -> n_prgs 1).
    lnc = 2 if s_prior % (2 * P_MAX) == 0 else 1
    if lnc == 2:
        batch_sharded = (bs % lnc == 0) and (
            bs * q_head * s_active > P_MAX or s_prior <= 2 * P_MAX
        )
        sprior_sharded = (not batch_sharded) and s_prior >= lnc * P_MAX
        sprior_n_prgs = lnc if sprior_sharded else 1
    else:
        sprior_n_prgs = 1

    num_blocks_per_batch = s_prior // block_len
    bucket_len = num_blocks_per_batch * block_len
    min_multiple = sprior_n_prgs * P_MAX
    if bucket_len % min_multiple != 0:
        return block_len

    new_block_len, _ = _resize_cache_block_len_for_attention_tkg_kernel(
        num_blocks_per_batch, block_len, sprior_n_prgs, P_MAX
    )
    return new_block_len


def _resize_cache_block_len_for_attention_tkg_kernel(
    num_blocks_per_batch: int, block_len: int, n_prgs: int, p_max: int
):
    """
    Block KV in token gen attention requires number of blocks per batch to be a multiple of (lnc * p_max).
    This allows loading p_max blocks onto SBUF partitions in parallel.
    If the block count is not divisible by (lnc * p_max), we will reduce block_len to increase num_blocks_per_batch.
    As long as the bucket_len is divisible by lnc * p_max, there is always a block_len (min. 1) that satisfies the requirement.

    Args:
      num_blocks_per_batch: Number of blocks in each batch. Generally the second dimension of the active blocks table.
      block_len: The size of each block.
      n_prgs: Sharding level.
      p_max: Maximum number of partitions.
      full_sprior: Maximum KV cache capacity (bucket size). Used for warning suggestions.

    NOTE: This function is borrowed from NKI due to them having prints which aren't dynamo safe.
    """

    bucket_len = num_blocks_per_batch * block_len
    min_multiple = n_prgs * p_max

    # Find the greatest multiple of block_len that also divides the maximum block length.
    reduced_blk_len = math.gcd(block_len, bucket_len // min_multiple)
    resize_factor = block_len // reduced_blk_len

    return reduced_blk_len, resize_factor


def _build_block_kv_iota(s_prior: int, block_len: int, device, dtype) -> Tensor:
    """
    Build a [s_prior, 1, 1, 1] iota tensor that mirrors the NKI kernel's
    block-KV index layout.

    The HBM output of the kernel is [n_tile, P_MAX, ...] which is reshaped
    to [s_prior, ...].  For a linear output index ``i``:

        tile_idx  = i // P_MAX
        partition = i  % P_MAX
        fold_idx  = tile_idx  // block_len
        blk_off   = tile_idx  %  block_len
        position  = fold_idx * P_MAX * block_len + partition * block_len + blk_off

    This matches the kernel's ``nisa.iota`` with
    ``pattern=[[1, block_len]], channel_multiplier=block_len``.
    """
    i = torch.arange(s_prior, device=device, dtype=dtype)
    tile_idx = i // P_MAX
    partition = i % P_MAX
    fold_idx = tile_idx // block_len
    blk_off = tile_idx % block_len
    iota = fold_idx * P_MAX * block_len + partition * block_len + blk_off
    return iota.view(s_prior, 1, 1, 1)


def _seq_pos_to_linear_idx(pos: int, block_len: int) -> int:
    """
    Inverse of the block-KV iota: given a sequential position, return the
    linear index in the kernel's output layout.

        fold_idx   = pos // (P_MAX * block_len)
        within     = pos %  (P_MAX * block_len)
        partition  = within // block_len
        blk_off    = within %  block_len
        tile_idx   = fold_idx * block_len + blk_off
        linear_idx = tile_idx * P_MAX + partition
    """
    fold_idx = pos // (P_MAX * block_len)
    within = pos % (P_MAX * block_len)
    partition = within // block_len
    blk_off = within % block_len
    tile_idx = fold_idx * block_len + blk_off
    return tile_idx * P_MAX + partition


def _build_seq_to_linear_map(s_prior: int, block_len: int) -> Tensor:
    """Build a [s_prior] tensor mapping sequential position → linear output index.

    ``result[seq_pos]`` gives the linear index in the kernel output where
    sequential position ``seq_pos`` is stored.
    """
    seq_positions = torch.arange(s_prior)
    fold_idx = seq_positions // (P_MAX * block_len)
    within = seq_positions % (P_MAX * block_len)
    partition = within // block_len
    blk_off = within % block_len
    tile_idx = fold_idx * block_len + blk_off
    return tile_idx * P_MAX + partition


def _torch_gen_attention_decode_mask_impl(
    pos_ids: Tensor,
    bs: int,
    q_head: int,
    s_active: int,
    s_prior: int,
    start_pos: Optional[Tensor] = None,
    block_len: int = 0,
    active_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    PyTorch reference implementation of attention mask generation for TKG.

    Produces output in the same block-KV shuffled layout as the NKI kernel
    (``gen_mask_tkg_hbm``).

    The prior mask is **uniform** across all active tokens — it uses the
    minimum pos_id (and maximum start_pos) per batch element, since the
    KV cache is only filled up to the earliest active token.  The per-token
    causal staircase among active tokens is captured entirely by
    ``active_mask``.  Even for ``s_active == 1`` the active mask is needed
    because the ``iota < pos_id`` comparison does not include the active
    token's own position (``pos == pos`` is not ``< pos``).

    Args:
        pos_ids:     [1, bs * s_active]  Position IDs (end positions, exclusive).
        bs:          Batch size.
        q_head:      Number of query heads.
        s_active:    Active (new-token) sequence length.
        s_prior:     Prior (KV-cache) sequence length.
        start_pos:   Optional [1, bs * s_active]  Per-query SWA window start
                     (inclusive).  ``None`` → standard causal mask.
        block_len:   Block length for block-KV cache.  Must be > 0.
        active_mask: [s_active, bs, q_head, s_active] active-token mask
                     loaded into the positions corresponding to the last
                     ``s_active`` sequence slots.  **Always required**.

    Returns:
        Tensor of shape [s_prior, bs, q_head, s_active] with values in {0, 1}.

    Raises:
        NotImplementedError: If ``block_len <= 0``.
        ValueError: If ``active_mask is None``.
    """
    if block_len <= 0:
        raise NotImplementedError(
            f"Block KV cache (block_len={block_len}) is not supported "
            "in the PyTorch fallback path; block_len must be > 0"
        )

    device = pos_ids.device
    dtype = torch.float32

    # Apply the same block_len resize the kernel does internally
    block_len = _resize_block_len(block_len, bs, q_head, s_active, s_prior)

    # Build the block-KV shuffled iota: [s_prior, 1, 1, 1]
    iota = _build_block_kv_iota(s_prior, block_len, device, dtype)

    # ── collapse to a single threshold per batch element ──
    pos_2d = pos_ids.view(bs, s_active)  # [bs, s_active]
    min_pos = pos_2d.min(dim=1, keepdim=True).values  # [bs, 1]
    min_pos = min_pos.unsqueeze(0).unsqueeze(2)  # [1, bs, 1, 1]

    if start_pos is None:
        # Standard causal: prior positions < min_pos are valid
        mask = (iota < min_pos).to(dtype)
    else:
        # SWA: use max(start_pos) per batch — most conservative start
        start_2d = start_pos.view(bs, s_active)  # [bs, s_active]
        max_start = start_2d.max(dim=1, keepdim=True).values  # [bs, 1]
        max_start = max_start.unsqueeze(0).unsqueeze(2)  # [1, bs, 1, 1]

        ge_start = iota >= max_start  # [s_prior, bs, 1, 1]
        lt_end = iota < min_pos  # [s_prior, bs, 1, 1]

        normal = ge_start & lt_end
        wrap = ge_start | lt_end

        is_wrap = max_start > min_pos  # [1, bs, 1, 1]
        mask = torch.where(is_wrap, wrap, normal).to(dtype)

    # Broadcast across q_head and s_active: [s_prior, bs, 1, 1] → full shape
    mask = mask.expand(s_prior, bs, q_head, s_active).contiguous()

    # Overlay active mask at the shuffled positions corresponding to the
    # last s_active sequential slots.
    for k in range(s_active):
        seq_pos = s_prior - s_active + k
        lin_idx = _seq_pos_to_linear_idx(seq_pos, block_len)
        mask[lin_idx, :, :, :] = active_mask[k, :, :, :]

    return mask
