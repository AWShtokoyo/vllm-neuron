# SPDX-License-Identifier: Apache-2.0
"""
Bucket generation and validation utilities for Neuron.

This module provides utilities for managing buckets used in Neuron's
compiled model execution :
- num_batched_tokens_buckets: Prefill segment sizes (token counts)
- num_seqs_buckets: Decode batch sizes (request counts)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Default Bucket Generation
# =============================================================================


def get_default_num_batched_tokens_buckets(max_num_batched_tokens: int) -> list[int]:
    """Generate default prefill token buckets (powers of 2 starting from 128).

    Args:
        max_num_batched_tokens: Maximum number of batched tokens (must be last bucket).

    Returns:
        List of power-of-2 bucket sizes ending with max_num_batched_tokens.
        e.g., [128, 256, 512, 1024, 2048] for max_num_batched_tokens=2048
    """
    buckets = []
    bucket = 128
    while bucket < max_num_batched_tokens:
        buckets.append(bucket)
        bucket *= 2

    # Always add max_num_batched_tokens as the last bucket
    buckets.append(max_num_batched_tokens)
    return buckets


def get_default_num_seqs_buckets(max_num_seqs: int) -> list[int]:
    """Generate default decode batch size buckets (powers of 2 starting from 1).

    Args:
        max_num_seqs: Maximum number of sequences (must be last bucket).

    Returns:
        List of power-of-2 bucket sizes ending with max_num_seqs.
        e.g., [1, 2, 4, 8] for max_num_seqs=8
    """
    buckets = []
    size = 1
    while size < max_num_seqs:
        buckets.append(size)
        size *= 2

    # Always add max_num_seqs as the last bucket
    buckets.append(max_num_seqs)
    return buckets


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n.

    Args:
        n: Input value.

    Returns:
        Smallest power of 2 that is >= n.
        e.g., next_power_of_2(5) = 8, next_power_of_2(8) = 8
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def get_default_num_vision_tokens_buckets(
    max_bucket: int, vision_attention_block_size: int
) -> list[int]:
    """Generate default vision token buckets (power-of-2 from block_size up to max_bucket).

    Args:
        max_bucket: Maximum bucket size (largest bucket).
        vision_attention_block_size: Starting bucket size.

    Returns:
        List of power-of-2 bucket sizes ending with max_bucket.
        e.g., [2048, 4096, 8192, 16384, 32768, 65536] for max_bucket=65536
    """
    buckets = []
    bucket = vision_attention_block_size
    while bucket < max_bucket:
        buckets.append(bucket)
        bucket *= 2
    buckets.append(max_bucket)
    return buckets


# =============================================================================
# DCP Bucket Alignment
# =============================================================================


def align_prefill_buckets_for_dcp(
    buckets: list[int],
    dcp_stride: int,
) -> list[int]:
    """Filter prefill buckets to those divisible by the DCP shard stride.

    DCP prefill reshapes the token dim as ``(S // dcp_stride, dcp, interleave)``
    to select this rank's slice, so ``S % dcp_stride == 0`` is required — a
    violating bucket crashes at warmup/runtime. ``dcp_stride <= 1`` = no-op.

    Args:
        buckets: Resolved prefill bucket list (ascending, non-empty).
        dcp_stride: ``decode_context_parallel_size * block_size``.
            For non-DCP configs pass 1.

    Returns:
        Subset of ``buckets`` where every entry divides evenly by
        ``dcp_stride``, preserving the original order.

    Raises:
        ValueError: If no bucket is divisible by ``dcp_stride`` (config
            incompatible with DCP prefill), or if the last (``max_num_batched_tokens``)
            isn't — dropping it would silently cap prefill. Other misaligned
            buckets are dropped with a warning.
    """
    if dcp_stride <= 1:
        return buckets

    aligned = [b for b in buckets if b % dcp_stride == 0]
    if not aligned:
        raise ValueError(
            f"No prefill buckets are divisible by the DCP shard stride "
            f"(decode_context_parallel_size * block_size = {dcp_stride}). "
            f"Buckets: {buckets}. Either reduce decode_context_parallel_size, "
            f"reduce block_size, or set num_batched_tokens_buckets to values "
            f"that are multiples of {dcp_stride}."
        )

    # The last bucket is max_num_batched_tokens (validate_num_batched_tokens_buckets).
    # Silently dropping it would cap prefill below the scheduler's max; fail loud instead.
    if aligned[-1] != buckets[-1]:
        raise ValueError(
            f"The largest prefill bucket (max_num_batched_tokens = {buckets[-1]}) "
            f"is not divisible by the DCP shard stride "
            f"(decode_context_parallel_size * block_size = {dcp_stride}), so it "
            f"cannot be used with DCP prefill. Set max_num_batched_tokens (or the "
            f"last num_batched_tokens_buckets entry) to a multiple of {dcp_stride}, "
            f"or reduce decode_context_parallel_size / block_size."
        )

    if len(aligned) < len(buckets):
        logger.warning(
            "DCP prefill alignment: dropped %d bucket(s) not divisible by %d. "
            "Remaining: %s (from %s)",
            len(buckets) - len(aligned),
            dcp_stride,
            aligned,
            buckets,
        )

    return aligned


# =============================================================================
# Bucket Validation
# =============================================================================


def validate_num_batched_tokens_buckets(
    buckets: Any,
    max_num_batched_tokens: int,
) -> list[int]:
    """Validate prefill token bucket configuration.

    Validates that:
    1. Bucket list is not empty
    2. All values are positive integers
    3. Values are in strictly ascending order
    4. Last bucket equals max_num_batched_tokens

    Args:
        buckets: List of bucket sizes to validate.
        max_num_batched_tokens: Expected value of the last bucket.

    Returns:
        The validated bucket list.

    Raises:
        ValueError: If validation fails.
    """
    param_name = "num_batched_tokens_buckets"

    if not buckets:
        raise ValueError(f"{param_name} cannot be empty")

    if not isinstance(buckets, list):
        raise ValueError(f"{param_name} must be a list, got {type(buckets).__name__}")

    # Check all values are positive integers
    for i, bucket in enumerate(buckets):
        if not isinstance(bucket, int):
            raise ValueError(f"{param_name}[{i}] must be an integer, got {bucket}")
        if bucket <= 0:
            raise ValueError(f"{param_name}[{i}] must be positive, got {bucket}")

    # Check strictly ascending order
    for i in range(1, len(buckets)):
        if buckets[i] <= buckets[i - 1]:
            raise ValueError(
                f"{param_name} must be in strictly ascending order, got {buckets}"
            )

    # Check last bucket matches max_num_batched_tokens
    if buckets[-1] != max_num_batched_tokens:
        raise ValueError(
            f"Last bucket in {param_name} must equal max_num_batched_tokens "
            f"({max_num_batched_tokens}), got {buckets[-1]}"
        )

    return buckets


def resolve_num_batched_tokens_buckets(
    max_num_batched_tokens: int,
    *,
    configured_buckets: Any = None,
    use_configured_buckets: bool = False,
    auto_buckets: list[int] | None = None,
    dcp_stride: int = 1,
) -> list[int]:
    """Resolve and DCP-align the prefill token buckets in one place.

    Explicit user configuration takes precedence over segmented-prefill
    defaults, which take precedence over the standard power-of-two defaults.
    ``use_configured_buckets`` distinguishes an absent setting from an explicit
    invalid value such as ``None``.

    Args:
        max_num_batched_tokens: Required value of the largest bucket.
        configured_buckets: User-provided bucket list, when configured.
        use_configured_buckets: Whether ``configured_buckets`` was explicitly set.
        auto_buckets: Optional segmented-prefill bucket list.
        dcp_stride: DCP size multiplied by cache block size, or 1 without DCP.

    Returns:
        The selected, validated, and DCP-aligned bucket list.

    Example:
        ``resolve_num_batched_tokens_buckets(2048, dcp_stride=512)`` returns
        ``[512, 1024, 2048]``.
    """
    if use_configured_buckets:
        buckets = validate_num_batched_tokens_buckets(
            configured_buckets, max_num_batched_tokens
        )
    elif auto_buckets is not None:
        buckets = auto_buckets
    else:
        buckets = get_default_num_batched_tokens_buckets(max_num_batched_tokens)

    return align_prefill_buckets_for_dcp(buckets, dcp_stride)


def validate_num_seqs_buckets(
    buckets: Any,
    max_num_seqs: int,
) -> list[int]:
    """Validate decode batch size bucket configuration.

    Validates that:
    1. Bucket list is not empty
    2. All values are positive integers
    3. Values are in strictly ascending order
    4. Last bucket equals max_num_seqs

    Args:
        buckets: List of bucket sizes to validate.
        max_num_seqs: Expected value of the last bucket.

    Returns:
        The validated bucket list.

    Raises:
        ValueError: If validation fails.
    """
    param_name = "num_seqs_buckets"

    if not buckets:
        raise ValueError(f"{param_name} cannot be empty")

    if not isinstance(buckets, list):
        raise ValueError(f"{param_name} must be a list, got {type(buckets).__name__}")

    # Check all values are positive integers
    for i, bucket in enumerate(buckets):
        if not isinstance(bucket, int):
            raise ValueError(f"{param_name}[{i}] must be an integer, got {bucket}")
        if bucket <= 0:
            raise ValueError(f"{param_name}[{i}] must be positive, got {bucket}")

    # Check strictly ascending order
    for i in range(1, len(buckets)):
        if buckets[i] <= buckets[i - 1]:
            raise ValueError(
                f"{param_name} must be in strictly ascending order, got {buckets}"
            )

    # Check last bucket matches max_num_seqs
    if buckets[-1] != max_num_seqs:
        raise ValueError(
            f"Last bucket in {param_name} must equal max_num_seqs "
            f"({max_num_seqs}), got {buckets[-1]}"
        )

    return buckets


# Segment sizes currently supported by the segmented attention NKI kernel.
SUPPORTED_KV_SEGMENT_SIZES = {512, 1024, 2048, 4096, 8192}

# Upper bound on max_model_len for which single-shot prefill
# (max_num_batched_tokens == max_model_len) is permitted. Above this,
# chunked / segmented prefill is required.
MAX_MODEL_LEN_SINGLE_SHOT = 16 * 1024


def resolve_segmented_prefill_config(
    max_num_batched_tokens: int,
    max_model_len: int,
) -> tuple[list[int] | None, list[int] | None]:
    """Decide whether to enable segmented prefill by default.

    Only called when the user did not explicitly set
    ``kv_segment_size_buckets``.

    Two outcomes:

    - Chunked prefill (``max_num_batched_tokens < max_model_len``):
      ``max_num_batched_tokens`` must be one of
      ``SUPPORTED_KV_SEGMENT_SIZES``. Returns
      ``([max_num_batched_tokens], [max_num_batched_tokens])`` — the
      caller should auto-enable segmented prefill with these buckets.
    - Single-shot prefill (``max_num_batched_tokens >= max_model_len``):
      allowed only when ``max_model_len <= MAX_MODEL_LEN_SINGLE_SHOT``.
      Returns ``(None, None)`` — the caller should not configure
      segmented prefill and should fall back to its own defaults for
      ``num_batched_tokens_buckets`` (e.g. power-of-2 buckets).

    Raises:
        ValueError: On unsupported combinations of ``max_model_len`` and
            ``max_num_batched_tokens``.
    """
    if max_num_batched_tokens >= max_model_len:
        if max_model_len > MAX_MODEL_LEN_SINGLE_SHOT:
            raise ValueError(
                f"Single-shot prefill (max_num_batched_tokens="
                f"{max_num_batched_tokens} >= max_model_len="
                f"{max_model_len}) is only supported when max_model_len "
                f"<= {MAX_MODEL_LEN_SINGLE_SHOT}. Set "
                f"max_num_batched_tokens to one of "
                f"{sorted(SUPPORTED_KV_SEGMENT_SIZES)} to enable "
                f"chunked prefill."
            )
        return (None, None)

    if max_num_batched_tokens not in SUPPORTED_KV_SEGMENT_SIZES:
        supported_sorted = sorted(SUPPORTED_KV_SEGMENT_SIZES)
        msg = (
            f"max_num_batched_tokens={max_num_batched_tokens} is not a "
            f"supported chunked prefill size on Neuron. Supported values: "
            f"{supported_sorted}."
        )
        if max_model_len <= MAX_MODEL_LEN_SINGLE_SHOT:
            msg += (
                f" Alternatively, set max_num_batched_tokens="
                f"{max_model_len} (equal to max_model_len) to disable "
                f"chunked prefill."
            )
        raise ValueError(msg)

    return ([max_num_batched_tokens], [max_num_batched_tokens])


def validate_kv_segment_size_buckets(
    buckets: Any,
    num_batched_tokens_buckets: list[int] | None,
) -> list[int]:
    """Validate kv_segment_size_buckets configuration for segmented prefill.

    Validates the following interface constraints (always enforced):
        1. Bucket list is a non-empty list of integers.
        2. Buckets must be in strictly ascending order.
        3. Each value must be one of the sizes supported by the segmented
           attention NKI kernel (see ``SUPPORTED_KV_SEGMENT_SIZES``).
        4. Every prefill bucket must be a legal query length that fits the
           largest segment size (multiple of 128, and <= max(buckets)).

    Multiple segment sizes are supported: warmup compiles the cross product
    ``kv_segment_size x num_batched_tokens_buckets`` and the NEFF cache keys the
    two dimensions independently, so the only requirement is that each prefill
    bucket is legal against the *largest* compiled segment. The per-request
    choice among the compiled segment sizes is made at runtime by
    :func:`pick_kv_segment_size`.

    Args:
        buckets: List of segment size buckets to validate.
        num_batched_tokens_buckets: Explicitly configured batched tokens buckets,
            or None if not set by user.

    Returns:
        The validated bucket list.

    Raises:
        ValueError: If validation fails.

    Example:
        >>> validate_kv_segment_size_buckets([2048], None)
        [2048]
        >>> validate_kv_segment_size_buckets([2048], [2048])
        [2048]
        >>> validate_kv_segment_size_buckets([512, 2048], [128, 512])
        [512, 2048]
    """
    param_name = "kv_segment_size_buckets"

    # 1. Must be a non-empty list of integers
    if not isinstance(buckets, list) or len(buckets) == 0:
        raise ValueError(f"{param_name} must be a non-empty list")

    for i, s in enumerate(buckets):
        if not isinstance(s, int):
            raise ValueError(f"{param_name}[{i}] must be an integer, got {s}")

    # 2. Strictly ascending order
    # TODO: Ordering matters once multiple segment sizes are supported,
    # so that bucket selection can pick the smallest fitting segment.
    for i in range(1, len(buckets)):
        if buckets[i] <= buckets[i - 1]:
            raise ValueError(
                f"{param_name} must be in strictly ascending order, got {buckets}"
            )

    # 3. Each value must be a kernel-supported segment size
    for i, s in enumerate(buckets):
        if s not in SUPPORTED_KV_SEGMENT_SIZES:
            raise ValueError(
                f"{param_name}[{i}] = {s} is not a supported segment size. "
                f"The segmented attention kernel only supports: "
                f"{SUPPORTED_KV_SEGMENT_SIZES}. "
                f"Please use one of these values."
            )

    # 4. Prefill buckets must be legal query lengths that fit the largest
    #    segment size.
    #    Previously this required an exact match against a single segment size, on
    #    the premise that "the segmented kernel requires the prefill bucket length
    #    to equal the segment size". That premise does not hold: the kernel derives
    #    the active segment from seqlen_q and the prior segments from
    #    prior_seg_size independently, then sizes SBUF as max() of the two. So the
    #    only requirement is that each prefill bucket is a legal query length
    #    (multiple of 128) that fits the segment it will run against.
    #
    #    Bound against max(buckets), not buckets[0]: with several compiled segment
    #    sizes a bucket only has to fit the segment actually selected for it at
    #    runtime, and pick_kv_segment_size() never returns a segment smaller than
    #    the active query length (see its clamping contract). Bounding on the
    #    smallest segment would reject legal ladders such as
    #    nb=[128,512] x ks=[512,2048].
    #
    #    Warmup already compiles the cross product kv_seg x bucket and the NEFF
    #    cache keys them separately, so multiple segment sizes need no further
    #    plumbing here.
    if num_batched_tokens_buckets is not None:
        seg_max = max(buckets)
        bad = [b for b in num_batched_tokens_buckets if b > seg_max]
        if bad:
            raise ValueError(
                f"num_batched_tokens_buckets entries must not exceed the largest "
                f"{param_name}={seg_max}; got {bad}."
            )
        illegal = [b for b in num_batched_tokens_buckets if b % 128 != 0]
        if illegal:
            raise ValueError(
                f"num_batched_tokens_buckets entries must be multiples of 128 "
                f"(the segmented kernel asserts seqlen_q % 128 == 0); "
                f"got {illegal}."
            )

    return buckets


def pick_kv_segment_size(
    cached_seq_len: int,
    kv_segment_size_buckets: list[int],
    active_seq_len: int = 0,
) -> int:
    """Pick the kv_segment_size to use for one prefill step.

    Chooses the compiled segment size that minimizes kernel launches over the
    prior (already-cached) KV, preferring fewer iterations, then less wasted
    tail, then larger segments:

        num_segments = ceil(cached_seq_len / kv_segment_size)
        tail_waste   = num_segments * kv_segment_size - cached_seq_len
        score        = (num_segments, tail_waste, -kv_segment_size)

    and returns the lexicographically smallest. With a single compiled size the
    result is always that size, so behaviour is unchanged from the pre-``#5``
    single-segment configuration.

    ``active_seq_len`` clamps the result upwards: the segmented kernel requires
    ``seqlen_q <= kv_segment_size`` (see ``attention_segmented_cte``), so a
    segment smaller than the active query length is never legal even when it
    scores better on the prior. The smallest compiled size that still admits the
    query wins; if none does, the largest compiled size is returned and the
    kernel wrapper's own validation reports the violation.

    Args:
        cached_seq_len: Number of prior (already-cached) KV tokens.
        kv_segment_size_buckets: Compiled KV segment sizes (ascending).
        active_seq_len: Active query length for this step (0 = unconstrained).

    Returns:
        The selected kv_segment_size.

    Example:
        >>> pick_kv_segment_size(0, [512, 2048])
        512
        >>> pick_kv_segment_size(4096, [512, 2048])
        2048
        >>> pick_kv_segment_size(0, [512, 2048], active_seq_len=1024)
        2048
    """
    if not kv_segment_size_buckets:
        raise ValueError("kv_segment_size_buckets cannot be empty")

    # Only segments that can legally hold the active query are candidates.
    legal = [s for s in kv_segment_size_buckets if s >= active_seq_len]
    if not legal:
        # Let the kernel wrapper raise with its own precise message.
        return max(kv_segment_size_buckets)

    if cached_seq_len <= 0:
        # No prior to iterate over, so launch overhead dominates: take the
        # smallest legal segment.
        return min(legal)

    def _score(kv_segment_size: int) -> tuple[int, int, int]:
        num_segments = (cached_seq_len + kv_segment_size - 1) // kv_segment_size
        tail_waste = num_segments * kv_segment_size - cached_seq_len
        return num_segments, tail_waste, -kv_segment_size

    return min(legal, key=_score)


# NKI attention kernel tile constraint. Same constant as
# `_compute_swa_num_blocks` rounds to. The block-table seq dim must be a
# multiple of P_MAX // block_size.
_DECODE_CONTEXT_LENGTH_PMAX = 128


def validate_decode_context_length_buckets(
    buckets: Any,
    max_model_len: int,
) -> list[int]:
    """Validate decode_context_length_buckets configuration.

    Validates that:
        1. Bucket list is a non-empty list of positive ints in strictly
           ascending order.
        2. Every value is strictly less than ``max_model_len``. Equality
           is redundant — the ``max_model_len`` fallback NEFF is always
           compiled separately.
        3. Every value is divisible by ``P_MAX = 128`` (the NKI attention
           kernel tile constraint).

    Args:
        buckets: List of decode context length bucket sizes to validate.
        max_model_len: Model's maximum sequence length.

    Returns:
        The validated bucket list.

    Raises:
        ValueError: If validation fails.

    Example:
        >>> validate_decode_context_length_buckets([2048, 4096], max_model_len=16384)
        [2048, 4096]
    """
    param_name = "decode_context_length_buckets"

    if not isinstance(buckets, list):
        raise ValueError(f"{param_name} must be a list, got {type(buckets).__name__}")
    if len(buckets) == 0:
        raise ValueError(f"{param_name} must be a non-empty list")

    for i, bucket in enumerate(buckets):
        if not isinstance(bucket, int) or isinstance(bucket, bool):
            raise ValueError(f"{param_name}[{i}] must be an integer, got {bucket!r}")
        if bucket <= 0:
            raise ValueError(f"{param_name}[{i}] must be positive, got {bucket}")

    for i in range(1, len(buckets)):
        if buckets[i] <= buckets[i - 1]:
            raise ValueError(
                f"{param_name} must be in strictly ascending order, got {buckets}"
            )

    for i, bucket in enumerate(buckets):
        if bucket >= max_model_len:
            raise ValueError(
                f"{param_name}[{i}]={bucket} must be strictly less than "
                f"max_model_len={max_model_len}; max_model_len is the "
                f"implicit fallback bucket."
            )
        if bucket % _DECODE_CONTEXT_LENGTH_PMAX != 0:
            raise ValueError(
                f"{param_name}[{i}]={bucket} must be divisible by "
                f"{_DECODE_CONTEXT_LENGTH_PMAX} to satisfy the attention "
                f"kernel tile constraint."
            )

    return buckets


# =============================================================================
# Bucket Lookup
# =============================================================================


def get_bucket_for_count(count: int, buckets: list[int]) -> int:
    """Find the smallest bucket that can accommodate the given count.

    Args:
        count: The count to find a bucket for.
        buckets: List of bucket sizes in ascending order.

    Returns:
        The smallest bucket size >= count, or the largest bucket if count
        exceeds all buckets.

    Example:
        >>> get_bucket_for_count(100, [128, 256, 512, 1024])
        128
        >>> get_bucket_for_count(128, [128, 256, 512, 1024])
        128
        >>> get_bucket_for_count(129, [128, 256, 512, 1024])
        256
        >>> get_bucket_for_count(2000, [128, 256, 512, 1024])
        1024
    """
    if not buckets:
        raise ValueError("Bucket list cannot be empty")

    if count <= 0:
        logger.warning("Count is %d, returning smallest bucket", count)
        return buckets[0]

    for bucket in buckets:
        if bucket >= count:
            return bucket

    # Count exceeds all buckets, return largest
    logger.warning(
        "Count %d exceeds all buckets %s, returning largest bucket %d",
        count,
        buckets,
        buckets[-1],
    )
    return buckets[-1]


def get_decode_padded_batch_size(
    num_reqs: int,
    max_query_len: int,
    num_seqs_buckets: list[int],
    decode_token_threshold: int = 1,
) -> int:
    """Calculate padded batch size for decode phase to match compiled buckets.

    Applies during decode. For non-spec decode, `max_query_len == 1`. For
    spec decode, `max_query_len` can be up to `1 + num_speculative_tokens`;
    pass that value via `decode_token_threshold`. Returns `num_reqs`
    unchanged for prefill (query len exceeds the decode cap) or when no
    buckets are configured.

    Args:
        num_reqs: Actual number of requests in the batch.
        max_query_len: Maximum query length (<=decode_token_threshold
            for decode, >decode_token_threshold for prefill).
        num_seqs_buckets: Compiled decode batch size buckets.
        decode_token_threshold: Largest query length still considered
            decode (1 + num_speculative_tokens when spec decode is on,
            else 1).

    Returns:
        Padded batch size matching the next compiled bucket.

    Example:
        >>> get_decode_padded_batch_size(3, max_query_len=1, num_seqs_buckets=[1, 2, 4, 8])
        4
        >>> get_decode_padded_batch_size(3, max_query_len=256, num_seqs_buckets=[1, 2, 4, 8])
        3
        >>> get_decode_padded_batch_size(1, max_query_len=4, num_seqs_buckets=[2],
        ...     decode_token_threshold=4)
        2
    """
    if (
        max_query_len < 1
        or max_query_len > decode_token_threshold
        or not num_seqs_buckets
    ):
        return num_reqs

    return get_bucket_for_count(num_reqs, num_seqs_buckets)


# =============================================================================
# Other Utils
# =============================================================================


def get_max_num_batched_tokens(
    configured_value: int,
    max_model_len: int,
) -> int:
    """Clamp max_num_batched_tokens to not exceed max_model_len.

    Args:
        configured_value: The configured max_num_batched_tokens from scheduler config.
        max_model_len: The maximum model sequence length.

    Returns:
        min(configured_value, max_model_len).
    """
    return min(configured_value, max_model_len)
