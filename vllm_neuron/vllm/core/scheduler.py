# SPDX-License-Identifier: Apache-2.0
"""
Neuron Scheduler implementation for vLLM

This module provides a custom scheduler optimized for Neuron hardware.
It uses a holdback queue pattern for cleaner request management and
bucket-aware admission control.
"""

import inspect
import logging
import os
from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from vllm_neuron.metrics import NUM_BATCHED_TOKENS_PADDING, NUM_SEQS_PADDING
from vllm_neuron.utils.bucket_utils import (
    get_bucket_for_count,
    get_default_num_batched_tokens_buckets,
    get_default_num_seqs_buckets,
    get_max_num_batched_tokens,
    resolve_segmented_prefill_config,
    validate_num_batched_tokens_buckets,
    validate_num_seqs_buckets,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.multimodal import MultiModalRegistry
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.structured_output import StructuredOutputManager

logger = logging.getLogger(__name__)


class SchedulerState(Enum):
    """Scheduler state for Neuron platform.

    State machine:
        IDLE → ACTIVE_PREFILL → ACTIVE_DECODE → IDLE
                     ↑    ↓           ↑    ↓
                     └────┘           └────┘
                           ↑          ↓
                           └──────────┘

    Transitions:
        - IDLE → ACTIVE_PREFILL: Prefill request arrives
        - ACTIVE_PREFILL → ACTIVE_PREFILL: More segments
        - ACTIVE_PREFILL → ACTIVE_DECODE: Prefill completes
        - ACTIVE_DECODE → ACTIVE_DECODE: Decode continues
        - ACTIVE_DECODE → ACTIVE_PREFILL: Decode has capacity AND prefill waiting
        - ACTIVE_DECODE → IDLE: All decode complete AND no prefill waiting

    States:
        IDLE: No requests in system (waiting=[], running=[])
        ACTIVE_PREFILL: Last schedule() call scheduled prefill.
        This state indicates there are prefill request(s) in waiting or running queue,
        and we don't have a full batch of decode requests
        ACTIVE_DECODE: Last schedule() call scheduled decode,
        This state indicates the running queue only contains decode requests

    """

    IDLE = auto()
    ACTIVE_PREFILL = auto()
    ACTIVE_DECODE = auto()

    def __str__(self) -> str:
        return self.name


class NeuronScheduler(Scheduler):
    """
    Scheduler for Neuron platform with prefill/decode separation.

    This scheduler extends vLLM's Scheduler with Neuron-specific
    constraints:
    - Holdback queue for managing request scheduling constraints
    - Prefill/decode separation (Neuron hardware requirement)
    - Token padding for compiled model bucket sizes
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: "MultiModalRegistry | None" = None,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the Neuron async scheduler.

        Args:
            vllm_config: vLLM configuration object.
            kv_cache_config: KV cache configuration.
            structured_output_manager: Manager for structured outputs.
            block_size: Size of KV cache blocks.
            mm_registry: Multimodal registry (optional).
            include_finished_set: Whether to include finished requests in output.
            log_stats: Whether to log scheduler statistics.
        """
        if mm_registry is None:
            mm_registry = MULTIMODAL_REGISTRY

        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            hash_block_size=hash_block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        self.log_stats = log_stats
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len
        self.model_name = vllm_config.model_config.model

        self._cap_encoder_budget_to_vision_bucket(vllm_config)

        # TODO: Support additional scheduling policies
        scheduler_policy = vllm_config.scheduler_config.policy
        if scheduler_policy != "fcfs":
            raise ValueError(
                f"NeuronAsyncScheduler currently only supports 'fcfs' scheduling policy, "
                f"got '{scheduler_policy}'."
            )

        # Holdback queue: requests temporarily held back from the base scheduler
        # to enforce Neuron-specific constraints
        self.holdback_queue: deque[Request] = deque()

        # The async-KV last-token-recompute reclassify (see schedule()) is only
        # needed for the isolated-decode bench connector, which reports the whole
        # prompt as external KV. Regular DI (NixlConnector) does not need it, so
        # gate on the specific connector name rather than "any connector".
        kv_transfer_config = vllm_config.kv_transfer_config
        self._is_decode_bench_connector = (
            kv_transfer_config is not None
            and kv_transfer_config.kv_connector == "NeuronDecodeBenchConnector"
        )

        # Current scheduler state (updated after each schedule() call)
        self._state: SchedulerState = SchedulerState.IDLE
        self._kv_exhaustion_warned: bool = False

        # Get configuration from neuron_config
        neuron_config = vllm_config.additional_config.get("neuron_config", {})

        max_num_batched_tokens = get_max_num_batched_tokens(
            vllm_config.scheduler_config.max_num_batched_tokens, self.max_model_len
        )

        user_set_kv_segment_size_buckets = "kv_segment_size_buckets" in neuron_config
        user_set_num_batched_tokens_buckets = (
            "num_batched_tokens_buckets" in neuron_config
        )

        auto_num_batched_tokens_buckets: list[int] | None = None
        if not user_set_kv_segment_size_buckets:
            _, auto_num_batched_tokens_buckets = resolve_segmented_prefill_config(
                max_num_batched_tokens, self.max_model_len
            )

        if user_set_num_batched_tokens_buckets:
            self.num_batched_tokens_buckets = validate_num_batched_tokens_buckets(
                neuron_config["num_batched_tokens_buckets"],
                max_num_batched_tokens,
            )
            logger.info(
                "Using num_batched_tokens_buckets from neuron_config: %s",
                self.num_batched_tokens_buckets,
            )
        elif auto_num_batched_tokens_buckets is not None:
            self.num_batched_tokens_buckets = auto_num_batched_tokens_buckets
            logger.info(
                "Segmented prefill auto-enabled; num_batched_tokens_buckets=%s",
                self.num_batched_tokens_buckets,
            )
        else:
            self.num_batched_tokens_buckets = get_default_num_batched_tokens_buckets(
                max_num_batched_tokens
            )
            logger.info(
                "Using default num_batched_tokens_buckets: %s",
                self.num_batched_tokens_buckets,
            )

        # Pooling/embedding models are prefill-only (no decode/KV/sampling), so
        # MULTIPLE sequences can be PACKED into one prefill bucket for throughput
        # (short embedding prompts otherwise waste most of a padded bucket). This
        # is gated behind VLLM_NEURON_POOLING_PACK (default ON) so it can be
        # disabled to fall back to the simpler one-prefill-per-batch path; the
        # generative (CausalLM) path is never affected either way.
        self.is_pooling = (
            getattr(vllm_config.model_config, "runner_type", None) == "pooling"
        )
        self.pooling_pack = self.is_pooling and (
            os.environ.get("VLLM_NEURON_POOLING_PACK", "1") == "1"
        )

        # Prefill batch size limit. Generative models: 1 (Neuron prefill/decode
        # separation assumes a single prefill, then decode). Pooling/embedding
        # models with packing enabled admit up to max_num_seqs prefills and pack
        # them into one bucket (see _apply_pooling_pack_padding and the runner's
        # _create_padded_inputs pooling branch) — the throughput win.
        self.max_prefills_per_batch: int = (
            self.max_num_seqs if self.pooling_pack else 1
        )

        # Parse num_seqs_buckets for decode batch size buckets
        if "num_seqs_buckets" in neuron_config:
            self.num_seqs_buckets = validate_num_seqs_buckets(
                neuron_config["num_seqs_buckets"],
                self.max_num_seqs,
            )
        else:
            self.num_seqs_buckets = get_default_num_seqs_buckets(self.max_num_seqs)

        # Pre-initialize metric labels so they appear at /metrics immediately
        for bucket_size in self.num_batched_tokens_buckets:
            bucket_name = f"prefill_s{bucket_size}"
            NUM_BATCHED_TOKENS_PADDING.labels(
                model_name=self.model_name, bucket_name=bucket_name
            )
        for bucket_size in self.num_seqs_buckets:
            bucket_name = f"decode_b{bucket_size}"
            NUM_SEQS_PADDING.labels(model_name=self.model_name, bucket_name=bucket_name)

        # Padding statistics
        self.total_padding_tokens: int = 0
        self.total_scheduled_tokens: int = 0

        # Warn if max_num_seqs exceeds KV cache capacity.
        # Use the same calculation as upstream vLLM (accounts for HMA groups,
        # sliding window, etc.) so the number matches the "Maximum concurrency"
        # log line from vllm/v1/core/kv_cache_utils.py.
        if kv_cache_config.kv_cache_groups:
            from vllm.v1.core.kv_cache_utils import (
                get_max_concurrency_for_kv_cache_config,
            )

            max_concurrency = get_max_concurrency_for_kv_cache_config(
                vllm_config, kv_cache_config
            )
            max_concurrent = int(max_concurrency)
        else:
            # Fallback for attention-free models or empty groups
            total_kv_tokens = kv_cache_config.num_blocks * block_size
            max_concurrent = (
                total_kv_tokens // self.max_model_len if self.max_model_len > 0 else 0
            )

        self._max_kv_concurrent = (
            max_concurrent if max_concurrent > 0 else self.max_num_seqs
        )

        if 0 < max_concurrent < self.max_num_seqs:
            logger.warning(
                "max_num_seqs (%d) exceeds worst-case KV cache capacity (%d "
                "concurrent requests at max_model_len=%d, assuming no prefix "
                "cache hits). Effective concurrency will be capped at %d to "
                "prevent KV cache exhaustion and request preemption. To "
                "increase concurrency, raise "
                "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION or reduce "
                "max_model_len.",
                self.max_num_seqs,
                max_concurrent,
                self.max_model_len,
                max_concurrent,
            )

        # Retain the window-low sliding-window block on the DI KV-pull admission path so a decode
        # SWA query never attends a null (uninitialized-K) block. Installed on this scheduler's own
        # kv_cache_manager now that super().__init__() has fully built it (once per DP replica).
        # No-op for models without a sliding-window group and for all non-DI admissions.
        self._install_swa_di_eviction_fix()

        logger.info("Initialized NeuronAsyncScheduler for Neuron platform")
        logger.info("Max prefills per batch: %d", self.max_prefills_per_batch)

    def _install_swa_di_eviction_fix(self) -> None:
        """Retain the window-low sliding-window block on the DI KV-pull admission path.

        A disaggregated-inference consumer admits a request with ``num_external_computed_tokens``
        ``= N`` (num_prompt), but ``_update_waiting_for_remote_kv`` then resets ``num_computed`` to
        ``N - 1`` (the decoder recomputes the last prompt token). Anchoring the sliding-window
        skip at N instead of N-1 nulls one extra block — the window's oldest — at the alignment
        ``N % block_size == block_size - 1``; decode then attends that never-transferred null block
        and reads uninitialized K (garbled harmony header -> HTTP 500 / empty response).

        Fix: for a DI admission (``num_external_computed_tokens > 0``), anchor every
        ``SlidingWindowManager``'s skip count at ``num_computed - 1`` for the duration of the call.
        We wrap ``allocate_slots`` — not ``get_num_skipped_tokens`` directly — because one
        ``allocate_slots`` call consults the skip count in three places (block-count predictor,
        ``remove_skipped_blocks``, allocator) that must agree; anchoring one alone under-reserves
        by a block. The transfer count stays N (producer sends all N): a retain, not an under-pull.

        Instance-scoped to this scheduler's own ``kv_cache_manager`` (one per DP replica),
        idempotent, and a strict no-op for non-DI admissions and models with no SWA group.
        """
        from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

        kv_cache_manager = self.kv_cache_manager
        orig_allocate_slots = kv_cache_manager.allocate_slots
        if getattr(orig_allocate_slots, "_neuron_swa_di_eviction_installed", False):
            return  # idempotent

        # The gate below reads num_external_computed_tokens as a keyword. Assert the pinned vLLM
        # still accepts it so a signature change becomes a loud startup failure, never a silent
        # disable of this DI-correctness fix.
        assert (
            "num_external_computed_tokens"
            in inspect.signature(orig_allocate_slots).parameters
        ), "vLLM allocate_slots signature changed; re-validate the SWA DI fix."

        sliding_window_managers = tuple(
            mgr
            for mgr in getattr(kv_cache_manager.coordinator, "single_type_managers", ())
            if isinstance(mgr, SlidingWindowManager)
        )

        def _skip_anchored_at_recompute(
            mgr, num_computed_tokens, _orig=SlidingWindowManager.get_num_skipped_tokens
        ):
            # Anchor one token earlier; the original's max(0, ...) keeps short sequences at 0.
            return _orig(mgr, num_computed_tokens - 1)

        def allocate_slots(request, num_new_tokens, *args, **kwargs):
            # Both in-tree callers pass num_external_computed_tokens by keyword or omit it; the
            # default 0 (non-DI admission / post-transfer decode) is byte-identical to upstream.
            num_external_computed_tokens = kwargs.get("num_external_computed_tokens", 0)
            if num_external_computed_tokens <= 0 or not sliding_window_managers:
                return orig_allocate_slots(request, num_new_tokens, *args, **kwargs)

            for mgr in sliding_window_managers:
                mgr.get_num_skipped_tokens = _skip_anchored_at_recompute.__get__(
                    mgr, type(mgr)
                )
            try:
                return orig_allocate_slots(request, num_new_tokens, *args, **kwargs)
            finally:
                for mgr in sliding_window_managers:
                    mgr.__dict__.pop("get_num_skipped_tokens", None)

        allocate_slots._neuron_swa_di_eviction_installed = True
        kv_cache_manager.allocate_slots = allocate_slots
        logger.info(
            "Installed DI SWA window-low retention on scheduler's kv_cache_manager "
            "(%d sliding-window manager(s)).",
            len(sliding_window_managers),
        )

    def _cap_encoder_budget_to_vision_bucket(self, vllm_config: "VllmConfig") -> None:
        """Limit images-per-step so total vision tokens fit in one compiled bucket.

        The vision encoder is compiled for fixed bucket sizes (e.g. [2048, 4096]).
        If the scheduler sends more images than the largest bucket can hold,
        embed_multimodal overflows. Capping the budget makes the base scheduler
        defer excess images to subsequent prefill steps automatically.

        Budget is in embed (post-merge) space; buckets are in raw (pre-merge)
        token space. Conversion: raw_tokens = embeds * merge_factor.
        """
        vision_config_dict = vllm_config.additional_config.get("vision_neuron_config")
        if not vision_config_dict or self.max_num_encoder_input_tokens <= 0:
            return

        buckets = vision_config_dict.get("num_vision_tokens_buckets")
        if not buckets:
            return

        from vllm_neuron.utils.vision_utils import get_vision_token_merge_factor

        max_bucket = max(buckets)
        merge_factor = get_vision_token_merge_factor(vllm_config.model_config.hf_config)
        cap = max(1, max_bucket // merge_factor)

        if self.max_num_encoder_input_tokens > cap:
            logger.info(
                "Capping encoder_compute_budget from %d to %d embeds "
                "(max_vision_bucket=%d, merge_factor=%d)",
                self.max_num_encoder_input_tokens,
                cap,
                max_bucket,
                merge_factor,
            )
            self.max_num_encoder_input_tokens = cap

    @property
    def state(self) -> SchedulerState:
        """Get the current scheduler state.

        State is updated at the end of each schedule() call based on
        the scheduler output, using num_computed_tokens vs num_prompt_tokens to
        determine prefill vs decode phase.

        Returns:
            IDLE: No requests in system (waiting=[], running=[])
            ACTIVE_PREFILL: Last schedule() call scheduled prefill.
            This state indicates there are prefill request(s) in waiting or running queue,
            and we don't have a full batch of decode requests
            ACTIVE_DECODE: Last schedule() call scheduled decode,
            This state indicates the running queue only contains decode requests
        """
        return self._state

    @property
    def at_capacity(self) -> bool:
        """Check if the running queue is at capacity. The running queue
        may contain either prefill or decode requests.

        When at capacity, new prefill requests are blocked until decode
        requests complete and free up slots.

        Returns:
            True if running requests >= max_num_seqs, False otherwise.
        """
        return len(self.running) >= self.max_num_seqs

    @property
    def has_prefill_in_running(self) -> bool:
        """Check if any requests in the running queue are prefills

        When there are ongoing prefill requests in the running queue,
        new prefill requests are blocks and ongoing decode requests
        are held back from the base scheduler.

        Returns:
            True if any requests in the running queue is in prefill
            phase, False otherwise.
        """
        return any(self._is_prefill_request(req) for req in self.running)

    def _is_prefill_request(self, request: Request) -> bool:
        """Check if a request is in prefill phase.

        A request is in prefill if it hasn't finished processing all prompt tokens.
        This is independent of which queue (waiting/running) it's in.

        Args:
            request: The request to check.

        Returns:
            True if request is in prefill phase, False if in decode phase.
        """
        return request.num_computed_tokens < request.num_prompt_tokens

    def _update_state(self, scheduler_output: "SchedulerOutput") -> None:
        """Update state based on scheduler output after scheduling.

        State is determined by checking if any scheduled request is in prefill
        phase using the SchedulerOutput's num_computed_tokens. This is different
        from the request.num_computed_tokens which gets incremented before
        going through model runner.

        Args:
            scheduler_output: The output from schedule().
        """
        old_state = self._state

        if len(self.waiting) == 0 and len(self.running) == 0:
            self._state = SchedulerState.IDLE
        elif scheduler_output.num_scheduled_tokens:
            # Check if any scheduled request is in prefill phase
            # Note: Need to use SchedulerOutput data which has pre-increment num_computed_tokens
            # The request.num_computed_tokens will already be incremented at this point
            has_prefill = False

            # Check new requests (NewRequestData has num_computed_tokens)
            for new_req in scheduler_output.scheduled_new_reqs:
                request = self.requests.get(new_req.req_id)
                if request is not None:
                    # Use num_computed_tokens from SchedulerOutput, not request
                    if new_req.num_computed_tokens < request.num_prompt_tokens:
                        has_prefill = True
                        break

            # Check cached requests if no prefill found yet
            if not has_prefill and scheduler_output.scheduled_cached_reqs is not None:
                cached_reqs = scheduler_output.scheduled_cached_reqs
                for i, req_id in enumerate(cached_reqs.req_ids):
                    request = self.requests.get(req_id)
                    if request is not None:
                        num_computed = cached_reqs.num_computed_tokens[i]
                        if num_computed < request.num_prompt_tokens:
                            has_prefill = True
                            break

            if has_prefill:
                self._state = SchedulerState.ACTIVE_PREFILL
            else:
                self._state = SchedulerState.ACTIVE_DECODE
        # else: keep previous state if nothing was scheduled

        if self._state != old_state:
            logger.debug(
                "State transition: %s -> %s",
                old_state,
                self._state,
            )

    def _fits_in_bucket(self, num_tokens: int) -> bool:
        """Check if token count fits within available buckets.

        Args:
            num_tokens: Number of tokens to check.

        Returns:
            True if tokens fit in any bucket, False otherwise.
        """
        if not self.num_batched_tokens_buckets:
            return True  # No buckets configured, allow all
        return num_tokens <= self.num_batched_tokens_buckets[-1]

    def _calculate_padded_count(self, token_count: int) -> int:
        """Calculate padded token count using bucket list.

        Finds the smallest bucket that can accommodate the token count.

        Args:
            token_count: Original number of tokens.

        Returns:
            Padded token count (from bucket list).
        """
        if token_count == 0:
            logger.warning("Request has zero tokens, skipping padding")
            return 0

        return get_bucket_for_count(token_count, self.num_batched_tokens_buckets)

    def can_schedule(self, request: Request) -> bool:
        """Admission control for scheduling requests.

        Determines if a new request can be scheduled based on:
        - Scheduler state
        - Decode batch capacity
        - Prefill batch capacity (max 1 prefill at a time)
        - KV cache concurrency cap (worst-case max_model_len per request)

        Note: We don't check bucket fit here because chunked prefill allows
        long prompts to be processed in multiple iterations. Each chunk will
        fit in a bucket. The max_model_len limit is enforced by vLLM.

        Args:
            request: The request to check.

        Returns:
            True if the new request can be scheduled, False otherwise.
        """
        # Scheduler is already running at capacity
        if self.at_capacity:
            return False

        # There is already an active prefill request in the running queue
        if self.has_prefill_in_running:
            return False

        # Check prefill batch capacity (generative: 1; pooling+pack: up to
        # max_num_seqs).
        if len(self.waiting) >= self.max_prefills_per_batch:
            return False

        # Pooling+pack: multiple sequences are PACKED into one prefill bucket, so
        # the COMBINED token count (already-admitted in self.waiting + this
        # candidate) must fit the largest bucket. Otherwise stop admitting and
        # let the already-admitted ones run this step (the candidate waits for
        # the next batch). The generative path skips this (single prefill;
        # chunked prefill handles length).
        if self.pooling_pack and self.num_batched_tokens_buckets:
            already = sum(
                self.requests[r.request_id].num_prompt_tokens
                for r in self.waiting
                if r.request_id in self.requests
            )
            candidate = request.num_prompt_tokens
            if already > 0 and already + candidate > self.num_batched_tokens_buckets[-1]:
                logger.debug(
                    "Pooling pack full: %d already + %d candidate > bucket %d; "
                    "deferring this request to next batch.",
                    already, candidate, self.num_batched_tokens_buckets[-1],
                )
                return False

        # Prevent livelock: don't admit a new prefill if running requests
        # already saturate KV cache capacity at worst-case (max_model_len).
        # This is conservative — it doesn't account for prefix cache sharing
        # — but guarantees no decode-decode preemption.
        if len(self.running) >= self._max_kv_concurrent:
            if not self._kv_exhaustion_warned:
                logger.warning(
                    "Prefill deferred: %d running requests already at KV "
                    "cache concurrency limit (%d, assuming max_model_len=%d "
                    "per request). Waiting for decode requests to complete. "
                    "To increase concurrency, raise "
                    "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION or reduce "
                    "max_model_len.",
                    len(self.running),
                    self._max_kv_concurrent,
                    self.max_model_len,
                )
                self._kv_exhaustion_warned = True
            else:
                logger.debug(
                    "Prefill deferred: %d running >= KV limit %d.",
                    len(self.running),
                    self._max_kv_concurrent,
                )
            return False

        return True

    def _should_skip_prefilling_padding_in_di(
        self, scheduler_output: "SchedulerOutput", req_id: str
    ) -> bool:
        """Skip prefill padding on the decode node in disaggregated inference.

        In DI, the decode node receives KV transfers from the prefill node.
        Padding should only be applied on the prefill node, not the decode node.

        Args:
            scheduler_output: The scheduler output to check.
            req_id: The request ID to check.

        Returns:
            True if padding should be skipped (decode node in DI), False otherwise.
        """
        request = self.requests.get(req_id)
        if request is None:
            return False
        kv_transfer_params = request.kv_transfer_params
        return (
            scheduler_output.kv_connector_metadata is not None
            and kv_transfer_params is not None
            and not kv_transfer_params.get("do_remote_decode", False)
        )

    def _apply_padding_and_log_stats(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "SchedulerOutput":
        """Apply padding to scheduler output for the prefill request.

        Since max_prefills_per_batch=1, there's at most one prefill request.
        We determine prefill status from SchedulerOutput data (which has
        pre-increment num_computed_tokens) rather than request state.

        Populates num_scheduled_tokens_padded for the prefill request while
        keeping num_scheduled_tokens as actual (unpadded) token counts.

        Args:
            scheduler_output: Original scheduler output.

        Returns:
            Modified scheduler output with padded token counts.
        """
        # Initialize padded counts dictionary if needed
        # Use getattr/setattr since num_scheduled_tokens_padded is not part of
        # the base vLLM SchedulerOutput dataclass
        if (
            not hasattr(scheduler_output, "num_scheduled_tokens_padded")
            or getattr(scheduler_output, "num_scheduled_tokens_padded") is None
        ):
            scheduler_output.num_scheduled_tokens_padded = {}

        # Build a map of req_id -> num_computed_tokens from SchedulerOutput
        # This has the pre-increment values (before _update_after_schedule)
        req_num_computed: dict[str, int] = {}

        # From new requests
        for new_req in scheduler_output.scheduled_new_reqs:
            req_num_computed[new_req.req_id] = new_req.num_computed_tokens

        # From cached requests
        if scheduler_output.scheduled_cached_reqs is not None:
            cached_reqs = scheduler_output.scheduled_cached_reqs
            for i, req_id in enumerate(cached_reqs.req_ids):
                req_num_computed[req_id] = cached_reqs.num_computed_tokens[i]

        # Pooling+pack: PACK all prefill sequences into ONE bucket (throughput).
        # The generative path below is left entirely unchanged.
        if self.pooling_pack:
            return self._apply_pooling_pack_padding(scheduler_output, req_num_computed)

        prefill_count = 0
        decode_count = 0
        total_padding_this_batch = 0

        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            if request is None:
                continue

            # Use num_computed_tokens from SchedulerOutput (pre-increment value)
            num_computed = req_num_computed.get(req_id, request.num_computed_tokens)
            is_prefill = num_computed < request.num_prompt_tokens

            # With the isolated-decode bench connector, the whole prompt is
            # reported as external KV and vLLM re-computes only the last prompt
            # token to get logits for sampling. That's 1 token of work — treat
            # it as decode, not prefill. If classified as prefill, it pads to a
            # prefill bucket size while idle DP ranks run the decode NEFF,
            # causing a shape mismatch in EP collectives. Scoped to that
            # connector: regular DI (NixlConnector) works without this, and
            # segmented prefill also produces 1-token re-computes.
            if (
                is_prefill
                and num_tokens == 1
                and num_computed == request.num_prompt_tokens - 1
                and self._is_decode_bench_connector
            ):
                is_prefill = False
                logger.debug(
                    "Async KV last-token re-compute for request %s: "
                    "treating as decode (1 token, num_computed=%d)",
                    req_id,
                    num_computed,
                )

            if is_prefill:
                # Skip padding on decode node in DI setting
                if self._should_skip_prefilling_padding_in_di(scheduler_output, req_id):
                    decode_count += 1
                    scheduler_output.num_scheduled_tokens_padded[req_id] = num_tokens
                    logger.debug(
                        "DI decode node: skipping padding for prefill request %s",
                        req_id,
                    )
                    continue

                prefill_count += 1
                padded_tokens = self._calculate_padded_count(num_tokens)
                padding_added = padded_tokens - num_tokens

                # Store padded count separately
                scheduler_output.num_scheduled_tokens_padded[req_id] = padded_tokens

                # Observe padded token count metric
                bucket_name = f"prefill_s{padded_tokens}"
                NUM_BATCHED_TOKENS_PADDING.labels(
                    model_name=self.model_name,
                    bucket_name=bucket_name,
                ).observe(padded_tokens)

                # Track statistics
                self.total_padding_tokens += padding_added
                self.total_scheduled_tokens += num_tokens
                total_padding_this_batch += padding_added

                # Log padding info
                if num_tokens > 0:
                    overhead_pct = (padding_added / num_tokens) * 100
                    logger.debug(
                        "PADDING: Prefill request %s: %d -> %d tokens "
                        "(+%d padding, %.1f%% overhead)",
                        req_id,
                        num_tokens,
                        padded_tokens,
                        padding_added,
                        overhead_pct,
                    )
            else:
                decode_count += 1
                scheduler_output.num_scheduled_tokens_padded[req_id] = num_tokens
                logger.debug(
                    "Decode request %s: %d tokens (no padding)", req_id, num_tokens
                )

        # Log batch summary
        if prefill_count > 0 and total_padding_this_batch > 0:
            original_total = sum(
                num_tokens
                for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items()
                if self.requests.get(req_id)
                and req_num_computed.get(req_id, 0)
                < self.requests[req_id].num_prompt_tokens
            )
            if original_total > 0:
                batch_overhead_pct = (total_padding_this_batch / original_total) * 100
                logger.debug(
                    "Batch padding summary: %d prefill reqs, "
                    "%d padding tokens (%.1f%% overhead)",
                    prefill_count,
                    total_padding_this_batch,
                    batch_overhead_pct,
                )

        # Observe padded batch size metric for decode
        if decode_count > 0:
            padded_batch_size = get_bucket_for_count(
                decode_count, self.num_seqs_buckets
            )
            bucket_name = f"decode_b{padded_batch_size}"
            NUM_SEQS_PADDING.labels(
                model_name=self.model_name,
                bucket_name=bucket_name,
            ).observe(decode_count)

        # Safety check: Neuron requires prefill and decode to be separate
        if prefill_count > 0 and decode_count > 0:
            raise RuntimeError(
                f"Neuron constraint violated: Mixed prefill ({prefill_count}) "
                f"and decode ({decode_count}) in the same batch. This indicates a bug "
                "in the scheduling logic."
            )

        # Log cumulative statistics periodically
        if self.log_stats and self.total_scheduled_tokens > 0:
            overhead = (self.total_padding_tokens / self.total_scheduled_tokens) * 100
            logger.debug(
                "Cumulative padding overhead: %d / %d = %.2f%%",
                self.total_padding_tokens,
                self.total_scheduled_tokens,
                overhead,
            )

        return scheduler_output

    def _apply_pooling_pack_padding(
        self,
        scheduler_output: "SchedulerOutput",
        req_num_computed: dict[str, int],
    ) -> "SchedulerOutput":
        """Pooling-only: pack all prefill sequences into ONE bucket.

        Unlike the generative path (one prefill, padded to its own bucket), an
        embedding batch holds several short sequences that we want to run in a
        single prefill. We size the batch to ONE bucket = bucket(sum of actual
        tokens), and put ALL of the padding on the LAST request. The runner's
        _create_padded_inputs pooling branch then lays tokens out densely as
        [seq0_real, seq1_real, ..., trailing_pad], which is exactly what the
        upstream pooling cursor (built from the ACTUAL per-seq token counts) and
        the model's per-sequence key_bounds expect. num_scheduled_tokens stays
        the ACTUAL per-seq count (those are the pooling boundaries); only
        num_scheduled_tokens_padded is inflated, and only on the final request,
        to reach the bucket total.

        Pooling is prefill-only, so there are no decode requests to consider.
        """
        # Per-req actual prefill token counts, in scheduler dict order.
        actual = {}
        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            if request is None:
                continue
            actual[req_id] = num_tokens

        total_actual = sum(actual.values())
        if total_actual == 0:
            return scheduler_output

        bucket_total = self._calculate_padded_count(total_actual)
        pad_total = bucket_total - total_actual

        # Default: padded == actual for every request; dump all padding on the last.
        req_ids = list(actual.keys())
        for req_id in req_ids:
            scheduler_output.num_scheduled_tokens_padded[req_id] = actual[req_id]
        last = req_ids[-1]
        scheduler_output.num_scheduled_tokens_padded[last] = actual[last] + pad_total

        # Stats / metric (mirror the generative branch's accounting).
        self.total_padding_tokens += pad_total
        self.total_scheduled_tokens += total_actual
        bucket_name = f"prefill_s{bucket_total}"
        NUM_BATCHED_TOKENS_PADDING.labels(
            model_name=self.model_name, bucket_name=bucket_name
        ).observe(bucket_total)
        logger.debug(
            "POOLING PACK: %d seqs, %d actual tokens -> bucket %d (+%d pad on %s)",
            len(req_ids), total_actual, bucket_total, pad_total, last,
        )
        return scheduler_output

    def schedule(self) -> "SchedulerOutput":
        """Schedule requests with prefill/decode separation and bucket padding.

        Flow:
        1. Move all waiting requests to holdback queue
        2. Selectively restore based on can_schedule() (includes KV capacity check)
        3. Separate prefill and decode based on state
        4. Delegate to parent scheduler
        5. Restore queues
        6. Apply bucket padding to prefill request
        7. Attach grammar bitmask for structured outputs
        8. Update state

        Returns:
            SchedulerOutput with padded token counts for prefill requests.
        """
        logger.debug(
            "schedule() called in state %s (waiting=%d, running=%d)",
            self._state,
            len(self.waiting),
            len(self.running),
        )

        # Step 1: Move all waiting to holdback queue (preserving priority)
        while self.waiting:
            self.holdback_queue.append(self.waiting.popleft())

        # Step 2: Selectively restore based on can_schedule()
        while self.holdback_queue:
            if self.can_schedule(self.holdback_queue[0]):
                self.waiting.append(self.holdback_queue.popleft())
            else:
                # Stop to preserve priority order
                break

        # Step 3: Separate prefill/decode
        running_holdback: list[Request] = []
        if self.has_prefill_in_running:
            # 3.1 Ongoing prefill segments in running queue - hide decode requests
            running_holdback = [
                req for req in self.running if not self._is_prefill_request(req)
            ]
            self.running = [
                req for req in self.running if self._is_prefill_request(req)
            ]
            logger.debug(
                "Scheduling prefill step: keeping %d prefill in running, "
                "hiding %d decode, %d in waiting, %d in holdback",
                len(self.running),
                len(running_holdback),
                len(self.waiting),
                len(self.holdback_queue),
            )
        elif len(self.waiting) > 0:
            # 3.2 New requests waiting for prefill - hide decode requests
            running_holdback = self.running
            self.running = []
            logger.debug(
                "Scheduling prefill step: %s request(s), "
                "holding back %s waiting, %s running",
                len(self.waiting),
                len(self.holdback_queue),
                len(running_holdback),
            )
        else:
            # 3.3 No prefill requests - no need to hide decodes
            logger.debug(
                "Scheduling decode step: scheduling %d decode request(s)",
                len(self.running),
            )

        # Step 4: Delegate to parent scheduler (sync or async)
        # Reduce capacity by hidden requests so base scheduler doesn't over-admit.
        # Without this, the base scheduler sees self.running == [] and admits up
        # to max_num_running_reqs from waiting; step 5 then restores the hidden
        # decodes and the total exceeds the cap, tripping
        # `assert len(self.running) <= self.max_num_running_reqs`.
        # finally: max_num_running_reqs persists on the instance, so a raise
        # from _call_base_schedule must not leave it permanently decremented.
        hidden_count = len(running_holdback)
        if hidden_count > 0:
            self.max_num_running_reqs -= hidden_count
        try:
            scheduler_output = self._call_base_schedule()
        finally:
            if hidden_count > 0:
                self.max_num_running_reqs += hidden_count

        # Step 5: Restore holdbacks
        self.running = self.running + running_holdback
        while self.holdback_queue:
            self.waiting.append(self.holdback_queue.popleft())

        # Step 6: Apply padding to prefill request
        scheduler_output = self._apply_padding_and_log_stats(scheduler_output)

        # Step 7: Attach grammar bitmask for structured outputs
        # Rows = logit rows (SO reqs + spec), NOT prompt tokens / prefill bucket
        # TODO: Consider upstreaming via SamplingMetadata instead of SchedulerOutput
        grammar_output = self.get_grammar_bitmask(scheduler_output)
        if grammar_output is not None and grammar_output.grammar_bitmask is not None:
            scheduler_output._grammar_bitmask = grammar_output.grammar_bitmask
            scheduler_output._structured_output_request_ids = (
                grammar_output.structured_output_request_ids
            )
            logger.debug(
                "[SO] scheduler: attached bitmask ids=%s bitmask_shape=%s",
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask.shape,
            )

        # Step 8: Update states
        self._update_state(scheduler_output)

        return scheduler_output

    def _call_base_schedule(self) -> "SchedulerOutput":
        """Delegate to base vllm Scheduler. Override for async behavior."""
        return Scheduler.schedule(self)


class NeuronAsyncScheduler(NeuronScheduler, AsyncScheduler):
    """
    Async Neuron scheduler combining:
    - NeuronScheduler: Neuron-specific scheduling (holdback, prefill/decode separation, padding)
    - AsyncScheduler: Output placeholders, async token handling
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: "MultiModalRegistry | None" = None,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            hash_block_size=hash_block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

    def _call_base_schedule(self) -> "SchedulerOutput":
        """Override to use AsyncScheduler.schedule() instead of Scheduler.schedule().

        NeuronScheduler.schedule() calls this method to delegate to the base scheduler.
        For async scheduling, we need AsyncScheduler's behavior.
        """
        return AsyncScheduler.schedule(self)

    def _update_after_schedule(self, scheduler_output: "SchedulerOutput") -> None:
        """Override to suppress spec-token placeholders near max_model_len.

        ``AsyncScheduler._update_after_schedule`` unconditionally sets
        ``request.spec_token_ids = [-1] * num_spec_tokens`` so the next
        scheduling step reserves ``1 + num_spec`` slots per request. Near
        ``max_model_len``, the scheduler's own trim logic
        (``num_new = min(1 + num_spec, max_model_len - 1 - num_computed)``)
        then produces partial shapes (e.g. 3 or 2 tokens instead of 4).
        Those partial shapes trigger a recompile against the target NEFF
        that was only warmed for the full ``1 + num_spec`` bucket.

        On Neuron the draft NEFF is a single compiled graph with fixed
        input shape, so we can't vary draft count per request. The clean
        transition is: near max_model_len, stop proposing drafts entirely
        and fall through to the non-spec target NEFF (which is warmed at
        ``decode_b{bs}_s1``). We signal "no drafts next step" by clearing
        ``spec_token_ids`` instead of setting placeholders, whenever the
        next step's full spec bucket wouldn't fit within max_model_len.

        Engine-core-side note: ``post_step`` only calls
        ``update_draft_token_ids`` when async scheduling is disabled
        (see ``vllm/v1/engine/core.py``). So the worker's
        ``take_draft_token_ids`` value is never propagated to the scheduler
        in async mode; the placeholder must be gated here instead.
        """
        # Advance num_computed_tokens via the plain Scheduler path, skipping
        # AsyncScheduler's unconditional placeholder injection.
        Scheduler._update_after_schedule(self, scheduler_output)

        # Re-apply AsyncScheduler's bookkeeping. The placeholder-count
        # increment (``+= 1 + cur_num_spec_tokens``) is required for both
        # spec and non-spec async paths — ``_update_request_with_output``
        # decrements it once per emitted token, so skipping the increment
        # would assert-fail. The spec-only logic (sticky disable near
        # max_model_len, ``spec_token_ids`` placeholder injection) is
        # gated on ``num_spec_tokens > 0``.
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Scheduler trims when `num_new = min(1 + num_spec, max_model_len - 1 - num_computed)`
        # falls below `1 + num_spec`. To avoid trim: `num_computed <= max_model_len - 2 - num_spec`.
        # Note: `num_computed` can TEMPORARILY go back down when
        # `update_from_output` applies rejection correction from an
        # in-flight step. That can make a previously-disabled spec
        # placeholder re-enable, producing positions that regress into
        # slots already written by later steps. We keep a sticky
        # ``_async_spec_disabled`` flag per request to prevent that
        # oscillation: once disabled, stays disabled for the rest of the
        # request's lifetime.
        # Extra margin (another 1+num_spec) so we transition to non-spec
        # before the last "safe" spec step. In async mode, the spec→non-spec
        # transition step fires the fallback path in the worker because the
        # previous step's sampled-token future has shape [bs, num_spec+1]
        # while the current non-spec input_ids is [bs]. That fallback
        # materializes async output and rebuilds inputs — but reads
        # input_ids from positions still reflecting the scheduler's
        # OPTIMISTIC num_computed_tokens (off by the previous step's
        # rejections). Transitioning earlier avoids the last spec step,
        # which tends to have heavy rejection near max_model_len.
        is_spec = self.num_spec_tokens > 0
        safe_num_computed = (
            self.max_model_len - 3 - 2 * self.num_spec_tokens if is_spec else 0
        )
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            if not is_spec:
                continue
            # Sticky: once disabled for this request, stay disabled.
            disabled = getattr(request, "_async_spec_disabled", False)
            if not disabled and request.num_computed_tokens > safe_num_computed:
                disabled = True
                request._async_spec_disabled = True
            if disabled:
                request.spec_token_ids = []
            else:
                request.spec_token_ids = self._spec_token_placeholders
