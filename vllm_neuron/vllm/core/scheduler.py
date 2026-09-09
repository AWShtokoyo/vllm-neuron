# SPDX-License-Identifier: Apache-2.0
"""
Neuron Scheduler implementation for vLLM

This module provides a custom scheduler optimized for Neuron hardware.
It uses a holdback queue pattern for cleaner request management and
bucket-aware admission control.
"""

import logging
from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from vllm_neuron.metrics import NUM_BATCHED_TOKENS_PADDING, NUM_SEQS_PADDING
from vllm_neuron.utils.bucket_utils import (
    get_bucket_for_count,
    get_default_num_seqs_buckets,
    get_max_num_batched_tokens,
    resolve_num_batched_tokens_buckets,
    resolve_segmented_prefill_config,
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
        self._pool_exhaustion_warned: bool = False

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

        dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        dcp_stride = (
            dcp_size * vllm_config.cache_config.block_size if dcp_size > 1 else 1
        )
        self.num_batched_tokens_buckets = resolve_num_batched_tokens_buckets(
            max_num_batched_tokens,
            configured_buckets=neuron_config.get("num_batched_tokens_buckets"),
            use_configured_buckets=user_set_num_batched_tokens_buckets,
            auto_buckets=auto_num_batched_tokens_buckets,
            dcp_stride=dcp_stride,
        )

        if user_set_num_batched_tokens_buckets:
            logger.info(
                "Using num_batched_tokens_buckets from neuron_config: %s",
                self.num_batched_tokens_buckets,
            )
        elif auto_num_batched_tokens_buckets is not None:
            logger.info(
                "Segmented prefill auto-enabled; num_batched_tokens_buckets=%s",
                self.num_batched_tokens_buckets,
            )
        else:
            logger.info(
                "Using default num_batched_tokens_buckets: %s",
                self.num_batched_tokens_buckets,
            )

        # Prefill batch size limit (currently 1, may expand in future)
        self.max_prefills_per_batch: int = 1

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

        logger.info("Initialized NeuronAsyncScheduler for Neuron platform")
        logger.info("Max prefills per batch: %d", self.max_prefills_per_batch)

        # Maps mm_hash -> encoder-cache locator dict, carrying the locator from
        # the worker's ModelRunnerOutput (read in update_from_output) to the
        # response (emitted in _free_request). See those methods.
        #
        # TODO(vLLM PR #40695): this map and the update_from_output / _free_request
        # overrides below are an interim shim. When #40695 (ec_transfer_params on
        # EngineCoreOutput + scheduler-driven ec_connector.request_finished) lands
        # in the pinned vLLM, delete all three and use the stock path.
        self._ec_locator_by_mm_hash: dict[str, dict] = {}

    def update_from_output(self, scheduler_output, model_runner_output):
        """Record EC locators from the worker output and drop freed ones.

        Thin wrapper around the base implementation. Stock update_from_output
        never reads ec_connector_output for the EC connector (stock only wires
        the KV connector output), so we read it here after super() returns and
        stash mm_hash -> locator for _free_request to emit on the response.

        TODO(vLLM PR #40695): remove this override (see __init__).
        """
        ec_out = getattr(model_runner_output, "ec_connector_output", None)
        embed_locator = getattr(ec_out, "embed_locator", None)
        if embed_locator:
            self._ec_locator_by_mm_hash.update(embed_locator)

        result = super().update_from_output(scheduler_output, model_runner_output)

        # Drop locators whose cache blocks the engine freed this step.
        freed = getattr(scheduler_output, "free_encoder_mm_hashes", None)
        if freed:
            for mm_hash in freed:
                self._ec_locator_by_mm_hash.pop(mm_hash, None)
        return result

    def _free_request(self, request, delay_free_blocks: bool = False):
        """Attach this request's EC locators onto kv_transfer_params for the PD.

        The base _free_request returns the kv_transfer_params dict the scheduler
        puts on EngineCoreOutput; we merge this request's stashed locators under an
        "ec_locator" key so they ride that already-wired field to the router/PD.
        Borrowing the field is not using the KV connector slot.

        TODO(vLLM PR #40695): remove this override (see __init__).
        """
        kv_xfer_params = super()._free_request(request, delay_free_blocks)
        if not self._ec_locator_by_mm_hash:
            return kv_xfer_params
        # KNOWN LIMITATION (prototype): the emitted locator is not guaranteed
        # fresh at consume time -- the referenced VE block may be evicted/reused
        # before the PD pulls it (no transfer-completion pin; we rely on the VE
        # cache's min_hold_time window). The prod fix (stamp the block with
        # mm_hash so the PD validates the embedding, plus pin until consumed) is
        # planned for next milestone.
        ec_locator = {}
        for feature in getattr(request, "mm_features", None) or []:
            mm_hash = getattr(feature, "identifier", None)
            if mm_hash is not None and mm_hash in self._ec_locator_by_mm_hash:
                ec_locator[mm_hash] = self._ec_locator_by_mm_hash[mm_hash]
        if not ec_locator:
            return kv_xfer_params
        params = dict(kv_xfer_params) if kv_xfer_params else {}
        params["ec_locator"] = ec_locator
        return params

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

        # Check prefill batch capacity (only 1 prefill at a time)
        if len(self.waiting) >= self.max_prefills_per_batch:
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

        # Prevent the hybrid-pool livelock: the count-based `_max_kv_concurrent`
        # guard above bounds the NUMBER of running requests, but is blind to the
        # actual per-request block footprint of the shared KV pool. For a hybrid
        # (full-attention + GatedDeltaNet) model the GDN mamba page dominates, so
        # at small max_model_len only a handful of requests fit even though
        # `_max_kv_concurrent` is large (device: 3 decodes hold ~803 blocks each
        # of a 2713-block pool -> free=303, yet _max_kv_concurrent=542). Admitting
        # a 4th request there makes schedule() Step 3.2 hide all running decodes
        # for a prefill the base scheduler then cannot allocate (pool saturated)
        # and cannot preempt (decodes hidden) -> zero-token batch every step ->
        # livelock. Defer instead so decodes keep running, free blocks, and the
        # prefill is admitted once one completes.
        if not self._pool_admission_ok(request):
            return False

        return True

    def _pool_admission_ok(self, request: "Request") -> bool:
        """Whether admitting ``request`` fits the free KV block pool (worst case).

        Predicts the request's worst-case (full-sequence, no prefix reuse) block
        requirement with the KV manager's own coordinator predictor — the exact
        computation ``allocate_slots(full_sequence_must_fit=True)`` performs — and
        returns ``False`` if it would not fit in the currently free blocks. This
        is the physically correct admission condition for the hybrid unified pool,
        of which the token-based ``_max_kv_concurrent`` count is only a proxy that
        ignores the mamba/GatedDeltaNet page footprint.

        Strictly additive and fail-open: it can only DEFER admission (return
        False), never grant it, and any predictor error reverts to the prior
        behavior (no deferral from this check). Disable with
        ``VLLM_NEURON_POOL_ADMISSION_GATE=0`` (default on) — kept as an A/B
        kill-switch for the device livelock counter-test.
        """
        import os

        if os.environ.get("VLLM_NEURON_POOL_ADMISSION_GATE", "1") != "1":
            return True
        try:
            km = self.kv_cache_manager
            full_num_tokens = min(request.num_tokens, self.max_model_len)
            # Worst case: no prefix-cache reuse, request not yet registered.
            # Both the full-attention and (non-align) mamba managers read
            # req_to_blocks via .get(id, ()), so this is safe & side-effect-free
            # for an unadmitted request.
            need = km.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=km.empty_kv_cache_blocks.blocks,
                num_encoder_tokens=0,
                total_computed_tokens=0,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,
            )
            free = km.block_pool.get_num_free_blocks()
            if need > free:
                if not self._pool_exhaustion_warned:
                    logger.warning(
                        "Prefill deferred (hybrid pool): request needs %d KV "
                        "blocks worst-case but only %d free (%d running). "
                        "Waiting for a decode to complete. This bounds "
                        "concurrency by the real per-request block footprint "
                        "(mamba/GDN page), which the token-based concurrency "
                        "cap does not capture.",
                        need,
                        free,
                        len(self.running),
                    )
                    self._pool_exhaustion_warned = True
                else:
                    logger.debug(
                        "Prefill deferred (hybrid pool): need %d > free %d.",
                        need,
                        free,
                    )
                return False
            return True
        except Exception as exc:  # noqa: BLE001 - gate must never break scheduling
            logger.debug(
                "Pool admission predictor failed (%s); skipping pool gate.", exc
            )
            return True

    @staticmethod
    def _is_structured_output_grammar_waiter(request: Request) -> bool:
        return request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

    def _has_ready_structured_output_waiter(self) -> bool:
        """Return whether skipped_waiting has an SO request ready to prefill."""
        for req in self.skipped_waiting:
            if not self._is_structured_output_grammar_waiter(req):
                continue
            structured_output_req = req.structured_output_request
            if structured_output_req and structured_output_req.grammar:
                return True
        return False

    def _has_structured_output_waiter(self) -> bool:
        """Return whether skipped_waiting contains any SO grammar request."""
        return any(
            self._is_structured_output_grammar_waiter(req)
            for req in self.skipped_waiting
        )

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

    def _log_empty_batch_diag(
        self, scheduler_output: "SchedulerOutput", running_holdback: list
    ) -> None:
        """Log the cause of an empty schedule() step (diagnostic, default-OFF).

        Enabled only under VLLM_NEURON_SCHED_DIAG=1. Fires when the base
        scheduler scheduled zero tokens while there are unfinished requests
        (running or hidden) — i.e. the livelock signature. Reports the real free
        block count and each unfinished request's prefill/decode progress so the
        empty-batch mechanism can be identified from a single device run. Pure
        observation: no state is mutated.
        """
        import os

        if os.environ.get("VLLM_NEURON_SCHED_DIAG") != "1":
            return
        try:
            total_tok = getattr(scheduler_output, "total_num_scheduled_tokens", None)
            if total_tok is None:
                total_tok = sum(scheduler_output.num_scheduled_tokens.values())
            # visible running (self.running) + hidden decodes (running_holdback)
            live = list(self.running) + list(running_holdback)
            n_unfinished = len(live) + len(self.holdback_queue) + len(self.waiting)
            if total_tok > 0 or n_unfinished == 0:
                return  # not an empty-batch-with-work step

            free_blocks = None
            total_blocks = None
            usage = None
            try:
                bp = self.kv_cache_manager.block_pool
                free_blocks = bp.get_num_free_blocks()
                total_blocks = bp.num_gpu_blocks
                usage = bp.get_usage()
            except Exception:
                pass

            # Base-scheduler decisions this step: preemption (allocate_slots pool
            # pressure) and completions distinguish a pool-eviction churn from a
            # mamba-align reason-4 zero-chunk.
            preempted = getattr(scheduler_output, "preempted_req_ids", None)
            finished = getattr(scheduler_output, "finished_req_ids", None)
            n_preempted = len(preempted) if preempted else 0
            n_finished = len(finished) if finished else 0

            reasons = []
            for req in live:
                try:
                    ncomp = req.num_computed_tokens
                    ntok = req.num_tokens
                    nspec = getattr(req, "num_tokens_with_spec", ntok)
                    nph = getattr(req, "num_output_placeholders", 0)
                    nprompt = req.num_prompt_tokens
                    num_new = nspec + nph - ncomp
                    phase = "prefill" if ncomp < nprompt else "decode"
                    reasons.append(
                        f"{req.request_id[:8]}:{phase} "
                        f"comp={ncomp} prompt={nprompt} tok={ntok} "
                        f"spec={nspec} ph={nph} num_new={num_new}"
                    )
                except Exception as exc:  # noqa: BLE001
                    reasons.append(f"<req-introspect-failed: {exc}>")

            logger.warning(
                "[SCHED_DIAG] EMPTY BATCH with work: scheduled_tokens=0, "
                "state=%s, running(visible)=%d, hidden_decode=%d, holdback=%d, "
                "waiting=%d | free_blocks=%s/%s usage=%.3f | preempted=%d "
                "finished=%d | mamba_align_split=%s max_kv_concurrent=%s | "
                "reqs=[%s]",
                self._state,
                len(self.running),
                len(running_holdback),
                len(self.holdback_queue),
                len(self.waiting),
                free_blocks,
                total_blocks,
                usage if usage is not None else -1.0,
                n_preempted,
                n_finished,
                getattr(self, "need_mamba_block_aligned_split", "n/a"),
                getattr(self, "_max_kv_concurrent", "n/a"),
                "; ".join(reasons),
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must never break scheduling
            logger.warning("[SCHED_DIAG] diagnostic failed: %s", exc)

    def schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
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

        Args:
            throttle_prefills: DP prefill-balancing hint from the engine core;
                forwarded unchanged to the base scheduler.

        Returns:
            SchedulerOutput with padded token counts for prefill requests.
        """
        logger.debug(
            "schedule() called in state %s (waiting=%d, skipped_waiting=%d, "
            "running=%d)",
            self._state,
            len(self.waiting),
            len(self.skipped_waiting),
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

        # Step 3: Separate prefill/decode. Upstream can promote ready
        # structured-output grammar requests from skipped_waiting in this step.
        has_ready_structured_output_waiting = self._has_ready_structured_output_waiter()
        has_structured_output_waiting = self._has_structured_output_waiter()
        has_prefill_waiting = bool(self.waiting) or has_ready_structured_output_waiting
        can_start_prefill = (
            has_prefill_waiting
            and not self.at_capacity
            and not self.has_prefill_in_running
            and len(self.running) < self._max_kv_concurrent
        )
        running_holdback: list[Request] = []
        max_num_running_reqs_override: int | None = None
        # Whether THIS schedule() call hid decode requests to run a prefill-only step
        # (branch 3.1 = mid-chunk prefill in running; branch 3.2 = new waiting prefill).
        # Either way, if the prefill-only step comes back with 0 scheduled tokens we can
        # safely fall back to a DECODE-ONLY step over the hidden decodes (empty-batch fix).
        _hid_decodes_for_prefill = False
        if self.has_prefill_in_running:
            # 3.1 Ongoing prefill segments in running queue - hide decode requests
            running_holdback = [
                req for req in self.running if not self._is_prefill_request(req)
            ]
            self.running = [
                req for req in self.running if self._is_prefill_request(req)
            ]
            # Only mark for the decode-only fallback if we actually hid some decodes.
            _hid_decodes_for_prefill = len(running_holdback) > 0
            if has_structured_output_waiting:
                max_num_running_reqs_override = len(self.running)
            logger.debug(
                "Scheduling prefill step: keeping %d prefill in running, "
                "hiding %d decode, %d in waiting, %d in skipped_waiting, "
                "%d in holdback",
                len(self.running),
                len(running_holdback),
                len(self.waiting),
                len(self.skipped_waiting),
                len(self.holdback_queue),
            )
        elif can_start_prefill:
            # 3.2 New requests waiting for prefill - hide decode requests
            running_holdback = self.running
            self.running = []
            _hid_decodes_for_prefill = len(running_holdback) > 0
            if has_structured_output_waiting:
                available_slots = max(
                    0, self.max_num_running_reqs - len(running_holdback)
                )
                max_num_running_reqs_override = min(
                    self.max_prefills_per_batch,
                    available_slots,
                )
            logger.debug(
                "Scheduling prefill step: %s request(s), "
                "%s skipped request(s), holding back %s waiting, %s running, "
                "temporary max running=%s",
                len(self.waiting),
                len(self.skipped_waiting),
                len(self.holdback_queue),
                len(running_holdback),
                max_num_running_reqs_override,
            )
        else:
            # 3.3 No prefill requests - no need to hide decodes
            if (
                has_ready_structured_output_waiting
                and len(self.running) < self.max_num_running_reqs
            ):
                # Prevent ready SO waiters from admitting a prefill into a
                # decode batch after the Neuron pre-check.
                max_num_running_reqs_override = len(self.running)
            logger.debug(
                "Scheduling decode step: scheduling %d decode request(s), "
                "waiting=%d, skipped_waiting=%d, prefill_waiting=%s, "
                "structured_output_waiting=%s, temporary max running=%s",
                len(self.running),
                len(self.waiting),
                len(self.skipped_waiting),
                has_prefill_waiting,
                has_structured_output_waiting,
                max_num_running_reqs_override,
            )

        # Step 4: Delegate to parent scheduler (sync or async)
        _pre_running = len(self.running)
        _pre_waiting = len(self.waiting)
        _pre_holdback = len(self.holdback_queue)
        _pre_holdback_running = len(running_holdback)
        original_max_num_running_reqs = self.max_num_running_reqs
        if max_num_running_reqs_override is not None:
            self.max_num_running_reqs = max_num_running_reqs_override
        else:
            # Reduce capacity by hidden requests so base scheduler doesn't
            # over-admit. Without this, the base scheduler sees self.running
            # == [] and admits up to max_num_running_reqs from waiting; step 5
            # then restores the hidden decodes and the total exceeds the cap,
            # tripping `assert len(self.running) <= self.max_num_running_reqs`.
            # finally: max_num_running_reqs persists on the instance, so a
            # raise from _call_base_schedule must not leave it permanently
            # decremented.
            hidden_count = len(running_holdback)
            if hidden_count > 0:
                self.max_num_running_reqs -= hidden_count
        try:
            scheduler_output = self._call_base_schedule(throttle_prefills)
        finally:
            self.max_num_running_reqs = original_max_num_running_reqs

        # EMPTY-BATCH LIVELOCK FALLBACK: a prefill-only step (branch 3.1 mid-chunk prefill,
        # or 3.2 new waiting prefill) can come back with 0 scheduled tokens — e.g. the
        # prefill couldn't allocate blocks / advance its next chunk this step. Because we
        # hid the decodes, the base scheduler had nothing else to schedule, so it returns an
        # empty batch. EngineCore then loops forever on the 0-token step (core.py
        # sleep(0.001) + has_unfinished_requests) while workers spin in shm dequeue.
        # FIX: if a prefill-oriented step produced 0 tokens but we have decodes held back,
        # run a DECODE-ONLY step over those decodes instead. Decode-only is a legal Neuron
        # step (no prefill/decode mixing), guarantees forward progress, and defers the
        # blocked prefill one step. Works for BOTH branches: hide ALL prefills (the mid-chunk
        # prefill from 3.1 lives in self.running; new prefills are in self.waiting) into
        # holdback, restore the decodes, and re-run base as decode-only.
        #
        # This is the LAST-RESORT complement to _pool_admission_ok(): the gate DEFERS a
        # prefill before it can saturate the hybrid state pool (preventing the stall), while
        # this fallback guarantees progress if a 0-token prefill step happens anyway.
        if (
            not scheduler_output.total_num_scheduled_tokens
            and _hid_decodes_for_prefill
            and _pre_holdback_running > 0
        ):
            logger.warning(
                "[EMPTY-BATCH] prefill-only step scheduled 0 tokens "
                "(pre_running=%d waiting=%d holdback=%d held_decodes=%d); falling "
                "back to decode-only step to avoid livelock.",
                _pre_running, _pre_waiting, _pre_holdback, _pre_holdback_running,
            )
            # Temporarily hide any prefill reqs still in self.running (branch 3.1 mid-chunk
            # prefill, status==RUNNING) so the decode-only re-run does NOT mix prefill+decode.
            # These must be restored to self.RUNNING (NOT waiting — the base scheduler's
            # waiting loop assumes status==WAITING and would mishandle a RUNNING req), so we
            # stash them separately and re-append after the re-run.
            deferred_running_prefills = [
                r for r in self.running if self._is_prefill_request(r)
            ]
            self.running = [
                r for r in self.running if not self._is_prefill_request(r)
            ]
            # Restore the hidden decodes as the running set for the decode-only re-run.
            self.running = self.running + running_holdback
            running_holdback = deferred_running_prefills  # Step 5 re-appends to running
            # Hide new waiting prefills (branch 3.2, status==WAITING) in holdback; Step 5
            # restores them to self.waiting so they retry next step.
            while self.waiting:
                self.holdback_queue.append(self.waiting.popleft())
            # Re-apply the same hidden-capacity correction as above for the re-run.
            hidden_count = len(running_holdback)
            _rerun_orig_max = self.max_num_running_reqs
            if hidden_count > 0:
                self.max_num_running_reqs -= hidden_count
            try:
                scheduler_output = self._call_base_schedule(throttle_prefills)
            finally:
                self.max_num_running_reqs = _rerun_orig_max

        # DIAGNOSTIC (default-OFF): when VLLM_NEURON_SCHED_DIAG=1, log why a
        # schedule() step produced an empty batch while requests are unfinished
        # (the hybrid mml512/bs>1 livelock). Captures free-block count, per-req
        # progress, and the base scheduler's decisions so the empty-batch cause
        # (mamba-align reason-4 chunk vs allocate_slots preemption vs pool) can
        # be pinned on device. No behavioral change; pure observation.
        self._log_empty_batch_diag(scheduler_output, running_holdback)

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

    def _call_base_schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
        """Delegate to base vllm Scheduler. Override for async behavior."""
        return Scheduler.schedule(self, throttle_prefills)


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

    def _call_base_schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
        """Override to use AsyncScheduler.schedule() instead of Scheduler.schedule().

        NeuronScheduler.schedule() calls this method to delegate to the base scheduler.
        For async scheduling, we need AsyncScheduler's behavior.
        """
        return AsyncScheduler.schedule(self, throttle_prefills)

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
