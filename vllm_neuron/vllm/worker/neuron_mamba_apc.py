# SPDX-License-Identifier: Apache-2.0
"""Neuron-native mamba prefix-cache (APC / align-mode) state copy.

T2 cross-request prefix caching for the hybrid GatedDeltaNet MoE model.

This is a faithful port of upstream ``vllm.v1.worker.mamba_utils``
``preprocess_mamba`` / ``postprocess_mamba`` bookkeeping (per-request
``mamba_state_idx`` tracking + src/dst block-index math), with ONE
Neuron-specific substitution:

  * Upstream ``collect_mamba_copy_meta`` + ``do_mamba_copy_block`` execute the
    block->block copy as a Triton ``batch_memcpy`` over raw ``data_ptr()``
    arrays. Triton does NOT run on Neuron, and ``data_ptr()`` byte-copies are
    not expressible on a Device Tensor. We instead accumulate ``(src_block_id,
    dst_block_id)`` pairs per paged state tensor and execute a per-row slice
    ``copy_`` (``state[dst] <- gathered_src``) on the CONTIGUOUS packed slab
    views (VLLM_GDN_SLAB_PACKED). NOTE: a batched ``index_copy_`` was tried first
    but crashes on device ("Can't call ReserveSpace on shared storage") because
    the slab view aliases shared storage; the slice ``copy_`` is a scratch-free
    DMA (the proven decode-path idiom). See ``_execute_plan``.

The paged state tensors are the model's bound ``[num_blocks, *state_shape]``
slab views (``lin.conv_state`` / ``lin.recurrent_state``), passed in as a
``{layer_name: [state0, state1, ...]}`` map. Their per-layer order matches the
tuple returned by ``model.get_mamba_state_copy_func()`` (conv first, then
temporal for GatedDeltaNet), exactly as upstream zips them.

ALL of this is gated by the caller on ``mamba_cache_mode == 'align'`` /
``enable_prefix_caching``; when off, nothing here is invoked and the runner is
byte-identical to the non-APC path.
"""
from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

import logging
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class NeuronMambaCopyState:
    """Holds the paged mamba state views + group metadata for align-mode APC.

    ``state_tensors`` maps a mamba layer_name to the list of its paged
    ``[num_blocks, *state_shape]`` views (order == get_mamba_state_copy_func
    tuple order: conv_state, recurrent_state for GatedDeltaNet).
    """

    mamba_group_ids: list[int]
    mamba_spec: MambaSpec
    state_tensors: dict[str, list[torch.Tensor]]
    # UNIFIED-CACHE: per (layer_name, state_pos) the CONTIGUOUS raw slab [n_blocks, page_elems]
    # (dtype-viewed) whose row k is GDN block k's whole page. The eager block->block copy uses
    # THIS (whole-page contiguous copy is RT-safe) instead of the page-strided component view
    # (a strided-row copy trips nrt_tensor_copy status=2). Empty on the default/packed path ->
    # the copy falls back to the component-view path (contiguous there). See _execute_plan.
    raw_slabs: dict[tuple, torch.Tensor] = dataclasses.field(default_factory=dict)

    @classmethod
    def create(
        cls,
        kv_cache_config: KVCacheConfig,
        state_tensors: dict[str, list[torch.Tensor]],
        copy_funcs: tuple = (),
        raw_slabs: dict | None = None,
    ) -> "NeuronMambaCopyState":
        # copy_funcs accepted for call-compat with the current runner (which
        # passes model.get_mamba_state_copy_func()); the working block-pair
        # executor addresses paged slab views directly and does not need them
        # FOR ALIGN (all copies are full-block, so per-state conv-vs-temporal
        # copy-func identity is unneeded). `all` mode / spec-decode partial-block
        # copies WOULD need them (upstream uses the identity there) — wire then.
        mamba_group_ids: list[int] = []
        mamba_specs: list[MambaSpec] = []
        for i in range(len(kv_cache_config.kv_cache_groups)):
            spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec
            if isinstance(spec, MambaSpec):
                mamba_group_ids.append(i)
                mamba_specs.append(spec)
        assert len(mamba_group_ids) > 0, "no mamba layers in the model"
        assert all(mamba_specs[0] == s for s in mamba_specs)
        return cls(
            mamba_group_ids=mamba_group_ids,
            mamba_spec=mamba_specs[0],
            state_tensors=state_tensors,
            raw_slabs=raw_slabs or {},
        )


def _collect_block_pairs(
    plan: dict[tuple[str, int], list[tuple[int, int]]],
    copy_state: NeuronMambaCopyState,
    kv_cache_config: KVCacheConfig,
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_block_ids: list[list[int]],
) -> None:
    """Mirror upstream ``collect_mamba_copy_meta`` block resolution, but record
    ``(src_block_id, dst_block_id)`` per paged state tensor instead of pointers.
    The plan is keyed by ``(layer_name, state_pos)`` — unique per physical
    paged tensor (multiple layer_names share a group id, so the group id alone
    is NOT a unique key).

    ``accept_token_bias`` is (num_accepted_tokens - 1). For align-mode WITHOUT
    speculative decoding it is always 0 (postprocess/preprocess reset accepted
    counts to 1), giving a clean full-block copy. A non-zero bias would require
    an intra-block offset copy (spec-decode partial block) — not expressible as
    a full-block index_copy; we fail loud rather than silently mis-copy.
    """
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return
    num_accepted = accept_token_bias + 1
    if num_accepted != 1:
        raise NotImplementedError(
            "Neuron mamba align-mode APC does not support speculative-decode "
            f"partial-block copies (num_accepted_tokens={num_accepted}). "
            "Only full-block (num_accepted_tokens==1) copies are wired."
        )
    for gid in copy_state.mamba_group_ids:
        block_ids = req_block_ids[gid]
        # conv uses cur_block_idx; temporal uses cur_block_idx + num_accepted-1.
        # With num_accepted==1 both resolve to block_ids[src_block_idx].
        src_block_id = int(block_ids[src_block_idx])
        dst_block_id = int(block_ids[dest_block_idx])
        if src_block_id == dst_block_id:
            continue
        layer_names = kv_cache_config.kv_cache_groups[gid].layer_names
        for layer_name in layer_names:
            for state_pos in range(len(copy_state.state_tensors[layer_name])):
                plan.setdefault((layer_name, state_pos), []).append(
                    (src_block_id, dst_block_id)
                )


def _execute_plan(
    plan: dict[tuple[str, int], list[tuple[int, int]]],
    copy_state: NeuronMambaCopyState,
) -> None:
    """Run the accumulated block->block copies via a PER-ROW slice ``copy_`` on the
    paged slab views (NOT index_copy_ — that hit the "Can't call ReserveSpace on shared
    storage" crash on the slab-aliased view; see the inline note below and the module
    header). index_select gathers all srcs into a fresh temp FIRST (breaking src/dst
    aliasing), then each row is written back with state[dst].copy_(gathered[j])."""
    # UNIFIED-CACHE dedup: conv (state_pos 0) and recurrent (state_pos 1) of a layer share the
    # SAME physical raw slab (row k = block k's WHOLE page — all components co-located), so a
    # single whole-page copy moves both. Without dedup the loop copies each page TWICE (once per
    # state_pos) and, worse, a src==dst chain across the two passes can latently alias. Track the
    # (slab storage, block-pair set) already executed this call and skip the redundant pass.
    _done_raw: set = set()
    for (layer_name, state_pos), pairs in plan.items():
        if not pairs:
            continue
        # DEFENSIVE BOUNDS GUARD (2026-07-14, task #106): under concurrent serve the shared BlockPool
        # hands the mamba group physical pool block ids that can EXCEED the mamba state slab's own row
        # count (device-confirmed crash: `TDRV:dmem_copy ... out of bounds (ret=-7)` at a 32-wide decode
        # block-open step; a raw `state[dst_block].copy_()` with dst_block >= slab rows is an OOB DMA that
        # HARD-CRASHES all 8 ranks / the whole engine). Until the root sizing/remap fix lands (see
        # APC_ON_SERVE_CRASH_DEBUG.md HANDOFF), filter out-of-range pairs and warn LOUDLY instead of
        # issuing the fatal copy: this keeps the engine alive (a skipped state migration degrades at most
        # ONE request's recurrent state; the alternative kills the entire batch). NOT the root fix.
        _bound_t = copy_state.raw_slabs.get((layer_name, state_pos))
        if _bound_t is None:
            _bound_t = copy_state.state_tensors[layer_name][state_pos]
        _n_rows = int(_bound_t.shape[0])
        _safe = [(s, d) for (s, d) in pairs if 0 <= s < _n_rows and 0 <= d < _n_rows]
        if len(_safe) != len(pairs):
            _bad = [(s, d) for (s, d) in pairs if not (0 <= s < _n_rows and 0 <= d < _n_rows)]
            logger.warning(
                "neuron_mamba_apc._execute_plan: %d/%d state-copy pair(s) reference a block id >= mamba "
                "slab rows (N=%d) for %s[state_pos=%d] — SKIPPING to avoid an OOB dmem_copy engine crash "
                "(task #106, APC concurrent block-id > mamba-slab-size). Offending pairs: %s",
                len(pairs) - len(_safe), len(pairs), _n_rows, layer_name, state_pos, _bad,
            )
        pairs = _safe
        if not pairs:
            continue
        # UNIFIED-CACHE: prefer the CONTIGUOUS raw slab (row k = GDN block k's whole page). A
        # whole-page copy raw[dst].copy_(raw[src].clone()) is a contiguous device copy the Neuron
        # RT accepts; the page-strided component view's strided-row copy trips nrt_tensor_copy
        # status=2. Disjoint block ids => a GDN page never holds KV, so copying the WHOLE page is
        # exactly the block-state move (conv+recurrent components co-move, so one raw-slab copy per
        # layer suffices — dedupe by layer so we don't copy the same page twice for state_pos 0/1).
        _raw = copy_state.raw_slabs.get((layer_name, state_pos))
        if _raw is not None:
            # Dedup key = physical storage ptr + the exact block-pair set. If another state_pos of
            # this layer already whole-page-copied the SAME slab with the SAME pairs, skip: the page
            # move already carried every component. (Different pairs on the same slab must still run.)
            _raw_key = (_raw.data_ptr(), tuple(pairs))
            if _raw_key in _done_raw:
                continue
            _done_raw.add(_raw_key)
            # _raw is CONTIGUOUS [n_blocks, page_elems] (row k = block k's whole page). Use the
            # PROVEN index_select-then-per-row-copy_ idiom (the hotfix's working path), NOT a
            # python-indexed .clone(): _raw[src_block] is a scalar-indexed row whose eager
            # nrt_tensor_copy the RT rejects (status=2, src/dst=_unknown_). index_select gathers all
            # src rows into ONE fresh contiguous tensor (breaks src/dst aliasing) via an op the RT
            # accepts; then per-row slice copy_ writes them back.
            _dev = _raw.device
            _src_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=_dev)
            _gathered = _raw.index_select(0, _src_idx)
            for j, (_, dst_block) in enumerate(pairs):
                _raw[dst_block].copy_(_gathered[j])
            continue
        state = copy_state.state_tensors[layer_name][state_pos]
        dev = state.device
        src_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=dev)
        # `state` is a CONTIGUOUS view aliasing the shared paged raw-slab storage
        # (VLLM_GDN_SLAB_PACKED). index_select gathers the source rows into a
        # FRESH tensor first (that op lowers fine and breaks src/dst aliasing, so
        # overlapping copies are safe). We then write each row back with a plain
        # slice `.copy_` rather than a batched `index_copy_`: index_copy_ tries to
        # ReserveSpace (scratch) on the shared-storage-backed view, which the
        # Neuron/XLA runtime rejects ("Can't call ReserveSpace on shared storage").
        # A per-row slice copy is the proven slab-write idiom (T_5f48855b) — no
        # scratch reservation, no indirect scatter DGE. `pairs` is small
        # (≈ one per prefix-extending request per step), so the unrolled loop is cheap.
        # UNIFIED-CACHE: under the page-strided shared-slab layout `state` is a NON-CONTIGUOUS
        # as_strided view; index_select on it raises "Expected self.is_contiguous()". This copy runs
        # EAGER (between NEFF calls), so a per-row gather+write via explicit index (no index_select,
        # no scratch) is correct and contiguity-agnostic: read src row (a strided row -> clone to a
        # fresh contiguous tensor), write dst row. Preserves the block->block move semantics exactly.
        if not state.is_contiguous():
            # Gather ALL src rows to fresh contiguous tensors FIRST (breaks src/dst aliasing, so an
            # overlapping copy where a src is also a later dst stays correct), THEN write each dst.
            _srcs = [state[src_block].clone() for src_block, _ in pairs]
            for j, (_, dst_block) in enumerate(pairs):
                state[dst_block].copy_(_srcs[j])
        else:
            gathered = state.index_select(0, src_idx)
            for j, (_, dst_block) in enumerate(pairs):
                state[dst_block].copy_(gathered[j])


def collect_carry_forward_ids(
    scheduler_output: Any,
    mamba_state_idx: dict[str, int],
    input_batch: Any,
    requests: dict[str, Any],
    copy_state: NeuronMambaCopyState,
    device: torch.device,
    max_rows: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """IN-GRAPH carry-forward (task #106 fix): instead of the eager per-row copy in
    _execute_plan (which overflows the DMA descriptor at concurrency -> dmem_copy ret=-7),
    resolve the SAME upstream block-index bookkeeping to a pair of FIXED-SHAPE int32 block-id
    vectors (src_ids, dst_ids) of length ``max_rows`` (== max_num_seqs), padded with -1.

    All GDN layers share ONE block table, so (src,dst) pairs are IDENTICAL across layers -> one
    vector pair per step, applied per-layer in the forward via paged_state_gather(src)->
    paged_state_scatter(dst): ONE .ap dispatch per layer (O(layers), not O(pairs*layers)),
    mirroring upstream's single batched batch_memcpy launch. Returns (src,dst) int32 on ``device``
    (-1 = skip), or None if no carry copies this step. Block-index math == verbatim upstream port."""
    mamba_spec = copy_state.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    block_size = mamba_spec.block_size
    gid = copy_state.mamba_group_ids[0]

    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        mamba_state_idx.pop(req_id, None)

    src_list: list[int] = []
    dst_list: list[int] = []
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)
            + num_speculative_blocks
        )
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            if (input_batch.num_accepted_tokens_cpu[i] - 1) != 0:
                raise NotImplementedError(
                    "Neuron mamba align-mode APC does not support spec-decode partial-block "
                    "carry copies (num_accepted != 1)."
                )
            block_ids = req_state.block_ids[gid]
            s = int(block_ids[prev_state_idx]); d = int(block_ids[curr_state_idx])
            if s != d:
                src_list.append(s); dst_list.append(d)
            input_batch.num_accepted_tokens_cpu[i] = 1
    if not src_list:
        return None
    src = torch.full((max_rows,), -1, dtype=torch.int32, device=device)
    dst = torch.full((max_rows,), -1, dtype=torch.int32, device=device)
    n = min(len(src_list), max_rows)
    src[:n] = torch.tensor(src_list[:n], dtype=torch.int32, device=device)
    dst[:n] = torch.tensor(dst_list[:n], dtype=torch.int32, device=device)
    return src, dst


def preprocess_mamba_neuron(
    scheduler_output: Any,
    kv_cache_config: KVCacheConfig,
    cache_config: Any,
    mamba_state_idx: dict[str, int],
    input_batch: Any,
    requests: dict[str, Any],
    copy_state: NeuronMambaCopyState,
) -> None:
    """Faithful port of upstream ``preprocess_mamba`` bookkeeping (block-index
    math verbatim), executing the copy via a PER-ROW slice ``copy_`` on the paged
    slab views (NOT index_copy_ — ReserveSpace crash; see _execute_plan).

    Runs BEFORE the forward: copies the previous step's running mamba state to
    the block that will hold this step's running state (prev_block -> curr_block
    on a cache-extended prefill)."""
    assert cache_config.enable_prefix_caching
    mamba_spec = copy_state.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    block_size = mamba_spec.block_size

    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(
        finished_req_ids, preempted_req_ids, resumed_req_ids
    ):
        mamba_state_idx.pop(req_id, None)

    plan: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)
            + num_speculative_blocks
        )
        # FAITHFUL ALIGN (upstream mamba_utils.py:204): the running recurrent/conv state FLOATS on
        # the last block (num_blocks-1-num_speculative_blocks). The runner's state_indices now
        # matches this via the align gather ((seq_lens-1)//block_size), so the model reads/writes
        # the running state at THIS same block, and this preprocess copies the prior running state
        # prev_block -> curr_block in lockstep when a step opens a new block. Block 0 and completed
        # full blocks are thus free to retain their frozen boundary state for cache-hit consumers.
        # (The earlier curr_state_idx=0 pin collapsed the running slot onto the reused block -> the
        # boundary state was clobbered -> reuse collapse; see doc/APC_DESIGN.md §3.)
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            _collect_block_pairs(
                plan,
                copy_state,
                kv_cache_config,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state.block_ids,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1
    _execute_plan(plan, copy_state)


def postprocess_mamba_neuron(
    scheduler_output: Any,
    kv_cache_config: KVCacheConfig,
    input_batch: Any,
    requests: dict[str, Any],
    mamba_state_idx: dict[str, int],
    copy_state: NeuronMambaCopyState,
) -> None:
    """Faithful port of upstream ``postprocess_mamba`` bookkeeping, executing
    the copy via a PER-ROW slice ``copy_`` on the paged slab views (NOT
    index_copy_ — ReserveSpace crash; see _execute_plan).

    Runs AFTER the forward: when a running-state block becomes a completed full
    block at a block boundary, persists (copies) that state into the full block
    so a later cache-hit request that shares the block reads it (GAP-3 fix)."""
    num_scheduled_tokens_dict = scheduler_output.num_scheduled_tokens
    scheduled_spec = scheduler_output.scheduled_spec_decode_tokens
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    mamba_spec = copy_state.mamba_spec

    plan: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        num_computed_tokens = req_state.num_computed_tokens
        num_draft_tokens = len(scheduled_spec.get(req_id, []))
        num_scheduled_tokens = num_scheduled_tokens_dict[req_id]
        num_accepted_tokens = num_accepted_tokens_cpu[i]
        num_tokens_running_state = (
            num_computed_tokens + num_scheduled_tokens - num_draft_tokens
        )
        new_num_computed_tokens = (
            num_tokens_running_state + num_accepted_tokens - 1
        )
        aligned_new_computed_tokens = (
            new_num_computed_tokens // mamba_spec.block_size * mamba_spec.block_size
        )
        if aligned_new_computed_tokens >= num_tokens_running_state:
            accept_token_bias = (
                aligned_new_computed_tokens - num_tokens_running_state
            )
            src_block_idx = mamba_state_idx[req_id]
            dest_block_idx = (
                aligned_new_computed_tokens // mamba_spec.block_size - 1
            )
            _collect_block_pairs(
                plan,
                copy_state,
                kv_cache_config,
                src_block_idx,
                dest_block_idx,
                accept_token_bias,
                req_state.block_ids,
            )
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1
    _execute_plan(plan, copy_state)
