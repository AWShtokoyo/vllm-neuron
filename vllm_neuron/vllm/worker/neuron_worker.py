# SPDX-License-Identifier: Apache-2.0
"""
NeuronWorker - Worker implementation for vLLM integration

This provides the main worker class that implements vLLM's WorkerBase interface
for AWS Neuron/Trainium hardware integration through vLLM Neuron.
"""

import logging
import os
import math
import re
import time
from datetime import timedelta
from typing import Any

import torch
import vllm.distributed.parallel_state as parallel_state
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    has_kv_transfer_group,
    get_kv_transfer_group,
)
from vllm.profiler.wrapper import TorchProfilerWrapper, WorkerProfiler
from vllm.utils.torch_utils import set_random_seed
from vllm.tasks import SupportedTask
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase
from vllm.v1.outputs import (
    DraftTokenIds,
    ModelRunnerOutput,
    AsyncModelRunnerOutput,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.v1.core.sched.output import GrammarOutput

from vllm_neuron import envs
from vllm_neuron.metrics import STARTUP_TIME
from vllm_neuron.model.interfaces import SupportsVisionWarmup
from vllm_neuron.utils.hardware_config import (
    get_efa_interface,
    parse_range_list,
)
from vllm.utils.network_utils import get_open_port
from vllm_neuron.parallel.neuron_parallel_state import tp_barrier
from vllm_neuron.vllm.platform import NeuronPlatform
from vllm_neuron.vllm.worker.neuron_profiler import (
    NeuronProfilerConfig,
    NeuronProfiler,
)

logger = logging.getLogger(__name__)

_LOOPBACK_ADDRS = (None, "", "127.0.0.1", "localhost", "::1")


def validate_cross_node_master_addr(
    nnodes: int,
    master_addr: str | None,
    data_parallel_master_ip: str | None,
) -> None:
    """Assert ``parallel_config.master_addr`` is set for the cross-node
    _WORLD rendezvous.

    vllm core's ``init_distributed_environment`` uses
    ``parallel_config.master_addr`` (not ``data_parallel_master_ip``) for the
    flat ``world_size_across_dp = TP*PP*DP`` group when ``nnodes>1``.
    ``master_addr`` defaults to the loopback ``"127.0.0.1"`` and is populated
    only by the ``--master-addr`` CLI flag — so a multi-pod DP launch that
    omits ``--master-addr`` leaves ``master_addr`` on loopback and the
    secondary pod's ranks deadlock rendezvousing on their own loopback.
    ``master_addr`` and ``data_parallel_master_ip`` represent two different
    things in vLLM (the TP/PP/DP world rendezvous vs. the DP coordinator), so
    we surface a clear error rather than silently substituting one for the
    other.
    """
    if nnodes > 1 and master_addr in _LOOPBACK_ADDRS:
        raise ValueError(
            f"nnodes={nnodes} but parallel_config.master_addr={master_addr!r} "
            "is the loopback default — the cross-node _WORLD rendezvous would "
            "deadlock on the secondary pod's own loopback. Pass "
            "--master-addr <HEAD_NODE_IP> on every pod (typically the same IP "
            f"as --data-parallel-address, e.g. {data_parallel_master_ip!r})."
        )


def resolve_ep_degree(
    ep_degree: int,
    world_size_across_dp: int,
    enable_expert_parallel: bool,
) -> int:
    """Resolve and validate the expert-parallel degree.

    Rules:
      * ``ep_degree > 1`` requires ``--enable-expert-parallel``.
      * When EP is enabled and ``ep_degree`` is left at its default of 1,
        it expands to ``world_size_across_dp`` (TP * DP) so experts span
        every rank, including across DP replicas.
      * The only hard constraint on an explicit ``ep_degree`` is that
        ``world_size_across_dp`` be divisible by it. Wide-EP layouts where
        ``ep_degree < TP * DP`` (e.g. TP=1 DP=128 EP=32 -> ep_tp=4) are
        valid: the model shards the MoE intermediate dim within each EP
        partition via ``_NEURON_EP_TP`` collectives. The previous strict
        ``ep_degree == world_size_across_dp`` check was the only thing
        blocking those topologies from configuring.

    Returns the resolved ``ep_degree``.
    """
    if not enable_expert_parallel:
        if ep_degree > 1:
            raise ValueError(
                f"ep_degree={ep_degree} requires --enable-expert-parallel to be set."
            )
        return ep_degree
    if ep_degree == 1:
        ep_degree = world_size_across_dp
    if world_size_across_dp % ep_degree != 0:
        raise ValueError(
            f"world_size (TP*DP={world_size_across_dp}) "
            f"must be divisible by ep_degree ({ep_degree})."
        )
    return ep_degree


def rendezvous_ccom_bootstrap():
    """Broadcast NEURON_RT_ROOT_COMM_ID from rank 0 via gloo.

    Must be called after ``init_distributed_environment()`` (gloo is up) and
    before the first NEFF execution (which triggers CCOM bootstrap).

    Rank 0 constructs the CCOM address from ``MASTER_ADDR`` and a free port,
    then broadcasts it to all workers. This ensures every worker — regardless
    of which DP engine spawned it — uses the same CCOM bootstrap endpoint.

    Works for all DP modes:
      - **No DP / external DP**: each process has its own gloo world with its
        own rank 0, so each independently picks its own port.
      - **Internal DP** (``--data-parallel-size > 1``): one gloo world spans
        all DP engines. Rank 0 (from engine DP0) broadcasts its address, so
        all TP*DP workers share one CCOM communicator.

    No-op in CPU mode (no CCOM needed).
    """
    import torch.distributed as dist

    if envs.VLLM_NEURON_CPU_MODE or envs.VLLM_NEURON_CPU_COMPILE:
        return

    if dist.get_rank() == 0:
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        port = get_open_port()
        addr = f"{master_addr}:{port}"
    else:
        addr = None

    addr_list = [addr]
    dist.broadcast_object_list(addr_list, src=0, device=torch.device("cpu"))
    addr = addr_list[0]

    os.environ["NEURON_RT_ROOT_COMM_ID"] = addr
    logger.info(
        "CCOM bootstrap rendezvous: NEURON_RT_ROOT_COMM_ID=%s (rank %d/%d)",
        addr,
        dist.get_rank(),
        dist.get_world_size(),
    )


class _SuppressModelRegistryOverwrite(logging.Filter):
    """Filter to suppress expected model registration overwrite warnings from vLLM.

    This filter is used when registering vLLM Neuron's Neuron-optimized model implementations
    with vLLM's ModelRegistry. Since we intentionally override vLLM's default models,
    the "already registered" warnings are expected and not actionable.
    """

    # Matches: "Model architecture LlamaForCausalLM is already registered, and will be overwritten..."
    _PATTERN = re.compile(
        r"Model architecture \w+ is already registered.*will be overwritten"
    )

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False to suppress matching messages, True to allow them through."""
        return not self._PATTERN.search(record.getMessage())


class NeuronWorker(WorkerBase):
    """
    Worker implementation for AWS Neuron/Trainium hardware.
    This class implements the vLLM WorkerBase interface.
    """

    def __init__(
        self,
        vllm_config: Any | None = None,
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method: str | None = None,
        is_driver_worker: bool = True,
    ):
        """
        Initialize the Neuron worker.

        Args:
            vllm_config: vLLM configuration object
            local_rank: Local rank of this worker
            rank: Global rank of this worker
            distributed_init_method: Method for distributed initialization
            is_driver_worker: Whether this is the driver worker
        """
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        self.use_async_scheduling = vllm_config.scheduler_config.async_scheduling

        self.vllm_config = vllm_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        self._startup_start_time = time.perf_counter()

        # Model runner will be created in init_device()
        # Following GPU/TPU worker pattern
        self.model_runner = None
        self.device = None

        # Register models with ModelRegistry
        # Put it in the worker as vLLM's model registry overwrites vllm_neuron model registry
        # if we put it too early in the platform.py.
        # We need to use or create a platform API for OOT platform model registry.
        # Tracked as TODO below.
        # TODO [CHRYS-72]: Investigate vLLM model registry
        from vllm import ModelRegistry
        from vllm_neuron.model import registry

        # Suppress the vLLM warning about overwriting existing model registrations.
        # This is expected behavior - we intentionally override vLLM's default model
        # implementations with vLLM Neuron's Neuron-optimized versions.
        vllm_registry_logger = logging.getLogger("vllm.model_executor.models.registry")
        suppress_filter = _SuppressModelRegistryOverwrite()
        vllm_registry_logger.addFilter(suppress_filter)
        try:
            for arch, model_cls in registry.get_models():
                ModelRegistry.register_model(arch, model_cls)
        finally:
            vllm_registry_logger.removeFilter(suppress_filter)

        self._profiler: WorkerProfiler | None = None

        logger.info(
            "Initialized NeuronWorker with rank %s, local_rank %s", rank, local_rank
        )

    def _set_efa_affinity(self, visible_devices) -> str:
        """
        Set EFA interface affinity for optimal network performance.

        Delegates the full device-to-EFA-interface resolution to
        hardware_config.get_efa_interface().

        Args:
            visible_devices: List of visible Neuron device indices

        Returns:
            str: Name of the EFA interface assigned to this host worker

        Raises:
            RuntimeError: If the Neuron device cannot be accessed or EFA interface not found
            IndexError: If the computed device index is out of bounds
        """
        efa = get_efa_interface(self.local_rank, visible_devices)
        os.environ["FI_EFA_IFACE"] = efa
        logger.debug(
            "Set EFA interface to %s for local_rank %s",
            efa,
            self.local_rank,
        )
        return efa

    def _set_cpu_affinity(self) -> None:
        """
        Set CPU affinity for all threads based on EFA's NUMA node.

        Pins all threads in the current process to CPUs in the same NUMA node as
        the EFA interface to minimize interconnection overhead and improve performance.
        This is critical for both single-instance and disaggregated inference workloads.

        Queries the EFA interface name from the FI_EFA_IFACE environment variable.
        """
        efa = os.environ.get("FI_EFA_IFACE")
        numa_node_path = (
            f"/sys/class/infiniband/{efa}/device/numa_node" if efa else None
        )

        if efa and numa_node_path and os.path.exists(numa_node_path):
            with open(f"/sys/class/infiniband/{efa}/device/numa_node") as f:
                numa = int(f.read())
            with open(f"/sys/devices/system/node/node{numa}/cpulist") as f:
                cpulist = parse_range_list(f.read())

            # Set affinity for all existing threads in the process
            # This handles any vLLM threads that may have been created before init_device()
            # e.g., ZeroMQ I/O or reaper threads
            threads_set = 0
            try:
                thread_ids = os.listdir("/proc/self/task")
                logger.debug(
                    "Setting affinity for %s threads to NUMA node %s",
                    len(thread_ids),
                    numa,
                )
                for tid in thread_ids:
                    try:
                        os.sched_setaffinity(int(tid), cpulist)
                        threads_set += 1
                    except OSError as ose:
                        # Thread may have exited or we don't have permission
                        logger.warning(
                            "Failed to set affinity for thread %s: %s", tid, ose
                        )
            except FileNotFoundError as fnfe:
                logger.warning(
                    "Could not enumerate threads (/proc/self/task not found): %s", fnfe
                )
                os.sched_setaffinity(0, cpulist)
                threads_set = 1

            logger.debug(
                "Set CPU affinity for %s thread(s) to NUMA node %s, CPUs: %s",
                threads_set,
                numa,
                cpulist,
            )
        else:
            logger.warning(
                "EFA device %s not found, skipping CPU affinity settings", efa
            )

    def init_device(self) -> None:
        """
        Initialize the device for this worker.

        Sets up Neuron environment variables, device, and creates model runner.
        Follows the vLLM GPU/TPU worker pattern.

        Initialization order:
        1. ``_init_neuron_distributed_environment_and_runtime()`` – affinity,
           env vars, gloo init, rendezvous_ccom_bootstrap, runtime.initialize(),
           ensure_kv_transfer_initialized
        2. Create torch device and model runner
        """
        # Pin device-worker CPU threads to 1. Workers inherit the frontend's
        # OMP_NUM_THREADS default (set for multimodal patchify), which skips
        # vLLM's own worker cap; without this reset the workers would
        # oversubscribe the host. Mirrors vllm_neuron/utils/executor.py.
        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        if envs.VLLM_NEURON_CPU_MODE:
            self.device = torch.device("cpu")
        elif envs.VLLM_NEURON_CPU_COMPILE:
            self.device = torch.device("meta")

        self._init_neuron_distributed_environment_and_runtime(
            self.vllm_config, self.rank, self.distributed_init_method, self.local_rank
        )

        if self._use_neuron_device():
            self.device = torch.device(f"neuron:{self.local_rank}")

        # Create model runner with vllm_config and device
        from .neuron_model_runner import NeuronModelRunner

        self.model_runner = NeuronModelRunner(
            vllm_config=self.vllm_config, device=self.device
        )

        # Set random seed
        set_random_seed(self.model_config.seed)

        logger.debug(
            "Device initialized: device=%s, local_rank=%s, rank=%s",
            self.device,
            self.local_rank,
            self.rank,
        )

    def _get_visible_devices(self):
        if "NEURON_RT_VISIBLE_CORES" in os.environ:
            raise RuntimeError(
                "NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution on vLLM. "
                "Set NEURON_VISIBLE_DEVICES to control device visibility."
            )
        # local_world_size is the per-node rank count within a DP replica
        # (TP*PP / nnodes_within_dp). With DP>1, parallel_config.world_size
        # is per-DP-replica and nnodes is the cluster total, so dividing
        # them directly yields the wrong answer for cross-DP-on-multi-node.
        num_local_ranks = self.vllm_config.parallel_config.local_world_size

        if "NEURON_VISIBLE_DEVICES" not in os.environ:
            # NEURON_VISIBLE_DEVICES must be set for libtorchneuron
            os.environ["NEURON_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in range(num_local_ranks)
            )

        neuron_visible_devices = os.getenv("NEURON_VISIBLE_DEVICES")
        neuron_visible_devices = parse_range_list(neuron_visible_devices)
        # Single-chip fan-out: on instances with a single Neuron device that
        # contains multiple logical cores (e.g. trn2.3xlarge has 1 device with
        # 4 logical cores at LNC=2), all TP ranks share the single chip and
        # each gets a distinct core via NEURON_RT_VISIBLE_CORES (set below).
        # Replicate the device id so every rank sees the shared chip.
        if len(neuron_visible_devices) == 1 and num_local_ranks > 1:
            logger.info(
                "Single-chip TP fan-out: %d ranks share device %s",
                num_local_ranks, neuron_visible_devices,
            )
            neuron_visible_devices = neuron_visible_devices * num_local_ranks
        assert len(neuron_visible_devices) == num_local_ranks
        return neuron_visible_devices

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        tp_rank = get_tp_group().rank_in_group
        return {tp_rank: metadata}

    def _patch_in_same_node_as_function(self):
        # WORKAROUND: Patch in_the_same_node_as() to avoid PyTorch barrier() bug with Neuron device backend
        #
        # THE PROBLEM:
        # vLLM's parallel_state.in_the_same_node_as() function calls torch.distributed.barrier(group=pg)
        # to synchronize ranks and find ranks that share the same shared memory.
        # However, PyTorch's barrier() implementation
        # (in torch/distributed/distributed_c10d.py, line ~4792) unconditionally calls:
        #     device = torch._C._get_accelerator()
        # BEFORE checking the backend type or dispatching to the backend-specific barrier implementation.
        #
        # For Neuron (which uses PyTorch's PrivateUse1 extension mechanism), _get_accelerator() requires
        # PrivateUse1HooksInterface to be registered. Since we haven't registered it, this causes:
        #     RuntimeError: Please register PrivateUse1HooksInterface by `RegisterPrivateUse1HooksInterface` first.
        #
        # This error occurs even though:
        # 1. The process group (pg) is using the Gloo backend (CPU-based)
        # 2. Gloo's barrier implementation doesn't need device/accelerator information
        # 3. The barrier would work fine if PyTorch didn't try to detect the device first
        #
        # CALL CHAIN WHERE THIS BREAKS:
        # init_model_parallel_group()
        #   → GroupCoordinator.__init__()
        #   → MessageQueue.create_from_process_group()
        #   → in_the_same_node_as(pg)
        #   → torch.distributed.barrier(group=pg)
        #   → torch._C._get_accelerator()  ← CRASH HERE
        #
        # THIS WORKAROUND:
        # We patch in_the_same_node_as() to bypass the barrier() call entirely and calculate
        # the result based on ParallelConfig, rather than probe the system. This allows:
        # - MessageQueue to use shared memory for inter-rank communication
        # - Process group initialization to complete successfully
        # - Single-node and multi-node distributed inference/training to work
        #
        # PROPER SOLUTIONS (either one will work):
        # 1. Register PrivateUse1HooksInterface for Neuron:
        #    - Will come with consolidation with Torch Eager
        # 2. Register vLLM Neuron plugin from a separate python package
        #    - Will mean that PrivateUse1 is not bound and hence error isn't hit.
        def patched_in_the_same_node_as(pg, source_rank=0):
            # Compute ranks_per_node from the cluster-wide world (TP*PP*DP),
            # not parallel_config.world_size (TP*PP per DP replica). With
            # DP>1, an EP/cross-DP group spans all nodes, and we need to
            # map each group-local rank back to its global rank to assign
            # it to a physical node correctly. Mirrors the patch in
            # neuron_parallel_state._ensure_vllm_parallel_state.
            parallel_config = self.vllm_config.parallel_config
            nnodes = parallel_config.nnodes
            ranks_per_node = max(parallel_config.world_size_across_dp // nnodes, 1)
            global_ranks = [
                torch.distributed.get_global_rank(pg, r)
                for r in range(torch.distributed.get_world_size(group=pg))
            ]
            source_node = global_ranks[source_rank] // ranks_per_node
            return [g // ranks_per_node == source_node for g in global_ranks]

        parallel_state.in_the_same_node_as = patched_in_the_same_node_as

    def _init_neuron_distributed_environment_and_runtime(
        self,
        vllm_config: VllmConfig,
        rank: int,
        distributed_init_method: str | None,
        local_rank: int,
    ) -> None:
        """Initialize the distributed environment and Neuron runtime.

        For non-CPU mode the full sequence is:
        1. Set EFA affinity for network performance
        2. Set CPU affinity to match EFA's NUMA node (all threads including main)
        3. Set visible cores, device count, and HBM mapping
        4. Initialize gloo distributed backend
        5. rendezvous_ccom_bootstrap  →  sets unique COMM_ID per server
        6. runtime.initialize()       →  latches COMM_ID, maps HBM
        7. ensure_kv_transfer_initialized  →  NIXL registers HBM memory
        """

        # --- Prepare runtime environment (affinity, env vars) ---------------
        if self._use_neuron_device():
            visible_devices = self._get_visible_devices()

            if os.environ.get("NEURON_SKIP_EFA_AFFINITY", "0") == "0":
                self._set_efa_affinity(visible_devices)
            self._set_cpu_affinity()

            # Set visible cores and platform device count. The worker thread
            # has one visible core, while the platform device count is the
            # number of visible cores on the local node.
            NeuronPlatform.set_device_count(len(visible_devices))
            # If several ranks share the same visible chip (single-chip fan-out
            # on trn2.3xlarge etc.), each rank must own a distinct core on that
            # chip. Compute the per-rank core id as this rank's position among
            # the ranks that share the same chip. For the common one-rank-per-
            # chip case the shared list contains only self.local_rank, so
            # _rank_within_chip == 0 and NEURON_RT_VISIBLE_CORES falls back to
            # the visible chip id -- unchanged from the previous behavior.
            _visible_chip = visible_devices[self.local_rank]
            _same_chip_ranks = [
                i for i, d in enumerate(visible_devices) if d == _visible_chip
            ]
            if len(_same_chip_ranks) > 1:
                _core_id = _same_chip_ranks.index(self.local_rank)
            else:
                _core_id = _visible_chip
            os.environ["NEURON_RT_VISIBLE_CORES"] = f"{_core_id}"

            # Enable HBM mapping (required for RDMA in disaggregated
            # inference and other use cases)
            os.environ["NEURON_RT_MAP_HBM"] = "1"

            # Default the runtime execution queue depth (8 -> 32) to avoid
            # intermittent Execution Queue Full issues under the async flow.
            # Temporary mitigation, tracking P443231617.
            os.environ.setdefault("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS", "32")

            # Default the DMA I/O ring cache (1 -> 32) so the runtime reuses
            # the per-step DMA ring across decode steps instead of rebuilding
            # it every step (kbl_exec_pre), which otherwise dominates
            # async-decode model-submit time. A cache-size sweep (see PR
            # description) shows 32 is the knee: it captures the full
            # latency win while the TPOT/HBM curves have both flattened, so
            # going higher only costs HBM for no gain. Override via env.
            # See P453665601.
            os.environ.setdefault("NEURON_RT_IO_RING_CACHE_SIZE", "32")

            # Disable the per-exec execution barrier: the cross-rank rendezvous
            # wait (enc_barrier) is the other major component of async-decode
            # model-submit time. Override via env. See P453665601.
            os.environ.setdefault("NEURON_RT_DISABLE_EXECUTION_BARRIER", "1")

        # --- Distributed environment init -----------------------------------
        parallel_config = vllm_config.parallel_config

        # vllm/distributed/parallel_state.py init_distributed_environment
        # rendezvouses the flat cross-node _WORLD group on
        # parallel_config.master_addr when nnodes>1; that field defaults to
        # "127.0.0.1" and is only set by --master-addr. Surface a clear error
        # (rather than letting the secondary pod deadlock on loopback) if
        # --master-addr was omitted. master_addr and data_parallel_master_ip
        # are distinct in vLLM, so we don't auto-substitute.
        validate_cross_node_master_addr(
            nnodes=parallel_config.nnodes,
            master_addr=parallel_config.master_addr,
            data_parallel_master_ip=parallel_config.data_parallel_master_ip,
        )

        self._patch_in_same_node_as_function()

        logger.debug(
            f"Initializing distributed environment: world_size={parallel_config.world_size}, {rank=}, {local_rank=}"
        )

        from vllm_neuron.parallel.neuron_parallel_state import (
            init_neuron_distributed_environment,
        )
        from vllm_neuron.envs import get_dist_backend

        neuron_config_dict = vllm_config.additional_config.get("neuron_config", {})
        dist_backend = get_dist_backend()

        ods_config = neuron_config_dict.get("on_device_sampling_config", {})
        sampling_dp_degree = (
            int(ods_config.get("sampling_dp_degree", 1)) if ods_config else 1
        )

        # Resolve ep_degree (defaults to TP*DP under EP) and validate it
        # against the cluster-wide world. See resolve_ep_degree for the rules.
        ep_degree = resolve_ep_degree(
            ep_degree=neuron_config_dict.get("ep_degree", 1),
            world_size_across_dp=parallel_config.world_size_across_dp,
            enable_expert_parallel=parallel_config.enable_expert_parallel,
        )

        attention_dp_size = neuron_config_dict.get("attention_dp_size", 1)
        embedding_dp_size = neuron_config_dict.get("embedding_dp_size", 1)
        lm_head_dp_size = neuron_config_dict.get("lm_head_dp_size", 1)
        mlp_dp_size = neuron_config_dict.get("mlp_dp_size", 1)

        vision_config_dict = vllm_config.additional_config.get(
            "vision_neuron_config", {}
        )
        from vllm_neuron.model.neuron_config import VisionNeuronConfig

        vnc = VisionNeuronConfig.from_dict(vision_config_dict or {})
        vision_tp_size, vision_dp_size = vnc.resolve_tp_dp(parallel_config.world_size)

        init_neuron_distributed_environment(
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=distributed_init_method,
            world_size=parallel_config.world_size,
            backend=dist_backend,
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            decode_context_parallel_size=parallel_config.decode_context_parallel_size,
            ep_degree=ep_degree,
            sampling_dp_degree=sampling_dp_degree,
            attention_dp_size=attention_dp_size,
            embedding_dp_size=embedding_dp_size,
            lm_head_dp_size=lm_head_dp_size,
            mlp_dp_size=mlp_dp_size,
            vision_tp_size=vision_tp_size,
            vision_dp_size=vision_dp_size,
            nnodes=parallel_config.nnodes,
            timeout=(
                timedelta(seconds=parallel_config.distributed_timeout_seconds)
                if parallel_config.distributed_timeout_seconds is not None
                else None
            ),
        )

        # Rendezvous CCOM bootstrap address via gloo (now that dist is up).
        rendezvous_ccom_bootstrap()

        # Initialize the Neuron runtime NOW — after rendezvous has set the
        # correct NEURON_RT_ROOT_COMM_ID but before NIXL/libfabric tries to
        # register device memory.  The runtime latches COMM_ID during
        # initialize() and never re-reads it, so the ordering is critical:
        #   1. gloo init  (above)
        #   2. rendezvous_ccom_bootstrap  →  sets unique COMM_ID per server
        #   3. runtime.initialize()       →  latches COMM_ID, maps HBM
        if self._use_neuron_device():
            runtime = torch.classes.neuron.Runtime()
            runtime.initialize()

    def load_model(self) -> None:
        """
        Load the model on this worker.

        This initializes the model runner and loads the model.
        After loading, attempts to fetch custom prefill buckets from
        the model if available.
        """
        from vllm.config import set_current_vllm_config

        logger.info("Loading model on NeuronWorker")
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()
            self.model_runner.init_tensor_replacement()

    def _get_local_worker_count(self) -> int:
        """Return number of worker processes sharing host memory on this node."""
        return max(self.vllm_config.parallel_config.local_world_size, 1)

    def _query_host_runtime_memory(self) -> int:
        """Query host available memory and return this worker's fair-share bytes."""
        from vllm_neuron.utils.memory import get_available_memory_bytes

        host_available = get_available_memory_bytes()

        if host_available <= 0:
            raise RuntimeError(
                f"Non-positive host available memory detected: {host_available} bytes."
            )

        local_workers = self._get_local_worker_count()
        fair_share = host_available // local_workers
        if fair_share <= 0:
            raise RuntimeError(
                "Host memory fair-share per worker is non-positive. "
                f"host_available={host_available} bytes, local_workers={local_workers}."
            )

        logger.info(
            "Host memory: %.2f GiB available, %d local worker(s), %.2f GiB fair-share per worker",
            host_available / (1024**3),
            local_workers,
            fair_share / (1024**3),
        )
        return fair_share

    def _query_runtime_memory(self) -> int:
        """Query Neuron runtime for available HBM (raw bytes_free)."""
        _, bytes_free = self._query_runtime_memory_stats()
        return bytes_free

    def _query_runtime_memory_stats(self) -> tuple[int, int]:
        """Query Neuron runtime for used/free HBM bytes."""
        runtime = torch.classes.neuron.Runtime()
        bytes_used, bytes_free = runtime.get_vnc_memory_stats()

        if bytes_used < 0 or bytes_free <= 0:
            raise RuntimeError(
                f"Neuron runtime reported unusable memory stats: "
                f"used={bytes_used}, free={bytes_free}"
            )

        logger.info(
            "Neuron HBM: %.2f GiB used, %.2f GiB free",
            bytes_used / (1024**3),
            bytes_free / (1024**3),
        )
        return bytes_used, bytes_free

    def _get_kv_cap_fraction(self) -> float:
        """Return validated KV cap fraction shared by CPU and Neuron modes."""
        cap_fraction = envs.VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION
        if cap_fraction <= 0 or cap_fraction > 1:
            raise RuntimeError(
                "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION must be in (0, 1]. "
                f"Got {cap_fraction}."
            )
        return cap_fraction

    def _determine_available_memory_cpu(self, gpu_mem_util: float) -> int:
        """Compute CPU-mode KV memory budget from fair-share host memory."""
        bytes_free = self._query_host_runtime_memory()
        total_budget = int(bytes_free * gpu_mem_util)
        cap_fraction = self._get_kv_cap_fraction()
        available = int(total_budget * cap_fraction)

        logger.debug(
            "CPU mode KV cache memory: %.2f GiB free-share × %.1f%% gpu_memory_utilization = %.2f GiB total budget, %.2f GiB available (cap_fraction=%.2f)",
            bytes_free / (1024**3),
            gpu_mem_util * 100,
            total_budget / (1024**3),
            available / (1024**3),
            cap_fraction,
        )
        return available

    def _compute_kv_budget(
        self,
        total_hbm_bytes: int,
        bytes_used: int,
        gpu_mem_util: float,
    ) -> int:
        """Shared KV cache budget logic for both estimate and runtime paths.

        Applies gpu_memory_utilization, cap-fraction guardrail, and min-KV
        threshold check, then returns the available bytes for KV cache.
        """
        total_budget = int(total_hbm_bytes * gpu_mem_util)
        # Preserve vLLM semantics: GMU applies to total device memory first.
        user_budget = max(total_budget - bytes_used, 0)

        # Apply a fixed cap to the GMU-scaled total HBM budget.
        # TODO: Replace this cap with a cleaner solution later.
        cap_fraction = self._get_kv_cap_fraction()
        heuristic_cap = int(total_budget * cap_fraction)
        available = max(min(user_budget, heuristic_cap), 0)

        logger.debug(
            "KV cache heuristic: user_budget=%.2f GiB, cap=%.2f GiB "
            "(total_budget=%.2f GiB, total_hbm=%.2f GiB, bytes_used=%.2f GiB, "
            "cap_fraction=%.2f), "
            "effective=%.2f GiB",
            user_budget / (1024**3),
            max(heuristic_cap, 0) / (1024**3),
            total_budget / (1024**3),
            total_hbm_bytes / (1024**3),
            bytes_used / (1024**3),
            cap_fraction,
            available / (1024**3),
        )

        min_kv_gib = envs.VLLM_NEURON_MIN_KV_BUDGET_GIB
        if min_kv_gib > 0:
            min_kv_bytes = int(min_kv_gib * (1024**3))
            if available < min_kv_bytes:
                raise RuntimeError(
                    "Computed KV cache budget is below minimum threshold. "
                    f"effective={available / (1024**3):.2f} GiB, "
                    f"minimum={min_kv_gib:.2f} GiB. "
                    "Increase gpu_memory_utilization or lower "
                    "VLLM_NEURON_MIN_KV_BUDGET_GIB."
                )

        logger.debug(
            "KV cache memory: %.2f GiB total × %.1f%% gpu_memory_utilization "
            "- %.2f GiB used = %.2f GiB user budget, %.2f GiB available",
            total_hbm_bytes / (1024**3),
            gpu_mem_util * 100,
            bytes_used / (1024**3),
            user_budget / (1024**3),
            available / (1024**3),
        )
        return available

    def _get_byte_used_from_model(self) -> int:
        """
        Use the model params and buffers to calculate the size of tensors
        allocated on device.
        """
        bytes_used_params = sum(p.nbytes for p in self.model_runner.model.parameters())
        bytes_used_buffers = sum(b.nbytes for b in self.model_runner.model.buffers())
        # Add spec decode model weights.
        if self.model_runner.drafter is not None:
            bytes_used_params += sum(
                p.nbytes for p in self.model_runner.drafter.model.parameters()
            )
            bytes_used_buffers += sum(
                b.nbytes for b in self.model_runner.drafter.model.buffers()
            )
        return bytes_used_params + bytes_used_buffers

    def _estimate_available_memory_neuron(self, gpu_mem_util: float) -> int:
        """
        Estimate KV budget from static HBM size and model parameter bytes.
        This is used during CPU Compilation flow and computes bytes used
        based on the model params and buffer sizes.
        """
        from vllm_neuron.compile.platform import get_total_available_memory

        total_hbm_bytes = get_total_available_memory() * 1024 * 1024 * 1024
        bytes_used = self._get_byte_used_from_model()
        return self._compute_kv_budget(total_hbm_bytes, bytes_used, gpu_mem_util)

    def _determine_available_memory_neuron(self, gpu_mem_util: float) -> int:
        """
        Compute Neuron-mode KV memory budget using static HBM size and model size

        Uses model parameter and buffer sizes for bytes_used (same as the CPU
        compile estimate path) instead of runtime.get_vnc_memory_stats().
        This ensures the KV cache block count is identical between CPU compile
        and execution, which is required for compilation cache hits.

        Without this, mismatch happens because runtime.get_vnc_memory_stats()
        itself uses ~3MB of device memory which goes unaccounted in CPU compile.
        Ignoring this is safe as the total memory is in order of GBs.
        """
        bytes_used, bytes_free = self._query_runtime_memory_stats()
        total_hbm_bytes = bytes_used + bytes_free
        bytes_used = self._get_byte_used_from_model()
        return self._compute_kv_budget(total_hbm_bytes, bytes_used, gpu_mem_util)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """
        Determine the amount of memory available for KV cache.

        In CPU mode, queries host available memory and uses a fair-share per
        local worker process. On Neuron device, queries runtime free HBM and
        applies gpu_memory_utilization.

        Note: In vLLM v1, gpu_memory_utilization is applied inside the worker
        (not by vLLM core). The GPU worker does the same pattern.

        Returns:
            int: Available memory in bytes (already scaled by gpu_memory_utilization)
        """
        gpu_mem_util = self.cache_config.gpu_memory_utilization
        if envs.VLLM_NEURON_CPU_COMPILE:
            avail = self._estimate_available_memory_neuron(gpu_mem_util)
        elif envs.VLLM_NEURON_CPU_MODE:
            avail = self._determine_available_memory_cpu(gpu_mem_util)
        else:
            avail = self._determine_available_memory_neuron(gpu_mem_util)
        return self._cap_kv_budget_for_unified_cache(avail)

    def _cap_kv_budget_for_unified_cache(self, avail_bytes: int) -> int:
        """Cap the KV budget so the shared unified slab stays within the .ap
        indirect-DMA addressing limits.

        Under VLLM_UNIFIED_KV_GATHER the full-attn KV and GDN state share ONE raw
        slab, read/written by .ap indirect DMA (functional/paged_kv_gather.py).

        HISTORY: the ORIGINAL bound was the 32-bit ELEMENT index — the kernels
        formed a flat `block*page_stride + intra` int32 offset that overflowed at
        num_blocks*page_elems >= 2**31 ("dst_indirect_max_index exceeds 32-bit
        range"), forcing num_blocks <= 16383 (~4 GiB) and HALVING usable KV. Both
        kernels were rewritten BLOCK-INDEXED (block base on the wide indirect axis,
        intra-page column as a small static offset), device-proven to read/write
        correctly at num_blocks=17000 (base offset 2.228e9 > 2**31). So the element
        bound no longer applies.

        REMAINING bound (device-discovered): the .ap indirect DIM's row count. The
        gather (read) resolves element addresses PAST 2**32 (device-verified block
        39999, base 5.24e9). The WRITE is an indirect SCATTER; on trn2 HWDGE has no
        indirect scatter, so SWDGE (uint32-addressed) is the only path and the
        resolved element address is hardware-capped at 2**32. Device-measured: write
        PASS to block 32767 (base 4.29e9, V-end exactly 2**32), FAIL at 32768. So the
        binding limit is now the SCATTER 2**32-element ceiling on the slab:
            num_blocks * page_elems <= 2**32   ->   32768 blocks / 8 GiB
        which is 2x the old int32 element cap (2**31 -> 16384 / 4 GiB). Full
        write->gather round-trip PROVEN at num_blocks=32768. To address >8 GiB in one
        slab would require host-side slab-sharding (per-request base pointer within a
        <=2**32 window) — a separate change, not a kernel one. No-op unless unified
        path on + hybrid model. See functional/paged_kv_gather.py.
        """
        import os as _os
        if not (_os.environ.get("VLLM_UNIFIED_KV_GATHER") == "1"
                or _os.environ.get("VLLM_KV_GATHER_KERNEL") == "1"):
            return avail_bytes
        try:
            kv_spec = self.get_kv_cache_spec()
        except Exception:
            return avail_bytes
        # Find the largest page_size among groups (the shared slab uses the uniform
        # / max page), the full-attn KV element size, and whether a Mamba group exists.
        from vllm.v1.kv_cache_interface import (
            MambaSpec, FullAttentionSpec, SlidingWindowSpec,
        )
        import torch as _torch
        page_bytes = 0
        is_hybrid = False
        dtype_elem = None  # derive from the full-attn KV dtype (NOT hardcoded — fp8/fp32 differ)
        for spec in kv_spec.values():
            pb = getattr(spec, "page_size_bytes", 0) or 0
            page_bytes = max(page_bytes, pb)
            if isinstance(spec, MambaSpec):
                is_hybrid = True
            if isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                _dt = getattr(spec, "dtype", None)
                if _dt is not None:
                    dtype_elem = _torch.empty(0, dtype=_dt).element_size()
        if not is_hybrid or page_bytes <= 0:
            return avail_bytes
        if dtype_elem is None:
            dtype_elem = 2  # no full-attn spec found; fall back to bf16 (shouldn't happen for hybrid)
        # The .ap index is an ELEMENT index in the SLAB's element size. The slab is int8-backed but
        # addressed per-component in that component's dtype; the tightest bound is the SMALLEST
        # element size present (more elements per page). Use the KV dtype (the full-attn region that
        # drives paged_kv_write/gather). page_elems = page_bytes // dtype_elem.
        page_elems = page_bytes // dtype_elem
        if page_elems <= 0:
            return avail_bytes
        # SCATTER-DGE 2**32-ELEMENT bound (device-measured, 2026-07-09). Both kernels were rewritten
        # block-indexed and validated: the GATHER (read) resolves element addresses past 2**32; the
        # WRITE (indirect SCATTER) resolves in UINT32 and is hardware-capped at 2**32 elements. On
        # trn2 HWDGE has no indirect gather/scatter, so SWDGE (uint32) is the only scatter path — no
        # kernel restructure exceeds 2**32 (verified: write PASS to block 32767 / base 4.29e9, FAIL
        # at 32768 / >2**32). So the slab (which the write must address) is bound by:
        #     num_blocks * page_elems <= 2**32
        # This is 2x the OLD int32 bound (2**31): 32768 blocks / 8 GiB vs the old 16384 / 4 GiB.
        # Round-trip PROVEN at num_blocks=32768 (V-region-end exactly 2**32). To exceed 8 GiB in one
        # slab would require host-side slab-sharding (per-request base pointer) — a separate change;
        # 8 GiB doubles unified KV vs the old cap and clears the KV portion of the 396K-hotfix compare.
        max_blocks = (2**32) // page_elems
        max_bytes = max_blocks * page_bytes
        if avail_bytes > max_bytes:
            logger.warning(
                "Unified KV cache: capping KV budget %.2f GiB -> %.2f GiB (num_blocks <= %d, "
                "page_elems=%d) — scatter-DGE 2**32-element (uint32) hardware ceiling on the WRITE. "
                "2x the old int32 cap; >8 GiB in one slab needs host-side slab-sharding.",
                avail_bytes / 2**30, max_bytes / 2**30, max_blocks, page_elems)
            return max_bytes
        return avail_bytes

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """
        Initialize the KV cache with the given number of blocks.

        Args:
            num_gpu_blocks: Number of GPU blocks to allocate
            num_cpu_blocks: Number of CPU blocks to allocate
        """
        logger.debug(
            "Initializing cache with %s GPU blocks, %s CPU blocks",
            num_gpu_blocks,
            num_cpu_blocks,
        )
        # TODO: Determine if initialize_cache() needs implementation for Neuron
        # Why required: Clarify whether explicit cache initialization is needed or handled by vLLM Neuron
        # Current: Empty method implementation
        # Target: Either implement cache allocation or document that it's handled elsewhere

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache from configuration object.

        This is the vLLM v1 API method that replaces initialize_cache.
        Required by vLLM v1 WorkerBase interface.

        Args:
            kv_cache_config: KV cache configuration object
        """
        logger.debug("Initializing KV cache from config (NeuronWorker)")

        # Initialize KV transfer connector before KV cache so register_kv_caches
        # can find the connector (has_kv_transfer_group() must be True).
        # Skip in CPU compile mode — no device memory to register.
        if not envs.VLLM_NEURON_CPU_COMPILE:
            ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        # Delegate to model runner to initialize KV cache
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> CompilationTimes:
        """
        Compile or warm up the model for optimal performance.

        Pre-compiles the model for all prefill bucket sizes and decode batch sizes
        to ensure optimal performance on Neuron hardware. Each bucket/batch size
        triggers a separate compilation that is cached by the Neuron compiler.

        In disaggregated inference (DI) mode, only the relevant warmup phase
        runs: prefill-only servers skip decode warmup, and decode-only servers
        skip prefill warmup. Servers with kv_role="kv_both" run both.

        Set VLLM_NEURON_SKIP_PREFILL_WARMUP=1 to skip prefill compilation
        without requiring a kv-transfer-config (useful for decode-only profiling).
        Set VLLM_NEURON_SKIP_DECODE_WARMUP=1 to skip decode compilation
        without requiring a kv-transfer-config (useful for prefill-only profiling).

        This method calls the model directly without using SchedulerOutput or
        execute_model(), avoiding unnecessary side effects on internal state.

        Returns:
            CompilationTimes: Compilation/warmup time.


        """
        if self.model_runner is None or self.model_runner.model is None:
            logger.warning(
                "Model not loaded yet. Skipping warmup. "
                "This may result in compilation happening during first inference."
            )
            return CompilationTimes(language_model=0.0, encoder=0.0)

        # SyntheticNeuronModel: skip warmup (no compiled graphs needed)
        from vllm_neuron.model.synthetic import SyntheticNeuronModel

        if isinstance(self.model_runner.model, SyntheticNeuronModel):
            logger.info("SyntheticNeuronModel detected — skipping warmup")
            return CompilationTimes(language_model=0.0, encoder=0.0)

        # CPU eager mode: skip warmup (no NEFFs to compile, shapes are dynamic)
        if envs.VLLM_NEURON_CPU_MODE and self.vllm_config.model_config.enforce_eager:
            logger.info("CPU eager mode — skipping warmup")
            return CompilationTimes(language_model=0.0, encoder=0.0)

        # Determine which warmup phases to run based on kv_role.
        # kv_producer = prefill server → only prefill warmup
        # kv_consumer = decode server  → only decode warmup
        # kv_both / no DI             → both
        warmup_start = time.perf_counter()
        kv_cfg = getattr(self.vllm_config, "kv_transfer_config", None)
        skip_prefill_warmup = (
            kv_cfg is not None and kv_cfg.is_kv_consumer and not kv_cfg.is_kv_producer
        ) or envs.VLLM_NEURON_SKIP_PREFILL_WARMUP
        skip_decode_warmup = (
            kv_cfg is not None and kv_cfg.is_kv_producer and not kv_cfg.is_kv_consumer
        ) or envs.VLLM_NEURON_SKIP_DECODE_WARMUP

        # === Graph extraction (capture HLO for prefill + decode + vision) ===
        # All extracts run before any warmup so the parallel-trace fork
        # children inherit a clean NRT/torch_xla state.

        # TODO: For E/P/D disaggregation, give skip_vision_warmup its own
        # derivation from kv_role (like skip_prefill/decode_warmup above).
        skip_vision_warmup = (
            self.model_runner.vision_neuron_config is None or skip_prefill_warmup
        )
        has_capture_backend = (
            self.model_runner.capture_backend_model is not None
            or self.model_runner.vision_capture_backend is not None
        )
        if has_capture_backend:
            if skip_prefill_warmup:
                logger.info(
                    "Skipping prefill graph extraction (kv_role=%s, decode-only server)",
                    kv_cfg.kv_role if kv_cfg is not None else "kv_consumer",
                )
            if skip_decode_warmup:
                logger.info(
                    "Skipping decode graph extraction (kv_role=%s, prefill-only server)",
                    kv_cfg.kv_role if kv_cfg is not None else "kv_producer",
                )
            self._extract_graphs(
                skip_prefill=skip_prefill_warmup,
                skip_decode=skip_decode_warmup,
                skip_vision=skip_vision_warmup,
            )
            logger.info("Barrier: waiting for all ranks to finish graph extraction")
            tp_barrier()
            self.model_runner.parallel_compile()
            logger.info("Barrier: waiting for all ranks to finish compilation")
            tp_barrier()

        # === Prefill warmup ===
        if skip_prefill_warmup:
            logger.info(
                "Skipping prefill warmup (kv_role=%s, this is a decode-only server)",
                kv_cfg.kv_role if kv_cfg is not None else "kv_consumer",
            )
        elif not envs.VLLM_NEURON_CPU_COMPILE:
            self._warmup_prefill()

        # === Decode warmup ===
        if skip_decode_warmup:
            logger.info(
                "Skipping decode warmup (kv_role=%s, this is a prefill-only server)",
                kv_cfg.kv_role if kv_cfg is not None else "kv_producer",
            )
        elif not envs.VLLM_NEURON_CPU_COMPILE:
            self._warmup_decode()

        # === Vision encoder warmup ===
        if skip_vision_warmup:
            if self.model_runner.vision_neuron_config is not None:
                logger.info(
                    "Skipping vision warmup (kv_role=%s, this is a decode-only server)",
                    kv_cfg.kv_role if kv_cfg is not None else "kv_consumer",
                )
        elif not envs.VLLM_NEURON_CPU_COMPILE:
            self._warmup_vision_encoder()

        # Enable tensor capture after warmup (skip warmup captures)
        if self.model_runner._capture_registry is not None:
            self.model_runner.enable_capture()
            logger.info("Tensor capture enabled after warmup")

        num_batched_tokens_buckets = (
            self.model_runner.neuron_config.num_batched_tokens_buckets
        )
        num_seqs_buckets = self.model_runner.neuron_config.num_seqs_buckets
        logger.info("=" * 80)
        logger.info(
            "Model warmup completed: %s num_batched_tokens_buckets, %s num_seqs_buckets",
            len(num_batched_tokens_buckets),
            len(num_seqs_buckets) if num_seqs_buckets else 0,
        )
        logger.info("=" * 80)

        STARTUP_TIME.labels(
            model_name=self.vllm_config.model_config.model,
        ).set(time.perf_counter() - self._startup_start_time)
        return CompilationTimes(
            language_model=time.perf_counter() - warmup_start, encoder=0.0
        )

    def _prefill_buckets(self) -> tuple[list[int], list[int]]:
        num_batched_tokens_buckets = (
            self.model_runner.neuron_config.num_batched_tokens_buckets
        )
        kv_segment_size_buckets = (
            self.model_runner.neuron_config.kv_segment_size_buckets
        )
        # Use [0] as a sentinel for non-segmented mode (cached_seq_len=0)
        effective_kv_buckets = (
            kv_segment_size_buckets if kv_segment_size_buckets else [0]
        )
        return num_batched_tokens_buckets, effective_kv_buckets

    def _build_prefill_trace_jobs(self, buckets: list[tuple[int, int]]) -> list[tuple]:
        """Build (callable, kwargs) jobs that capture the prefill HLO for
        each (bucket_size, kv_seg_size) pair, on the meta device. Each
        job, when invoked, drives one Dynamo trace and writes one
        ``graph.hlo`` to the cache. Eagle (drafter) graph extraction is
        not included here — that runs sequentially after the pool.

        Buckets are emitted **largest-first** (longest-processing-time-
        first scheduling). The fork pool's wall-clock equals the slowest
        lane in each wave; LPT minimizes the long-pole by ensuring big
        jobs get scheduled first while small jobs fill in around them.
        """
        # Sort descending. Trace cost grows with bucket_size primarily
        # and kv_seg_size secondarily; tuple ordering captures both.
        meta = torch.device("meta")
        jobs: list[tuple] = []
        for bucket_size, kv_seg_size in sorted(buckets, reverse=True):
            kwargs = self.model_runner._build_prefill_synthetic_inputs(
                bucket_size, kv_seg_size, device=meta
            )
            jobs.append((self.model_runner.capture_backend_model, kwargs))
        return jobs

    def _build_decode_trace_jobs(self, targets: list[tuple[int, int]]) -> list[tuple]:
        """Build (callable, kwargs) jobs that capture the decode HLO for
        each (batch, ctx_bucket) compile target, on the meta device.

        For spec-decode configs the target model has both with-spec and
        no-spec decode shapes — both flow through the pool. The drafter's
        decode graph runs sequentially in the post-pool fallback.

        Targets are emitted **largest-first** (LPT) — see
        ``_build_prefill_trace_jobs`` for rationale.
        """
        if not targets:
            return []
        spec_decode_enabled = (
            self.model_runner.drafter is not None
            and self.model_runner.speculative_config is not None
        )
        # Sort descending. Decode cost grows with both batch_size and
        # ctx_bucket; tuple ordering captures both.
        targets = sorted(targets, reverse=True)
        meta = torch.device("meta")
        jobs: list[tuple] = []
        for batch_size, ctx_bucket in targets:
            kwargs = self.model_runner._build_decode_synthetic_inputs(
                batch_size,
                256,
                spec_decode_enabled=spec_decode_enabled,
                ctx_bucket=ctx_bucket,
                device=meta,
            )
            jobs.append((self.model_runner.capture_backend_model, kwargs))
            if spec_decode_enabled:
                # Target model decode WITHOUT spec decode — separate
                # cache key, different decode_token_threshold.
                kwargs_no_spec = self.model_runner._build_decode_synthetic_inputs(
                    batch_size,
                    256,
                    spec_decode_enabled=False,
                    decode_token_threshold=1,
                    ctx_bucket=ctx_bucket,
                    device=meta,
                )
                jobs.append((self.model_runner.capture_backend_model, kwargs_no_spec))
        return jobs

    def _unwrap_vision_model(self) -> "SupportsVisionWarmup | None":
        """Unwrap model wrappers and return it if it implements SupportsVisionWarmup."""
        m = self.model_runner.model
        if type(m).__name__ == "OptimizedModule":
            m = m._orig_mod
        if type(m).__name__ == "TensorCaptureModel":
            m = m.model
        return m if isinstance(m, SupportsVisionWarmup) else None

    def _build_vision_trace_jobs(self) -> list[tuple]:
        """Build (callable, kwargs) jobs for vision encoder graph capture.

        Each job traces one vision bucket through the capture backend.
        Buckets are sorted largest-first (LPT scheduling).
        """
        if self.model_runner.vision_capture_backend is None:
            return []
        vnc = self.model_runner.vision_neuron_config
        if vnc is None or vnc.num_vision_tokens_buckets is None:
            return []

        unwrapped = self._unwrap_vision_model()
        if unwrapped is None:
            return []

        capture_backend = self.model_runner.vision_capture_backend
        meta = torch.device("meta")
        cache = self.model_runner.encoder_cache
        jobs: list[tuple] = []
        for bucket in sorted(vnc.num_vision_tokens_buckets, reverse=True):
            kwargs = unwrapped.build_vision_synthetic_inputs(bucket, vnc, meta)
            num_blocks = (
                math.ceil(
                    math.ceil(bucket / vnc.vision_attention_block_size) / vnc.dp_size
                )
                * vnc.dp_size
            )
            kwargs["encoder_cache_buffer"] = torch.empty(
                cache.num_blocks,
                cache.block_size,
                cache.fat_dim,
                dtype=cache.dtype,
                device=meta,
            )
            kwargs["write_block_ids"] = torch.zeros(
                num_blocks, dtype=torch.int64, device=meta
            )
            jobs.append((capture_backend, kwargs))
        return jobs

    def _run_parallel_trace_jobs(self, jobs: list[tuple]) -> bool:
        """Send target-model trace jobs to the parallel-trace fork pool.

        Returns True if the pool ran, False if it was bypassed via
        ``VLLM_NEURON_DISABLE_PARALLEL_TRACE`` (caller falls back to
        the in-process sequential path).
        """
        from vllm_neuron.compile.parallel_trace import parallel_trace

        if envs.VLLM_NEURON_DISABLE_PARALLEL_TRACE:
            return False
        parallel_trace(jobs=jobs, parent_rank=self.rank)
        return True

    def _prefill_compile_targets(self) -> list[tuple[int, int]]:
        """Enumerate (num_batched_tokens, kv_segment_size) pairs to compile.

        Cartesian product of the configured prefill buckets and KV
        segment buckets. Mirror of ``_decode_compile_targets`` — both
        return the flat tuple list the trace pool consumes.
        """
        num_batched_tokens_buckets, effective_kv_buckets = self._prefill_buckets()
        return [
            (bucket_size, kv_seg_size)
            for kv_seg_size in effective_kv_buckets
            for bucket_size in num_batched_tokens_buckets
        ]

    def _extract_graphs(
        self, *, skip_prefill: bool, skip_decode: bool, skip_vision: bool
    ) -> None:
        """Extract all target-model HLOs (prefill + decode + vision) in a
        single parallel-trace pass, then capture eagle drafter graphs in
        this process.

        Combining vision + prefill + decode into one fork pool is strictly better
        than three: it amortizes the per-child meta-swap (the dominant
        per-fork cost — ~3-4 GB / child) and gives the round-robin
        partitioner a bigger work pool to balance against, so the
        long-pole bucket has less idle time around it.

        Sequential fallback (when ``VLLM_NEURON_DISABLE_PARALLEL_TRACE``
        is set): runs extraction in the existing order, including drafter
        graph capture interleaved per-bucket. Eagle drafter graph extraction wraps
        ``propose()`` which isn't a torch.compile wrapper, so it can't slot into the
        ``(model, kwargs)`` primitive without a dedicated adapter — left as a follow-up.
        """
        prefill_buckets = [] if skip_prefill else self._prefill_compile_targets()
        decode_targets = [] if skip_decode else self._decode_compile_targets()
        vision_jobs = [] if skip_vision else self._build_vision_trace_jobs()

        if prefill_buckets:
            logger.info(
                "Extracting graphs for %d prefill buckets: %s",
                len(prefill_buckets),
                prefill_buckets,
            )
        if decode_targets:
            logger.info(
                "Extracting %d decode compile targets: %s",
                len(decode_targets),
                decode_targets,
            )

        if vision_jobs:
            logger.info(
                "Extracting graphs for %d vision buckets",
                len(vision_jobs),
            )

        prefill_jobs = self._build_prefill_trace_jobs(prefill_buckets)
        decode_jobs = self._build_decode_trace_jobs(decode_targets)

        target_jobs = prefill_jobs + decode_jobs + vision_jobs

        if self._run_parallel_trace_jobs(target_jobs):
            # Drafter graphs run sequentially against ``self.device`` —
            # the fork pool only handled target-model jobs.
            has_drafter = self.model_runner.drafter is not None
            if has_drafter and not skip_prefill:
                self._extract_prefill_drafter_graphs(prefill_buckets)
            if has_drafter and not skip_decode:
                self._extract_decode_drafter_graphs(decode_targets)
            return

        # Sequential fallback: in-process extraction (target + drafter
        # together) for each prefill bucket, then each decode target.
        if not skip_prefill:
            self._extract_prefill_graphs_sequential(prefill_buckets)
        if not skip_decode:
            self._extract_decode_graphs_sequential(decode_targets)
        if not skip_vision:
            self._extract_vision_graphs()

    def _extract_prefill_graphs_sequential(
        self, buckets: list[tuple[int, int]]
    ) -> None:
        """In-process prefill extraction for every bucket. Used when
        the parallel-trace pool is disabled (workers<=1).

        Delegates to ``model_runner.extract_prefill_graphs`` which
        captures both the target and drafter graphs together against
        ``self.device``."""
        total = len(buckets)
        for i, (bucket_size, kv_seg_size) in enumerate(buckets, 1):
            logger.info(
                "\n[%s/%s] Extracting for prefill: bucket_size=%s, kv_segment_size=%s",
                i,
                total,
                bucket_size,
                kv_seg_size,
            )
            try:
                self.model_runner.extract_prefill_graphs(bucket_size, kv_seg_size)
                logger.info(
                    "  Successfully extracted graph for prefill bucket %s "
                    "with kv_segment_size %s",
                    bucket_size,
                    kv_seg_size,
                )
            except Exception as e:
                logger.error(
                    "  Failed to extract graph for prefill bucket %s "
                    "with kv_segment_size %s: %s",
                    bucket_size,
                    kv_seg_size,
                    e,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Model graph extraction failed for prefill bucket {bucket_size} "
                    f"with kv_segment_size {kv_seg_size}. Error: {e}"
                ) from e

    def _extract_prefill_drafter_graphs(self, buckets: list[tuple[int, int]]) -> None:
        """Sequentially extract the eagle drafter prefill graphs.

        Runs in the parent process after the target-model fork pool
        completes, on ``self.device``. Caller must ensure a drafter is
        attached.
        """
        if self.model_runner.drafter is None:
            raise RuntimeError(
                "_extract_prefill_drafter_graphs called without a drafter "
                "attached; gate the call on self.model_runner.drafter."
            )
        for bucket_size, kv_seg_size in buckets:
            kwargs = self.model_runner._build_prefill_synthetic_inputs(
                bucket_size, kv_seg_size
            )
            logger.info(
                "Capturing EAGLE3 prefill graphs for bucket size: %d", bucket_size
            )
            self.model_runner.drafter.graph_extract(
                num_tokens=bucket_size,
                num_reqs=1,
                attn_metadata=kwargs["attn_metadata"],
            )

    def _extract_vision_graphs(self) -> None:
        """Extract vision encoder HLO graphs.

        Drives the capture backend directly with synthetic per-bucket inputs.
        """
        from vllm_neuron.compile.capture_backend import CaptureComplete

        vnc = self.model_runner.vision_neuron_config
        unwrapped = self._unwrap_vision_model()
        if unwrapped is None:
            return

        sorted_buckets = sorted(vnc.num_vision_tokens_buckets)
        logger.info(
            "Extracting vision graphs for %d buckets: %s",
            len(sorted_buckets),
            sorted_buckets,
        )

        capture_backend = self.model_runner.vision_capture_backend
        meta = torch.device("meta")
        for i, bucket in enumerate(sorted_buckets, 1):
            logger.info(
                "  [%s/%s] Extracting vision graph: bucket=%s",
                i,
                len(sorted_buckets),
                bucket,
            )
            vision_inputs = unwrapped.build_vision_synthetic_inputs(bucket, vnc, meta)
            num_blocks = (
                math.ceil(
                    math.ceil(bucket / vnc.vision_attention_block_size) / vnc.dp_size
                )
                * vnc.dp_size
            )
            cache = self.model_runner.encoder_cache
            vision_inputs["encoder_cache_buffer"] = torch.empty(
                cache.num_blocks,
                cache.block_size,
                cache.fat_dim,
                dtype=cache.dtype,
                device=meta,
            )
            vision_inputs["write_block_ids"] = torch.zeros(
                num_blocks, dtype=torch.int64, device=meta
            )
            try:
                capture_backend(**vision_inputs)
            except CaptureComplete:
                pass
            logger.info(
                "  Successfully extracted vision graph for bucket %s",
                bucket,
            )

    def _warmup_prefill(self) -> None:
        """Run prefill warmup for all bucket and KV segment size combinations."""
        num_batched_tokens_buckets, effective_kv_buckets = self._prefill_buckets()

        logger.info(
            "Warming up model for %s num_batched_tokens_buckets: %s",
            len(num_batched_tokens_buckets),
            num_batched_tokens_buckets,
        )

        total_warmups = len(effective_kv_buckets) * len(num_batched_tokens_buckets)
        warmup_count = 0

        for kv_seg_size in effective_kv_buckets:
            for bucket_size in num_batched_tokens_buckets:
                warmup_count += 1
                logger.info(
                    "\n[%s/%s] Warming up for prefill: bucket_size=%s, "
                    "kv_segment_size=%s",
                    warmup_count,
                    total_warmups,
                    bucket_size,
                    kv_seg_size,
                )
                try:
                    self.model_runner.warmup_prefill(bucket_size, kv_seg_size)
                    logger.info(
                        "  Successfully warmed up for prefill bucket %s "
                        "with kv_segment_size %s",
                        bucket_size,
                        kv_seg_size,
                    )
                except Exception as e:
                    logger.error(
                        "  Failed to warmup for prefill bucket %s "
                        "with kv_segment_size %s: %s",
                        bucket_size,
                        kv_seg_size,
                        e,
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Model warmup failed for prefill bucket {bucket_size} "
                        f"with kv_segment_size {kv_seg_size}. Error: {e}"
                    ) from e

    def _decode_compile_targets(self) -> list[tuple[int, int]]:
        """Enumerate (batch_bucket, ctx_bucket) pairs to compile.

        Without ``decode_context_length_buckets``: one pair per batch bucket,
        seq=max_model_len (today's behavior).

        With ``decode_context_length_buckets`` set: cartesian product of batch
        buckets × (configured seq buckets + max_model_len fallback).

        Returns:
            List of (batch_bucket, ctx_bucket) tuples. Empty when
            ``num_seqs_buckets`` is None.
        """
        num_seqs_buckets = self.model_runner.neuron_config.num_seqs_buckets
        if not num_seqs_buckets:
            return []

        decode_ctx_buckets = (
            self.model_runner.neuron_config.decode_context_length_buckets
        )
        max_model_len = self.model_runner.max_model_len

        if decode_ctx_buckets is None:
            return [(b, max_model_len) for b in num_seqs_buckets]

        ctx_buckets_with_fallback = list(decode_ctx_buckets) + [max_model_len]
        return [(b, s) for b in num_seqs_buckets for s in ctx_buckets_with_fallback]

    def _extract_decode_graphs_sequential(self, targets: list[tuple[int, int]]) -> None:
        """In-process decode extraction for every (batch, seq) compile
        target. Used when the parallel-trace pool is disabled.

        Delegates to ``model_runner.extract_decode_graphs`` which
        captures both target and drafter graphs together against
        ``self.device``."""
        for i, (batch_size, ctx_bucket) in enumerate(targets, 1):
            logger.info(
                "\n[%s/%s] Extracting graph for decode (batch=%s, seq=%s)",
                i,
                len(targets),
                batch_size,
                ctx_bucket,
            )
            try:
                self.model_runner.extract_decode_graphs(
                    batch_size, ctx_bucket=ctx_bucket
                )
                logger.info(
                    "  Extracted graph for decode (batch=%s, seq=%s)",
                    batch_size,
                    ctx_bucket,
                )
            except Exception as e:
                logger.error(
                    "  Failed to extract graph for decode (batch=%s, seq=%s): %s",
                    batch_size,
                    ctx_bucket,
                    e,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Model graph extraction failed for decode "
                    f"(batch={batch_size}, seq={ctx_bucket}). Error: {e}"
                ) from e

    def _extract_decode_drafter_graphs(self, targets: list[tuple[int, int]]) -> None:
        """Sequentially extract eagle drafter decode graphs.

        Runs in the parent process after the target-model fork pool, on
        ``self.device``. Caller must ensure a drafter is attached.

        TODO: migrate the drafter to the parallel-trace fork pool. Eagle
        wraps ``propose()`` rather than calling ``capture_backend_model``
        directly, so it doesn't fit the ``(model, kwargs)`` primitive
        without an adapter that exposes ``_orig_mod`` for the meta swap.
        """
        runner = self.model_runner
        if runner.drafter is None or runner.speculative_config is None:
            raise RuntimeError(
                "_extract_decode_drafter_graphs called without a drafter "
                "attached; gate the call on self.model_runner.drafter."
            )
        num_spec_tokens = runner.speculative_config.num_speculative_tokens
        for batch_size, ctx_bucket in targets:
            draft_num_tokens = batch_size * (1 + num_spec_tokens)
            draft_attn_metadata = runner._build_warmup_attention_metadata(
                num_tokens=draft_num_tokens,
                num_reqs=batch_size,
                ctx_bucket=ctx_bucket,
            )
            logger.info("Capturing EAGLE3 decode graphs for batch size: %d", batch_size)
            runner.drafter.graph_extract(
                num_tokens=draft_num_tokens,
                num_reqs=batch_size,
                attn_metadata=draft_attn_metadata,
            )

    def _warmup_decode(self) -> None:
        """Run decode warmup for all (batch, seq) compile targets."""
        targets = self._decode_compile_targets()
        if not targets:
            return
        logger.info(
            "\nWarming up %d decode compile targets: %s",
            len(targets),
            targets,
        )

        for i, (batch_size, ctx_bucket) in enumerate(targets, 1):
            logger.info(
                "\n[%s/%s] Warming up decode (batch=%s, seq=%s)",
                i,
                len(targets),
                batch_size,
                ctx_bucket,
            )
            try:
                self.model_runner.warmup_decode(batch_size, ctx_bucket=ctx_bucket)
                logger.info(
                    "  Warmed up decode (batch=%s, seq=%s)",
                    batch_size,
                    ctx_bucket,
                )
            except Exception as e:
                logger.error(
                    "  Failed to warmup decode (batch=%s, seq=%s): %s",
                    batch_size,
                    ctx_bucket,
                    e,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Model warmup failed for decode "
                    f"(batch={batch_size}, seq={ctx_bucket}). Error: {e}"
                ) from e

    def _warmup_vision_encoder(self) -> None:
        """Warm up vision encoder by loading NEFFs for each bucket.

        Drives the compiled vision encoder directly with synthetic per-bucket inputs.
        """
        vnc = self.model_runner.vision_neuron_config
        unwrapped = self._unwrap_vision_model()
        has_vision_warmup = unwrapped is not None

        if vnc.num_vision_tokens_buckets is None:
            if has_vision_warmup:
                raise ValueError(
                    "Model supports vision but num_vision_tokens_buckets is not "
                    "configured. Vision encoder requires bucket configuration "
                    "for Neuron compilation."
                )
            logger.info("No vision token buckets configured, skipping vision warmup")
            return

        if not has_vision_warmup:
            logger.info("Model does not support vision warmup, skipping")
            return

        sorted_buckets = sorted(vnc.num_vision_tokens_buckets)
        logger.info(
            "Warming up vision encoder for %d buckets: %s",
            len(sorted_buckets),
            sorted_buckets,
        )

        device = next(unwrapped.visual.parameters()).device
        for i, bucket in enumerate(sorted_buckets):
            num_blocks = math.ceil(bucket / vnc.vision_attention_block_size)
            # Pad num_blocks to be divisible by dp_size for even DP scatter
            if vnc.dp_size > 1:
                num_blocks = math.ceil(num_blocks / vnc.dp_size) * vnc.dp_size
            logger.info(
                "  Vision warmup [%d/%d]: bucket=%d, num_blocks=%d",
                i + 1,
                len(sorted_buckets),
                bucket,
                num_blocks,
            )

            vision_inputs = unwrapped.build_vision_synthetic_inputs(bucket, vnc, device)
            try:
                vision_inputs["encoder_cache_buffer"] = (
                    self.model_runner.encoder_cache.buffer
                )
                vision_inputs["write_block_ids"] = torch.zeros(
                    num_blocks, dtype=torch.int64, device=device
                )
                _ = unwrapped.visual(**vision_inputs)
                logger.info("  Successfully warmed up vision bucket %d", bucket)
            except Exception as e:
                logger.error(
                    "  Failed to warmup vision bucket %d: %s",
                    bucket,
                    e,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Vision encoder warmup failed for bucket {bucket}. Error: {e}"
                ) from e

        logger.info("Vision encoder warmup complete")

    def execute_model(
        self,
        scheduler_output: Any,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """
        Execute the model with the given scheduler output.

        If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        NOTE (vllm) Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.

        Args:
            scheduler_output: Output from the vLLM scheduler

        Returns:
            - ModelRunnerOutput
            - or AsyncModelRunnerOutput
            - or None
        """
        if self._profiler is not None:
            self._profiler.step()

        logger.debug("Executing model on NeuronWorker")
        output = self.model_runner.execute_model(scheduler_output)

        return output

    def _create_profiler(self) -> WorkerProfiler:
        """Create the appropriate profiler based on config.

        Called lazily on first /start_profile request.

        Raises:
            ValueError: If profiler type is not recognized.
        """
        profiler_config = self.vllm_config.profiler_config

        if profiler_config.profiler == "cuda":
            neuron_profiler_config = NeuronProfilerConfig(
                self.vllm_config.additional_config.get("neuron_profiler")
            )
            return NeuronProfiler(profiler_config, neuron_profiler_config)
        elif profiler_config.profiler == "torch":
            worker_name = f"neuron-rank-{self.rank}"
            return TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU"],
            )
        else:
            raise ValueError(
                f"Unknown profiler type: '{profiler_config.profiler}'. "
                "Supported types: 'cuda', 'torch'."
            )

    def _should_profile(self) -> bool:
        """Return True if this worker should participate in profiling."""
        profiler_config = self.vllm_config.profiler_config
        if profiler_config is None or profiler_config.profiler is None:
            return False

        if profiler_config.profiler == "cuda":
            neuron_profiler_config = NeuronProfilerConfig(
                self.vllm_config.additional_config.get("neuron_profiler")
            )
            neuron_cores = neuron_profiler_config.neuron_cores
            if neuron_cores is not None:
                return self.local_rank in neuron_cores
            return self.local_rank == 0

        return True

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        """Toggle profiling on/off via /start_profile and /stop_profile.

        Supports two profiler types:
        - "cuda": NeuronProfiler for device/system profiling (NRT inspect APIs)
        - "torch": TorchProfilerWrapper for CPU-side torch profiler spans

        The profiler is created lazily on the first start call.
        """
        if not self._should_profile():
            return

        if is_start:
            if self._profiler is None:
                self._profiler = self._create_profiler()
            self._profiler.start()
        else:
            if self._profiler is None:
                logger.warning("Profiler was not started, nothing to stop.")
                return
            self._profiler.stop()

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Get KV cache specifications for all model layers.

        This method is required by vLLM v1 WorkerBase interface.

        Returns:
            Dictionary mapping layer names to their KV cache specifications
        """
        logger.debug("Getting KV cache spec from NeuronWorker")
        return self.model_runner.get_kv_cache_spec()

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """
        Get draft token ids from the model runner.

        This method is required by vLLM v1 WorkerBase interface.
        Delegates to the model runner to get draft token ids.

        Returns:
            DraftTokenIds or None if not available
        """
        logger.debug("Getting draft token ids from NeuronWorker")
        return self.model_runner.take_draft_token_ids()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """
        Get the tasks supported by this worker.

        This method is required by vLLM v1 WorkerBase interface.
        Delegates to the model runner to determine supported tasks.

        Returns:
            Tuple of supported tasks
        """
        logger.debug("Getting supported tasks from NeuronWorker")
        return self.model_runner.get_supported_tasks()

    def execute_dummy_batch(self) -> None:
        """
        Execute a dummy batch for Data Parallel synchronization.

        This method is called by the vLLM engine when Data Parallel workload
        imbalance is detected (i.e., this rank has no real requests but other
        DP ranks do). It runs a minimal dummy forward pass to keep all ranks
        synchronized during collective operations.

        The implementation simply delegates to the model runner's
        execute_dummy_batch method, which handles the actual dummy execution.
        """
        logger.debug("Executing dummy batch on NeuronWorker")
        self.model_runner.execute_dummy_batch()

    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        if self._profiler is not None:
            self._profiler.shutdown()

        # Disable tensor capture to prevent writes during teardown
        if self.model_runner is not None:
            self.model_runner.disable_capture()

        self.model_runner.ensure_kv_transfer_shutdown()

        # This triggers an unexpected state error. Lets try avoiding unsafe_close()
        #
        # if not envs.VLLM_NEURON_CPU_MODE:
        #     # force the neuron profiles to dump the profiles
        #     torch.classes.neuron.Runtime().unsafe_close()

    def get_kv_caches(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Extract KV cache tensors from this worker's model (CPU tensors for serialization).

        Example:
            >>> kv = worker.get_kv_caches()
        """
        return self.model_runner.get_kv_caches()

    def get_block_table_info(self) -> dict:
        """Extract block table and sequence info for KV cache reconstruction.

        Example:
            >>> info = worker.get_block_table_info()
        """
        return self.model_runner.get_block_table_info()

    def get_kv_cache_config(self) -> dict:
        """Return KVCacheConfig for model-agnostic KV reconstruction.

        Example:
            >>> cfg = worker.get_kv_cache_config()
        """
        return self.model_runner.get_kv_cache_config()

    def get_block_tables(self) -> list[bytes]:
        """Return block tables for all KV cache groups.

        Example:
            >>> tables = worker.get_block_tables()
        """
        return self.model_runner.get_block_tables()

    def clear_kv_snapshot(self) -> None:
        """Release block table snapshot memory and disable snapshotting.

        Example:
            >>> worker.clear_kv_snapshot()
        """
        self.model_runner.clear_kv_snapshot()

    def get_encoder_cache(self) -> dict[str, bytes]:
        """Return encoder cache (vision embeddings) as serialized bytes.

        Example:
            >>> enc = worker.get_encoder_cache()
        """
        return self.model_runner.get_encoder_cache()

    def enable_encoder_cache_snapshot(self) -> None:
        """Enable encoder cache snapshotting (lightweight, no serialization).

        Example:
            >>> worker.enable_encoder_cache_snapshot()
        """
        self.model_runner.enable_encoder_cache_snapshot()

    def clear_encoder_cache_snapshot(self) -> None:
        """Release encoder cache snapshot memory and disable snapshotting.

        Example:
            >>> worker.clear_encoder_cache_snapshot()
        """
        self.model_runner.clear_encoder_cache_snapshot()

    def get_async_scheduling_stats(self) -> dict:
        """Return async scheduling step counters from the model runner.

        Returns a dict with ``async_steps`` and ``sync_fallback_steps``.
        Returns zeros when async scheduling is not enabled.

        Example:
            >>> stats = worker.get_async_scheduling_stats()
            >>> stats["async_steps"]
            42
        """
        if not getattr(self.model_runner, "use_async_scheduling", False):
            return {"async_steps": 0, "sync_fallback_steps": 0}
        return {
            "async_steps": self.model_runner._async_steps,
            "sync_fallback_steps": self.model_runner._sync_fallback_steps,
        }

    def _use_neuron_device(self) -> bool:
        """
        VLLM_NEURON_CPU_MODE forces cpu as the device with vLLM.
        VLLM_NEURON_CPU_COMPILE forces meta as the device type with vLLM.

        This check returns if neuron is the device type being used with vLLM.
        """
        return not envs.VLLM_NEURON_CPU_MODE and not envs.VLLM_NEURON_CPU_COMPILE
