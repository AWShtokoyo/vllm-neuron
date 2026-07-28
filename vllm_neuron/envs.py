# SPDX-License-Identifier: Apache-2.0
"""
vLLM Neuron Environment Variables Configuration

This module provides centralized environment variable management for vLLM Neuron

All environment variables are:
- Lazily evaluated when accessed
- Type-safe with proper validation
- Prefixed with VLLM_NEURON_ for namespace isolation
"""

import functools
import logging
import os
import subprocess
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # Core System Variables
    VLLM_NEURON_CPU_MODE: bool = False
    VLLM_NEURON_CPU_COMPILE: bool = False
    VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND: bool = False
    VLLM_NEURON_LOG_LEVEL: str = "INFO"
    VLLM_NEURON_DEBUG_MODE: bool = False
    VLLM_NEURON_DISABLE_COMPILE_CACHE: bool = False
    VLLM_NEURON_COMPILATION_TIMEOUT: int = 600
    VLLM_NEURON_BARRIER_TIMEOUT: int = 3600
    # TODO: The parallel compile worker needs to be determined automatically
    # based on underlying CPU and model being compiled.
    VLLM_NEURON_PARALLEL_COMPILE_WORKERS: int = 8
    # TODO: Determine the optimal forks-per-worker automatically based on
    # underlying CPU, number of ranks, and buckets being compiled.
    VLLM_NEURON_PARALLEL_TRACE_WORKERS: int = 8
    VLLM_NEURON_DISABLE_PARALLEL_TRACE: bool = False
    # TODO: Remove VLLM_NEURON_SWITCH_CC and derive topology from instance type.
    VLLM_NEURON_SWITCH_CC: bool = False
    VLLM_NEURON_MIN_KV_BUDGET_GIB: float = 1.0
    VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION: float = 0.30
    VLLM_NEURON_REMOTE_CACHE: Optional[str] = None
    VLLM_NEURON_WORKER_TERMINATION_TIMEOUT: int = 5
    VLLM_NEURON_DISABLE_WARMUP_COMPILE: bool = False
    VLLM_NEURON_ASSERT_CACHE_HIT: bool = False
    VLLM_NEURON_DISABLE_NKI_KERNELS: bool = False
    VLLM_NEURON_KERNEL_DEVICE_DUMP: bool = False
    VLLM_NEURON_SKIP_PREFILL_WARMUP: bool = False
    VLLM_NEURON_SKIP_DECODE_WARMUP: bool = False
    VLLM_NEURON_LIBTORCH_NEURONX_LITE: bool = True
    # When True, the compile backend skips FX-graph input deduplication.
    # Multiple FX placeholders that alias the same runtime storage become
    # multiple HLO parameters; the runtime still allocates one byte buffer
    # per bucket (KVCacheTensor.shared_by) so HBM usage should not change.
    VLLM_NEURON_DISABLE_INPUT_DEDUP: bool = False
    VLLM_NEURON_EFA_INSTANCE_FAMILY: str = ""


def maybe_convert_bool(value: str | None) -> bool | None:
    """
    Safely convert string to boolean using numeric conversion.

    Args:
        value: String value to convert ("0", "1") or None

    Returns:
        True if value is "1", False if value is "0", None if value is None

    Raises:
        ValueError: If value cannot be converted to int

    Examples:
        >>> maybe_convert_bool("1")
        True
        >>> maybe_convert_bool("0")
        False
        >>> maybe_convert_bool(None)
        None
        >>> maybe_convert_bool("invalid")  # Raises ValueError
    """
    if value is None:
        return None
    return bool(int(value))


def maybe_convert_int(value: str | None) -> int | None:
    """
    Safely convert string to integer.

    Args:
        value: String value to convert or None

    Returns:
        Integer value if conversion successful, None if value is None

    Raises:
        ValueError: If value cannot be converted to int

    Examples:
        >>> maybe_convert_int("600")
        600
        >>> maybe_convert_int("0")
        0
        >>> maybe_convert_int(None)
        None
        >>> maybe_convert_int("invalid")  # Raises ValueError
    """
    if value is None:
        return None
    return int(value)


def maybe_convert_float(value: str | None) -> float | None:
    """Safely convert string to float."""
    if value is None:
        return None
    return float(value)


environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Core System Variables ==================
    # Enable CPU fallback mode instead of using Neuron accelerators
    "VLLM_NEURON_CPU_MODE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_CPU_MODE")) or False
    ),
    # Enable CPU graph capture and compilation
    "VLLM_NEURON_CPU_COMPILE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_CPU_COMPILE")) or False
    ),
    # Disable graph capture backend and parallel compilation.
    # This env is used when remote cache already has the compiled artifacts.
    # So we avoid graph capture and parallel compile all together.
    "VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND"))
        or False
    ),
    # Logging level for vLLM Neuron components
    "VLLM_NEURON_LOG_LEVEL": lambda: os.getenv("VLLM_NEURON_LOG_LEVEL", "INFO").upper(),
    # Enable debug mode for verbose output and additional diagnostics
    "VLLM_NEURON_DEBUG_MODE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DEBUG_MODE")) or False
    ),
    # Disable compilation cache
    "VLLM_NEURON_DISABLE_COMPILE_CACHE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DISABLE_COMPILE_CACHE")) or False
    ),
    # Timeout in seconds for compilation
    "VLLM_NEURON_COMPILATION_TIMEOUT": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_COMPILATION_TIMEOUT")) or 600
    ),
    # tp_barrier timeout in seconds (raise for long serial warmups)
    "VLLM_NEURON_BARRIER_TIMEOUT": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_BARRIER_TIMEOUT")) or 3600
    ),
    # Number of compilation workers
    "VLLM_NEURON_PARALLEL_COMPILE_WORKERS": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_PARALLEL_COMPILE_WORKERS")) or 8
    ),
    # Number of fork children each worker spawns to parallelize per-bucket
    # graph trace. Each fork runs an independent Dynamo + torch_xla state to
    # sidestep the process-global trace bottleneck. Set to 1 to fall back to
    # a single forked child running every bucket sequentially.
    # TODO: Determine the optimal value automatically based on CPU/RAM
    # headroom, bucket count, and per-bucket trace cost.
    "VLLM_NEURON_PARALLEL_TRACE_WORKERS": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_PARALLEL_TRACE_WORKERS")) or 8
    ),
    # When True, disable the parallel-trace fork pool entirely and run
    # graph extraction sequentially in the parent process.
    "VLLM_NEURON_DISABLE_PARALLEL_TRACE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DISABLE_PARALLEL_TRACE")) or False
    ),
    # Minimum KV budget (GiB) guardrail
    "VLLM_NEURON_MIN_KV_BUDGET_GIB": lambda: (
        maybe_convert_float(os.getenv("VLLM_NEURON_MIN_KV_BUDGET_GIB"))
        if os.getenv("VLLM_NEURON_MIN_KV_BUDGET_GIB") is not None
        else 1.0
    ),
    # KV cap fraction applied to GMU-scaled total HBM budget.
    # TODO: Interim global safety cap. Replace with a better compile-safe
    # estimator that adapts across model families and hardware generations.
    "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION": lambda: (
        maybe_convert_float(os.getenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION"))
        if os.getenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION") is not None
        else 0.30
    ),
    # Local cache directory for model checkpoints
    "VLLM_NEURON_CHECKPOINT_CACHE": lambda: os.getenv(
        "NXDI_CHECKPOINT_CACHE", "/tmp/vllm_neuron-checkpoints"
    ),
    # Golden cache directory (disk tier)
    "VLLM_NEURON_GOLDEN_CACHE_DIR": lambda: os.path.expandvars(
        os.getenv("VLLM_NEURON_GOLDEN_CACHE_DIR", "/tmp/vllm_neuron-goldens-$USER")
    ),
    # Golden cache S3 URI (secondary tier, empty = disabled)
    "VLLM_NEURON_S3_GOLDENS_URI": lambda: os.getenv("VLLM_NEURON_S3_GOLDENS_URI", ""),
    # TODO: Remove VLLM_NEURON_SWITCH_CC and derive topology from instance type.
    # When True, uses contiguous groups for 8x8 topology instead of custom TRN2 mesh.
    "VLLM_NEURON_SWITCH_CC": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SWITCH_CC")) or False
    ),
    # Optional NFS/FSx path for shared persistent compile cache.
    # When set, compile fetches from this path on a local cache miss before
    # falling back to compilation. When not set, behaviour is identical to having no remote cache.
    "VLLM_NEURON_REMOTE_CACHE": lambda: os.getenv("VLLM_NEURON_REMOTE_CACHE", None),
    # Timeout in seconds for worker termination (SIGTERM→SIGKILL).
    # Default is 5s (upstream vLLM uses 4s for workers, 5s for API servers).
    # Increase when system profiling is enabled (NEURON_RT_INSPECT_ENABLE=1),
    # as the Neuron runtime needs time to flush profiling data after SIGTERM.
    "VLLM_NEURON_WORKER_TERMINATION_TIMEOUT": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_WORKER_TERMINATION_TIMEOUT")) or 5
    ),
    # Disable warmup compilation: treat local compile cache miss as a fatal error.
    # Incompatible with VLLM_NEURON_DISABLE_COMPILE_CACHE.
    "VLLM_NEURON_DISABLE_WARMUP_COMPILE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DISABLE_WARMUP_COMPILE")) or False
    ),
    # Expect cache hits for artifacts: treat local + remote compile cache miss as a fatal error.
    # Incompatible with VLLM_NEURON_DISABLE_COMPILE_CACHE.
    "VLLM_NEURON_ASSERT_CACHE_HIT": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_ASSERT_CACHE_HIT")) or False
    ),
    # Disable NKI kernels — forces can_run_kernel() to return False
    "VLLM_NEURON_DISABLE_NKI_KERNELS": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DISABLE_NKI_KERNELS")) or False
    ),
    # Enable NKI kernel device dumps: compile NKI kernels with the NKI Tracer
    # frontend + enable_device_dump=True, dumping all intermediate SBUF tensors.
    # Capture at runtime via NEURON_RT_DEBUG_OUTPUT_DIR.
    "VLLM_NEURON_KERNEL_DEVICE_DUMP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_KERNEL_DEVICE_DUMP")) or False
    ),
    # Skip prefill warmup/compilation without requiring kv-transfer-config.
    # Useful for decode-only profiling workflows.
    "VLLM_NEURON_SKIP_PREFILL_WARMUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SKIP_PREFILL_WARMUP")) or False
    ),
    # Skip decode warmup/compilation without requiring kv-transfer-config.
    # Useful for prefill-only profiling workflows.
    "VLLM_NEURON_SKIP_DECODE_WARMUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SKIP_DECODE_WARMUP")) or False
    ),
    "VLLM_NEURON_LIBTORCH_NEURONX_LITE": lambda: (
        # Honor an explicit env value; default True only when unset. Using
        # `... or True` forced True even when the env was set to "0"
        # (False or True == True), so lite mode could never be disabled.
        _v
        if (_v := maybe_convert_bool(os.getenv("VLLM_NEURON_LIBTORCH_NEURONX_LITE")))
        is not None
        else True
    ),
    # Explicit instance-family override. When set, get_instance_family()
    # returns this verbatim instead of resolving from the sysfs product_name
    # (which defaults any trn3* to trn3pds and any trn2* to trn2).
    "VLLM_NEURON_EFA_INSTANCE_FAMILY": lambda: os.getenv(
        "VLLM_NEURON_EFA_INSTANCE_FAMILY", ""
    ),
    # Skip FX-graph input deduplication (compile/backend.py).
    "VLLM_NEURON_DISABLE_INPUT_DEDUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DISABLE_INPUT_DEDUP")) or False
    ),
}


def __getattr__(name: str) -> Any:
    """
    Gets environment variables lazily.

    Args:
        name: Name of the environment variable to retrieve

    Returns:
        Value of the environment variable (type depends on variable)

    Raises:
        AttributeError: If environment variable name is not defined

    Examples:
        >>> import vllm_neuron.envs as envs
        >>> cpu_mode = envs.VLLM_NEURON_CPU_MODE  # Returns bool
        >>> log_level = envs.VLLM_NEURON_LOG_LEVEL  # Returns str
        >>> envs.VLLM_NEURON_NONEXISTENT  # Raises AttributeError
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    Return list of available environment variables.

    Returns:
        List of all defined environment variable names

    Examples:
        >>> import vllm_neuron.envs as envs
        >>> variables = dir(envs)
        >>> 'VLLM_NEURON_CPU_MODE' in variables
        True
    """
    return list(environment_variables.keys())


def is_set(name: str) -> bool:
    """
    Check if an environment variable is explicitly set.

    Args:
        name: Name of the environment variable to check

    Returns:
        True if environment variable is explicitly set, False otherwise

    Raises:
        AttributeError: If environment variable name is not defined

    Examples:
        >>> import os
        >>> import vllm_neuron.envs as envs
        >>> os.environ['VLLM_NEURON_CPU_MODE'] = '1'
        >>> envs.is_set('VLLM_NEURON_CPU_MODE')
        True
        >>> envs.is_set('VLLM_NEURON_LOG_LEVEL')  # Not set, uses default
        False
        >>> envs.is_set('VLLM_NEURON_NONEXISTENT')  # Raises AttributeError
    """
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_native_backend() -> bool:
    """Return True when VLLM_NEURON_BACKEND=neuron_native."""
    return os.getenv("VLLM_NEURON_BACKEND", "").lower() == "neuron_native"


def get_compile_backend_name() -> str:
    """Return the torch.compile backend name: 'neuron' for native, 'vllm_neuron' for XLA."""
    return "neuron" if is_native_backend() else "vllm_neuron"


def get_dist_backend() -> str:
    """Return the distributed backend: 'neuron' for native, 'gloo' for XLA."""
    return "neuron" if is_native_backend() else "gloo"


def get_neuron_compile_cache_dir() -> str:
    """Return the local compile cache directory derived from VLLM_CACHE_ROOT.

    If ``VLLM_CACHE_ROOT`` is explicitly set by the user, returns
    ``$VLLM_CACHE_ROOT/neuron/compile_cache`` unconditionally — the user's
    explicit choice is always respected.

    If ``VLLM_CACHE_ROOT`` is not set (default ``~/.cache/vllm``), probes
    the home directory filesystem type.  On multi-node clusters,the home directory
    can be NFS-mounted, which is incompatible with FileLock semantics.  In that case
    the function falls back to ``/tmp/vllm_neuron_wdir_$USER/neuron/compile_cache`` and logs a warning.

    Returns:
        str: Absolute path to the local Neuron compile cache directory.

    Examples:
        >>> import os
        >>> os.environ["VLLM_CACHE_ROOT"] = "/tmp/my_cache"
        >>> get_neuron_compile_cache_dir()
        '/tmp/my_cache/neuron/compile_cache'
    """
    # Cache by VLLM_CACHE_ROOT so the FS probe + NFS-fallback warning only
    # runs once per cache root per process.
    return _resolve_neuron_compile_cache_dir(os.environ.get("VLLM_CACHE_ROOT"))


@functools.lru_cache
def _resolve_neuron_compile_cache_dir(cache_root_override: Optional[str]) -> str:
    """Cached resolver for ``get_neuron_compile_cache_dir``.

    Keyed on the caller-supplied ``VLLM_CACHE_ROOT`` value so different
    overrides each get their own cache slot.
    """
    if cache_root_override is not None:
        # User made an explicit choice — respect it unconditionally.
        return os.path.join(cache_root_override, "neuron/compile_cache")

    # Probe the home directory to detect remote filesystem mounts (NFS or Lustre).
    # We probe ~ (guaranteed to exist) rather than ~/.cache/vllm (may not exist yet).
    if _path_is_remote_filesystem(os.path.expanduser("~")):
        fallback = os.path.join(
            os.path.expandvars("/tmp/vllm_neuron_wdir_$USER"), "neuron/compile_cache"
        )
        logger.warning(
            "Default compile cache path ~/.cache/vllm/neuron/compile_cache is on a remote filesystem (NFS or Lustre). "
            "Falling back to %s.",
            fallback,
        )
        return fallback

    return os.path.join(os.path.expanduser("~/.cache/vllm"), "neuron/compile_cache")


def _path_is_remote_filesystem(path: str) -> bool:
    """Return True if *path* is on a remote filesystem (NFS or Lustre).

    Args:
        path: An existing filesystem path to check.

    Returns:
        bool: True if the filesystem is NFS (any variant) or Lustre
        (including FSx for Lustre), False otherwise.
    """
    result = subprocess.check_output(
        ["stat", "-f", "-c", "%T", path],
        stderr=subprocess.STDOUT,
        text=True,
    )
    fs_type = result.strip().lower()
    return fs_type.startswith("nfs") or fs_type == "lustre"
