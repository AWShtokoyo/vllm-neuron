# SPDX-License-Identifier: Apache-2.0
import argparse
import contextlib
import logging
import os
import shlex
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

import torch
import torch.distributed as dist
import torch.fx

from vllm_neuron import envs
from vllm_neuron.compile.cache import get_neff_filename
from vllm_neuron.utils.timer import timer


logger = logging.getLogger(__name__)


def model_forward_context(vllm_config: "VllmConfig"):
    """Context manager for model forward: skips fail_on_recompile in CPU mode."""
    if envs.VLLM_NEURON_CPU_MODE and vllm_config.model_config.enforce_eager:
        return contextlib.nullcontext()
    return torch.compiler.set_stance("fail_on_recompile")


# ---------------------------------------------------------------------------
# Input deduplication (mirrors torch._functorch._aot_autograd.AOTDedupeWrapper)
#
# The vllm_neuron backend bypasses aot_autograd, so we don't get PyTorch's built-in
# duplicate-input handling.  When the same underlying tensor is passed as
# multiple FX graph inputs (e.g. shared KV cache views in hybrid-attention
# models), Dynamo creates separate placeholders.  This causes the Neuron
# compiler to double-count memory.
#
# The functions below detect duplicates, rewrite the FX graph to remove them,
# and wrap the compiled executable so that Dynamo's full arg list is filtered
# down to the deduped set at runtime — exactly as AOTDedupeWrapper does for
# inductor.
# ---------------------------------------------------------------------------


def _detect_duplicate_inputs(
    example_inputs,
) -> tuple[List[bool], List[int]]:
    """Detect duplicate inputs by storage identity.

    Two inputs are considered duplicates when they share the same data pointer,
    storage offset, shape, stride, **and** dtype — i.e. they are
    indistinguishable at the tensor level.

    Returns:
        keep_mask:  ``keep_mask[i]`` is True for the first occurrence of each
                    unique input and False for subsequent duplicates.
        dupe_map:   ``dupe_map[i]`` gives the index *in the deduped list* that
                    original index ``i`` maps to.
    """
    seen: Dict[Any, int] = {}  # key -> deduped index
    keep_mask: List[bool] = []
    dupe_map: List[int] = []
    j = 0  # running deduped index

    for inp in example_inputs:
        if isinstance(inp, torch.Tensor):
            # Use storage object identity + offset to detect views of the
            # same buffer (e.g. shared KV cache).  We use id(storage)
            # rather than data_ptr() because Neuron/XLA tensors return
            # data_ptr()==0, but their storage objects are distinct per
            # allocation.
            key = (
                id(inp.untyped_storage()),
                inp.storage_offset(),
                tuple(inp.shape),
                tuple(inp.stride()),
                inp.dtype,
            )
        else:
            key = id(inp)

        if key in seen:
            keep_mask.append(False)
            dupe_map.append(seen[key])
        else:
            seen[key] = j
            keep_mask.append(True)
            dupe_map.append(j)
            j += 1

    return keep_mask, dupe_map


def _dedup_fx_graph(
    gm: torch.fx.GraphModule,
    keep_mask: List[bool],
    dupe_map: List[int],
) -> torch.fx.GraphModule:
    """Remove duplicate placeholder nodes from *gm*, redirecting their uses.

    For every placeholder whose ``keep_mask`` entry is False, all downstream
    uses are rewritten to reference the corresponding *kept* placeholder
    (identified via ``dupe_map``), and the duplicate node is erased.
    """
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    kept_placeholders: List[torch.fx.Node] = []

    for i, node in enumerate(placeholders):
        if keep_mask[i]:
            kept_placeholders.append(node)
        else:
            primary = kept_placeholders[dupe_map[i]]
            node.replace_all_uses_with(primary)
            gm.graph.erase_node(node)

    gm.recompile()

    duped_count = sum(1 for k in keep_mask if not k)
    logger.info(
        f"Input dedup: removed {duped_count} duplicate placeholder(s) "
        f"({len(placeholders)} -> {len(kept_placeholders)})"
    )
    return gm


def compile(gm: torch.fx.GraphModule, example_inputs, options: dict = {}) -> callable:
    """Compile a PyTorch FX GraphModule for AWS Neuron execution.

    This function compiles a PyTorch FX GraphModule to run on AWS Neuron devices
    (Trainium/Inferentia). Uses a unified compilation approach that automatically
    handles both shared cache and non-cached compilation modes.

    Dynamo calls this function with the traced fx graph. It expects this function to return an executable
    which it caches.

    Compilation Flow:
        Step 1. fx.graph -> HLO
                (current) torch_xla does this translation
                (future) fx.graph to StableHLO is being built to replace torch_xla
        Step 2. neuronx_cc compiled HLO -> NEFF
        Step 3: Build an Executable - This is an invocation to custom op that invokes NRT runtime.

    Args:
        gm (torch.fx.GraphModule): The PyTorch FX GraphModule to compile.
        example_inputs: Example inputs used for tracing the model.
        options (dict, optional): Compilation options. Supported options:
            - debug_hlo (bool): If True, dumps HLO debug information. Defaults to True.
            - compiler_workdir (str): Base directory for compiler work files. Defaults to "/tmp/vllm_neuron_wdir/".
            - compilation_timeout (int): Timeout in seconds for shared compilation coordination. Defaults to 600 seconds (10 minutes).

    Returns:
        callable: Either the compiled Neuron model or the original forward method.

    Examples:
        >>> import torch
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend, rank=rank, world_size=world_size)  # doctest: +SKIP
        >>>
        >>> rank = dist.get_rank()  # doctest: +SKIP
        >>> world_size = dist.get_world_size()  # doctest: +SKIP
        >>>
        >>> # Load and prepare model
        >>> model = Model()  # doctest: +SKIP
        >>> checkpoint = torch.load(checkpoint_path)  # doctest: +SKIP
        >>> model.load_state_dict(checkpoint)  # doctest: +SKIP
        >>>
        >>> # Move to Neuron device before compilation (required!)
        >>> model.to("neuron:0")  # doctest: +SKIP
        >>>
        >>> # Compile with vLLM Neuron backend
        >>> model = torch.compile(model, backend="vllm_neuron")  # doctest: +SKIP
        >>>
        >>> # Configure compilation timeout via environment variable
        >>> import os
        >>> os.environ["VLLM_NEURON_COMPILATION_TIMEOUT"] = "1200"  # 20 minutes  # doctest: +SKIP
        >>> model = torch.compile(model, backend="vllm_neuron")  # doctest: +SKIP
        >>>
        >>> # Disable shared cache via environment variable (enabled by default)
        >>> os.environ["VLLM_NEURON_DISABLE_COMPILE_CACHE"] = "1"  # doctest: +SKIP
        >>> model = torch.compile(model, backend="vllm_neuron")  # doctest: +SKIP
    """
    # Return original GraphModule if CPU mode is enabled
    if envs.VLLM_NEURON_CPU_MODE:
        return gm.forward

    # Lazy import to avoid torch_neuronx/torch_xla in CPU mode
    from vllm_neuron.compile import cache

    gm, example_inputs = preprocess_and_validate_inputs(gm, example_inputs)

    # Get device info
    g_device_id = dist.get_rank() if dist.is_initialized() else 0
    g_device_count = dist.get_world_size() if dist.is_initialized() else 1

    # Cache configuration
    use_cache = not envs.VLLM_NEURON_DISABLE_COMPILE_CACHE
    assert_cache_hits = envs.VLLM_NEURON_ASSERT_CACHE_HIT
    disable_warmup_compile = envs.VLLM_NEURON_DISABLE_WARMUP_COMPILE
    remote_cache_dir = envs.VLLM_NEURON_REMOTE_CACHE

    if (disable_warmup_compile or assert_cache_hits) and not use_cache:
        raise ValueError(
            "VLLM_NEURON_DISABLE_COMPILE_CACHE cannot both be set with VLLM_NEURON_DISABLE_WARMUP_COMPILE or  VLLM_NEURON_ASSERT_CACHE_HIT"
        )

    compile_base_dir = options.get(
        "compiler_workdir", envs.get_neuron_compile_cache_dir()
    )

    # Normalize compiler_args before hashing so the cache key reflects what
    # neuroncc_compile will actually run (e.g. trn2 fp8 flag injection).
    # Without this, equivalent configs produce different hashes and compile twice.
    options = _apply_platform_compiler_args(options)

    hash_key = cache.create_cache_hash(gm, example_inputs, options)
    logger.info(f"Compilation cache key: {hash_key}")

    # Compilation settings
    compilation_timeout = options.get(
        "compilation_timeout", envs.VLLM_NEURON_COMPILATION_TIMEOUT
    )
    artifacts = None

    def compilation_callback():
        return execute_compile(gm, example_inputs, options, hash_key)

    if use_cache:
        # Fast local check
        artifacts = cache.get_local(hash_key, compile_base_dir)
        if artifacts:
            logger.info(f"Local cache hit for key: {hash_key}")
    else:
        logger.info(f"Cache disabled - forcing recompilation for key: {hash_key}")

    if disable_warmup_compile and not artifacts:
        raise RuntimeError(
            f"VLLM_NEURON_DISABLE_WARMUP_COMPILE: local compile cache miss for key "
            f"{hash_key}. All graphs must be pre-compiled when warmup compilation is disabled."
        )
    elif assert_cache_hits and not artifacts and not remote_cache_dir:
        raise RuntimeError(
            f"VLLM_NEURON_ASSERT_CACHE_HIT: local compile cache miss for key "
            f"{hash_key} and remote cache is disabled. All graphs must be a cache hit."
        )

    # Cache miss / cache disabled — try remote fetch then compile
    if not artifacts:
        artifacts = cache.fetch_remote_or_compile(
            hash_key,
            compile_base_dir,
            compilation_callback,
            compilation_timeout,
            invalidate=not use_cache,
            remote_cache_dir=remote_cache_dir,
            assert_cache_hits=assert_cache_hits,
        )

    executable = build_executable(
        hlo_filename=artifacts.hlo_filename,
        neff_filename=artifacts.neff_filename,
        g_device_id=g_device_id,
        g_device_count=g_device_count,
        artifacts=artifacts,
    )

    return executable


def execute_compile(
    gm: torch.fx.GraphModule, example_inputs, options: dict, hash_key: str
):
    """Execute the complete FX -> HLO -> NEFF compilation pipeline.

    Args:
        gm: The PyTorch FX GraphModule to compile
        example_inputs: Example inputs for the model
        options: Compilation options

    Returns:
        CompilationArtifacts: Compilation artifacts including HLO, NEFF, and I/O mapping
    """
    # Lazy imports to avoid torch_neuronx/torch_xla in CPU mode
    from vllm_neuron.compile.artifacts import CompilationArtifacts
    from vllm_neuron.compile.hlo import log_hlo_debug
    from vllm_neuron.compile.schema import create_metadata
    from vllm_neuron.compile.capture_backend import run_fx_to_hlo_pipeline

    workdir = _setup_workdir(gm, example_inputs, options)

    if _is_remote_filesystem(workdir):
        raise RuntimeError(
            f"compiler_workdir ({workdir}) is on a remote filesystem (NFS or Lustre). "
            "Operations that might be atomic on a local filesystem are not guaranteed to be atomic over a remote filesystem "
            "and this could lead to unexpected behavior."
        )

    (
        hlo_module,
        unused_input_indices,
        has_rng_seed_parameter,
        io_map,
        output_count,
        fx_to_hlo_time,
    ) = run_fx_to_hlo_pipeline(gm, example_inputs, options, workdir)

    compilation_times = {"fx_to_hlo": fx_to_hlo_time}

    log_hlo_debug(hlo_module, workdir, options)

    compiler_args = _parse_compiler_args(options.get("compiler_args", ""))

    # Compile to NEFF
    with timer() as compile_timer:
        neff_filename = neuroncc_compile(
            hlo_module.SerializeToString(),
            workdir,
            compiler_args=compiler_args,
        )

    compilation_times["compile"] = compile_timer()

    _log_compilation_times(compilation_times)

    metadata = create_metadata(
        cache_key=hash_key,
        output_count=output_count,
        unused_input_indices=unused_input_indices,
        has_rng_seed_parameter=has_rng_seed_parameter,
        io_map=io_map,
    )

    hlo_filename = os.path.join(workdir, "graph.hlo")
    return CompilationArtifacts(
        hlo_filename=hlo_filename,
        neff_filename=neff_filename,
        metadata=metadata,
    )


def neuroncc_compile(hlo, compiler_workdir, compiler_args=None):
    """Compile HLO to NEFF using neuronx-cc compiler.

    This function builds the minimal neuronx-cc invocation (input, framework,
    target, output) and appends ``compiler_args`` verbatim.

    Args:
        hlo: HLO binary data to compile.
        compiler_workdir (str): Working directory for compilation artifacts.
        compiler_args (list, optional): Additional arguments for the neuronx-cc
            compiler.

    Returns:
        str: Path to the compiled NEFF file.
    """
    from vllm_neuron.compile.platform import resolve_target

    if compiler_args is None:
        compiler_args = []

    hlo_filename = os.path.join(compiler_workdir, "graph.hlo")
    command_filename = os.path.join(compiler_workdir, "command.txt")
    log_filename = os.path.join(compiler_workdir, "log-neuron-cc.txt")

    # Allow caller to override --output via compiler_args; otherwise default
    # to the hash-based NEFF filename in compiler_workdir.
    if "--output" in compiler_args:
        idx = compiler_args.index("--output")
        neff_filename = compiler_args[idx + 1]
        compiler_args = compiler_args[:idx] + compiler_args[idx + 2 :]
    else:
        neff_filename = os.path.join(
            compiler_workdir, get_neff_filename(compiler_workdir)
        )

    with open(hlo_filename, "wb") as f:
        f.write(hlo)
        f.flush()

    # Extract --target from caller args or fall back to platform default.
    # Precedence: env var > --target from args > NRT auto-detect.
    target = resolve_target(compiler_args)

    parser = argparse.ArgumentParser()
    known_flags, remaining_args = parser.parse_known_args(compiler_args)

    neuron_cc = shutil.which("neuronx-cc")
    if neuron_cc is None:
        raise RuntimeError("neuronx-cc compiler binary does not exist")

    command = [
        neuron_cc,
        "compile",
        hlo_filename,
        "--framework",
        "XLA",
        "--target",
        target,
        "--output",
        neff_filename,
        "--logfile",
        log_filename,
    ]

    command.extend(remaining_args)

    with open(command_filename, "w") as f:
        f.write(shlex.join(command))

    logger.info("Compiling...")
    start_time = time.time()
    status = subprocess.run(command, stdout=subprocess.DEVNULL).returncode
    elapsed_time = time.time() - start_time

    if status != 0:
        if status == -9:
            logger.warning(
                "The neuronx-cc (neuron compiler) process was killed (SIG_KILL).  "
                "This typically happens when there is insufficient memory to compile and the linux "
                "Out Of Memory (OOM) killer terminates the compiler. "
                "Consider trying compilation on an instance with more memory"
            )
        elif status == -6:
            logger.warning(
                "The neuronx-cc (neuron compiler) process aborted (SIG_ABORT). "
                "This is likely due to an unexpected condition internally (a bug).  "
                "Please lodge an issue at 'https://github.com/aws/aws-neuron-sdk/issues'"
            )
        elif status == -11:
            logger.warning(
                "The neuronx-cc (neuron compiler) crashed (SEGFAULT). "
                "This is likely due to a bug in the compiler.  "
                "Please lodge an issue at 'https://github.com/aws/aws-neuron-sdk/issues'"
            )

        raise RuntimeError(
            f"neuronx-cc compilation failed with {status}. Check compiler logs."
        )
    logger.info(f"Compilation complete in {elapsed_time:.2f} seconds")

    return neff_filename


def build_executable(
    hlo_filename, neff_filename, g_device_id, g_device_count, artifacts
):
    from torch_neuronx.pyhlo import xla_data_pb2
    from vllm_neuron.compile.hlo import load_hlo_module

    xla_dtype_to_torch_dtype = {
        xla_data_pb2.F32: torch.float32,
        xla_data_pb2.F64: torch.float64,
        xla_data_pb2.BF16: torch.bfloat16,
        xla_data_pb2.F16: torch.float16,
        xla_data_pb2.U8: torch.uint8,
        xla_data_pb2.S8: torch.int8,
        xla_data_pb2.U16: torch.uint16,
        xla_data_pb2.S16: torch.int16,
        xla_data_pb2.U32: torch.uint32,
        xla_data_pb2.S32: torch.int32,
        xla_data_pb2.U64: torch.uint64,
        xla_data_pb2.S64: torch.int64,
        xla_data_pb2.C64: None,
        xla_data_pb2.C128: None,
        xla_data_pb2.PRED: torch.bool,
        xla_data_pb2.F8E4M3FN: torch.float8_e4m3fn,
        xla_data_pb2.F8E5M2: torch.float8_e5m2,
    }

    hlo = load_hlo_module(hlo_filename)

    def get_hlo_entry_computation(hlo):
        for computation in hlo.computations:
            if computation.id == hlo.entry_computation_id:
                return computation

    computations = get_hlo_entry_computation(hlo)

    program_shape = computations.program_shape

    class Executable:
        def __init__(self, neff_filename, g_device_id, g_device_count, artifacts):
            start_nc = 0
            self.executor = torch.classes.neuron.Executor(
                neff_filename, start_nc, g_device_id, g_device_count
            )
            metadata = artifacts.metadata
            self.io_map = metadata.io_map or {}
            # Set of input indices that were DCE'd by XLA and should be filtered out
            self.unused_input_indices = set(metadata.unused_input_indices or [])
            self.original_output_count = metadata.output_count
            self.append_rng_seed_parameter = metadata.has_rng_seed_parameter

        def __call__(self, *inputs):
            # Filter out unused inputs that were DCE'd by XLA
            if self.unused_input_indices:
                filtered_inputs = [
                    inp
                    for i, inp in enumerate(inputs)
                    if i not in self.unused_input_indices
                ]
            else:
                filtered_inputs = list(inputs)

            outputs = []
            for i, metadata in enumerate(program_shape.result.tuple_shapes):
                shape = list(metadata.dimensions)
                dtype = xla_dtype_to_torch_dtype[metadata.element_type]

                # Check if this output is aliased to an input
                if i in self.io_map:
                    input_idx = self.io_map[i]
                    outputs.append(inputs[input_idx])
                else:
                    outputs.append(
                        # NOTE: We cannot use `g_device_id` here because it is defined in the global
                        # world size, which may differ from the local device index in more complex
                        # parallelism setups.
                        # For example, with TP=2 and DP=2, `g_device_id` ranges from 0–3,
                        # while tensor device indices are always tp_group local (0 or 1).
                        # So here we just use the same device as input.
                        torch.empty(shape, dtype=dtype, device=inputs[0].device)
                    )

            if self.append_rng_seed_parameter:
                # Not even though RNG seed is s64 parameter in HLO, it's range is limited to s32.
                rng_seed = torch.randint(high=1 << 31, size=(), dtype=torch.int64)
                filtered_inputs.append(rng_seed.to(filtered_inputs[0].device))

            self.executor.execute(tuple(filtered_inputs), outputs)

            # The aliasing map can add additional outputs not part of the original return
            # For example, the forward does not return KV cache but updates KV cache in-place
            # The aliasing pass will add KV cache to the output, but this should be hidden from the final output.
            if self.original_output_count:
                return outputs[0 : self.original_output_count]

            return outputs

    return Executable(neff_filename, g_device_id, g_device_count, artifacts)


def preprocess_and_validate_inputs(gm, example_inputs):
    """Validate inputs are on device and deduplicate them.

    Combines device validation with AOTDedupeWrapper-style input deduplication.
    Duplicate inputs are detected by storage identity and removed from both the
    FX graph and the example_inputs list so that the cache key reflects the
    deduped graph.

    Args:
        gm: The PyTorch FX GraphModule
        example_inputs: Example input tensors

    Returns:
        tuple: (gm, example_inputs) with duplicates removed
    """
    _validate_inputs_on_device(example_inputs)

    if envs.VLLM_NEURON_DISABLE_INPUT_DEDUP:
        logger.info(
            "Input dedup skipped (VLLM_NEURON_DISABLE_INPUT_DEDUP=1); "
            "every storage-aliased FX placeholder will become its own HLO parameter"
        )
        return gm, example_inputs

    keep_mask, dupe_map = _detect_duplicate_inputs(example_inputs)
    has_dupes = any(not k for k in keep_mask)

    if has_dupes:
        gm = _dedup_fx_graph(gm, keep_mask, dupe_map)
        example_inputs = [inp for inp, keep in zip(example_inputs, keep_mask) if keep]

    return gm, example_inputs


def _is_remote_filesystem(path: str) -> bool:
    """Check if a path is on a remote filesystem (NFS or Lustre).

    Uses the 'stat -f' command to determine the filesystem type of the given path.
    Returns True if the filesystem is NFS or Lustre (including FSx for Lustre), False otherwise.

    Args:
        path: An existing filesystem path to check.

    Returns:
        bool: True if the filesystem is NFS (any variant) or Lustre, False otherwise.
    """
    cmd = ["stat", "-f", "-c", "%T", path]
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        fs_type = result.strip().lower()
        return fs_type.startswith("nfs") or fs_type == "lustre"
    except subprocess.CalledProcessError:
        return False


def _validate_inputs_on_device(example_inputs):
    """Validate all inputs are on expected device (neuron/meta).

    Args:
        example_inputs: Input tensors to validate

    Raises:
        ValueError: If any input tensor is not on neuron device
    """
    if envs.VLLM_NEURON_CPU_COMPILE:
        # CPU Compilation uses meta as the device type.
        device = "meta"
    else:
        device = "neuron"
    for tensor in example_inputs:
        if tensor.device.type != device:
            raise ValueError(
                f"vLLM Neuron Compilation called with tensors not on {device}. Move all parameters, buffers and inputs to {device}. "
                f"Tensor with shape {tensor.shape} is on device {tensor.device}"
            )


def _setup_workdir(gm: torch.fx.GraphModule, example_inputs, options: dict) -> str:
    """Setup working directory for compilation.

    Args:
        gm: The PyTorch FX GraphModule
        example_inputs: Example inputs
        options: Compilation options

    Returns:
        str: Path to the working directory
    """
    from vllm_neuron.compile.capture_backend import setup_workdir_common

    workdir, _, _ = setup_workdir_common(gm, example_inputs, options, per_rank=False)
    os.makedirs(workdir, exist_ok=True)
    return workdir


def _log_compilation_times(compilation_times: Dict[str, float]) -> None:
    """Log compilation time breakdown with color formatting.

    Args:
        compilation_times: Dictionary containing timing information
    """
    fx_to_hlo_time = compilation_times["fx_to_hlo"]
    compile_time = compilation_times["compile"]

    red = "\033[91m"
    reset = "\033[0m"
    fx_str = (
        f"{red}{fx_to_hlo_time:.2f}{reset}"
        if fx_to_hlo_time > 1
        else f"{fx_to_hlo_time:.2f}"
    )
    compile_str = (
        f"{red}{compile_time:.2f}{reset}" if compile_time > 1 else f"{compile_time:.2f}"
    )
    total = fx_to_hlo_time + compile_time
    total_str = f"{red}{total:.2f}{reset}" if total > 1 else f"{total:.2f}"

    logger.info(
        f"Compilation time split - FX to HLO: {fx_str}s | Compile: {compile_str}s | Total: {total_str}s"
    )


def _parse_compiler_args(raw: str | list | None) -> List[str]:
    """Normalize compiler_args from options into a list of strings.

    Accepts a string (shell-quoted), a list of strings, or None.

    Args:
        raw: Compiler args in string, list, or None form.

    Returns:
        List of individual argument strings.
    """
    if not raw:
        return []

    if isinstance(raw, str):
        return shlex.split(raw)

    if isinstance(raw, list):
        return list(raw)

    raise TypeError(
        f"compiler_args must be a str, list, or None, got {type(raw).__name__}"
    )


_HLO2TENSORIZER_FLAG = "--internal-hlo2tensorizer-options="


def _apply_platform_compiler_args(options: dict) -> dict:
    """Return ``options`` with platform-dependent compiler_args applied.

    Called before cache-hash computation so the hash reflects what the
    compiler actually runs. On trn2, injects
    ``--experimental-unsafe-fp8e4m3fn-as-fp8e4m3`` into
    ``--internal-hlo2tensorizer-options`` (trn2 lacks native OCP fp8_e4m3fn).
    No-op when the graph has no fp8 tensors, so it is safe to apply
    unconditionally on trn2. Trn3+ supports OCP natively and adding this flag
    triggers NCC_EOCP001 against NKI kernels that emit OCP.
    """
    from vllm_neuron.compile.platform import resolve_target

    compiler_args = _parse_compiler_args(options.get("compiler_args", ""))
    target = resolve_target(compiler_args)

    if target == "trn2":
        compiler_args = _inject_hlo2tensorizer_opt(
            compiler_args, "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"
        )
    else:
        return options

    return {**options, "compiler_args": compiler_args}


def _inject_hlo2tensorizer_opt(compiler_args: List[str], opt: str) -> List[str]:
    """Append ``opt`` to ``--internal-hlo2tensorizer-options``, adding the flag
    if missing. Idempotent: does not duplicate an already-present option.
    """
    result = list(compiler_args)
    for i, arg in enumerate(result):
        if arg.startswith(_HLO2TENSORIZER_FLAG):
            existing = arg[len(_HLO2TENSORIZER_FLAG) :]
            if opt in shlex.split(existing):
                return result
            result[i] = f"{_HLO2TENSORIZER_FLAG}{existing} {opt}".strip()
            return result
    result.append(f"{_HLO2TENSORIZER_FLAG}{opt}")
    return result
