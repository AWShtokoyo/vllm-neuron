# SPDX-License-Identifier: Apache-2.0
"""vLLM Neuron plugin module."""

import glob
import os
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from vllm_neuron import envs

# Redirect torch_neuronx → libtorch_neuronx_lite before anything imports it.
from vllm_neuron.utils.import_redirector import install as _install_redirector

if envs.VLLM_NEURON_LIBTORCH_NEURONX_LITE:
    _install_redirector()

# Initialize logging early so VLLM_NEURON_LOG_LEVEL takes effect
from vllm_neuron.logging_config import setup_logging as _setup_logging

_setup_logging()
import logging

logger = logging.getLogger(__name__)
# Enable prometheus multiprocess mode so that metrics observed in vLLM Neuron
# EngineCore/Worker processes (e.g. scheduler padding metrics) are written to
# shared mmap files and visible at the API server's /metrics endpoint.
# Note: Prometheus requires PROMETHEUS_MULTIPROC_DIR to be set before
# prometheus_client is imported anywhere. Typically a user would set the env
# var manually, but we set it during package init to set this up for the user.
# In case prometheus_client is imported before vLLM Neuron, users can
# set PROMETHEUS_MULTIPROC_DIR manually in their env to enable metrics.
if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
    import tempfile

    os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(
        prefix="vllm_neuron_prometheus_"
    )

# Suppress PyTorch kernel override warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Overriding a previously registered kernel.*",
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*other operators may also be overridden.*"
)


def _is_neuron_dev() -> bool:
    """Detect Neuron device by checking for /dev/neuron* devices."""
    neuron_devices = glob.glob("/dev/neuron*")
    return len(neuron_devices) > 0


def _is_cpu_mode() -> bool:
    """Check if VLLM_NEURON_CPU_MODE is enabled via environment variable."""
    return os.environ.get("VLLM_NEURON_CPU_MODE", "0") == "1"


def _is_cpu_compile() -> bool:
    """Check if VLLM_NEURON_CPU_COMPILE is enabled via environment variable."""
    return os.environ.get("VLLM_NEURON_CPU_COMPILE", "0") == "1"


def _init_backend():
    """Initialize the vllm_neuron backend (XLA path)."""
    from vllm_neuron import envs

    if envs.is_native_backend():
        return

    # In CPU mode, the NKI CPU simulator is OFF by default.
    # Users opt-in by setting NKI_SIMULATOR=1 for kernel development,
    # integration and accuracy validation on small shapes and tiny models.
    if envs.VLLM_NEURON_CPU_MODE:
        if os.environ.get("NKI_SIMULATOR") == "1":
            os.environ.setdefault("NKI_PRECISE_FP", "1")

    if envs.VLLM_NEURON_CPU_MODE and _is_cpu_compile():
        raise RuntimeError(
            "VLLM_NEURON_CPU_MODE and VLLM_NEURON_CPU_COMPILE are not compatible with each other"
        )

    os.environ["PJRT_DEVICE"] = "CPU"

    import torch
    import torch._dynamo.backends.registry as registry

    _has_neuron_hw = _is_neuron_dev()

    torch_neuronx = None
    if not envs.VLLM_NEURON_CPU_MODE or _has_neuron_hw:
        try:
            import torch_neuronx  # noqa: E402
        except ImportError:
            pass

    from vllm_neuron.compile.backend import compile
    from vllm_neuron.compile.capture_backend import capture

    if "vllm_neuron" not in registry.list_backends():
        registry.register_backend(compiler_fn=compile, name="vllm_neuron")

    if "vllm_neuron_graph_capture" not in registry.list_backends():
        registry.register_backend(compiler_fn=capture, name="vllm_neuron_graph_capture")

    if torch_neuronx is not None:
        from vllm_neuron import overrides  # noqa: F401

        sys.modules["torch.neuron"] = torch_neuronx
        torch.neuron = torch_neuronx
        torch.neuron.is_available = lambda: True
        torch.neuron.current_device = lambda: 0

        if not envs.VLLM_NEURON_LIBTORCH_NEURONX_LITE:
            # libtorch_neuronx_lite does this internally
            torch.utils.rename_privateuse1_backend("neuron")
            unsupported_dtype = [
                torch.float64,
                torch.uint16,
                torch.uint32,
                torch.uint64,
            ]
            try:
                torch.utils.generate_methods_for_privateuse1_backend(
                    for_tensor=True,
                    for_module=True,
                    for_storage=True,
                    unsupported_dtype=unsupported_dtype,
                )
            except RuntimeError as e:
                if "already been registered" not in str(e):
                    raise
                logger.debug("privateuse1 backend already registered, skipping: %s", e)

    else:
        # CPU-only: register minimal privateuse1 backend so dynamo's
        # SymbolicStreamState doesn't fall through to CUDA
        import types

        torch.utils.rename_privateuse1_backend("neuron")
        neuron_module = types.ModuleType("torch.neuron")
        neuron_module.is_available = lambda: False
        neuron_module.current_device = lambda: 0
        neuron_module.current_stream = lambda device=None: type(
            "_S", (), {"device": torch.device("cpu")}
        )()
        neuron_module.set_stream = lambda stream: None
        sys.modules["torch.neuron"] = neuron_module
        torch.neuron = neuron_module

        try:
            torch.utils.generate_methods_for_privateuse1_backend(
                for_tensor=True,
                for_module=True,
                for_storage=True,
            )
        except RuntimeError:
            pass

    _original_current_accelerator = torch.accelerator.current_accelerator

    def _current_accelerator_wrapper(check_available: bool = False):
        device = _original_current_accelerator(check_available)
        if device is not None:
            if envs.VLLM_NEURON_CPU_MODE:
                return torch.device("cpu")
            elif envs.VLLM_NEURON_CPU_COMPILE:
                return torch.device("neuron", torch.neuron.current_device())
            elif device.type == "neuron" and device.index is None:
                return torch.device("neuron", torch.neuron.current_device())
            else:
                return device
        return device

    torch.accelerator.current_accelerator = _current_accelerator_wrapper

    # Patch accelerator stream/device APIs for CPU mode to prevent CUDA fallthrough
    if envs.VLLM_NEURON_CPU_MODE:
        torch.accelerator.current_stream = lambda device=None: type(
            "_S", (), {"device": torch.device("cpu")}
        )()
        torch.accelerator.current_device_index = lambda: 0


try:
    _init_backend()
except (ImportError, KeyError):
    pass

# Import-time so it survives spawn-mode re-imports (EngineCore subprocess
# never calls check_and_update_config).
from vllm_neuron.vllm.patches.port_hold_patch import apply_port_hold_patch

apply_port_hold_patch()


def register():
    """Register the Neuron platform if Neuron devices are present, else return None.

    The backend is selected based on the VLLM_NEURON_BACKEND environment variable:
    - "vllm_neuron": Use the vLLM Neuron backend (default)
    - "neuron_native": Use the neuron native backend

    If VLLM_NEURON_BACKEND is not set, defaults to vllm_neuron.
    """
    if not _is_cpu_mode() and not _is_cpu_compile() and not _is_neuron_dev():
        warnings.warn(
            "No Neuron devices found. Skipping Neuron plugin registration.",
            category=UserWarning,
        )
        return None

    from vllm_neuron.backend import get_platform_class
    from vllm_neuron.vllm.platform import _patch_dcp_config_validation

    _patch_dcp_config_validation()
    _register_ministral3_hf_config()

    return get_platform_class()


# Module-level handle for the Ministral3 HF config class (bound lazily inside
# _register_ministral3_hf_config). Kept at module scope so instances are
# picklable across engine-core subprocesses under data_parallel_size>1.
_Ministral3HFConfig = None


def _register_ministral3_hf_config() -> None:
    """Register the Ministral3 model_type with HuggingFace AutoConfig.

    Ministral3 (e.g. Devstral-2-123B-Instruct-2512) targets transformers 5.x and
    is not in the installed transformers' CONFIG_MAPPING, so vLLM's
    AutoConfig.from_pretrained() would raise KeyError('ministral3'). We register a
    minimal PretrainedConfig subclass that simply absorbs all fields from the HF
    config dict (the Neuron model reads them via Ministral3Config.from_configs).
    """
    from transformers import AutoConfig, PretrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    global _Ministral3HFConfig
    if _Ministral3HFConfig is None:
        class _Ministral3HFConfig(PretrainedConfig):  # noqa: F811
            model_type = "ministral3"

        # Make the class picklable: with data_parallel_size>1 vLLM spawns
        # engine-core subprocesses and pickles the HF config across the process
        # boundary. A function-local class has qualname
        # "_register_ministral3_hf_config.<locals>._Ministral3HFConfig", which
        # pickle cannot resolve. Bind it to a module-level name so pickle finds
        # it as vllm_neuron._Ministral3HFConfig (TP-only never crosses the
        # boundary, which is why this only surfaces under DP).
        _Ministral3HFConfig.__qualname__ = "_Ministral3HFConfig"
        _Ministral3HFConfig.__module__ = __name__

    if "ministral3" not in CONFIG_MAPPING:
        AutoConfig.register("ministral3", _Ministral3HFConfig)


def __getattr__(name):
    import importlib

    if name == "nn":
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
