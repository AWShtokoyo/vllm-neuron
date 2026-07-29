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
    _register_gemma4_hf_config()

    return get_platform_class()


def _register_gemma4_hf_config() -> None:
    """Register a text-only Gemma4 model_type with HuggingFace AutoConfig.

    google/gemma-4-31B-it uses model_type `gemma4` (text params nested under
    `text_config`, typed as `gemma4_text`). vLLM calls AutoConfig.from_pretrained()
    before dispatching to our Neuron model, so we register minimal PretrainedConfig
    subclasses that carry the nested text_config through. The actual model is built
    from these via vllm_neuron.model.gemma4.config.Gemma4Config.from_configs().

    We serve Gemma4 **text-only**, which mirrors how transformers 5.x itself
    separates gemma4 into a TEXT architecture (`Gemma4ForCausalLM` /
    `gemma4_text`) and a MULTIMODAL one (`Gemma4ForConditionalGeneration` /
    `gemma4_mm`). Two things are forced here so the model routes down the text
    path end-to-end:

    1. **`architectures = ['Gemma4ForCausalLM']`.** The gemma-4-31B checkpoint
       declares the *multimodal* arch name `Gemma4ForConditionalGeneration`,
       which vLLM-core maps to `gemma4_mm` (a multimodal model). Multimodality
       is decided from the arch name in the *main* process — before our Neuron
       worker registers anything — so overriding the config class alone is not
       enough: vLLM-core's renderer still selects `gemma4_mm` and its
       `get_supported_mm_limits()` rejects our config type. Rewriting the arch
       to the native text name `Gemma4ForCausalLM` (vLLM-core: non-multimodal
       text) is what actually disengages the multimodal renderer. This is the
       orthodox alignment to upstream's own text/mm split.
    2. **Drop `vision_config` / `audio_config`.** With no `vision_config`
       attribute, the Neuron platform also keeps the model off its multimodal
       load/forward path (`text_neuron_config`/`vision_neuron_config` kwargs and
       `vision_embedding_blocks`/`vision_positions` forward inputs), which the
       text-only decoder does not accept.

    Newer transformers releases (5.x) ship a *native* multimodal `gemma4` config
    that always synthesizes a `vision_config`. We therefore register with
    `exist_ok=True` to override that native mapping with our text-only config,
    rather than skipping when the model_type is already present.
    """
    from transformers import AutoConfig, PretrainedConfig

    class _Gemma4TextConfig(PretrainedConfig):
        model_type = "gemma4_text"

    class _Gemma4Config(PretrainedConfig):
        model_type = "gemma4"
        sub_configs = {"text_config": _Gemma4TextConfig}

        def __init__(self, text_config=None, **kwargs):
            # Text-only: drop the multimodal sub-configs so downstream
            # `hasattr(hf_config, "vision_config")` checks stay False.
            kwargs.pop("vision_config", None)
            kwargs.pop("audio_config", None)
            # Present as the transformers-5.x native TEXT architecture so
            # vLLM-core routes to the non-multimodal text model (see docstring).
            kwargs["architectures"] = ["Gemma4ForCausalLM"]
            if isinstance(text_config, dict):
                text_config = _Gemma4TextConfig(**text_config)
            self.text_config = text_config
            super().__init__(**kwargs)

    for mt, cls in (("gemma4_text", _Gemma4TextConfig), ("gemma4", _Gemma4Config)):
        AutoConfig.register(mt, cls, exist_ok=True)


def __getattr__(name):
    import importlib

    if name == "nn":
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
