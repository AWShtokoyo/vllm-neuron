# SPDX-License-Identifier: Apache-2.0
"""FP8 quantization scheme parsing for the Qwen3.5 hybrid-dense port.

The ``Qwen3.8-27B-FP8`` checkpoint is a DeepSeek-style **block-FP8** model:
every quantized ``nn.Linear`` stores a ``.weight`` (``float8_e4m3fn``) plus a
companion ``.weight_scale_inv`` (bf16, one scalar per ``[128, 128]`` block along
BOTH the output and input dims), with a top-level ``quantization_config`` of::

    {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
     "weight_block_size": [128, 128]}

The Neuron NKI GEMMs do NOT support a resident block-``[128, 128]`` scale (that
path is CUDA/Hopper-only). The supported resident-FP8 schemes are per-tensor
STATIC, per-channel ROW, and MX block-32. Because the checkpoint's activation
scheme is *dynamic* (no calibrated input scales are shipped), the natural
Neuron-side target is **ROW**: the weights are re-quantized at load into
per-output-channel FP8 (dequant the ``[128, 128]`` blocks → per-row absmax →
FP8), and the activations are quantized dynamically per-token online via
``NF.rmsnorm_quant(ROW)``. See ``weight_loaders_fp8_row.py`` for the load-time
transform.

This module only *parses* the HF ``quantization_config`` into the local
:class:`QuantScheme`. Default (no quant config) is :attr:`QuantScheme.NONE`,
which keeps the model byte-identical to the bf16 path.
"""

from __future__ import annotations

from enum import Enum


class QuantScheme(Enum):
    """Resident-weight quantization scheme selected for the served model."""

    NONE = "none"  # bf16 weights (default; byte-identical to the un-quantized port)
    FP8_ROW = "fp8_row"  # block-FP8 checkpoint re-quantized at load to per-channel ROW FP8


# The only block geometry this port re-quantizes. DeepSeek block-FP8 uses a
# square [128, 128] block along both weight dims; every quantized dim in this
# checkpoint is an exact multiple of 128 (no partial blocks).
_SUPPORTED_WEIGHT_BLOCK_SIZE = [128, 128]


def parse_quant_config(hf_quant_config: dict | None) -> QuantScheme:
    """Map an HF ``quantization_config`` dict to a :class:`QuantScheme`.

    Args:
        hf_quant_config: the raw ``quantization_config`` sub-dict from the HF
            ``config.json`` (or ``None`` for an un-quantized checkpoint).

    Returns:
        :attr:`QuantScheme.NONE` when there is no quantization config;
        :attr:`QuantScheme.FP8_ROW` for a block-``[128, 128]`` FP8 checkpoint.

    Raises:
        NotImplementedError: for an FP8 checkpoint whose block size this port
            does not handle, or for any non-FP8 ``quant_method`` — failing loud
            rather than silently loading FP8 bytes into bf16 params (a dtype
            mismatch that would surface as an obscure load error downstream).
    """
    if not hf_quant_config:
        return QuantScheme.NONE

    quant_method = hf_quant_config.get("quant_method")
    if quant_method == "fp8":
        weight_block_size = hf_quant_config.get("weight_block_size")
        if weight_block_size == _SUPPORTED_WEIGHT_BLOCK_SIZE:
            return QuantScheme.FP8_ROW
        raise NotImplementedError(
            "Qwen3.5-dense FP8 port only supports DeepSeek block-FP8 with "
            f"weight_block_size {_SUPPORTED_WEIGHT_BLOCK_SIZE}; got "
            f"{weight_block_size!r}. Per-tensor/ModelOpt FP8 is not wired for "
            "this model."
        )

    raise NotImplementedError(
        f"Qwen3.5-dense port does not handle quant_method={quant_method!r}; "
        "only DeepSeek block-FP8 (quant_method='fp8', weight_block_size "
        f"{_SUPPORTED_WEIGHT_BLOCK_SIZE}) or an un-quantized (bf16) checkpoint."
    )
