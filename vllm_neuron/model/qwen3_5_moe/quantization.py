# SPDX-License-Identifier: Apache-2.0
"""FP8 quantization scheme parsing for the Qwen3.5 hybrid-MoE port.

The ``Qwen3.6-35B-A3B-FP8`` checkpoint is a DeepSeek-style **block-FP8** model:
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
FP8). See ``weight_loaders_fp8_row.py`` for the load-time transform.

Where the saving comes from: almost all of this model's parameters are the 256
experts, so FP8 is only worth anything if the EXPERT weights stay fp8-resident.
They do: the decode MoE kernel (``moe_block_tkg``) has an FP8-ROW path that
takes ``[E_L, 2, I]`` / ``[E_L, H]`` fp32 scales alongside fp8 expert weights
(``QuantizationType.ROW``, documented TRN2).

Prefill needs a specific ``moe_cte`` implementation. ``shard_on_block`` — the one
the bf16 path uses — accepts ``gate_up_proj_scale`` / ``down_proj_scale`` in its
signature but never loads them: ``bwmm_shard_on_block.py:244`` declares the local
scale tiles ``# Placeholder for FP8`` and always passes ``None`` down to the
projections, so the dequant multiply never runs and the fp8 bytes would be
consumed as though they were already dequantised — silently wrong output, no
error. ``shard_on_i`` (also TRN2; it asserts ``NUM_SHARDS == 2``, i.e. LNC-2)
does implement per-channel FP8: it reshapes the flat ``[E, 1, 2*I_TP]`` scale to
``[E, 2, I_TP]`` and applies it moving psum→sbuf. (The kernels' native
DeepSeek-block path is no help here: ``BLOCK_QUANT_SIZE`` is 256 and this
checkpoint's blocks are 128×128.)

So prefill has two routes, and MEASUREMENT picked the second one: block-dequant
the experts to bf16 per layer and keep the fast ``shard_on_block``. That beats
``shard_on_i`` at both low and high concurrency because shard_on_i's own cost over
shard_on_block (+74% TTFT, isolated with a bf16-through-shard_on_i control) exceeds
the dequant DMA. See the table in ``model.py``'s ``forward_prefill``.
``VLLM_QWEN35_MOE_FP8_PREFILL=shard_on_i`` selects the other route; the default is
spelled ``dequant_to_bf16`` because both routes dequantise and only the *place* differs. Everything else
(attention QKV/o_proj, the GatedDeltaNet projections, the shared expert) is
block-dequantised to bf16 at load, which costs nothing in HBM terms because
those tensors are a small fraction of the model.

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
            "Qwen3.5-MoE FP8 port only supports DeepSeek block-FP8 with "
            f"weight_block_size {_SUPPORTED_WEIGHT_BLOCK_SIZE}; got "
            f"{weight_block_size!r}. Per-tensor/ModelOpt FP8 is not wired for "
            "this model."
        )

    raise NotImplementedError(
        f"Qwen3.5-MoE port does not handle quant_method={quant_method!r}; "
        "only DeepSeek block-FP8 (quant_method='fp8', weight_block_size "
        f"{_SUPPORTED_WEIGHT_BLOCK_SIZE}) or an un-quantized (bf16) checkpoint."
    )
