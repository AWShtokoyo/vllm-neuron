# SPDX-License-Identifier: Apache-2.0
"""
Ministral3 Configuration
================================
<-- MODEL-SPECIFIC: All fields in this config are model-specific.
Ported from the Llama dense config for Mistral AI's Ministral3 architecture
(e.g. Devstral-2-123B-Instruct-2512).

Architecture overview:
- Dense Transformer decoder (no MoE)
- GQA (Grouped Query Attention) with separate Q and KV head counts
- YaRN RoPE for position encoding (transformers _compute_yarn_parameters)
- SiLU-gated MLP (SwiGLU), no bias on any projection
- Pre-attention and pre-MLP RMSNorm
- Untied embeddings (separate lm_head)
- FP8 (per-tensor static) checkpoint, dequantized to BF16 at load time
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Ministral3Config:
    """Configuration for the Ministral3 dense decoder.

    <-- MODEL-SPECIFIC: Defaults reflect Devstral-2-123B-Instruct-2512. Values are
    overridden from the HuggingFace config in ``from_configs``.
    """

    # ── Model architecture (MODEL-SPECIFIC) ──────────────────────────────
    vocab_size: int = 131072
    hidden_size: int = 12288
    unpadded_hidden_size: int | None = None
    intermediate_size: int = 28672
    num_hidden_layers: int = 88
    num_attention_heads: int = 96  # Q heads
    num_key_value_heads: int = 8  # KV heads (GQA)
    head_dim: int = 128  # Per-head dimension (explicit in HF config)
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-5

    # ── RoPE settings (MODEL-SPECIFIC: YaRN scaling) ─────────────────────
    # rope_scaling carries the YaRN parameters (rope_type, factor, beta_fast,
    # beta_slow, original_max_position_embeddings, mscale, mscale_all_dim).
    rope_theta: float = 1000000.0
    rope_scaling: dict | None = None

    # ── Embeddings ────────────────────────────────────────────────────────
    tie_word_embeddings: bool = False

    torch_dtype: torch.dtype = torch.bfloat16

    # ── Framework config (not model-specific) ────────────────────────────
    neuron_config: NeuronConfig | None = None

    # ── FP8-native dense projections (MODEL-SPECIFIC) ────────────────────
    # When set, the seven linear projections (q/k/v/o_proj, gate/up/down) are
    # kept in FP8 (E4M3) and run as FP8×FP8 STATIC per-tensor matmuls on TRN2,
    # instead of being dequantized to BF16 at load time. Default OFF → the
    # existing, validated BF16-dequant path is unchanged.
    #
    # The value is a comma-separated set of projection families to run FP8-native
    # ("qkv", "o_proj", "mlp"), or "all", or None/"" (off). Per-family granularity
    # exists for bring-up isolation (enable qkv, then o_proj, then mlp). Families
    # not listed fall back to BF16-dequant. Only meaningful on-device (TRN2): the
    # CPU/torch fallbacks reject quantization, so a CPU-sim run must leave this off.
    dense_fp8_static: str | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.unpadded_hidden_size is None:
            self.unpadded_hidden_size = self.hidden_size
        self._fp8_families = self._parse_fp8_families(self.dense_fp8_static)

    @staticmethod
    def _parse_fp8_families(spec: str | None) -> frozenset[str]:
        """Normalize ``dense_fp8_static`` into a set of {"qkv","o_proj","mlp"}."""
        if not spec:
            return frozenset()
        spec = spec.strip().lower()
        if spec == "all":
            return frozenset({"qkv", "o_proj", "mlp"})
        valid = {"qkv", "o_proj", "mlp"}
        fams = {f.strip() for f in spec.split(",") if f.strip()}
        unknown = fams - valid
        if unknown:
            raise ValueError(
                f"dense_fp8_static: unknown projection family/families {sorted(unknown)}; "
                f"expected a comma-separated subset of {sorted(valid)}, 'all', or None."
            )
        return frozenset(fams)

    def fp8_enabled(self, family: str) -> bool:
        """True if ``family`` ("qkv"|"o_proj"|"mlp") should run FP8-native."""
        return family in getattr(self, "_fp8_families", frozenset())

    @property
    def any_fp8(self) -> bool:
        return bool(getattr(self, "_fp8_families", frozenset()))

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig = None
    ):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            if hasattr(hf_config, "torch_dtype") and hf_config.torch_dtype is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = hf_config

        # <-- MODEL-SPECIFIC: Ministral3 (transformers 5.x) stores YaRN parameters
        # under the new key ``rope_parameters``. Older configs use ``rope_scaling``.
        # Normalize to ``rope_scaling`` (and to a plain dict) for the RoPE module.
        if config_dict.get("rope_scaling") is None and config_dict.get(
            "rope_parameters"
        ):
            config_dict["rope_scaling"] = dict(config_dict["rope_parameters"])
        # ``rope_theta`` may live inside the rope dict in the new schema.
        rope = config_dict.get("rope_scaling")
        if rope is not None and "rope_theta" in rope and "rope_theta" not in config_dict:
            config_dict["rope_theta"] = rope["rope_theta"]

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config
            # FP8-native dense projections are opted into at runtime via
            # ``additional_config={'neuron_config': {'quantization': ...}}``
            # (mirrors solar-open2). Accepted values: "fp8"/"fp8_all"/"all" → all
            # families; a comma list like "fp8:qkv,mlp" or just "qkv,mlp" → those
            # families; None/"bf16"/"" → off (BF16-dequant). The HF config key
            # ``dense_fp8_static`` (if present) takes precedence over neuron_config.
            quant = getattr(neuron_config, "quantization", None)
            if "dense_fp8_static" not in filtered_dict and quant:
                filtered_dict["dense_fp8_static"] = cls._normalize_quant_flag(quant)

        return cls(**filtered_dict)

    @staticmethod
    def _normalize_quant_flag(quant: str) -> str | None:
        """Map a ``neuron_config.quantization`` string to ``dense_fp8_static``.

        "fp8" / "fp8_all" / "all" → "all"; "bf16" / None / "" → None (off); a
        "fp8:qkv,mlp" form → "qkv,mlp"; a bare "qkv,mlp" / family name → itself.
        """
        q = quant.strip().lower()
        if q in ("", "bf16", "none"):
            return None
        if q in ("fp8", "fp8_all", "all"):
            return "all"
        if q.startswith("fp8:"):
            return q[len("fp8:"):]
        return q  # bare family list, validated by _parse_fp8_families
