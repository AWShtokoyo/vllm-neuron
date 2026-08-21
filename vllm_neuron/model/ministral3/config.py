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
import os
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

    # ── 🔬 PER-LAYER FP8 GATING — root-cause debugging only (2026-08-14) ──────
    # WHY THIS EXISTS. Sessions 1–5 established that `o_proj`'s static FP8 is the
    # main contributor to the numeric-corruption defect, and that the residual
    # 4/24 is NOT attention-side (`fp8:mlp`, i.e. attention entirely BF16, still
    # scores 4/24). That leaves the MLP or something outside FP8 — and the
    # family-level knob cannot distinguish them, because putting all 88 MLP
    # layers in BF16 needs 26.9 GB/core against 24 available.
    #
    # 🔑 THE MEMORY WALL IS ONLY A WALL FOR *ALL* LAYERS. One MLP layer moved from
    # FP8 to BF16 costs 126 MB/core, so with the KV cache shrunk to debugging size
    # (num_seqs=1, max_model_len=512 → 0.011 GB/core) there is room for ~67 of 88
    # MLP layers. That makes a LAYER BISECTION feasible for the MLP too, which is
    # what this knob is for: find *where* in the 88 layers the defect originates.
    # A defect localised to a few layers implies layer-specific numerics (which
    # would explain why accuracy is NOT monotonic in quantization error:
    # 8/24 → 8/24 → 13/24 → 4/24 across the four repair attempts); a defect spread
    # evenly implies accumulation and sends us elsewhere.
    #
    # SYNTAX — `MINISTRAL3_FP8_LAYERS`, comma-separated `family:ranges` clauses:
    #     "o_proj:0-43"            o_proj FP8 only on layers 0..43; BF16 on 44..87
    #     "mlp:0-43,o_proj:all"    MLP FP8 on 0..43, o_proj FP8 everywhere
    #     "mlp:none"               MLP BF16 on every layer (needs the memory)
    #     "o_proj:0-9,20-29"       multiple ranges per family
    # A family absent from the spec keeps its `dense_fp8_static` setting on ALL
    # layers, so **an unset env var is byte-identical to the shipped behaviour**
    # and every recorded arm reproduces.
    # ⚠️ This gates the WEIGHT DTYPE, so an arm here is a distinct compiled graph
    # AND a distinct weight layout. It is a diagnostic, never a shipping config.
    @staticmethod
    def _parse_layer_spec(spec: str | None) -> dict:
        """Parse ``MINISTRAL3_FP8_LAYERS`` into {family: set(layer_idx) | None}.

        ``None`` for a family means "no per-layer restriction" (all layers).
        """
        if not spec or not spec.strip():
            return {}
        valid = {"qkv", "o_proj", "mlp"}
        out: dict[str, set | None] = {}
        for clause in spec.split(","):
            clause = clause.strip()
            if not clause:
                continue
            if ":" in clause:
                fam, rng = clause.split(":", 1)
                fam, rng = fam.strip().lower(), rng.strip().lower()
            else:
                # A bare range continues the previous family: "o_proj:0-9,20-29"
                if not out:
                    raise ValueError(
                        f"MINISTRAL3_FP8_LAYERS: range {clause!r} has no family prefix"
                    )
                fam, rng = list(out)[-1], clause.lower()
            if fam not in valid:
                raise ValueError(
                    f"MINISTRAL3_FP8_LAYERS: unknown family {fam!r}; expected one of "
                    f"{sorted(valid)}"
                )
            if rng == "all":
                out[fam] = None
                continue
            if rng == "none":
                out.setdefault(fam, set())
                if out[fam] is None:
                    out[fam] = set()
                continue
            cur = out.get(fam)
            if cur is None and fam in out:
                # "all" already seen for this family: a later range narrows nothing
                continue
            acc = cur if isinstance(cur, set) else set()
            if "-" in rng:
                lo, hi = rng.split("-", 1)
                acc |= set(range(int(lo), int(hi) + 1))
            else:
                acc.add(int(rng))
            out[fam] = acc
        return out

    def _layer_spec(self) -> dict:
        cached = getattr(self, "_fp8_layer_spec_cache", None)
        if cached is None:
            cached = self._parse_layer_spec(os.environ.get("MINISTRAL3_FP8_LAYERS"))
            object.__setattr__(self, "_fp8_layer_spec_cache", cached)
        return cached

    def fp8_enabled(self, family: str, layer_idx: int | None = None) -> bool:
        """True if ``family`` ("qkv"|"o_proj"|"mlp") should run FP8-native.

        ``layer_idx`` restricts the answer to one layer when
        ``MINISTRAL3_FP8_LAYERS`` names that family; otherwise the family-level
        answer applies to every layer (the shipped behaviour).
        """
        if family not in getattr(self, "_fp8_families", frozenset()):
            return False
        spec = self._layer_spec()
        if family not in spec or layer_idx is None:
            return True
        allowed = spec[family]
        return True if allowed is None else (layer_idx in allowed)

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
