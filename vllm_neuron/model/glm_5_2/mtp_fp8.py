# SPDX-License-Identifier: Apache-2.0
"""FP8 (fp8_fwd ROW) variant of the GLM-5.2 MTP draft head.

Mirrors `model_fp8_fwd_dequant.py` (the FP8 ROW target model) but scoped to the
single layer-78 MTP head. The decoder body reuses the FP8 MoE / dense / shared-expert
classes (swapped in at construction via the same monkeypatch trick the target uses),
and `load_weights` loads ONLY `model.layers.78.*` plus the base `embed_tokens` /
`lm_head`:

  * self_attn projections: block-FP8 → dequant → shard → re-quantise per-row (row scale).
  * MoE 256 routed experts (EP-sliced) + shared_expert: same FP8 ROW loaders.
  * mlp.gate.weight [256,6144] + e_score_correction_bias [256] (BF16 / f32, NOT quantised).
  * MTP-only: eh_proj (BF16 replicated, D4), enorm, hnorm, shared_head.norm (D5).
  * SKIP all self_attn.indexer.* keys (present but unused in this port).
  * reuse base lm_head (D6, untied) + base embed_tokens.

We deliberately do NOT reuse the target `load_weights` (it loops `range(78)` and binds
`self.model.layers[id]` — it would IndexError at 78 and cannot populate a draft head;
verdict C3).
"""

import logging

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.parallel.neuron_parallel_state import (
    get_neuron_ep_degree,
    get_neuron_ep_rank,
)
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

from .config import Glm52Config
from .mtp import Glm52MtpForCausalLM as Glm52MtpForCausalLMBF16, MTP_LAYER_IDX
from .model_fp8_fwd_dequant import (
    Glm52DenseMLPFP8Fwd,
    Glm52MoELayerFP8Fwd,
    Glm52SharedExpertMLPFP8Fwd,
    _fp8_row_moe_weight_loader,
    _fp8_row_weight_loader,
)
from .weight_loaders_fp8 import (
    fp8_dequant_row_parallel_weight_loader,
    fp8_dequant_weight_loader,
)

logger = logging.getLogger(__name__)


class Glm52MtpForCausalLM(Glm52MtpForCausalLMBF16):
    """GLM-5.2 MTP draft head with FP8 ROW quantization (per-row weight scales)."""

    @torch.no_grad()
    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        tp_rank = self.sp_group.rank_in_group
        tp_size = self.sp_world_size

        if self.ep_degree > 1:
            ep_rank = get_neuron_ep_rank()
            num_local_experts = self.config.n_routed_experts // self.ep_degree
            local_expert_start = ep_rank * num_local_experts
        else:
            num_local_experts = self.config.n_routed_experts
            local_expert_start = 0

        cfg = self.config
        hidden = cfg.hidden_size
        hidden_per_sp = hidden // tp_size
        qk_rope_head_dim = cfg.qk_rope_head_dim
        qk_nope_head_dim = cfg.qk_nope_head_dim
        v_head_dim = cfg.v_head_dim
        num_heads_per_rank = cfg.num_attention_heads // tp_size
        q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_b_out_per_rank = num_heads_per_rank * (qk_nope_head_dim + v_head_dim)
        o_proj_in_per_rank = num_heads_per_rank * v_head_dim

        ep_tp_size = tp_size // self.ep_degree if self.ep_degree > 1 else tp_size
        moe_intermediate_per_rank = cfg.moe_intermediate_size // ep_tp_size
        shared_intermediate = cfg.moe_intermediate_size * cfg.n_shared_experts
        shared_intermediate_per_rank = shared_intermediate // tp_size

        prefix = f"model.layers.{MTP_LAYER_IDX}"
        decoder = self.mtp.decoder
        attn = decoder.self_attn
        moe = decoder.mlp  # Glm52MoELayerFP8Fwd (layer 78 is MoE)

        # ---- checkpoint key mappings (module-attr path -> checkpoint key(s)) ----
        mappings: dict = {}

        # MLA self-attention projections (block-FP8: weight + scale_inv pair).
        mappings["mtp.decoder.self_attn.q_a_proj_weight"] = [
            f"{prefix}.self_attn.q_a_proj.weight",
            f"{prefix}.self_attn.q_a_proj.weight_scale_inv",
        ]
        mappings["mtp.decoder.self_attn.q_b_proj_weight"] = [
            f"{prefix}.self_attn.q_b_proj.weight",
            f"{prefix}.self_attn.q_b_proj.weight_scale_inv",
        ]
        mappings["mtp.decoder.self_attn.kv_a_proj_weight"] = [
            f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
            f"{prefix}.self_attn.kv_a_proj_with_mqa.weight_scale_inv",
        ]
        mappings["mtp.decoder.self_attn.kv_b_proj_weight"] = [
            f"{prefix}.self_attn.kv_b_proj.weight",
            f"{prefix}.self_attn.kv_b_proj.weight_scale_inv",
        ]
        mappings["mtp.decoder.self_attn.o_proj_weight"] = [
            f"{prefix}.self_attn.o_proj.weight",
            f"{prefix}.self_attn.o_proj.weight_scale_inv",
        ]
        # Attention + layer RMSNorms (plain BF16 weights).
        mappings["mtp.decoder.self_attn.q_a_layernorm.weight"] = (
            f"{prefix}.self_attn.q_a_layernorm.weight"
        )
        mappings["mtp.decoder.self_attn.kv_a_layernorm.weight"] = (
            f"{prefix}.self_attn.kv_a_layernorm.weight"
        )
        mappings["mtp.decoder.input_layernorm.weight"] = (
            f"{prefix}.input_layernorm.weight"
        )
        # D5: the decoder's MoE-input norm (distinct from shared_head.norm).
        mappings["mtp.decoder.post_attention_layernorm.weight"] = (
            f"{prefix}.post_attention_layernorm.weight"
        )

        # MoE router (replicated, NOT quantised).
        mappings["mtp.decoder.mlp.gate_weight"] = f"{prefix}.mlp.gate.weight"
        mappings["mtp.decoder.mlp.e_score_correction_bias"] = (
            f"{prefix}.mlp.gate.e_score_correction_bias"
        )
        # Routed experts (local EP slice) — weight + scale_inv per expert.
        expert_range = range(local_expert_start, local_expert_start + num_local_experts)
        mappings["mtp.decoder.mlp.gate_proj_weights"] = (
            [f"{prefix}.mlp.experts.{j}.gate_proj.weight" for j in expert_range]
            + [f"{prefix}.mlp.experts.{j}.gate_proj.weight_scale_inv" for j in expert_range]
        )
        mappings["mtp.decoder.mlp.up_proj_weights"] = (
            [f"{prefix}.mlp.experts.{j}.up_proj.weight" for j in expert_range]
            + [f"{prefix}.mlp.experts.{j}.up_proj.weight_scale_inv" for j in expert_range]
        )
        mappings["mtp.decoder.mlp.down_proj_weights"] = (
            [f"{prefix}.mlp.experts.{j}.down_proj.weight" for j in expert_range]
            + [f"{prefix}.mlp.experts.{j}.down_proj.weight_scale_inv" for j in expert_range]
        )
        # Shared expert (FP8 ROW).
        mappings["mtp.decoder.mlp.shared_expert.gate_proj_weight"] = [
            f"{prefix}.mlp.shared_experts.gate_proj.weight",
            f"{prefix}.mlp.shared_experts.gate_proj.weight_scale_inv",
        ]
        mappings["mtp.decoder.mlp.shared_expert.up_proj_weight"] = [
            f"{prefix}.mlp.shared_experts.up_proj.weight",
            f"{prefix}.mlp.shared_experts.up_proj.weight_scale_inv",
        ]
        mappings["mtp.decoder.mlp.shared_expert.down_proj_weight"] = [
            f"{prefix}.mlp.shared_experts.down_proj.weight",
            f"{prefix}.mlp.shared_experts.down_proj.weight_scale_inv",
        ]

        # MTP-only tensors (D1/D2/D4/D5): eh_proj (BF16, replicated), enorm, hnorm,
        # shared_head.norm. eh_proj ckpt is [6144,12288] == nn.Linear(12288,6144).weight.
        mappings["mtp.eh_proj.weight"] = f"{prefix}.eh_proj.weight"
        mappings["mtp.enorm.weight"] = f"{prefix}.enorm.weight"
        mappings["mtp.hnorm.weight"] = f"{prefix}.hnorm.weight"
        mappings["mtp.shared_head_norm.weight"] = f"{prefix}.shared_head.norm.weight"

        # Base shared embedding + untied lm_head (D6).
        mappings["embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["lm_head.weight"] = "lm_head.weight"
        # NOTE: self_attn.indexer.* keys are intentionally NOT mapped (skipped, R-idx).

        # ---- attach FP8 loaders (mirrors model_fp8_fwd_dequant.load_weights) ----
        _scale_store: dict = {}

        set_weight_loader(
            attn.q_a_proj_weight,
            fp8_dequant_row_parallel_weight_loader(hidden_per_sp, tp_size),
        )
        set_weight_loader(
            attn.q_b_proj_weight,
            fp8_dequant_weight_loader(0, num_heads_per_rank * q_head_dim, tp_size),
        )
        set_weight_loader(
            attn.kv_a_proj_weight,
            fp8_dequant_row_parallel_weight_loader(hidden_per_sp, tp_size),
        )
        set_weight_loader(
            attn.kv_b_proj_weight,
            fp8_dequant_weight_loader(0, kv_b_out_per_rank, tp_size),
        )
        set_weight_loader(
            attn.o_proj_weight,
            fp8_dequant_weight_loader(1, o_proj_in_per_rank, tp_size),
        )

        def _make_mlp_loader(proj_loader, scale_key):
            def transform(slices, rank):
                fp8_w, dequant_scale = proj_loader.transform(slices, rank)
                _scale_store[scale_key] = dequant_scale
                return fp8_w

            return SafetensorsWeightLoader(transform=transform)

        def _make_moe_loader(proj_loader, scale_key):
            def transform(slices, rank):
                fp8_ws, scales = proj_loader.transform(slices, rank)
                _scale_store[scale_key] = scales
                return fp8_ws

            return SafetensorsWeightLoader(transform=transform)

        set_weight_loader(
            moe.gate_proj_weights,
            _make_moe_loader(
                _fp8_row_moe_weight_loader(num_local_experts, 0, moe_intermediate_per_rank, ep_tp_size),
                f"{prefix}.mlp.gate_proj_scales",
            ),
        )
        set_weight_loader(
            moe.up_proj_weights,
            _make_moe_loader(
                _fp8_row_moe_weight_loader(num_local_experts, 0, moe_intermediate_per_rank, ep_tp_size),
                f"{prefix}.mlp.up_proj_scales",
            ),
        )
        set_weight_loader(
            moe.down_proj_weights,
            _make_moe_loader(
                _fp8_row_moe_weight_loader(num_local_experts, 1, moe_intermediate_per_rank, ep_tp_size),
                f"{prefix}.mlp.down_proj_scales",
            ),
        )

        shared = moe.shared_expert
        set_weight_loader(
            shared.gate_proj_weight,
            _make_mlp_loader(
                _fp8_row_weight_loader(0, shared_intermediate_per_rank, tp_size),
                f"{prefix}.mlp.shared_expert.gate_w_scale",
            ),
        )
        set_weight_loader(
            shared.up_proj_weight,
            _make_mlp_loader(
                _fp8_row_weight_loader(0, shared_intermediate_per_rank, tp_size),
                f"{prefix}.mlp.shared_expert.up_w_scale",
            ),
        )
        set_weight_loader(
            shared.down_proj_weight,
            _make_mlp_loader(
                _fp8_row_weight_loader(1, shared_intermediate_per_rank, tp_size),
                f"{prefix}.mlp.shared_expert.down_w_scale",
            ),
        )

        # eh_proj / enorm / hnorm / shared_head_norm are replicated (no shard, no scale):
        # the default SafetensorsWeightLoader loads the full tensor as-is.
        set_weight_loader(self.mtp.eh_proj.weight, SafetensorsWeightLoader())
        set_weight_loader(self.mtp.enorm.weight, SafetensorsWeightLoader())
        set_weight_loader(self.mtp.hnorm.weight, SafetensorsWeightLoader())
        set_weight_loader(self.mtp.shared_head_norm.weight, SafetensorsWeightLoader())

        # ---- load ----
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        load_result = checkpoint.load_sharded(
            tp_rank, tp_size, self, mappings, device, strict=False
        )
        rank_sharded = load_result.state_dict

        # Inject the per-row dequant scales captured by the loaders. Scale keys
        # follow the FP8 module attribute paths under mtp.decoder.mlp.*.
        for scale_key, scale_val in _scale_store.items():
            # rewrite "model.layers.78.mlp.X" -> "mtp.decoder.mlp.X" to match param names
            attr_key = scale_key.replace(f"{prefix}.mlp.", "mtp.decoder.mlp.")
            rank_sharded[attr_key] = scale_val

        self.load_state_dict(rank_sharded, strict=False, assign=True)

        real_missing = [k for k in (load_result.missing_keys or []) if k not in rank_sharded]
        if real_missing:
            logger.error("MTP MISSING weights (%d): %s", len(real_missing), real_missing[:10])
        if load_result.unexpected_keys:
            logger.error("MTP UNEXPECTED weights (%d): %s", len(load_result.unexpected_keys),
                         load_result.unexpected_keys[:10])
        logger.info(
            "MTP FP8 ROW weight loading complete: %d params loaded, %d scales injected",
            len(rank_sharded), len(_scale_store),
        )

        # Prepare moe_tkg decode weights (stack gate+up, free originals) — mirrors
        # model_fp8_fwd_dequant.load_weights:761-764. The stacked _tkg_gate_up_w +
        # scales are ALSO what the einsum decode path dequantizes, so this prep is
        # needed regardless of which decode path runs.
        if hasattr(moe, "_prepare_tkg_weights"):
            moe._prepare_tkg_weights()
        # Route the DRAFT's decode MoE through the kernel-free einsum path. The
        # moe_tkg NKI kernel's internal indirect DMA goes out-of-bound at the
        # γ=1 verify token shape (T=bs*(1+γ)); the einsum path is per-token and
        # shape-agnostic. Only the draft's layer-78 MoE is flagged; the base
        # target model keeps the fast kernel path.
        moe.use_einsum_decode = True
        logger.info("MTP draft decode MoE → einsum path (moe_tkg over-read workaround)")

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        start_layer_idx: int = MTP_LAYER_IDX,
        neuron_config: NeuronConfig | None = None,
    ):
        config = Glm52Config.from_configs(hf_config, neuron_config)
        # Swap the base MLP classes for their FP8 ROW variants during construction,
        # so Glm52DecoderLayer@78 builds an FP8 MoE + FP8 shared expert. Mirrors
        # model_fp8_fwd_dequant.from_configs:766-782.
        import vllm_neuron.model.glm_5_2.model as model_mod

        orig_dense = model_mod.Glm52DenseMLP
        orig_shared = model_mod.Glm52SharedExpertMLP
        orig_moe = model_mod.Glm52MoE
        model_mod.Glm52DenseMLP = Glm52DenseMLPFP8Fwd
        model_mod.Glm52SharedExpertMLP = Glm52SharedExpertMLPFP8Fwd
        model_mod.Glm52MoE = Glm52MoELayerFP8Fwd
        try:
            model = cls(config, start_layer_idx=start_layer_idx)
        finally:
            model_mod.Glm52DenseMLP = orig_dense
            model_mod.Glm52SharedExpertMLP = orig_shared
            model_mod.Glm52MoE = orig_moe
        return model
