# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.2 FP8 Implementation
====================================

Loads weights from an FP8 block-quantized checkpoint, dequantizes to BF16
at load time, and reuses the BF16 model's forward path unchanged. This
enables using FP8 checkpoints without modifying the compiled graph.

Block-wise dequantization: bf16 = fp8_weight * downscale * (1/scale_inv)
where blocks are 128x128 and scale_inv is [ceil(out/128), ceil(in/128)].
"""

import logging

import torch
from torch import nn

from vllm_neuron.parallel.neuron_parallel_state import (
    get_neuron_ep_rank,
)

from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig

from .config import Glm52Config
from .model import Glm52ForCausalLM as Glm52ForCausalLMBF16
from .weight_loaders_fp8 import (
    fp8_dequant_weight_loader,
    fp8_dequant_row_parallel_weight_loader,
    fp8_dequant_moe_expert_weight_loader,
)

logger = logging.getLogger(__name__)


class Glm52ForCausalLM(Glm52ForCausalLMBF16):
    """GLM-5.2 with FP8 checkpoint loading.

    Inherits the BF16 model's architecture and forward pass. Overrides
    load_weights() to dequantize FP8 block-quantized checkpoints to BF16
    at load time.
    """

    def load_weights(self, checkpoint_path, device, cache_dir=None):
        tp_rank = self.sp_group.rank_in_group
        tp_size = self.sp_world_size

        if self.ep_degree > 1:
            ep_rank = get_neuron_ep_rank()
            num_local_experts = self.config.n_routed_experts // self.ep_degree
            local_expert_start = ep_rank * num_local_experts
        else:
            num_local_experts = self.config.n_routed_experts
            local_expert_start = 0

        hidden = self.config.hidden_size
        hidden_per_sp = hidden // tp_size
        q_lora_rank = self.config.q_lora_rank
        kv_lora_rank = self.config.kv_lora_rank
        qk_rope_head_dim = self.config.qk_rope_head_dim
        qk_nope_head_dim = self.config.qk_nope_head_dim
        v_head_dim = self.config.v_head_dim
        num_heads_per_rank = self.config.num_attention_heads // tp_size
        q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_b_out_per_rank = num_heads_per_rank * (qk_nope_head_dim + v_head_dim)
        o_proj_in_per_rank = num_heads_per_rank * v_head_dim
        intermediate = self.config.intermediate_size
        intermediate_per_rank = intermediate // tp_size

        # EP-internal TP for MoE
        ep_tp_size = tp_size // self.ep_degree if self.ep_degree > 1 else tp_size
        moe_intermediate = self.config.moe_intermediate_size
        moe_intermediate_per_rank = moe_intermediate // ep_tp_size

        mappings = dict()

        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"
            is_dense = layer_id < self.config.first_k_dense_replace

            # Attention: FP8 weight + scale_inv paired as lists
            mappings[f"{prefix}.self_attn.q_a_proj_weight"] = [
                f"{prefix}.self_attn.q_a_proj.weight",
                f"{prefix}.self_attn.q_a_proj.weight_scale_inv",
            ]
            mappings[f"{prefix}.self_attn.q_b_proj_weight"] = [
                f"{prefix}.self_attn.q_b_proj.weight",
                f"{prefix}.self_attn.q_b_proj.weight_scale_inv",
            ]
            mappings[f"{prefix}.self_attn.kv_a_proj_weight"] = [
                f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
                f"{prefix}.self_attn.kv_a_proj_with_mqa.weight_scale_inv",
            ]
            mappings[f"{prefix}.self_attn.kv_b_proj_weight"] = [
                f"{prefix}.self_attn.kv_b_proj.weight",
                f"{prefix}.self_attn.kv_b_proj.weight_scale_inv",
            ]
            mappings[f"{prefix}.self_attn.o_proj_weight"] = [
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.o_proj.weight_scale_inv",
            ]

            # Layer norms (BF16 passthrough, no scale)
            mappings[f"{prefix}.self_attn.q_a_layernorm.weight"] = (
                f"{prefix}.self_attn.q_a_layernorm.weight"
            )
            mappings[f"{prefix}.self_attn.kv_a_layernorm.weight"] = (
                f"{prefix}.self_attn.kv_a_layernorm.weight"
            )
            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )
            mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )

            if is_dense:
                # Dense MLP FP8
                mappings[f"{prefix}.mlp.gate_proj_weight"] = [
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.gate_proj.weight_scale_inv",
                ]
                mappings[f"{prefix}.mlp.up_proj_weight"] = [
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.up_proj.weight_scale_inv",
                ]
                mappings[f"{prefix}.mlp.down_proj_weight"] = [
                    f"{prefix}.mlp.down_proj.weight",
                    f"{prefix}.mlp.down_proj.weight_scale_inv",
                ]
            else:
                # MoE Router (BF16, no scale)
                mappings[f"{prefix}.mlp.gate_weight"] = (
                    f"{prefix}.mlp.gate.weight"
                )
                mappings[f"{prefix}.mlp.e_score_correction_bias"] = (
                    f"{prefix}.mlp.gate.e_score_correction_bias"
                )

                # MoE Routed experts: [weight_0..N, scale_0..N] paired
                expert_range = range(local_expert_start, local_expert_start + num_local_experts)
                mappings[f"{prefix}.mlp.gate_proj_weights"] = (
                    [f"{prefix}.mlp.experts.{j}.gate_proj.weight" for j in expert_range]
                    + [f"{prefix}.mlp.experts.{j}.gate_proj.weight_scale_inv" for j in expert_range]
                )
                mappings[f"{prefix}.mlp.up_proj_weights"] = (
                    [f"{prefix}.mlp.experts.{j}.up_proj.weight" for j in expert_range]
                    + [f"{prefix}.mlp.experts.{j}.up_proj.weight_scale_inv" for j in expert_range]
                )
                mappings[f"{prefix}.mlp.down_proj_weights"] = (
                    [f"{prefix}.mlp.experts.{j}.down_proj.weight" for j in expert_range]
                    + [f"{prefix}.mlp.experts.{j}.down_proj.weight_scale_inv" for j in expert_range]
                )

                # Shared expert FP8
                mappings[f"{prefix}.mlp.shared_expert.gate_proj_weight"] = [
                    f"{prefix}.mlp.shared_experts.gate_proj.weight",
                    f"{prefix}.mlp.shared_experts.gate_proj.weight_scale_inv",
                ]
                mappings[f"{prefix}.mlp.shared_expert.up_proj_weight"] = [
                    f"{prefix}.mlp.shared_experts.up_proj.weight",
                    f"{prefix}.mlp.shared_experts.up_proj.weight_scale_inv",
                ]
                mappings[f"{prefix}.mlp.shared_expert.down_proj_weight"] = [
                    f"{prefix}.mlp.shared_experts.down_proj.weight",
                    f"{prefix}.mlp.shared_experts.down_proj.weight_scale_inv",
                ]

        # Backbone (BF16 passthrough)
        mappings["model.norm.weight"] = "model.norm.weight"
        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["lm_head.weight"] = "lm_head.weight"

        # Override weight loaders: replace BF16 loaders with FP8 dequant loaders
        for layer_id in range(len(self.model.layers)):
            layer = self.model.layers[layer_id]
            attn = layer.self_attn
            is_dense = layer_id < self.config.first_k_dense_replace

            # Attention weight loaders
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

            if is_dense:
                mlp = layer.mlp
                set_weight_loader(
                    mlp.gate_proj_weight,
                    fp8_dequant_weight_loader(0, intermediate_per_rank, tp_size),
                )
                set_weight_loader(
                    mlp.up_proj_weight,
                    fp8_dequant_weight_loader(0, intermediate_per_rank, tp_size),
                )
                set_weight_loader(
                    mlp.down_proj_weight,
                    fp8_dequant_weight_loader(1, intermediate_per_rank, tp_size),
                )
            else:
                moe = layer.mlp
                set_weight_loader(
                    moe.gate_proj_weights,
                    fp8_dequant_moe_expert_weight_loader(
                        num_local_experts, 0, moe_intermediate_per_rank, ep_tp_size,
                    ),
                )
                set_weight_loader(
                    moe.up_proj_weights,
                    fp8_dequant_moe_expert_weight_loader(
                        num_local_experts, 0, moe_intermediate_per_rank, ep_tp_size,
                    ),
                )
                set_weight_loader(
                    moe.down_proj_weights,
                    fp8_dequant_moe_expert_weight_loader(
                        num_local_experts, 1, moe_intermediate_per_rank, ep_tp_size,
                    ),
                )
                # Shared expert uses full TP group (same as BF16 model)
                shared = moe.shared_expert
                set_weight_loader(
                    shared.gate_proj_weight,
                    fp8_dequant_weight_loader(0, shared.intermediate_size_per_rank, tp_size),
                )
                set_weight_loader(
                    shared.up_proj_weight,
                    fp8_dequant_weight_loader(0, shared.intermediate_size_per_rank, tp_size),
                )
                set_weight_loader(
                    shared.down_proj_weight,
                    fp8_dequant_weight_loader(1, shared.intermediate_size_per_rank, tp_size),
                )

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)

        load_result = checkpoint.load_sharded(
            tp_rank, tp_size, self, mappings, device, strict=False,
        )
        rank_sharded = load_result.state_dict

        self.load_state_dict(rank_sharded, strict=False, assign=True)
        if load_result.missing_keys:
            logger.error("MISSING weights (%d): %s", len(load_result.missing_keys), load_result.missing_keys[:20])
        if load_result.unexpected_keys:
            logger.error("UNEXPECTED weights (%d): %s", len(load_result.unexpected_keys), load_result.unexpected_keys[:20])
        loaded_keys = set(rank_sharded.keys())
        all_params = set(n for n, _ in self.named_parameters())
        not_loaded = all_params - loaded_keys
        if not_loaded:
            logger.error("PARAMS NOT LOADED (%d): %s", len(not_loaded), sorted(not_loaded)[:20])
        logger.info("FP8 weight loading: %d params loaded, %d total params", len(loaded_keys), len(all_params))

        # Pre-fuse decode weights. This mode dequantizes to BF16 and reuses the
        # BF16 forward path, whose decode MoE reads the fused ``_decode_gate_up_w``
        # buffer. ``load_weights`` fully overrides the BF16 implementation, so the
        # fuse step has to be repeated here — without it every MoE layer raises
        # ``AttributeError: 'Glm52MoE' object has no attribute '_decode_gate_up_w'``
        # while the decode graph is being traced.
        for layer in self.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "_fuse_decode_weights"):
                layer.mlp._fuse_decode_weights()
        logger.info("Pre-fused decode MoE weights (BF16 einsum path)")

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Glm52Config.from_configs(hf_config, neuron_config)
        return cls(config)
