# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.2 FP8 Native — Option 2: Re-quantize to per-tensor FP8 at load time.

At load time: dequant block-wise FP8 → BF16, compute per-tensor scale,
re-quantize to TRN2 FP8 (e4m3, max 240). Store FP8 + scalar scales on HBM.
Forward uses NF.mlp / NF.moe_cte with quantization_type=STATIC, so kernels
handle dequant internally on tensor engine.

This halves weight HBM vs BF16, with slight precision loss from collapsing
block scales to per-tensor.
"""

import logging
import math

import torch
import torch.nn as nn
from transformers import PretrainedConfig

import neuronxcc.nki.language as nl
import vllm_neuron.functional as NF
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    QuantizationType,
)

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.parallel.neuron_parallel_state import (
    get_neuron_ep_degree,
    get_neuron_ep_rank,
    get_neuron_ep_tp_group,
)
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

from .config import Glm52Config
from .model import (
    Glm52ForCausalLM as Glm52ForCausalLMBF16,
    Glm52MoE as BF16MoELayer,
    Glm52DenseMLP as BF16DenseMLP,
    Glm52SharedExpertMLP as BF16SharedExpertMLP,
    get_tp_group,
)

logger = logging.getLogger(__name__)

_FP8_DTYPE = torch.float8_e4m3fn
_BLOCK_SIZE = 128
_PMAX = 128  # kernel scale broadcast dimension

_FP8_E4M3_MAX = 240.0
_FP8_E4M3FN_MAX = 448.0
_FP8_WEIGHT_DOWNSCALE = _FP8_E4M3_MAX / _FP8_E4M3FN_MAX


def _dequant_block_fp8(weight_fp8, scale_inv):
    """Dequant block-wise FP8 → f32. scale_inv is the dequant multiplier."""
    out_dim, in_dim = weight_fp8.shape
    w_f32 = weight_fp8.float()
    scale_expanded = (
        scale_inv.float()
        .repeat_interleave(_BLOCK_SIZE, dim=0)
        .repeat_interleave(_BLOCK_SIZE, dim=1)
    )[:out_dim, :in_dim]
    return w_f32 * scale_expanded


def _requantize_to_trn2_fp8(w_bf16):
    """Re-quantize BF16 weight to TRN2 FP8 (max 240). Returns (fp8_weight, scalar_scale)."""
    w_f32 = w_bf16.float()
    amax = w_f32.abs().max().clamp(min=1e-12)
    scale = _FP8_E4M3_MAX / amax  # quantization scale
    w_scaled = (w_f32 * scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    w_fp8 = w_scaled.to(_FP8_DTYPE)
    dequant_scale = amax / _FP8_E4M3_MAX  # = 1/scale
    return w_fp8, dequant_scale


def _fp8_native_weight_loader(shard_dim, shard_size, num_shards):
    """Load FP8 block-quantized weight, re-quantize to per-tensor FP8.
    Returns (fp8_weight_transposed, scalar_dequant_scale)."""

    def transform(slices, rank):
        tp_rank = rank % num_shards
        assert len(slices) == 2
        weight_fp8 = slices[0][:]
        scale_inv = slices[1][:]

        w_bf16 = _dequant_block_fp8(weight_fp8, scale_inv).to(torch.bfloat16)

        # Shard
        start_idx = tp_rank * shard_size
        end_idx = start_idx + shard_size
        sl = [slice(None)] * 2
        sl[shard_dim] = slice(start_idx, end_idx)
        w_shard = w_bf16[tuple(sl)]

        # Transpose (storage layout [in, out] for NF kernels)
        w_shard_t = w_shard.T.contiguous()

        # Re-quantize to per-tensor TRN2 FP8
        w_fp8_new, dequant_scale = _requantize_to_trn2_fp8(w_shard_t)
        return w_fp8_new, dequant_scale

    return SafetensorsWeightLoader(transform=transform)


def _fp8_native_moe_weight_loader(num_experts, shard_dim, shard_size, num_shards):
    """Load FP8 block-quantized MoE weights, re-quantize per expert."""

    def transform(slices, rank):
        tp_rank = rank % num_shards
        assert len(slices) == 2 * num_experts
        weight_slices = slices[:num_experts]
        scale_slices = slices[num_experts:]

        fp8_experts = []
        scales = []
        for w_slice, s_slice in zip(weight_slices, scale_slices):
            w_fp8_raw = w_slice[:]
            s_inv = s_slice[:]

            w_bf16 = _dequant_block_fp8(w_fp8_raw, s_inv).to(torch.bfloat16)

            start_idx = tp_rank * shard_size
            end_idx = start_idx + shard_size
            sl = [slice(None)] * 2
            sl[shard_dim] = slice(start_idx, end_idx)
            w_shard = w_bf16[tuple(sl)]

            w_shard_t = w_shard.T.contiguous()
            w_fp8_new, dequant_scale = _requantize_to_trn2_fp8(w_shard_t)
            fp8_experts.append(w_fp8_new)
            scales.append(dequant_scale)

        return torch.stack(fp8_experts, dim=0), torch.stack(scales, dim=0)

    return SafetensorsWeightLoader(transform=transform)


# =============================================================================
# FP8 Native Dense MLP
# =============================================================================


class Glm52DenseMLPFP8Native(nn.Module):
    def __init__(self, config: Glm52Config):
        super().__init__()

        self.tp_group = get_tp_group()
        self.sp_group = self.tp_group
        self.world_size = self.tp_group.world_size

        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size

        # FP8 weights
        self.gate_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.intermediate_size_per_rank, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.intermediate_size_per_rank, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_size_per_rank, config.hidden_size, dtype=_FP8_DTYPE),
            requires_grad=False,
        )

        # Scalar scales broadcast to [PMAX, 1] for kernel
        self.gate_w_scale = nn.Parameter(
            torch.ones(_PMAX, 1, dtype=torch.float32), requires_grad=False
        )
        self.up_w_scale = nn.Parameter(
            torch.ones(_PMAX, 1, dtype=torch.float32), requires_grad=False
        )
        self.down_w_scale = nn.Parameter(
            torch.ones(_PMAX, 1, dtype=torch.float32), requires_grad=False
        )

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill and self.sp_group.world_size > 1:
            hidden_states = self.sp_group.all_gather(hidden_states, dim=0)

        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
            quantization_type=QuantizationType.STATIC,
            gate_w_scale=self.gate_w_scale,
            up_w_scale=self.up_w_scale,
            down_w_scale=self.down_w_scale,
        )

        if self.sp_group.world_size > 1:
            if is_prefill:
                output = self.sp_group.reduce_scatter(output, dim=0)
            else:
                self.sp_group.all_reduce(output)

        return output


# =============================================================================
# FP8 Native Shared Expert MLP
# =============================================================================


class Glm52SharedExpertMLPFP8Native(nn.Module):
    def __init__(self, config: Glm52Config):
        super().__init__()

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
        self.intermediate_size_per_rank = shared_intermediate // self.world_size

        self.gate_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.intermediate_size_per_rank, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.intermediate_size_per_rank, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_size_per_rank, config.hidden_size, dtype=_FP8_DTYPE),
            requires_grad=False,
        )

        self.gate_w_scale = nn.Parameter(
            torch.ones(_PMAX, 1, dtype=torch.float32), requires_grad=False
        )
        self.up_w_scale = nn.Parameter(
            torch.ones(_PMAX, 1, dtype=torch.float32), requires_grad=False
        )
        self.down_w_scale = nn.Parameter(
            torch.ones(_PMAX, 1, dtype=torch.float32), requires_grad=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
            quantization_type=QuantizationType.STATIC,
            gate_w_scale=self.gate_w_scale,
            up_w_scale=self.up_w_scale,
            down_w_scale=self.down_w_scale,
        )


# =============================================================================
# FP8 Native MoE Layer
# =============================================================================


class Glm52MoELayerFP8Native(BF16MoELayer):
    """MoE with FP8 expert weights + per-expert scalar scales.

    Inherits routing from BF16 MoE. Overrides weight storage and forward
    to use FP8 with native kernel support.
    """

    def __init__(self, config: Glm52Config):
        nn.Module.__init__(self)

        self.ep_degree = get_neuron_ep_degree()
        self.ep_enabled = self.ep_degree > 1

        if self.ep_enabled:
            self.ep_rank = get_neuron_ep_rank()
            self.ep_tp_group = get_neuron_ep_tp_group()
            self.moe_group = get_tp_group()
        else:
            self.ep_rank = 0
            self.ep_tp_group = get_tp_group()
            self.moe_group = get_tp_group()

        self.tp_degree = self.ep_tp_group.world_size
        self.rank = self.ep_tp_group.rank_in_group

        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.num_local_experts = config.n_routed_experts // self.ep_degree
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.dtype = config.torch_dtype
        self.routed_scaling_factor = config.routed_scaling_factor

        self.moe_intermediate_size = config.moe_intermediate_size
        self.intermediate_size_per_rank = config.moe_intermediate_size // self.tp_degree

        # Router (BF16)
        self.gate_weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=self.dtype)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32)
        )

        # FP8 expert weights [E_local, in, out] after transpose
        self.gate_proj_weights = nn.Parameter(
            torch.empty(
                self.num_local_experts, self.hidden_size, self.intermediate_size_per_rank,
                dtype=_FP8_DTYPE,
            ),
            requires_grad=False,
        )
        self.up_proj_weights = nn.Parameter(
            torch.empty(
                self.num_local_experts, self.hidden_size, self.intermediate_size_per_rank,
                dtype=_FP8_DTYPE,
            ),
            requires_grad=False,
        )
        self.down_proj_weights = nn.Parameter(
            torch.empty(
                self.num_local_experts, self.intermediate_size_per_rank, self.hidden_size,
                dtype=_FP8_DTYPE,
            ),
            requires_grad=False,
        )

        # Per-expert scalar dequant scales: [E_local, 1, I_TP] / [E_local, 1, H]
        I_TP = self.intermediate_size_per_rank
        self.gate_up_proj_scale = nn.Parameter(
            torch.ones(self.num_local_experts, 1, 2 * I_TP, dtype=torch.float32),
            requires_grad=False,
        )
        self.down_proj_scale = nn.Parameter(
            torch.ones(self.num_local_experts, 1, self.hidden_size, dtype=torch.float32),
            requires_grad=False,
        )

        # Shared expert (FP8 native)
        self.shared_expert = Glm52SharedExpertMLPFP8Native(config)

    def _fuse_decode_weights(self):
        """Pre-fuse dequanted BF16 weights for batched einsum decode path."""
        I_TP = self.intermediate_size_per_rank
        gate_scales = self.gate_up_proj_scale.data.cpu()[:, 0, :I_TP].float()
        up_scales = self.gate_up_proj_scale.data.cpu()[:, 0, I_TP:].float()
        down_scales = self.down_proj_scale.data.cpu()[:, 0, :].float()

        gate_f = self.gate_proj_weights.data.cpu().to(dtype=torch.float32)
        up_f = self.up_proj_weights.data.cpu().to(dtype=torch.float32)
        down_f = self.down_proj_weights.data.cpu().to(dtype=torch.float32)

        gate_w = (gate_f * gate_scales.unsqueeze(1)).to(torch.bfloat16)
        up_w = (up_f * up_scales.unsqueeze(1)).to(torch.bfloat16)
        self._decode_gate_up_w = nn.Parameter(
            torch.cat([gate_w, up_w], dim=-1).contiguous(), requires_grad=False
        )
        self._decode_down_w = nn.Parameter(
            (down_f * down_scales.unsqueeze(1)).to(torch.bfloat16).contiguous(),
            requires_grad=False,
        )

    def _forward_decode(self, hidden_states_2d: torch.Tensor) -> torch.Tensor:
        """Decode: batched einsum over all experts (contiguous DMA)."""
        import torch.nn.functional as F

        T = hidden_states_2d.shape[0]
        topk_weights, topk_indices = self._compute_routing(hidden_states_2d)

        local_affinities = torch.zeros(
            T, self.num_local_experts,
            dtype=topk_weights.dtype, device=hidden_states_2d.device
        )
        for e_local in range(self.num_local_experts):
            e_global = self.local_expert_start + e_local
            mask = (topk_indices == e_global)
            local_affinities[:, e_local] = (topk_weights * mask.to(topk_weights.dtype)).sum(dim=-1)

        gate_up = torch.einsum("th,ehi->eti", hidden_states_2d, self._decode_gate_up_w)
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = F.silu(gate) * up

        down = torch.einsum("eti,eih->eth", intermediate, self._decode_down_w)
        output = torch.einsum("eth,te->th", down, local_affinities)

        return output

    def _forward_prefill(self, hidden_states_2d: torch.Tensor) -> torch.Tensor:
        """Prefill: use moe_cte kernel with FP8 scales."""
        from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation

        num_tokens = hidden_states_2d.shape[0]
        topk_weights, topk_indices = self._compute_routing(hidden_states_2d)

        full_affinities = torch.zeros(
            num_tokens, self.num_experts, dtype=torch.float32, device=hidden_states_2d.device
        )
        full_affinities.scatter_(1, topk_indices, topk_weights.to(torch.float32))

        local_expert_indices = torch.arange(
            self.local_expert_start,
            self.local_expert_start + self.num_local_experts,
            device=hidden_states_2d.device,
            dtype=torch.int64,
        )
        local_affinities = NF.get_local_expert_affinities(full_affinities, local_expert_indices)

        block_size = 128
        affinities_masked, token_pos_to_id, block_to_expert, conditions = NF.build_blockwise_mapping(
            expert_affinities=local_affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.num_experts_per_tok,
            block_size=block_size,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
        )

        # Dequant FP8 → BF16 for CTE (transient, not stored)
        I_TP = self.intermediate_size_per_rank
        gate_scales = self.gate_up_proj_scale[:, 0, :I_TP]  # [E, I_TP]
        up_scales = self.gate_up_proj_scale[:, 0, I_TP:]  # [E, I_TP]
        down_scales = self.down_proj_scale[:, 0, :]  # [E, H]

        gate_bf16 = self.gate_proj_weights.to(torch.bfloat16) * gate_scales.unsqueeze(1)
        up_bf16 = self.up_proj_weights.to(torch.bfloat16) * up_scales.unsqueeze(1)
        down_bf16 = self.down_proj_weights.to(torch.bfloat16) * down_scales.unsqueeze(1)

        gate_up_weight = torch.stack([gate_bf16, up_bf16], dim=2)

        output = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=hidden_states_2d,
            expert_affinities_masked=affinities_masked,
            gate_up_proj_weight=gate_up_weight,
            down_proj_weight=down_bf16,
            activation_function=ActFnType.SiLU,
            block_size=block_size,
            token_position_to_id=token_pos_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        )

        return output


# =============================================================================
# Top-level model
# =============================================================================


class Glm52ForCausalLM(Glm52ForCausalLMBF16):
    """GLM-5.2 with native FP8 (re-quantized per-tensor at load time)."""

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

        ep_tp_size = tp_size // self.ep_degree if self.ep_degree > 1 else tp_size
        moe_intermediate = self.config.moe_intermediate_size
        moe_intermediate_per_rank = moe_intermediate // ep_tp_size

        shared_intermediate = self.config.moe_intermediate_size * self.config.n_shared_experts
        shared_intermediate_per_rank = shared_intermediate // tp_size

        # Build checkpoint mappings (same as model_fp8.py)
        mappings = dict()

        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"
            is_dense = layer_id < self.config.first_k_dense_replace

            # Attention weights (dequant to BF16 — MLA has no kernel FP8 support)
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

            # Layer norms (BF16)
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
                # Dense MLP FP8 → native FP8 (weights + scales separate)
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
                # MoE Router (BF16)
                mappings[f"{prefix}.mlp.gate_weight"] = f"{prefix}.mlp.gate.weight"
                mappings[f"{prefix}.mlp.e_score_correction_bias"] = (
                    f"{prefix}.mlp.gate.e_score_correction_bias"
                )

                # Routed experts FP8
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

        # Backbone (BF16)
        mappings["model.norm.weight"] = "model.norm.weight"
        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["lm_head.weight"] = "lm_head.weight"

        # --- Setup weight loaders ---
        # Scales are accumulated here during transform, then injected into state dict
        _scale_store = {}

        from .weight_loaders_fp8 import (
            fp8_dequant_weight_loader,
            fp8_dequant_row_parallel_weight_loader,
        )

        for layer_id in range(len(self.model.layers)):
            layer = self.model.layers[layer_id]
            attn = layer.self_attn
            is_dense = layer_id < self.config.first_k_dense_replace
            prefix = f"model.layers.{layer_id}"

            # Attention: still dequant to BF16 (no MLA kernel FP8 support)
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
                # Dense MLP: re-quantize to FP8 native
                gate_loader = _fp8_native_weight_loader(0, intermediate_per_rank, tp_size)
                up_loader = _fp8_native_weight_loader(0, intermediate_per_rank, tp_size)
                down_loader = _fp8_native_weight_loader(1, intermediate_per_rank, tp_size)

                def _make_mlp_loader(proj_loader, scale_key):
                    """Wrapper that splits (fp8_weight, scale) tuple and stores scale."""
                    def transform(slices, rank):
                        result = proj_loader.transform(slices, rank)
                        fp8_w, dequant_s = result
                        # Build [PMAX, 1] scale tensor
                        scale_tensor = torch.full((_PMAX, 1), dequant_s.item(), dtype=torch.float32)
                        _scale_store[scale_key] = scale_tensor
                        return fp8_w
                    return SafetensorsWeightLoader(transform=transform)

                set_weight_loader(mlp.gate_proj_weight, _make_mlp_loader(gate_loader, f"{prefix}.mlp.gate_w_scale"))
                set_weight_loader(mlp.up_proj_weight, _make_mlp_loader(up_loader, f"{prefix}.mlp.up_w_scale"))
                set_weight_loader(mlp.down_proj_weight, _make_mlp_loader(down_loader, f"{prefix}.mlp.down_w_scale"))
            else:
                moe = layer.mlp
                # MoE experts: re-quantize per expert
                gate_moe_loader = _fp8_native_moe_weight_loader(
                    num_local_experts, 0, moe_intermediate_per_rank, ep_tp_size
                )
                up_moe_loader = _fp8_native_moe_weight_loader(
                    num_local_experts, 0, moe_intermediate_per_rank, ep_tp_size
                )
                down_moe_loader = _fp8_native_moe_weight_loader(
                    num_local_experts, 1, moe_intermediate_per_rank, ep_tp_size
                )

                I_TP = moe_intermediate_per_rank
                H = self.config.hidden_size

                def _make_gate_moe_loader(proj_loader, scale_key, i_tp):
                    def transform(slices, rank):
                        result = proj_loader.transform(slices, rank)
                        fp8_ws, scales_vec = result
                        # Store gate half of gate_up_proj_scale [E, 1, 2*I_TP]
                        _scale_store[scale_key + "_gate"] = scales_vec
                        return fp8_ws
                    return SafetensorsWeightLoader(transform=transform)

                def _make_up_moe_loader(proj_loader, scale_key, i_tp):
                    def transform(slices, rank):
                        result = proj_loader.transform(slices, rank)
                        fp8_ws, scales_vec = result
                        # Store up half of gate_up_proj_scale
                        _scale_store[scale_key + "_up"] = scales_vec
                        return fp8_ws
                    return SafetensorsWeightLoader(transform=transform)

                def _make_down_moe_loader(proj_loader, scale_key, h):
                    def transform(slices, rank):
                        result = proj_loader.transform(slices, rank)
                        fp8_ws, scales_vec = result
                        # Store down_proj_scale [E, 1, H]
                        _scale_store[scale_key] = scales_vec
                        return fp8_ws
                    return SafetensorsWeightLoader(transform=transform)

                set_weight_loader(
                    moe.gate_proj_weights,
                    _make_gate_moe_loader(gate_moe_loader, f"{prefix}.mlp.gate_up_proj_scale", I_TP),
                )
                set_weight_loader(
                    moe.up_proj_weights,
                    _make_up_moe_loader(up_moe_loader, f"{prefix}.mlp.gate_up_proj_scale", I_TP),
                )
                set_weight_loader(
                    moe.down_proj_weights,
                    _make_down_moe_loader(down_moe_loader, f"{prefix}.mlp.down_proj_scale", H),
                )

                # Shared expert: re-quantize native FP8
                shared = moe.shared_expert
                sh_gate_loader = _fp8_native_weight_loader(0, shared_intermediate_per_rank, tp_size)
                sh_up_loader = _fp8_native_weight_loader(0, shared_intermediate_per_rank, tp_size)
                sh_down_loader = _fp8_native_weight_loader(1, shared_intermediate_per_rank, tp_size)

                set_weight_loader(shared.gate_proj_weight, _make_mlp_loader(sh_gate_loader, f"{prefix}.mlp.shared_expert.gate_w_scale"))
                set_weight_loader(shared.up_proj_weight, _make_mlp_loader(sh_up_loader, f"{prefix}.mlp.shared_expert.up_w_scale"))
                set_weight_loader(shared.down_proj_weight, _make_mlp_loader(sh_down_loader, f"{prefix}.mlp.shared_expert.down_w_scale"))

        # Load
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        load_result = checkpoint.load_sharded(
            tp_rank, tp_size, self, mappings, device, strict=False,
        )
        rank_sharded = load_result.state_dict

        # Inject accumulated scales into the state dict
        _injected = set()
        for scale_key, scale_val in _scale_store.items():
            if scale_key.endswith("_gate") or scale_key.endswith("_up"):
                base_key = scale_key.rsplit("_", 1)[0]
                if base_key in _injected:
                    continue
                gate_key = base_key + "_gate"
                up_key = base_key + "_up"
                if gate_key in _scale_store and up_key in _scale_store:
                    gate_s = _scale_store[gate_key]  # [E] scalar per expert
                    up_s = _scale_store[up_key]  # [E] scalar per expert
                    E = gate_s.shape[0]
                    i_tp = moe_intermediate_per_rank
                    combined = torch.zeros(E, 1, 2 * i_tp, dtype=torch.float32)
                    combined[:, 0, :i_tp] = gate_s.unsqueeze(1).expand(-1, i_tp)
                    combined[:, 0, i_tp:] = up_s.unsqueeze(1).expand(-1, i_tp)
                    rank_sharded[base_key] = combined
                    _injected.add(base_key)
            elif "down_proj_scale" in scale_key:
                scales_vec = scale_val  # [E] scalar per expert
                E = scales_vec.shape[0]
                h = self.config.hidden_size
                expanded = scales_vec.unsqueeze(1).unsqueeze(2).expand(E, 1, h).contiguous()
                rank_sharded[scale_key] = expanded
                _injected.add(scale_key)
            else:
                # Dense MLP scales — already [PMAX, 1]
                rank_sharded[scale_key] = scale_val
                _injected.add(scale_key)

        self.load_state_dict(rank_sharded, strict=False, assign=True)

        if load_result.missing_keys:
            # Filter out scales we injected manually
            real_missing = [k for k in load_result.missing_keys if k not in rank_sharded]
            if real_missing:
                logger.error("MISSING weights (%d): %s", len(real_missing), real_missing[:10])
        if load_result.unexpected_keys:
            logger.error("UNEXPECTED weights (%d): %s", len(load_result.unexpected_keys), load_result.unexpected_keys[:10])
        logger.info("FP8 native weight loading complete: %d params loaded, %d scales injected",
                    len(load_result.state_dict), len(_scale_store))

        # Pre-fuse dequanted decode weights before model moves to device/compiles
        for layer in self.model.layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, '_fuse_decode_weights'):
                layer.mlp._fuse_decode_weights()
        logger.info("Pre-fused decode MoE weights (BF16 einsum path)")

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Glm52Config.from_configs(hf_config, neuron_config)
        # Monkey-patch to use FP8 native components
        import vllm_neuron.model.glm_5_2.model as model_mod
        orig_dense = model_mod.Glm52DenseMLP
        orig_shared = model_mod.Glm52SharedExpertMLP
        orig_moe = model_mod.Glm52MoE
        model_mod.Glm52DenseMLP = Glm52DenseMLPFP8Native
        model_mod.Glm52SharedExpertMLP = Glm52SharedExpertMLPFP8Native
        model_mod.Glm52MoE = Glm52MoELayerFP8Native
        try:
            model = cls(config)
        finally:
            model_mod.Glm52DenseMLP = orig_dense
            model_mod.Glm52SharedExpertMLP = orig_shared
            model_mod.Glm52MoE = orig_moe
        return model
