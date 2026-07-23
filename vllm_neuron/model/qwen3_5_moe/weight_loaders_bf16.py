# SPDX-License-Identifier: Apache-2.0
"""MoE expert weight loaders for the Qwen3.5-MoE rebase (self-contained).

Qwen3.5-MoE stores FUSED stacked-expert tensors in the HF checkpoint:
  - model.....mlp.experts.gate_up_proj: [E, 2*I, H]   (gate then up on dim 1)
  - model.....mlp.experts.down_proj:    [E, H, I]

The NF kernels (moe_cte / moe_block_tkg) want this parameter layout:
  - gate_up_proj_weight: [E_local, H, 2*I_per_rank]   (reshaped to [E,H,2,I])
  - down_proj_weight:    [E_local, I_per_rank, H]

These loaders transform HF -> kernel layout: transpose + TP-shard the
intermediate dim. EP filtering on dim 0 (expert axis) is applied AFTER this
transform by expert_parallel_weight_loader wrapping in the MoE block, so these
loaders return the FULL expert axis [E, ...].

Self-contained copy (only the two expert loaders) so the rebase does NOT import
from vllm_neuron/model/qwen3_5_moe. The rebase keeps the dense attention /
GatedDeltaNet inline loaders in model.py — the other loaders from the source
qwen3_5_moe/weight_loaders_bf16.py are intentionally NOT copied (unused here).
"""

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader


# =============================================================================
# GatedDeltaNet (linear-attention) TP head-sharding loaders.
#
# IMPORTANT — storage orientation: this rebase stores the GDN in_proj / out_proj
# params TRANSPOSED vs HF: our params are [hidden, out] (in-major, used as
# ``h @ W``), whereas HF stores [out, hidden] (Linear, out-major). So every
# loader here must (1) transpose HF [out,H] -> [H,out], AND (2) head-slice the
# OUT axis, which after transpose is dim 1 (NOT dim 0 like the qwen3_5_moe
# reference, whose params stay out-major). Getting the axis wrong silently
# corrupts weights, so each loader documents which axis it slices.
#
# Head geometry: conv_dim = key_dim*2 + value_dim, laid out on the out axis as
# [q(key_dim) ; k(key_dim) ; v(value_dim)]. value/key heads shard evenly across
# TP (e.g. 32/8=4, 16/8=2); a naive contiguous slice of conv_dim would mis-split
# the q/k/v blocks, so we slice each sub-block on its own head axis and re-concat.
# =============================================================================


def gated_deltanet_in_proj_qkv_loader(key_dim, value_dim, num_shards):
    """HF in_proj_qkv [conv_dim, H] -> our [H, conv_dim/TP] (3-way head slice)."""

    def transform(slices, rank):
        w = slices[0][:].t()  # HF [conv_dim, H] -> [H, conv_dim]
        tp = rank % num_shards
        kpr = key_dim // num_shards
        vpr = value_dim // num_shards
        q = w[:, :key_dim]
        k = w[:, key_dim:2 * key_dim]
        v = w[:, 2 * key_dim:]
        q_s = q[:, tp * kpr:(tp + 1) * kpr]
        k_s = k[:, tp * kpr:(tp + 1) * kpr]
        v_s = v[:, tp * vpr:(tp + 1) * vpr]
        return torch.cat([q_s, k_s, v_s], dim=1).contiguous()  # [H, conv_dim/TP]

    return SafetensorsWeightLoader(transform=transform)


def gated_deltanet_conv1d_loader(key_dim, value_dim, num_shards):
    """HF conv1d [conv_dim, 1, k] -> our [conv_dim/TP, k], matching the qkv split.

    Depthwise conv (groups=conv_dim_local): channels must be EXACTLY the q/k/v
    channels this rank holds. conv stays out-major (channels on dim 0); slice dim 0.
    """

    def transform(slices, rank):
        w = slices[0][:].squeeze(1)  # [conv_dim, k]
        tp = rank % num_shards
        kpr = key_dim // num_shards
        vpr = value_dim // num_shards
        q = w[:key_dim, :]
        k = w[key_dim:2 * key_dim, :]
        v = w[2 * key_dim:, :]
        q_s = q[tp * kpr:(tp + 1) * kpr, :]
        k_s = k[tp * kpr:(tp + 1) * kpr, :]
        v_s = v[tp * vpr:(tp + 1) * vpr, :]
        return torch.cat([q_s, k_s, v_s], dim=0).contiguous()  # [conv_dim/TP, k]

    return SafetensorsWeightLoader(transform=transform)


def gated_deltanet_dim1_head_loader(num_shards):
    """HF [N_heads*X, H] -> our [H, N/TP]: transpose, shard the OUT axis (dim 1).

    Used for in_proj_z ([value_dim, H]), in_proj_a / in_proj_b ([num_v_heads, H]).
    """

    def transform(slices, rank):
        w = slices[0][:].t()  # [out, H] -> [H, out]
        tp = rank % num_shards
        shard = w.shape[1] // num_shards
        return w[:, tp * shard:(tp + 1) * shard].contiguous()

    return SafetensorsWeightLoader(transform=transform)


def gated_deltanet_dim0_head_loader(num_shards):
    """1-D params [num_v_heads] (dt_bias, A_log): even split on dim 0, no transpose."""

    def transform(slices, rank):
        w = slices[0][:]
        tp = rank % num_shards
        shard = w.shape[0] // num_shards
        return w[tp * shard:(tp + 1) * shard].contiguous()

    return SafetensorsWeightLoader(transform=transform)


def gated_deltanet_out_proj_loader(value_dim, num_shards):
    """HF out_proj [H, value_dim] -> our [value_dim/TP, H] (row-parallel input).

    Our param is stored [value_dim, H] (in-major; used as ``core @ out_proj``).
    Shard the INPUT (value_dim) axis = dim 0 after transpose; outputs are partial
    sums combined by reduce_scatter (prefill) / all_reduce (decode) in forward.
    """

    def transform(slices, rank):
        w = slices[0][:].t()  # HF [H, value_dim] -> [value_dim, H]
        tp = rank % num_shards
        vpr = value_dim // num_shards
        return w[tp * vpr:(tp + 1) * vpr, :].contiguous()  # [value_dim/TP, H]

    return SafetensorsWeightLoader(transform=transform)


def expert_gate_up_weight_loader(
    num_experts: int,
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Transform HF fused gate_up_proj [E, 2*I, H] -> [E, H, 2*I/TP].

    HF dim 1 is laid out as [gate(I) ; up(I)]. Shard each of gate and up on the
    intermediate dim by tp_rank, re-concat as [gate_shard ; up_shard], then
    transpose (2*I/TP, H) -> (H, 2*I/TP) to match the kernel layout (which
    reshapes [E, H, 2*I/TP] -> [E, H, 2, I/TP]).

    Args:
        num_experts: total experts E (full, before EP filtering).
        shard_size: 2 * intermediate_size_per_rank (the per-rank 2*I slice).
        num_shards: tp_degree.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1, "expert_gate_up_weight_loader expects one fused tensor"
        w = slices[0][:]  # [E, 2*I, H]
        E, two_I, H = w.shape
        I = two_I // 2

        tp_rank = rank % num_shards
        half = shard_size // 2  # = I_per_rank
        start = tp_rank * half
        end = start + half

        gate = w[:, :I, :]   # [E, I, H]
        up = w[:, I:, :]     # [E, I, H]
        gate_shard = gate[:, start:end, :]  # [E, I/TP, H]
        up_shard = up[:, start:end, :]      # [E, I/TP, H]
        fused = torch.cat([gate_shard, up_shard], dim=1)
        return fused.transpose(1, 2).contiguous()  # [E, H, 2*I/TP]

    return SafetensorsWeightLoader(transform=transform)


def expert_down_weight_loader(
    num_experts: int,
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Transform HF fused down_proj [E, H, I] -> [E, I/TP, H].

    Shard the intermediate dim (dim 2) by tp_rank, then transpose (H, I/TP)
    -> (I/TP, H) to match the kernel layout.

    Args:
        num_experts: total experts E (full, before EP filtering).
        shard_size: intermediate_size_per_rank (I/TP).
        num_shards: tp_degree.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1, "expert_down_weight_loader expects one fused tensor"
        w = slices[0][:]  # [E, H, I]
        tp_rank = rank % num_shards
        start = tp_rank * shard_size
        end = start + shard_size
        down_shard = w[:, :, start:end]   # [E, H, I/TP]
        return down_shard.transpose(1, 2).contiguous()  # [E, I/TP, H]

    return SafetensorsWeightLoader(transform=transform)
