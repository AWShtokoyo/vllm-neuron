# SPDX-License-Identifier: Apache-2.0
"""GatedDeltaNet (linear-attention) TP head-sharding weight loaders for the
Qwen3.5 dense (Qwen3.6-27B) port (self-contained).

These loaders shard the GatedDeltaNet in_proj / conv1d / out_proj parameters
across TP by q/k/v head geometry (see the orientation note below). They are
imported by Qwen3_5GatedDeltaNet.__init__ in model.py.

The dense SwiGLU MLP (Qwen3_5DenseMLP) uses the framework's standard
``sharding_weight_loader`` directly (see model.py), so — unlike the sibling
qwen3_5_moe port — there are NO expert/MoE loaders here.
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
