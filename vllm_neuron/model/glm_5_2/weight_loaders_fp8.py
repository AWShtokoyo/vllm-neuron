# SPDX-License-Identifier: Apache-2.0
"""Weight loaders for GLM-5.2 FP8 checkpoint format.

GLM-5.2 FP8 checkpoints use block-wise quantization (128×128 blocks):
  - weights stored as float8_e4m3fn
  - weight_scale_inv: [ceil(out/128), ceil(in/128)] inverse scales per block

Strategy: dequantize to BF16 at load time, store BF16 on device. This avoids
block-boundary alignment issues when shard_size < block_size (e.g., hidden=6144
/ TP=64 = 96 < 128). The forward path is identical to the BF16 model.

Dequantization formula: bf16 = fp8_value * scale_inv
(scale_inv is the dequantization multiplier per 128x128 block.)
"""

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

_BLOCK_SIZE = 128


def _dequant_and_shard(weight_fp8, scale_inv, shard_dim, shard_size, tp_rank):
    """Dequant full FP8 weight to BF16, then shard and transpose.

    Dequantization: bf16 = fp8_value * scale_inv_per_block
    (scale_inv is the dequantization multiplier, named "inv" because it's
    the inverse of the quantization scale used during calibration.)

    Args:
        weight_fp8: [out, in] float8_e4m3fn from checkpoint
        scale_inv: [ceil(out/128), ceil(in/128)] float32 dequant multipliers
        shard_dim: which dim to shard (0=out, 1=in)
        shard_size: elements per shard on shard_dim
        tp_rank: current TP rank

    Returns:
        BF16 tensor, sharded on shard_dim then transposed.
    """
    out_dim, in_dim = weight_fp8.shape
    w_f32 = weight_fp8.float()

    # Expand block scales to full weight dimensions
    scale_expanded = scale_inv.float().repeat_interleave(_BLOCK_SIZE, dim=0).repeat_interleave(_BLOCK_SIZE, dim=1)
    scale_expanded = scale_expanded[:out_dim, :in_dim]

    w_bf16 = w_f32 * scale_expanded

    # Shard
    start_idx = tp_rank * shard_size
    end_idx = start_idx + shard_size
    sl = [slice(None)] * 2
    sl[shard_dim] = slice(start_idx, end_idx)
    w_shard = w_bf16[tuple(sl)]

    return w_shard.to(torch.bfloat16).T.contiguous()


def fp8_dequant_weight_loader(
    shard_dim: int,
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Load FP8 weight: dequant to BF16, shard, transpose.

    Loads both the weight and its scale_inv from checkpoint. The loader
    is configured to receive 2 slices: [weight, scale_inv].
    """

    def transform(slices, rank):
        tp_rank = rank % num_shards
        assert len(slices) == 2, f"Expected [weight, scale_inv], got {len(slices)} slices"
        weight_fp8 = slices[0][:]
        scale_inv = slices[1][:]
        return _dequant_and_shard(weight_fp8, scale_inv, shard_dim, shard_size, tp_rank)

    return SafetensorsWeightLoader(transform=transform)


def fp8_dequant_row_parallel_weight_loader(
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Row-parallel: dequant to BF16, shard on in_features (dim=1), transpose."""

    def transform(slices, rank):
        tp_rank = rank % num_shards
        assert len(slices) == 2
        weight_fp8 = slices[0][:]
        scale_inv = slices[1][:]
        return _dequant_and_shard(weight_fp8, scale_inv, 1, shard_size, tp_rank)

    return SafetensorsWeightLoader(transform=transform)


def fp8_dequant_moe_expert_weight_loader(
    num_experts: int,
    shard_dim: int,
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Load FP8 MoE expert weights: dequant to BF16, shard, transpose.

    Receives 2*num_experts slices: [weight_0, ..., weight_N, scale_0, ..., scale_N].
    """

    def transform(slices, rank):
        tp_rank = rank % num_shards
        assert len(slices) == 2 * num_experts, (
            f"Expected {2 * num_experts} slices (weights + scales), got {len(slices)}"
        )
        weight_slices = slices[:num_experts]
        scale_slices = slices[num_experts:]

        expert_tensors = []
        for w_slice, s_slice in zip(weight_slices, scale_slices):
            w_fp8 = w_slice[:]
            s_inv = s_slice[:]
            expert_tensors.append(
                _dequant_and_shard(w_fp8, s_inv, shard_dim, shard_size, tp_rank)
            )
        return torch.stack(expert_tensors, dim=0)

    return SafetensorsWeightLoader(transform=transform)
