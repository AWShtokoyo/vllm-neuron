# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.2 FP8 Forward Dequant — ROW quantization mode.

Store FP8 weights + per-row (per-output-channel) scales on HBM. Forward uses
NF.mlp with quantization_type=ROW which handles dequant inside the kernel —
no transient BF16 materialization needed, and no activation scales required.

At load time: dequant block-wise FP8 → BF16, compute per-column amax,
re-quantize to TRN2 FP8 (e4m3, max 240) per column, store FP8 weight +
per-column dequant scale [P_MAX, out_dim] on HBM.

This halves weight HBM vs BF16, with slight precision loss from collapsing
block scales to per-row. Compilation is fast because the kernel natively
handles FP8+scales (same graph structure as STATIC, no shape-dependent ops).
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Enable FP8 support in moe_tkg kernel (gated by TODO in upstream)
import vllm_neuron.functional.moe.moe_tkg as _moe_tkg_mod
from vllm_neuron.utils.neuron_utils import can_run_kernel as _can_run_kernel

def _can_use_kernel_fp8(hidden_input, expert_down_weights):
    if not _can_run_kernel(hidden_input):
        return False
    if expert_down_weights.dtype == torch.uint16:
        return True
    if expert_down_weights.dtype == torch.bfloat16:
        return True
    if expert_down_weights.dtype == _FP8_DTYPE:
        return True
    return False

_moe_tkg_mod._can_use_kernel = _can_use_kernel_fp8

logger = logging.getLogger(__name__)

_FP8_DTYPE = torch.float8_e4m3fn
_BLOCK_SIZE = 128
_PMAX = 128

_FP8_E4M3_MAX = 240.0

# The token count at or below which nkilib routes an MLP to the TKG kernel, taken
# from nkilib rather than copied: `is_mlp_tkg` compares `batch_size *
# sequence_len` against it, and nkilib's own comment on the constant says
# "Threshold currently set to 96 based on existing tuning; subject to future
# refinement." The shared expert picks its weight FORMAT from this boundary
# (BF16 for CTE, FP8+ROW for TKG), so a stale copy would feed the kernel the
# wrong one. The fallback keeps the module importable against an nkilib that has
# moved or renamed the constant; 96 is the value as of nkilib 0.6.0.
try:
    from nkilib.core.mlp.mlp_parameters import TKG_BS_SEQLEN_THRESHOLD as _TKG_THRESHOLD
except Exception:  # noqa: BLE001 - keep the model importable; log so it is not silent
    _TKG_THRESHOLD = 96
    logger.warning(
        "nkilib TKG_BS_SEQLEN_THRESHOLD not importable; falling back to %d. If "
        "nkilib's threshold has changed, the shared expert may hand the MLP kernel "
        "the wrong weight format.", _TKG_THRESHOLD
    )


def _dequant_block_fp8(weight_fp8, scale_inv):
    """Dequant block-wise FP8 → f32."""
    out_dim, in_dim = weight_fp8.shape
    w_f32 = weight_fp8.float()
    scale_expanded = (
        scale_inv.float()
        .repeat_interleave(_BLOCK_SIZE, dim=0)
        .repeat_interleave(_BLOCK_SIZE, dim=1)
    )[:out_dim, :in_dim]
    return w_f32 * scale_expanded


def _requantize_row_fp8(w_t):
    """Re-quantize transposed weight [in, out] to per-column FP8.

    Returns (fp8_weight [in, out], dequant_scale [1, out]).
    Each output column is independently scaled by its amax.
    The scale is stored compact; callers expand to [PMAX, out] at forward time.
    """
    w_f32 = w_t.float()
    # Per-column amax (along input dimension = dim 0)
    col_amax = w_f32.abs().amax(dim=0).clamp(min=1e-12)  # [out]
    # Per-column quantization scale
    col_scale = _FP8_E4M3_MAX / col_amax  # [out]
    # Quantize
    w_scaled = w_f32 * col_scale.unsqueeze(0)  # [in, out] * [1, out]
    w_fp8 = w_scaled.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(_FP8_DTYPE)
    # Compact scale [1, out] — expand to [PMAX, out] deferred to forward()
    dequant_scale = (col_amax / _FP8_E4M3_MAX).to(torch.float32).unsqueeze(0)
    return w_fp8, dequant_scale


def _fp8_row_weight_loader(shard_dim, shard_size, num_shards):
    """Load FP8 block-quantized weight → shard → transpose → re-quantize per-row."""

    def transform(slices, rank):
        tp_rank = rank % num_shards
        assert len(slices) == 2
        weight_fp8 = slices[0][:]
        scale_inv = slices[1][:]

        w_bf16 = _dequant_block_fp8(weight_fp8, scale_inv).to(torch.bfloat16)

        start_idx = tp_rank * shard_size
        end_idx = start_idx + shard_size
        sl = [slice(None)] * 2
        sl[shard_dim] = slice(start_idx, end_idx)
        w_shard = w_bf16[tuple(sl)]

        # Transpose to [in, out] for NF kernels
        w_shard_t = w_shard.T.contiguous()

        # Re-quantize per column (= per output channel)
        w_fp8_new, dequant_scale = _requantize_row_fp8(w_shard_t)
        return w_fp8_new, dequant_scale

    return SafetensorsWeightLoader(transform=transform)


def _fp8_row_moe_weight_loader(num_experts, shard_dim, shard_size, num_shards):
    """Load FP8 block-quantized MoE weights → re-quantize per-row per expert."""

    def transform(slices, rank):
        tp_rank = rank % num_shards
        assert len(slices) == 2 * num_experts
        weight_slices = slices[:num_experts]
        scale_slices = slices[num_experts:]

        fp8_experts = []
        scale_experts = []
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
            w_fp8_new, dequant_scale = _requantize_row_fp8(w_shard_t)
            fp8_experts.append(w_fp8_new)
            scale_experts.append(dequant_scale)

        return torch.stack(fp8_experts, dim=0), torch.stack(scale_experts, dim=0)

    return SafetensorsWeightLoader(transform=transform)


# =============================================================================
# FP8 ROW Dense MLP
# =============================================================================


class Glm52DenseMLPFP8Fwd(nn.Module):
    def __init__(self, config: Glm52Config):
        super().__init__()

        self.tp_group = get_tp_group()
        self.sp_group = self.tp_group
        self.world_size = self.tp_group.world_size

        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size

        # FP8 weights [in, out] layout
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

        # Per-column dequant scales stored compact [1, out_dim]
        I = self.intermediate_size_per_rank
        H = config.hidden_size
        self.gate_w_scale = nn.Parameter(
            torch.ones(1, I, dtype=torch.float32), requires_grad=False
        )
        self.up_w_scale = nn.Parameter(
            torch.ones(1, I, dtype=torch.float32), requires_grad=False
        )
        self.down_w_scale = nn.Parameter(
            torch.ones(1, H, dtype=torch.float32), requires_grad=False
        )

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill and self.sp_group.world_size > 1:
            hidden_states = self.sp_group.all_gather(hidden_states, dim=0)

        if is_prefill:
            # CTE path: dequant weights to BF16 (CTE ROW requires packed-scale FP8 input)
            gate_w = self.gate_proj_weight.to(torch.bfloat16) * self.gate_w_scale
            up_w = self.up_proj_weight.to(torch.bfloat16) * self.up_w_scale
            down_w = self.down_proj_weight.to(torch.bfloat16) * self.down_w_scale
            output = NF.mlp(hidden_states, gate_w, up_w, down_w)
        else:
            # TKG path: FP8 weights + ROW scales — expand [1,D]→[PMAX,D] for kernel
            output = NF.mlp(
                hidden_states,
                self.gate_proj_weight,
                self.up_proj_weight,
                self.down_proj_weight,
                quantization_type=QuantizationType.ROW,
                gate_w_scale=self.gate_w_scale.expand(_PMAX, -1),
                up_w_scale=self.up_w_scale.expand(_PMAX, -1),
                down_w_scale=self.down_w_scale.expand(_PMAX, -1),
            )

        if self.sp_group.world_size > 1:
            if is_prefill:
                output = self.sp_group.reduce_scatter(output, dim=0)
            else:
                self.sp_group.all_reduce(output)

        return output


# =============================================================================
# FP8 ROW Shared Expert MLP
# =============================================================================


class Glm52SharedExpertMLPFP8Fwd(nn.Module):
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

        I = self.intermediate_size_per_rank
        H = config.hidden_size
        self.gate_w_scale = nn.Parameter(
            torch.ones(1, I, dtype=torch.float32), requires_grad=False
        )
        self.up_w_scale = nn.Parameter(
            torch.ones(1, I, dtype=torch.float32), requires_grad=False
        )
        self.down_w_scale = nn.Parameter(
            torch.ones(1, H, dtype=torch.float32), requires_grad=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        T = hidden_states.shape[0]
        # 🔴 This branch must agree with nkilib's OWN CTE/TKG decision, because the
        # two paths hand the kernel different weight formats: BF16 for CTE, FP8+ROW
        # for TKG. Disagree and the kernel gets the format the other path wanted.
        # nkilib decides in `is_mlp_tkg` by `batch_size * sequence_len <=
        # TKG_BS_SEQLEN_THRESHOLD`, and that constant's own comment reads "subject
        # to future refinement" -- so it is read from nkilib rather than copied as a
        # literal 96, which is what this used to be.
        if T > _TKG_THRESHOLD:
            # CTE path: dequant to BF16 (CTE ROW requires packed-scale FP8 input)
            gate_w = self.gate_proj_weight.to(torch.bfloat16) * self.gate_w_scale
            up_w = self.up_proj_weight.to(torch.bfloat16) * self.up_w_scale
            down_w = self.down_proj_weight.to(torch.bfloat16) * self.down_w_scale
            return NF.mlp(hidden_states, gate_w, up_w, down_w)
        else:
            # TKG path: FP8 + ROW — expand [1,D]→[PMAX,D] for kernel
            return NF.mlp(
                hidden_states,
                self.gate_proj_weight,
                self.up_proj_weight,
                self.down_proj_weight,
                quantization_type=QuantizationType.ROW,
                gate_w_scale=self.gate_w_scale.expand(_PMAX, -1),
                up_w_scale=self.up_w_scale.expand(_PMAX, -1),
                down_w_scale=self.down_w_scale.expand(_PMAX, -1),
            )


# =============================================================================
# FP8 ROW MoE Layer
# =============================================================================


class Glm52MoELayerFP8Fwd(BF16MoELayer):
    """MoE with FP8 expert weights + per-row scales.
    Prefill: transient dequant → moe_cte (BF16).
    Decode: transient dequant → matmul loop."""

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

        # FP8 expert weights [E_local, in, out]
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

        # Per-expert per-column scales stored compact [E_local, 1, out_dim]
        I_TP = self.intermediate_size_per_rank
        H = self.hidden_size
        self.gate_proj_scales = nn.Parameter(
            torch.ones(self.num_local_experts, 1, I_TP, dtype=torch.float32),
            requires_grad=False,
        )
        self.up_proj_scales = nn.Parameter(
            torch.ones(self.num_local_experts, 1, I_TP, dtype=torch.float32),
            requires_grad=False,
        )
        self.down_proj_scales = nn.Parameter(
            torch.ones(self.num_local_experts, 1, H, dtype=torch.float32),
            requires_grad=False,
        )

        # Shared expert (FP8 ROW)
        self.shared_expert = Glm52SharedExpertMLPFP8Fwd(config)

    def _prepare_tkg_weights(self):
        """Reshape weights for moe_tkg kernel: stack gate+up into [E_L, H, 2, I].

        Frees the original gate/up weight params to avoid doubling HBM at 78L.
        Prefill path slices back from the stacked tensor.
        """
        self._tkg_gate_up_w = nn.Parameter(
            torch.stack([self.gate_proj_weights, self.up_proj_weights], dim=2),
            requires_grad=False,
        )
        del self.gate_proj_weights
        del self.up_proj_weights

        gate_s = self.gate_proj_scales[:, 0, :]  # [E_L, I]
        up_s = self.up_proj_scales[:, 0, :]  # [E_L, I]
        self._tkg_gate_up_scale = nn.Parameter(
            torch.stack([gate_s, up_s], dim=1),  # [E_L, 2, I]
            requires_grad=False,
        )
        del self.gate_proj_scales
        del self.up_proj_scales

        self._tkg_down_scale = nn.Parameter(
            self.down_proj_scales[:, 0, :],  # [E_L, H]
            requires_grad=False,
        )
        del self.down_proj_scales

    def _forward_decode_einsum(self, hidden_states_2d: torch.Tensor) -> torch.Tensor:
        """Kernel-free FP8 decode MoE: transient dequant → BF16 einsum.

        Mirrors the BF16 ``Glm52MoE._forward_decode`` (model.py) exactly, but
        dequantizes the FP8 stacked weights first (same per-row dequant as
        ``_forward_prefill``). Used by the MTP draft (``use_einsum_decode=True``)
        to AVOID the ``moe_tkg`` NKI kernel, whose internal indirect DMA goes
        out-of-bound at the γ=1 verify token shape (T=bs*(1+γ)). The
        MoE is per-token, so any T is valid here. Slower than the kernel, but the
        draft is 1 layer of 78 so the per-step cost is small, and it unblocks the
        greedy-losslessness gate + α measurement.
        """
        T = hidden_states_2d.shape[0]
        topk_weights, topk_indices = self._compute_routing(hidden_states_2d)

        # Per-local-expert affinities [T, E_local] (global index → local slot).
        # _compute_routing returns FP32; this path multiplies the affinities into a
        # bf16 einsum, so pin the previous dtype explicitly instead of inheriting it.
        topk_weights = topk_weights.to(self.dtype)
        local_affinities = torch.zeros(
            T, self.num_local_experts,
            dtype=topk_weights.dtype, device=hidden_states_2d.device,
        )
        for e_local in range(self.num_local_experts):
            e_global = self.local_expert_start + e_local
            mask = (topk_indices == e_global)
            local_affinities[:, e_local] = (
                topk_weights * mask.to(topk_weights.dtype)
            ).sum(dim=-1)

        # Transient FP8→BF16 dequant of the stacked gate/up + down (same pattern
        # as _forward_prefill lines: slice [E_L,H,2,I], apply per-row scales).
        gate_w = self._tkg_gate_up_w[:, :, 0, :]          # [E_L, H, I]
        up_w = self._tkg_gate_up_w[:, :, 1, :]            # [E_L, H, I]
        gate_scale = self._tkg_gate_up_scale[:, 0, :]     # [E_L, I]
        up_scale = self._tkg_gate_up_scale[:, 1, :]       # [E_L, I]
        gate_bf16 = gate_w.to(torch.bfloat16) * gate_scale.unsqueeze(1)
        up_bf16 = up_w.to(torch.bfloat16) * up_scale.unsqueeze(1)
        down_bf16 = self.down_proj_weights.to(torch.bfloat16) * self._tkg_down_scale.unsqueeze(1)

        # Batched einsum over local experts (matches BF16 Glm52MoE._forward_decode).
        gate = torch.einsum("th,ehi->eti", hidden_states_2d, gate_bf16)
        up = torch.einsum("th,ehi->eti", hidden_states_2d, up_bf16)
        intermediate = F.silu(gate) * up
        down = torch.einsum("eti,eih->eth", intermediate, down_bf16)
        output = torch.einsum("eth,te->th", down, local_affinities)
        return output

    def _forward_decode(self, hidden_states_2d: torch.Tensor) -> torch.Tensor:
        # The MTP draft routes decode MoE through the kernel-free einsum path
        # (moe_tkg's indirect DMA over-reads at the γ=1 verify shape). Set on the
        # draft's layer-78 MoE only; the base target keeps the fast kernel path.
        if getattr(self, "use_einsum_decode", False):
            return self._forward_decode_einsum(hidden_states_2d)

        T = hidden_states_2d.shape[0]
        topk_weights, topk_indices = self._compute_routing(hidden_states_2d)

        # DGE guard: moe_tkg's affinity-mask indirect DMA (moe_tkg_affinity_masking.py)
        # falls to the swdge path when T % _DGE_ALIGNMENT(=16) != 0, whose GpSimd
        # descriptor over-reads the affinity source up to the next multiple of 16
        # partitions and aborts with a DGE out-of-bound (that kernel omits
        # oob_mode.skip, unlike its MX sibling). Pad T up to a multiple of 16 so the
        # DMA takes the aligned (dge_mode.unknown) path and never over-reads; pad rows
        # route to expert 0 with zero affinity (masked, contribute nothing) and are
        # sliced off. Covers BOTH the base decode shape (T=bs) and the MTP γ=1 verify
        # shape (T=bs*(1+γ)) — both non-multiples of 16. The MoE is per-token so real
        # rows are unaffected.
        _DGE_ALIGNMENT = 16
        T_pad = ((T + _DGE_ALIGNMENT - 1) // _DGE_ALIGNMENT) * _DGE_ALIGNMENT
        pad = T_pad - T

        # Build full [T, E_global] affinity matrix for the kernel, then pad rows.
        full_affinities = torch.zeros(
            T, self.num_experts, dtype=torch.float32, device=hidden_states_2d.device
        )
        full_affinities.scatter_(1, topk_indices, topk_weights.to(torch.float32))

        expert_index = topk_indices.to(torch.int32)
        if pad:
            hidden_states_2d = F.pad(hidden_states_2d, (0, 0, 0, pad))
            full_affinities = F.pad(full_affinities, (0, 0, 0, pad))
            expert_index = F.pad(expert_index, (0, 0, 0, pad))  # pad rows -> expert 0

        ep_rank = torch.tensor([[self.ep_rank]], dtype=torch.int32, device=hidden_states_2d.device)

        output = NF.moe_tkg(
            hidden_input=hidden_states_2d,
            expert_gate_up_weights=self._tkg_gate_up_w,
            expert_down_weights=self.down_proj_weights,
            expert_affinities=full_affinities,
            expert_index=expert_index,
            is_all_expert=True,
            rank_id=ep_rank,
            expert_gate_up_weights_scale=self._tkg_gate_up_scale,
            expert_down_weights_scale=self._tkg_down_scale,
            mask_unselected_experts=True,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
            # The optimized gate-up weight layout halves MoE weight HBM traffic but is
            # flat on full-model throughput (the fragmented DGE weight packets already
            # overlap compute, so cutting bytes doesn't cut wall-clock). It also requires
            # a matching nkilib kernel flag that is not yet available upstream, so it is
            # left off by default. Re-enable once the nkilib kernel lands and packet
            # coalescing moves the weight load off the critical path:
            # use_tkg_gate_up_proj_optimized_layout=True,
        )

        # Slice off the DGE alignment pad rows (if any) → back to T real tokens.
        if pad:
            output = output[:T]
        return output

    def _forward_prefill(self, hidden_states_2d: torch.Tensor) -> torch.Tensor:
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

        # Slice gate/up from the stacked tkg tensor [E_L, H, 2, I]
        gate_w = self._tkg_gate_up_w[:, :, 0, :]  # [E_L, H, I]
        up_w = self._tkg_gate_up_w[:, :, 1, :]  # [E_L, H, I]
        gate_scale = self._tkg_gate_up_scale[:, 0, :]  # [E_L, I]
        up_scale = self._tkg_gate_up_scale[:, 1, :]  # [E_L, I]
        down_scale = self._tkg_down_scale  # [E_L, H]

        gate_bf16 = gate_w.to(torch.bfloat16) * gate_scale.unsqueeze(1)
        up_bf16 = up_w.to(torch.bfloat16) * up_scale.unsqueeze(1)
        down_bf16 = self.down_proj_weights.to(torch.bfloat16) * down_scale.unsqueeze(1)

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
    """GLM-5.2 with FP8 ROW quantization (per-row weight scales, no activation scales)."""

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

        # Build checkpoint mappings
        mappings = dict()

        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"
            is_dense = layer_id < self.config.first_k_dense_replace

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

            # DSA indexer, `full` layers only and only when DSA is on. 🔴 The five
            # tensors are NOT uniformly quantized: `wq_b` and `wk` are float8_e4m3fn
            # with a weight_scale_inv, while `weights_proj`, `k_norm.weight` and
            # `k_norm.bias` are plain BF16 with no scale at all. Mapping the BF16 three
            # as 2-slice FP8 entries would fail on a missing scale key; mapping the FP8
            # two as 1-slice entries would store raw fp8 bytes reinterpreted as bf16.
            # Verified against the reference checkpoint: wk.weight_scale_inv is (1, 48) and
            # wq_b.weight_scale_inv is (32, 16), both matching a 128-block grid.
            if self.model.layers[layer_id].self_attn.dsa_is_full:
                ix = f"{prefix}.self_attn.indexer"
                for ours, theirs in (("wq_b_weight", "wq_b"), ("wk_weight", "wk")):
                    mappings[f"{ix}.{ours}"] = [
                        f"{ix}.{theirs}.weight",
                        f"{ix}.{theirs}.weight_scale_inv",
                    ]
                mappings[f"{ix}.weights_proj_weight"] = f"{ix}.weights_proj.weight"
                mappings[f"{ix}.k_norm_weight"] = f"{ix}.k_norm.weight"
                mappings[f"{ix}.k_norm_bias"] = f"{ix}.k_norm.bias"

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
                mappings[f"{prefix}.mlp.gate_weight"] = f"{prefix}.mlp.gate.weight"
                mappings[f"{prefix}.mlp.e_score_correction_bias"] = (
                    f"{prefix}.mlp.gate.e_score_correction_bias"
                )

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

        mappings["model.norm.weight"] = "model.norm.weight"
        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["lm_head.weight"] = "lm_head.weight"

        # --- Weight loaders ---
        _scale_store = {}

        from .model import _replicated_transposed_weight_loader
        from .weight_loaders_fp8 import (
            fp8_dequant_weight_loader,
            fp8_dequant_row_parallel_weight_loader,
            fp8_dequant_replicated_weight_loader,
        )

        # Defined here, OUTSIDE the loop, because both branches of the per-layer
        # `if is_dense` need it: the dense branch for its MLP and the MoE branch for
        # the shared expert. It used to live inside the dense branch, which worked
        # for this checkpoint only by accident -- `first_k_dense_replace=3` means
        # layer 0 is dense, so the name was bound on the first iteration and still
        # in scope when a later MoE layer's shared expert reached for it. With
        # `first_k_dense_replace=0` no layer is dense, nothing binds the name, and
        # the first shared-expert loader raises UnboundLocalError.
        def _make_mlp_loader(proj_loader, scale_key):
            def transform(slices, rank):
                result = proj_loader.transform(slices, rank)
                fp8_w, dequant_scale = result
                _scale_store[scale_key] = dequant_scale
                return fp8_w
            return SafetensorsWeightLoader(transform=transform)

        for layer_id in range(len(self.model.layers)):
            layer = self.model.layers[layer_id]
            attn = layer.self_attn
            is_dense = layer_id < self.config.first_k_dense_replace
            prefix = f"model.layers.{layer_id}"

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

            if attn.indexer is not None:
                # Replicated, so no shard size is threaded in. wq_b/wk are FP8 and get
                # the dequantizing loader; weights_proj and both k_norm tensors are BF16
                # in the checkpoint and take the bare loader, transposed only where the
                # rank-2 layout differs (k_norm is 1-D, so it does not).
                fp8_rep = fp8_dequant_replicated_weight_loader()
                set_weight_loader(attn.indexer.wq_b_weight, fp8_rep)
                set_weight_loader(attn.indexer.wk_weight, fp8_rep)
                set_weight_loader(
                    attn.indexer.weights_proj_weight,
                    _replicated_transposed_weight_loader(),
                )
                set_weight_loader(attn.indexer.k_norm_weight, SafetensorsWeightLoader())
                set_weight_loader(attn.indexer.k_norm_bias, SafetensorsWeightLoader())

            if is_dense:
                mlp = layer.mlp
                gate_loader = _fp8_row_weight_loader(0, intermediate_per_rank, tp_size)
                up_loader = _fp8_row_weight_loader(0, intermediate_per_rank, tp_size)
                down_loader = _fp8_row_weight_loader(1, intermediate_per_rank, tp_size)

                set_weight_loader(mlp.gate_proj_weight, _make_mlp_loader(gate_loader, f"{prefix}.mlp.gate_w_scale"))
                set_weight_loader(mlp.up_proj_weight, _make_mlp_loader(up_loader, f"{prefix}.mlp.up_w_scale"))
                set_weight_loader(mlp.down_proj_weight, _make_mlp_loader(down_loader, f"{prefix}.mlp.down_w_scale"))
            else:
                moe = layer.mlp
                gate_moe_loader = _fp8_row_moe_weight_loader(
                    num_local_experts, 0, moe_intermediate_per_rank, ep_tp_size
                )
                up_moe_loader = _fp8_row_moe_weight_loader(
                    num_local_experts, 0, moe_intermediate_per_rank, ep_tp_size
                )
                down_moe_loader = _fp8_row_moe_weight_loader(
                    num_local_experts, 1, moe_intermediate_per_rank, ep_tp_size
                )

                def _make_moe_loader(proj_loader, scale_key):
                    def transform(slices, rank):
                        result = proj_loader.transform(slices, rank)
                        fp8_ws, scales = result  # [E, in, out], [E, 1, out]
                        _scale_store[scale_key] = scales
                        return fp8_ws
                    return SafetensorsWeightLoader(transform=transform)

                set_weight_loader(moe.gate_proj_weights, _make_moe_loader(gate_moe_loader, f"{prefix}.mlp.gate_proj_scales"))
                set_weight_loader(moe.up_proj_weights, _make_moe_loader(up_moe_loader, f"{prefix}.mlp.up_proj_scales"))
                set_weight_loader(moe.down_proj_weights, _make_moe_loader(down_moe_loader, f"{prefix}.mlp.down_proj_scales"))

                # Shared expert
                shared = moe.shared_expert
                sh_gate_loader = _fp8_row_weight_loader(0, shared_intermediate_per_rank, tp_size)
                sh_up_loader = _fp8_row_weight_loader(0, shared_intermediate_per_rank, tp_size)
                sh_down_loader = _fp8_row_weight_loader(1, shared_intermediate_per_rank, tp_size)

                set_weight_loader(shared.gate_proj_weight, _make_mlp_loader(sh_gate_loader, f"{prefix}.mlp.shared_expert.gate_w_scale"))
                set_weight_loader(shared.up_proj_weight, _make_mlp_loader(sh_up_loader, f"{prefix}.mlp.shared_expert.up_w_scale"))
                set_weight_loader(shared.down_proj_weight, _make_mlp_loader(sh_down_loader, f"{prefix}.mlp.shared_expert.down_w_scale"))

        # Load
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        load_result = checkpoint.load_sharded(
            tp_rank, tp_size, self, mappings, device, strict=False,
        )
        rank_sharded = load_result.state_dict

        # Inject scales into state dict
        for scale_key, scale_val in _scale_store.items():
            rank_sharded[scale_key] = scale_val

        self.load_state_dict(rank_sharded, strict=False, assign=True)

        real_missing = [k for k in (load_result.missing_keys or []) if k not in rank_sharded]
        if real_missing:
            logger.error("MISSING weights (%d): %s", len(real_missing), real_missing[:10])
        if load_result.unexpected_keys:
            logger.error("UNEXPECTED weights (%d): %s", len(load_result.unexpected_keys), load_result.unexpected_keys[:10])
        logger.info("FP8 ROW weight loading complete: %d params loaded, %d scales injected",
                    len(load_result.state_dict), len(_scale_store))

        for layer in self.model.layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, '_prepare_tkg_weights'):
                layer.mlp._prepare_tkg_weights()
        logger.info("Prepared moe_tkg weights (FP8 ROW decode kernel path)")

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Glm52Config.from_configs(hf_config, neuron_config)
        import vllm_neuron.model.glm_5_2.model as model_mod
        orig_dense = model_mod.Glm52DenseMLP
        orig_shared = model_mod.Glm52SharedExpertMLP
        orig_moe = model_mod.Glm52MoE
        model_mod.Glm52DenseMLP = Glm52DenseMLPFP8Fwd
        model_mod.Glm52SharedExpertMLP = Glm52SharedExpertMLPFP8Fwd
        model_mod.Glm52MoE = Glm52MoELayerFP8Fwd
        try:
            model = cls(config)
        finally:
            model_mod.Glm52DenseMLP = orig_dense
            model_mod.Glm52SharedExpertMLP = orig_shared
            model_mod.Glm52MoE = orig_moe
        return model
