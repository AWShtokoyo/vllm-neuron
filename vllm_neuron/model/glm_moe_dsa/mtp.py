# SPDX-License-Identifier: Apache-2.0
"""GLM MTP (Multi-Token Prediction) draft head — layer 78.

GLM ships a native MTP head at checkpoint layer 78: a FULL MLA+MoE decoder
layer wrapped by a fusion front-end (enorm/hnorm/eh_proj) and a shared output
head (shared_head.norm), reusing the base `embed_tokens` and (untied) `lm_head`.
Used here as a γ=1 self-speculative draft: predict token t+1 from the accepted
token x_t and the target hidden state h_t.

Draft forward (one propose step):

    emb   = embed_tokens(x_t)                                  # base shared embedding
    fused = eh_proj( cat([ enorm(emb), hnorm(h_t) ], dim=-1) ) # emb FIRST (D1), [.,12288]->[.,6144]
    hid   = GlmMoeDsaDecoderLayer@78(fused)                        # MLA + MoE (own KV), residuals internal (D7)
    out   = shared_head.norm(hid)                              # final norm (D5, distinct tensor)
    logits= lm_head(out)                                       # base untied lm_head (D6)
    draft = argmax(logits)                                     # greedy (reuse eagle3 helper)

Design decisions (RESOLVED):
  D1 concat = embedding FIRST (glm4_moe_mtp.py:112-118).
  D2 enorm→embedding, hnorm→target hidden.
  D3 target hidden fed to hnorm defaults to PRE-`model.norm` (the capture point;
     a 1-line flip switches to post-norm if a checkpoint requires it).
  D4 eh_proj replicated per rank (BF16, no shard, no scale) — 151MB, negligible vs ~10GB draft.
  D5 shared_head.norm is a SEPARATE tensor from decoder.post_attention_layernorm.
  D6 reuse base lm_head (no shared_head.head in checkpoint; tie_word_embeddings=False).
  D7 GlmMoeDsaDecoderLayer applies both residual adds internally — feed `fused` directly.

The DSA indexer (`self_attn.indexer.*`) IS present in the layer-78 checkpoint but is
SKIPPED in this port (full attention; a no-op at ctx ≤ 2048).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.parallel.neuron_parallel_state import (
    get_neuron_ep_degree,
    get_neuron_ep_rank,
)
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import set_weight_loader

# Reuse the base GLM building blocks verbatim (layer-idx-parameterized).
from .config import GlmMoeDsaConfig
from .model import (
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaRMSNorm,
    GlmMoeDsaRotaryEmbedding,
)
# Reuse the greedy sampler + accepted-token extraction from Eagle3 (model-agnostic).
from ..llama3.eagle3_model import extract_accepted_tokens

logger = logging.getLogger(__name__)

MTP_LAYER_IDX = 78  # config.num_hidden_layers == 78 → the MTP head is checkpoint layer 78


class GlmMoeDsaMtpLayer(nn.Module):
    """Fusion front-end + decoder-78 + final norm for the GLM MTP head.

    Wraps a standard `GlmMoeDsaDecoderLayer` (reused verbatim at layer_idx=78) with:
      * enorm / hnorm  — RMSNorms over the token embedding and target hidden (D2)
      * eh_proj        — replicated Linear(2*hidden, hidden) fusing the concat (D1/D4)
      * shared_head_norm — the final pre-lm_head RMSNorm (D5, separate tensor)
    """

    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int = MTP_LAYER_IDX):
        super().__init__()
        self.layer_idx = layer_idx
        hidden = config.hidden_size
        eps = config.rms_norm_eps
        dtype = config.torch_dtype

        # Fusion front-end (D1/D2/D4). enorm/hnorm replicated; eh_proj replicated BF16.
        self.enorm = GlmMoeDsaRMSNorm(hidden, eps, dtype)
        self.hnorm = GlmMoeDsaRMSNorm(hidden, eps, dtype)
        self.eh_proj = nn.Linear(2 * hidden, hidden, bias=False, dtype=dtype)

        # The decoder body — MLA self-attn (own KV cache) + MoE (256 experts + shared).
        # layer_idx=78 >= first_k_dense_replace=3 ⇒ GlmMoeDsaDecoderLayer builds a MoE layer.
        self.decoder = GlmMoeDsaDecoderLayer(config, layer_idx=layer_idx)

        # Final norm applied to the decoder output before lm_head (D5).
        self.shared_head_norm = GlmMoeDsaRMSNorm(hidden, eps, dtype)

    def fuse(self, emb: torch.Tensor, h_t: torch.Tensor,
             positions: torch.Tensor | None = None) -> torch.Tensor:
        """eh_proj(cat([enorm(emb), hnorm(h)], -1)) — emb FIRST (D1).

        Optional position-0 embedding mask for prefill-safety parity with the vLLM
        reference (glm4_moe_mtp.py:112); decode never sees position 0 in the draft
        window, so it is a no-op there.
        """
        if positions is not None:
            emb = torch.where(
                positions.view(*positions.shape, 1) == 0, torch.zeros_like(emb), emb
            )
        emb = self.enorm(emb)
        h_t = self.hnorm(h_t)
        return self.eh_proj(torch.cat([emb, h_t], dim=-1))

    def layer78(
        self,
        fused: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None,
    ) -> torch.Tensor:
        # GlmMoeDsaDecoderLayer handles both residual adds internally (D7).
        return self.decoder(
            fused,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )


class GlmMoeDsaMtpForCausalLM(nn.Module):
    """GLM γ=1 MTP draft head.

    Public surface mirrors `Eagle3LlamaForCausalLM` so `MtpProposer` (which mirrors
    `EagleProposer`) can drive it: `forward`, `from_configs`, `load_weights`,
    `get_kv_spec`, `bind_kv_cache`, `num_speculative_tokens`. Unlike Eagle3 there is
    NO 3× aux concat, NO combine/fc layer, and NO recurrent loop (γ=1 ⇒ single pass).
    """

    def __init__(self, config: GlmMoeDsaConfig, start_layer_idx: int = MTP_LAYER_IDX):
        super().__init__()
        self.config = config
        self.start_layer_idx = start_layer_idx

        self.tp_group = get_tp_group()
        self.sp_group = self.tp_group
        self.world_size = self.tp_group.world_size
        self.sp_world_size = self.sp_group.world_size

        self.ep_degree = get_neuron_ep_degree()

        # γ is pinned to 1 for v1; the proposer overwrites this from the
        # speculative_config (mirrors eagle.py:338 pattern).
        self.num_speculative_tokens = 1

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        self.on_device_sampling = self.on_device_sampling_config is not None

        # Base shared embedding (reused; same VocabDimSharded layout as the target).
        from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
        from vllm_neuron.utils.weight_loader import sharding_weight_loader

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.sp_group.device_group,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.sp_world_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )

        # The MTP layer (fusion + decoder-78 + shared_head.norm).
        self.mtp = GlmMoeDsaMtpLayer(config, layer_idx=start_layer_idx)

        # RoPE — same construction as the base backbone.
        self.rotary_emb = GlmMoeDsaRotaryEmbedding(config)

        # Base untied lm_head. Same ColumnParallel layout as the target.
        # Structural fix (draft only): GATHER the draft's logits to full vocab
        # (gather_output=True) so the sampler can run a PLAIN torch.argmax instead of
        # the tensor-parallel `distributed_argmax`. distributed_argmax is built from
        # two `all_gather_into_tensor` collectives + a final gather over tiny
        # [batch, tp_degree] index/value buffers; in the draft's tiny single-layer
        # decode graph those transients sit at the arena tail, so their DGE indirect
        # scatter/gather tiles over-read into a guard page → a DGE abort (the
        # argmax-gather and the all_gather(index) scatter). A
        # full-vocab logits all-gather is one CONTIGUOUS collective (no indirect DMA)
        # + a plain argmax, eliminating that whole subsystem from the draft graph. The
        # extra all-gather moves [batch, vocab] bf16 (~1.2MB/rank at bs=4) — negligible
        # for the 1-layer draft. Bit-identical to distributed_argmax for greedy
        # (vocab divides evenly across ranks; ties break to the lowest token id in
        # both). Target keeps sharded logits + distributed_argmax (unaffected).
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=True,
            tp_group=self.sp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=config.vocab_size // self.sp_world_size,
                num_shards=self.sp_world_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )

        if self.on_device_sampling:
            # Structural fix (draft only): pass process_group=None so the
            # sampler runs a PLAIN torch.argmax on the already-gathered full-vocab
            # logits (see lm_head gather_output=True above) instead of the
            # tensor-parallel `distributed_argmax`. This removes distributed_argmax's
            # two all_gather scatters + final index gather — the tiny [batch,
            # tp_degree] transients whose DGE tiles over-read into a guard page in the
            # draft's single-layer graph. The draft's logits
            # are already full-vocab, so a non-distributed argmax is exact.
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=None,
            )

    # ------------------------------------------------------------------ #
    # Draft forward (γ=1). Mirrors eagle3_model.py:685-877 WITHOUT the 3×
    # aux concat / combine_hidden_states / recurrent loop.
    # ------------------------------------------------------------------ #
    def _sample_draft_token(self, logits: torch.Tensor) -> torch.Tensor:
        if self.on_device_sampling:
            return self.sampler(logits).to(torch.int32)
        return torch.argmax(logits, dim=-1).to(torch.int32)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,                     # [T] shifted target token ids
        positions: torch.LongTensor,                 # [T]
        initial_target_hidden_states: torch.Tensor,  # [T, hidden] SINGLE (not 3x)
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,   # [bs]
        rank: torch.Tensor | None = None,
        raw_sampled_token_ids: torch.Tensor | None = None,  # [bs, 1+γ] / [bs,1] prefill
        prev_sampled_token_ids: torch.Tensor | None = None,   # accepted, IGNORED (γ=1)
        prev_num_draft_tokens: torch.Tensor | None = None,    # accepted, IGNORED
        req_indices_per_token: torch.Tensor | None = None,    # accepted, IGNORED
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: NO NF.correct_spec_decode_positions_and_slot_mapping prologue.
        # MTP has no recurrent rejection; prev_* kwargs are accepted (for graph-shape
        # parity with EagleProposer) but ignored. v1 runs sync-only so the runner never
        # injects meaningful prev_* values.

        # Clamp sampling_positions into this draft graph's token range BEFORE the
        # extract helper scatters `input_ids` at those positions (eagle3_model.py:88).
        # sampling_positions comes from the target's bonus_logits_indices, computed
        # over the runner's (padded / concurrency-capped) token space; padding/fake
        # batch rows can point past the draft's `input_ids` length and trip the
        # on-device DGE scatter/gather out-of-bound abort. Real rows are already
        # in-bounds, so this only redirects discarded padding rows — greedy
        # losslessness is unchanged. (A second clamp guards the hidden gather below.)
        if sampling_positions is not None:
            sampling_positions = torch.clamp(
                sampling_positions, min=0, max=input_ids.shape[0] - 1
            )

        # On-device extract accepted tokens + bonus (reuse eagle3 helper verbatim).
        # NOTE: gather_free is NOT set here — the draft graph disables neuronx-cc DGE
        # on indirect DMA (MtpProposer.compile_and_load_draft_model), so the plain
        # `.gather` lowers to bounded DMA with no over-read (structural fix for the
        # padding-row indirect-DMA over-read).
        # The gather-free one-hot form is avoided because it tripped a
        # neuronx-cc scheduler internal error in the full draft graph.
        if raw_sampled_token_ids is not None:
            input_ids, sampling_positions, bonus_token_ids = extract_accepted_tokens(
                input_ids,
                sampling_positions,
                raw_sampled_token_ids,
                self.config.vocab_size,
                self.num_speculative_tokens,
            )
        else:
            bonus_token_ids = None

        positions = positions.to(torch.int32)

        # Fusion + decoder-78.
        emb = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        fused = self.mtp.fuse(emb, initial_target_hidden_states, positions=positions)

        position_embeddings = self.rotary_emb(
            positions, device=fused.device, dtype=fused.dtype
        )

        # SP-sharding contract for the reused GlmMoeDsaDecoderLayer.
        # The base backbone's PREFILL path expects hidden to arrive SP-SHARDED
        # (T/sp per rank): the target's model.forward reduce-scatters it via
        # embed_tokens(scatter_tokens=True), and GlmMoeDsaAttention.forward_prefill
        # then ALL-GATHERs it back to full T (model.py:262-263) so q_len matches
        # the full-T cos/sin. Its output is reduce-scattered back to sharded, and
        # the residual add needs a sharded residual. The DECODE path takes full-T
        # hidden as-is (no input gather).
        #
        # Our draft builds `fused` FULL-T on every rank (embed scatter_tokens=False
        # + already-all-gathered target hidden), which is correct for DECODE but
        # DOUBLE-inflates in PREFILL (all-gather 128→128*sp vs cos T=128 → crash).
        # So on the prefill step we slice `fused` to this rank's contiguous T/sp
        # shard before layer78 (mirroring the base reduce_scatter chunk order, so
        # the layer's internal all_gather restores full T), then all-gather the
        # sharded layer output back to full T before the sampling gather below.
        # cos/sin stay full-T (built from full-T `positions`) exactly as the base.
        layer_name = f"layers.{self.start_layer_idx}.self_attn"
        md = attn_metadata[layer_name]
        is_prefill = md["max_query_len"] > md["decode_token_threshold"]
        sp = self.sp_world_size

        if is_prefill and sp > 1:
            T = fused.shape[0]
            # 🔴 The divisibility this slice needs is enforced in a DIFFERENT class.
            # `GlmMoeDsaForCausalLM.forward` rejects `T % sp_world_size != 0` for the target,
            # and the draft inherits a valid T only because `extract_accepted_tokens`
            # SCATTERS into input_ids rather than resizing it. That coupling is invisible
            # from here: if it ever breaks, `T // sp` silently drops the remainder, the
            # all_gather below returns fewer than T rows, and the clamped gather then
            # reads the WRONG hidden for a real request — wrong drafts with no error.
            # Assert rather than trust, on Python ints so it resolves at trace time.
            if T % sp != 0:
                raise ValueError(
                    f"MTP draft prefill: token count {T} is not divisible by "
                    f"sp_world_size {sp}, so the SP shard would drop "
                    f"{T % sp} token(s) and the gather after all_gather would read the "
                    "wrong rows. The target model rejects this T; the draft should never "
                    "see it."
                )
            shard = T // sp
            start = self.sp_group.rank_in_group * shard
            fused = fused[start:start + shard]

        hidden_states = self.mtp.layer78(
            fused, positions, position_embeddings, attn_metadata
        )

        if is_prefill and sp > 1:
            # Layer output is SP-sharded (forward_prefill reduce-scatters); restore
            # full T so sampling_positions indexes the same space the target used.
            hidden_states = self.sp_group.all_gather(hidden_states, dim=0)

        # Gather per-request states at the sampling positions. Clamp to the valid row
        # range: on the decode/verify step `sampling_positions` comes from the target's
        # bonus_logits_indices, computed over the runner's (possibly padded /
        # concurrency-capped) token space, so it can point past this draft graph's
        # hidden_states for PADDING or fake batch rows. An out-of-range index would trip
        # the on-device DGE gather OOB. Clamping only moves those padding rows to a valid
        # (discarded) slot — real request indices are already in-bounds, so drafts and
        # greedy losslessness are unchanged. The draft graph disables neuronx-cc DGE on
        # indirect DMA (MtpProposer.compile_and_load_draft_model), so this gather lowers
        # to a bounded copy with no descriptor over-read.
        #
        # Guarded for None like the clamp at the top of this method. The parameter is
        # declared optional, and only the `raw_sampled_token_ids is not None` path
        # guarantees `extract_accepted_tokens` filled it in; clamping unconditionally
        # would raise a bare TypeError on the one path that legitimately omits it.
        if sampling_positions is None:
            raise ValueError(
                "MTP draft forward needs sampling_positions to select the per-request "
                "hidden state; it was None and no raw_sampled_token_ids were supplied to "
                "derive it from."
            )
        sampling_positions = torch.clamp(
            sampling_positions, min=0, max=hidden_states.shape[0] - 1
        )
        hidden_states = hidden_states[sampling_positions]

        # Final norm + base lm_head → greedy draft token.
        hidden_states = self.mtp.shared_head_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        draft_token_ids = self._sample_draft_token(logits)

        # Collect [bonus, draft] → [bs, 1+γ]; drafts-only → [bs, γ]. Matches
        # eagle3_model.py:794-866 (γ=1 ⇒ single draft, no recurrent appends).
        all_draft_tokens = (
            [bonus_token_ids, draft_token_ids]
            if bonus_token_ids is not None
            else [draft_token_ids]
        )
        stacked_tokens = torch.stack(all_draft_tokens, dim=1)
        drafts_only = torch.stack(
            all_draft_tokens[1:] if bonus_token_ids is not None else all_draft_tokens,
            dim=1,
        )
        return stacked_tokens, drafts_only, None

    # ------------------------------------------------------------------ #
    # KV cache — single MLA layer (layer 78). Mirrors model.py:1164-1191.
    # ------------------------------------------------------------------ #
    def get_kv_spec(self) -> KVSpec:
        attn = self.mtp.decoder.self_attn
        layer_name = f"layers.{self.start_layer_idx}.self_attn"
        return KVSpec(
            layers=[
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=attn.num_kv_cache_heads,
                    head_size=attn.kv_cache_head_dim,
                    dtype=attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                    is_mla=True,  # draft KV must be MLA to merge with the target MLA specs
                )
            ]
        )

    # NOTE: on device, the draft's DGE indirect-DMA can abort with an
    # out-of-bound access on the spec/verify step. The draft KV read itself is
    # not the aborting op (a small KV over-read would be absorbed by the guard
    # blocks); the remaining indexed-DMA transients live in the draft's tiny
    # single-layer graph at the arena edge. A robust fix needs either a
    # KV-sharing draft architecture (Q-only, reading the target's cache
    # read-only, so there is no standalone single-layer draft KV graph doing
    # indexed DMA at the pool edge) or a vendor neuronx-cc/nkilib DGE/OOBMode
    # change. Plain binding until then; on-device validation is in progress.
    def bind_kv_cache(self, kv_caches: dict) -> None:
        layer_name = f"layers.{self.start_layer_idx}.self_attn"
        if layer_name not in kv_caches:
            raise Exception(f"KV cache for draft layer {layer_name} not initialized")
        self.mtp.decoder.self_attn.k_cache = kv_caches[layer_name][0]
        self.mtp.decoder.self_attn.v_cache = kv_caches[layer_name][1]

    # ------------------------------------------------------------------ #
    # Weight loading — layer 78 only.
    # ------------------------------------------------------------------ #
    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        raise NotImplementedError(
            "GlmMoeDsaMtpForCausalLM.load_weights is provided by the FP8 subclass "
            "(model_fp8_per_channel path). The BF16 base head is CPU-reference only."
        )

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        start_layer_idx: int = MTP_LAYER_IDX,
        neuron_config: NeuronConfig | None = None,
    ):
        config = GlmMoeDsaConfig.from_configs(hf_config, neuron_config)
        return cls(config, start_layer_idx=start_layer_idx)


class GlmMoeDsaMtpForCausalLMFactory(nn.Module):
    """Registry entry for the GLM MTP draft head.

    Selects BF16 vs FP8-ROW (`fp8_per_channel`) based on `neuron_config.quantization`,
    mirroring `glm_moe_dsa/factory.py`. `MtpProposer.compile_and_load_draft_model`
    resolves the draft architecture name to this class and calls `from_configs`.
    """

    @classmethod
    def from_configs(
        cls,
        config: PretrainedConfig,
        start_layer_idx: int = MTP_LAYER_IDX,
        neuron_config: NeuronConfig | None = None,
    ) -> nn.Module:
        quantization = neuron_config.quantization if neuron_config else None
        if quantization == "fp8_per_channel":
            from .mtp_fp8 import GlmMoeDsaMtpForCausalLM as Model
        elif quantization in (None, "bf16"):
            Model = GlmMoeDsaMtpForCausalLM
        else:
            raise ValueError(
                f"quantization='{quantization}' is not supported for the GlmMoeDsa MTP "
                "draft head. Supported: 'fp8_per_channel' or None/bf16."
            )
        return Model.from_configs(
            config, start_layer_idx=start_layer_idx, neuron_config=neuron_config
        )
