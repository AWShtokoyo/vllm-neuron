# SPDX-License-Identifier: Apache-2.0
"""MTP (Multi-Token Prediction) self-speculative-decode proposer for GLM-5.2.

The layer-78 MTP head of GLM-5.2 is used as a γ=1 draft: it proposes one token
per step which the target verifies in the same forward (S_decode=2 per seq).
Acceptance is decided by the already-wired greedy rejection sampler.

This proposer mirrors ``EagleProposer`` (``vllm_neuron/vllm/spec_decode/eagle.py``)
one-for-one — same public surface so the runner's Eagle3 branch points reuse it —
with exactly three MTP-specific differences:

  1. ``method == "mtp"`` (not ``"eagle3"``).
  2. Single target hidden state ``[T, hidden]`` (no 3x aux-layer concat) in the
     synthetic warmup inputs.
  3. The draft arch is hard-set to ``Glm52MtpForCausalLM`` — vLLM's
     ``SpeculativeConfig.hf_config_override`` rewrites a ``glm_moe_dsa`` draft's
     ``architectures`` to ``["DeepSeekMTPModel"]`` (speculative.py:290-295), so the
     Eagle3 ``f"Eagle3{arch}"`` prefix logic cannot be reused verbatim.

There is NO recurrent draft loop (γ=1 ⇒ a single pass in the draft forward), and
v1 runs SYNC-ONLY so the ``@async_speculative_decoding`` correction prologue never
fires for MTP.
"""

import contextlib
import logging
import time

import torch
import torch.nn as nn
from vllm.v1.attention.backend import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_neuron import envs
from vllm_neuron.compile.backend import model_forward_context
from vllm_neuron.compile.capture_backend import CaptureComplete
from vllm_neuron.metrics import (
    COMPILATION_TIME,
    MODEL_LOAD_SIZE,
    MODEL_LOAD_TIME,
    NEFF_EXECUTION_COUNT,
)
from vllm_neuron.model.neuron_config import (
    NeuronConfig,
    OnDeviceSamplingConfig,
)
from vllm_neuron.model.registry import get_models

logger = logging.getLogger(__name__)


class MtpProposer:
    """γ=1 self-speculative proposer backed by GLM-5.2's layer-78 MTP head.

    Public surface is identical to ``EagleProposer`` (the runner asserts
    ``isinstance(self.drafter, (EagleProposer, MtpProposer))`` at three sites).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        on_device_sampling: bool = True,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None

        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        assert self.method == "mtp", (
            f"MtpProposer requires speculative method 'mtp', got '{self.method}'"
        )

        self.device = device
        self.on_device_sampling = on_device_sampling
        # v1: pinned to 1 (γ=1).
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        self.attn_layer_names: list[str] = []
        self.capture_backend_model = None
        self.model: nn.Module | None = None

        # Pass rank as a tensor input to avoid it becoming a compile-time
        # constant (see eagle.py:53-59).
        world_group = get_world_group()
        world_rank = world_group.rank if world_group else 0
        self.rank_tensor = torch.tensor(
            world_rank, dtype=torch.int32, device=self.device
        )

    # ------------------------------------------------------------------ #
    # Async-correction no-op kwargs (copied verbatim from eagle.py:61-104).
    # v1 MTP never runs the correction (sync-only), but the kwargs are still
    # passed for graph-shape parity with the Eagle3 draft NEFF signature.
    # ------------------------------------------------------------------ #
    def _build_noop_async_spec_correction_kwargs(
        self,
        num_tokens: int,
        num_reqs: int,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        if device is None:
            device = self.device
        prev_sampled_token_ids = torch.zeros(
            num_reqs, self.num_speculative_tokens + 1, dtype=torch.int32, device=device
        )
        prev_num_draft_tokens = torch.full(
            (num_reqs,), self.num_speculative_tokens, dtype=torch.int32, device=device
        )
        assert num_reqs > 0
        assert num_tokens % num_reqs == 0
        tokens_per_req = num_tokens // num_reqs
        req_indices_per_token = (
            torch.arange(num_reqs, dtype=torch.int64)
            .repeat_interleave(tokens_per_req)
            .to(device)
        )
        return {
            "prev_sampled_token_ids": prev_sampled_token_ids,
            "prev_num_draft_tokens": prev_num_draft_tokens,
            "req_indices_per_token": req_indices_per_token,
        }

    def load_model(self, target_hidden_size: int) -> None:
        target_num_layers = self.vllm_config.model_config.hf_config.num_hidden_layers
        self.model = self.compile_and_load_draft_model(
            start_layer_idx=target_num_layers,
            target_hidden_size=target_hidden_size,
        )
        logger.info("Completed compilation for MTP draft model.")

        # The MTP head is a SINGLE layer (layer 78). attn_layer_names must match
        # the draft head's get_kv_spec() layer name for the KV-group binding.
        self.attn_layer_names = [f"layers.{target_num_layers}.self_attn"]

    def _build_synthetic_inputs(
        self,
        num_tokens: int,
        num_reqs: int,
        device: torch.device | None = None,
    ) -> dict:
        """Synthetic inputs for warmup/graph_extract. MTP delta vs eagle.py:124-185:
        target_hidden_states is a SINGLE ``[T, hidden]`` (not ``hidden*3``)."""
        assert self.model is not None
        if device is None:
            device = self.device
        hidden_size = self.model.config.hidden_size

        target_token_ids = torch.ones(num_tokens, dtype=torch.int32).to(device)
        target_positions = torch.zeros(num_tokens, dtype=torch.long).to(device)
        # MTP: single hidden (NO 3x concat).
        target_hidden_states = torch.ones(
            num_tokens, hidden_size, dtype=torch.bfloat16
        ).to(device)

        tokens_per_req = num_tokens // num_reqs
        last_token_indices = torch.tensor(
            [(i + 1) * tokens_per_req - 1 for i in range(num_reqs)], dtype=torch.long
        ).to(device)

        if num_tokens == num_reqs * (1 + self.num_speculative_tokens):
            raw_sampled_cols = self.num_speculative_tokens + 1
        else:
            raw_sampled_cols = 1
        raw_sampled_token_ids = torch.ones(
            num_reqs, raw_sampled_cols, dtype=torch.int32
        ).to(device)

        return dict(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            last_token_indices=last_token_indices,
            raw_sampled_token_ids=raw_sampled_token_ids,
            **self._build_noop_async_spec_correction_kwargs(
                num_tokens, num_reqs, device=device
            ),
        )

    def warmup(
        self,
        num_tokens: int,
        num_reqs: int,
        attn_metadata: AttentionMetadata,
    ) -> None:
        assert self.model is not None
        logger.info("MTP warmup: num_tokens=%d, num_reqs=%d", num_tokens, num_reqs)
        kwargs = self._build_synthetic_inputs(num_tokens, num_reqs)
        draft_output = self.propose(
            attn_metadata=attn_metadata, is_warmup=True, **kwargs
        )
        if self.vllm_config.scheduler_config.async_scheduling:
            outputs = (
                draft_output
                if isinstance(draft_output, (tuple, list))
                else (draft_output,)
            )
            for out in outputs:
                if torch.is_tensor(out):
                    out.cpu()

    def graph_extract(
        self,
        num_tokens: int,
        num_reqs: int,
        attn_metadata: AttentionMetadata,
        device: torch.device | None = None,
    ) -> None:
        if self.capture_backend_model is None:
            logger.debug("MTP draft graph extraction skipped (no capture backend)")
            return
        logger.info("MTP graph capture: num_tokens=%d, num_reqs=%d", num_tokens, num_reqs)
        kwargs = self._build_synthetic_inputs(num_tokens, num_reqs, device=device)
        try:
            _ = self.propose(
                attn_metadata=attn_metadata,
                is_warmup=True,
                model_override=self.capture_backend_model,
                **kwargs,
            )
        except CaptureComplete:
            logger.debug(
                "MTP draft graph capture completed: num_tokens=%d, num_reqs=%d",
                num_tokens,
                num_reqs,
            )

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """Single draft layer (layer 78) → trivially one group. Copied from
        eagle.py:278-291; the check passes for a single-layer draft."""
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert (
            len({kv_cache_groups[layer_name] for layer_name in self.attn_layer_names})
            == 1
        ), "All MTP draft layers should belong to the same kv cache group"

    def compile_and_load_draft_model(
        self,
        start_layer_idx: int,
        target_hidden_size: int,
    ) -> nn.Module:
        # MTP delta (see module docstring #3): hard-set the draft arch. vLLM's
        # hf_config_override rewrote the raw glm_moe_dsa draft arch to
        # "DeepSeekMTPModel", so we cannot derive the name from
        # draft_model_config.architecture.
        draft_model_arch = "Glm52MtpForCausalLM"

        vllm_neuron_models = dict(get_models())
        if draft_model_arch not in vllm_neuron_models:
            raise ValueError(
                f"MTP draft architecture '{draft_model_arch}' not found in vLLM Neuron "
                f"model registry. Supported: {list(vllm_neuron_models.keys())}"
            )
        model_cls = vllm_neuron_models[draft_model_arch]

        draft_hf_config = self.draft_model_config.hf_config
        draft_hf_config.unpadded_hidden_size = draft_hf_config.hidden_size
        if target_hidden_size != draft_hf_config.hidden_size:
            draft_hf_config.hidden_size = target_hidden_size

        on_device_sampling_config = None
        if self.on_device_sampling:
            on_device_sampling_config = OnDeviceSamplingConfig(all_greedy=True)

        # The draft head reuses the target's FP8/EP config (self-speculation on the
        # SAME base checkpoint). The neuron_config lives in additional_config, NOT as
        # an attribute of model_config — mirror the runner (neuron_model_runner.py:448,
        # NeuronConfig.from_dict(additional_config["neuron_config"])). Rebuild it so the
        # draft's quantization ("fp8_fwd") + ep_degree (8) route the factory to the FP8
        # head loader. Overriding on_device_sampling_config forces greedy drafting.
        target_neuron_dict = dict(
            self.vllm_config.additional_config.get("neuron_config", {})
        )
        neuron_config = NeuronConfig.from_dict(target_neuron_dict)
        neuron_config.on_device_sampling_config = on_device_sampling_config

        cpu_compile = envs.VLLM_NEURON_CPU_COMPILE

        with torch.device("meta"):
            self.model = model_cls.from_configs(
                config=draft_hf_config,
                start_layer_idx=start_layer_idx,
                neuron_config=neuron_config,
            )
        self.model.num_speculative_tokens = self.num_speculative_tokens

        load_start = time.perf_counter()
        if not cpu_compile:
            self.model.load_weights(
                self.speculative_config.model,
                self.device,
                self.vllm_config.load_config.download_dir,
            )
            logger.info("Moving MTP draft model to device: %s", self.device)
            self.model = self.model.to(self.device)

            draft_model_name = self.speculative_config.model
            MODEL_LOAD_TIME.labels(model_name=draft_model_name).set(
                time.perf_counter() - load_start
            )
            MODEL_LOAD_SIZE.labels(model_name=draft_model_name).set(
                sum(p.nbytes for p in self.model.parameters())
            )
        else:
            if hasattr(self.model, "load_weights_lite"):
                self.model.load_weights_lite(
                    self.speculative_config.model,
                    torch.device("cpu"),
                    self.vllm_config.load_config.download_dir,
                )
            logger.info("CPU Compilation enabled. Skipping full MTP weight loading.")
            self.model = self.model.to("meta")

        debug_mode = envs.VLLM_NEURON_DEBUG_MODE
        fullgraph_enabled = not debug_mode

        logger.info("Compiling MTP draft model with vllm_neuron backend")
        compile_backend = envs.get_compile_backend_name()

        # NOTE: the draft decode-graph DGE indirect-DMA over-read is addressed by
        # the trailing guard blocks on the draft KV cache (see bind_kv_cache), NOT by
        # compiler flags — neuronxcc 2.25.3371 exposes no CLI flag to disable DGE on
        # indirect DMA (only a bare `--enable-dge`; the granular
        # `--enable-dge-on-indirect-dma` is a Tensorizer-internal default that
        # hlo2penguin rejects on the command line). So the draft uses the standard
        # compile path (default options), same as the target.

        cpu_mode = envs.VLLM_NEURON_CPU_MODE
        eager_mode = self.vllm_config.model_config.enforce_eager
        skip_graph_capture_backend = envs.VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND
        tensor_capture_enabled = bool(
            self.vllm_config.additional_config.get("neuron_config", {}).get(
                "tensor_capture"
            )
        )
        if (
            eager_mode
            or cpu_mode
            or debug_mode
            or skip_graph_capture_backend
            or tensor_capture_enabled
        ):
            self.capture_backend_model = None
        else:
            self.capture_backend_model = torch.compile(
                self.model,
                backend="vllm_neuron_graph_capture",
                fullgraph=fullgraph_enabled,
            )

        return torch.compile(
            self.model,
            backend=compile_backend,
            fullgraph=fullgraph_enabled,
        )

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        attn_metadata: AttentionMetadata,
        raw_sampled_token_ids: torch.Tensor,
        prev_sampled_token_ids: torch.Tensor | None = None,
        prev_num_draft_tokens: torch.Tensor | None = None,
        req_indices_per_token: torch.Tensor | None = None,
        is_warmup: bool = False,
        model_override: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the γ=1 draft propose. Copied from eagle.py:412-519; the only
        behavioural difference is downstream in the draft model (single hidden,
        no recurrent loop). Returns ``(draft_token_ids [bs,1+γ], drafts_only [bs,γ])``.
        """
        assert self.model is not None
        model = model_override if model_override is not None else self.model

        target_device = (
            torch.device("meta")
            if target_token_ids.device.type == "meta"
            else self.device
        )

        batch_size = last_token_indices.shape[0]
        num_tokens = target_token_ids.shape[0]
        if num_tokens == 0:
            return torch.empty(
                (batch_size, self.num_speculative_tokens),
                device=target_token_ids.device,
                dtype=torch.int32,
            )

        # Shift input ids by one token (next-token patching is on-device).
        if num_tokens > 1:
            shifted = target_token_ids.narrow(0, 1, num_tokens - 1)
            last_token = target_token_ids.narrow(0, num_tokens - 1, 1)
            input_ids = torch.cat([shifted, last_token])
        else:
            input_ids = target_token_ids.clone()

        input_ids = input_ids.to(target_device)
        target_positions = target_positions.to(target_device)
        last_token_indices = last_token_indices.to(target_device)
        raw_sampled_token_ids = raw_sampled_token_ids.to(target_device)
        target_hidden_states = target_hidden_states.to(target_device)
        if prev_sampled_token_ids is None:
            correction_kwargs = self._build_noop_async_spec_correction_kwargs(
                num_tokens, batch_size, device=target_device
            )
            prev_sampled_token_ids = correction_kwargs["prev_sampled_token_ids"]
            prev_num_draft_tokens = correction_kwargs["prev_num_draft_tokens"]
            req_indices_per_token = correction_kwargs["req_indices_per_token"]
        else:
            prev_sampled_token_ids = prev_sampled_token_ids.to(target_device)
            assert prev_num_draft_tokens is not None
            assert req_indices_per_token is not None
            prev_num_draft_tokens = prev_num_draft_tokens.to(target_device)
            req_indices_per_token = req_indices_per_token.to(target_device)

        first_attn_metadata = {ln: attn_metadata[ln] for ln in self.attn_layer_names}

        draft_model_name = self.speculative_config.model
        model_forward_start = time.perf_counter()
        with (
            contextlib.nullcontext()
            if is_warmup
            else model_forward_context(self.vllm_config)
        ):
            rank_tensor = self.rank_tensor.to(target_device)
            draft_token_ids, drafts_only, _ = model(
                input_ids=input_ids,
                positions=target_positions,
                initial_target_hidden_states=target_hidden_states,
                attn_metadata=first_attn_metadata,
                sampling_positions=last_token_indices,
                rank=rank_tensor,
                raw_sampled_token_ids=raw_sampled_token_ids,
                prev_sampled_token_ids=prev_sampled_token_ids,
                prev_num_draft_tokens=prev_num_draft_tokens,
                req_indices_per_token=req_indices_per_token,
            )
        model_forward_elapsed = time.perf_counter() - model_forward_start
        bucket_name = f"mtp_draft_fused_s{num_tokens}"
        if is_warmup:
            COMPILATION_TIME.labels(
                model_name=draft_model_name, bucket_name=bucket_name
            ).set(model_forward_elapsed)
        else:
            NEFF_EXECUTION_COUNT.labels(
                model_name=draft_model_name, bucket_name=bucket_name
            ).inc()

        return draft_token_ids, drafts_only
