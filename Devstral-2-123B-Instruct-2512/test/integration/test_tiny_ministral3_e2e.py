# SPDX-License-Identifier: Apache-2.0
"""E2E inference smoke test for a tiny Ministral3 model on the NKI CPU simulator.

Validates the full vLLM pipeline (model load -> forward pass -> NKI kernel
dispatch -> sampling -> output) for the Ministral3 architecture
(Devstral-2-123B-Instruct-2512 family).

Ministral3 is architecturally a dense GQA decoder identical in structure to
Llama, differing only in RoPE (YaRN) and FP8 weights. transformers 4.x has no
``Ministral3ForCausalLM`` class, so we build a tiny random-weight BF16 checkpoint
with transformers' Llama and then patch ``config.json`` to advertise
``model_type=ministral3`` + ``architectures=[Ministral3ForCausalLM]`` and a YaRN
``rope_parameters`` block. vLLM then routes to our native Neuron implementation.

This is a BF16 smoke test (no FP8): it exercises the model logic, weight
loading, RoPE, and kernel dispatch. FP8 dequant is covered separately once real
weights are available.

Run with:
    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python3 -m pytest test/vllm_neuron/model/ministral3/tiny/test_tiny_ministral3_e2e.py -v
"""

import json
import os

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from vllm import LLM, SamplingParams

pytestmark = pytest.mark.fast

# Smallest config exercising ALL NKI kernels (prefill + decode) at TP=1 and TP=2.
# Mirrors the Llama tiny config constraints:
#   kv_heads=2 so TP=2 gives 1/rank (decode megakernel requires kv_heads==1/rank)
#   hidden_size % 256 == 0 (MLP decode grid)
#   head_dim (hidden/n_heads) <= 128
#   ceil(intermediate/512) <= 8 (MLP PSUM banks)
# Architecture-shaped like Ministral3 (untied embeddings, SiLU, RMSNorm, GQA).
TINY_BASE_CONFIG = LlamaConfig(
    vocab_size=256,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=4096,
    rms_norm_eps=1e-5,
    rope_theta=1000000.0,
    tie_word_embeddings=False,
    torch_dtype="bfloat16",
)

# YaRN rope params shaped like the real Devstral config (small factor to stay
# numerically tame on a tiny model).
_TINY_ROPE_PARAMETERS = {
    "rope_type": "yarn",
    "factor": 8.0,
    "beta_fast": 4.0,
    "beta_slow": 1.0,
    "original_max_position_embeddings": 512,
    "mscale": 1.0,
    "mscale_all_dim": 0.0,
    "rope_theta": 1000000.0,
}


@pytest.fixture(scope="module")
def tiny_ministral3_dir(tmp_path_factory):
    """Create a random-weight tiny Ministral3-shaped model (BF16) once per module.

    Built with transformers' Llama (same dense GQA structure), then config.json
    is patched to advertise the Ministral3 architecture + YaRN rope_parameters so
    vLLM routes to the native Neuron Ministral3 implementation.
    """
    model_dir = str(tmp_path_factory.mktemp("tiny_ministral3"))
    torch.manual_seed(42)
    model = LlamaForCausalLM(TINY_BASE_CONFIG).to(torch.bfloat16)
    model.save_pretrained(model_dir)

    # Patch config.json: model_type + architecture name + YaRN rope params.
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["model_type"] = "ministral3"
    cfg["architectures"] = ["Ministral3ForCausalLM"]
    cfg["rope_parameters"] = _TINY_ROPE_PARAMETERS
    cfg.pop("rope_scaling", None)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    return model_dir


def _run_inference(model_dir, tp_size):
    """Run full pipeline: model load, prefill, decode, sampling."""
    llm = LLM(
        model=model_dir,
        max_num_seqs=1,
        max_model_len=128,
        block_size=128,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        num_gpu_blocks_override=4,
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [16, 128],
            }
        },
    )

    prompt_token_ids = list(range(1, 11))  # 10 tokens
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_token_ids}],
        SamplingParams(temperature=0.0, max_tokens=3),
    )

    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
    assert len(outputs[0].outputs[0].token_ids) > 0, "Expected non-empty token output"


def test_tp2(tiny_ministral3_dir):
    """TP=2: distributed weight loading, multi-worker NKI kernel dispatch."""
    _run_inference(tiny_ministral3_dir, tp_size=2)


def test_tp1(tiny_ministral3_dir):
    """TP=1: single-worker full pipeline."""
    _run_inference(tiny_ministral3_dir, tp_size=1)
