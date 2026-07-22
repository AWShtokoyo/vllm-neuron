# Devstral-2-123B (Ministral3, Dense FP8) Model Recipe

<!-- meta: description: Model recipe for deploying Devstral-2-123B-Instruct-2512
(Ministral3 dense GQA decoder) with vLLM on Neuron, covering supported
checkpoints, feature support, quantization modes (BF16-dequant and FP8-native),
performance, and a link to the end-to-end deployment tutorial on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Devstral, Devstral-2-123B, Ministral3, dense,
GQA, FP8, per-tensor static FP8, YaRN, model recipe, model card, LLM serving,
Trn2, Trainium -->
<!-- meta: date_updated: 2026-07-22 -->
<!-- Content type: model-card -->

## Introduction

[Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512)
is a 123B dense causal language model from Mistral AI (model_type `ministral3`,
architecture `Ministral3ForCausalLM`), instruction-tuned for coding and agentic
tasks. It is a dense GQA Transformer decoder — closely related to Llama 3, with
three differences: **YaRN** RoPE scaling, a **per-tensor static FP8 (E4M3)**
checkpoint, and untied embeddings. The published checkpoint ships FP8 weights
with per-tensor static scales — the same quantization scheme the GPU reference
deployment (`vllm serve … --tp 8`) runs.

Devstral-2-123B is supported for inference serving with
[vLLM](https://github.com/vllm-project/vllm) using the Neuron SDK on AWS
Trainium2 (`trn2`) hardware.

**Compatible model checkpoints:**

| Model | HuggingFace |
|-------|-------------|
| Devstral-2-123B-Instruct-2512 | [mistralai/Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512) |

> The checkpoint repository ships **both** the HuggingFace safetensors format
> (`model-*`) and the Mistral-native format (`consolidated-*`). Serving on Neuron
> uses the HF format — pass `--config-format hf` (and, for offline `LLM`,
> `load_format="safetensors"`) so vLLM does not fall back to its built-in
> `MistralForCausalLM`. When downloading, `--exclude "consolidated-*"` halves the
> transfer.

## Architecture

Dense GQA decoder. Key dimensions:

| Field | Value |
|---|---|
| hidden_size | 12288 |
| num_hidden_layers | 88 |
| num_attention_heads (Q) | 96 |
| num_key_value_heads (KV) | 8 (GQA) |
| head_dim | 128 |
| intermediate_size | 28672 |
| vocab_size | 131072 |
| max_position_embeddings | 262144 (256k) |
| RoPE | YaRN (θ=1e6, factor=64, orig_max=4096, β_fast=4, β_slow=1, mscale=1) |
| activation / norm | SiLU (SwiGLU) / RMSNorm (eps 1e-5), pre-attn + pre-MLP |
| attention bias / qk_norm | none |
| tie_word_embeddings | false (separate lm_head) |

> **Tensor-parallel degree must divide Q=96.** `TP=32` (half a `trn2.48xlarge`),
> `TP=16`, and `TP=8` are all valid; `TP=64` is **not** (96 is not divisible by
> 64). See the [tutorial](../tutorials/tutorial-devstral-2-123b.md) for the
> recommended layouts.

## Features

Per-model feature availability for Devstral-2-123B. See the
[features guide](../guides/features-guide.md) for configuration details.

| Category | Feature | Status |
|---|---|---|
| **Quantization** | BF16 (FP8 weights dequantized at load) | ✅ |
| | FP8-native `qkv,o_proj` (attention, BF16-token-faithful) | ✅ |
| | FP8-native `qkv,o_proj,mlp` (full FP8, −50% weight HBM) | ✅ |
| | FP8 KV cache (`kv_cache_dtype=fp8_e4m3`, scale=1.0) | ✅ |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Data parallelism (DP, whole-box throughput) | ✅ |
| | Pipeline parallelism (PP) | ❌ |
| | Context parallelism (CP) | ❌ |
| **Performance** | Single-shot prefill (`bucket == max_model_len`) | ✅ |
| | Multi-bucket prefill (opt-in, short-input TTFT) | ✅ |
| | On-device sampling (greedy, top-k, top-p) | ✅ |
| | Prefix caching (APC) | ❌ |
| | Chunked prefill (mixed batching) | ❌ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (testing) | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Devstral-2-123B
- ❌ Not supported: may be considered for future releases

## Quantization modes

The checkpoint is per-tensor static FP8. Two serving paths are supported:

- **BF16-dequant (default).** FP8 weights are dequantized to BF16 at load
  (`weight × weight_scale_inv`). Safe baseline; no HBM saving. Requires no kernel
  patch.
- **FP8-native (opt-in).** Keeps the dense projections in FP8 and runs FP8×FP8
  static matmuls on Trainium2 — the GPU-equivalent deployment. Enabled per
  projection family via `neuron_config.quantization`:
  - `"fp8:qkv,o_proj"` — attention projections in FP8, MLP stays BF16.
    **BF16-token-faithful** (first-token 30/30 vs BF16); removes attention-weight
    HBM at no quality cost.
  - `"fp8:qkv,o_proj,mlp"` (or `"fp8"`) — **full FP8**. Weight HBM **−50%**,
    decode throughput **+19%**. Carries the per-tensor-static-FP8 accuracy
    tradeoff described below.

  FP8-native requires the `integration_nkilib.patch` kernel patch applied to the
  installed `nkilib` (bundled inside `neuronx-cc`); full FP8 additionally needs
  `NKILIB_MLP_BF16_XPOSE_SRC=1` in the environment. See the
  [tutorial](../tutorials/tutorial-devstral-2-123b.md) for the exact steps.

> vLLM rejects the `fp8` quant method, so the checkpoint's `quantization_config`
> is emptied via `--hf-overrides '{"quantization_config": {}}'`; the model's
> weight loader auto-detects FP8 from the checkpoint (slice count /
> `*.weight_scale_inv`), not from the config, in both paths.

## Accuracy Evaluation

**Faithfulness — HuggingFace model-card "Tests" prompts** (greedy, Mistral chat
template). The card's bar is qualitative correctness, not BF16-token-exact match.

| Card prompt | BF16-dequant | Full FP8 (`qkv,o_proj,mlp`) |
|---|---|---|
| self-ID (who / maker / date) | Devstral-Medium, Mistral AI, 2025-12-09 ✓ | facts ✓, occasional token glitch |
| Python `capital_of_japan()` | `return "Tokyo"` ✓ | `return "Tokyo"` ✓ |
| capital of Japan / gold symbol / Mona Lisa | Tokyo / Au / Leonardo ✓ | Tokyo / Au / Leonardo ✓ |
| long open-ended (web server) | correct | occasional early derail |

Full FP8 gets the **facts and code right** but shows occasional per-token
corruption and, rarely, an early derail on long open-ended generation.

**This error is inherent to per-tensor static FP8 — it occurs on GPU too, not a
Trainium or implementation bug.** Offline emulation of the exact FP8 math (real
checkpoint scales) matches BF16 at cos ≈ 0.9999 per layer, but ~1% per-layer error
compounds across 88 layers and flips the greedy argmax on a minority of tokens.
The GPU reference runs the *same* per-tensor static FP8 checkpoint
(`weight_block_size=null`, `qscheme_act="TENSOR"`), so a GPU FP8 deployment
exhibits the same class of per-token deviation from BF16.

**Recommendation:** use `fp8:qkv,o_proj` (BF16-token-faithful, first-token 30/30)
— or `TP=32` — when byte-faithful output matters; use `fp8:qkv,o_proj,mlp` (full
FP8) where the HBM / throughput win outweighs occasional token-level deviation,
matching how the model ships for GPU.

## Performance

Measured on `trn2.48xlarge` (Neuron 2.31: vLLM 0.21 / vllm-neuron 0.21.0.1.0.0 /
neuronx-cc 2.26 / nki 0.5.0 / torch 2.11), full FP8 (`fp8:qkv,o_proj,mlp`) +
`kv_cache_dtype=fp8_e4m3`, `vllm bench serve` (random, input 1024 / output 256,
greedy).

| Config (full FP8, fp8 KV) | chips | total tok/s | TPOT |
|---|---|---|---|
| TP=32 (half box, low-latency, bs=2) | 8 | ~1,395 | ~25–30 ms |
| **TP=8 × DP=1** (one replica, 128 prompts @ conc 128) | 2 | **~1,645** | 245 ms |
| **TP=8 × DP=8** (whole box, 512 prompts @ conc 1024) | 16 | **~10,240** | 176 ms |

A single 2-chip `TP=8 × DP=1` replica reaches roughly the whole `TP=32` half-box
throughput on **1/4 the chips** — the most cost-efficient point (≈3.8× per chip).
`TP=8 × DP=8` packs the whole box. FP8-native full FP8 also lifts decode
throughput ~19% and halves weight HBM (243.6 → 121.8 GB) versus BF16-dequant.

> `TP=8 × DP=1` is pure configuration. The whole-box `data_parallel_size>1`
> variant needs two launcher fixes (a module-scope HF config for pickling across
> engine-core subprocesses, and `VLLM_WORKER_MULTIPROC_METHOD=fork`), both handled
> by the reproduction scripts. See the
> [tutorial](../tutorials/tutorial-devstral-2-123b.md).

**Multi-bucket prefill (opt-in).** A single `[2048]` prefill bucket pads every
short prompt to 2048. Adding smaller buckets lets the scheduler pick the tightest
fit, cutting TTFT on short-input bursts (whole-box DP=8, `in=128`: median TTFT
**4.4× faster**, total throughput **2.36×**). The win scales with how much shorter
the prompt is than 2048, so at `in=1024` it is effectively neutral. Opt-in via an
ascending bucket list whose last element equals `max_model_len` (keeps every
bucket single-shot). Quality is unchanged — it only selects which prefill graph
runs.

## Tutorials

- [Tutorial: Deploy Devstral-2-123B with vLLM Neuron](../tutorials/tutorial-devstral-2-123b.md)
