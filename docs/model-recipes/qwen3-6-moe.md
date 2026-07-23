# Qwen3.6-35B-A3B (MoE) Model Recipe

<!-- meta: description: Model recipe for deploying Qwen3.6-35B-A3B with vLLM on
Neuron, including the supported checkpoint, feature support, accuracy results,
throughput, and a link to the end-to-end deployment tutorial for the text-only
hybrid Mixture-of-Experts model on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3.6, Qwen3.6-35B-A3B, qwen3_5_moe, MoE,
GatedDeltaNet, hybrid, linear attention, model recipe, model card, LLM serving,
Trn2, Trainium -->
<!-- meta: date_updated: 2026-07-23 -->
<!-- Content type: model-card -->

## Introduction

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) is a hybrid
Mixture-of-Experts (MoE) language model from the Qwen team. Its 40 decoder layers
follow a 3:1 pattern of **30 GatedDeltaNet (linear-attention) layers + 10
full-attention layers**, and every token is routed through a 256-expert top-8 MoE
plus a shared expert (35B total parameters, ~3B active per token).

Qwen3.6-35B-A3B is supported for inference serving with
[vLLM](https://github.com/vllm-project/vllm) using the Neuron SDK on AWS
Trainium2 (`trn2`) hardware. This is a **text-only** deployment: the native
checkpoint ships a `Qwen3_5MoeForConditionalGeneration` (multimodal) architecture,
and the Neuron path loads its text decoder via the `Qwen3_5MoeForCausalLM` alias.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Qwen3.6-35B-A3B | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | Trn2 | BF16 |

> The checkpoint is served text-only. Pass
> `--hf-overrides '{"architectures":["Qwen3_5MoeForCausalLM"]}'` to load the
> text decoder and `--limit-mm-per-prompt '{"image":0,"video":0}'` to skip the
> vision tower. Omitting the override forces the multimodal config path (fp32 SSM
> cache), which raises the mamba block size and can trigger a compile-time OOM.

## Features

Per-model feature availability for Qwen3.6-35B-A3B. See the
[features guide](../guides/features-guide.md) for configuration details and the
cross-model feature compatibility matrix.

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| | Vision (image / video) | ❌ (text-only port) |
| **Quantization** | BF16 weights | ✅ |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Expert parallelism (EP) | ✅ |
| | Pipeline parallelism (PP) | ❌ |
| **Performance** | Continuous batching | ✅ |
| | Segmented GDN prefill | ✅ |
| | On-device sampling (greedy) | ✅ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (testing) | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Qwen3.6-35B-A3B
- ❌ Not supported: may be considered for future releases

Tensor parallelism is used at `tensor_parallel_size=8` with expert parallelism
(`ep_degree=8`, 32 experts per rank) on a `trn2.48xlarge`, matching the
configuration used throughout the [tutorial](../tutorials/tutorial-qwen3-6-moe.md).
`tensor_parallel_size` must divide the 16 full-attention heads (maximum valid
value is 8).

The 30 GatedDeltaNet layers use a **segmented sequential scan** for prefill. For
sequences of 512 tokens or longer this bounded-graph prefill is required; the
unsegmented monolithic prefill trips a scatter/gather out-of-bound access at long
sequence length. Keep `max_model_len`, `max_num_batched_tokens`, and
`kv_segment_size_buckets` at 512 or below so no long-extent prefill graph is
built.

## Accuracy Evaluation

**Benchmark:** GSM8K (grade-school math word problems), exact-match on the final
answer. Measured on real hardware (`trn2.48xlarge`, TP8/EP8, BF16) with greedy
on-device sampling.

| Metric | Qwen3.6-35B-A3B, Neuron Trn2 BF16 |
|--------|:---------------------------------:|
| GSM8K exact-match (batch size 1) | 95.0% |

**Reproduce:** Serve the checkpoint following the
[tutorial](../tutorials/tutorial-qwen3-6-moe.md), then run the GSM8K task from
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
against the running server over its OpenAI-compatible endpoint. Keep the few-shot
prompt plus generation length within the 1024-token context window (e.g. 4-shot,
≤ 448 generated tokens) so requests are not rejected for exceeding
`max_model_len`.

## Performance

Output throughput measured on a `trn2.48xlarge` (TP8/EP8, BF16, greedy) —
single batch offline with `vllm bench throughput`, multi-batch online with
`vllm bench serve`. Throughput scales with batch size; time-per-output-token
stays flat, indicating near-linear batch scaling. On the recommended
`max_model_len=1024` recipe, KV-cache utilization stays low (≤ 6%) even at
batch size 8.

| Configuration | Batch size | Output throughput | KV usage |
|---------------|:----------:|:-----------------:|:--------:|
| `max_model_len=1024`, offline | 1 | ~103 tok/s | — |
| `max_model_len=1024`, `kv_segment_size=512`, online | 4 | ~59 tok/s | ~1% |
| `max_model_len=1024`, `kv_segment_size=512`, online | 8 | ~131 tok/s | ~6% |

Per-request decode is approximately 21 tok/s at batch size 1, with a flat
time-per-output-token of roughly 55 ms across batch sizes 1, 4, and 8.

> **Multi-batch serving: use `max_model_len=1024` with `kv_segment_size=512`.**
> This is the verified multi-batch recipe (batch sizes 4 and 8 both run cleanly).
> Do **not** run batch size > 1 with `max_model_len=512`: the tight single
> bucket makes each request reserve its full worst-case KV allocation, so the
> hybrid (attention + GatedDeltaNet) unified block pool saturates at ~3
> concurrent decodes. A fourth request then cannot be admitted and cannot
> preempt, and once a running decode crosses the 256-token attention block
> boundary the scheduler returns empty batches indefinitely (an admission-control
> limitation for the hybrid pool, tracked separately). `max_model_len=512` is
> fine for batch size 1.

> Use `vllm bench serve` (online) for batch sizes greater than 1. Any change to
> the segmentation, sequence-length, or bucket configuration changes the traced
> graph and triggers a full cold recompile, so freeze one recipe rather than
> sweeping bucket sizes.

## Tutorials

- [Tutorial: Deploy Qwen3.6-35B-A3B with vLLM Neuron](../tutorials/tutorial-qwen3-6-moe.md)
  — End-to-end text-only deployment recipe (environment setup, model download,
  online serving, and offline inference) on a `trn2.48xlarge`.
