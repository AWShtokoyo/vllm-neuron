# Llama-Embed-Nemotron-8B (Embedding) Model Recipe

<!-- meta: description: Model recipe for deploying nvidia/llama-embed-nemotron-8b
with vLLM on Neuron, an 8B bidirectional Llama embedding (pooling) model, including
feature support, accuracy results, and a link to the end-to-end deployment tutorial
on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, llama-embed-nemotron, embedding, pooling,
sentence embedding, bidirectional Llama, retrieval, model recipe, model card,
Trn2, Trn3, Trainium -->
<!-- meta: date_updated: 2026-07-22 -->
<!-- Content type: model-card -->

## Introduction

[llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b)
is an 8B text embedding model from NVIDIA. It runs a **bidirectional** Llama-3.1-8B
backbone, then mask-weighted mean-pools the final hidden states and L2-normalizes
them to produce a 4096-dimensional sentence embedding. It targets retrieval,
semantic search, clustering, and reranking workloads.

Unlike a generative LLM, this is a **pooling** model: it is prefill-only (no decode,
no KV cache, no sampler) and is served with `--runner pooling`, returning an
embedding vector per input rather than generated tokens.

llama-embed-nemotron-8b is supported for inference serving with
[vLLM](https://github.com/vllm-project/vllm) using the Neuron SDK on AWS Trainium2
(`trn2`).

**Compatible model checkpoints:**

| Model | HuggingFace |
|-------|-------------|
| Llama-Embed-Nemotron-8B | [nvidia/llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b) |

The checkpoint uses `model_type` `llama_bidirec` (architecture
`LlamaBidirectionalModel`). It shares the Llama-3.1-8B dense GQA backbone
(hidden_size 4096, 32 layers, 32/8 GQA heads, head_dim 128, intermediate_size
14336, vocab 128256, llama3 RoPE scaling) with three embedding-specific
differences: bidirectional (non-causal) attention, a mask-weighted mean-pool + L2
head instead of an `lm_head`, and no KV cache (`use_cache=false`).

## Features

Per-model feature availability for llama-embed-nemotron-8b. See the
[features guide](../guides/features-guide.md) for configuration details.

| Category | Feature | Status |
|---|---|---|
| **Task** | Embedding / pooling (`--runner pooling`) | ✅ |
| | Text generation / decode | ❌ (embedding model, prefill-only) |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor parallelism (TP=1/2/4) | ✅ |
| | Pipeline parallelism (PP) | ❌ |
| | Context parallelism (CP) | ❌ |
| **Performance** | Batched prefill packing (`VLLM_NEURON_POOLING_PACK`) | ✅ |
| | On-device pooling gather (`VLLM_NEURON_POOLING_ONDEVICE_GATHER`) | ✅ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (testing) | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for llama-embed-nemotron-8b
- ❌ Not supported: not applicable to this embedding model or a future consideration

**Attention.** The model uses bidirectional (non-causal) attention. Per-query
`key_bounds` restrict each query to its own sequence's KV range so real queries do
not attend to padding, and — when multiple sequences are packed into one prefill —
do not attend across sequence boundaries.

**Tensor parallelism.** TP=1, TP=2, and TP=4 are all validated. TP=2 scales
super-linearly for throughput (2× compute plus the bf16 weights split across two
cores doubles per-core HBM bandwidth), and TP=4 maximizes absolute throughput and
minimizes latency. At **TP=1** set `VLLM_NEURON_FORCE_LNC1=1` (a codegen workaround
for the all-PyTorch graph; it does not change HBM usage). **Drop
`VLLM_NEURON_FORCE_LNC1` at TP≥2** — otherwise the compiled collectives fail to
load.

**Prefill bucket sizes must be greater than 128** (`num_batched_tokens_buckets`).

## Accuracy Evaluation

Correctness is validated as **embedding equivalence** against the official
HuggingFace `LlamaBidirectionalModel` reference with sentence-transformers
mean-pool + L2, measured as cosine similarity (higher is better; `1.0` is
identical).

| Test | Result |
|---|---|
| CPU equivalence (full vLLM pooling path, tiny weights) | worst cos = 0.999962 |
| On-device equivalence (trn2 TP=1, real 8B) | worst cos = 0.999961 |
| Batched on-device (4 seqs / 1 prefill, vs solo) | cos = 1.000000 (no cross-sequence leak) |
| vLLM serving `/v1/embeddings` vs HF reference | worst cos = 0.999962 |
| Cross-prompt similarity (discriminative check) | ~0.28–0.30 (distinct prompts stay distinct) |

> **Reference build note.** On the Neuron 2.31 stack (transformers ≥ 5.13) the HF
> *reference* class must be built on a **transformers-4.x** venv — the equivalence
> tests are split into a reference-build stage (CPU, 4.x) and an on-device compare
> stage. The port itself is bidirectional and correct on the current stack; only
> the upstream HF reference class breaks on transformers ≥ 5.

## Tutorials

- [Tutorial: Deploy llama-embed-nemotron-8b with vLLM Neuron](../tutorials/tutorial-llama-embed-nemotron-8b.md)
</content>
</invoke>
