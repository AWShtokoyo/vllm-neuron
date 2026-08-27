# Tutorial: Deploy Qwen3.8-27B with vLLM Neuron (BF16 and FP8)

<!-- meta: description: End-to-end tutorial for deploying the Qwen3.8-27B hybrid
dense model with vLLM on Neuron, in BF16 and in per-channel ROW FP8, with text and
vision (image / video) input. Pointer to the
consolidated Qwen3.8-27B port README, which is the single source of truth for setup,
serving, long context, sampling limitations, verification, and performance at TP=4 on
Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3.8, Qwen3.8-27B, Qwen3.8-27B-FP8,
qwen3_5_dense, dense, FP8, quantization, GatedDeltaNet, hybrid, vision, multimodal,
image, video, long context, chunked prefill, tutorial, LLM serving, Trn2, Trainium -->
<!-- meta: date_updated: 2026-08-26 -->
<!-- Content type: procedural-tutorial -->

> **The full end-to-end walkthrough of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) and [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) lives in the port's consolidated README, which is the
> single source of truth:**
>
> **➜ [`Qwen3.8-27B/README.md`](https://github.com/htokoyo/vllm-neuron/tree/add-qwen38-27b/Qwen3.8-27B/README.md)**

See also the [model recipe / card](../model-recipes/qwen3-8-27b.md).
