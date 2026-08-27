# Qwen3.8-27B (Dense, BF16 and FP8) Model Recipe

<!-- meta: description: Model recipe for deploying Qwen3.8-27B with vLLM on
Neuron, a hybrid dense model served in BF16 and in per-channel ROW FP8, with text and
vision (image / video) input. Pointer to
the consolidated Qwen3.8-27B port README, which is the single source of truth for
architecture, the FP8 quantisation path, configuration, long context, sampling
limitations, verification, and performance on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3.8, Qwen3.8-27B, Qwen3.8-27B-FP8,
qwen3_5_dense, dense, FP8, quantization, ROW scaling, GatedDeltaNet, hybrid, linear
attention, SwiGLU, vision, multimodal, image, video, long context, chunked prefill,
model recipe, model card, LLM serving, Trn2, Trainium -->
<!-- meta: date_updated: 2026-08-26 -->
<!-- Content type: model-card -->

> **The full model recipe for [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) and [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) lives in the port's consolidated README, which is the
> single source of truth:**
>
> **➜ [`Qwen3.8-27B/README.md`](https://github.com/htokoyo/vllm-neuron/tree/add-qwen38-27b/Qwen3.8-27B/README.md)**

See also the [deployment tutorial](../tutorials/tutorial-qwen3-8-27b.md).
