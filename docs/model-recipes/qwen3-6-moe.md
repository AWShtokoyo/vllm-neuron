# Qwen3.6-35B-A3B (MoE) Model Recipe

<!-- meta: description: Model recipe for deploying Qwen3.6-35B-A3B with vLLM on
Neuron, a hybrid Mixture-of-Experts model served in BF16 or FP8, with image and video
input, on a single Trn2 device. The default recipe runs without expert parallelism.
Pointer to the consolidated Qwen3.6-35B-A3B port README, which is the single source of
truth for architecture, configuration, verification, and TP=4 performance on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3.6, Qwen3.6-35B-A3B, qwen3_5_moe, MoE,
mixture of experts, expert parallelism, FP8, vision, image, video, multimodal,
GatedDeltaNet, hybrid, linear attention, chunked prefill, long context, model recipe,
model card, LLM serving, Trn2, Trainium -->
<!-- meta: date_updated: 2026-09-07 -->
<!-- Content type: model-card -->

> **The full model recipe for [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) lives in the port's consolidated README, which is the
> single source of truth:**
>
> **➜ [`Qwen3.6-35B-A3B/README.md`](https://github.com/htokoyo/vllm-neuron/tree/add-qwen36-moe/Qwen3.6-35B-A3B/README.md)**

See also the [deployment tutorial](../tutorials/tutorial-qwen3-6-moe.md).
