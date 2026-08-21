# Qwen3.6-27B (Dense) Model Recipe

<!-- meta: description: Model recipe for deploying Qwen3.6-27B with vLLM on
Neuron, a text-only hybrid dense model. Pointer to the consolidated Qwen3.6-27B
port README, which is the single source of truth for architecture, configuration,
the TP=4 SBUF workaround, verification, and performance on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3.6, Qwen3.6-27B, qwen3_5_dense, dense,
GatedDeltaNet, hybrid, linear attention, SwiGLU, model recipe, model card, LLM
serving, Trn2, Trainium -->
<!-- meta: date_updated: 2026-07-25 -->
<!-- Content type: model-card -->

> **The full model recipe for [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) lives in the port's consolidated README, which is the
> single source of truth:**
>
> **➜ [`Qwen3.6-27B/README.md`](https://github.com/AWShtokoyo/vllm-neuron/tree/add-qwen36-27b/Qwen3.6-27B/README.md)**

See also the [deployment tutorial](../tutorials/tutorial-qwen3-6-27b.md).
