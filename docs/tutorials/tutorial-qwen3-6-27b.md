# Tutorial: Deploy Qwen3.6-27B with vLLM Neuron

<!-- meta: description: End-to-end tutorial for deploying the text-only
Qwen3.6-27B hybrid dense model with vLLM on Neuron. Pointer to the consolidated
Qwen3.6-27B port README, which is the single source of truth for setup, serving,
verification, and performance at TP=4 on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3.6, Qwen3.6-27B, qwen3_5_dense, dense,
GatedDeltaNet, hybrid, tutorial, LLM serving, Trn2, Trainium -->
<!-- meta: date_updated: 2026-07-25 -->
<!-- Content type: procedural-tutorial -->

> **The full end-to-end walkthrough of [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) lives in the port's consolidated README, which is the
> single source of truth:**
>
> **➜ [`Qwen3.6-27B/README.md`](https://github.com/AWShtokoyo/vllm-neuron/tree/add-qwen36-27b/Qwen3.6-27B/README.md)**

See also the [model recipe / card](../model-recipes/qwen3-6-27b.md).
