# Tutorial: Deploy GLM-5.2 with vLLM Neuron

<!-- meta: description: End-to-end tutorial for deploying GLM-5.2 with vLLM on
Neuron. Pointer to the consolidated GLM-5.2 port README, which is the single
source of truth for environment setup, model download, offline and online
serving, the FP8-only quantization modes and why BF16 does not fit, and
verification on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, GLM, GLM-5.2, MoE, MLA, FP8, expert
parallelism, tutorial, LLM serving, Trn2, Trainium -->
<!-- meta: date_updated: 2026-07-29 -->
<!-- Content type: procedural-tutorial -->

> **The full end-to-end walkthrough of [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8) lives in the port's consolidated README, which is the
> single source of truth:**
>
> **➜ [`GLM-5.2/README.md`](https://github.com/AWShtokoyo/vllm-neuron/tree/add-glm-5-2/GLM-5.2/README.md)**
