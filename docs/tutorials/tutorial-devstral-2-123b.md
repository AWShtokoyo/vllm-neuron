# Tutorial: Deploy Devstral-2-123B-Instruct-2512 (Ministral3) with vLLM Neuron

<!-- meta: description: End-to-end tutorial for deploying
mistralai/Devstral-2-123B-Instruct-2512 with vLLM on Neuron. Pointer to the
consolidated Devstral-2-123B-Instruct-2512 port README, which is the single
source of truth for setup, the two required correctness knobs, serving,
verification, and performance on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Devstral, Ministral3, FP8, static FP8, E4M3,
YaRN RoPE, GQA, automatic prefix caching, data parallelism, tutorial, Trn2,
Trainium -->
<!-- meta: date_updated: 2026-08-21 -->
<!-- Content type: procedural-tutorial -->

> **The full end-to-end walkthrough of [`mistralai/Devstral-2-123B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512) lives in the port's consolidated README, which is the
> single source of truth:**
>
> **➜ [`Devstral-2-123B-Instruct-2512/README.md`](https://github.com/AWShtokoyo/vllm-neuron/tree/add-devstral-ministral3/Devstral-2-123B-Instruct-2512/README.md)**

🔴 Before serving, export both required knobs —
`MINISTRAL3_FP8_LAYERS="mlp:0-1,3-87"` and `FORCE_MLP_KERNEL=cte`. Omitting
either produces wrong output that raises no kernel asserts.

See also the [model recipe / card](../model-recipes/devstral-2-123b.md).
