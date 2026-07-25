# Tutorial: Deploy Devstral-2-123B with vLLM Neuron

<!-- meta: description: End-to-end tutorial for deploying Devstral-2-123B-Instruct-2512
(Ministral3 dense FP8 GQA decoder) with vLLM on Neuron. Pointer to the
consolidated Devstral-2-123B port README, which is the single source of truth for
environment setup, model download, online serving, offline inference, the
optional FP8-native kernel path, whole-box data-parallel throughput, and
verification on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Devstral, Devstral-2-123B, Ministral3, dense,
FP8, per-tensor static FP8, YaRN, tutorial, LLM serving, Trn2, Trainium -->
<!-- meta: date_updated: 2026-07-25 -->
<!-- Content type: procedural-tutorial -->

> **The full end-to-end walkthrough of [`mistralai/Devstral-2-123B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512) lives in the port's consolidated README, which is the
> single source of truth:**
>
> **➜ [`Devstral-2-123B-Instruct-2512/README.md`](https://github.com/AWShtokoyo/vllm-neuron/tree/add-devstral-ministral3/Devstral-2-123B-Instruct-2512/README.md)**
