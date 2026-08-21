# Devstral-2-123B-Instruct-2512 (Ministral3) Model Recipe

<!-- meta: description: Model recipe for deploying
mistralai/Devstral-2-123B-Instruct-2512 with vLLM on Neuron, a 123B FP8
dense-GQA causal LM. Pointer to the consolidated Devstral-2-123B-Instruct-2512
port README, which is the single source of truth for architecture, the two
required correctness knobs, configuration, verification, and TP=8 / TP=8xDP=8
performance on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Devstral, Ministral3, FP8, static FP8, E4M3,
YaRN RoPE, GQA, coding model, automatic prefix caching, model recipe, model card,
Trn2, Trainium -->
<!-- meta: date_updated: 2026-08-21 -->
<!-- Content type: model-card -->

> **The full model recipe for [`mistralai/Devstral-2-123B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512) lives in the port's consolidated README, which is the
> single source of truth:**
>
> **➜ [`Devstral-2-123B-Instruct-2512/README.md`](https://github.com/AWShtokoyo/vllm-neuron/tree/add-devstral-ministral3/Devstral-2-123B-Instruct-2512/README.md)**

🔴 Two environment knobs are **required for correctness** on this model —
`MINISTRAL3_FP8_LAYERS="mlp:0-1,3-87"` and `FORCE_MLP_KERNEL=cte`. Omitting
either produces wrong output that raises no kernel asserts. The README explains
both.

See also the [deployment tutorial](../tutorials/tutorial-devstral-2-123b.md).
