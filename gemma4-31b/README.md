# Contributed Model: gemma4-31b

## Introduction

[Gemma 4](https://huggingface.co/google/gemma-4-31B-it) is a family of open
text-generation models developed by Google. This is the vllm-neuron
implementation of the 31B instruction-tuned variant
[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) — a
decoder-only transformer with **heterogeneous attention** (local sliding-window
layers with head_dim=256 interleaved with full-global layers with head_dim=512),
plus QK/V normalization, partial rotary embeddings on the global layers, a GeGLU
MLP, and final-logit softcapping. It is served text-only.

**Compatible model checkpoints:**

| Model | HuggingFace |
|-------|-------------|
| Gemma 4 31B IT | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) |
| Gemma 4 31B (base) | [`google/gemma-4-31B`](https://huggingface.co/google/gemma-4-31B) |

> The instruction-tuned (`-it`) checkpoint is the verified target. The pretrained
> base checkpoint (`google/gemma-4-31B`) shares the identical architecture,
> serving path, and configuration, but only the `-it` variant was verified on
> this hardware.

## Verification scope

Verified at **TP=4 and TP=8**, BF16, on the Neuron 2.31 stack (vLLM 0.21 /
vllm-neuron 0.21.0.1.0.0), with `max_model_len=2048` and multi-bucket prefill
`[512, 1024, 2048]`. Setup guidance targets `trn2.3xlarge` (4 logical
NeuronCores) because TP=4 fits there; the measurements below were taken on a
`trn2.48xlarge`, TP=4 on a 4-device slice and TP=8 on an 8-device slice.

| Configuration | Status |
|---|---|
| TP=4, BF16, `max_model_len=2048`, single-shot prefill — generation smoke | ✅ device |
| TP=4, BF16 — 3-way logit validation (FP32 → BF16 → Neuron), `<bos>`-anchored | ✅ device, aggregate σ-ratio 0.9567 |
| TP=4, BF16, NKI CTE prefill attention | ✅ device (all runs above route through it) |
| TP=4, BF16, segmented prefill `kv_segment_size_buckets: [512]` — logit validation | ✅ device, aggregate σ-ratio 0.9877 |
| TP=4, BF16, segmented prefill — 1400-token prompt (3 chunks, `cached_seq_len > 0`) | ✅ device, token-identical to unsegmented |
| TP=4, BF16 — `vllm bench serve` TTFT + throughput sweep | ✅ device |
| TP=8, BF16, uncapped KV budget — `vllm bench serve` TTFT + throughput sweep | ✅ device |
| TP=4, BF16, opt-in NKI decode kernel (`VLLM_GEMMA4_DECODE_KERNEL=1`) | ✅ device, aggregate σ-ratio 0.9236 — correct but **2.35× slower**, see below |
| TP=8, BF16, opt-in NKI decode kernel (`VLLM_GEMMA4_DECODE_KERNEL=1`) | ✅ device, aggregate σ-ratio 0.9802 — correct but **1.60× slower**, see below |

**The optional NKI decode kernel is correct on device but slower than the default
path at both tensor-parallel degrees tested, so it stays off.**
`VLLM_GEMMA4_DECODE_KERNEL=1` was verified at TP=4 and TP=8: 3-way logit validation
passes at both (aggregate σ-ratio 0.9236 at TP=4 and 0.9802 at TP=8, against 0.9567
for the default path, same goldens), and the kernel demonstrably engages on both layer
types. But the same `vllm bench serve` case is slower with it enabled at both TPs:

| | default path | kernel enabled | |
|---|---:|---:|---|
| TP=4 output tok/s / mean TPOT | 10.3 / 87 ms | **4.39 / 220 ms** | 2.35× lower throughput |
| TP=8 output tok/s / mean TPOT | 15.02 / 57.8 ms | **9.41 / 98.0 ms** | 1.60× lower throughput |

TTFT is unchanged in both comparisons (731 vs 730 ms at TP=4, 624 vs 622 ms at TP=8;
prefill runs the CTE kernel either way), which confirms only the decode path differed.
The penalty shrinks as TP grows but does not invert. The default fp32 PyTorch decode
path is therefore the recommended and the faster one, and it is what every measurement
in this document uses.

**Not verified on this hardware:**

- **The NKI decode kernel beyond that single TP=4 case** — no TP=8 leg, no
  segmented-prefill leg, and no concurrency sweep. Whether the gap narrows at higher
  concurrency is unmeasured.
- **Prompt-embedding input** (`inputs_embeds` / `is_token_ids`) and
  **`load_weights_lite`**. Both are wired to match the other in-tree models but
  were not exercised.
- **Prefix caching beyond a single repeated request.** A segmented-prefill run
  with `--enable-prefix-caching` produced output token-identical to the
  non-cached leg, but that was one request repeated twice, not a multi-request
  shared-prefix workload. The serving recipes below therefore do not enable it.
- Tensor-parallel sizes above 8. TP=16 and TP=32 are architecturally supported —
  Gemma 4 has 32 attention heads, so valid TP sizes are 1, 2, 4, 8, 16, 32 — but
  were not run.
- The pretrained base checkpoint (`google/gemma-4-31B`); only `-it` was run.
- Context lengths beyond `max_model_len=2048` (the architecture declares
  262,144).
- Vision / audio input — this port serves the text decoder only.

Every number in this document is this port's own measurement.

## Model Overview

| | |
|---|---|
| HuggingFace ID | `google/gemma-4-31B-it` (base: `google/gemma-4-31B`) |
| model_type | served text arch `Gemma4ForCausalLM` (`gemma4_text`); the checkpoint declares the multimodal arch `Gemma4ForConditionalGeneration` (`gemma4`) |
| Task | Text generation (`--runner generate`), prefill + decode |
| Params | ~31B |
| Context length | 262,144 (`max_position_embeddings`); verified `max_model_len=2048` |
| Vocabulary | 262,144 tokens; tied word embeddings |
| Tensor type | BF16 |
| License | Gemma license (Apache-2.0); public, not gated on HuggingFace |

## Model Architecture

Gemma 4 31B is a decoder-only transformer with **heterogeneous attention**: most
layers use local sliding-window attention, and a smaller set use full global
attention — the two types differ in shape (head_dim, KV heads) and RoPE.

**Core configuration**

| Field | Value |
|---|---|
| hidden_size | 5376 |
| num_hidden_layers | 60 |
| num_attention_heads | 32 |
| intermediate_size | 21504 |
| MLP activation | GeGLU (`gelu_pytorch_tanh`) |
| norm / eps | RMSNorm / 1e-6 |
| max_position_embeddings | 262,144 |
| vocab_size | 262,144 (tied embeddings) |
| attention scaling | `1.0` (no `1/sqrt(head_dim)`; QK norm handles it) |

**Heterogeneous attention layers**

| | Sliding-window layers | Global layers |
|---|---|---|
| head_dim | 256 | 512 |
| KV heads | 16 | 4 |
| Attention | Sliding window (1024) | Full causal |
| RoPE θ | 10,000 | 1,000,000 |
| RoPE coverage | 100% | 25% (partial) |
| V projection | Separate | K = V (V copies K) |

Layer pattern (60 layers): mostly sliding-window, with global layers at indices
5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55.

**Embedding-/Gemma-specific differences** (see `vllm_neuron/model/gemma4/model.py`):

- **QK normalization** — RMSNorm (learnable scale) on Q and K after projection.
- **V normalization** — RMSNorm (no learnable scale) on V.
- **layer_scalar** — learned per-layer multiplicative factor on the residual.
- **4 norms per layer** — input, post-attention, pre-feedforward,
  post-feedforward.
- **Final-logit softcapping** — `tanh(logits / 30.0) * 30.0` before sampling.
- **Scaled input embeddings** — multiply by `sqrt(hidden_size)`; tied word
  embeddings.
- **BOS anchor** — a leading `<bos>` token (id 2) is required for stable greedy
  decoding (see the [BOS note](#step-2-download-the-model) under Setup).


## Feature status

| Feature | Status | Notes |
|---|---|---|
| Text input | ✅ | Text-only serving |
| BF16 inference | ✅ | Only BF16 currently |
| Tensor parallelism (TP) | ✅ | Verified TP=4 and TP=8; 1/2/4/8/16/32 valid (32 heads) |
| Sequence parallelism (SP) | ✅ | All-gather/reduce-scatter on prefill |
| Heterogeneous attention (SWA + global) | ✅ | Per-layer head_dim via LayerSpec |
| NKI prefill attention (CTE) | ✅ | Both layer types; head_dim ≤ 512. Verified on device |
| Segmented prefill | ✅ | Sliding window + prior KV; verified on device at `kv_segment_size 512` |
| Multi-bucket prefill | ✅ | `[512, 1024, 2048]` |
| NKI decode attention | ⚠️ | Opt-in `VLLM_GEMMA4_DECODE_KERNEL=1`. Verified correct on device at TP=4 and TP=8 but slower than the default path at both (2.35× / 1.60×) — leave it off |
| Prompt embeddings (`inputs_embeds`) | ⚠️ | Wired; not verified on device |
| On-device sampling (greedy, top-k, top-p) | ✅ | |
| Logit softcapping | ✅ | `tanh(x/30)*30` before sampling |
| Tied embeddings | ✅ | lm_head shares embed_tokens |
| torch.compile (XLA) | ✅ | Full-graph compilation |
| Pipeline / context parallelism | ❌ | |
| Vision encoder | ❌ | Text decoder only (vision weights skipped) |
| MXFP4 / FP8 KV cache | ❌ | Not implemented |



## Setup

### Native integration (nothing to apply)

The model is **natively integrated** into this repository — checkout the branch
and it works; you do not need to apply any patch.

- **Model package:** [`vllm_neuron/model/gemma4/`](../vllm_neuron/model/gemma4/).
  Served text-only under the transformers-5.x native TEXT architecture
  `Gemma4ForCausalLM` (`model_type` `gemma4_text`). The checkpoint declares the
  multimodal arch `Gemma4ForConditionalGeneration` (`model_type` `gemma4`);
  `vllm_neuron` rewrites it to the text arch at config registration so vLLM
  routes it down the text path (vision/audio encoders are not served).
- **Registration touch-points** — `vllm_neuron/model/registry.py` and the
  `gemma4` HuggingFace `AutoConfig` registration in `vllm_neuron/__init__.py` —
  are committed directly on this branch.

**Prerequisites:**

- A `trn2` instance with Neuron SDK `2.31` or later. Verified on `trn2.3xlarge`
  (4 NeuronCores, TP=4). See [setup guide](../docs/getting-started/setup-guide.md).
- vLLM Neuron plugin `0.21.0.1.0.0` or above installed.
- Python 3.10+.
- The `google/gemma-4-31B-it` weights. The repository is public and not gated, so
  no HuggingFace token is required to download it; the weights are governed by the
  Gemma license (Apache-2.0).

### Step 1: Environment Setup

Verify Neuron devices are visible:

```bash
neuron-ls
# Expected: 4 NeuronCores listed for trn2.3xlarge
```

Set environment variables before running any inference / compile:

```bash
# Extend timeouts for large-model compilation
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_NEURON_COMPILATION_TIMEOUT=1200

# Serial compile/trace — avoid host OOM (kernel kill) on trn2.3xlarge
export VLLM_NEURON_PARALLEL_COMPILE_WORKERS=1
export VLLM_NEURON_PARALLEL_TRACE_WORKERS=1

# Cap the KV budget so 31B weights (~15 GiB/core) + KV fit 24 GiB/core at TP=4.
# Required at TP=4; leave it unset at TP>=8, where the weights-per-core drop
# makes the default budget fit.
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.08

# Required if your home directory is on NFS. Note that NEURON_CC_FLAGS REPLACES
# the framework's composed compiler-flag string rather than appending to it, so
# set it only when you need to and check the resulting command line. To relocate
# compiler scratch and NEFF cache without touching the flags, prefer
# VLLM_CACHE_ROOT.
export NEURON_CC_FLAGS="--temp-dir=/tmp/neuroncc_tmp"; mkdir -p /tmp/neuroncc_tmp
```

### Step 2: Download the Model

```bash
# `huggingface-cli` no longer works in the huggingface_hub 1.x this stack pins.
hf download google/gemma-4-31B-it --local-dir /path/to/gemma-4-31B-it
```

> **Tip:** On a `trn2` cluster, download to a shared filesystem instead of your
> home directory to avoid NFS write issues.

> **Tokenizer note.** Some Gemma 4 checkpoint revisions ship
> `extra_special_tokens` in `tokenizer_config.json` as a list, while
> transformers 4.5x expects a dict. If tokenizer loading fails, edit
> `tokenizer_config.json` in your local model directory and replace the
> `extra_special_tokens` list with an empty object `{}` for text-only serving.

> **BOS note (important).** Gemma uses a leading `<bos>` token (id 2) as an
> attention anchor; without it, greedy decoding can degenerate into a repetition
> loop. The `google/gemma-4-31B-it` chat template already prepends `<bos>`, so
> the normal `apply_chat_template` path is correct. If you feed **raw completion
> prompts** and the checkpoint's fast tokenizer does not add BOS, prepend it
> explicitly (e.g. `TokensPrompt(prompt_token_ids=[2] + tok.encode(prompt,
> add_special_tokens=False))`).

## Serving

### Verified — offline `LLM` API (TP=4, `trn2.3xlarge`)

This is the configuration verified for this port. The ready-to-run example
[`examples/vllm_neuron/models/gemma4/run.py`](../examples/vllm_neuron/models/gemma4/run.py)
defaults to it (and prepends `<bos>`):

```bash
python examples/vllm_neuron/models/gemma4/run.py \
    --model-checkpoint /path/to/gemma-4-31B-it
```

Equivalent direct `LLM` call:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/gemma-4-31B-it",
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=4,
    tensor_parallel_size=4,
    additional_config={
        "neuron_config": {
            "quantization": "bf16",
            "num_batched_tokens_buckets": [512, 1024, 2048],  # multi-bucket prefill
            "num_seqs_buckets": [4],
            "on_device_sampling_config": {"all_greedy": True},
        },
    },
)
outputs = llm.generate(["The capital of France is "],
                       SamplingParams(max_tokens=50, temperature=0.0))
print(outputs[0].outputs[0].text)
```

With the multi-bucket schedule `[512, 1024, 2048]` each request lands in its
smallest fitting NEFF, so a short prompt does not pay a full-bucket TTFT.

### Online serving (OpenAI-compatible)

```bash
vllm serve /path/to/gemma-4-31B-it \
    --served-model-name gemma-4-31B-it \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --tensor-parallel-size 4 \
    --no-enable-prefix-caching \
    --additional-config '{"neuron_config": {"quantization": "bf16", "num_batched_tokens_buckets": [512, 1024, 2048], "num_seqs_buckets": [4], "on_device_sampling_config": {"all_greedy": true}}}'
```

The first launch compiles one NEFF per bucket (several minutes each). Watch for
`Application startup complete.`

**Configuration constraints:**

- `max-num-batched-tokens` must be one of the bucket values or equal
  `max-model-len`, and the last entry of `num_batched_tokens_buckets` must equal
  `max-num-batched-tokens`.
- `--no-enable-prefix-caching` is **required** with this recipe. vLLM 0.21
  enables prefix caching by default, and the backend requires segmented prefill
  whenever prefix caching is on; here `max-num-batched-tokens` equals
  `max-model-len`, so prefill is single-shot and segmented prefill stays off.
  Without the flag every rank fails at worker init with `ValueError: Automatic
  Prefix Caching (APC) requires segmented prefill to be enabled`. The
  offline `LLM(...)` example above passes `enable_prefix_caching=False`
  for the same reason. To use prefix caching instead, set
  `--max-num-batched-tokens` to a segmented-prefill size smaller than
  `max-model-len` (512, 1024, 2048, 4096 or 8192) so segmentation auto-enables.
- When `kv_segment_size_buckets` is set explicitly, `num_batched_tokens_buckets`
  must be **equal** to it, not merely compatible.

**TP=8 on a `trn2.48xlarge`** uses the identical command with
`--tensor-parallel-size 8` and `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` left
unset — the cap is a TP=4 necessity, not a general one. This configuration is
verified on device and is the faster of the two on every metric (see
[Measured performance (TP=8)](#measured-performance-tp8-bf16)).

For still larger deployments (e.g. TP=32, `max_model_len=4096`, buckets
`[512, 1024, 2048, 4096]`) the same config shape applies with a larger KV budget;
this was not verified on this hardware.

### Segmented prefill

Segmented prefill splits a long prompt into fixed-size chunks and attends each
chunk against the KV already written by the earlier chunks. Enable it by setting
`kv_segment_size_buckets`, keeping `num_batched_tokens_buckets` equal to it:

```bash
vllm serve /path/to/gemma-4-31B-it \
    --served-model-name gemma-4-31B-it \
    --max-model-len 2048 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 4 \
    --tensor-parallel-size 4 \
    --no-enable-prefix-caching \
    --additional-config '{"neuron_config": {"quantization": "bf16", "kv_segment_size_buckets": [512], "num_batched_tokens_buckets": [512], "num_seqs_buckets": [4], "on_device_sampling_config": {"all_greedy": true}}}'
```

Verified on device at `kv_segment_size 512`: a 1400-token prompt (3 chunks, so
chunks 2 and 3 attend against prior KV) produced greedy output token-identical to
the same prompt served with single-shot prefill. Note that setting
`kv_segment_size_buckets` alone does not exercise the prior-KV path — a prompt
shorter than one segment yields `cached_seq_len == 0`, and the gathered prior KV
is then fully masked. Use a prompt longer than the segment size when validating
this path.


## Accuracy Evaluation

Use the repository's generic accuracy scripts in
[`examples/vllm_neuron/accuracy/`](../examples/vllm_neuron/accuracy/), which
compare HuggingFace CPU goldens (FP32 → BF16) against vLLM-on-Neuron for any
text causal LM. Point them at gemma4 with `--model` and `--tp-size 4`:

```bash
# 3-way logit validation (FP32 baseline → BF16 expected → Neuron target), offline
python examples/vllm_neuron/accuracy/run_logit_validation_offline.py \
    --model /path/to/gemma-4-31B-it --tp-size 4

# Same, against a running server over /v1/completions
python examples/vllm_neuron/accuracy/run_logit_validation_online.py \
    --model /path/to/gemma-4-31B-it --tp-size 4 [--server-url http://localhost:8000]

# Intermediate-tensor HF-vs-Neuron comparison (per-module)
python examples/vllm_neuron/accuracy/compare_hf_vs_vllm_neuron.py \
    --model /path/to/gemma-4-31B-it --tp-size 4
```

Notes for gemma4:

- These scripts tokenize raw prompts (no chat template). The HF-golden and
  Neuron sides are tokenized identically, so the numerical A/B comparison is
  self-consistent and catches port bugs — but the fed tokens omit `<bos>`, so
  they are not representative of correct `-it` generation. To check
  *generation* content, prepend `<bos>` (id 2) via the offline/online
  `prompt_token_ids` path (see the BOS note above).
- `compare_hf_vs_vllm_neuron_with_reconstruction.py` hardcodes a Llama module
  layout (`LLAMA_MODULES` / `llama_execution_order`) and does **not** capture
  Gemma's extra norms / softcap; do not use it for gemma4 without Gemma-specific
  edits.
- `run_encoder_cache_analysis.py` is a vision-encoder tool — not applicable to
  text-only gemma4.

**Measured result.** With `<bos>`-anchored prompts at TP=4 the 3-way logit
validation passes: aggregate σ-ratio **0.9567** with single-shot prefill and
**0.9877** with segmented prefill (`kv_segment_size 512`), both under the 1.0
acceptance threshold, with aggregate Bhattacharyya coefficient passing in each
case. Read the aggregate `Multi-prompt validation` verdict rather than the
per-prompt lines: the per-prompt `Overall Status` is a two-way FP32-vs-Neuron
comparison, whereas the run of record is the three-way aggregate. A small number
of generated tokens differ on near-tie logits where the CPU BF16 golden itself
flips between the top two candidates; the divergence is *smaller* segmented than
unsegmented, so segmentation does not introduce it.

## Benchmark — `vllm bench serve`

Performance is measured with the built-in, model-agnostic `vllm bench serve`
client against a running gemma4 server (start it as in [Serving](#online-serving-openai-compatible)):

```bash
vllm bench serve \
    --base-url http://localhost:8000 \
    --model gemma-4-31B-it \
    --dataset-name random \
    --random-input-len 256 --random-output-len 128 \
    --num-prompts 32 --max-concurrency 4 \
    --ignore-eos \
    --save-result --result-filename gemma4_bench.json
```

`--ignore-eos` fixes the output length so throughput/latency numbers are stable;
`random`/`prefix_repetition` payloads omit `<bos>`, which does not affect timing
but may produce degenerate content — use a `sharegpt` dataset (which flows
through the chat template) if representative content is needed.

### Measured performance (TP=4, bf16)

Measured on this repository's port on a `trn2.48xlarge` (4-device slice): TP=4,
on-device greedy sampling, `max_num_seqs=4`, `max_model_len=2048`, multi-bucket
prefill `[512, 1024, 2048]`, KV budget capped
(`VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.08`), prefill on the NKI CTE
flash-attention kernel. This is a functional/latency-oriented configuration, not
a maximum-throughput one. Each sweep was run twice against a warm server and the
second pass is reported; the two passes agree to within 1%.

**Time-to-first-token (multi-bucket prefill),** `--random-output-len 8`,
concurrency 1:

| Prompt size | TTFT (median) |
|---:|---:|
| 256 | 730 ms |
| 512 | 732 ms |
| 1024 | 1,357 ms |
| 1920 | 2,672 ms |

Prompts of 256 and 512 tokens land in the same 512-token prefill bucket and
therefore share a TTFT; 1024 and 1920 land in the 1024- and 2048-token buckets.

**Throughput,** `--random-input-len 256 --random-output-len 64`:

| Concurrency | Output tok/s | Output tok/min | Mean TPOT |
|---:|---:|---:|---:|
| 1 | 10.3 | 617 | 87 ms |
| 2 | 18.5 | 1,108 | 93 ms |
| 4 | 30.5 | 1,829 | 104 ms |

Per-request decode is bound by the decode-attention path: both layer types
(head_dim 256 and 512) exceed the fused TKG decode megakernel's 128 head_dim
cap, so decode runs a decomposed PyTorch attention. (This cap is specific to the
decode megakernel — the CTE prefill kernel accepts head dims up to 512.)
Single-stream decode is ~87 ms/token, and per-token latency degrades only ~19%
from concurrency 1 to 4, so aggregate throughput still scales roughly 3× over
that range.

### Measured performance (TP=8, bf16)

Same recipe on an 8-device slice of the same `trn2.48xlarge`, with the KV budget
left at its **default** (the `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.08` cap
that TP=4 requires is unnecessary at TP=8):

**Time-to-first-token,** `--random-output-len 8`, concurrency 1:

| Prompt size | TTFT (median) |
|---:|---:|
| 256 | 622 ms |
| 512 | 623 ms |
| 1024 | 1,216 ms |
| 1920 | 2,433 ms |

**Throughput,** `--random-input-len 256 --random-output-len 64`:

| Concurrency | Output tok/s | Output tok/min | Mean TPOT |
|---:|---:|---:|---:|
| 1 | 15.0 | 901 | 58 ms |
| 2 | 26.2 | 1,574 | 63 ms |
| 4 | 41.9 | 2,514 | 72 ms |

Doubling TP improves every axis: TTFT drops 9–15%, single-stream decode goes from
~87 to ~58 ms/token (+46% tok/s), and concurrency-4 throughput rises 37% to 41.9
tok/s. Decode gains more than prefill because the decomposed decode attention is
the per-request bottleneck and it shards with TP. All 32 requests succeeded at
every concurrency level at both parallel degrees.


## Contents of this bundle

The model is integrated in-tree (committed directly on this branch), so the
top-level `gemma4-31b/` bundle is supplementary — it carries this
source-of-truth README.

```text
gemma4-31b/
└── README.md            # This file — the source of truth for the gemma4-31b port
```

The paths the port touches are listed below; that listing is the record of which
shared framework files this model changes.

### Paths this port touches across the repository

**New — model package** (`vllm_neuron/model/gemma4/`)

```text
vllm_neuron/model/gemma4/
├── __init__.py                  # Package exports (Gemma4ForCausalLM)
├── README.md                    # Package-level module structure (points here)
├── config.py                    # _Gemma4Config: rewrites the checkpoint's multimodal
│                                #   arch to the text arch Gemma4ForCausalLM at registration
├── factory.py                   # Model construction / weight-loading factory
├── attention_decode_kernel.py   # Optional NKI decode attention for head_dim > 128
│                                #   (opt-in via VLLM_GEMMA4_DECODE_KERNEL=1; correct
│                                #   on device but slower (2.35x TP=4 / 1.60x TP=8)
│                                #   — leave it off)
└── model.py                     # Full text model: heterogeneous attention (sliding
                                 #   head_dim=256 / global head_dim=512), QK/V norm,
                                 #   partial RoPE, GeGLU MLP, logit softcap
```

**New — docs & example**

```text
docs/model-recipes/gemma4-31b.md              # Model recipe / card (pointer to this README)
docs/tutorials/tutorial-gemma4-31b.md         # Deployment tutorial (pointer to this README)
examples/vllm_neuron/models/gemma4/run.py     # Offline generation example (TP=4 default, <bos> anchor)
```

**Modified — shared framework touch-points**

```text
vllm_neuron/model/registry.py       # Register Gemma4ForCausalLM
vllm_neuron/__init__.py             # gemma4 HuggingFace AutoConfig registration
                                    #   + multimodal→text arch rewrite
vllm_neuron/functional/mlp.py       # Inline tanh-GELU for GELU_Tanh_Approx (traceable on
                                    #   libtorch_neuronx_lite; numerically identical)
vllm_neuron/functional/attention/attention_cte.py            # MAX_HEAD_DIM 128 -> 512, matching
                                    #   nkilib attention_cte's own _MAX_HEAD_DIM; below 512 large
                                    #   head dims silently fell back to PyTorch
vllm_neuron/functional/attention/attention_segmented_cte.py  # Same ceiling, for consistency
                                    #   with the nkilib segmented kernel's assert
README.md                           # Add Gemma 4 31B to the Contributed Models list
docs/model-recipes/index.md         # Link the model recipe
docs/tutorials/index.md             # Link the tutorial
```



