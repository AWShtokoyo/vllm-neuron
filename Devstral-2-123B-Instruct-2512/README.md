# Contributed Model: Devstral-2-123B-Instruct-2512 (Ministral3)

vllm-neuron implementation of
[`mistralai/Devstral-2-123B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512),
a 123B **FP8 dense-GQA causal LM** (`model_type` `ministral3`, architecture
`Ministral3ForCausalLM`). This README is the **single source of truth** for the
port: architecture, setup, serving, verification, and measured performance. The
`docs/` recipe and tutorial and the model-package README are thin pointers to
this file.

> **Neuron 2.32 / vllm-neuron 0.24.** This branch hosts the port on the public
> `vllm-neuron` **0.24** framework (Neuron 2.32). Two things changed materially
> versus the 0.21 port on `add-devstral-ministral3-231`: **automatic prefix
> caching (APC) now works** and is part of the recommended configuration, and the
> `nkilib` kernel-library patch the 0.21 port needed is **obsolete** — the only
> shared-file edit is `integration.patch`.

## Introduction

Devstral-2-123B-Instruct-2512 is a 123B coding-oriented instruct model. The
checkpoint ships **per-tensor static FP8 (E4M3)** weights with `activation_scheme
= "static"`, and two paths are possible:

- **BF16-dequant (the code default):** FP8 weights are dequantized to BF16 at
  load (`weight × weight_scale_inv`). No HBM saving, and it needs **TP ≥ 16**
  (~30.75 GiB/core at TP=8 does not fit). **Out of scope for this branch.**
- **FP8-native (opt-in, the validated path):** the dense projections stay in FP8
  and run FP8×FP8 static matmuls on Trainium2 — weight HBM roughly halves. Set
  the `quantization` knob (below) to take it.

vLLM rejects the `fp8` quant method, so `quantization_config` is emptied via
`--hf-overrides` on both paths.

> 🔴 **Two environment knobs are required for correctness, not for tuning.**
> Omitting either produces output that passes every automated guard the stack
> offers — `asserts=0`, no exceptions, plausible throughput — **and is wrong.**
> They are `MINISTRAL3_FP8_LAYERS="mlp:0-1,3-87"` and `FORCE_MLP_KERNEL=cte`,
> both explained under **Required knobs**.

**Verification scope.** Validated at **TP=8** (one replica on 2 chips) and
**TP=8 × DP=8** (whole `trn2.48xlarge`) on Neuron 2.32 / `vllm-neuron
0.24.0.1.1.0`. TP=16 and TP=32 are **not** recommended for this port — see
**Feature status**.

**Compatible checkpoints:**

| Model | HuggingFace |
|-------|-------------|
| Devstral-2-123B-Instruct-2512 | [mistralai/Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512) |

## Model Architecture

Dense GQA decoder, nearly identical to `llama3`, with three differences:

| Field | Value |
|---|---|
| hidden_size | 12288 |
| num_hidden_layers | 88 |
| num_attention_heads (Q) | 96 |
| num_key_value_heads (KV) | 8 (GQA) |
| head_dim | 128 |
| intermediate_size | 28672 |
| vocab_size | 131072 |
| max_position_embeddings | 262144 |
| RoPE | **YaRN** (θ=1e6, factor=64, orig_max=4096, β_fast=4, β_slow=1, mscale=1) |
| activation / norm | SiLU (SwiGLU) / RMSNorm (eps 1e-5), pre-attn + pre-MLP |
| attention bias / qk_norm | none |
| tie_word_embeddings | false (separate `lm_head`) |

The three differences versus `llama3`:

1. **YaRN RoPE** — YaRN frequency computation; the rotation stays interleaved
   (`rotate_half`, Mistral/Llama style), **not** the GPT-OSS split-half form.
2. **FP8 per-tensor quantization** — on the dequant path, `weight ×
   weight_scale_inv` (embeddings / norms / `lm_head` are already BF16). The
   **adaptive weight loaders** detect FP8 from the checkpoint's slice count and
   the presence of `*.weight_scale_inv`, **not** from `config.json` (which is
   emptied — see **Setup**).
3. **`tie_word_embeddings=false`** — an independent `lm_head`.

## Required knobs

### `MINISTRAL3_FP8_LAYERS="mlp:0-1,3-87"` — keep layer 2's MLP in BF16

**Symptom without it.** With every layer's MLP in static FP8, output corruption
appears wherever the model emits digits or list separators: spurious `' ,'` after
digits, non-Latin token injection (Cyrillic `' січ'`, Italian `Perché`), and
mid-list collapse.

**Cause — the anomaly is in the checkpoint; Trainium2's narrower FP8 range
exposes it.** `mlp.down_proj.activation_scale`, read directly from the
checkpoint, runs layer 0 `0.0737` → 1 `0.0097` → **2 `1.9141`** → 3 `0.0093`:
layer 2 is a **197× discontinuity against its immediate neighbours**, and 3 of
the 88 layers sit above 20× the all-layer median. Trainium2's e4m3 saturates at
**±240** while the checkpoint is calibrated for OCP **±448**, and the generic FP8
path applies the stored scales unchanged, with no compensation. ⇒ Neither side
alone produces the defect — the combination does, and layer 2's MLP is where it
lands.

**Fix.** Keep layer 2's MLP in BF16, leave the other 87 layers in FP8:

```bash
export MINISTRAL3_FP8_LAYERS="mlp:0-1,3-87"
```

Cost: weights **+0.12 GiB/core** (14.94 → 15.06) and **no KV-headroom loss**
(315,712 KV tokens either way, comfortably above the 131,072 that
`num_seqs=64 × max_model_len=2048` needs).

⚠️ **Magnitude alone is not the mechanism, so do not generalise to "put the
biggest outlier in BF16".** Layer 87 carries a *larger* scale and stays in FP8
under this fix with no corruption; `mlp:0-1,3-87` is the **minimal sufficient**
set, and widening it was measured and does not help.

> 🔴 **Generality:** *which* layer fails is decided by the checkpoint's
> calibration run, so **another FP8 checkpoint may fail at a different layer**.
> The per-layer knob is general; the index `2` is not. If you bring up a
> different FP8 checkpoint, look for the `activation_scale` outlier.
>
> ⚠️ **Omit the variable entirely for a baseline. Setting it to an empty string
> is not the same thing** — an empty spec means "all 88 layers in FP8", i.e. the
> corrupted configuration. The server log's per-layer census line distinguishes
> them: `mlp=87/88 [0-1,3-87]` with the knob, `mlp=88/88 all` without.

### `FORCE_MLP_KERNEL=cte` — the decode MLP must run the CTE kernel

`nkilib` routes the MLP on `batch_size × sequence_len ≤ 96`. In decode
`sequence_len = 1`, so **`num_seqs` itself is compared against 96** — and at
`num_seqs ≤ 96` the router selects the TKG MLP, which has **no static-FP8 path**
on this stack. The result is numerically wrong output with **zero kernel
asserts**.

```bash
export FORCE_MLP_KERNEL=cte
```

🎯 It is also **~17–19% faster** than avoiding the routing boundary by raising
`num_seqs` to 128, so the correctness knob is the fast choice too.

## Feature status

| Category | Feature | Status |
|---|---|---|
| Quantization | FP8-native (`fp8:qkv,o_proj,mlp`), per-tensor static | ✅ Validated at TP=8 |
| | BF16-dequant | ⚠️ Needs TP ≥ 16; out of scope for this branch |
| | FP8 KV cache (`--kv-cache-dtype fp8_e4m3`, packed) | ✅ |
| Attention | YaRN RoPE, GQA, segmented prefill | ✅ |
| Prefix caching | Automatic prefix caching (`--enable-prefix-caching`) | 🟢 **New on 2.32.** Needs segmented prefill (see below) |
| Parallelism | TP=8 (2 chips, one replica) | ✅ Recommended |
| | TP=8 × DP=8 (whole `trn2.48xlarge`) | ✅ Validated |
| | TP=16 / TP=32 | ⚠️ Not recommended for this port — an unexplained accuracy residual remains above TP=8 |
| | Dependent DP, DCP | Untested for this model |
| Sampling | On-device greedy sampling | ✅ |
| | `logprobs` | ✅ with `max_logprobs` in `neuron_config` and `--no-async-scheduling` (⚠️ changes ITL — accuracy runs only) |
| | `prompt_logprobs` / `echo` | ⛔ On-device sampling keeps only top-k, so full-vocabulary loglikelihood is unavailable |
| Prefill | Multi-bucket prefill (`num_batched_tokens_buckets`) | ✅ Part of the recommended configuration |
| Speculative decoding | Eagle3 | Not wired (no draft checkpoint) |

## Setup

### Native integration (nothing to apply)

The model is **committed in-tree on this branch**: the package lives at
[`vllm_neuron/model/ministral3/`](../vllm_neuron/model/ministral3/) and the
shared-framework edits are already applied. Checking out this branch and
installing the framework is enough — `Devstral-2-123B-Instruct-2512/integration.patch`
is a standalone copy of the same shared-file edits, for grafting the port onto a
separately-installed `vllm_neuron`.

### Step 1: Environment setup

```bash
VENV_DIR=~/vllm_neuron_venv
python3.13 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
python -m pip install neuronx-cc==2.*

git clone https://github.com/AWShtokoyo/vllm-neuron.git
cd vllm-neuron && git checkout add-devstral-ministral3
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .

# `-e .` already resolves torch, torch-xla and libtorch-neuronx-lite. Pin them
# explicitly so a later install cannot drag torch-xla off torch's minor version.
# ⚠️ This must come AFTER `-e .` — a fresh venv has no torch, so `import torch`
#    here would fail.
TORCH_MM=$(python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")
python -m pip install torch-xla==${TORCH_MM}.* libtorch-neuronx-lite==${TORCH_MM}.*

neuron-ls          # confirm the NeuronCores are visible and idle
python -V          # → Python 3.13.x
pip list | grep -e neuron -e nki -e torch
#   torch                         2.11.0
#   torch-xla                     2.11.0
#   libtorch-neuronx-lite         2.11.0.1.0.1284+f49d8626
#   neuronx-cc                    2.27.5334.0+f702b353
#   nki                           0.6.0+31049202112.g85070674
#   vllm-neuron                   0.24.0.1.1.0
```

```bash
# Point the compile caches at fast local storage (NOT an NFS home directory —
# compile and load are otherwise dominated by NFS latency).
export VLLM_CACHE_ROOT=/opt/nvme/vllm_cache
export NKI_COMPILE_CACHE_URL=$HOME/nki_cache
# A single cold graph has been measured at 1,808 s. Raise the readiness timeout
# AND any outer wait loop together — raising only the outer loop is useless
# because the engine gives up first.
export VLLM_ENGINE_READY_TIMEOUT_S=10800
```

### Step 2: Download the model

```bash
hf download mistralai/Devstral-2-123B-Instruct-2512 \
  --local-dir /opt/nvme/devstral-ckpt
```

The checkpoint is **239 GB**. Put it on local NVMe, not network storage — load
time dominates otherwise.

> 🔴 **Run the compiler from a neutral working directory.** `neuronx-cc` dumps
> NKI artefacts into the current directory, so launching from inside the
> repository litters it.

## Serving

### Online serving (OpenAI-compatible)

```bash
export MINISTRAL3_FP8_LAYERS="mlp:0-1,3-87"   # 🔴 required — see Required knobs
export FORCE_MLP_KERNEL=cte                   # 🔴 required — see Required knobs
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2

vllm serve /opt/nvme/devstral-ckpt \
  --served-model-name mistralai/Devstral-2-123B-Instruct-2512 \
  --tensor-parallel-size 8 \
  --max-model-len 2048 --max-num-batched-tokens 1024 --max-num-seqs 64 \
  --enable-prefix-caching --config-format hf --tokenizer-mode mistral \
  --load-format safetensors --hf-overrides '{"quantization_config": {}}' \
  --kv-cache-dtype fp8_e4m3 \
  --additional-config '{"neuron_config": {
      "on_device_sampling_config": {"all_greedy": "true"},
      "num_batched_tokens_buckets": [128,256,512,1024],
      "kv_segment_size_buckets": [512,1024],
      "num_seqs_buckets": [64],
      "quantization": "fp8:qkv,o_proj,mlp",
      "fp8_packed_kv": true}}'
```

For the whole box, add `--data-parallel-size 8`, set
`NEURON_VISIBLE_DEVICES` to all 64 logical cores, and export
`VLLM_WORKER_MULTIPROC_METHOD=fork` (the `DPCoordinator` child otherwise re-runs
the ~56 s plugin import and trips its own 30 s ZMQ timeout).

### Offline inference (`LLM`)

See [`examples/vllm_neuron/models/ministral3/run.py`](../examples/vllm_neuron/models/ministral3/run.py)
for a runnable example. The load-bearing arguments:

```python
llm = LLM(
    model="/opt/nvme/devstral-ckpt",
    served_model_name="mistralai/Devstral-2-123B-Instruct-2512",
    tensor_parallel_size=8,               # divides Q=96
    max_model_len=2048,
    max_num_seqs=64,
    max_num_batched_tokens=1024,          # must equal the largest prefill bucket
    config_format="hf",
    load_format="safetensors",
    tokenizer_mode="mistral",
    enable_prefix_caching=True,
    hf_overrides={"quantization_config": {}},
    kv_cache_dtype="fp8_e4m3",
    additional_config={"neuron_config": {
        "num_batched_tokens_buckets": [128, 256, 512, 1024],
        "kv_segment_size_buckets": [512, 1024],
        "num_seqs_buckets": [64],
        "quantization": "fp8:qkv,o_proj,mlp",
        "fp8_packed_kv": True,
        "on_device_sampling_config": {"all_greedy": "true"}}},
)
```

### Required flags (summary)

| Flag / knob | Why |
|---|---|
| `MINISTRAL3_FP8_LAYERS="mlp:0-1,3-87"` | 🔴 correctness — see **Required knobs** |
| `FORCE_MLP_KERNEL=cte` | 🔴 correctness at `num_seqs ≤ 96` — see **Required knobs** |
| `--hf-overrides '{"quantization_config": {}}'` | vLLM rejects the `fp8` quant method |
| `--config-format hf --tokenizer-mode mistral` | Mistral tokenizer / HF config layout |
| `"quantization": "fp8:qkv,o_proj,mlp"` | selects the FP8-native path |
| `--kv-cache-dtype fp8_e4m3` + `"fp8_packed_kv": true` | packed FP8 KV cache |
| `max_num_batched_tokens` == largest `num_batched_tokens_buckets` entry | bucket contract |
| `--enable-prefix-caching` with `max_num_batched_tokens` **<** `max_model_len` | APC needs segmented prefill; equal values resolve to single-shot and APC is rejected at startup |

### Automatic prefix caching on 2.32

APC requires **segmented** prefill, i.e. `max_num_batched_tokens <
max_model_len`. `max_num_batched_tokens` **is** the last entry of the prefill
bucket list, so 🔑 **what disables APC is the bucket ladder topping out at
`max_model_len`** — not how many buckets it has. Enabling APC without lowering
the top bucket makes the engine refuse to start with `ValueError: Automatic
Prefix Caching (APC) requires segmented prefill to be enabled`.

`kv_segment_size_buckets: [512, 1024]` is the best segment ladder measured at
this shape (`max_model_len=2048`, `in=1024 / out=256`); a single `512` and
`[512, 2048]` are both worse. ⚠️ **Re-tune the list if you change
`max_model_len` or the input-length distribution.**

> 🔴 **`--dataset-name random` is deterministic in `--seed`**, so re-running a
> sweep regenerates a byte-identical prompt set and hits the cache its own first
> pass populated. Restart the server between measurement points and report
> **cold and warm separately** — every figure below follows that rule.

## Verification

Use the repository's **generic, model-independent** tooling; this port ships no
model-specific accuracy harness.

**1. Accuracy — generic logit comparison.** The scripts in
[`examples/vllm_neuron/accuracy/`](../examples/vllm_neuron/accuracy/) work on any
text causal LM:

```bash
# 3-way logit validation (FP32 baseline → BF16 expected → Neuron target), offline
python examples/vllm_neuron/accuracy/run_logit_validation_offline.py \
  --model /opt/nvme/devstral-ckpt --tp-size 8

# HF vs vllm-neuron intermediate-tensor comparison
python examples/vllm_neuron/accuracy/compare_hf_vs_vllm_neuron.py \
  --model /opt/nvme/devstral-ckpt --tp-size 8
```

🔴 **Gate every run on the server log before believing any number:**

| Gate | Required | If it fails |
|---|---|---|
| per-layer census | `mlp=87/88 [0-1,3-87]` | `mlp=88/88 all` means the env var never reached the workers — **discard the run** |
| knob echo | the knob appears in the `SCALE KNOBS:` line | the knob did not apply |
| `tensor_parallel_size` | matches what you asked for | wrong arm |
| completion | every request completed | `completed == 0` is **missing data, not a result** |
| kernel asserts | 0 | ⚠️ **necessary but nowhere near sufficient** — corrupted output raises no assert |

⚠️ **Do not validate with `asserts=0` or with TTFT.** Corrupted output is
*faster* than correct output on this stack (one repeated token is cheap), so a
throughput number without an accuracy check beside it is not quotable.

**2. Benchmark — `vllm bench serve`.** The built-in, model-independent client:

```bash
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --model mistralai/Devstral-2-123B-Instruct-2512 \
  --tokenizer /opt/nvme/devstral-ckpt --tokenizer-mode mistral \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 192 --max-concurrency 64 --ignore-eos --temperature 0 --seed 0
```

`--ignore-eos` fixes the output length, which is what makes the numbers stable.

## Measured performance

Self-measured on `trn2.48xlarge`, Neuron 2.32 / `vllm-neuron 0.24.0.1.1.0`,
FP8-native with both required knobs, `in=1024 / out=256`, greedy, random
dataset. **cold** = fresh server, nothing cached; **warm** = the same prompt set
replayed against a populated prefix cache.

### TP=8 × DP=1, `num_seqs=64` fixed — the recommended configuration

One compiled graph serves every concurrency.

| conc | N | cold tok/s | **warm tok/s** | cold TTFT med | warm TTFT med | **warm ITL med** |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16 | 148.3 | **155.6** | 1,150 ms | 427 ms | **62.42 ms** |
| 8 | 64 | 483.3 | **573.0** | 2,977 ms | 1,138 ms | **62.47 ms** |
| 32 | 192 | 1,107.7 | **1,742.3** | 10,878 ms | 3,952 ms | **62.53 ms** |
| 64 | 192 | 1,410.5 | **2,637.9** | 21,455 ms | 7,721 ms | **62.57 ms** |
| 128 | 192 | 1,414.9 | **2,640.1** | 68,987 ms | 34,980 ms | **62.57 ms** |

⇒ **ITL is flat at 62.4–62.6 ms across the whole ladder** — a 0.24% spread over a
**64× range** of offered concurrency. Per-token decode cost does not degrade with
load; what grows is queueing, which shows up in TTFT. Throughput saturates at
conc=64. Memory: **15.06 GiB/core**, KV **315,712 tokens**.

> Offered load beyond 64 on this graph only *queues* — conc=64 → 128 gains
> **+0.08%** throughput while median TTFT rises **4.53×**. Cap admitted
> concurrency at 64, or compile a larger `num_seqs` bucket and accept padding
> waste at low concurrency.

### TP=8 × DP=1, `num_seqs=conc` — peak throughput

The bucket is matched to the load at every point, i.e. **one compiled graph per
concurrency**.

| conc | N | cold tok/s | **warm tok/s** | warm TTFT med | **warm ITL med** |
|---:|---:|---:|---:|---:|---:|
| 2 | 16 | 214.3 | **230.6** | 404 ms | **41.46 ms** |
| 8 | 64 | 615.7 | **771.9** | 1,120 ms | **44.45 ms** |
| 32 | 192 | 1,192.1 | **1,966.4** | 3,935 ms | **52.07 ms** |
| 64 | 192 | 1,407.8 | **2,638.4** | 7,730 ms | **62.55 ms** |
| 128 | 192 | 1,435.4 | **2,722.1** | 11,544 ms | **88.15 ms** |

⇒ Peak **2,722.1 tok/s** at conc=128 (+3.2% over the fixed-64 graph's saturation
point), and materially better latency at low concurrency — **ITL 41.5 ms at
conc=2** versus 62.4 ms — because a `num_seqs=2` graph does no padded work. The
cost is one compile per concurrency point and no headroom above the bucket.

**Which to use:** `num_seqs=64` fixed if the offered load varies (one graph, flat
ITL); `num_seqs=conc` if the concurrency is known and pinned.

### TP=8 × DP=8 — whole box

A `trn2.48xlarge` is **64 logical NeuronCores** (16 chips × 4 at LNC2), and on
this stack `TP × DP == world_size == cores`, so **TP=8 = 2 chips** and 8 replicas
fill the node. Measured by the same driver, in the same shape as the DP=1 ladder
above, so **DP is the only variable**. Every point completed with `asserts=0`
and the census confirmed on all 64 ranks.

| total conc | per replica | cold tok/s | **warm tok/s** | warm TTFT med | ITL med |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 592.7 | 608.0 | 676 ms | 62.37 ms |
| 32 | 4 | 2,090.4 | 2,096.6 | 772 ms | 62.46 ms |
| 128 | 16 | 5,844.0 | 5,977.6 | 935 ms | 62.52 ms |
| 256 | 32 | 8,348.7 | 8,437.4 | 1,076 ms | 62.62 ms |
| **512** | **64** | **10,417.7** | **10,586.4** | 19,779 ms | 62.60 ms |
| 1024 | 128 | 10,387.2 | 10,585.5 | 71,067 ms | 62.58 ms |

**conc=512 is the saturation point, and it is measured, not assumed.** There each
replica fills its compiled `num_seqs=64` bucket exactly; doubling the offered
load to 1024 changes throughput by **−0.3% cold / −0.008% warm** — i.e. not at
all — while median TTFT rises from 19.8 s to **71.1 s**.

🔑 **The TTFT jump at conc=512 is queueing, not slowdown.** Server capacity is
`DP × num_seqs` = **512** concurrent sequences: below it a slot is always free
and TTFT is prefill-only (~1.1 s), at it every slot is occupied so a new request
waits for one to finish (256 × 62.6 ms ≈ 16 s). ⇒ **Cap admitted concurrency at
`DP × num_seqs`, and below it if TTFT matters.**

**Compute scales at 92% per replica.** Comparing DP=1 and DP=8 at the same
per-replica load (64 concurrent each) on the cold pass, **1,410.5 → 10,417.7
tok/s**: a **7.39×** gain on 8× the chips, i.e. **92.3% scaling efficiency**.

> **DP > 1 requires the code fix in `integration.patch`** (module-scope
> `_Ministral3HFConfig`; vLLM pickles the HF config across engine-core
> subprocesses) plus `VLLM_WORKER_MULTIPROC_METHOD=fork`, and
> `NEURON_VISIBLE_DEVICES` must enumerate all **64 logical cores** (vLLM slices
> it per DP rank). The DP=1 path is pure configuration.

### What the prefill bucket ladder is worth

The recommended configuration already uses a four-bucket ladder. Measured
against a **single** `[2048]` bucket, which pads every short prompt to 2048:

🔴 **This pair is measured in a different configuration from every table above**,
because the baseline needs an un-bucketed ladder: `max_num_batched_tokens ==
max_model_len == 2048` (single-shot prefill), which **forces APC off**.
`num_seqs=64`, both required knobs on, `in=128 / out=128` — a short-input burst,
the case the ladder exists for. **Do not compare these rows against the tables
above.**

| | prefill buckets | total tok/s | TTFT median | TTFT p99 | ITL median |
|---|---|---:|---:|---:|---:|
| **DP=1** (256 prompts @ conc 256) | `[2048]` | 384.5 | 82,340 ms | 162,024 ms | 65.68 ms |
| | ⭐ `[128,256,512,1024,2048]` | **1,146.3** | **25,038 ms** | **49,012 ms** | 65.63 ms |
| **DP=8** (512 prompts @ conc 1024) | `[2048]` | 2,977.5 | 18,737 ms | 35,848 ms | 65.68 ms |
| | ⭐ `[128,256,512,1024,2048]` | **8,361.9** | **4,213 ms** | **7,305 ms** | 65.74 ms |

⇒ On a 128-token burst the ladder is worth **2.8–3.0× throughput** and cuts
median TTFT by **3.3× (DP=1)** and **4.5× (DP=8)**; p99 TTFT improves **4.9×** at
DP=8. 🔑 **ITL is identical to two decimal places on all four arms** — the ladder
changes only which prefill graph runs, never the decode path or what is
computed, so quality is unaffected.

⚠️ **The padding ratio is the whole story**: 128 into 2048 is 16×, the best case.
At `in=1024` against a 2048 bucket there is nothing to recover. **Measure it on
your own input-length distribution.**

## Contents of this bundle

The model is integrated in-tree (committed directly on this branch), so the
top-level `Devstral-2-123B-Instruct-2512/` bundle is supplementary — it carries
this source-of-truth README and a standalone patch of the same shared-file edits.

```text
Devstral-2-123B-Instruct-2512/
├── README.md            # This file — the source of truth for the port
└── integration.patch    # Shared-framework touch-points only, as a standalone patch,
                         #   for grafting the port onto a separately-installed
                         #   vllm_neuron (already committed in-tree here)
```

No `test/` directory is shipped: verification is the two generic methods under
**Verification**. The bring-up benches and serving/sweep scripts used to produce
the numbers above are kept internal and are not part of this bundle; their
measured values are transcribed into **Measured performance**.

### Paths this port touches across the repository

**New — model package** (`vllm_neuron/model/ministral3/`)

```text
vllm_neuron/model/ministral3/
├── __init__.py          # Package exports (Ministral3ForCausalLM)
├── README.md            # Package-level module structure (points here)
├── config.py            # Ministral3 config + neuron_config plumbing (YaRN, FP8 families)
├── factory.py           # Model construction / weight-loading factory
├── model.py             # Dense GQA decoder: YaRN RoPE, static-FP8 QKV/o_proj/MLP,
│                        #   per-layer FP8/BF16 split, packed FP8 KV
└── weight_loaders.py    # Adaptive FP8 / BF16 loaders (fused QKV, TP sharding)
```

**New — docs & example**

```text
docs/model-recipes/devstral-2-123b.md            # Model recipe / card (pointer to this README)
docs/tutorials/tutorial-devstral-2-123b.md       # Deployment tutorial (pointer to this README)
examples/vllm_neuron/models/ministral3/run.py    # Offline inference example
```

**Modified — shared framework touch-points** (10 files)

```text
vllm_neuron/model/registry.py                        # Register Ministral3ForCausalLM
vllm_neuron/model/__init__.py                        # Add ministral3 to the lazy-import set
vllm_neuron/__init__.py                              # Register the ministral3 HuggingFace AutoConfig
vllm_neuron/vllm/platform.py                         # Pre-register the _ModelInfo
vllm_neuron/utils/bucket_utils.py                    # Segmented prefill: allow seqlen_q <= kv_segment_size
                                                     #   and a list of segment sizes (this is what enables APC)
vllm_neuron/vllm/worker/neuron_model_runner.py       # Q padding / buffers sized for the largest segment
vllm_neuron/vllm/worker/neuron_worker.py             # Skip unreachable (bucket, segment) pairs at warmup/compile
vllm_neuron/functional/attention/attention_segmented_cte.py  # Derive active/prior segments independently
vllm_neuron/functional/mlp.py                        # FORCE_MLP_KERNEL + a ROW eligibility fix
vllm_neuron/functional/attention/o_proj.py           # ROW layout support on the output projection
docs/model-recipes/index.md                          # Link the model recipe
docs/tutorials/index.md                              # Link the tutorial
```
