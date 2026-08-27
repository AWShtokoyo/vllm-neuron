# Contributed Model: Qwen3.8-27B (Qwen3.5 dense hybrid, BF16 and FP8)

vllm-neuron port of `Qwen/Qwen3.8-27B` and `Qwen/Qwen3.8-27B-FP8` for AWS Trainium2,
on the **Neuron 2.32 stack (vLLM 0.24 / plugin 0.24.0.1.1.0)**.

It adds the [`qwen3_5_dense`](../vllm_neuron/model/qwen3_5_dense/README.md) model package and serves
Qwen3.8-27B in **both BF16 and FP8**, so this is the single 0.24 branch for the whole `qwen3_5` dense
family: Qwen3.5-27B, Qwen3.6-27B and Qwen3.8-27B, BF16 and FP8, all served through one code path
(see [Compatible checkpoints](#introduction) below).

> If you are on the **Neuron 2.31 / vLLM 0.21** stack, use the
> [`add-qwen36-27b-231`](https://github.com/htokoyo/vllm-neuron/blob/add-qwen36-27b-231/Qwen3.6-27B/README.md)
> tag instead: it holds the predecessor `Qwen3.6-27B` port for that stack, which this branch supersedes.

## Introduction

Qwen3.8-27B is a **hybrid dense** decoder: 64 layers made of 48 GatedDeltaNet (linear
attention) layers and 16 full-attention GQA layers (`full_attention_interval=4`), a plain
SwiGLU FFN, `head_dim=256`, 24 query heads / 4 KV heads, and partial RoPE
(`partial_rotary_factor=0.25`, so only 64 of the 256 head dims are rotated) with a 10M RoPE
base. The architecture is identical to Qwen3.6-27B, so one hybrid attention stack, one set of
GatedDeltaNet NKI kernels, and one forced-bf16 SSM state serve the whole family.

The FP8 checkpoint is a DeepSeek-style **block-quantised** release: each quantised tensor
ships `.weight` (`float8_e4m3fn`) plus `.weight_scale_inv` (bf16, one scale per `[128,128]`
block). Neuron's NKI GEMMs support per-tensor, **per-output-channel (ROW)**, and MX block-32
scaling — but not `[128,128]` block scaling — so this port **re-quantises the block scales to
per-channel ROW fp8 at load time**. Weights stay resident as fp8, which is where the memory
saving comes from; dequantising to bf16 at load would save nothing.

Scope of the FP8 path in this release: the **MLP** `gate`/`up`/`down` projections of all 64
layers are fp8-resident. QKV, `o_proj`, and the GatedDeltaNet projections are stored fp8 in
the checkpoint and **dequantised to bf16 at load**. The MLP is where the weight mass is
(3 × 17408 × 5120 × 64 ≈ 17.1B of the 27B parameters), so this captures the bulk of the
saving. The MTP head (`mtp_num_hidden_layers: 1`) is not used.

**Compatible checkpoints:**

Every checkpoint below shares the `qwen3_5` hybrid dense architecture (64 layers,
`full_attention_interval=4`, hidden 5120, FFN 17408), so this branch's `qwen3_5_dense` package
serves them all through one code path. The only thing that varies is `quantization_config`:
absent means BF16, and DeepSeek block-FP8 (`quant_method: fp8`, `weight_block_size: [128,128]`)
selects the FP8 path. Selection keys off the block size alone, so the FP8 checkpoints are
accepted whether or not they also declare `fmt: e4m3`; any other quantisation scheme is
rejected loudly rather than silently mis-served.

| Model | HuggingFace | Hardware | Quantization | Verified on this branch |
|-------|-------------|----------|--------------|-------------------------|
| Qwen3.8-27B | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Trn2 | BF16 | ✅ served + scored on device |
| Qwen3.8-27B-FP8 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | Trn2 | FP8 (block → per-channel ROW at load) | ✅ the primary target of this port |
| Qwen3.6-27B | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | Trn2 | BF16 | ✅ served on device |
| Qwen3.6-27B-FP8 | [Qwen/Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) | Trn2 | FP8 (block → per-channel ROW at load) | ⚠️ config verified compatible; not run on device |
| Qwen3.5-27B | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | Trn2 | BF16 | ⚠️ config verified compatible; not run on device |
| Qwen3.5-27B-FP8 | [Qwen/Qwen3.5-27B-FP8](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) | Trn2 | FP8 (block → per-channel ROW at load) | ⚠️ config verified compatible; not run on device |

## Verification scope

Everything below was measured on real hardware — a single **trn2.3xlarge** (1 Trainium2
device, 4 NeuronCores, 96 GB HBM), **TP=4**, LNC=2, on the Neuron 2.32 stack.

**Verified on device**

| Area | What was verified |
|---|---|
| FP8 serving | Compiles and serves; coherent text at every configuration below |
| BF16 serving | The BF16 recipe serves on device with a real BF16 release |
| Long context | `max-model-len` up to 32768; the opt-in prefill kernel was exercised there too |
| Accuracy at long context | GSM8K 4-shot with a ~25,000-token irrelevant prefix, on both prefill kernels — a few points below the short-prompt figure and inside the measurement error, see [Accuracy at long context](#accuracy-at-long-context) |
| Batching | `max-num-seqs` 1, 4 and 8 — coherent at every step, including a 6,300-token prompt with several requests prefilling concurrently |
| Memory | HBM measured per rank via `neuron-monitor`, summed across all four ranks |
| Throughput / latency | Output tok/s, TTFT p50 and TPOT at concurrency 1 and 8, natural-text prompts |
| Accuracy | GSM8K-CoT at n=100, and a BF16-vs-FP8 comparison on the same checkpoint (see [Accuracy Evaluation](#accuracy-evaluation)) |
| Sampling | Greedy on-device sampling, and on-device random sampling honouring `temperature`/`top_p`/`top_k` |
| Logit fidelity (BF16 path) | 3-way validation against a CPU FP32 baseline: aggregate σ-ratio ≤ 1 on both BF16 checkpoints (see [Logit validation](#logit-validation-bf16-path)) |
| Vision (image / video) | Image and video both described correctly on device with the FP8 checkpoint (see [Vision serving](#vision-serving-image--video)) |

**Not verified / out of scope**

| Area | Status |
|---|---|
| Speculative decoding, EAGLE | Not exercised |
| Expert parallelism | Not applicable — dense FFN, no experts |
| Pipeline parallelism | Not implemented |
| Per-request `seed` | **Unsupported — see [Sampling](#sampling)** |
| Reasoning *over* long relevant material | Long context does not degrade the model: with 25,000 tokens of prefix it still follows the instructions and few-shot format that sit after the prefix, and answers correctly. What is untested is comprehension of a long **relevant** context — the filler is deliberately content-free, so nothing here measures reading and reasoning across a long document. That would need a multi-document QA suite |
| Multi-instance / multi-node | Not exercised (single trn2.3xlarge only) |

### What to expect on a larger instance (`trn2.48xlarge`, TP=8) — not measured

Nothing in this port is specific to the small instance, but **TP=8 has not been exercised**: neither
the 0.24 stack at that degree, nor the FP8 path at any parallel degree other than 4. What follows is
what to expect, not what was measured.

One recipe change is expected there: the `VLLM_QWEN_QKV_MATMUL=1` workaround should be **dropped**,
because it exists only to work around the fused-QKV kernel's per-rank width of 3584 overrunning the
SBUF budget by ~8 % at TP=4. At TP=8 that width halves to 1792 and the kernel fits, which is why the
flag defaults off.

The headroom should mainly buy concurrency: weights and the fixed GatedDeltaNet per-sequence state
shard across twice as many ranks, so the per-rank footprint roughly halves while the four-core instance
measured here is much tighter. FP8's *relative* benefit should carry over
unchanged, because it comes from halving the MLP weight bytes read on the decode path, which is
independent of the parallel degree; its absolute saving simply matters less when far more HBM
is available.

Properties of the backend or the model do not change with instance size: prefill remains batch-1 and
chunked prefill is supported only at batch size 1, long context still needs chunked prefill at
`max-num-batched-tokens` 512, the SSM state stays forced to bf16, and the per-request `seed`
limitation in [Sampling](#sampling) still applies. The unified KV-gather slab ceiling should be
**re-checked** rather than assumed to scale, since it derives from a uint32 addressing limit in the
indirect scatter rather than from available memory. Multi-node and data-parallel deployment are
likewise unexercised.

## Model Architecture

| Property | Value |
|---|---|
| Architecture aliases | `Qwen3_5ForCausalLM` (text), `Qwen3_5ForConditionalGeneration` (text + vision) |
| `model_type` | `qwen3_5` |
| Layers | 64 = 48 GatedDeltaNet + 16 full attention (`full_attention_interval=4`) |
| Hidden size | 5120 |
| FFN intermediate | 17408 (dense SwiGLU) |
| Attention heads | 24 query / 4 KV (GQA), `head_dim` 256 |
| Attention output gate | Yes (`attn_output_gate=True`) |
| RoPE | Partial (`partial_rotary_factor=0.25` → 64 rotated dims), M-RoPE sections `[11,11,10]`, base 10,000,000 |
| GatedDeltaNet | key dim 2048 / value dim 6144, head dim 128, conv kernel 4 |
| Max position embeddings | 262144 |
| SSM state dtype | **Forced bf16, not tunable** (see below) |

### Forced bf16 GatedDeltaNet state (not tunable)

`get_mamba_state_dtype()` pins the recurrent SSM state to bf16 regardless of
`--mamba-ssm-cache-dtype`. This is load-bearing: a bf16 state keeps the mamba page small
enough that the hybrid block-size aligner stays at 512. An fp32 state doubles the mamba page,
pushes the attention `block_size` to 896, and drives the GatedDeltaNet kernels outside their
validated regime, at which point decode diverges to out-of-vocabulary tokens. Pass
`--mamba-ssm-cache-dtype bfloat16` explicitly so the resolved configuration is unambiguous.

### FP8 quantisation path

| Stage | Behaviour |
|---|---|
| Load | `[128,128]` block dequantisation → per-output-channel absmax → fp8 re-quantisation. Scales are emitted as `[128, out_dim]` (row 0 is the per-channel scale, broadcast across the 128 partition rows) as the NKI ROW contract requires |
| Prefill (CTE) | Manual dequantisation to bf16 followed by a bf16 SwiGLU. The nkilib `mlp_cte` ROW path produces incorrect logits at this model's dimensions on trn2 (device-verified: the first sampled token, which comes from prefill, is a garbage token id, while a manual dequantisation using the *identical* fp8 weights and ROW scales is accurate). Weights stay resident as fp8; only a transient bf16 view is materialised per projection |
| Decode (TKG) | `NF.mlp(ROW)` fp8 kernel, which fuses the RMSNorm and the per-row activation quantisation online from a bf16 input. This is the hot path, and it is where the throughput gain comes from |
| Not quantised | QKV, `o_proj`, GatedDeltaNet projections (dequantised to bf16 at load), all norms, `lm_head`, embeddings |

The fp8 dynamic range on trn2 is the legacy `float8_e4m3` range (max 240), so setting
`neuron_config.quantization="fp8"` is required — it also injects the
`--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` compiler flag that the ROW kernels depend on.

### Prefill kernel selection (`VLLM_GDN_PREFILL`)

The GatedDeltaNet layers admit two prefill formulations, and this port implements both:

| Mode | How it computes the recurrence |
|---|---|
| `sequential` (**default**) | The recurrence exactly as defined, advanced one token at a time inside a single bounded NKI kernel (`nl.sequential_range`) |
| `parallel` (opt-in) | An algebraically equivalent chunked form: within each 64-token chunk the inter-token coupling is resolved by solving `T = (I − A)⁻¹` in fp32 by blocked forward substitution, so chunks are processed as matmuls instead of as a step loop |

Decode follows whichever mode is selected — the two are not independently settable, because a prefill
state produced by one form and advanced by the other diverges to NaN after a few tens of decode steps.
Which one to pick, and what it costs, is in
[Choosing between the two kernels](#choosing-between-the-two-kernels).

## Feature status

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| | Vision — image / video | ✅ Verified on device (FP8, `max-model-len` 4096) |
| **Quantization** | BF16 weights | ✅ |
| | FP8 weights (per-channel ROW, MLP) | ✅ |
| | FP8 KV cache | ❌ Not implemented |
| **Parallelism** | Tensor parallelism (TP) | ✅ (TP=4 verified) |
| | Expert parallelism | ⚠️ Not applicable — this is a dense FFN with no experts |
| | Pipeline parallelism | ❌ Not implemented |
| **Performance** | Continuous batching | ✅ (`max-num-seqs` 1 / 4 / 8 verified) |
| | Chunked prefill (long context) | ✅ (`max-model-len` up to 32768 verified) |
| **Sampling** | On-device sampling (greedy) | ✅ — the recipe default |
| | On-device sampling (`temperature`/`top_p`/`top_k`) | ✅ — verified; omit `all_greedy` |
| | Per-request `seed` | ❌ Platform limitation — see [Sampling](#sampling) |
| **Compilation** | torch.compile (XLA backend) | ✅ |

## Sampling

The serving recipes below set `"on_device_sampling_config": {"all_greedy": "true"}`, which is
the default across this repository. In that mode the compiled sampler returns `argmax`, so a
request's `temperature`, `top_p` and `top_k` are **ignored**. Use it for deterministic output
and the lowest decode latency.

To honour per-request sampling parameters instead, **omit** `on_device_sampling_config`
entirely. `all_greedy` then defaults to `False` and the compiled sampler consumes a
per-request `[top_k, top_p, temperature]` tensor. This was verified on device
(`temperature=1.0, top_p=0.95, top_k=20` produced varied, coherent output on repeated calls)
and costs no host round-trip. Note that the sampler is part of the traced graph, so switching modes needs
its own compilation and cache directory. The model card recommends `temperature=1.0, top_p=0.95,
top_k=20` for thinking mode and `0.7 / 0.80 / 20` for instruct mode. Note that the sampler is
part of the traced graph, so switching modes requires its own compilation and cache directory.

> ⚠️ **Do not send a per-request `seed`.** A request with `temperature > 0` **and** `seed` set takes
> vLLM's `SamplingType.RANDOM_SEED` path, which constructs a `torch.Generator` on the Neuron device;
> PyTorch has no generator registered for that backend, so the worker dies and the engine core goes
> down for **all** in-flight requests. This is a limitation of the Neuron backend rather than of this
> model — the failing path is in the shared model runner, before any model logic, and `all_greedy` does
> not protect against it. For reproducible output use `temperature=0`. Clients that always send `seed`
> (`lm-eval-harness` does) need it stripped from the payload.

## Setup

Nothing to apply to the repository — the model is in-tree on this branch. What follows is the environment
the measured figures were taken with: three recipe flags that are load-bearing, and the compile-time
settings this instance size needs.

### Native integration (nothing to apply)

The model package and every framework touch-point are implemented in-tree on this branch.
Check the branch out and install as usual; there is no patch to apply.

### Step 1: Environment Setup

```bash
# Confirm the device is visible — expect 4 NeuronCores on trn2.3xlarge
neuron-ls

# Point the vLLM/NEFF and NKI compile caches at a large volume (not /)
export VLLM_CACHE_ROOT=/path/to/cache/vllm
export NKI_COMPILE_CACHE_URL=/path/to/cache/nki

# Required recipe flags for this model — all three are load-bearing
# GatedDeltaNet prefill kernel. Unset means "sequential": the exact recurrence, evaluated one
# step at a time, and the conservative default. "parallel" is an algebraically equivalent chunked
# reformulation that is markedly faster on long prefills but solves a matrix inverse in fp32 per
# chunk; both score the same on GSM8K. See Prefill kernel selection below.
# export VLLM_GDN_PREFILL=parallel
export VLLM_UNIFIED_KV_GATHER=1      # shared KV + GDN-state slab, .ap indirect gather
export VLLM_QWEN_QKV_MATMUL=1        # QKV matmul fallback: the fused NKI QKV overruns SBUF at TP=4

# Skip EFA affinity when the instance exposes no EFA device — which includes trn2.3xlarge and
# also single-node deployments on larger instances. Affinity is a performance optimisation,
# not a correctness requirement, so skipping it is safe; drop this only if you have EFA.
export NEURON_SKIP_EFA_AFFINITY=1

# Serial compile/trace workers — parallel workers x this model's graphs x TP4 host-OOMs
# on this instance size (observed as a kernel panic, and as a killed neuronx-cc)
export VLLM_NEURON_PARALLEL_COMPILE_WORKERS=1
export VLLM_NEURON_PARALLEL_TRACE_WORKERS=1

# Long-model compilation needs extended timeouts
export VLLM_NEURON_COMPILATION_TIMEOUT=5400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
```

### Step 2: Download the Model

```bash
# FP8
hf download Qwen/Qwen3.8-27B-FP8 --local-dir /path/to/models/Qwen3.8-27B-FP8
# BF16
hf download Qwen/Qwen3.8-27B     --local-dir /path/to/models/Qwen3.8-27B
```

## Serving

Four recipes follow: FP8, BF16, vision, and the flags that govern long context. `--hf-overrides` and
`--limit-mm-per-prompt` are **mandatory** in the text-only recipes — omitting the `Qwen3_5ForCausalLM`
override takes the multimodal config path, which pins the SSM cache to fp32 and fails to compile at this
instance size.

### Offline inference (`llm.generate()`)

[`examples/vllm_neuron/models/qwen3_5_dense/run.py`](../examples/vllm_neuron/models/qwen3_5_dense/run.py)
applies the recipe flags itself via `os.environ.setdefault`, so the only thing it needs is the checkpoint:
`python run.py --model-checkpoint /path/to/models/Qwen3.8-27B-FP8`. An explicit `export` still wins, since
`setdefault` only fills unset variables.

### Online serving — FP8 (OpenAI-compatible)

```bash
vllm serve /path/to/models/Qwen3.8-27B-FP8 \
  --served-model-name Qwen3.8-27B-FP8 \
  --tensor-parallel-size 4 \
  --optimization-level 2 \
  --max-model-len 32768 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
  --mamba-ssm-cache-dtype bfloat16 \
  --additional-config '{"neuron_config":{"quantization":"fp8","on_device_sampling_config":{"all_greedy":"true"},"num_batched_tokens_buckets":[512],"num_seqs_buckets":[8]}}'
```

`neuron_config.quantization="fp8"` is what selects the ROW fp8 path and injects the required
trn2 fp8 compiler flag. `--hf-overrides` forces the text-only architecture alias so the ViT is
left unbuilt.

### Online serving — BF16

Identical, minus the fp8 selection:

```bash
vllm serve /path/to/models/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --tensor-parallel-size 4 \
  --optimization-level 2 \
  --max-model-len 32768 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
  --mamba-ssm-cache-dtype bfloat16 \
  --additional-config '{"neuron_config":{"on_device_sampling_config":{"all_greedy":"true"},"num_batched_tokens_buckets":[512],"num_seqs_buckets":[8]}}'
```

### Vision serving (image / video)

To serve image and video input, use the `Qwen3_5ForConditionalGeneration` alias, allow multimodal
items, and add a `vision_neuron_config` block. Everything else — the Step 1 recipe flags, the
QKV-matmul workaround — is unchanged, except that prefix caching is **off** here: at
`max-model-len 4096` with a single 4096-token bucket the prompt is not split, and the backend rejects
`--enable-prefix-caching` outright with `ValueError: Automatic Prefix Caching (APC) requires
segmented prefill`. Vision prompts are short, so nothing is lost.

```bash
vllm serve /path/to/models/Qwen3.8-27B-FP8 \
  --served-model-name Qwen3.8-27B-FP8 \
  --tensor-parallel-size 4 \
  --optimization-level 2 \
  --max-model-len 4096 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":1,"video":1}' \
  --hf-overrides '{"architectures":["Qwen3_5ForConditionalGeneration"]}' \
  --mamba-ssm-cache-dtype bfloat16 \
  --additional-config '{"neuron_config":{"quantization":"fp8","on_device_sampling_config":{"all_greedy":"true"},"num_batched_tokens_buckets":[512],"num_seqs_buckets":[1]},"vision_neuron_config":{"num_vision_tokens_buckets":[2048],"vision_attention_block_size":2048,"tp_size":4,"dp_size":1}}'
```

Then post an OpenAI-compatible chat request with an `image_url` or `video_url` content part; a
`data:` base64 URI works, and a video must be a single container URI (`data:video/mp4;base64,...`),
not a list of frames. `num_vision_tokens_buckets` bounds the ViT graph — raise it and recompile if
your media exceeds ~2048 vision tokens after patch-merge. The vision tower is TP-sharded across the
same four cores.

Verified on device at this recipe: HBM 74.28 GB, an image described correctly ("a solid red circle
positioned on the right side of a plain white background") and a video's object *and* motion
described correctly ("a red circle moving from left to right across the screen").

> **Text-only is unaffected.** Serving under `Qwen3_5ForCausalLM` leaves the vision tower unbuilt,
> so the text recipes above and their measured numbers do not change.

### Long context

`max-num-batched-tokens` is the prefill chunk size and must be one of
`{512, 1024, 2048, 4096, 8192}`; the last entry of `num_batched_tokens_buckets` must equal it.
Keeping it at **512** while raising `--max-model-len` is what makes long context work: each
prefill step then feeds at most 512 new tokens, which stays inside the GatedDeltaNet prefill
graph's compile limit.

**Prefix caching is off in the recipes above**, and that is fine at any prompt length: the
GatedDeltaNet state is carried across chunk boundaries either way, so a 25,000-token prefix costs at
most a few points of GSM8K rather than collapsing it ([Accuracy at long context](#accuracy-at-long-context)). Add `--enable-prefix-caching` if your requests
**share a long prefix** — it needs `max-model-len` strictly greater than `max-num-batched-tokens` or it
refuses to start, and it costs throughput when prefixes are not shared
([Measured performance](#measured-performance)).

## Accuracy Evaluation

Evaluated through the OpenAI-compatible endpoint with `lm-eval-harness`.

### GSM8K-CoT

Measured on `trn2.3xlarge` at **TP=4**, FP8, greedy on-device sampling, via
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (`gsm8k_cot`)
driven against the server's OpenAI-compatible endpoint:

| Metric | Value | Stderr |
|:---|:---:|:---:|
| `exact_match` (strict-match) | **94.0%** | ±2.4 |
| `exact_match` (flexible-extract) | **97.0%** | ±1.7 |

Recipe: 100 questions, 4-shot with the chat template applied
(`--apply_chat_template --fewshot_as_multiturn`), `max_tokens 448` and
`enable_thinking: false` so prompt plus generation stays inside the 1024-token window,
`max-model-len 1024`, `max-num-batched-tokens 512`, `--max-num-seqs 1`,
and the default sequential GatedDeltaNet prefill. The ± figures are one binomial `SE` at n=100.

The spread between the two filters is a formatting effect: strict-match requires the answer in
the exact `The answer is X` form, and this model frequently answers correctly without it. Prefer
flexible-extract when comparing across models.

### What FP8 costs in accuracy

Same checkpoint, same serve configuration, same harness; the only difference is whether the MLP weights
stay fp8-resident or are dequantised to bf16 at load. n=100, 4-shot, `max-model-len` 1024,
`max-num-seqs` 1, greedy, `sequential` prefill; 1 SE ≈ 2.2 pt.

| Precision | strict | flex |
|---|---|---|
| BF16 | 0.950 | 0.980 |
| FP8 | 0.940 | 0.970 |

−1.0 pt on both filters, roughly 0.5 SE: **fp8 costs no measurable accuracy** at these dimensions.

### Accuracy at long context

Chunked prefill splits a prompt into `max-num-batched-tokens` pieces, and the 48 GatedDeltaNet layers
carry a recurrent state across those boundaries (the 16 full-attention layers carry theirs in the KV
cache). The table below is the same GSM8K recipe with ~25,000 tokens of content-free filler prepended,
so the prompt spans ~50 prefill chunks instead of two:

| Prompt | `max-model-len` | Filler | Prefill kernel | Prefix caching | strict | flex | n |
|---|---|---|---|---|---|---|---|
| short | 1024 | — | `sequential` | off | **0.940** | 0.970 | 100 |
| long | 32768 | ~25,000 tok | `sequential` | off | **0.880** | 0.920 | 50 |
| long | 32768 | ~25,000 tok | `parallel` | off | **0.900** | 0.933 | 30 |
| long | 32768 | ~25,000 tok | `sequential` | on | 0.900 | 0.967 | 30 |

4-shot, greedy, FP8, `max-num-batched-tokens 512`, `max-num-seqs 1` at every row; one binomial `SE` is
2.4 pt at n=100, 4.6 pt at n=50 and 6 pt at n=30. The long rows sit 4–6 pt below the short one, inside
the error bars (1.2 `SE` on the short-vs-long difference). The claim the table supports is that a
25,000-token irrelevant prefix costs at most a few points, on either prefill kernel and with or without
prefix caching.

### Logit validation (BF16 path)

The generic 3-way validation from `examples/vllm_neuron/accuracy/`: an FP32 CPU baseline fixes the
token sequence, a BF16 CPU run teacher-forced on those tokens gives the expected logits, and the
Neuron target is compared against both. Three prompts × 16 generated tokens, greedy.

The headline metric is the aggregate σ-ratio,
`RMS(device error vs FP32) / RMS(bf16 error vs FP32)`. **Lower is better**, and **σ ≤ 1 means the
device is at least as accurate as running the same model in bf16** — the deviation you already accept
by not using fp32.

| Checkpoint | aggregate σ-ratio | Per-prompt gate |
|---|---|---|
| `Qwen/Qwen3.8-27B` (BF16) | **0.9282** | 2 / 3 |
| `Qwen/Qwen3.6-27B` (BF16) | **0.9448** | 3 / 3 |

Both are inside the bf16 noise floor. Measured on the current code at TP=4, `max-model-len` 512,
`kv_segment_size_buckets: [512]`, the default `sequential` prefill.

> **On Qwen3.8's 2 / 3 — a tolerance artefact, not an accuracy problem.** The gate is a worst-**single-token**
> peak against fixed TopK tolerances, so one token anywhere in the run can fail it while σ — an RMS over
> every token — stays inside the bf16 noise floor. One prompt fails on one token of sixteen, and only on
> K50 (0.0348 against a 0.02 tolerance) and K1000 (0.0416 against 0.03). **K5, which decides which token
> is emitted, and the full vocabulary both pass on every prompt**, so what exceeds tolerance is the
> 50th-to-1000th ranked tail that greedy decoding never consults.

> **Scope: BF16 only.** This harness assumes the device runs the **same dtype** as the baseline it is
> compared against, which is what lets it separate target-specific error from dtype-inherent error.
> Running the FP8 path against a BF16 baseline would fold fp8's own quantisation into the numerator,
> so σ would exceed 1 for a reason that is not a defect. **σ is therefore not a meaningful gate for
> the FP8 path, and GSM8K-CoT remains the accuracy basis there.**

## Measured performance

All figures from a single **trn2.3xlarge**, TP=4, `max-model-len` 4096,
`max-num-batched-tokens` 512, `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.30`, greedy sampling,
natural-text prompts (~200 words in, 128 tokens out). HBM is the sum over all four ranks as reported by
`neuron-monitor`. TTFT is a **p50 over 24 requests** — see [How to read TTFT](#how-to-read-ttft) below,
which matters at concurrency. **Prefix caching is off in every row**, matching the serve recipes above.
Every table covers `max-num-seqs` 1 and 8.

Both prefill kernels were measured on the same hardware with the same recipe, because the choice between
them is a real performance trade — see
[Prefill kernel selection](#prefill-kernel-selection-vllm_gdn_prefill).

**Default — `sequential`** (FP8)

| `max-num-seqs` | HBM used (GB) | Output tok/s | TTFT p50 (ms) | Mean TPOT (ms) |
|---|---|---|---|---|
| 1 | **74.37** | **23.65** | 1315.9 | **32.19** |
| 8 | **74.50** | **60.54** | 6604.1 | **85.52** |

**Opt-in — `VLLM_GDN_PREFILL=parallel`** (FP8)

| `max-num-seqs` | HBM used (GB) | Output tok/s | TTFT p50 (ms) | Mean TPOT (ms) |
|---|---|---|---|---|
| 1 | **74.05** | **27.07** | 369.6 | **34.22** |
| 8 | **74.19** | **98.56** | 1880.2 | **67.56** |

**Batch scaling.** Going from `max-num-seqs` 1 to 8 multiplies output throughput by **2.56×** on
`sequential` and **3.64×** on `parallel` — both sub-linear against the 8× in batch size, and the default
scales less well for the prefill-stall reason below.

### How to read TTFT

The two concurrency regimes measure different things.

* At **`max-num-seqs` 1** nothing else is in flight, so TTFT is the prefill path plus one decode step.
  It is tight and repeatable — of 24 consecutive requests on `sequential`, 23 fell in 1313–1321 ms
  (p50 1315.9) and only the first, cold one sat outside at 1512 ms.
* At **`max-num-seqs` > 1** TTFT additionally includes **waiting for in-flight decode**, because this
  backend runs prefill batch-1 and does not mix it with decode in one batch: an arriving prompt cannot
  start until the current decode step yields. It therefore depends on arrival order and spreads widely
  — at `max-num-seqs` 8 on `sequential`, 24 requests ran min 1367 / p50 6604 / max 10836 ms, because the
  first few met an idle server and the rest queued.

So the concurrency > 1 rows are a queueing-inclusive latency, not a prefill cost. Compare TTFT between
the two kernels only at equal concurrency.

**Why p50 and not the mean.** The first request after startup costs up to 1.9× the steady-state value
(686 ms against a 369.6 ms p50 on `parallel`; 1512 ms against 1315.9 ms on `sequential`). Over only four
requests that single cold sample moves the mean enough to matter: the same four arms measured that way
read 7 % high at `max-num-seqs` 1, which is large enough to invent a regression that is not there.

### Choosing between the two kernels

`sequential` is the default because it evaluates the recurrence definition directly, with no matrix
inverse and no chunk-boundary reconstruction — the conservative choice for a path whose numerics are
load-bearing. It pays **≈3.5× higher TTFT p50** at both concurrencies for that. TPOT moves in the other
direction and then reverses:

| `max-num-seqs` | TPOT `sequential` | TPOT `parallel` | Output tok/s |
|---|---|---|---|
| 1 | **32.19** (6 % better) | 34.22 | 13 % lower |
| 8 | 85.52 (27 % worse) | **67.56** | 39 % lower |

`VLLM_GDN_PREFILL` selects the **decode** implementation as well as the prefill one — the recurrent-state
convention differs between the two forms, so they cannot be mixed. At `max-num-seqs` 1, with nothing else
in flight, the TPOT column is therefore a direct comparison of those two decode paths, and the default's
torch scan is the faster of the two by 6 %.

At higher concurrency TPOT also absorbs *other* requests' prefill time, because prefill runs batch-1 and
is not mixed with decode in one batch, so each prefill stalls every decoding sequence. That is the likely
reason the ordering reverses and the gap grows with concurrency — but the two contributions (decode speed
versus stall time) **were not measured separately**, so read the concurrency > 1 rows as end-to-end
behaviour rather than as a statement about decode. HBM is essentially unaffected (+0.3 GB). Both TPOT
columns reproduced across two independent measurement rounds to within 0.05 %, so these differences are
not run-to-run noise.

Pick `parallel` when time-to-first-token or aggregate throughput at concurrency matters; keep the default
when you want the exact recurrence. The two score the same on GSM8K within noise — see
[Prefill kernel selection](#prefill-kernel-selection-vllm_gdn_prefill).

### FP8 vs BF16

> **What "BF16" means here.** The FP8 checkpoint with its MLP weights dequantised to bf16 at load
> (`VLLM_QWEN38_FP8_MLP_BF16=1`, off by default), **not** the separate `Qwen/Qwen3.8-27B` release — that
> holds the checkpoint and the serve configuration fixed so the only variable is MLP precision. The real
> BF16 release is verified separately (see [Verification scope](#verification-scope)).

On the default `sequential` kernel, both columns on the current code:

| `max-num-seqs` | HBM FP8 / BF16 (GB) | Output tok/s FP8 / BF16 | TTFT p50 FP8 / BF16 (ms) |
|---|---|---|---|
| 1 | **74.37** / 89.66 | **23.65** / 21.65 | 1315.9 / 1331.0 |
| 8 | **74.50** / 89.76 | **60.54** / 57.97 | 6604.1 / 6690.9 |

FP8 saves **−15.3 GB** of HBM at both concurrencies and raises output throughput by **4.4 % to 9.2 %**;
TTFT is unchanged, as expected for a change confined to the decode weight path. The memory saving matches
the arithmetic — the MLP holds ≈17.1B of the 27B parameters, so bf16 ≈34.2 GB against fp8 ≈17.1 GB — and
the speed-up is consistent with decode being byte-bound, since the fp8 kernel halves the weight bytes read
on the hot path. That mechanism sits in the MLP and is independent of the prefill kernel, so it carries
over to the opt-in kernel unchanged. FP8's throughput edge narrows as concurrency rises, as
non-weight-read overheads grow relative to the halved weight traffic.

### What prefix caching costs

Same configuration with `--enable-prefix-caching` added, on `sequential`:

| `max-num-seqs` | Output tok/s off / on | TTFT p50 off / on (ms) | HBM off / on (GB) |
|---|---|---|---|
| 1 | 23.65 / **19.05** (−19.5 %) | 1315.9 / 1279.9 | 74.37 / 74.40 |
| 8 | 60.54 / **54.61** (−9.8 %) | 6604.1 / 6385.7 | 74.50 / 74.53 |

The cost lands on decode throughput; TTFT and HBM are unchanged. **These are worst-case figures**: the
bench varies every prompt at its first tokens, so no request can reuse another's cached blocks — it
measures the bookkeeping with none of the benefit. A workload whose requests share a long prefix skips
re-prefilling the shared part, which can dominate: a 100-question run whose 25,000-token prefix was
identical every time took 18 minutes with caching on against ~3 hours with it off.

### Long context: memory and speed

`max-num-seqs` 1, prefill chunk 512, `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.30`.

| `max-model-len` | FP8 HBM (GB) | BF16 HBM (GB) |
|---|---|---|
| 1024 | 74.36 | — |
| 32768 | **77.96** | 93.25 |

Raising `max-model-len` from 1024 to 32768 costs **+3.6 GB** of HBM, because the KV pool is
sized by the GMU budget fraction rather than by demand. **BF16 reaches 32K as well** — fp8 is not what
makes long context possible here; what fp8 buys is headroom (**18.0 GB** free at 32K against 2.8 GB) and the throughput
above. That headroom is what leaves room to raise the KV budget fraction or the batch size further.

**Throughput and latency at 6,319 prompt tokens**, same recipe at `max-model-len` 32768, 128 output
tokens, every request filling the budget:

| `max-num-seqs` | HBM (GB) | Output tok/s | TTFT p50 (ms) | Mean TPOT (ms) |
|---|---|---|---|---|
| 1 | 77.96 | 5.30 | 19,354 | 37.45 |
| 8 | 78.43 | 6.13 | 97,064 | 627.92 |

Against the ~200-word figures above, throughput falls to 0.22× at `max-num-seqs` 1 and TTFT rises 14.7×,
which is what a 12× longer prompt costs in a prefill that is O(N²) in the full-attention layers.
**Concurrency barely helps at this length** — 5.30 → 6.13 tok/s, against 2.56× on short prompts — because
prefill runs batch-1 and is not mixed with decode, so eight 6,319-token prefills serialise and stall every
decoding sequence. That is also why TPOT rises 17× while the per-token decode cost itself does not change.
Size the batch for short prompts, not for long ones.

The opt-in `parallel` kernel was within 0.3 GB of the default at every length measured (71.90–74.64 GB
across `max-model-len` 1024–16384), so the prefill kernel does not materially
change the long-context memory profile — the trade between the two kernels is latency, not memory. Those
figures predate the state-carry change; the comparison is a difference between kernels, which a uniform
offset does not affect.

## Contents of this bundle

The model is integrated in-tree (committed directly on this branch), so the top-level
`Qwen3.8-27B/` bundle is just this source-of-truth README. There is **no `src/` copy**
here (that would duplicate the in-tree package).

```text
Qwen3.8-27B/
└── README.md            # This file — the source of truth for the Qwen3.8-27B port
```

The paths the port touches are listed below; that listing is the record of which shared
framework files this model changes.

### Paths this port touches across the repository

Everything below is stated against the branch this port is based on,
`release-0.24.0.1.1.0` (vllm-neuron 0.24 / Neuron 2.32).

**New — model package** (`vllm_neuron/model/qwen3_5_dense/`)

```text
vllm_neuron/model/qwen3_5_dense/
├── __init__.py                   # Package export
├── config.py                     # Config plumbing; reads quantization_config → sets quant_scheme on the text config
├── factory.py                    # Model builder wiring
├── model.py                      # The hybrid backbone (48 GatedDeltaNet + 16 full-attention layers), FP8 params + ROW scale buffers, the hybrid CTE/TKG MLP forward, loader wiring
├── quantization.py               # QuantScheme (NONE / FP8_ROW) + quantization_config parsing
├── weight_loaders_bf16.py        # GatedDeltaNet TP head-sharding loaders (in_proj / conv1d / out_proj)
├── weight_loaders_fp8_row.py     # Block dequant → per-output-channel ROW fp8 re-quant; plus block dequant → bf16 for the tensors kept in bf16 (wraps each module's existing loader so GQA/GDN head-aware sharding is reused)
└── README.md                     # Module structure (points here)
```

FP8 is conditional on `quant_scheme`: with a BF16 checkpoint the package takes the unquantized
path and none of the fp8 code runs.

**New — GatedDeltaNet and paged-KV kernels** (carried in-tree so the branch builds and serves
standalone)

```text
vllm_neuron/functional/
├── gated_delta_rule.py             # Chunked GDN prefill entry point (the opt-in VLLM_GDN_PREFILL=parallel kernel)
├── gated_delta_rule_seq.py         # Sequential GDN prefill kernel (the default)
├── gdn_conv_update.py              # Slot-indexed decode conv-state update, addressed by indirect DMA
├── gdn_state_update.py             # Slot-indexed decode recurrent-state update, addressed by indirect DMA
├── paged_kv_gather.py              # Paged prefix-KV gather for the shared unified KV slab
└── linear_attention/
    ├── __init__.py
    ├── gated_delta_rule.py         # Chunked GDN prefill public API
    ├── _gated_delta_rule_kernel.py # The NKI chunk kernel
    ├── causal_conv1d.py            # Short-conv public API
    └── _causal_conv1d_kernel.py    # The NKI short-conv kernel
```

**New — hybrid prefix-cache support**

```text
vllm_neuron/vllm/worker/neuron_mamba_apc.py      # Mamba/GDN state slotting for the prefix-cache path
```

**Modified — shared framework files.** These are the shared files this model changes. The other
checkpoints of the family run through the same files (see
[Compatible checkpoints](#introduction)).

```text
vllm_neuron/functional/mlp.py                              # _can_use_kernel: intermediate-sharding prediction corrected to match the kernel's own thresholds; strip the packed +4 scale tail for plain ROW
vllm_neuron/functional/attention/attention_decode_mask.py  # Derive lnc from s_prior instead of hardcoding 2, so a block_size that is a multiple of 128 but not 256 still yields a valid mask layout
vllm_neuron/model/kv_cache.py                              # HybridKVSpec: KVSpec plus the names of the stateful (non-attention) layers
vllm_neuron/model/registry.py                              # Register Qwen3_5ForConditionalGeneration / Qwen3_5ForCausalLM
vllm_neuron/vllm/attention/attn.py                         # Declare MultipleOf(128) as the decode-mask kernel's block-size alignment
vllm_neuron/vllm/core/scheduler.py                         # KV-pool admission check for hybrid requests, plus a default-off empty-batch diagnostic
vllm_neuron/vllm/platform.py                               # Accept "fp8" as a supported quantization on this platform; hybrid block-size alignment; vision-enabled detection
vllm_neuron/vllm/worker/neuron_model_runner.py             # Hybrid-model metadata: state slots, and has_initial_state so the GatedDeltaNet state carries across prefill chunks
vllm_neuron/vllm/worker/neuron_worker.py                   # Cap the KV budget so the shared unified slab stays inside the indirect-DMA addressing limits
```

**New — offline inference example**

```text
examples/vllm_neuron/models/qwen3_5_dense/run.py   # Applies the recipe flags via os.environ.setdefault; --model-checkpoint is the only required argument
```

**Documentation**

```text
README.md                                 # Contributed Models table row (modified)
docs/model-recipes/index.md               # Grid card + toctree entry (modified)
docs/tutorials/index.md                   # Grid card + toctree entry (modified)
docs/model-recipes/qwen3-8-27b.md         # Model recipe pointer to this README (new)
docs/tutorials/tutorial-qwen3-8-27b.md    # Deployment tutorial pointer to this README (new)
```
