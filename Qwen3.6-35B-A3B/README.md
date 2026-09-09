# Contributed Model: Qwen3.6-35B-A3B (Qwen3.5-MoE hybrid)

vllm-neuron implementation of
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), a **hybrid
Mixture-of-Experts** decoder on vllm-neuron 0.24 / Neuron 2.32. **BF16 and FP8** weights
are both supported, and so are **image and video input**.

This README is the single source of truth for the port: architecture, setup, serving,
verification, and measured performance. The `docs/` recipe and tutorial and the model-package
README are thin pointers to this file.

> On the **Neuron 2.31 / vLLM 0.21** stack, use the
> [`add-qwen36-moe-231`](https://github.com/htokoyo/vllm-neuron/blob/add-qwen36-moe-231/Qwen3.6-35B-A3B/README.md)
> tag instead: it holds the predecessor port for that stack, which this branch supersedes.

## Change History

Newest first. Each entry links to the section with the full detail.

| Date | Change |
|---|---|
| 2026-09-09 | Re-hosted onto vllm-neuron 0.24 / Neuron 2.32, targeting **TP=4 on one `trn2.3xlarge`** with **`ep_degree` 1** and a KV budget cap of **0.05**. Adds **FP8** weights alongside BF16, **image and video** input, and **selective expert loading** in decode. Takes the shared hybrid framework from the dense sibling port. Chunked prefill verified to `max-model-len` 32768. |
| 2026-07-25 | Initial port on vllm-neuron 0.21 / Neuron 2.31, verified at TP8/EP8 on `trn2.48xlarge`. Superseded by this branch, kept at the [`add-qwen36-moe-231`](https://github.com/htokoyo/vllm-neuron/blob/add-qwen36-moe-231/Qwen3.6-35B-A3B/README.md) tag. |

## Introduction

Qwen3.6-35B-A3B is a hybrid Mixture-of-Experts (MoE) language model from the Qwen
team. Its 40 decoder layers follow a 3:1 pattern — **30 GatedDeltaNet
(linear-attention) layers + 10 full-attention layers** — and every token is routed
through a 256-expert top-8 MoE plus a shared expert (35B total parameters, ~3B
active per token). BF16 and FP8 weights are both supported.

The native checkpoint ships a `Qwen3_5MoeForConditionalGeneration` (multimodal)
architecture. **Both aliases are supported and both are documented below**: the text
command loads the decoder via `Qwen3_5MoeForCausalLM` and leave the vision tower unbuilt,
which keeps the compile smaller; the [vision command](#vision-serving-image--video) uses the
multimodal alias and builds the ViT. `--hf-overrides` and `--limit-mm-per-prompt` are what
select between them, and they are **mandatory** for the text path — omitting them lets
vLLM pick the multimodal architecture and build a ViT the text path never uses.

> **Qwen3.5 and Qwen3.6 share this architecture** (weights-only difference; HF loads
> both under `qwen3_5_moe`). Validated on device with Qwen3.6-35B-A3B weights; serves
> Qwen3.5-35B-A3B unchanged.

**Compatible checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Qwen3.6-35B-A3B | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | Trn2 | BF16 |
| Qwen3.5-35B-A3B | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | Trn2 | BF16 (same arch) |
| Qwen3.6-35B-A3B-FP8 | [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | Trn2 | FP8 (block-FP8 → ROW fp8 experts) |

The FP8 checkpoint is supported and correct: faster than BF16 at `max-model-len` 32768, level with
it at 2048, and equal in accuracy. See [FP8](#fp8-resident-per-channel-row-fp8-experts).

## Verification scope

**Measured on this stack** (`trn2.3xlarge`, 1 device / 4 NeuronCores / 96 GiB, LNC=2,
vllm-neuron 0.24 / Neuron 2.32, **TP=4**).

**Verified on device**

| Area | What was verified |
|---|---|
| BF16 serving | Compiles and serves at every configuration below; every probe coherent |
| FP8 serving | Compiles and serves on both prefill routes; every probe coherent |
| Accuracy | GSM8K-CoT 4-shot on both dtypes and both expert layouts, at two prompt lengths: **short** (0.89–0.94, `n=100`) and **~25k tokens** (0.8500 against 0.8950 short at `n=200`, a 4.5-point cost). ⚠️ Scored with a **no-think** task variant, because this checkpoint otherwise emits a `<think>` trace that the stock task cannot terminate. See [Accuracy Evaluation](#accuracy-evaluation) |
| Long context | Chunked prefill to `max-model-len` 32768 — 25k-token prompts spanning 50 chunks, with prefix caching off and on. **BF16 and FP8 both serve 32768, with and without expert parallelism** |
| Retrieval from a long prompt | A unique 6-digit code at 2% / 50% / 98% depth of a ~25k-token prompt, asked for at the end, was returned **90/90** |
| Selective expert loading | Decode at `ep_degree` 1 reads only the experts the batch routed to, on both dtypes, with no accuracy difference |
| Batching | Concurrency from 1 up to the kernel ceiling of 16, at both `max-model-len` 2048 and 32768, on both dtypes and both expert layouts. Per-point figures are in [Measured performance](#measured-performance) |
| Memory | Device HBM via `neuron-monitor`, summed over the 4 ranks, sampled to steady state. **At the shipped cap and `max-model-len` 2048**: BF16 **72.33 GiB**, FP8 **43.95 GiB** — FP8 frees **28.38 GiB**. ⚠️ Quote these only with their cap and length; the cap moves the total |
| Throughput / latency | Output tok/s and **p50** TTFT / TPOT on natural-text prompts, for all four text configurations at the shipped cap, every arm at the default MoE prefill `block_size` **128** (see [Measured performance](#measured-performance)) |
| KV capacity | KV pool read from the server at startup at four `max-model-len` values, identical on both dtypes and both expert layouts — see [the cap](#the-kv-budget-cap-and-what-it-decides) |
| Prefill kernels | `VLLM_GDN_PREFILL` grouped / parallel / sequential all correct; `grouped` fastest, and it is the default |
| Sampling | Greedy on-device sampling |
| Vision (image / video) | A synthetic three-shape image described correctly (colour and position of all three), and the object and direction of travel over 16 frames |
| Documented commands | Each was run exactly as written; the offline example with **no environment variables set**, which is what its `os.environ.setdefault` block claims |

**Not verified / out of scope**

| Area | Status |
|---|---|
| FP8 KV cache | Not implemented (separate from FP8 weights, which are supported) |
| Pipeline parallelism | Not implemented |
| Speculative decoding | Not exercised |
| Multi-instance / multi-node | Not exercised — a single `trn2.3xlarge` only |
| Tensor/expert-parallel degrees other than TP=4 with `ep_degree` 1 or 4 | Not exercised |
| Prompts at `max-model-len` 65536 | **Starting and serving is verified; nothing else is** |
| `run_logit_validation_offline.py`'s stock thresholds | **Not usable as a gate on this model** — its 0.011 top-5 tolerance is below the BF16 noise floor (the CPU baseline's own L-inf error against FP32 is 0.03–0.11), so it reports FAILED on a healthy model. Gate with GSM8K ([Accuracy Evaluation](#accuracy-evaluation)) instead |
| Per-request `seed` | **Unsupported — see [Sampling](#sampling)** |
| Reading a long **relevant** document | Only a content-free filler was used, so comprehension across a long document is untested |
| Correctness of the GatedDeltaNet state carry | Bounded, not proven — no measurement isolates it |
| `VLLM_QWEN_FLASH_ATTN=1` | **Not exercised on this port.** Default **off**, and every figure here is from the default (eager) decode attention path |

## Model Architecture

| Property | Value |
|---|---|
| Architecture aliases | `Qwen3_5MoeForCausalLM` (text), `Qwen3_5MoeForConditionalGeneration` (text + vision) |
| `model_type` | `qwen3_5_moe` |
| Layers | 40 = 30 GatedDeltaNet + 10 full attention (`full_attention_interval=4`) |
| Full-attn layer indices | 3, 7, 11, … , 39 |
| Hidden size | 2048 |
| Attention heads | 16 query / 2 KV (GQA), `head_dim` 256 |
| Attention output gate | Yes (`attn_output_gate=True`) |
| RoPE | Partial (`partial_rotary_factor=0.25` → 64 rotated dims), M-RoPE |
| GatedDeltaNet | 16 key heads / 32 value heads (K repeated 2× to V), head dim 128, conv kernel 4 |
| Experts | 256, top-8 per token, plus one shared expert |
| MoE intermediate | 512 |
| Vocab size | 248320 |
| Max position embeddings | 262144 |
| SSM state dtype | **Forced bf16, not tunable** (see below) |

**Hybrid state.** The 10 full-attention layers use a paged KV cache; the 30 GatedDeltaNet
layers keep their recurrent + conv state in the same unified paged pool, through vLLM's
hybrid-model interfaces.

**Parallelism — the recommended layout is `TP=4` with `ep_degree` 1 (no expert parallelism).**
`tensor_parallel_size` must divide the 16 full-attention heads, so 4, 8 and 16 are the candidates;
**TP=4** is one `trn2.3xlarge` device. The two expert layouts are not a capacity trade-off: EP=4
holds 64 experts at the full intermediate width per rank, EP=1 holds all 256 with the intermediate
width split across the 4 ranks, and **256 × 512/4 = 64 × 512, so the resident bytes are identical.**
EP=1 is recommended because only it can skip the experts a batch did not route to
([selective expert loading](#selective-expert-loading-decode)), and EP=4 was not faster at any
concurrency measured — the numbers are in [Measured performance](#measured-performance). Accuracy is
the same either way.

### Forced bf16 GatedDeltaNet state (not tunable)

The port pins the recurrent SSM state to bf16 regardless of `--mamba-ssm-cache-dtype`, and this
is load-bearing: an fp32 state grows the mamba page, pushes the KV page `block_size` up, and
drives the GatedDeltaNet kernels outside their validated regime, at which point decode diverges
to out-of-vocabulary tokens. This checkpoint's `text_config` declares
`mamba_ssm_dtype: float32`, which is exactly the value the pin overrides — pass
**`--mamba-ssm-cache-dtype bfloat16`** explicitly so the resolved configuration is unambiguous.
(That KV page `block_size` is unrelated to the MoE prefill block size in
[Measured performance](#measured-performance): one is in KV slots and framework-derived, the
other is in tokens and set by `VLLM_QWEN35_MOE_BLOCK_SIZE`.)

### Prefill kernel selection (`VLLM_GDN_PREFILL`)

The GatedDeltaNet layers admit more than one prefill formulation, and this port carries **three NKI
kernels** for them — listed in the order they were added, so the progression is visible:

| Mode | How it computes the recurrence |
|---|---|
| `sequential` | The recurrence exactly as defined, advanced one token at a time inside a single bounded NKI kernel (`nl.sequential_range`) |
| `parallel` | An algebraically equivalent chunked form: within each 64-token chunk the inter-token coupling is resolved by solving `T = (I − A)⁻¹` by blocked forward substitution, so chunks are processed as matmuls instead of as a step loop |
| **`grouped`** (**default**) | The chunked form as well, but the intra-chunk solve is split — forward substitution *between* 16×16 sub-blocks, recursive doubling *inside* each — and eight chunks are processed as one group so their per-sub-block work packs onto the 128-partition axis. It also takes a carry-in state, so a chunked prefill resumes from the previous boundary instead of restarting |

All three compute the same recurrence and all three are correct on device; they differ in how much of
it is turned into matmuls. `grouped` is the default because it is the fastest, and there is no reason
to change it — every figure in this README uses it.

Decode follows whichever mode is selected — the three are not independently settable, because a
prefill state produced by one form and advanced by another's decode path diverges to NaN.

### MoE expert-GEMM route (`VLLM_QWEN35_MOE_FP8_PREFILL`, FP8 only)

A separate knob, on a different sub-block of the same layer. It applies to the MoE FFN in **all 40**
layers and exists only on the FP8 path:

| Mode | What it does |
|---|---|
| **`dequant_to_bf16`** (**default**) | Upcasts each layer's experts to bf16 ahead of the kernel so the `shard_on_block` route becomes legal |
| `shard_on_i` | Feeds the fp8 weights and their ROW scales straight to the kernel, which splits the work on the intermediate dimension |

Both compute the same product — the choice is *where* the dequantisation happens, not whether. It
changes nothing in decode (the experts stay fp8-resident either way and `moe_block_tkg` runs its ROW
path), so this knob moves prefill only. `dequant_to_bf16` is the default because it is faster at
every concurrency measured ([Measured performance](#measured-performance)).

### Selective expert loading (decode)

At `ep_degree` 1 the decode path loads only the experts the current batch actually routed to,
instead of every local expert on every step. It engages when `top_k × max-num-seqs` is below the
256 experts — always true here, since the concurrency ceiling is 16 and `top_k` is 8. Reading 8 of
256 experts instead of all of them is what makes the default configuration faster than `ep_degree` 4
([Measured performance](#measured-performance)), for identical accuracy.

> ⚠️ **The saving shrinks as the batch grows**, in proportion to `top_k × max-num-seqs ÷ 256`: 1/32
> of the expert reads at 1 concurrent request, but 1/2 at 16, and nothing at all from 32 up. That
> decay is why the EP-free advantage is large at c=1 and gone by c=16.

> 🔴 **It cannot be combined with expert parallelism.** The decode kernel's `rank_id` — the tensor
> that tells a rank which slice of the global expert ids it owns — is consumed only by its
> all-expert branch, so an expert-parallel rank would index global expert ids into its local expert
> weights. `ep_degree` 4 therefore always loads all local experts.

### FP8 (resident per-channel ROW fp8 experts)

`Qwen/Qwen3.6-35B-A3B-FP8` is a DeepSeek-style `[128,128]` block-FP8 checkpoint. It is supported:
the block scales are dequantised at load and the **expert** weights are re-quantised to
per-output-channel ROW fp8 — almost all of this model's parameters are expert weights. Everything
else (attention QKV/o_proj, the GatedDeltaNet projections, the shared expert) is
block-dequantised to bf16 at load, which costs little because those tensors are a small fraction
of the model. Serving it requires `neuron_config.quantization: "fp8"`, which is load-bearing
rather than a label — see [Online serving](#online-serving-openai-compatible).

**FP8 halves the resident expert bytes**, cutting the device total from **72.33 to 43.95 GiB** — a
**28.38 GiB** saving, both measured at the shipped cap and `max-model-len` 2048. That headroom is
what lets the KV pool run at caps BF16 cannot reach: above 0.08 BF16 fails to load the NEFF
(`NRT_RESOURCE`) **at any `max-model-len`, including 256**.

🔑 **FP8 is faster at `max-model-len` 32768** on decode-dominant traffic — both at a matched cap
(+9.8%) and at each dtype's best reachable operating point (+11.0%, because only FP8 can raise the
cap far enough for 8 concurrent sequences at that length). It is level with BF16 at 2048, and
accuracy is equal either way. The numbers are in [Measured performance](#measured-performance).

## Feature status

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| | Vision — image | ✅ See [vision serving](#vision-serving-image--video) |
| | Vision — video | ✅ See [vision serving](#vision-serving-image--video) |
| **Quantization** | BF16 weights | ✅ |
| | FP8 weights | ✅ See [FP8](#fp8-resident-per-channel-row-fp8-experts) |
| | FP8 KV cache | ❌ Not implemented |
| **Parallelism** | Tensor parallelism (TP) | ✅ TP=4 — the recommended layout |
| | Expert parallelism (EP) | ✅ EP=4 supported, **not recommended** — see [Parallelism](#model-architecture) |
| | Pipeline parallelism | ❌ Not implemented |
| **Performance** | [Selective expert loading](#selective-expert-loading-decode) (decode) | ✅ `ep_degree` 1 only |
| | Continuous batching | ✅ |
| | Chunked prefill | ✅ |
| | Prefix caching (APC) | ✅ Verified at long context |
| **Sampling** | On-device sampling (greedy) | ✅ — the default |
| | Per-request `seed` | ❌ Platform limitation — see [Sampling](#sampling) |
| **Compilation** | torch.compile (XLA backend) | ✅ |

## Sampling

The serving command below sets `"on_device_sampling_config": {"all_greedy": "true"}`,
which is the default across this repository. In that mode the compiled sampler returns
`argmax`, so a request's `temperature`, `top_p` and `top_k` are **ignored**. Use it for
deterministic output and the lowest decode latency. To honour per-request sampling
parameters instead, **omit** `on_device_sampling_config` entirely; the sampler is part
of the traced graph, so switching modes needs its own compilation and cache directory.

> ⚠️ **Do not send a per-request `seed`.** A request with `temperature > 0` **and**
> `seed` set takes vLLM's `SamplingType.RANDOM_SEED` path, which constructs a
> `torch.Generator` on the Neuron device; PyTorch has no generator registered for that
> backend, so the worker dies and the engine core goes down for **all** in-flight
> requests. This is a limitation of the Neuron backend rather than of this model — the
> failing path is in the shared model runner, before any model logic, and `all_greedy`
> does not protect against it. For reproducible output use `temperature=0`. Clients
> that always send `seed` (`lm-eval-harness` does) need it stripped from the payload.

## Setup

Nothing to apply to the repository — the model is in-tree on this branch.

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

# GatedDeltaNet prefill kernel. Unset means "grouped": the chunk-group form.
# "parallel" and "sequential" are the earlier kernels, kept for comparison and for
# anyone who wants the exact recurrence. See GatedDeltaNet prefill kernel selection.
# export VLLM_GDN_PREFILL=parallel     # or sequential; unset = grouped
export VLLM_UNIFIED_KV_GATHER=1      # shared KV + GDN-state slab, .ap indirect gather

# Skip EFA affinity when the instance exposes no EFA device — which includes
# trn2.3xlarge and also single-node deployments on larger instances. Affinity is a
# performance optimisation, not a correctness requirement, so skipping it is safe.
export NEURON_SKIP_EFA_AFFINITY=1

# Serial compile/trace workers — parallel workers x this model's graphs x TP4 can
# host-OOM on this instance size (observed as a kernel panic, and as a killed
# neuronx-cc). Drop these on a larger-RAM host if compile is slow.
export VLLM_NEURON_PARALLEL_COMPILE_WORKERS=1
export VLLM_NEURON_PARALLEL_TRACE_WORKERS=1

# Large-model compilation needs extended timeouts
export VLLM_NEURON_COMPILATION_TIMEOUT=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# 🔴 REQUIRED on both dtypes, not a tuning knob: at the framework default
# (0.30) BF16 fails to load the NEFF with `NRT_RESOURCE`.
# What the cap decides, and what changing it costs: see "The KV budget cap".
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.05
```

### Step 2: Download the Model

```bash
# BF16
hf download Qwen/Qwen3.6-35B-A3B     --local-dir /path/to/models/Qwen3.6-35B-A3B
# FP8
hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir /path/to/models/Qwen3.6-35B-A3B-FP8
```

## Serving

The `hf-overrides` and `limit-mm-per-prompt` flags select the text-only decoder path
and are **mandatory** — omitting the `Qwen3_5MoeForCausalLM` override forces the
multimodal config path, whose fp32 SSM cache changes the resolved block size.

### Offline inference (`llm.generate()`)

The ready-to-run example
[`examples/vllm_neuron/models/qwen3_5_moe/run.py`](../examples/vllm_neuron/models/qwen3_5_moe/run.py)
sets the required flags via `os.environ.setdefault`, so it works standalone:

```bash
python examples/vllm_neuron/models/qwen3_5_moe/run.py \
    --model-checkpoint /path/to/Qwen3.6-35B-A3B
```

### Online serving (OpenAI-compatible)

```bash
vllm serve /path/to/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 1 \
    --mamba-ssm-cache-dtype bfloat16 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --hf-overrides '{"architectures":["Qwen3_5MoeForCausalLM"]}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "bf16",
            "ep_degree": 1,
            "on_device_sampling_config": {"all_greedy": "true"},
            "num_batched_tokens_buckets": [512],
            "num_seqs_buckets": [1]
        }
    }'
```

**Configuration notes.** The command above is the shipped default. Each knob below is independent;
nothing else needs to change.

- **`quantization` + the checkpoint** — `bf16` with `Qwen3.6-35B-A3B`, `fp8` with
  `Qwen3.6-35B-A3B-FP8`. They must match, and `"fp8"` is required rather than optional on the FP8
  checkpoint. FP8 halves the expert weights, 16.36 → 8.89 GiB per core.
- **`ep_degree`** — **leave it at 1.** 4 is verified as well, but 1 is always faster
  ([Measured performance](#measured-performance)). If you do set 4, set `--enable-expert-parallel`
  at the same time.
- **`--max-model-len`** — declare what you actually serve. It divides the KV pool to set the
  concurrency ceiling, so over-declaring costs concurrency for nothing: on 6.1k-token prompts,
  8192 instead of 32768 gave **1.47× the throughput and half the TTFT**. Chunked prefill is
  verified with prompts up to 32768.
- **`--max-num-batched-tokens`** — the prefill chunk size. Keep **512** above `max-model-len` 2048;
  raising it takes long prompts outside the GatedDeltaNet prefill graph. At 2048 with ~1024-token
  prompts, 1024 cuts **TTFT p50 by 33%** (379 → 252 ms) with TPOT unchanged.
- **`--max-num-seqs`** (and `"num_seqs_buckets"`, which must equal it) — 🔴 **16 is the maximum**:
  the GatedDeltaNet decode kernel supports up to 16 sequences. At long context the KV pool caps it
  lower still — see [The KV budget cap](#the-kv-budget-cap-and-what-it-decides).
- **KV-budget cap** (`VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION`, [Step 1](#step-1-environment-setup))
  — **`0.05` on both dtypes**, and the single most valuable knob here. Details and limits in
  [The KV budget cap](#the-kv-budget-cap-and-what-it-decides).
- **`--enable-prefix-caching`** — optional, for requests sharing a long prefix. Accuracy-neutral at
  25k tokens.

> **Buckets** are the graph shapes that get compiled, and matching them to your traffic is what
> pays: with ~1024-token prompts, a 1024 prefill bucket instead of 512 cut TTFT p50 by **33%**. So
> if your prompt lengths or batch sizes vary, list the sizes you actually serve. The last entry of
> `num_batched_tokens_buckets` must equal `max-num-batched-tokens`.

Once the server is up, send requests using the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
resp = client.chat.completions.create(
    model="Qwen3.6-35B-A3B",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

#### Vision serving (image / video)

To serve image and video input, use the `Qwen3_5MoeForConditionalGeneration` alias and allow
multimodal items. Everything else — the Step 1 environment flags, the KV-budget cap for BF16,
the bucket rules — is unchanged:

```bash
vllm serve /path/to/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 1 \
    --mamba-ssm-cache-dtype bfloat16 \
    --limit-mm-per-prompt '{"image":1,"video":1}' \
    --hf-overrides '{"architectures":["Qwen3_5MoeForConditionalGeneration"]}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "bf16",
            "ep_degree": 1,
            "on_device_sampling_config": {"all_greedy": "true"},
            "num_batched_tokens_buckets": [512],
            "num_seqs_buckets": [1]
        }
    }'
```

Then post an OpenAI-compatible chat request with an `image_url` or `video_url` content part
(a video is one media item, not a list of frames). The ViT is the Qwen3-VL tower, reused
byte-for-byte from the dense sibling's package, and this checkpoint's `model.visual.*`
tensors match it key-for-key.

**Verified on device with this exact command** — one server with both media limits at 1 and
no expert parallelism: a synthetic three-shape image described with the correct colour and
position of all three shapes (red circle upper-left, green triangle, blue square), and a
16-frame clip described with the correct object and direction of travel (a red circle moving
left to right).

## Accuracy Evaluation

GSM8K-CoT, 4-shot, n=100, greedy on-device sampling, raw completions, 1 concurrent request, at the
shipped cap 0.05:

| Arm | strict-match | flexible-extract |
|---|:---:|:---:|
| BF16, no expert parallelism (the default) | 0.8900 | 0.9000 |
| BF16 at EP=4 | 0.9200 | 0.9200 |
| FP8 | 0.9400 | 0.9400 |

All three are equal within the standard error (±0.024–0.031): neither re-quantising the experts to
ROW fp8 nor dropping expert parallelism costs measurable accuracy.

**Long prompts** — ~25k tokens of content-free filler spans 50 prefill chunks instead of two. Same
server and `max-model-len` 32768 in every row:

| Arm | Prompt | Concurrency | Prefix caching | strict | flex | n |
|---|---|:---:|:---:|:---:|:---:|:---:|
| BF16 | short (no filler) | 1 | off | **0.8950** | 0.9050 | 200 |
| BF16 | ~25k filler | 1 | off | **0.8500** | 0.8600 | 200 |
| BF16 | ~25k filler | 4 asked, 3 served | off | 0.8550 | 0.8700 | 200 |
| FP8 | ~25k filler | 1 | off | 0.8300 | 0.8300 | 200 |

**Long context costs 4.5 points — 1.35 standard errors.** Batching does not add to it (3 concurrent
scores the same as 1), and retrieval *from* the long prompt is unaffected — a unique 6-digit code at
2% / 50% / 98% depth came back **90/90**. ⇒ what degrades is reasoning while holding a long context,
not recall out of it. ⚠️ The KV budget cap does not affect any of this (two caps, identical scores).

> 🔴 **Use a no-think GSM8K variant.** This checkpoint emits a `<think>` trace the stock `gsm8k_cot`
> stop sequences do not terminate, so the stock task scores **0.3000 / 0.3400** on a model that
> solves ~0.9 of the same set. Scoring goes through raw `/v1/completions`, so
> `enable_thinking: false` cannot be passed and the marker has to be in the prompt: append the chat
> template's own `enable_thinking: false` output to `doc_to_text`, leaving everything else stock —
>
> ```yaml
> doc_to_text: "Q: {{question}}\n\nA: <think>\n\n</think>\n\n"
> ```

## Measured performance

`vllm bench serve`, natural-text prompts, 1024 input / 256 output tokens, `--ignore-eos`,
`max-model-len 2048`, `max-num-batched-tokens 512`, MoE prefill `block_size` **128** (the
default) on **every** arm, both prefill routes at their defaults. Concurrency equals
`max-num-seqs`, and `n` is the request count. Every run completed every request, and **every
figure here is text-only**.

> ⚠️ **Do not benchmark this model with a random-token dataset.** Uniformly sampled token ids drive
> decode outside the trained vocabulary on this stack and have stopped the engine with a NaN. Use
> natural text.

> **TTFT and TPOT are the median (p50) over `n` requests, not the mean** — the first request after
> startup is a one-time warm-up that pulls the mean ~5% above the p50, and above concurrency 1 the
> TTFT standard deviation is **26–53% of the p50** because prompts queue (the backend does not mix
> prefill with decode in one step, and this model prefills one sequence at a time — see
> [Sequence parallelism](#sequence-parallelism)). **So TTFT above concurrency 1 is a
> queueing-inclusive latency, not a prefill cost**, comparable only between arms at equal concurrency.

**Throughput and latency.** All four text configurations at the **shipped cap 0.05**,
`max-model-len` 2048, 1024-in/256-out. Every arm completed every request.

**`ep_degree` 1 — the shipped layout:**

| dtype | c | Output tok/s | Total tok/s | TTFT p50 | TPOT p50 |
|---|:---:|:---:|:---:|:---:|:---:|
| **bf16** (shipped) | 1 | **53.42** | 266.82 | 366 ms | **17.30 ms** |
| **bf16** | 4 | 115.92 | 579.18 | 982 ms | 31.00 ms |
| **bf16** | 16 | 179.56 | 897.10 | 3,122 ms | 77.32 ms |
| fp8 | 1 | 50.52 | 252.34 | 472 ms | 17.95 ms |
| fp8 | 4 | **120.84** | **603.74** | 1,256 ms | 28.51 ms |
| fp8 | 16 | **192.62** | **962.33** | 4,034 ms | **67.67 ms** |

**`ep_degree` 4 — supported, not recommended:**

| dtype | c | Output tok/s | Total tok/s | TTFT p50 | TPOT p50 |
|---|:---:|:---:|:---:|:---:|:---:|
| bf16 | 1 | 20.48 | 102.27 | **247 ms** | 48.00 ms |
| bf16 | 4 | 69.32 | 346.37 | **698 ms** | 55.43 ms |
| bf16 | 16 | 189.93 | 948.89 | **2,122 ms** | 76.37 ms |
| fp8 | 1 | 22.84 | 114.10 | 383 ms | 42.37 ms |
| fp8 | 4 | 71.79 | 358.70 | 1,023 ms | 52.14 ms |
| fp8 | 16 | 180.53 | 901.94 | 3,269 ms | 76.27 ms |

Three things the table says:

- 🔑 **No expert parallelism is the right default, at every concurrency measured.** Without EP,
  decode loads only the experts the batch routed to, which is worth **2.61×** the output throughput
  at 1 concurrent request and **1.67×** at 4. At 16 — where that advantage has mostly decayed — the
  fastest arm in the whole table is still an EP-free one (**fp8 / `ep_degree` 1, 192.62**), above the
  best EP=4 arm (189.93). ⇒ **`ep_degree` 4 is not the fastest choice at any concurrency here.**
  It does win TTFT everywhere, by making prefill's expert GEMMs 4× wider; that is a one-time cost per
  request, while the decode penalty is paid on every token — which is why TPOT is 1.5–2.8× worse at
  low concurrency. Memory is identical either way (256 × I/4 = 64 × I), so EP buys no capacity.
- **A *smaller* KV pool is faster, which is why the cap ships at 0.05.** Raising it costs throughput
  at every step measured — up to **9.6%** for a modest increase and **16.5–37.5%** at the framework
  default — and changes no accuracy. The per-step cost tracks the pool's **bytes**, so size the pool
  to what the concurrency ceiling can actually use and no larger
  — see [the cap](#the-kv-budget-cap-and-what-it-decides).
- **FP8 pays at long context.** At `max-model-len` 2048 it lands within **0.95–1.12×** of BF16; at
  32768 on decode-dominant traffic it is **+9.8%** (88.93 against 80.97 at the same cap and
  concurrency). Accuracy is equal throughout. Its freed memory also buys concurrency BF16 cannot
  reach at that length.

**Each dtype's best operating point at `max-model-len` 32768**, same 256-in / 2048-out shape:

| dtype | cap | `max-num-seqs` | Output tok/s |
|---|:---:|:---:|:---:|
| bf16 | 0.08 (its ceiling — it cannot load higher) | 5 | 91.08 |
| **fp8** | 0.30 | **8** | **101.07** |

⇒ **FP8's reachable optimum is 11.0% above BF16's** at this length, because only FP8 can raise the
cap far enough for 8 concurrent sequences. That is the one place the freed memory converts into
throughput.

### The KV budget cap, and what it decides

`VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` sets the KV pool, and the pool sets how many requests can
be in flight. 🔑 **vllm-neuron's own default is 0.30; this port sets it to 0.05** in
[Step 1](#step-1-environment-setup) — "the shipped cap" below always means **0.05**, the value this
README recommends and measured everything at, not a framework default.

```
effective concurrency = min( max_num_seqs ,  KV pool ÷ max_model_len ,  16 )
```

> ⚠️ The middle term is a **worst case** — the scheduler assumes every request fills
> `max_model_len`. Exceeding it does not fail to start; the server admits fewer requests and queues
> the rest, which is expensive (declaring 32768 for 6.1k-token prompts cost **32% of output
> throughput** against declaring 8192). Only `max_num_seqs` above 16 fails outright.

Measured pools at the shipped cap (0.05), identical on both dtypes and both expert layouts:

| `max-model-len` | KV pool | ceiling = pool ÷ `max-model-len` |
|---|:---:|:---:|
| 2048 | **68,494 tokens** | **33** (the kernel caps it at 16) |
| 8192 | **98,631 tokens** | **12** |
| 32768 | **110,822 tokens** | **3** |
| 65536 | **113,369 tokens** | **1** |

**`max-model-len` is the stronger lever — set it to what you actually serve, then leave the cap
alone unless the ceiling above is what binds you.**

- **At 2048, keep 0.05.** The ceiling is already 33, so the kernel's 16 binds; a larger pool only adds
  capacity you cannot reach, and it measures slower.
- **At 32768, shortening the length beats raising the cap.** 16 concurrent sequences at 8192 return
  **196.77** output tok/s, nearly 2× the best 32768 result (101.07). If you must serve 32768, raising
  the cap does help — BF16 **91.08** at 0.08 with 5 concurrent, FP8 **101.07** at the framework
  default with 8 — but only up to those measured points; do not extrapolate past them.
- ⚠️ **Never below 0.05.** A floor (`VLLM_NEURON_MIN_KV_BUDGET_GIB`) rejects anything under ≈0.045,
  and the pool also holds the prefix cache — 51,370 with caching on at 2048 against 68,494 off.

### Sequence parallelism

The GatedDeltaNet backbone runs **sequence-parallel during prefill** whenever `world_size > 1`,
which is every configuration here. There is nothing to enable and no flag to tune. Hidden states are
`all_gather`ed to the full sequence at entry, each rank runs its own head slice over that full
sequence, and the `out_proj` partial sum is `reduce_scatter`ed back at exit; decode uses a plain
`all_reduce` instead.

⇒ **prefill processes one sequence at a time**, which is why TTFT above concurrency 1 is
queueing-inclusive.

## Contents of this bundle

The model is integrated in-tree (committed directly on this branch), so the top-level
`Qwen3.6-35B-A3B/` bundle is just this source-of-truth README. There is **no `src/` copy**
here (that would duplicate the in-tree package).

```text
Qwen3.6-35B-A3B/
└── README.md            # This file — the source of truth for the Qwen3.6-35B-A3B port
```

The paths the port touches are listed below, split into what it adds and what it changes;
the second list is the record of which shared framework files this model touches.

### Paths this port touches across the repository

Everything below is stated against the branch this port is based on,
`release-0.24.0.1.1.0` (vllm-neuron 0.24 / Neuron 2.32).

**New — model package** (`vllm_neuron/model/qwen3_5_moe/`)

```text
vllm_neuron/model/qwen3_5_moe/
├── __init__.py                   # Package exports
├── config.py                     # HF → Neuron config translation: the 3:1 hybrid layer pattern, the 256-expert MoE dims, the GatedDeltaNet dims; reads quantization_config → sets quant_scheme
├── factory.py                    # Model builder + weight-loader wiring + the hybrid mamba-state classmethods
├── model.py                      # The hybrid MoE decoder (30 GatedDeltaNet + 10 full-attention layers, 256-expert top-8 MoE + shared expert), the FP8 params and ROW scale buffers, the prefill/decode MoE kernel calls, and the ViT wiring for the multimodal alias
├── quantization.py               # QuantScheme (NONE / FP8_ROW) + quantization_config parsing
├── weight_loaders_bf16.py        # BF16 expert fusion / TP-EP sharding loaders, and the GatedDeltaNet head-aware loaders
├── weight_loaders_fp8_row.py     # Block-FP8 → per-output-channel ROW fp8 re-quant at load; plus block dequant → bf16 for the tensors kept in bf16 (wraps each module's existing loader so the GQA/GDN head-aware sharding is reused verbatim)
└── README.md                     # Module structure (points here)
```

FP8 is conditional on `quant_scheme`: with a BF16 checkpoint the package takes the
unquantized path and none of the fp8 code runs.

**New — GatedDeltaNet and paged-KV kernels** (carried in-tree so the branch builds and serves
standalone)

```text
vllm_neuron/functional/
├── gated_delta_rule.py             # Chunked GDN prefill entry point (VLLM_GDN_PREFILL=parallel)
├── gated_delta_rule_seq.py         # Sequential GDN prefill kernel (VLLM_GDN_PREFILL=sequential)
├── gdn_conv_update.py              # Slot-indexed decode conv-state update, addressed by indirect DMA
├── gdn_state_update.py             # Slot-indexed decode recurrent-state update, addressed by indirect DMA
├── paged_kv_gather.py              # Paged prefix-KV gather for the shared unified KV slab
└── linear_attention/
    ├── __init__.py
    ├── gated_delta_rule.py         # Chunked GDN prefill public API
    ├── _gated_delta_rule_kernel.py # The NKI chunk kernel (parallel) and the decode single-step kernel
    ├── _gated_delta_rule_grouped_kernel.py  # The NKI chunk-group prefill kernel (the default), fp32, carry-in state
    ├── causal_conv1d.py            # Short-conv public API
    └── _causal_conv1d_kernel.py    # The NKI short-conv kernel
```

**New — hybrid prefix-cache support**

```text
vllm_neuron/vllm/worker/neuron_mamba_apc.py      # Mamba/GatedDeltaNet state slotting for the prefix-cache path (`align` mode)
```

**New — example and docs**

```text
examples/vllm_neuron/models/qwen3_5_moe/run.py   # Offline inference example; applies the recipe flags via os.environ.setdefault so it runs standalone
docs/model-recipes/qwen3-6-moe.md                # Recipe page — a pointer to this README
docs/tutorials/tutorial-qwen3-6-moe.md           # Tutorial page — a pointer to this README
```

**Modified — shared framework files.** These are the shared files this model changes.

```text
vllm_neuron/functional/attention/attention_decode_mask.py  # Decode-mask reuse on the hybrid path
vllm_neuron/model/kv_cache.py                              # HybridKVSpec: KVSpec plus the names of the stateful (non-attention) layers
vllm_neuron/model/registry.py                              # Register Qwen3_5MoeForConditionalGeneration / Qwen3_5MoeForCausalLM
vllm_neuron/vllm/attention/attn.py                         # Neuron attention backend: SSM awareness for the block-size aligner
vllm_neuron/vllm/core/scheduler.py                         # KV-pool admission check for hybrid requests
vllm_neuron/vllm/platform.py                               # Accept "fp8" as a supported quantization on this platform (without it vLLM rejects the FP8 checkpoint before the model is built); hybrid page-size alignment; mamba-cache-mode guard; hybrid+APC async-scheduling force-off; vision-bucket gating
vllm_neuron/vllm/worker/neuron_model_runner.py             # Hybrid-model metadata: state slots, the KV budget cap, and has_initial_state so the GatedDeltaNet state carries across prefill chunks
vllm_neuron/vllm/worker/neuron_worker.py                   # Hybrid state binding and reset; cap the KV budget so the shared unified slab stays inside the indirect-DMA addressing limits
```

**Modified — indexes and the top-level README**

```text
docs/model-recipes/index.md   # Add the recipe page
docs/tutorials/index.md       # Add the tutorial page
README.md                     # Contributed Models section -> link to the list on the release branch
```
