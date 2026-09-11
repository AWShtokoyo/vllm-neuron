# Contributed Model: GLM-5.3

vllm-neuron port of `zai-org/GLM-5.3` for AWS Trainium2, on the
**Neuron 2.32 stack (vLLM 0.24 / plugin 0.24.0.1.1.0)**.

> The **Neuron 2.31 / vLLM 0.21** stack is served by the
> [`add-glm-5-2-231`](https://github.com/htokoyo/vllm-neuron/blob/add-glm-5-2-231/GLM-5.2/README.md)
> tag, which is the **GLM-5.2-FP8** port. This branch supersedes it.

## Change History

Newest first. Each entry links to the section with the full detail.

| Date | Change |
|---|---|
| 2026-09-11 | Measured on GLM-5.3 weights, plus one code change. **[The two configuration levers compose](#the-two-configuration-levers-compose-2138-toks)** — **21.38 tok/s, 1.35× the published 15.79**, TPOT −31 %; **[MTP's speed](#speculative-decoding-mtp-speed)** measured (**+18.6 %** on prose, +4.3 % on random — the dataset moves it 4.3×), and its draft MoE **drops an obsolete einsum workaround** for the kernel path (**12.44 tok/s**); **[DSA gained a matched dense counterpart](#dsa-against-a-matched-dense-run)** — 2.1–3.0× slower on decode, so it stays off. |
| 2026-09-01 | Retargeted onto **`zai-org/GLM-5.3`**, which supersedes GLM-5.2-FP8. Verified on the new weights in the same pass: **[Automatic Prefix Caching](#automatic-prefix-caching-apc) enabled and measured** (60.2 % lower median TTFT, 1.25× output throughput, at the predicted hit rate); **[`decode_context_length_buckets`](#setting-decode_context_length_buckets-cuts-decode-time-by-a-third) measured for the first time** (−36.0 % TPOT, TTFT unchanged) and now recommended; **[batching](#batching-scales-to-612-at-the-engines-maximum) measured to the engine's ceiling** (6.12× at concurrency 8); **[DSA](#accuracy-evaluation) gained an identity control** (byte-identical to DSA off below the selection width) and a `gsm8k_cot` figure. |
| 2026-08-31 | Re-hosted onto Neuron 2.32 / vllm-neuron 0.24 (from Neuron 2.31 / 0.21), and extended in the same release: **segmented prefill** made correct and verified on device to `max-model-len` 65536; **speculative decoding (MTP)** at `num_speculative_tokens=1`; the **DSA sparse-attention indexer** as an opt-in path that runs on device at one configuration. Also fixes a prior-KV mask bound in `forward_decode` reachable only under speculation. See [Verification scope](#verification-scope). |
| 2026-07-28 | Initial contribution, on vllm-neuron 0.21 / Neuron 2.31. Adds the `glm_5_2` model package — a 78-layer decoder combining **Multi-head Latent Attention (MLA)** with a **256-expert MoE** (top-8 sigmoid routing plus one always-on shared expert, first three layers dense) — in **FP8 per-channel ROW** at TP=64 / EP=16, with the tiled MLA attention path, the FP8 dequant-and-shard weight loaders and the halved MLA KV page. Component equivalence against `transformers.models.glm_moe_dsa` 13/13 (all R < 1.2). The DSA indexer was **not** included and full attention was used instead. Superseded by this branch and kept at the [`add-glm-5-2-231`](https://github.com/htokoyo/vllm-neuron/blob/add-glm-5-2-231/GLM-5.2/README.md) tag for anyone still on that stack. |

## Introduction

[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) is a large Mixture-of-Experts
(MoE) text-generation model. This is the vllm-neuron implementation of it: a
78-layer decoder that combines **Multi-head Latent Attention (MLA)** with a
**256-expert MoE** feed-forward network (top-8 sigmoid routing plus one always-on
shared expert; the first three layers are dense). It is served text-only under the
architecture `GlmMoeDsaForCausalLM`.

**Compatible model checkpoints:**

| Model | Precision | HuggingFace | |
|-------|-----------|-------------|---|
| GLM-5.3 | FP8 (128×128 block-quantized) | [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) | current default |
| GLM-5.2-FP8 | FP8 (128×128 block-quantized) | [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8) | still loads; produced the figures marked 5.2-FP8 below |

Both work without a code change: the two checkpoints are identical in everything this port
touches — `config.json` bar `transformers_version`, all 118,629 weight keys, and
`quantization_config` — so they share the same compiled graphs. ⚠️ Note the naming is inverted
between the two releases: 5.2's FP8 build carries the `-FP8` suffix while 5.3's does not
(5.3's BF16 build is [`zai-org/GLM-5.3-BF16`](https://huggingface.co/zai-org/GLM-5.3-BF16)).

**This port targets FP8 weights on HBM, because BF16 weights do not fit a
`trn2.48xlarge`.** The model is large enough that the choice is made by HBM
capacity rather than by preference: keeping the weights in FP8 *on device* is what
makes a single-node deployment possible at all. The arithmetic is in
[Why FP8 only](#why-fp8-only-hbm-capacity). The supported configuration is
therefore a single one — `quantization: "fp8_per_channel"`, which keeps FP8 weights on HBM
and dequantizes inside the NKI kernels (see [Quantization](#quantization)).

> This mode was called `fp8_fwd` on the 0.21 branch; it was renamed to match vLLM's own
> [`fp8_per_channel`](https://github.com/vllm-project/vllm/blob/main/vllm/config/quantization.py)
> shorthand, and the old name is no longer accepted.

The BF16 releases — [`zai-org/GLM-5.3-BF16`](https://huggingface.co/zai-org/GLM-5.3-BF16) and
5.2's [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2) — are not supported inputs for
this port.

## Verification scope

Everything below was measured on real hardware — a single **trn2.48xlarge** (16 Trainium2
devices, 64 logical NeuronCores, 1.5 TB HBM), **TP=64 / EP=16**, LNC=2, `fp8_per_channel`,
on the Neuron 2.32 stack.

**Verified on device**

The port was retargeted onto GLM-5.3 partway through verification, so the **Weights** column says
which checkpoint produced each row. Shapes and graphs carry over between them; **accuracy does not**.

| Area | Weights | What was verified |
|---|---|---|
| FP8 serving | both | Compiles and serves; coherent text at every configuration below |
| Long context | 5.2-FP8 | `max-model-len` up to 65536 with an 8192-wide KV segment; 55/55 graphs, needles 3/3 at 29,212 and 3/3 at 58,408 tokens |
| Segmented prefill | 5.2-FP8 | Correct at every supported segment width; prefill cost scales linearly with segment count (see [Measured performance](#measured-performance)) |
| Accuracy | 5.2-FP8 | Component equivalence vs `transformers.models.glm_moe_dsa` 13/13 (all R < 1.2), and GSM8K-CoT at n=100 (see [Accuracy Evaluation](#accuracy-evaluation)) |
| Determinism | 5.2-FP8 | Byte-identical continuation across the offline `LLM` API and the served endpoint on the same NEFFs, 3/3 repeats |
| Speculative decoding (MTP) | 5.2-FP8 | Runs at `num_speculative_tokens=1`, repeatable 8/8, `gsm8k_cot` unchanged against a non-speculative run on the same build (see [Speculative decoding](#accuracy-evaluation)) |
| DSA sparse-attention indexer | 5.2-FP8 needle, **5.3** control | Opt-in (`VLLM_GLM_DSA=1`); serves at `max-model-len` 4096 with a 512-wide segment, needle retrieved at 10 / 50 / 90 % depth in a 3,518-token context; below the selection width, byte-identical to DSA off |
| Automatic Prefix Caching | **5.3** | 60.2 % lower median TTFT and 1.25× output throughput at a 73 % prefix-cache hit rate, against a no-overlap control on the same server |
| Throughput / latency | 5.2-FP8 prefill ladder, **5.3** decode and batching | TTFT and TPOT — single-request for the segmented-prefill runs (`max-num-seqs` 1), and a concurrency sweep to the engine's ceiling of 8 at `max_model_len=4096` |

**Not verified / out of scope**

| Area | Status |
|---|---|
| BF16 serving | Out of scope — the BF16 release does not fit one node, see [Why FP8 only](#why-fp8-only-hbm-capacity) |
| End-to-end logit comparison vs HuggingFace | Blocked by host memory and by the reference implementation, not skipped — see [Accuracy Evaluation](#accuracy-evaluation) |
| MTP byte-identity vs a non-speculative run | **Not expected to hold**, and not a defect; the verify step reads logits at a different shape — see [Accuracy Evaluation](#accuracy-evaluation) |
| DSA above `max_model_len=4096`, or alongside MTP | Not compiled, so not exercised — see [Accuracy Evaluation](#accuracy-evaluation) |
| DSA **selecting** rather than covering | Only the 3,518-token needle test runs above the 2,048-key selection width — see [Accuracy Evaluation](#accuracy-evaluation) |
| DSA speed against a matched dense run | **Measured, and DSA is slower** at every length tried up to 7,000 input tokens — see [DSA against a matched dense run](#dsa-against-a-matched-dense-run). DSA remains off by default |
| NKI MLA attention kernel | Opt-in (`VLLM_GLM_MLA_BLOCK_KERNEL=1`), CPU-simulator validated only, **never run on device** |
| On-device top-k / top-p sampling | Supported by `OnDeviceSamplingConfig` (`max_top_k`), but not exercised — every configuration here compiled the `all_greedy` sampling graph |
| `logprobs` on the served endpoint | **Works with `--no-async-scheduling`, but only without speculation** (HTTP 500 otherwise). No recompile — the flag selects a scheduler, not a graph — and costs about 3 % throughput. 🔴 With MTP enabled it still returns HTTP 500 — see [Accuracy Evaluation](#accuracy-evaluation) |
| Concurrency above `max_num_seqs=8`, or APC / DSA / MTP under concurrency | Not exercised — the batching figures are one prompt length at one `max_model_len` |
| Parallel degrees other than TP=64 / EP=16 | Not exercised |
| Pipeline parallelism, multi-node | Not implemented |

## Model Overview

| | |
|---|---|
| HuggingFace ID | `zai-org/GLM-5.3` |
| Architecture | `GlmMoeDsaForCausalLM` |
| Task | Text generation (`--runner generate`), prefill + decode |
| Layers / hidden size | 78 / 6144 |
| Experts | 256 routed (top-8) + 1 shared |
| Context length | 1,048,576 (`max_position_embeddings`) |
| Vocabulary | 154,880 tokens, untied word embeddings |
| Tensor type | FP8 (e4m3) weights on HBM with per-row scales (`fp8_per_channel`) |

## Model Architecture

Values below come from the port's configuration
([`vllm_neuron/model/glm_moe_dsa/config.py`](../vllm_neuron/model/glm_moe_dsa/config.py)).

**Core configuration**

| Field | Value |
|---|---|
| hidden_size | 6144 |
| num_hidden_layers | 78 (first 3 dense, remaining 75 MoE) |
| num_attention_heads | 64 |
| intermediate_size (dense layers) | 12288 |
| moe_intermediate_size (per expert) | 2048 |
| norm / eps | RMSNorm / 1e-5 |
| RoPE | interleaved, `theta = 8e6` |
| vocab_size | 154,880 (untied embeddings) |

**MLA (Multi-head Latent Attention)**

| Field | Value |
|---|---|
| q_lora_rank | 2048 |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 192 |
| qk_rope_head_dim | 64 |
| v_head_dim | 256 |
| KV cache footprint | 576 dims/token (`kv_lora_rank + qk_rope_head_dim`) |

**MoE routing**

| Field | Value |
|---|---|
| n_routed_experts | 256 |
| num_experts_per_tok | 8 (top-8) |
| scoring_func | sigmoid (`norm_topk_prob=True`) |
| routed_scaling_factor | 2.5 |
| n_shared_experts | 1 (always on) |
| n_group / topk_group | 1 / 1 — plain sigmoid top-k, no group-limited routing |
| first_k_dense_replace | 3 (layers 0–2 use a dense MLP) |

**Port-specific notes**

- **DSA indexer: implemented, opt-in (`VLLM_GLM_DSA=1`), off by default.** `indexer_types` marks each
  layer `full` or `shared`: 21 `full` layers compute a top-`index_topk` selection and the 57
  `shared` layers reuse the nearest preceding one, so every layer ends up sparse while only 21
  carry weights. Two consequences that decide whether you can turn it on:
  - It **requires segmented prefill** (`kv_segment_size_buckets`); the single-shot path has no DSA
    implementation and refuses to run rather than silently serving full attention.
  - It widens **every** layer's KV row by `index_head_dim` (the indexer key rides in the same paged
    buffer and the width cannot vary per layer), costing **22 % of KV cache**, and therefore
    **needs its own compilation** — it cannot be toggled against an existing cache or graph.

  What was verified, and what is not claimed: [Accuracy Evaluation](#accuracy-evaluation).

- **MLA single compressed latent.** MLA stores one compressed latent per token instead of separate
  per-head K and V, so the port emits an `MLAAttentionSpec` whose per-block page size drops the K+V
  factor of two a standard attention layer budgets — roughly halving KV-cache HBM.

- **Layer-78 MTP head.** The checkpoint carries one extra decoder layer intended as a Multi-Token
  Prediction head; the port can serve with it as a self-speculation draft at
  `num_speculative_tokens=1`. Measurements and why byte-identity is not the right test:
  [Accuracy Evaluation](#accuracy-evaluation).

## Why FP8 only: HBM capacity

A `trn2.48xlarge` has 16 Trainium2 chips × 96 GB = **1,536 GB of HBM**, and at LNC2
it presents 4 logical cores per chip = **64 ranks**, so **24.0 GB per rank**. The
default `gpu_memory_utilization=0.92` leaves **22.08 GB per rank** for weights, KV
cache, activations and scratch combined.

Counting this port's own configuration
([`config.py`](../vllm_neuron/model/glm_moe_dsa/config.py)) — 78 layers of MLA plus 75
MoE layers of 256 experts, 3 dense layers, untied embeddings, and the checkpoint's
layer-78 head — gives **≈753 B parameters** to place on device:

| Weights on HBM | Total | Per rank (TP=64) | Against the 22.08 GB/rank budget |
|---|---:|---:|---|
| **FP8** (`fp8_per_channel`) | ≈753 GB | **≈11.8 GB** | fits, with room for KV and activations |
| **BF16** | ≈1,506 GB | **≈23.5 GB** | **does not fit** — over budget on weights alone |

BF16 exceeds the per-rank budget **before** any KV cache, activation or scratch
memory is added, so it cannot be closed by tuning `gpu_memory_utilization`,
shrinking `max_model_len`, or reducing the batch size — it would need at least two
nodes.

What matters is **what is resident on HBM, not what the checkpoint holds.** A mode that
dequantizes to BF16 while loading, or keeps a BF16 copy for its decode path, lands in the BF16
row regardless of the checkpoint it read — which is why `fp8_per_channel` is the only
configuration this port documents.

> This table is a **derivation from the model configuration and the published HBM capacity, not
> a measurement.** The one cross-check available is that ≈753 GB matches the size of the
> `zai-org/GLM-5.3` checkpoint as downloaded.

## Required knobs

Four settings this model requires. Each is load-bearing: without it the run fails at startup
or at compile time, and none of them announce the cause.

| Item | What happens, and what to do |
|---|---|
| **`enable_prefix_caching=False`** | vLLM 0.24 turns APC on by default (`CacheConfig.enable_prefix_caching = True`), and vllm-neuron rejects APC unless segmented prefill is on. A single-shot recipe therefore **fails at startup** with "Automatic Prefix Caching (APC) requires segmented prefill to be enabled". Every configuration below disables it. APC does work on this port — the next row is what it takes. |
| **APC needs three settings together, or it breaks** | If you want prefix caching: (1) `max_num_batched_tokens` **strictly below** `max_model_len` — equal values give single-shot prefill and APC is rejected at startup; (2) `num_batched_tokens_buckets` set **explicitly** — leave it out and the plugin auto-sets it to the segment size alone, after which the first cache hit changes the prefill length, falls outside the compiled buckets and **takes the engine down** with `Detected recompile`; (3) prompts whose shared prefix is a **whole multiple of the segment size** — a prefix that does not fill a segment leaves nothing to skip. |
| **`VLLM_NEURON_BARRIER_TIMEOUT` (default 3600 s) is too short** | While rank 0 compiles, the other 63 ranks wait in `tp_barrier()`. If the barrier fires first they exit and the run dies mid-build — **after every graph has compiled and been cached**, so the work is not lost but the run is. Raise it well past rank 0's whole cold compile — which ran **4.63 h** in one of the configurations below; see [Step 1](#step-1-environment-setup). |
| **A 512-wide prefill segment when `VLLM_GLM_DSA=1`** | At a 1024-wide segment the compiler aborts with `[INTERNAL_ERROR] [NCC_ILSA901] LegalizeSundaAccess assertion error: unexpected AP of matmult dst`; the cause is under investigation. **Use a 512-wide segment** — every DSA configuration in this README was verified with it. |

## Feature status

| Category | Feature | Status | Notes |
|---|---|---|---|
| **Inputs** | Text | ✅ | Text-only serving |
| | Multimodal (image / video) | ❌ | |
| **Quantization** | FP8 per-row weights on HBM (`fp8_per_channel`) | ✅ | `ROW` mode, dequant in-kernel; the supported configuration |
| | BF16 weights on HBM | ❌ | Does not fit one node — [why](#why-fp8-only-hbm-capacity) |
| **Parallelism** | Tensor parallelism (TP) | ✅ | |
| | Expert parallelism (EP) | ✅ | `ep_degree` in `neuron_config` |
| | Pipeline / context parallelism | ❌ | |
| **Attention** | Multi-head Latent Attention (MLA) | ✅ | `MLAAttentionSpec`, halved KV page |
| | DSA sparse-attention indexer | ⚠️ experimental | `VLLM_GLM_DSA=1`; validated on CPU against `transformers.models.glm_moe_dsa` (selection set matches the reference exactly) and **runs on device at one verified configuration** (`max_model_len=4096`, 512-wide prefill segment, needle retrieved at 10/50/90% depth in a 3,518-token context; `gsm8k_cot` 0.91 at n=100). ⚠️ The `gsm8k_cot` figure does **not** test sparsity — its prompts sit below the 2,048-key selection width, where the selection is all-inclusive; only the needle test exercises real sparsity ([detail](#accuracy-evaluation)). Off by default — still under verification towards a supported implementation |
| | NKI MLA attention kernel | ⚠️ not on device | `VLLM_GLM_MLA_BLOCK_KERNEL=1`; **CPU-simulator validated only, never run on device** — off by default. Running it on device is planned |
| **Speculative decoding** | MTP self-speculation | ✅ opt-in | `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`. Runs on device, repeatable 8/8, and `gsm8k_cot` is unchanged against a non-speculative run on the same build (see [Accuracy Evaluation](#accuracy-evaluation)). Off unless explicitly enabled |
| **Context** | `max_model_len` ≤ 16,384 (single-shot prefill) | ✅ | vllm-neuron 0.24 caps single-shot at `MAX_MODEL_LEN_SINGLE_SHOT` = 16 KiB |
| | `max_model_len` > 16,384 | ✅ | Requires chunked prefill (✅ below). Verified on device at **65,536** with `kv_segment_size=8192` (the largest supported segment size): a 58,408-token prompt spanning 8 segments retrieved facts planted in segments 0, 3 and 6. Also verified at 16,384 with `kv_segment_size=4096` |
| **Performance** | Segmented (chunked) prefill | ✅ | Verified on device at `max_model_len=16384` / `kv_segment_size=4096`: facts planted in segments 0, 1 and 2 of an 8,800-token prompt and in 0, 1 and 3 of a 15,100-token one were all retrieved, so the prior-KV path resolves across segment boundaries on hardware. |
| | Automatic Prefix Caching (APC) | ✅ opt-in | Measured on device: **60.2 % lower median TTFT** and **1.25× output throughput** on a shared-prefix workload against a no-overlap control. Needs three settings or it fails — see [Required knobs](#required-knobs) and [Accuracy Evaluation](#accuracy-evaluation). Off in every other configuration here |
| | On-device sampling — greedy | ✅ | `on_device_sampling_config: {"all_greedy": true}`. Every configuration verified here used it; `SamplingParams(temperature=0)` alone does not select it |
| **Compilation** | torch.compile (XLA backend) | ✅ | |

Legend: **✅** verified on device and usable — `opt-in` means it works but is off by default ·
**⚠️** integrated but not yet dependable, with the qualifier saying why — `experimental` ran on
device in one narrow configuration, `not on device` was verified only off device · **❌** not
implemented. See [Verification scope](#verification-scope) for exactly what was measured.

## Setup

### Native integration (nothing to apply)

The model is **natively integrated** into this repository — check out this branch
and it works; there is no patch to apply.

- **Model package:** [`vllm_neuron/model/glm_moe_dsa/`](../vllm_neuron/model/glm_moe_dsa/)
  (per-file breakdown: [module structure](../vllm_neuron/model/glm_moe_dsa/README.md)).
- **Registration and framework touch-points** are committed directly on this
  branch — see
  [Paths this port touches](#paths-this-port-touches-across-the-repository).

**Prerequisites:**

- `trn2` hardware with Neuron SDK `2.32`. See the
  [setup guide](../docs/getting-started/setup-guide.md).
- vLLM Neuron plugin `0.24.0.1.1.0` — this branch's base.
- Python 3.11+ (the repository's `requires-python`). Verified on 3.13.7.
- **An FP8 checkpoint and an FP8 `quantization` mode are required**, not optional:
  see [Why FP8 only](#why-fp8-only-hbm-capacity). Provision the checkpoint
  filesystem for ≈753 GB.
- GLM-5.3 is large. The shipped example uses `tensor_parallel_size=64`, which at
  LNC2 is the 64 logical cores of one `trn2.48xlarge`. Size the checkpoint
  filesystem accordingly.

### Step 1: Environment setup

Verify the Neuron devices are visible:

```bash
neuron-ls   # 16 Trainium2 chips per trn2.48xlarge; 64 logical cores at LNC2
```

Export these before any compile / inference run:

```bash
# Extended timeouts for large MoE compilation
export NEURON_LIBTORCH_COMPILATION_TIMEOUT=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Scratchpad settings used during bring-up
export NEURON_SCRATCHPAD_PAGE_SIZE=512
export NEURON_SKIP_EFA_AFFINITY=1

# NEURON_CC_FLAGS is not set: this backend builds the compiler arguments itself and never reads it.

# Serial TRACE only — parallel trace host-OOMs. 🔴 Exactly this spelling: others are silently ignored.
export NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS=1

# COMPILE parallelism is a separate knob (no model resident); 8 is nkilib's own default.
export NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS=8

# 🔴 Far above rank 0's whole cold compile, or the other 63 ranks leave tp_barrier(). 14400 s was
# also too short; see "Required knobs".
export VLLM_NEURON_BARRIER_TIMEOUT=86400

# Where the compiled NEFFs land: a large LOCAL volume that survives a reboot, not NFS. Unset, the
# plugin falls back to a path under /tmp.
export VLLM_CACHE_ROOT=/path/to/cache/vllm

# 🔴 Only if a compile cache is reused across machines: the emitted HLO references NKI binaries in
# /var/tmp by absolute path, so a restored cache fails with [NCC_EVRF059].
export NKI_COMPILE_CACHE_URL=/path/to/cache/nki
```

> **The first run compiles every graph, and it is not quick.** At
> `NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS=8`, the minimal `max_model_len=128`
> configuration compiled its 128 graphs in **59.9 min**, and `max_model_len=16384` took
> **4.63 h**. It is one-time per cache — which is why `VLLM_CACHE_ROOT` above should point
> somewhere that survives a reboot.
>
> ⚠️ A cold compile can also lose a graph to `[NCC_ILKK005]` on a shared NKI transpose kernel.
> It is a race, not a shape problem — but the framework fails the whole load rather than
> re-checking the cache. The graphs that did compile are cached, so a relaunch only retries the
> rest; **the race can recur on the same graphs**, so budget for more than one attempt. Seen on
> 2 of 128 graphs, twice in a row here.

### Step 2: Download the model

```bash
huggingface-cli download zai-org/GLM-5.3 --local-dir /path/to/GLM-5.3
```

> **Tip:** Download to a filesystem provisioned for the full ≈753 GB, not a home
> directory — and to a shared one if you intend to serve across nodes.

## Serving

### Offline inference (`llm.generate()`)

[`examples/vllm_neuron/models/glm_moe_dsa/run.py`](../examples/vllm_neuron/models/glm_moe_dsa/run.py)
carries the verified recipe and applies most of the `export`s above via `os.environ.setdefault`,
so the only thing it needs is the checkpoint:
`python run.py --model-checkpoint /path/to/GLM-5.3`. An explicit `export` still wins, since
`setdefault` only fills unset variables.

### Online serving (OpenAI-compatible)

```bash
vllm serve /path/to/GLM-5.3 \
    --served-model-name GLM-5.3 \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel --no-enable-prefix-caching \
    --additional-config '{"neuron_config": {"ep_degree": 16, "quantization": "fp8_per_channel", "num_batched_tokens_buckets": [4096], "num_seqs_buckets": [8], "on_device_sampling_config": {"all_greedy": true}}}'
```

The first launch compiles one NEFF per bucket, which takes a long time at this
model size. Wait for `Application startup complete.`, then:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "GLM-5.3", "prompt": "The capital of France is",
         "max_tokens": 16, "temperature": 0}'
```

**Configuration notes:**

- `ep_degree` — expert-parallel degree. With `tensor_parallel_size=64` and
  `ep_degree=16`, each expert-parallel group holds `256 / 16 = 16` experts, and
  each expert's intermediate dimension is sharded across `tp_sub = 64 / 16 = 4`
  ranks. Both `world_size % ep_degree == 0` and `num_experts % ep_degree == 0`
  must hold, and `ep_degree` requires `--enable-expert-parallel`.
- `num_batched_tokens_buckets` / `num_seqs_buckets` — prefill-token and
  batch-size buckets. Every bucket adds compile time; start with one of each.
- `decode_context_length_buckets` — worth setting. Left unset, decode falls back to
  `max_model_len` and reads the whole bucket's latent every step; setting it to the context you
  actually serve cut TPOT by **36 %** here, with TTFT unchanged
  ([measurement](#setting-decode_context_length_buckets-cuts-decode-time-by-a-third)).
- `quantization` — set it to `fp8_per_channel`. This is **not optional**: without it the
  weights are held in BF16, which does not fit one node
  ([why](#why-fp8-only-hbm-capacity)).

### Quantization

Set `quantization` to **`fp8_per_channel`** in `neuron_config`. It is the one configuration
that holds FP8 weights on HBM end to end, and therefore the one that fits a single
node ([why](#why-fp8-only-hbm-capacity)):

| | `fp8_per_channel` |
|---|---|
| Checkpoint | `zai-org/GLM-5.3` (128×128 block-quantized) |
| Weights on HBM | FP8 `e4m3` + a per-output-channel (per-row) dequant scale vector |
| Where dequant happens | Inside the NKI kernels (`QuantizationType.ROW`) |
| Activation scales | Not required |
| MLP / MoE weights | FP8 on HBM |
| MLA attention weights | BF16 — MLA has no kernel-side FP8 support, so these are dequantized at load |
| Cost of the 128×128 → per-row scale collapse | **Unmeasured** — see [Verification scope](#verification-scope) |

At load time the checkpoint's block-FP8 weights are dequantized, an amax is taken per output
channel, and they are written back as `e4m3` alongside their per-row scale. Prefill dequantizes
to BF16 as a **transient** inside the forward pass, so no second copy stays resident. MLA in
BF16 is the one exception, and the MLP/MoE weights (75 layers × 256 experts) dominate, so the
capacity conclusion above holds.

```bash
vllm serve /path/to/GLM-5.3 \
    --served-model-name GLM-5.3 \
    --max-model-len 4096 --max-num-seqs 8 \
    --tensor-parallel-size 64 --enable-expert-parallel --no-enable-prefix-caching \
    --additional-config '{"neuron_config": {"ep_degree": 16, "quantization": "fp8_per_channel"}}'
```

### A note on naming

`neuron_config.quantization` and the checkpoint's own `quantization_config` describe
**different layers** in the same FP8 vocabulary. The checkpoint's field (`quant_method: "fp8"`,
`weight_block_size: [128, 128]`) says **how the checkpoint is stored**; `neuron_config.quantization`
says **what this port puts on HBM**. `config.py` never reads the checkpoint's field — the block
size of 128 lives in
[`weight_loaders_fp8.py`](../vllm_neuron/model/glm_moe_dsa/weight_loaders_fp8.py). And whatever
`activation_scheme` says, this port quantizes weights only; activations stay in BF16.

## Accuracy Evaluation

**Configurations exercised on device.** Four, all TP=64 / `ep_degree`=16 / `fp8_per_channel` /
APC off / on-device `all_greedy`:

| | A | B | C | D |
|---|---|---|---|---|
| `max_model_len` | 128 | 4,096 | 16,384 | **65,536** |
| `kv_segment_size` | — | — | 4,096 | **8,192** |
| `num_batched_tokens_buckets` | `[128]` | `[4096]` | `[4096]` | `[8192]` |
| `num_seqs_buckets` | `[1]` | `[8]` | `[1]` | `[1]` |
| KV cache | 79,136 tokens | 79,136 tokens | 79,136 tokens | 79,136 tokens |
| max concurrency | — | 19.32× @ 4,096 tok | 4.83× @ 16,384 tok | 1.21× @ 65,536 tok |

C and D are segmented-prefill, A and B single-shot. The KV cache is the same in all four
because it is sized from free HBM after weights, not from `max_model_len`; what changes is how
much of it one request can occupy. The MTP and DSA runs in **3** and **4** below each used a
further configuration of their own (`max_model_len` 2,048 and 4,096 respectively), stated there.

**1. Accuracy — component equivalence.** Each component of this port is compared
against the corresponding `transformers.models.glm_moe_dsa` class using the
three-tensor R-ratio method (FP32 reference, BF16 baseline, port under test), on
CPU at TP=1 with shared random weights, so it needs no checkpoint and is
independent of model size. **13/13 pass**, τ_R = 1.2:

| Component | R | | Component | R |
|---|---:|---|---|---:|
| RMSNorm | 0.8331 | | MLA compressed latent | 0.9060 |
| interleaved RoPE, values | 1.0000 | | **MLA scores via weight absorption** | **0.9599** |
| interleaved RoPE, q·kᵀ | 1.0000 | | dense MLP | 1.0000 |
| MoE router logits | 1.0000 | | shared-expert MLP | 1.0000 |
| MoE router top-k weights | 1.0000 | | tiled MLA vs single-shot prefill | 1.0315 |
| tiled MLA vs decode | 0.9520 | | segmented: 2 chunks == 1 shot | 0.9986 |
| segmented: first chunk E2E | 1.0196 | | | |

`mla_scores_absorption` is the load-bearing one: this port absorbs `kv_b_proj` into `q_nope`
and scores in latent space while HuggingFace expands the latent into per-head K, so agreement
rules out a wrong absorption axis or a transposed `W_uk` — errors a shape check cannot see.
R < 1 (0.83, 0.91, 0.96) means the port sits *closer* to the FP32 reference than HuggingFace's
own BF16 does; for RMSNorm the cause is explicit, HF rounds the normalised hidden to bf16
*before* the weight multiply while this port keeps the multiply in FP32.

> ⚠️ Run against the **GLM-5.2-FP8** config, before the retarget. It uses random weights rather
> than the checkpoint's, so what carries over to 5.3 is exactly what the two configs share — and
> `config.json` is identical bar `transformers_version`.

**2. Accuracy — downstream task.** GSM8K-CoT through the served endpoint, 8-shot,
`do_sample=False`, `num_concurrent=8`:

| Filter | exact_match | Stderr |
|---|---:|---:|
| flexible-extract | **90.0%** | ±3.02 |
| strict-match | **88.0%** | ±3.27 |

> ⚠️ **n = 100**, via `--limit`, on **GLM-5.2-FP8 weights** — measured before the retarget and not
> re-measured on 5.3, and accuracy is the one thing that depends on the weights rather than on the
> shapes the two checkpoints share. This run is configuration **B**
> (`max_model_len=4096`, `max_num_seqs=8`); the pair in **3** below is a **separate, later run**
> at `max_model_len=2048` / `max_num_seqs=1`, which is why its numbers differ from these by about
> one standard error.

```bash
lm_eval --model local-completions \
  --model_args "model=GLM-5.3,base_url=http://localhost:8000/v1/completions,\
num_concurrent=8,max_retries=3,timeout=1800,tokenized_requests=False,tokenizer=/path/to/GLM-5.3" \
  --tasks gsm8k_cot --batch_size 1 --limit 100
```

> ⚠️ `timeout=1800` above is load-bearing. lm-eval's `local-completions` backend defaults its
> client timeout to **300 s** (`api_models.py`), and an 8-shot CoT request against this model exceeds
> that, which surfaces as `TimeoutError` and `ServerDisconnectedError` after retries rather than as a
> server-side error. (`--batch_size` is ignored whenever `num_concurrent > 1`: the generative path
> passes `n=0` to the batcher, so it does not compound the concurrency.)

**3. Speculative decoding (MTP).** Measured against a non-speculative run on the same build
and the same instance:

| | MTP, `num_speculative_tokens=1` | non-speculative |
|---|---|---|
| `gsm8k_cot` flexible-extract | **0.89** ± 0.031 | 0.88 ± 0.033 |
| `gsm8k_cot` strict-match | **0.90** ± 0.030 | 0.89 ± 0.031 |
| same prompt → same output | **8 of 8** | 8 of 8 |
| byte-identical to the other column | 4 of 8 | — |

> ⚠️ Measured on **GLM-5.2-FP8** weights, before the retarget, and not re-measured on 5.3.

**Byte-identity is not the right test, and its absence is not a defect.** The verify step reads the
target's logits at a different shape from a non-speculative step, so the two are not the same
floating-point expression. Standing in for a direct check, and off device: a 5-layer CPU comparison
puts the logit difference between the shapes at **3.1e-02** against a top-1/top-2 gap of
**1.25e+00** — a 40× margin, so the shape rarely changes which token wins.

> ⚠️ So "every divergence is a near-tie" is *inferred*, not checked position by position, and it
> **cannot be checked on this build**: the check needs `logprobs` from the MTP arm, and `logprobs`
> fails under speculation — a non-speculative server with **`--no-async-scheduling`** returns them
> (no recompile, about **3 %** throughput), but the MTP server, already running with that flag,
> still returns **HTTP 500**.

**4. DSA sparse-attention indexer.** Opt-in (`VLLM_GLM_DSA=1`), and what was verified is narrow:

| | Weights | |
|---|---|---|
| Configuration | — | `max_model_len=4096`, 512-wide KV segment, one decode context bucket at 2048, `max_num_seqs=1` |
| Graphs compiled | 5.2-FP8 | 896, no compile errors |
| KV cache | 5.2-FP8 | 64,736 tokens, against 79,136 with DSA off — the 704/576 row-width ratio |
| Retrieval, 3,518-token context (the selection keeps 2,048 of 3,518 keys) | 5.2-FP8 | needle found at **10 %, 50 % and 90 % depth** |
| Decode latency at that context | 5.2-FP8 | **155.7 ms/token**, from the slope of two output lengths (8 and 72, `ignore_eos`) so prefill and client overhead cancel. **No dense counterpart at this configuration** — for DSA against a matched dense run see [Measured performance](#dsa-against-a-matched-dense-run) |
| Identity control below the selection width | **5.3** | 128 greedy tokens from a 1,012-token prompt are **byte-identical** to the same request with DSA off |
| Accuracy with the indexer active | **5.3** | `gsm8k_cot` at n=100: **0.91** ± 0.029 strict-match, **0.90** ± 0.030 flexible-extract |

> ⚠️ The rows marked 5.2-FP8 were measured before the retarget and not re-measured on 5.3. The
> graph count and the KV figure are shape facts, so they carry over; the needle result and the
> latency depend on the weights.

> ⚠️ The 90 % case is the one that means something. At 50 % the needle sits inside the first 2,048
> keys, so a selection that took the leading window — or one that was silently inert — would retrieve
> it too. At 90 % it is outside any leading window, so retrieval requires the scores to have ranked
> that region.

> ⚠️ **The last two rows do not test sparsity.** The code takes `min(index_topk, S_kv)`, so below
> 2,048 keys the selection is all-inclusive — the sparse path still runs, but it never has to choose.
> Both rows sit there (the GSM8K prompts are ~1,000 tokens), so what they check is the index
> arithmetic: drop one legitimate key and the softmax denominator moves, breaking byte-equality.
> Only the needle test above exercises real sparsity.

> 🔴 **Not claimed.** No speed figure, because there is no dense counterpart built the same way at
> the same configuration. No accuracy benchmark on a workload whose context exceeds the 2,048-key
> selection width — that needs a DSA build above `max_model_len=4096`, and none has been compiled.
> DSA has not been run alongside speculative decoding. And **DSA does not
> reduce work in this form**: the attention pass still visits every key and masks the unselected
> ones, so what it buys is fidelity to the trained model, not speed.

## Measured performance

> ⚠️ Measured on the configurations under [Accuracy Evaluation](#accuracy-evaluation), with greedy
> on-device sampling. Single-request except where a concurrency is stated. Not a deployment
> characterisation: every figure comes from one instance and one prompt distribution.

### Benchmarking this port yourself

The tables below were produced by a purpose-built harness, not by `vllm bench serve`, so this
command does not reproduce them — it is the model-agnostic starting point if you want your own
numbers. Run it against a server started as in
[Online serving](#online-serving-openai-compatible).

```bash
vllm bench serve \
    --base-url http://localhost:8000 \
    --model GLM-5.3 \
    --tokenizer /path/to/GLM-5.3 \
    --dataset-name random \
    --random-input-len 32 --random-output-len 96 \
    --random-range-ratio 0 \
    --num-prompts 4 --max-concurrency 1 \
    --ignore-eos \
    --save-result --result-filename glm53_bench_decode.json
```

Four flags are load-bearing. `--random-range-ratio 0` pins the input length — without it the
client samples a range and runs are not comparable. `--ignore-eos` fixes the output length.
`--max-concurrency` must equal a compiled `num_seqs_buckets` value, and the two lengths must fit
`max_model_len`. A shape outside the compiled buckets is not merely slow: `torch.compile` runs with
the `fail_on_recompile` stance here, so the recompile raises and **takes the engine down**
(`RuntimeError: Detected recompile ...`, observed on device), leaving the server returning HTTP 500.
`--model` must be the `--served-model-name`, which is not resolvable locally, so
`--tokenizer <checkpoint path>` is required too or every request fails with `is not a local folder`.

### Prefill scales linearly with segment count

Segmented prefill, 8 output tokens, sole client.

> ⚠️ Measured on **GLM-5.2-FP8** weights, before the retarget. Prefill cost is set by the bucket
> shapes, which the two checkpoints share, so these figures are expected to hold on 5.3 — but they
> were not re-measured there.

| Prompt tokens | Segments | `mml=16384` / `seg=4096` | `mml=65536` / `seg=8192` |
|---:|---:|---:|---:|
| ~1,000 | 1 | 3.41 s | — |
| ~3,400 | 1 | 3.60 s | — |
| 4,762 | 1 | — | 10.00 s |
| 6,814 | 2 | 6.30 s | — |
| 10,702 | 3 | 9.01 s | — |
| 12,754 | 2 | — | 18.47 s |
| 14,590 | 4 | 11.72 s | — |
| 20,710 | 3 | — | 26.93 s |
| 28,702 | 4 | — | 35.39 s |
| 36,658 | 5 | — | 43.86 s |
| 44,614 | 6 | — | 52.32 s |
| 52,606 | 7 | — | 60.78 s |

| Configuration | Least-squares fit over segment count | R² | Marginal cost per segment |
|---|---|---:|---|
| `mml=16384` / `seg=4096` | total ≈ 0.89 s + **2.71 s** × segments | 0.999999 | 2.70 / 2.71 / 2.71 s |
| `mml=65536` / `seg=8192` | total ≈ 1.54 s + **8.46 s** × segments | 1.0000 | 8.47 / 8.46 / 8.47 / 8.46 s |

The marginal cost does not grow with segment position, so the prior-KV sweep is bounded by
a fixed bucket rather than by the prior that actually exists — the bound has to be static
because a data-dependent one cannot be traced on device.

> ⚠️ **Practical consequence: a large `max_model_len` costs prefill time even for short
> prompts.** A 4,762-token prompt takes 10.00 s at `max_model_len=65536`, against 3.41 s for
> a ~1,000-token prompt at 16,384 — one segment of real work either way, against a
> four-times larger prior bucket. Size `max_model_len` to the context you need.

### At a fixed prefill bucket, latency does not vary with prompt length

`max_model_len=4096`, `max_num_seqs=8`, 32 output tokens:

> ⚠️ Measured on **GLM-5.2-FP8** weights, before the retarget. The 373.37 ms baseline in the next
> section is the same shape re-measured on 5.3, and it agrees to 0.2 %.

| Prompt tokens | 32 | 249 | 993 | 1,985 | 2,907 | 3,682 |
|---|---:|---:|---:|---:|---:|---:|
| TTFT | 2.16 s | 2.16 s | 2.16 s | 2.16 s | 2.16 s | 2.16 s |
| TPOT | 374.0 ms | 374.0 ms | 374.1 ms | — | 374.1 ms | 374.0 ms |

TTFT spread across the six prompts is 0.005 s (stdev 0.002 s); TPOT varies by 0.1 ms. The
1,985-token case reports no TPOT because the model emitted EOS after one token.

Both are bucketing, not scaling. Prefill pads to the single 4,096-token bucket, so it does
identical work regardless of prompt length. Decode falls back to `max_model_len` because
`decode_context_length_buckets` is unset, so it gathers the full bucket's latent every step
no matter how short the real context is.

### Setting `decode_context_length_buckets` cuts decode time by a third

Same server configuration, same benchmark (993-token prompts, 32 output tokens,
`--max-concurrency 1`), on **GLM-5.3** weights, with `decode_context_length_buckets` as the only
variable:

| | unset (falls back to `max_model_len` = 4,096) | `[2048]` |
|---|---:|---:|
| Median TPOT | 373.37 ms | **238.94 ms** |
| Median TTFT | 2,171.38 ms | 2,170.95 ms |

**−36.0 % on decode (1.56×), with TTFT unchanged to within 0.5 ms** — the knob narrows the decode
bucket only, and the measurement agrees. No kernel work behind it, so it applies to any deployment
whose real contexts are shorter than `max_model_len`.

> ⚠️ One prompt length, one bucket value, single concurrency. Adding the bucket costs a partial
> recompile: 64 decode graphs here, against the 128 the full configuration needs, because the prefill
> graphs are reused.

### Batching scales to 6.12× at the engine's maximum

Same server, same workload (993-token prompts, 128 output tokens, `--ignore-eos`), on **GLM-5.3**
weights, varying only `--max-concurrency`. `max_model_len=4096`, `max_num_seqs=8`, so 8 is the
ceiling the engine allows:

| Concurrency | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Output throughput | 2.58 tok/s | 4.91 tok/s | 9.08 tok/s | **15.79 tok/s** |
| Speedup over concurrency 1 | 1.00× | 1.90× | 3.52× | **6.12×** |
| Median TPOT | 373.06 ms | 381.69 ms | 398.65 ms | 432.48 ms |

At the engine's maximum batch, throughput is **6.12× the single-request figure — 77 % of the ideal
8×** — and per-token latency degrades only **15.9 %**.

TTFT is the cost side: **5,788 ms** at concurrency 4 against **2,172 ms** at 1. Prefill itself did
not slow down — median inter-token latency is 373.0 ms at both — the later requests are queuing for
a prefill slot.

> ⚠️ **This retires a prediction this README used to make.** MLA's compressed latent is not
> TP-shardable, so every rank reads the whole latent for every sequence every step, and batching was
> expected to help less than for a GQA model. It does not — so the latent read is not what binds
> decode at these batch sizes. What does has not been measured.

> ⚠️ Measured at one prompt length and one `max_model_len`, with `decode_context_length_buckets`
> unset. Nothing above `max_num_seqs=8` was tried.

### The two configuration levers compose: 21.38 tok/s

The table above varies concurrency with `decode_context_length_buckets` unset; the
[bucket section](#setting-decode_context_length_buckets-cuts-decode-time-by-a-third) varies the
bucket at concurrency 1. Setting **both** — `max_model_len=4096`,
`num_batched_tokens_buckets: [4096]`, `num_seqs_buckets: [8]`,
`decode_context_length_buckets: [2048]` — and re-running the same workload (993-token prompts, 128
output tokens, `--ignore-eos`) on the same weights gives:

| Concurrency | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Output throughput | 3.94 tok/s | 7.36 tok/s | 13.05 tok/s | **21.38 tok/s** |
| Speedup over concurrency 1 | 1.00× | 1.87× | 3.31× | **5.43×** |
| Median TPOT | 238.48 ms | 247.10 ms | 264.19 ms | **298.32 ms** |

**21.38 tok/s is 1.35× the 15.79 tok/s above, and no kernel was changed** — two configuration
values, each already documented separately, that had not been measured together. Median TPOT
improves by 31 % at concurrency 8 (432.48 → 298.32 ms) and by 36 % at concurrency 1.

⚠️ **The batch speedup falls to 5.43× from 6.12×, and that is not a regression.** Concurrency 1 got
faster (2.58 → 3.94 tok/s), so the ratio is measured against a larger denominator. Efficiency
against the ideal 8× is 68 % here against 77 % above.

> ✅ Validity control: median TPOT at concurrency 1 reproduced an earlier run of this same
> configuration to within **0.2 %** (238.48 ms against 238.94 ms) on a different instance of the same
> type. Without that the rest of the table would not be quotable.

> ⚠️ Same limits as the table above: one prompt length, one `max_model_len`, one bucket value,
> greedy on-device sampling, nothing above `max_num_seqs=8`. `max_num_seqs=16` was tried and does
> not fit this configuration.

### Automatic Prefix Caching (APC)

Two runs of the same benchmark on the same server, with
`--prefix-repetition-num-prefixes` as the only variable — 5 distinct prefixes shared across 50
prompts (heavy reuse) against 50 distinct prefixes (nothing to reuse). On **GLM-5.3** weights.
`max_model_len=2048`, `kv_segment_size=512`, prefix 1,536 tokens (three whole segments), suffix 384,
output 128, `--max-concurrency 1`, `--ignore-eos`:

| | Median TTFT | Output throughput | Prefix cache hit rate |
|---|---:|---:|---:|
| 5 shared prefixes | **1,762 ms** | **13.66 tok/s** | 73–74 % |
| 50 distinct prefixes (control) | 4,429 ms | 10.89 tok/s | ~0 % |
| | **−60.2 %** | **1.25×** | |

The hit rate matches the 72.0 % the workload's token counts predict, and the TTFT saving is 80 % of
the 75 % ceiling that skipping three of four segments allows.

> ⚠️ **The control is what makes this a measurement.** A first request is often faster than its
> successors here for reasons unrelated to caching — probing by hand with three repeats gave 0.905 s
> on a shared-prefix request *and* 1.789 s on an unrelated one, both against a 3.88 s steady state,
> so a single fast request proves nothing. Only the median over 50 prompts, against a workload with
> no overlap, separates prefix reuse from that effect.

> ⚠️ **What is not claimed.** One shape at one segment size, single concurrency. Nothing is measured
> for concurrent requests, for larger `max_model_len`, or with DSA or MTP enabled alongside it.

### Speculative decoding (MTP) speed

The accuracy of MTP is in [Accuracy Evaluation](#accuracy-evaluation); this is its speed, measured
separately at `max_model_len=2048`, segmented prefill 512, `max_num_seqs=1`,
`num_speculative_tokens=1`, both arms warm, 128 output tokens with `--ignore-eos`:

| Dataset | Mean acceptance length | non-speculative | MTP | MTP gain | Median TPOT |
|---|---:|---:|---:|---:|---|
| Random token sequences | 1.18 | 11.50 tok/s | 11.99 tok/s | **+4.3 %** | 71.07 → 67.27 ms |
| English prose | 1.73 | 10.38 tok/s | 12.31 tok/s | **+18.6 %** | 71.84 → **56.99 ms** |

**The dataset moves the answer by 4.3×.** Random tokens understate speculation structurally —
acceptance length collapses to 1.18 of a possible 2, against 1.73 on prose — so **+18.6 % is the
favourable end and +4.3 % the unfavourable one**, and production traffic should fall between them.

> ⚠️ Read within a row, not across rows: the prose loader packs whole lines, so its prompts come out
> at 845 tokens against 993. One concurrency, two requests per arm.

**The draft's MoE now takes the same kernel path as the target.** Earlier releases routed the draft's
layer-78 decode MoE through a kernel-free einsum path, to avoid an indirect-DMA over-read at the γ=1
verify shape; `_forward_decode` now pads the token count up to a multiple of 16, which covers that
shape, so the workaround was obsolete. Removing it, with the non-speculative control unchanged at
11.5 tok/s:

| MTP draft decode MoE | tok/s | Median TPOT |
|---|---:|---:|
| einsum path (previous) | 12.22 | 67.68 ms |
| `moe_tkg` kernel path (now) | **12.44** | **65.88 ms** |

> ⚠️ **+1.8 %** is small because decode here is not bandwidth-bound — the draft's MoE reads drop
> roughly 32× but the bytes were not what bound it. It is also **not numerically neutral**: mean
> acceptance length moved 1.40 → 1.33, so the two paths round differently enough to change which
> draft tokens are accepted. Two requests, one configuration.

### DSA against a matched dense run

Both arms were built at the same `max_model_len=8192` / 512-wide segmented prefill, with
`decode_context_length_buckets` of 2048 and 4096, so one compile per arm covers three lengths
(7,000 falls into the implicit `max_model_len` bucket). `max_num_seqs=1`, 128 output tokens,
`--ignore-eos`, both arms warm.

| Input tokens | Decode bucket | Dense TPOT | DSA TPOT | Dense TTFT | DSA TTFT |
|---:|---:|---:|---:|---:|---:|
| 1,500 | 2048 | **57.71 ms** | 121.81 ms | 3,853 ms | 33,358 ms |
| 3,500 | 4096 | **62.29 ms** | 159.55 ms | 9,005 ms | 77,716 ms |
| 7,000 | 8192 | **80.44 ms** | 243.87 ms | 18,113 ms | 155,338 ms |

**DSA costs 2.1× to 3.0× on decode here, and the gap widens with context** (2.11× → 2.56× → 3.03×)
instead of narrowing, so the overhead is not the indexer arithmetic alone; TTFT is a near-constant
8.6×. ⚠️ These lengths sit well below the ones sparse selection targets, so the claim is narrow —
*at 8,192 and below, on this port, enabling DSA costs 2–3× decode latency*, which is the reason it
stays off by default. One concurrency, two requests per point; `tok/s` is omitted because the
512-wide segment inflates TTFT for both arms for reasons unrelated to DSA.

## Contents of this bundle

The model is integrated in-tree (committed directly on this branch), so this
top-level `GLM-5.3/` bundle is supplementary: it carries the source-of-truth
README only. There is no `src/` copy here — that would duplicate the in-tree
package.

```text
GLM-5.3/
└── README.md            # This file — the source of truth for the port
```

The bundle ships no `test/` directory. The verification that was carried out is
described in [Accuracy Evaluation](#accuracy-evaluation), together with the procedure to reproduce
it and an explicit statement of what remains unverified.

### Paths this port touches across the repository

Everything below is stated against the branch this port is based on,
`release-0.24.0.1.1.0` (vllm-neuron 0.24 / Neuron 2.32).

**New — model package** (`vllm_neuron/model/glm_moe_dsa/`)

```text
vllm_neuron/model/glm_moe_dsa/
├── __init__.py                # Package exports (GlmMoeDsaConfig, GlmMoeDsaForCausalLM)
├── README.md                  # Module structure (points at this README)
├── config.py                  # GlmMoeDsaConfig: HF → Neuron config translation (MLA ranks,
│                              #   MoE routing, first_k_dense_replace, interleaved RoPE)
├── factory.py                 # GlmMoeDsaForCausalLM factory: validates the config and selects
│                              #   the implementation from `quantization`
├── mla_block.py               # NKI dispatch for the MLA inner attention block, opt-in via
│                              #   VLLM_GLM_MLA_BLOCK_KERNEL=1 — default OFF, simulator-only
├── mla_block_kernel.py        # The NKI kernel itself: one online-softmax step over a key
│                              #   segment (score + max + rescale + accumulate)
├── dsa_indexer.py             # The DSA indexer: per-query top-k key selection (score, relu
│                              #   before the head-weighted sum, LayerNorm k_norm, interleaved
│                              #   RoPE), plus the streaming running-top-k fold and the additive
│                              #   segment mask attention consumes. Opt-in via VLLM_GLM_DSA=1
├── model_fp8_per_channel.py   # quantization="fp8_per_channel": per-row (ROW) FP8 on HBM, dequant
│                              #   in-kernel — the supported configuration
├── mtp_fp8.py                 # FP8 ROW variant of mtp.py — unsupported path
├── weight_loaders_fp8.py      # Block-128 FP8 dequant + shard / re-quantize weight loaders
├── model.py                   # Base implementation the FP8 classes extend: MLA attention +
│                              #   256-expert top-8 sigmoid MoE with a shared expert, first 3
│                              #   layers dense. Holds the module graph and the BF16 forward
│                              #   path; its own BF16 weight footprint does not fit one node
└── mtp.py                     # GlmMoeDsaMtpForCausalLM: layer-78 head that mtp_fp8.py extends —
                               #   unsupported path, not exercised on device
```

**New — speculative-decoding proposer** (present in the tree; speculative decoding
is not a supported configuration — see [Feature status](#feature-status))

```text
vllm_neuron/vllm/spec_decode/mtp.py   # MtpProposer — mirrors the EagleProposer public surface
```

**New — docs & example**

```text
docs/model-recipes/glm-5.3.md               # Model recipe / card (pointer to this README)
docs/tutorials/tutorial-glm-5.3.md          # Deployment tutorial (pointer to this README)
examples/vllm_neuron/models/glm_moe_dsa/run.py  # Offline generation example (TP=64, ep_degree=16)
```

**Modified — shared framework touch-points**

```text
vllm_neuron/model/registry.py                   # Register GlmMoeDsaForCausalLM (served) and
                                                #   GlmMoeDsaMtpForCausalLM (unsupported path)
vllm_neuron/model/__init__.py                   # Add glm_moe_dsa to the lazy-import allow-list in
                                                #   __getattr__, so the package is only imported
                                                #   when this model is actually requested
vllm_neuron/model/kv_cache.py                   # Add the is_mla flag to LayerSpec (keys the
                                                #   runner's MLAAttentionSpec branch)
vllm_neuron/vllm/platform.py                    # Accept "fp8" in the supported quantization list,
                                                #   so vLLM admits the FP8 checkpoint's own
                                                #   quantization_config at startup
vllm_neuron/vllm/worker/neuron_model_runner.py  # MTP drafter wiring: is_mtp_spec + unified
                                                #   is_spec_decode, MtpProposer dispatch, MLA
                                                #   KV init + MLAAttentionSpec, hidden capture
vllm_neuron/vllm/worker/neuron_worker.py        # Draft prefill graph-extract/warmup covers MTP;
                                                #   post-warmup HBM probe
docs/model-recipes/index.md                     # Add the recipe grid card + toctree entry
docs/tutorials/index.md                         # Add the tutorial grid card + toctree entry
```
