# Contributed Model: GLM-5.2-FP8

vllm-neuron port of `zai-org/GLM-5.2-FP8` for AWS Trainium2, on the
**Neuron 2.32 stack (vLLM 0.24 / plugin 0.24.0.1.1.0)**.

> If you are on the **Neuron 2.31 / vLLM 0.21** stack, use the
> [`add-glm-5-2-231`](https://github.com/htokoyo/vllm-neuron/blob/add-glm-5-2-231/GLM-5.2/README.md)
> tag instead: it holds the predecessor port for that stack, which this branch supersedes.

## Change History

Newest first. Each entry links to the section with the full detail.

| Date | Change |
|---|---|
| 2026-08-31 | Re-hosted onto Neuron 2.32 / vllm-neuron 0.24 (from Neuron 2.31 / 0.21), and extended in the same release: **segmented prefill** made correct and verified on device to `max-model-len` 65536; **speculative decoding (MTP)** at `num_speculative_tokens=1`; the **DSA sparse-attention indexer** as an opt-in path that runs on device at one configuration. Also fixes a prior-KV mask bound in `forward_decode` reachable only under speculation. See [Verification scope](#verification-scope). |
| 2026-07-28 | Initial contribution, on vllm-neuron 0.21 / Neuron 2.31. Adds the `glm_5_2` model package — a 78-layer decoder combining **Multi-head Latent Attention (MLA)** with a **256-expert MoE** (top-8 sigmoid routing plus one always-on shared expert, first three layers dense) — in **FP8 per-channel ROW** at TP=64 / EP=16, with the tiled MLA attention path, the FP8 dequant-and-shard weight loaders and the halved MLA KV page. Component equivalence against `transformers.models.glm_moe_dsa` 13/13 (all R < 1.2). The DSA indexer was **not** included and full attention was used instead. Superseded by this branch and kept at the [`add-glm-5-2-231`](https://github.com/htokoyo/vllm-neuron/blob/add-glm-5-2-231/GLM-5.2/README.md) tag for anyone still on that stack. |

## Introduction

[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) is a large Mixture-of-Experts
(MoE) text-generation model. This is the vllm-neuron implementation of it: a
78-layer decoder that combines **Multi-head Latent Attention (MLA)** with a
**256-expert MoE** feed-forward network (top-8 sigmoid routing plus one always-on
shared expert; the first three layers are dense). It is served text-only under the
architecture `GlmMoeDsaForCausalLM`.

**Compatible model checkpoint:**

| Model | Precision | HuggingFace |
|-------|-----------|-------------|
| GLM-5.2-FP8 | FP8 (128×128 block-quantized) | [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8) |

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

The BF16 release [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2) is not a
supported input for this port.

## Verification scope

Everything below was measured on real hardware — a single **trn2.48xlarge** (16 Trainium2
devices, 64 logical NeuronCores, 1.5 TB HBM), **TP=64 / EP=16**, LNC=2, `fp8_per_channel`,
on the Neuron 2.32 stack.

**Verified on device**

| Area | What was verified |
|---|---|
| FP8 serving | Compiles and serves; coherent text at every configuration below |
| Long context | `max-model-len` up to 65536 with an 8192-wide KV segment; 55/55 graphs, needles 3/3 at 29,212 and 3/3 at 58,408 tokens |
| Segmented prefill | Correct at every supported segment width; prefill cost scales linearly with segment count (see [Measured performance](#measured-performance)) |
| Accuracy | Component equivalence vs `transformers.models.glm_moe_dsa` 13/13 (all R < 1.2), and GSM8K-CoT at n=100 (see [Accuracy Evaluation](#accuracy-evaluation)) |
| Determinism | Byte-identical continuation across the offline `LLM` API and the served endpoint on the same NEFFs, 3/3 repeats |
| Speculative decoding (MTP) | Runs at `num_speculative_tokens=1`, repeatable 8/8, `gsm8k_cot` unchanged against a non-speculative run on the same build (see [Speculative decoding](#accuracy-evaluation)) |
| DSA sparse-attention indexer | Opt-in (`GLM52_DSA=1`); serves at `max-model-len` 4096 with a 512-wide segment, needle retrieved at 10 / 50 / 90 % depth in a 3,521-token context |
| Throughput / latency | TTFT and TPOT, single-request — `max-num-seqs` 1 for the segmented-prefill runs, 8 for the fixed-bucket latency run |

**Not verified / out of scope**

| Area | Status |
|---|---|
| BF16 serving | Out of scope — the BF16 release does not fit one node, see [Why FP8 only](#why-fp8-only-hbm-capacity) |
| End-to-end logit comparison vs HuggingFace | Blocked by host memory and by the reference implementation, not skipped — see [Accuracy Evaluation](#accuracy-evaluation) |
| MTP byte-identity vs a non-speculative run | **Not expected to hold**, and not a defect; the verify step reads logits at a different shape — see [Accuracy Evaluation](#accuracy-evaluation) |
| DSA beyond the one configuration above | Not exercised — see [Accuracy Evaluation](#accuracy-evaluation) |
| NKI MLA attention kernel | Opt-in (`GLM52_MLA_BLOCK_KERNEL=1`), CPU-simulator validated only, **never run on device** |
| On-device top-k / top-p sampling | Supported by `OnDeviceSamplingConfig` (`max_top_k`), but not exercised — every configuration here compiled the `all_greedy` sampling graph |
| Parallel degrees other than TP=64 / EP=16 | Not exercised |
| Pipeline parallelism, multi-node | Not implemented |

## Model Overview

| | |
|---|---|
| HuggingFace ID | `zai-org/GLM-5.2-FP8` |
| Architecture | `GlmMoeDsaForCausalLM` |
| Task | Text generation (`--runner generate`), prefill + decode |
| Layers / hidden size | 78 / 6144 |
| Experts | 256 routed (top-8) + 1 shared |
| Context length | 1,048,576 (`max_position_embeddings`) |
| Vocabulary | 154,880 tokens, untied word embeddings |
| Tensor type | FP8 (e4m3) weights on HBM with per-row scales (`fp8_per_channel`) |

## Model Architecture

Values below come from the port's configuration
([`vllm_neuron/model/glm_5_2/config.py`](../vllm_neuron/model/glm_5_2/config.py)).

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

- **DSA indexer: implemented, opt-in (`GLM52_DSA=1`), off by default.** `indexer_types` marks each
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
([`config.py`](../vllm_neuron/model/glm_5_2/config.py)) — 78 layers of MLA plus 75
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
> `zai-org/GLM-5.2-FP8` checkpoint as downloaded.

## Required knobs

Three settings this model requires. Each is load-bearing: without it the run fails at startup
or at compile time, and none of them announce the cause.

| Item | What happens, and what to do |
|---|---|
| **`enable_prefix_caching=False`** | vLLM 0.24 turns APC on by default (`CacheConfig.enable_prefix_caching = True`), and vllm-neuron rejects APC unless `max_num_batched_tokens` ∈ {512, 1024, 2048, 4096, 8192}. A single-shot recipe therefore **fails at startup** with "Automatic Prefix Caching (APC) requires segmented prefill to be enabled". Every configuration below and in [Verification scope](#verification-scope) disables it; APC itself is neither enabled nor measured on this port. |
| **`VLLM_NEURON_BARRIER_TIMEOUT` (default 3600 s) is too short** | While rank 0 compiles, the other 63 ranks wait in `tp_barrier()`. If the barrier fires first they exit and the run dies mid-build — **after every graph has compiled and been cached**, so the work is not lost but the run is. Raise it well past rank 0's whole cold compile — which ran **4.63 h** in one of the configurations below; see [Step 1](#step-1-environment-setup). |
| **A 512-wide prefill segment when `GLM52_DSA=1`** | At a 1024-wide segment the compiler aborts with `[INTERNAL_ERROR] [NCC_ILSA901] LegalizeSundaAccess assertion error: unexpected AP of matmult dst`; the cause is under investigation. **Use a 512-wide segment** — every DSA configuration in this README was verified with it. |

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
| | DSA sparse-attention indexer | ⚠️ experimental | `GLM52_DSA=1`; validated on CPU against `transformers.models.glm_moe_dsa` (selection set matches the reference exactly) and **runs on device at one verified configuration** (`max_model_len=4096`, 512-wide prefill segment, needle retrieved at 10/50/90% depth in a 3,521-token context). Off by default — still under verification towards a supported implementation |
| | NKI MLA attention kernel | ⚠️ not on device | `GLM52_MLA_BLOCK_KERNEL=1`; **CPU-simulator validated only, never run on device** — off by default. Running it on device is planned |
| **Speculative decoding** | MTP self-speculation | ✅ opt-in | `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`. Runs on device, repeatable 8/8, and `gsm8k_cot` is unchanged against a non-speculative run on the same build (see [Accuracy Evaluation](#accuracy-evaluation)). Off unless explicitly enabled |
| **Context** | `max_model_len` ≤ 16,384 (single-shot prefill) | ✅ | vllm-neuron 0.24 caps single-shot at `MAX_MODEL_LEN_SINGLE_SHOT` = 16 KiB |
| | `max_model_len` > 16,384 | ✅ | Requires chunked prefill (✅ below). Verified on device at **65,536** with `kv_segment_size=8192` (the largest supported segment size): a 58,408-token prompt spanning 8 segments retrieved facts planted in segments 0, 3 and 6. Also verified at 16,384 with `kv_segment_size=4096` |
| **Performance** | Segmented (chunked) prefill | ✅ | Verified on device at `max_model_len=16384` / `kv_segment_size=4096`: facts planted in segments 0, 1 and 2 of an 8,800-token prompt and in 0, 1 and 3 of a 15,100-token one were all retrieved, so the prior-KV path resolves across segment boundaries on hardware. |
| | Automatic Prefix Caching (APC) | ⚠️ not exercised | The blocker is gone (0.24 requires segmented prefill for APC, which now works), but APC itself has not been switched on or measured. Every configuration in this README passes `enable_prefix_caching=False` |
| | On-device sampling — greedy | ✅ | `on_device_sampling_config: {"all_greedy": true}`. Every configuration verified here used it; `SamplingParams(temperature=0)` alone does not select it |
| **Compilation** | torch.compile (XLA backend) | ✅ | |

Legend: **✅** verified on device and usable — `opt-in` means it works but is off by default ·
**⚠️** integrated but not yet dependable, with the qualifier saying why — `experimental` ran on
device in one narrow configuration, `not on device` was verified only off device, `not exercised`
was never enabled · **❌** not implemented. See [Verification scope](#verification-scope) for
exactly what was measured.

## Setup

### Native integration (nothing to apply)

The model is **natively integrated** into this repository — check out this branch
and it works; there is no patch to apply.

- **Model package:** [`vllm_neuron/model/glm_5_2/`](../vllm_neuron/model/glm_5_2/)
  (per-file breakdown: [module structure](../vllm_neuron/model/glm_5_2/README.md)).
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
- GLM-5.2 is large. The shipped example uses `tensor_parallel_size=64`, which at
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
export NEURON_CC_FLAGS="--hbm-scratchpad-page-size=512"
export NEURON_SCRATCHPAD_PAGE_SIZE=512
export NEURON_SKIP_EFA_AFFINITY=1

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
huggingface-cli download zai-org/GLM-5.2-FP8 --local-dir /path/to/GLM-5.2-FP8
```

> **Tip:** Download to a filesystem provisioned for the full ≈753 GB, not a home
> directory — and to a shared one if you intend to serve across nodes.

## Serving

### Offline inference (`llm.generate()`)

[`examples/vllm_neuron/models/glm_5_2/run.py`](../examples/vllm_neuron/models/glm_5_2/run.py)
carries the verified recipe and applies most of the `export`s above via `os.environ.setdefault`,
so the only thing it needs is the checkpoint:
`python run.py --model-checkpoint /path/to/GLM-5.2-FP8`. An explicit `export` still wins, since
`setdefault` only fills unset variables.

### Online serving (OpenAI-compatible)

```bash
vllm serve /path/to/GLM-5.2-FP8 \
    --served-model-name GLM-5.2 \
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
    -d '{"model": "GLM-5.2", "prompt": "The capital of France is",
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
- `quantization` — set it to `fp8_per_channel`. This is **not optional**: without it the
  weights are held in BF16, which does not fit one node
  ([why](#why-fp8-only-hbm-capacity)).

### Quantization

Set `quantization` to **`fp8_per_channel`** in `neuron_config`. It is the one configuration
that holds FP8 weights on HBM end to end, and therefore the one that fits a single
node ([why](#why-fp8-only-hbm-capacity)):

| | `fp8_per_channel` |
|---|---|
| Checkpoint | `zai-org/GLM-5.2-FP8` (128×128 block-quantized) |
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
vllm serve /path/to/GLM-5.2-FP8 \
    --served-model-name GLM-5.2 \
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
[`weight_loaders_fp8.py`](../vllm_neuron/model/glm_5_2/weight_loaders_fp8.py). And whatever
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
much of it one request can occupy. The MTP and DSA runs in **4** and **5** below each used a
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

**2. Accuracy — downstream task.** GSM8K-CoT through the served endpoint, 8-shot,
`do_sample=False`, `num_concurrent=8`:

| Filter | exact_match | Stderr |
|---|---:|---:|
| flexible-extract | **90.0%** | ±3.02 |
| strict-match | **88.0%** | ±3.27 |

> ⚠️ **n = 100**, via `--limit`. See the caveat in
> [Verification scope](#verification-scope) before quoting it. This run is configuration **B**
> (`max_model_len=4096`, `max_num_seqs=8`); the pair in **4** below is a **separate, later run**
> at `max_model_len=2048` / `max_num_seqs=1`, which is why its numbers differ from these by about
> one standard error.

```bash
lm_eval --model local-completions \
  --model_args "model=GLM-5.2,base_url=http://localhost:8000/v1/completions,\
num_concurrent=8,max_retries=3,timeout=1800,tokenized_requests=False,tokenizer=/path/to/GLM-5.2-FP8" \
  --tasks gsm8k_cot --batch_size 1 --limit 100
```

> ⚠️ `timeout=1800` above is load-bearing. lm-eval's `local-completions` backend defaults its
> client timeout to **300 s** (`api_models.py`), and an 8-shot CoT request against this model exceeds
> that, which surfaces as `TimeoutError` and `ServerDisconnectedError` after retries rather than as a
> server-side error. (`--batch_size` is ignored whenever `num_concurrent > 1`: the generative path
> passes `n=0` to the batcher, so it does not compound the concurrency.)

**4. Speculative decoding (MTP).** Measured against a non-speculative run on the same build
and the same instance:

| | MTP, `num_speculative_tokens=1` | non-speculative |
|---|---|---|
| `gsm8k_cot` flexible-extract | **0.89** ± 0.031 | 0.88 ± 0.033 |
| `gsm8k_cot` strict-match | **0.90** ± 0.030 | 0.89 ± 0.031 |
| same prompt → same output | **8 of 8** | 8 of 8 |
| byte-identical to the other column | 4 of 8 | — |

**Byte-identity is not the right test, and its absence is not a defect.** The verify step reads the
target's logits at a different shape from a non-speculative step, so the two are not the same
floating-point expression. Standing in for a direct check, and off device: a 5-layer CPU comparison
puts the logit difference between the shapes at **3.1e-02** against a top-1/top-2 gap of
**1.25e+00** — a 40× margin, so the shape rarely changes which token wins.

> ⚠️ So "every divergence is a near-tie" is *inferred*, not checked position by position:
> `logprobs` on `/v1/completions` raises. It needs `max_logprobs` and `--no-async-scheduling`; the
> offline `LLM` API returns them without either.

**5. DSA sparse-attention indexer.** Opt-in (`GLM52_DSA=1`), and what was verified is narrow:

| | |
|---|---|
| Configuration | `max_model_len=4096`, 512-wide KV segment, one decode context bucket at 2048, `max_num_seqs=1` |
| Graphs compiled | 896, no compile errors |
| KV cache | 64,736 tokens, against 79,136 with DSA off — the 704/576 row-width ratio |
| Retrieval, 3,518-token context (the selection keeps 2,048 of 3,518 keys) | needle found at **10 %, 50 % and 90 % depth** |
| Decode latency at that context | **155.7 ms/token**, from the slope of two output lengths (8 and 72, `ignore_eos`) so prefill and client overhead cancel |

> ⚠️ The 90 % case is the one that means something. At 50 % the needle sits inside the first 2,048
> keys, so a selection that took the leading window — or one that was silently inert — would retrieve
> it too. At 90 % it is outside any leading window, so retrieval requires the scores to have ranked
> that region.

> 🔴 **Not claimed.** No speed figure, because there is no dense counterpart built the same way at
> the same configuration. No accuracy benchmark with DSA on — three needle prompts are not an
> evaluation. Nothing above `max_model_len=4096` has been compiled with DSA, and it has not been run alongside speculative decoding. And **DSA does not
> reduce work in this form**: the attention pass still visits every key and masks the unselected
> ones, so what it buys is fidelity to the trained model, not speed.

## Measured performance

> ⚠️ Single-request throughout, on the configurations under
> [Accuracy Evaluation](#accuracy-evaluation), warm compile cache, greedy on-device sampling. Not a
> deployment characterisation: `decode_context_length_buckets` has never been set, and no
> batched throughput figure has been measured on this port.

### Benchmarking this port yourself

The tables below were produced by a purpose-built harness, not by `vllm bench serve`, so this
command does not reproduce them — it is the model-agnostic starting point if you want your own
numbers. Run it against a server started as in
[Online serving](#online-serving-openai-compatible).

```bash
vllm bench serve \
    --base-url http://localhost:8000 \
    --model GLM-5.2 \
    --tokenizer /path/to/GLM-5.2-FP8 \
    --dataset-name random \
    --random-input-len 32 --random-output-len 96 \
    --random-range-ratio 0 \
    --num-prompts 4 --max-concurrency 1 \
    --ignore-eos \
    --save-result --result-filename glm52_bench_decode.json
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

Segmented prefill, 8 output tokens, sole client:

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

| Prompt tokens | 32 | 249 | 993 | 1,985 | 2,907 | 3,682 |
|---|---:|---:|---:|---:|---:|---:|
| TTFT | 2.16 s | 2.16 s | 2.16 s | 2.16 s | 2.16 s | 2.16 s |
| TPOT | 374.0 ms | 374.0 ms | 374.1 ms | — | 374.1 ms | 374.0 ms |

TTFT spread across the six prompts is 0.005 s (stdev 0.002 s); TPOT varies by 0.1 ms. The
1,985-token case reports no TPOT because the model emitted EOS after one token.

Both are bucketing, not scaling. Prefill pads to the single 4,096-token bucket, so it does
identical work regardless of prompt length. Decode falls back to `max_model_len` because
`decode_context_length_buckets` is unset, so it gathers the full bucket's latent every step
no matter how short the real context is. ⇒ **Setting `decode_context_length_buckets` is a
configuration-only decode improvement**, independent of any kernel work. It is not yet
measured.

> ⚠️ **No batched throughput number has been measured on this port.** MLA keeps one
> compressed latent per token and it is not TP-shardable, so every rank reads the whole
> latent for every sequence every step, which is a reason to expect batching to help less
> here than it would for a GQA model — but that expectation is untested, and the figures
> above are all single-request.

## Contents of this bundle

The model is integrated in-tree (committed directly on this branch), so this
top-level `GLM-5.2/` bundle is supplementary: it carries the source-of-truth
README only. There is no `src/` copy here — that would duplicate the in-tree
package.

```text
GLM-5.2/
└── README.md            # This file — the source of truth for the port
```

The bundle ships no `test/` directory. The verification that was carried out is
described in [Accuracy Evaluation](#accuracy-evaluation), together with the procedure to reproduce
it and an explicit statement of what remains unverified.

### Paths this port touches across the repository

Everything below is stated against the branch this port is based on,
`release-0.24.0.1.1.0` (vllm-neuron 0.24 / Neuron 2.32).

**New — model package** (`vllm_neuron/model/glm_5_2/`)

```text
vllm_neuron/model/glm_5_2/
├── __init__.py                # Package exports (Glm52Config, Glm52ForCausalLM)
├── README.md                  # Module structure (points at this README)
├── config.py                  # Glm52Config: HF → Neuron config translation (MLA ranks,
│                              #   MoE routing, first_k_dense_replace, interleaved RoPE)
├── factory.py                 # Glm52ForCausalLM factory: validates the config and selects
│                              #   the implementation from `quantization`
├── mla_block.py               # NKI dispatch for the MLA inner attention block, opt-in via
│                              #   GLM52_MLA_BLOCK_KERNEL=1 — default OFF, simulator-only
├── mla_block_kernel.py        # The NKI kernel itself: one online-softmax step over a key
│                              #   segment (score + max + rescale + accumulate)
├── dsa_indexer.py             # The DSA indexer: per-query top-k key selection (score, relu
│                              #   before the head-weighted sum, LayerNorm k_norm, interleaved
│                              #   RoPE), plus the streaming running-top-k fold and the additive
│                              #   segment mask attention consumes. Opt-in via GLM52_DSA=1
├── model_fp8_per_channel.py   # quantization="fp8_per_channel": per-row (ROW) FP8 on HBM, dequant
│                              #   in-kernel — the supported configuration
├── mtp_fp8.py                 # FP8 ROW variant of mtp.py — unsupported path
├── weight_loaders_fp8.py      # Block-128 FP8 dequant + shard / re-quantize weight loaders
├── model.py                   # Base implementation the FP8 classes extend: MLA attention +
│                              #   256-expert top-8 sigmoid MoE with a shared expert, first 3
│                              #   layers dense. Holds the module graph and the BF16 forward
│                              #   path; its own BF16 weight footprint does not fit one node
└── mtp.py                     # Glm52MtpForCausalLM: layer-78 head that mtp_fp8.py extends —
                               #   unsupported path, not exercised on device
```

**New — speculative-decoding proposer** (present in the tree; speculative decoding
is not a supported configuration — see [Feature status](#feature-status))

```text
vllm_neuron/vllm/spec_decode/mtp.py   # MtpProposer — mirrors the EagleProposer public surface
```

**New — docs & example**

```text
docs/model-recipes/glm-5.2.md               # Model recipe / card (pointer to this README)
docs/tutorials/tutorial-glm-5.2.md          # Deployment tutorial (pointer to this README)
examples/vllm_neuron/models/glm_5_2/run.py  # Offline generation example (TP=64, ep_degree=16)
```

**Modified — shared framework touch-points**

```text
vllm_neuron/model/registry.py                   # Register GlmMoeDsaForCausalLM (served) and
                                                #   Glm52MtpForCausalLM (unsupported path)
vllm_neuron/model/__init__.py                   # Add glm_5_2 to the lazy-import allow-list in
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
