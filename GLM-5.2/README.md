# Contributed Model: GLM-5.2

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
therefore a single one — `quantization: "fp8_fwd"`, which keeps FP8 weights on HBM
and dequantizes inside the NKI kernels (see [Quantization](#quantization)).

Note that reading an FP8 checkpoint is **not** the same as holding FP8 weights on
HBM: a mode that dequantizes to BF16 at load time has the BF16 footprint and so
does not fit either. The BF16 release
[`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2) is not a supported
input for this port.

## Verification scope

> **Validation in progress.** This port is functionally integrated: the
> `GlmMoeDsaForCausalLM` architecture registers, builds, compiles, and
> **runs end to end on one `trn2.48xlarge`** at TP=64 / EP=16 / `fp8_fwd`.
> **On-device accuracy validation has not passed yet**, so every figure in this
> document is **reference information from a single configuration, not a
> validated result** — see [Measured performance](#measured-performance).

**Device smoke check (not a gate).** Served at TP=64 / EP=16 / `fp8_fwd` with
`max_model_len=128`, the prompt `The capital of France is` returns the expected
continuation:

```text
"The capital of France is"  →  " Paris. Distance from Paris to Lyon is"
```

That output was **byte-for-byte identical** between the offline `LLM` API and the
`vllm serve` `/v1/completions` endpoint on the same compiled NEFFs, and identical
again across three repeats — so the graph is correctly wired and greedy decoding
is deterministic on device. It is a **sanity check, not accuracy verification**: a
plausible continuation shows the model is assembled and running, not that its
logits match a reference. Establishing the latter requires accuracy verification,
which has not been carried out — see [Verification](#verification).

Specifically **not verified** in this port:

- **On-device accuracy** of the supported `fp8_fwd` configuration — accuracy
  verification has not been carried out, see [Verification](#verification). The
  smoke check above is not a substitute for it.
- **Throughput and latency beyond the one configuration measured.** The figures
  under [Measured performance](#measured-performance) come from the single
  minimal configuration compiled at this stage — `max_model_len=128`, one prefill
  bucket, `max_num_seqs=1`. Nothing is measured for the `max_model_len=4096`
  recipes shown under [Serving](#serving), or for batched or concurrent decode.
- **Speculative decoding**, which this port does not support — see
  [Feature status](#feature-status). Nothing about it has been run on device.
- **The DSA (sparse-attention) indexer**, which this port deliberately does not
  implement: full attention is used for all tokens. Long-context behaviour past
  the bring-up context length is therefore untested.

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
| Tensor type | FP8 (e4m3) weights on HBM with per-row scales (`fp8_fwd`) |

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

- **DSA indexer omitted.** The reference checkpoint ships a DeepSeek-style Sparse
  Attention (DSA) indexer (`wq_b`, `wk`, `k_norm`, `weights_proj`). This port does
  not load or run the indexer and uses **full attention** over all tokens
  instead — functionally correct, but less efficient than sparse attention at very
  long context.
- **MLA single compressed latent.** Because MLA stores one compressed latent per
  token instead of separate per-head K and V, the port emits an
  `MLAAttentionSpec`, whose per-block page size drops the K+V factor of two that a
  standard attention layer budgets — roughly halving KV-cache HBM (equivalently,
  doubling the block count at a fixed KV budget).
- **Layer-78 MTP head (present in the checkpoint, not used).** The checkpoint
  carries one extra decoder layer intended as a Multi-Token Prediction head. This
  port does not serve with it; speculative decoding is not supported — see
  [Feature status](#feature-status).

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
| **FP8** (`fp8_fwd`) | ≈753 GB | **≈11.8 GB** | fits, with room for KV and activations |
| **BF16** | ≈1,506 GB | **≈23.5 GB** | **does not fit** — over budget on weights alone |

BF16 exceeds the per-rank budget **before** any KV cache, activation or scratch
memory is added, so it cannot be closed by tuning `gpu_memory_utilization`,
shrinking `max_model_len`, or reducing the batch size — it would need at least two
nodes.

The distinction that matters is **what is resident on HBM, not what the checkpoint
holds.** `fp8_fwd` stores e4m3 weights plus a per-output-channel scale vector and
dequantizes inside the NKI kernels, so the FP8 footprint above is what the device
actually carries. A mode that instead dequantizes the checkpoint to BF16 while
loading, or that keeps a BF16 copy of the weights for its decode path, lands in the
BF16 row regardless of the checkpoint it read — which is why `fp8_fwd` is the only
configuration this port documents.

> This table is a **derivation from the model configuration and the published HBM
> capacity, not a measurement.** It is stated here because it determines what this
> port supports; it is not an accuracy or performance result. The one
> cross-check available is that the ≈753 GB figure matches the size of the
> `zai-org/GLM-5.2-FP8` checkpoint as downloaded.

## Feature status

| Category | Feature | Status | Notes |
|---|---|---|---|
| **Inputs** | Text | ✅ | Text-only serving |
| | Multimodal (image / video) | ❌ | |
| **Quantization** | FP8 per-row weights on HBM (`fp8_fwd`) | ✅ | `ROW` mode, dequant in-kernel; the supported configuration |
| | BF16 weights on HBM | ❌ | Does not fit one node — [why](#why-fp8-only-hbm-capacity) |
| **Parallelism** | Tensor parallelism (TP) | ✅ | |
| | Expert parallelism (EP) | ✅ | `ep_degree` in `neuron_config` |
| | Pipeline / context parallelism | ❌ | |
| **Attention** | Multi-head Latent Attention (MLA) | ✅ | `MLAAttentionSpec`, halved KV page |
| | DSA sparse-attention indexer | ❌ | Skipped; full attention used |
| **Speculative decoding** | MTP self-speculation | ❌ | Not supported; do not pass `--speculative-config` |
| **Performance** | Segmented prefill | ✅ | |
| | On-device sampling (greedy, top-k, top-p) | ✅ | |
| **Compilation** | torch.compile (XLA backend) | ✅ | |

Every ✅ entry is integrated and compiles; see
[Verification scope](#verification-scope) for on-device status.

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

- `trn2` hardware with Neuron SDK `2.31` or later. See the
  [setup guide](../docs/getting-started/setup-guide.md).
- vLLM Neuron plugin `0.21.0.1.0.0` or above.
- Python 3.10+.
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
export VLLM_NEURON_COMPILATION_TIMEOUT=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Scratchpad settings used during bring-up
export NEURON_CC_FLAGS="--hbm-scratchpad-page-size=512"
export NEURON_SCRATCHPAD_PAGE_SIZE=512
export NEURON_SKIP_EFA_AFFINITY=1

# Serial compile/trace avoids host OOM (kernel kill) on graphs this large
export VLLM_NEURON_PARALLEL_COMPILE_WORKERS=1
export VLLM_NEURON_PARALLEL_TRACE_WORKERS=1
```

### Step 2: Download the model

Only the FP8 checkpoint is used — see
[Why FP8 only](#why-fp8-only-hbm-capacity):

```bash
huggingface-cli download zai-org/GLM-5.2-FP8 --local-dir /path/to/GLM-5.2-FP8
```

> **Tip:** Download to a filesystem provisioned for the full ≈753 GB, not a home
> directory — and to a shared one if you intend to serve across nodes.

## Serving

### Offline `LLM` API

```python
from vllm import LLM, SamplingParams

# GLM-5.2: 64 attention heads, 256 routed experts.
# EP config: world_size=64, ep_degree=16 → tp_sub=4.
llm = LLM(
    model="/path/to/GLM-5.2-FP8",
    max_model_len=128,
    max_num_seqs=1,
    max_num_batched_tokens=128,
    tensor_parallel_size=64,
    enable_expert_parallel=True,
    gpu_memory_utilization=0.92,
    additional_config={
        "neuron_config": {
            "ep_degree": 16,
            "quantization": "fp8_fwd",   # required — see "Why FP8 only"
            "num_batched_tokens_buckets": [128],
            "num_seqs_buckets": [1],
        }
    },
)
outputs = llm.generate(["The capital of France is"],
                       SamplingParams(max_tokens=16, temperature=0.0))
print(outputs[0].outputs[0].text)
```

> The shipped example
> [`examples/vllm_neuron/models/glm_5_2/run.py`](../examples/vllm_neuron/models/glm_5_2/run.py)
> predates this FP8-only scope: it defaults `--model-checkpoint` to the BF16
> repository and exposes no `quantization` argument, so it does not express a
> configuration that fits one node. Use the `LLM` call above, or set
> `neuron_config["quantization"]` in the script, until the example is updated.

### Online serving (OpenAI-compatible)

```bash
vllm serve /path/to/GLM-5.2-FP8 \
    --served-model-name GLM-5.2 \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --additional-config '{"neuron_config": {"ep_degree": 16, "quantization": "fp8_fwd", "num_batched_tokens_buckets": [4096], "num_seqs_buckets": [8], "on_device_sampling_config": {"all_greedy": true}}}'
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
- `quantization` — set it to `fp8_fwd`. This is **not optional**: without it the
  weights are held in BF16, which does not fit one node
  ([why](#why-fp8-only-hbm-capacity)).

### Quantization

Set `quantization` to **`fp8_fwd`** in `neuron_config`. It is the one configuration
that holds FP8 weights on HBM end to end, and therefore the one that fits a single
node ([why](#why-fp8-only-hbm-capacity)):

| | `fp8_fwd` |
|---|---|
| Checkpoint | `zai-org/GLM-5.2-FP8` (128×128 block-quantized) |
| Weights on HBM | FP8 `e4m3` + a per-output-channel (per-row) dequant scale vector |
| Where dequant happens | Inside the NKI kernels (`QuantizationType.ROW`) |
| Activation scales | Not required |
| MLP / MoE weights | FP8 on HBM |
| MLA attention weights | BF16 — MLA has no kernel-side FP8 support, so these are dequantized at load |

At load time the checkpoint's block-FP8 weights are dequantized, an amax is taken
per output channel, and the weights are written back as `e4m3` FP8 alongside their
per-row scale. Decode feeds those FP8 weights and scales straight to the MoE and
MLP kernels; prefill dequantizes to BF16 as a **transient** inside the forward
pass, so no second copy of the weights is kept resident. Attention is the one
exception — MLA runs in BF16 — but the MLP and MoE weights are the overwhelming
majority of the model (75 layers × 256 experts), so the capacity conclusion above
holds.

```bash
vllm serve /path/to/GLM-5.2-FP8 \
    --served-model-name GLM-5.2 \
    --max-model-len 4096 --max-num-seqs 8 \
    --tensor-parallel-size 64 --enable-expert-parallel \
    --additional-config '{"neuron_config": {"ep_degree": 16, "quantization": "fp8_fwd"}}'
```

> **Not verified:** `fp8_fwd` has not passed the on-device accuracy gate, so the
> cost of collapsing the checkpoint's 128×128 block scales to per-row scales is
> **unmeasured** on this port, and compounds over 78 layers. Do not assume it
> matches a BF16 reference until the gate under [Verification](#verification)
> passes.
>
> Two other quantization values are accepted by the code but are **not** documented
> configurations, because neither keeps FP8 weights on HBM: `fp8` dequantizes the
> checkpoint to BF16 at load and reuses the BF16 compute graph, and `fp8_native`
> (per-tensor scales) materializes a BF16 copy of the MoE weights for its decode
> path *in addition to* the FP8 originals its prefill path still needs. Both carry a
> BF16-or-larger footprint, so both fall in the BF16 row of
> [Why FP8 only](#why-fp8-only-hbm-capacity).

### A note on naming

`fp8` / `fp8_native` / `fp8_fwd` are **names internal to this GLM package**, not
standard PyTorch or HuggingFace terminology — the branch in
[`factory.py`](../vllm_neuron/model/glm_5_2/factory.py) is their whole definition.
They describe a different layer than the checkpoint's own
`quantization_config` in `config.json` (`quant_method: "fp8"`, `fmt: "e4m3"`,
`weight_block_size: [128, 128]`): that field states **how the checkpoint is
stored**, whereas `neuron_config.quantization` states **what this port puts on HBM
and which kernels it runs**. The same string `"fp8"` appears in both places with
different meanings. This port's `config.py` does not read the checkpoint's
`quantization_config` at all; the block size of 128 lives in
[`weight_loaders_fp8.py`](../vllm_neuron/model/glm_5_2/weight_loaders_fp8.py).
Note also that whatever `activation_scheme` says, this port quantizes weights
only — activations stay in BF16.

## Verification

**Scope of this stage.** The objective so far has been **functional
verification** — confirming that the model registers, compiles, and generates the
expected output end to end on device. That is a functional statement, not an
accuracy one: accuracy verification has not been carried out — see (1) below. For
that purpose the port has been compiled for a **single, minimal configuration**:

| | |
|---|---|
| `max_model_len` | 128 |
| `num_batched_tokens_buckets` | `[128]` |
| `num_seqs_buckets` | `[1]` |

Everything below — the commands and the recorded figures — was carried out within
that one configuration. **Larger context lengths and batch
sizes are simply not verified yet; they are not restricted by this port.** Each
additional shape requires its own compilation, which was outside the scope of
this stage. The `max_model_len=4096` recipes under [Serving](#serving) are the
intended deployment shapes and remain to be exercised.

**1. Accuracy — not carried out at this point.** No accuracy verification has been
run for this port.

**2. Benchmark — `vllm bench serve`.** The built-in, model-agnostic client, run
against a server started as in
[Online serving](#online-serving-openai-compatible). The command below carries the
parameters of the runs recorded under
[Measured performance](#measured-performance) — the `Decode` case. Its shape
flags are sized to the minimal configuration above, so the runs stay inside the
one compiled bucket and reuse the warm compile cache:

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

The other four cases are the same command with `--random-input-len` /
`--random-output-len` / `--num-prompts` set to the values in that table's first
three columns (e.g. `32 / 1 / 8` for `TTFT-32`).

When you compile for larger shapes — for example the `max_model_len=4096`,
`--max-num-seqs 8` recipes under [Serving](#serving) — scale these flags up
accordingly:

- **`--random-input-len` + `--random-output-len` must fit `max_model_len`**, and
  **`--max-concurrency` should equal one of the compiled `num_seqs_buckets`
  values.** Requesting a shape outside the compiled buckets triggers a
  compilation for that shape, and a `--max-concurrency` above the compiled batch
  bucket leaves the extra requests queueing, which distorts TPOT.
- **`--random-range-ratio 0`** pins the actual input length to
  `--random-input-len`. Without it the client samples a range, so the prompts are
  not the length requested and runs are not comparable.
- **`--ignore-eos`** fixes the output length, which is what makes throughput and
  latency stable across runs.
- **`--model` must be the `--served-model-name`** (`GLM-5.2` above), not the
  checkpoint path. Because that name is not resolvable locally, pass
  **`--tokenizer <checkpoint path>`** as well, or every request fails with
  `is not a local folder`.

Use `--dataset-name sharegpt` (which flows through the chat template) when
representative content matters rather than fixed-length shapes.

### Measured performance

> ⚠️ **Reference information only — the accuracy gate has not passed.** The
> numbers below were measured on device, but they describe a configuration whose
> **output correctness is unverified** (see
> [Verification scope](#verification-scope)). A performance figure for a
> configuration that is not known to compute the right answer cannot be used to
> qualify this port, compare it against another platform, or size a deployment.

Measured on one `trn2.48xlarge` (16 × Trainium2, LNC2 = 64 logical cores) with
`vllm bench serve` as shown under [Verification](#verification), against a server
started as in [Online serving](#online-serving-openai-compatible) — random
dataset, `--random-range-ratio 0`, `--ignore-eos`, greedy on-device sampling, warm
compile cache:

| Configuration | |
|---|---|
| Parallelism | `tensor_parallel_size=64`, `ep_degree=16` (single node) |
| Quantization | `fp8_fwd` (FP8 weights on HBM, per-row scales) |
| `max_model_len` | 128 |
| Prefill bucket / batch bucket | `num_batched_tokens_buckets=[128]` / `num_seqs_buckets=[1]` |
| Sampling | on-device, `all_greedy` |
| Compiler flags | `--hbm-scratchpad-page-size=512` |
| Speculative decoding | off (no `--speculative-config`) |
| KV cache | 79,136 tokens |

This is the minimal configuration described at the top of
[Verification](#verification), so all five cases share the one compiled bucket.
Larger `max_model_len` and batch sizes are **unmeasured, not unsupported**. Two
passes per case; pass 2 is recorded (pass 1 agreed within 1%).

| Case | Input | Output | Requests | Median TTFT | Mean TPOT | Output tok/s |
|---|---:|---:|---:|---:|---:|---:|
| TTFT-32 | 32 | 1 | 8 | 321.65 ms | — | 3.10 |
| TTFT-64 | 64 | 1 | 8 | 320.94 ms | — | 3.11 |
| TTFT-96 | 96 | 1 | 8 | 322.75 ms | — | 3.09 |
| Decode | 32 | 96 | 4 | 332.13 ms | 58.60 ms | 16.28 |
| Mixed | 64 | 64 | 4 | 325.58 ms | 58.59 ms | 15.93 |

**How to read these — the shape matters more than the values:**

- **Equal TTFT at input 32 / 64 / 96 is the expected result, not a measurement
  artifact.** All three prompts land in the same fixed 128-token prefill bucket,
  so prefill does identical work regardless of the real prompt length; identical
  TTFT is precisely what confirms bucketing behaves as designed. It also means
  none of these figures describes prefill scaling with input length.
- **Decode costs ~58.6 ms/token and is independent of input length**, consistent
  across both multi-token cases.
- **Concurrency was 1 because that is the configuration compiled at this
  stage**, not because the port is limited to one request: with
  `num_seqs_buckets=[1]` no batched decode graph exists, so concurrent requests
  simply queue. **No
  concurrency or batch-scaling figure can be derived from this table**, and the
  aggregate `output tok/s` is a single-stream number, not a throughput ceiling for
  the hardware. Batched serving needs a server compiled with larger
  `num_seqs_buckets` — see the `--max-num-seqs 8` recipes under
  [Serving](#serving).
- The single-token-output cases report no TPOT because TPOT needs at least two
  output tokens.

Once the accuracy gate passes, these should be re-measured at a realistic
`max_model_len` and `max_num_seqs` before any of them is treated as a
performance characterization of this port.

## Contents of this bundle

The model is integrated in-tree (committed directly on this branch), so this
top-level `GLM-5.2/` bundle is supplementary: it carries the source-of-truth
README only. There is no `src/` copy here — that would duplicate the in-tree
package.

```text
GLM-5.2/
└── README.md            # This file — the source of truth for the port
```

The bundle ships no `test/` directory. Verification has not been carried out at
this point — see [Verification](#verification) for the procedure and the scope of
what has been done so far.

### Paths this port touches across the repository

**New — model package** (`vllm_neuron/model/glm_5_2/`)

```text
vllm_neuron/model/glm_5_2/
├── __init__.py                # Package exports (Glm52Config, Glm52ForCausalLM)
├── README.md                  # Module structure (points at this README)
├── config.py                  # Glm52Config: HF → Neuron config translation (MLA ranks,
│                              #   MoE routing, first_k_dense_replace, interleaved RoPE)
├── factory.py                 # Glm52ForCausalLM factory: validates the config and selects
│                              #   the implementation from `quantization`
├── model_fp8_fwd_dequant.py   # quantization="fp8_fwd": per-row (ROW) FP8 on HBM, dequant
│                              #   in-kernel — the supported configuration
├── mtp_fp8.py                 # FP8 ROW variant of mtp.py — unsupported path
├── weight_loaders_fp8.py      # Block-128 FP8 dequant + shard / re-quantize weight loaders
├── model.py                   # Base implementation the FP8 classes extend: MLA attention +
│                              #   256-expert top-8 sigmoid MoE with a shared expert, first 3
│                              #   layers dense. Holds the module graph and the BF16 forward
│                              #   path; its own BF16 weight footprint does not fit one node
├── mtp.py                     # Glm52MtpForCausalLM: layer-78 head that mtp_fp8.py extends —
│                              #   unsupported path, not exercised on device
├── model_fp8.py               # quantization="fp8": dequantizes block-FP8 → BF16 at load and
│                              #   reuses model.py's graph — BF16 footprint on HBM
└── model_fp8_native.py        # quantization="fp8_native": per-tensor FP8 scales, but keeps a
                               #   BF16 decode copy alongside the FP8 weights — see
                               #   "Quantization" for why neither of these two is documented
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
vllm_neuron/model/kv_cache.py                   # Add the is_mla flag to LayerSpec (keys the
                                                #   runner's MLAAttentionSpec branch)
vllm_neuron/vllm/platform.py                    # Accept "fp8" in the supported quantization list,
                                                #   so vLLM admits the FP8 checkpoint's own
                                                #   quantization_config at startup
vllm_neuron/compile/backend.py                  # Honor VLLM_NEURON_DISABLE_INPUT_DEDUP
                                                #   (skip FX-graph input dedup)
vllm_neuron/envs.py                             # Add the VLLM_NEURON_DISABLE_INPUT_DEDUP flag
vllm_neuron/vllm/worker/neuron_model_runner.py  # MTP drafter wiring: is_mtp_spec + unified
                                                #   is_spec_decode, MtpProposer dispatch, MLA
                                                #   KV init + MLAAttentionSpec, hidden capture
vllm_neuron/vllm/worker/neuron_worker.py        # Draft prefill graph-extract/warmup covers MTP;
                                                #   post-warmup HBM probe
docs/model-recipes/index.md                     # Add the recipe grid card + toctree entry
docs/tutorials/index.md                         # Add the tutorial grid card + toctree entry
```
