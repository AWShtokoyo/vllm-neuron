# Contributed Model: llama-embed-nemotron-8b

vllm-neuron implementation of
[`nvidia/llama-embed-nemotron-8b`](https://huggingface.co/nvidia/llama-embed-nemotron-8b),
an 8B **bidirectional Llama embedding (pooling) model**. This README is the
**single source of truth** for the port: architecture, setup, serving,
verification, and measured performance. The `docs/` recipe and tutorial and the
model-package README are thin pointers to this file.

> **Neuron 2.32 / vllm-neuron 0.24.** This branch hosts the port on the public
> `vllm-neuron` **0.24** framework (Neuron 2.32). The five worker / scheduler /
> attention / compile edits the 0.21 port needed are now **upstreamed into the
> framework**, so the shared-file footprint shrinks to **2 files** and the model's
> env gates (`VLLM_NEURON_FORCE_LNC1`, `VLLM_NEURON_POOLING_PACK`,
> `VLLM_NEURON_POOLING_ONDEVICE_GATHER`) are **retired** — the framework runs the
> NKI `attention_cte` kernel natively for `causal_mask=False` at default LNC=2 and
> pools through its native `_pool` path. TP=1 is now ~1.8× faster than 0.21 and is
> the cost sweet spot.

## Introduction

llama-embed-nemotron-8b is an 8B text embedding model from NVIDIA. It runs a
**bidirectional** Llama-3.1-8B backbone, then mask-weighted mean-pools the final
hidden states and L2-normalizes them into a 4096-dimensional sentence embedding.
It targets retrieval, semantic search, clustering, and reranking workloads.

Unlike a generative LLM, this is a **pooling** model: it is prefill-only (no
decode, no KV cache, no sampler) and is served with `--runner pooling`, returning
an embedding vector per input rather than generated tokens.

**Verification scope.** Correctness was verified as embedding equivalence against
the official HuggingFace `LlamaBidirectionalModel` reference on **TP=1/2/4** on a
`trn2` instance (worst cosine ≥ 0.99996). Throughput/latency were measured with
`vllm bench serve` at TP=1/2/4.

**Compatible checkpoints:**

| Model | HuggingFace |
|-------|-------------|
| Llama-Embed-Nemotron-8B | [nvidia/llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b) |

## Model Architecture

The checkpoint uses `model_type` `llama_bidirec` (architecture
`LlamaBidirectionalModel`). It shares the Llama-3.1-8B dense GQA backbone
(hidden_size 4096, 32 layers, 32/8 GQA heads, head_dim 128, intermediate_size
14336, vocab 128256, llama3 RoPE scaling) with three embedding-specific
differences:

- **Bidirectional (non-causal) attention.** Per-query `key_bounds` restrict each
  query to its own sequence's KV range so real queries do not attend to padding,
  and — when multiple sequences are packed into one prefill — do not attend across
  sequence boundaries.
- **Mask-weighted mean-pool + L2 head** instead of an `lm_head`: the final hidden
  states are mean-pooled over real tokens and L2-normalized to a 4096-dim vector.
- **No KV cache** (`use_cache=false`): the model is prefill-only.

## Feature status

| Category | Feature | Status |
|---|---|---|
| **Task** | Embedding / pooling (`--runner pooling`) | ✅ |
| | Text generation / decode | ❌ (embedding model, prefill-only) |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor parallelism (TP=1/2/4) | ✅ Verified |
| | Pipeline parallelism (PP) | ❌ |
| | Context parallelism (CP) | ❌ |
| **Performance** | Batched multi-sequence prefill packing | ✅ native (scheduler) |
| | Non-causal `attention_cte` NKI kernel @ LNC=2 | ✅ native (all TP) |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (testing) | ✅ |

**Tensor parallelism.** TP=1, TP=2, and TP=4 are all validated. On 0.24 the
framework runs the native NKI `attention_cte` kernel for `causal_mask=False` at
default **LNC=2** (both cores), so a single 1/4 chip already saturates the
prefill path — **TP=1 is ~1.8× faster than on 0.21** (peak 3,583 → 6,440 tok/s).
Scaling above TP=1 is therefore **sub-linear**: TP=2 ≈ 1.46× and TP=4 ≈ 1.80×
raise the absolute ceiling and cut latency, but the *cost* sweet spot is **TP=1**
(see **Measured performance**). No `VLLM_NEURON_FORCE_LNC1` flag is needed at any
TP — it is retired.

**Prefill bucket sizes must be greater than 128** (`num_batched_tokens_buckets`).

## Setup

### Native integration (nothing to apply)

The model is **natively integrated** into this repository — you do not need to
apply anything to use it:

- Model package: [`vllm_neuron/model/llama_bidirec/`](../vllm_neuron/model/llama_bidirec/)
  (architecture `LlamaBidirectionalModel`, `model_type` `llama_bidirec`).
- Registration and the two shared-file touch-points (`model/registry.py`,
  `vllm_neuron/__init__.py`) are committed directly on this branch. On 0.24 the
  0.21 port's scheduler / worker / runner / attention / compile edits are
  **upstreamed into the framework**, so nothing else is patched. The bundled
  `integration.patch` reproduces the two shared-file edits for an out-of-tree
  install and is not needed here.

### Step 1: Environment Setup

Verify Neuron devices are visible:

```bash
neuron-ls
# Expected: NeuronCores listed for your trn2 instance
```

Set environment variables before running any inference script:

```bash
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2   # target Trainium2 (required on trn2)
export NEURON_SKIP_EFA_AFFINITY=1             # skip EFA NUMA-affinity probe
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# vLLM/NEFF compile cache — point at fast local storage (e.g. local NVMe)
export VLLM_CACHE_ROOT=/path/to/vllm_cache

# Extend timeouts for the first (compile) run
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_NEURON_COMPILATION_TIMEOUT=1200
```

All TP configurations run at the framework default **LNC=2** — there are no
model-specific env flags to set (the 0.21 `VLLM_NEURON_FORCE_LNC1` /
`VLLM_NEURON_POOLING_PACK` / `VLLM_NEURON_POOLING_ONDEVICE_GATHER` gates are
retired on 0.24).

### Step 2: Download the Model

```bash
huggingface-cli download \
    nvidia/llama-embed-nemotron-8b \
    --local-dir /path/to/llama-embed-nemotron-8b
```

> **Tip:** On a `trn2` cluster, download to a shared or local-NVMe filesystem
> instead of your home directory to avoid NFS write issues.

## Serving

### Online serving (OpenAI-compatible `/v1/embeddings`)

Embedding models are prefill-only, so there are no decode/sampling flags;
`num_batched_tokens_buckets` controls the compiled prefill shapes and its entries
**must be greater than 128**. TP=1 is the cost sweet spot; raise
`--tensor-parallel-size` to 2 or 4 only when you need higher absolute throughput
or lower latency.

```bash
vllm serve /path/to/llama-embed-nemotron-8b \
    --runner pooling \
    --tensor-parallel-size 1 \
    --max-model-len 512 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [256, 512]}}'
```

Request embeddings once the server is up:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
resp = client.embeddings.create(
    model="/path/to/llama-embed-nemotron-8b",
    input=[
        "The capital of France is Paris.",
        "Trainium is an AWS machine learning accelerator.",
    ],
)
for d in resp.data:
    print(len(d.embedding))   # 4096, L2-normalized
```

### Offline inference (`llm.embed()`)

The ready-to-run example
[`examples/vllm_neuron/models/llama_bidirec/run.py`](../examples/vllm_neuron/models/llama_bidirec/run.py)
defaults to this configuration. Equivalent direct `LLM` call:

```python
import vllm_neuron  # registers llama_bidirec + platform + AutoConfig
from vllm import LLM

llm = LLM(
    model="/path/to/llama-embed-nemotron-8b",
    runner="pooling",
    tensor_parallel_size=1,
    max_model_len=512,
    max_num_seqs=4,                       # >1 packs multiple sequences per prefill
    enable_prefix_caching=False,
    additional_config={"neuron_config": {"num_batched_tokens_buckets": [256, 512]}},
)
outputs = llm.embed([
    "The capital of France is Paris.",
    "Trainium is an AWS machine learning accelerator.",
    "Sentence embeddings power semantic search and retrieval.",
])
for out in outputs:
    print(len(out.outputs.embedding))     # 4096, L2-normalized
```

**Configuration notes.** On 0.24 the pooling throughput optimizations are handled
**natively by the framework** — the 0.21 env gates are gone:

- **Multi-sequence prefill packing** is handled by the scheduler. Set
  `max_num_seqs > 1` to pack multiple embedding prefills into one bucket for
  higher short-prompt throughput.
- **On-device MEAN + L2 pooling** runs through the framework's native `_pool`
  path, returning the per-sequence embedding directly without a per-step host copy
  of the full hidden-state buffer.

**Bucket sizes** (`num_batched_tokens_buckets`) control the discrete padded
prefill shapes compiled into each NEFF. Each bucket adds compile time; start with
one or two and add more as your input-length distribution requires. All entries
must be greater than 128.

## Verification

Correctness is validated as **embedding equivalence** against the official
HuggingFace `LlamaBidirectionalModel` reference (sentence-transformers mean-pool
+ L2), measured as cosine similarity (`1.0` is identical). The equivalence tests
ship in this bundle under [`test/equivalence/`](test/equivalence/):

```bash
# CPU equivalence (fast, no device): full vLLM pooling path vs HF reference
VLLM_NEURON_CPU_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python3 test/equivalence/cpu_equivalence.py

# Batched equivalence: 4 seqs in ONE prefill must match solo (no cross-seq leak)
VLLM_NEURON_CPU_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python3 test/equivalence/batch_equivalence.py

# On-device equivalence (trn2): builds the HF reference on a transformers-4.x
# venv (REF_VENV), then compares against the on-device 2.32 result.
REF_VENV=$HOME/tf4x_venv python3 test/equivalence/ondevice_equivalence.py
```

| Test | Result |
|---|---|
| CPU equivalence (full vLLM pooling path, tiny weights) | worst cos = 0.999966 |
| On-device equivalence (trn2 TP=1, real 8B) | worst cos = 0.999963 |
| Batched on-device (4 seqs / 1 prefill, vs solo) | cos = 1.000000 (no cross-sequence leak) |
| Batched vs HF reference | cos = 0.99995+ |
| vLLM serving `/v1/embeddings` vs HF reference | worst cos = 0.999963 |
| Cross-prompt similarity (discriminative check) | ~0.28–0.30 (distinct prompts stay distinct) |

> **Reference build note.** On the Neuron 2.32 stack (transformers ≥ 5) the HF
> *reference* class must be built on a **transformers-4.x** venv — the equivalence
> tests split into a reference-build stage (CPU, 4.x) and an on-device compare
> stage. The port itself is bidirectional and correct on the current stack; only
> the upstream HF reference class breaks on transformers ≥ 5.

Throughput/latency were benchmarked with the built-in, model-agnostic
`vllm bench serve` client against a running server:

```bash
vllm bench serve \
    --backend openai-embeddings \
    --model /path/to/llama-embed-nemotron-8b \
    --endpoint /v1/embeddings \
    --dataset-name random --random-input-len 256 \
    --num-prompts 64 --max-concurrency 4
```

## Measured performance

`trn2`, bf16, `--runner pooling`, `random` dataset, 64 prompts, `max_model_len`
512, Neuron 2.32 / vLLM 0.24 (default LNC=2, no env flags; re-measured
2026-08-18). Throughput is total token throughput (input tokens only — embedding
has no output tokens); latency is end-to-end per request.

| TP | input len | concurrency | req/s | tok/s | p50 (ms) | p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 500 | 4 | 12.88 | **6,440** ⭐️ | 308.6 | 314.7 |
| 1 | 256 | 2 | 24.21 | 6,198 | 81.3 | 86.3 |
| 1 | 128 | 4 | 24.23 | 3,101 | 163.4 | 168.3 |
| 2 | 500 | 4 | 18.83 | 9,416 | 209.7 | 216.7 |
| 2 | 256 | 4 | 31.40 | 8,039 | 125.5 | 130.7 |
| 2 | 128 | 4 | 31.93 | 4,087 | 123.2 | 128.5 |
| 4 | 500 | 2 | 23.20 | 11,601 | 84.0 | 90.7 |
| 4 | 256 | 4 | 36.86 | 9,436 | 106.1 | 112.0 |
| 4 | 128 | 1 | 29.62 | 3,791 | 32.5 | 37.5 |

Peak throughput per TP: **TP=1 ≈ 6,440 tok/s** (500/c4), **TP=2 ≈ 9,416 tok/s**
(1.46× TP=1, 500/c4), **TP=4 ≈ 11,601 tok/s** (1.80× TP=1, 500/c2). Because the
native LNC=2 kernel already saturates a 1/4 chip at TP=1, scaling is sub-linear:
per NeuronCore, **TP=1 delivers the most throughput** — normalising the peaks by
TP gives 6,440 tok/s/core at TP=1 versus 4,708 at TP=2 and 2,900 at TP=4, i.e.
**TP=1 is 1.37× TP=2 and 2.22× TP=4 per core.** Lowest latency is at TP=4 (p50 ~33 ms
at 128/c1). Full sweeps (TP=1/2/4 × input 128/256/500 × concurrency 1/2/4) are
recorded in `INTERNAL/bringup-benches/`.

## Contents of this bundle

The model is integrated in-tree (committed directly on this branch), so the
top-level `llama-embed-nemotron-8b/` bundle is supplementary — it carries this
source-of-truth README, a standalone patch of the same edits, and the pooling
equivalence tests (the repo's generic logit-comparison scripts are causal-LM-only
and do not apply to a pooling model, so the cosine-equivalence tests are retained
here).

```text
llama-embed-nemotron-8b/
├── README.md                          # This file — the source of truth for the port
├── integration.patch                  # The two shared-file edits as a standalone patch,
│                                       #   for applying the model on top of a separately-
│                                       #   installed vllm_neuron (already committed in-tree
│                                       #   here; kept for reference / out-of-tree installs)
└── test/equivalence/                  # Embedding-equivalence tests vs the HF reference
    ├── cpu_equivalence.py              #   Full vLLM pooling path vs HF ref (CPU, tiny weights)
    ├── batch_equivalence.py           #   4 seqs / 1 prefill vs solo (no cross-seq leak)
    └── ondevice_equivalence.py        #   Real 8B on trn2 vs transformers-4.x HF ref
```

Serving + performance sweep scripts and the recorded bench results (`results/tpN/`)
are kept internal under `INTERNAL/bringup-benches/` (not part of the public
bundle); their measured values are transcribed into **Measured performance** above.

### Paths this port touches across the repository

**New — model package** (`vllm_neuron/model/llama_bidirec/`)

```text
vllm_neuron/model/llama_bidirec/
├── __init__.py    # Package exports (LlamaBidirecConfig, LlamaBidirectionalModel)
├── README.md      # Package-level module structure (points here)
├── config.py      # LlamaBidirecConfig (Llama-3 8B backbone shape + neuron_config)
├── factory.py     # Model construction / weight-loading factory
└── model.py       # Bidirectional Llama backbone + mask-weighted mean-pool + L2-norm
                   #   → 4096-dim embedding; per-query key_bounds; no KV cache
```

**New — docs & example**

```text
docs/model-recipes/llama-embed-nemotron-8b.md           # Model recipe / card (pointer to this README)
docs/tutorials/tutorial-llama-embed-nemotron-8b.md      # Deployment tutorial (pointer to this README)
examples/vllm_neuron/models/llama_bidirec/run.py        # Offline embedding example (llm.embed(), TP=1 default)
```

**Modified — shared framework touch-points** (2 files; both additive)

```text
vllm_neuron/model/registry.py                       # Register LlamaBidirectionalModel
vllm_neuron/__init__.py                             # Register the llama_bidirec HuggingFace AutoConfig
docs/model-recipes/index.md                         # Link the model recipe
docs/tutorials/index.md                             # Link the tutorial
```

> On 0.24 the five 0.21 shared-file edits — pooling runner mode
> (`neuron_model_runner.py`), decode-graph skip (`neuron_worker.py`),
> multi-sequence prefill packing (`scheduler.py`), the non-causal attention route
> (`attention_cte.py`), and the LNC1 codegen flag (`compile/backend.py`) — are
> **upstreamed into the framework** and no longer patched by this port. In
> particular the framework runs the NKI `attention_cte` kernel natively for
> `causal_mask=False`, so it compiles at default LNC=2 without the old
> `VLLM_NEURON_FORCE_LNC1` workaround.
