# Contributed Model: llama-embed-nemotron-8b

vllm-neuron implementation of
[`nvidia/llama-embed-nemotron-8b`](https://huggingface.co/nvidia/llama-embed-nemotron-8b),
an 8B **bidirectional Llama embedding (pooling) model**. This README is the
**single source of truth** for the port: architecture, setup, serving,
verification, and measured performance. The `docs/` recipe and tutorial and the
model-package README are thin pointers to this file.

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
| **Performance** | Batched prefill packing (`VLLM_NEURON_POOLING_PACK`) | ✅ default on |
| | On-device pooling gather (`VLLM_NEURON_POOLING_ONDEVICE_GATHER`) | ✅ default on |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (testing) | ✅ |

**Tensor parallelism.** TP=1, TP=2, and TP=4 are all validated. TP=2 scales
super-linearly for throughput (2× compute plus the bf16 weights split across two
cores doubles per-core HBM bandwidth); TP=4 maximizes absolute throughput and
minimizes latency. At **TP=1** set `VLLM_NEURON_FORCE_LNC1=1` (a codegen
workaround for the all-PyTorch graph; it does not change HBM usage). **Drop
`VLLM_NEURON_FORCE_LNC1` at TP≥2** — otherwise the compiled collectives fail to
load.

**Prefill bucket sizes must be greater than 128** (`num_batched_tokens_buckets`).

## Setup

### Native integration (nothing to apply)

The model is **natively integrated** into this repository — you do not need to
apply anything to use it:

- Model package: [`vllm_neuron/model/llama_bidirec/`](../vllm_neuron/model/llama_bidirec/)
  (architecture `LlamaBidirectionalModel`, `model_type` `llama_bidirec`).
- Registration and the shared-file touch-points (`model/registry.py`,
  `vllm_neuron/__init__.py`, the scheduler / worker / runner pooling branches, the
  non-causal attention fallback, and the LNC1 codegen flag) are committed directly
  on this branch. The bundled `integration.patch` reproduces the shared-file edits
  for an out-of-tree install and is not needed here.

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

At **TP=1** also set the LNC1 codegen workaround (drop it at TP≥2):

```bash
export VLLM_NEURON_FORCE_LNC1=1   # TP=1 ONLY; must be unset for TP>=2
```

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
**must be greater than 128**.

```bash
vllm serve /path/to/llama-embed-nemotron-8b \
    --runner pooling \
    --tensor-parallel-size 2 \
    --max-model-len 512 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [256, 512]}}'
```

> At `--tensor-parallel-size 1`, export `VLLM_NEURON_FORCE_LNC1=1` before
> launching. For TP=2 or TP=4, make sure `VLLM_NEURON_FORCE_LNC1` is **unset**.

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
    tensor_parallel_size=2,
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

**Configuration notes.** The pooling path adds two throughput optimizations, both
default ON, gated by environment variables:

- **`VLLM_NEURON_POOLING_PACK`** (default `1`): packs multiple embedding prefills
  into one bucket for higher short-prompt throughput. Set `0` for one prefill per
  batch.
- **`VLLM_NEURON_POOLING_ONDEVICE_GATHER`** (default `1`): pools MEAN + L2 on
  device and returns the per-sequence embedding directly, avoiding a per-step host
  copy of the full hidden-state buffer. Set `0` to fall back to host-side pooling.

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
# venv (REF_VENV), then compares against the on-device 2.31 result.
REF_VENV=$HOME/tf4x_venv python3 test/equivalence/ondevice_equivalence.py
```

| Test | Result |
|---|---|
| CPU equivalence (full vLLM pooling path, tiny weights) | worst cos = 0.999962 |
| On-device equivalence (trn2 TP=1, real 8B) | worst cos = 0.999961 |
| Batched on-device (4 seqs / 1 prefill, vs solo) | cos = 1.000000 (no cross-sequence leak) |
| vLLM serving `/v1/embeddings` vs HF reference | worst cos = 0.999962 |
| Cross-prompt similarity (discriminative check) | ~0.28–0.30 (distinct prompts stay distinct) |

> **Reference build note.** On the Neuron 2.31 stack (transformers ≥ 5.13) the HF
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
512. Throughput is total token throughput (input tokens only — embedding has no
output tokens); latency is end-to-end per request.

| TP | input len | concurrency | req/s | tok/s | p50 (ms) | p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 4 | 18.3 | 2,339 | 216.8 | 232.2 |
| 1 | 256 | 2 | 14.0 | 3,583 | 141.2 | 156.9 |
| 1 | 500 | 2 | 6.9 | 3,427 | 290.6 | 295.1 |
| 2 | 128 | 4 | 66.7 | 8,532 | 58.1 | 71.8 |
| 2 | 256 | 4 | 35.3 | 9,049 | 111.2 | 128.7 |
| 2 | 500 | 2 | 17.8 | 8,912 | 110.7 | 116.9 |
| 4 | 128 | 4 | 55.4 | 7,087 | 69.1 | 87.1 |
| 4 | 256 | 4 | 44.5 | 11,381 | 86.9 | 104.9 |
| 4 | 500 | 2 | 22.6 | 11,278 | 86.9 | 92.1 |

Peak throughput per TP (256-token inputs, concurrency 4): **TP=1 ≈ 3,583 tok/s**,
**TP=2 ≈ 9,049 tok/s** (2.5× TP=1), **TP=4 ≈ 11,381 tok/s** (3.2× TP=1). Full
sweeps (TP=1/2/4 × input 128/256/500 × concurrency 1/2/4) are recorded in
`INTERNAL/bringup-benches/`.

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
├── integration.patch                  # The model package + shared-file edits as a
│                                       #   standalone patch, for applying the model on
│                                       #   top of a separately-installed vllm_neuron
│                                       #   (already committed in-tree here; kept for
│                                       #   reference / out-of-tree installs)
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
├── __init__.py    # Package exports (LlamaBidirectionalModel)
├── README.md      # Package-level module structure (points here)
├── config.py      # LlamaBidirectional config (pooling / temperature fields)
├── factory.py     # Model construction / weight-loading factory
└── model.py       # Bidirectional Llama backbone + mask-weighted mean-pool + L2-norm
                   #   → 4096-dim embedding; per-query key_bounds; no KV cache
```

**New — docs & example**

```text
docs/model-recipes/llama-embed-nemotron-8b.md           # Model recipe / card (pointer to this README)
docs/tutorials/tutorial-llama-embed-nemotron-8b.md      # Deployment tutorial (pointer to this README)
examples/vllm_neuron/models/llama_bidirec/run.py        # Offline embedding example (llm.embed(), TP=2 default)
```

**Modified — shared framework touch-points**

```text
vllm_neuron/model/registry.py                       # Register LlamaBidirectionalModel
vllm_neuron/__init__.py                             # Register the llama_bidirec HuggingFace AutoConfig
vllm_neuron/vllm/core/scheduler.py                  # Pooling multi-sequence prefill packing
                                                    #   (VLLM_NEURON_POOLING_PACK, default on)
vllm_neuron/vllm/worker/neuron_model_runner.py      # Pooling runner mode + dense multi-seq prefill layout
vllm_neuron/vllm/worker/neuron_worker.py            # Skip decode graph extraction + warmup for
                                                    #   prefill-only pooling models
vllm_neuron/functional/attention/attention_cte.py   # Route non-causal (bidirectional) attention through
                                                    #   the PyTorch fallback (nkilib non-causal kernel
                                                    #   corrupts the heap)
vllm_neuron/compile/backend.py                      # Opt-in VLLM_NEURON_FORCE_LNC1 single-core codegen
                                                    #   for all-PyTorch graphs
docs/model-recipes/index.md                         # Link the model recipe
docs/tutorials/index.md                             # Link the tutorial
```
</content>
