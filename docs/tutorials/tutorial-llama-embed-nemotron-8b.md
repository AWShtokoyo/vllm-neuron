# Tutorial: Deploy llama-embed-nemotron-8b with vLLM Neuron

<!-- meta: description: End-to-end tutorial for deploying nvidia/llama-embed-nemotron-8b
with vLLM on Neuron, covering environment setup, model download, online embedding
serving, and offline embedding inference for the 8B bidirectional Llama embedding
(pooling) model on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, llama-embed-nemotron, embedding, pooling,
sentence embedding, bidirectional Llama, retrieval, tutorial, Trn2, Trainium -->
<!-- meta: date_updated: 2026-07-22 -->
<!-- Content type: procedural-tutorial -->

This tutorial walks through deploying
[nvidia/llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b)
on a `trn2` instance using vLLM-Neuron. It covers environment setup, model
download, online embedding serving, and offline embedding inference.

This is a **pooling / embedding** model: it is prefill-only (no decode) and is
served with `--runner pooling`, returning a 4096-dimensional L2-normalized
embedding per input.

**Prerequisites:**

- A `trn2` instance with Neuron SDK `2.31.0` or later. See
  [setup guide](../getting-started/setup-guide.md).
- vLLM Neuron plugin `0.21.0` or above installed.
- Python 3.10+

## Step 1: Environment Setup

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

## Step 2: Download the Model

```bash
huggingface-cli download \
    nvidia/llama-embed-nemotron-8b \
    --local-dir /path/to/llama-embed-nemotron-8b
```

> **Tip:** On a `trn2` cluster, download to a shared or local-NVMe filesystem
> instead of your home directory to avoid NFS write issues.

## Step 3: Online Serving

Start a vLLM OpenAI-compatible embedding server. Embedding models are prefill-only,
so there are no decode/sampling flags; `num_batched_tokens_buckets` controls the
compiled prefill shapes and its entries **must be greater than 128**.

```bash
vllm serve /path/to/llama-embed-nemotron-8b \
    --runner pooling \
    --tensor-parallel-size 2 \
    --max-model-len 512 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --additional-config '{
        "neuron_config": {
            "num_batched_tokens_buckets": [256, 512]
        }
    }'
```

> At `--tensor-parallel-size 1`, export `VLLM_NEURON_FORCE_LNC1=1` before launching.
> For TP=2 or TP=4, make sure `VLLM_NEURON_FORCE_LNC1` is **unset**.

Once the server is up, request embeddings from the OpenAI-compatible
`/v1/embeddings` endpoint:

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

## Step 4: Offline Inference

vLLM-Neuron compiles the model on the first run and caches the artifacts under
`VLLM_CACHE_ROOT`. Subsequent runs skip recompilation and load from cache.

The pooling model exposes the `llm.embed()` API. Set `max_num_seqs > 1` to let
multiple sequences pack into a single prefill for higher short-prompt throughput.

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
    additional_config={
        "neuron_config": {
            "num_batched_tokens_buckets": [256, 512],
        }
    },
)

prompts = [
    "The capital of France is Paris.",
    "Trainium is an AWS machine learning accelerator.",
    "Sentence embeddings power semantic search and retrieval.",
]
outputs = llm.embed(prompts)
for out in outputs:
    emb = out.outputs.embedding
    print(len(emb))                       # 4096, L2-normalized
```

### Configuration notes

The pooling path adds two throughput optimizations, **both default ON**, gated by
environment variables:

- **`VLLM_NEURON_POOLING_PACK`** (default `1`): packs multiple embedding prefills
  into one bucket for higher short-prompt throughput. Set `0` for one prefill per
  batch.
- **`VLLM_NEURON_POOLING_ONDEVICE_GATHER`** (default `1`): pools MEAN + L2 on device
  and returns the per-sequence embedding directly, avoiding a per-step host copy of
  the full hidden-state buffer — the higher-performance path, especially at short
  prompt / high concurrency. Set `0` to fall back to host-side pooling.

**Bucket sizes** (`num_batched_tokens_buckets`) control the discrete padded prefill
shapes compiled into each NEFF. Each bucket adds compile time; start with one or two
and add more as your input-length distribution requires. All entries must be
greater than 128.

## Conclusion

You have successfully deployed llama-embed-nemotron-8b on a `trn2` instance. The
model produces 4096-dimensional L2-normalized sentence embeddings via both the
offline `llm.embed()` API and the OpenAI-compatible `/v1/embeddings` serving
endpoint. For accuracy validation results, see the
[model card](../model-recipes/llama-embed-nemotron-8b.md).
</content>
