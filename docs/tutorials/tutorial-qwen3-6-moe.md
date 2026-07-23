# Tutorial: Deploy Qwen3.6-35B-A3B with vLLM Neuron

<!-- meta: description: End-to-end tutorial for deploying the text-only
Qwen3.6-35B-A3B hybrid Mixture-of-Experts model with vLLM on Neuron, covering
environment setup, model download, online serving, and offline inference on
Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3.6, Qwen3.6-35B-A3B, qwen3_5_moe, MoE,
GatedDeltaNet, hybrid, tutorial, LLM serving, Trn2, Trainium -->
<!-- meta: date_updated: 2026-07-23 -->
<!-- Content type: procedural-tutorial -->

This tutorial walks through deploying
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) on a
`trn2.48xlarge` instance using vLLM-Neuron. It covers environment setup, model
download, online serving, and offline inference. The model is served **text-only**
(the vision tower of the native multimodal checkpoint is skipped).

**Prerequisites:**

- A `trn2.48xlarge` instance (16 NeuronCores) with Neuron SDK `2.31.0` or later.
  See [setup guide](../getting-started/setup-guide.md).
- vLLM Neuron plugin `0.21.0` or above installed.
- Python 3.10+

## Step 1: Environment Setup

Verify Neuron devices are visible:

```bash
neuron-ls
# Expected: 16 NeuronCores listed for trn2.48xlarge
```

Set environment variables before running any inference script:

```bash
# Extend timeouts for large model compilation
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_NEURON_COMPILATION_TIMEOUT=1200

# Required if your home directory is on NFS
mkdir -p /tmp/neuroncc_tmp

# Required recipe flags for this model (default-off; all three are load-bearing)
export VLLM_GDN_SEQ_NKI=1          # bounded-graph GatedDeltaNet prefill (required for seq >= 512)
export VLLM_UNIFIED_KV_GATHER=1    # unified block-indexed KV gather for the full-attention layers
export VLLM_MOE_TKG_ROUTER_FP32=1  # fp32 decode router for correct top-8 expert selection
```

> **Cores:** for a tensor-parallel serve, select devices with
> `NEURON_VISIBLE_DEVICES` (e.g. `0-7`). Do **not** use
> `NEURON_RT_VISIBLE_CORES` for a multi-process TP serve — it errors with
> "cannot be used with multi-processing execution".

> **Compiler flags:** leave `NEURON_CC_FLAGS` unset and let the framework compose
> them. Setting it replaces (rather than appends to) the framework's flag string
> and drops flags the model requires. Relocate caches to a large volume with
> `VLLM_CACHE_ROOT` instead.

## Step 2: Download the Model

```bash
huggingface-cli download \
    Qwen/Qwen3.6-35B-A3B \
    --local-dir /path/to/Qwen3.6-35B-A3B
```

> **Tip:** On a `trn2` cluster, download to a large shared filesystem instead of
> your home directory to avoid NFS write issues and to keep the checkpoint off the
> root volume.

## Step 3: Online Serving

Start a vLLM OpenAI-compatible server. The `hf-overrides` and `limit-mm-per-prompt`
flags select the text-only decoder path and are mandatory:

```bash
vllm serve /path/to/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-model-len 1024 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 1 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --hf-overrides '{"architectures":["Qwen3_5MoeForCausalLM"]}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "bf16",
            "ep_degree": 8,
            "on_device_sampling_config": {"all_greedy": "true"},
            "kv_segment_size_buckets": [512],
            "num_batched_tokens_buckets": [512],
            "num_seqs_buckets": [1]
        }
    }'
```

> **Bucket sizing:** `kv_segment_size_buckets` and `num_batched_tokens_buckets`
> must be **≤ 512** so no 1024-extent prefill graph is traced. The GatedDeltaNet
> layers require a bounded prefill graph at sequence length ≥ 512; a longer
> monolithic graph trips a scatter/gather out-of-bound access. Each bucket adds
> compile time, and changing any bucket forces a full cold recompile — freeze one
> recipe rather than sweeping sizes.

Once the server is up, send requests using the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="Qwen3.6-35B-A3B",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=50,
)
print(response.choices[0].message.content)
```

For higher throughput, keep `--max-model-len 1024` (with `kv_segment_size_buckets:
[512]` as above) and serve with a larger batch — for example `--max-num-seqs 8
--num-seqs-buckets 8` — then drive load with `vllm bench serve`. Do not drop
`--max-model-len` to 512 for batch size > 1; see the
[model recipe](../model-recipes/qwen3-6-moe.md#performance) for the reason and for
measured throughput at batch sizes 1, 4, and 8.

## Step 4: Offline Inference

vLLM-Neuron compiles the model on the first run and caches the artifacts. Subsequent
runs skip recompilation and load from cache.

The following script runs text-only offline inference. It sets the required recipe
flags via `os.environ.setdefault`, so it works standalone:

```python
import os

os.environ.setdefault("VLLM_GDN_SEQ_NKI", "1")
os.environ.setdefault("VLLM_UNIFIED_KV_GATHER", "1")
os.environ.setdefault("VLLM_MOE_TKG_ROUTER_FP32", "1")
os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "1200")

from vllm import LLM, SamplingParams

MODEL_PATH = "/path/to/Qwen3.6-35B-A3B"

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=8,
    enable_expert_parallel=True,
    max_model_len=1024,
    max_num_batched_tokens=512,
    max_num_seqs=1,
    limit_mm_per_prompt={"image": 0, "video": 0},
    hf_overrides={"architectures": ["Qwen3_5MoeForCausalLM"]},
    additional_config={
        "neuron_config": {
            "quantization": "bf16",
            "ep_degree": 8,
            "on_device_sampling_config": {"all_greedy": "true"},
            "kv_segment_size_buckets": [512],
            "num_batched_tokens_buckets": [512],
            "num_seqs_buckets": [1],
        }
    },
)

sampling_params = SamplingParams(max_tokens=200, temperature=0.0)
outputs = llm.generate(["What is the capital of France?"], sampling_params)
print(outputs[0].outputs[0].text)
```

A ready-to-run version of this script lives at
`examples/vllm_neuron/models/qwen3_5_moe/run.py`.

## Conclusion

You have successfully deployed Qwen3.6-35B-A3B text-only on a `trn2.48xlarge`
instance via both the offline `LLM` API and the OpenAI-compatible online serving
endpoint. For accuracy and throughput results, see the
[model recipe](../model-recipes/qwen3-6-moe.md).
