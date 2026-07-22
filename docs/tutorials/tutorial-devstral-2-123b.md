# Tutorial: Deploy Devstral-2-123B with vLLM Neuron

<!-- meta: description: End-to-end tutorial for deploying Devstral-2-123B-Instruct-2512
(Ministral3 dense FP8 GQA decoder) with vLLM on Neuron, covering environment
setup, model download, online serving, offline inference, and the optional
FP8-native kernel path on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Devstral, Devstral-2-123B, Ministral3, dense,
FP8, per-tensor static FP8, YaRN, tutorial, LLM serving, Trn2, Trainium -->
<!-- meta: date_updated: 2026-07-22 -->
<!-- Content type: procedural-tutorial -->

This tutorial walks through deploying
[Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512)
on a `trn2.48xlarge` instance using vLLM-Neuron. It covers environment setup,
model download, online serving, offline inference, and the optional FP8-native
path. For supported checkpoints, feature status, and accuracy results, see the
[model recipe](../model-recipes/devstral-2-123b.md).

**Prerequisites:**

- A `trn2.48xlarge` instance (64 logical NeuronCores at LNC2 / 16 Trainium2 chips)
  with Neuron SDK `2.31.0` or later. See [setup guide](../getting-started/setup-guide.md).
- vLLM Neuron plugin `0.21.0` or above installed.
- Python 3.10+ (3.12 used for validation).
- Access to the gated `mistralai/Devstral-2-123B-Instruct-2512` repository
  (`hf auth login` with a token that has access).

## Step 1: Environment Setup

Verify Neuron devices are visible:

```bash
neuron-ls
# Expected: 16 Trainium2 chips (64 NeuronCores at LNC2) on trn2.48xlarge
```

Set environment variables before running any inference script:

```bash
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2       # target Trainium2
export NEURON_SKIP_EFA_AFFINITY=1                 # skip EFA NUMA-affinity probe
export NKI_COMPILE_CACHE_URL=$HOME/nki_cache       # NKI kernel cache
export VLLM_CACHE_ROOT=/opt/nvme/vllm_cache        # vLLM/NEFF cache (local NVMe)

# Extend timeouts for large-model compilation (123B compiles for several minutes)
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_NEURON_COMPILATION_TIMEOUT=1200

# Required if your home directory is on NFS
export NEURON_CC_FLAGS="--temp-dir=/tmp/neuroncc_tmp"
mkdir -p /tmp/neuroncc_tmp
```

> **Do not launch `vllm serve` (or an offline script) with your shell's current
> directory inside the installed `vllm_neuron` package.** That package contains a
> `vllm/` subpackage which shadows the real `vllm` in vLLM's model-inspection
> subprocess (`python -m vllm.model_executor.models.registry`, launched from the
> cwd), failing with `Model architectures ['Ministral3ForCausalLM'] failed to be
> inspected` on a cold model-info cache. Launch from any neutral directory.

## Step 2: Download the Model

The checkpoint repository ships two copies (HF safetensors `model-*` and
Mistral-native `consolidated-*`, ~256 GB total). Serving uses the HF format, so
exclude the native copy to halve the download:

```bash
hf download \
    mistralai/Devstral-2-123B-Instruct-2512 \
    --exclude "consolidated-*" \
    --local-dir /path/to/Devstral-2-123B-Instruct-2512
```

> **Tip:** On a `trn2` instance, download to a large local disk (e.g. `/opt/nvme`)
> or a shared filesystem rather than your home directory to avoid NFS write issues.
> Note that instance-store volumes such as `/opt/nvme` are wiped on stop/terminate.

## Step 3: Online Serving

Start a vLLM OpenAI-compatible server. The recommended throughput layout is one
`TP=8` replica (2 chips); `TP=32` (half the box) is the low-latency point. TP must
divide Q=96, so `TP=8`, `TP=16`, and `TP=32` are valid — `TP=64` is not.

```bash
vllm serve /path/to/Devstral-2-123B-Instruct-2512 \
    --served-model-name Devstral-2-123B-Instruct-2512 \
    --config-format hf \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 2048 \
    --no-enable-prefix-caching \
    --hf-overrides '{"quantization_config": {}}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "bf16",
            "num_batched_tokens_buckets": [2048],
            "num_seqs_buckets": [128],
            "on_device_sampling_config": {"all_greedy": true}
        }
    }'
```

The required flags are specific to this model:

| Flag | Reason |
|---|---|
| `--config-format hf` | repo also ships Mistral-native files; `auto` would force the built-in `MistralForCausalLM` |
| `--hf-overrides '{"quantization_config": {}}'` | vLLM rejects the `fp8` quant method; the model's loader auto-detects FP8 from the checkpoint |
| `--no-enable-prefix-caching` | the stack does not support APC |
| `--max-num-batched-tokens 2048` | must equal the prefill bucket `[2048]` (single-shot prefill) |

Once the server is up, send requests using the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="Devstral-2-123B-Instruct-2512",
    messages=[{"role": "user", "content": "Write a Python function that returns the capital of Japan."}],
    max_tokens=128,
    temperature=0.0,
)
print(response.choices[0].message.content)
```

## Step 4: Offline Inference

vLLM-Neuron compiles the model on the first run and caches the artifacts under
`$VLLM_CACHE_ROOT`. Subsequent runs skip recompilation and load from cache.

```python
import vllm_neuron  # registers ministral3 + platform + AutoConfig
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Devstral-2-123B-Instruct-2512",
    tensor_parallel_size=8,
    max_model_len=2048,
    max_num_seqs=128,
    max_num_batched_tokens=2048,          # must equal the bucket [2048]
    config_format="hf",                   # avoid MistralForCausalLM override
    load_format="safetensors",
    enable_prefix_caching=False,          # stack has no APC
    hf_overrides={"quantization_config": {}},
    additional_config={"neuron_config": {
        "quantization": "bf16",
        "num_batched_tokens_buckets": [2048],
        "num_seqs_buckets": [128],
    }},
)

sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
outputs = llm.generate(["The capital of France is "], sampling_params)
print(outputs[0].outputs[0].text)
```

### Bucket sizes

`num_batched_tokens_buckets` controls the discrete padded prefill shapes compiled
into each NEFF. Keep the config single-shot by making the largest bucket equal
`max_model_len`. Each bucket adds compile time; start with one (`[2048]`) and add
smaller buckets only if you serve short prompts (see **Multi-bucket prefill**
below).

## Step 5: FP8-native (optional)

The checkpoint is per-tensor static FP8. The FP8-native path runs the dense
projections as FP8×FP8 static matmuls instead of dequantizing to BF16 — the
GPU-equivalent deployment, with up to **−50% weight HBM** and **+19% decode
throughput**. It requires one kernel patch on top of the framework install.

**1. Apply the nkilib kernel patch.** `nkilib` ships *inside* `neuronx-cc` (there
is nothing extra to `pip install`); the patch edits the tree that `import nkilib`
resolves to:

```bash
NKILIB="$(python3 -c 'import os,nkilib; print(os.path.dirname(nkilib.__file__))' | tail -1)"
NKROOT="$(dirname "$NKILIB")"
( cd "$NKROOT" && git apply --check -p1 /path/to/integration_nkilib.patch )   # verify
( cd "$NKROOT" && git apply        -p1 /path/to/integration_nkilib.patch )    # apply
# Confirm it applied:
grep -c NKILIB_MLP_BF16_XPOSE_SRC "$NKILIB/core/mlp/mlp_cte/mlp_cte_constants.py"   # >0
```

`integration_nkilib.patch` is distributed with the model bundle under
`Devstral-2-123B-Instruct-2512/`. It carries the committed qkv SBUF-budget fix and
the gen3 MLP fixes (`dma_transpose`, 32-byte alignment, activation pre-scale). The
default BF16-dequant path (Steps 3–4) does **not** need it.

**2. Enable via `neuron_config.quantization`.** Everything else in the serve /
offline command is identical:

```jsonc
"neuron_config": {
    "quantization": "fp8:qkv,o_proj,mlp",   // or "fp8:qkv,o_proj" (BF16-faithful)
    "num_batched_tokens_buckets": [2048],
    "num_seqs_buckets": [128]
}
```

Full FP8 (`…,mlp`) additionally needs `NKILIB_MLP_BF16_XPOSE_SRC=1` exported in the
environment (enables the gen3 MLP DMA-transpose path). `fp8:qkv,o_proj` is
BF16-token-faithful and does not; see the
[model recipe](../model-recipes/devstral-2-123b.md) for the accuracy tradeoff of
full FP8.

To add an FP8 KV cache, pass `--kv-cache-dtype fp8_e4m3` (runs at scale=1.0; only
valid when `bucket == max_model_len`).

## Whole-box throughput (TP=8 × DP=8)

A single `TP=8` replica is pure configuration. To pack all 16 chips as 8 replicas
for maximum throughput, set `data_parallel_size=8`. This requires two launcher
settings, both because vLLM 0.21 spawns fresh subprocesses on Neuron:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=fork    # DP coordinator child inherits the
                                            # initialized parent instead of re-running
                                            # the ~56 s plugin import (30 s ZMQ timeout)
export NEURON_VISIBLE_DEVICES=$(seq -s, 0 63)   # enumerate ALL 64 logical cores

vllm serve /path/to/Devstral-2-123B-Instruct-2512 \
    ... \
    --tensor-parallel-size 8 \
    --data-parallel-size 8 \
    --api-server-count 1        # single front-end; avoids the multi-API-server DP coordinator path
```

The module-scope HF config needed for pickling across engine-core subprocesses is
already handled by the model integration. The bundle's
`test/integration/perf_tp8_dp8.sh` reproduces this end to end.

### Multi-bucket prefill (optional)

For short-prompt / high-fan-out serving, add smaller prefill buckets so the
scheduler avoids padding every prompt to 2048. Keep the list ascending with the
last element equal to `max_model_len`, so every bucket stays single-shot:

```jsonc
"neuron_config": {
    "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048],
    ...
}
```

On a whole-box DP=8 short-input burst (`in=128`) this cuts median TTFT ~4.4× and
raises total throughput ~2.36×; at `in=1024` it is effectively neutral (the
padding it removes is small). Quality is unchanged — it only selects which prefill
graph runs, not the FP8 numerics. Multi-bucket cold-compiles one graph per bucket,
so raise `VLLM_ENGINE_READY_TIMEOUT_S` (e.g. `3600`) to avoid tripping vLLM's
readiness timeout mid-compile.

## Conclusion

You have deployed Devstral-2-123B-Instruct-2512 on a `trn2.48xlarge`, in both the
default BF16-dequant path and the optional FP8-native path, via the offline `LLM`
API and the OpenAI-compatible online serving endpoint. For feature status,
accuracy results, and performance figures, see the
[model card](../model-recipes/devstral-2-123b.md).
