# Contributed Model: Devstral-2-123B-Instruct-2512 (Ministral3)

## Introduction

[Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512)
is a 123B dense causal language model from Mistral AI (model_type `ministral3`,
architecture `Ministral3ForCausalLM`), instruction-tuned for coding and agentic
tasks. It is a dense GQA Transformer decoder — closely related to Llama 3, with
three differences: **YaRN** RoPE scaling, a **per-tensor static FP8 (E4M3)**
checkpoint, and untied embeddings.

This is the vllm-neuron implementation of that model on AWS Trainium2 (`trn2`).
Two serving paths are supported: the default **BF16-dequant** path (FP8 weights
dequantized at load, nothing extra to install) and the opt-in **FP8-native** path
(FP8×FP8 static matmuls — the same quantization scheme the GPU reference
deployment runs).

**Compatible model checkpoints:**

| Model | Precision | HuggingFace |
|-------|-----------|-------------|
| Devstral-2-123B-Instruct-2512 | per-tensor static FP8 (E4M3) | [`mistralai/Devstral-2-123B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512) |

> The checkpoint repository ships **both** the HuggingFace safetensors format
> (`model-*`) and the Mistral-native format (`consolidated-*`). Serving on Neuron
> uses the HF format — pass `--config-format hf` (and, for the offline `LLM` API,
> `load_format="safetensors"`) so vLLM does not fall back to its built-in
> `MistralForCausalLM`. When downloading, `--exclude "consolidated-*"` halves the
> transfer.

## Verification scope

Validated on `trn2.48xlarge` (Trainium2, 64 logical NeuronCores at LNC2 / 16
chips) on the Neuron 2.31 stack (vLLM 0.21 / vllm-neuron 0.21.0.1.0.0 /
neuronx-cc 2.26 / nki 0.5.0 / torch 2.11):

| Configuration | Status |
|---|---|
| TP=32 (half box), BF16-dequant — compile + generate | Verified on device |
| TP=32 (half box), FP8-native `qkv,o_proj` and full `qkv,o_proj,mlp` | Verified on device |
| TP=8 × DP=1 (one replica, 2 chips), full FP8 + FP8 KV cache | Verified on device |
| TP=8 × DP=8 (whole box, 8 replicas), full FP8 + FP8 KV cache | Verified on device |
| Multi-bucket prefill (opt-in), DP=8 short-input burst | Verified on device |

Accuracy was verified by a **same-build BF16-vs-FP8 A/B on device** (first-token
agreement plus the checkpoint's own HuggingFace model-card "Tests" prompts) — see
[Verification](#verification). The repository's generic 3-way logit-validation
scripts have **not** been run for this model; the procedure for doing so is
documented there. Every number in
[Measured performance](#measured-performance) is this port's own measurement.

> **Full FP8 is corrupted at TP=16.** The 16-way intermediate shard (per-rank
> `intermediate_size` = 28672/16 = 1792) produces corrupted output. Use **TP=8**
> or **TP=32** for full FP8.

## Model Overview

| | |
|---|---|
| HuggingFace ID | `mistralai/Devstral-2-123B-Instruct-2512` |
| Architecture | `Ministral3ForCausalLM` (model_type `ministral3`) |
| Task | Text generation (`--runner generate`), prefill + decode |
| Layers / hidden size | 88 / 12288 |
| Attention | Dense GQA (Q=96, KV=8, head_dim 128) |
| Context length | 262,144 (`max_position_embeddings`) |
| Vocabulary | 131,072 tokens, untied word embeddings |
| Tensor type | per-tensor static FP8 (E4M3) checkpoint; BF16 or FP8-native compute |

## Model Architecture

Dense GQA decoder. The values below come from the checkpoint config as translated
by [`vllm_neuron/model/ministral3/config.py`](../vllm_neuron/model/ministral3/config.py).

| Field | Value |
|---|---|
| hidden_size | 12288 |
| num_hidden_layers | 88 |
| num_attention_heads (Q) | 96 |
| num_key_value_heads (KV) | 8 (GQA) |
| head_dim | 128 |
| intermediate_size | 28672 |
| vocab_size | 131,072 |
| max_position_embeddings | 262,144 (256k) |
| RoPE | YaRN (θ=1e6, factor=64, orig_max=4096, β_fast=4, β_slow=1, mscale=1) |
| activation / norm | SiLU (SwiGLU) / RMSNorm (eps 1e-5), pre-attention + pre-MLP |
| attention bias / qk_norm | none |
| tie_word_embeddings | false (separate `lm_head`) |

**Port-specific notes**

- **The tensor-parallel degree must divide Q=96.** `TP=32` (half a
  `trn2.48xlarge`), `TP=16`, and `TP=8` are arithmetically valid; `TP=64` is
  **not** (96 is not divisible by 64). For full FP8, TP=16 additionally corrupts
  output — use TP=8 or TP=32.
- **YaRN parity.** The port reproduces the `transformers`
  `_compute_yarn_parameters` inverse frequencies exactly (maxdiff 0.0), including
  the `attention_factor` (mscale) of 1.4158883.
- **FP8 detection is checkpoint-driven.** The weight loaders detect FP8 from the
  number of slices they receive and the presence of `*.weight_scale_inv`, not from
  the HF `quantization_config` — which matters because that config has to be
  emptied for vLLM to accept the model (see
  [Required flags](#required-flags)).
- **E4M3 table difference.** Trainium2 decodes `float8_e4m3fn` bytes under its
  native table, whose exponent field `1111` is reserved, so the finite range caps
  at ±240 instead of OCP's ±448. The checkpoint does contain a handful of codes
  above 240, so the FP8-native loaders saturate those raw bytes onto the ±240 grid
  (exact for every other code; the per-tensor scale is untouched).

## Feature status

| Category | Feature | Status | Notes |
|---|---|---|---|
| **Quantization** | BF16 (FP8 weights dequantized at load) | ✅ | Default; nothing extra to install |
| | FP8-native `fp8:qkv,o_proj` | ✅ | BF16-token-faithful; needs the nkilib patch |
| | FP8-native `fp8:qkv,o_proj,mlp` (full FP8) | ✅ | −50% weight HBM; needs the nkilib patch + `NKILIB_MLP_BF16_XPOSE_SRC=1` |
| | FP8 KV cache (`--kv-cache-dtype fp8_e4m3`) | ✅ | Runs at scale 1.0; requires `bucket == max_model_len` |
| **Parallelism** | Tensor parallelism (TP) | ✅ | Must divide Q=96 |
| | Data parallelism (DP, whole-box throughput) | ✅ | Needs `VLLM_WORKER_MULTIPROC_METHOD=fork` |
| | Pipeline / context parallelism | ❌ | |
| **Performance** | Single-shot prefill (`bucket == max_model_len`) | ✅ | |
| | Multi-bucket prefill (opt-in, short-input TTFT) | ✅ | Keep the last bucket equal to `max_model_len` |
| | On-device sampling (greedy, top-k, top-p) | ✅ | |
| | Prefix caching (APC) | ❌ | Pass `--no-enable-prefix-caching` |
| | Chunked prefill (mixed batching) | ❌ | |
| **Compilation** | torch.compile (XLA backend) | ✅ | |
| | CPU mode (testing) | ✅ | `VLLM_NEURON_CPU_MODE=1` |

See the [features guide](../docs/guides/features-guide.md) for framework-level
configuration details.

## Setup

### Native integration (nothing to apply for the default path)

The model is **natively integrated** into this repository — check out this branch
and it serves Devstral out of the box on the default BF16-dequant path. There is
no in-repo patch to apply.

- **Model package:** [`vllm_neuron/model/ministral3/`](../vllm_neuron/model/ministral3/)
  (per-file breakdown: [module structure](../vllm_neuron/model/ministral3/README.md)).
- **Registration and framework touch-points** are committed directly on this
  branch — see
  [Paths this port touches](#paths-this-port-touches-across-the-repository).
- **One out-of-repo patch:** the optional FP8-native path needs
  [`integration_nkilib.patch`](integration_nkilib.patch) applied to the installed
  `nkilib`, which is a *different package* (bundled inside `neuronx-cc`) and
  therefore cannot be carried in this tree. Install it with
  [Step 3](#step-3-fp8-native-only-install-integration_nkilibpatch); skipping it
  leaves the default BF16-dequant path fully functional.

**Prerequisites:**

- A `trn2.48xlarge` instance (64 logical NeuronCores at LNC2 / 16 Trainium2 chips)
  with Neuron SDK `2.31` or later. See the
  [setup guide](../docs/getting-started/setup-guide.md).
- vLLM Neuron plugin `0.21.0.1.0.0` or above.
- Python 3.10+ (3.12 used for validation).
- Access to the gated `mistralai/Devstral-2-123B-Instruct-2512` repository
  (`hf auth login` with a token that has access).

### Step 1: Environment setup

Verify the Neuron devices are visible:

```bash
neuron-ls
# Expected: 16 Trainium2 chips (64 NeuronCores at LNC2) on trn2.48xlarge
```

Export these before any compile / inference run:

```bash
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2       # target Trainium2
export NEURON_SKIP_EFA_AFFINITY=1                 # skip the EFA NUMA-affinity probe
export NKI_COMPILE_CACHE_URL=$HOME/nki_cache      # NKI kernel cache
export VLLM_CACHE_ROOT=/opt/nvme/vllm_cache       # vLLM/NEFF cache (local NVMe)

# Extend the timeouts — a 123B model compiles for several minutes
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_NEURON_COMPILATION_TIMEOUT=1200

# Required if your home directory is on NFS
export NEURON_CC_FLAGS="--temp-dir=/tmp/neuroncc_tmp"; mkdir -p /tmp/neuroncc_tmp
```

> **Do not launch `vllm serve` (or an offline script) with your shell's current
> directory inside the installed `vllm_neuron` package.** That package contains a
> `vllm/` subpackage which shadows the real `vllm` in vLLM's model-inspection
> subprocess (launched from the cwd), failing with `Model architectures
> ['Ministral3ForCausalLM'] failed to be inspected` on a cold model-info cache.
> Launch from any neutral directory.

### Step 2: Download the model

The checkpoint repository ships two copies (HF safetensors `model-*` and
Mistral-native `consolidated-*`, ~256 GB together). Serving uses the HF format, so
excluding the native copy halves the download:

```bash
hf download \
    mistralai/Devstral-2-123B-Instruct-2512 \
    --exclude "consolidated-*" \
    --local-dir /path/to/Devstral-2-123B-Instruct-2512
```

> **Tip:** On a `trn2` instance, download to a large local disk (e.g.
> `/opt/nvme`) or a shared filesystem rather than your home directory to avoid
> NFS write issues. Instance-store volumes such as `/opt/nvme` are wiped on
> stop/terminate.

### Step 3 (FP8-native only): install `integration_nkilib.patch`

**Skip this step entirely if you serve on the default BF16-dequant path** — it
needs nothing beyond a normal build of this branch. The step is required only for
the FP8-native modes (`fp8:qkv,o_proj` and `fp8:qkv,o_proj,mlp`), because they run
kernels that this patch fixes.

[`integration_nkilib.patch`](integration_nkilib.patch) edits `nkilib`, the NKI
kernel library. `nkilib` ships **inside `neuronx-cc`** — there is nothing extra to
`pip install`, and because it is a different package from `vllm-neuron` the fix
cannot be carried in this repository. Apply it to the `nkilib` tree that
`import nkilib` actually resolves to:

```bash
# 0. From this bundle directory (the one holding integration_nkilib.patch)
cd /path/to/vllm-neuron/Devstral-2-123B-Instruct-2512

# 1. Locate the nkilib that `import nkilib` RESOLVES TO (not a guessed path).
#    Patch paths start with nkilib/... , so apply with -p1 from its parent.
NKILIB="$(python3 -c 'import os,nkilib; print(os.path.dirname(nkilib.__file__))' 2>/dev/null | tail -1)"
NKROOT="$(dirname "$NKILIB")"                   # parent of the nkilib/ package dir
echo "nkilib at: $NKILIB"
[ -d "$NKILIB" ] || { echo "ERROR: nkilib not importable — is neuronx-cc installed in this venv?"; }

# 2. Dry-run, then apply. Use an ABSOLUTE patch path so it resolves in the subshell.
NKPATCH="$PWD/integration_nkilib.patch"
( cd "$NKROOT" && git apply --check -p1 "$NKPATCH" )   # verify it applies clean
( cd "$NKROOT" && git apply        -p1 "$NKPATCH" )    # apply
# No git available? use patch(1) instead:
#   ( cd "$NKROOT" && patch -p1 --fuzz=3 < "$NKPATCH" )

# 3. Verify it landed (non-zero count = applied)
grep -c NKILIB_MLP_BF16_XPOSE_SRC "$NKILIB/core/mlp/mlp_cte/mlp_cte_constants.py"

# To undo:
#   ( cd "$NKROOT" && git apply -R -p1 "$NKPATCH" )
```

Always run `git apply --check` (or `patch --dry-run`) first: it reports up-front
that the `nkilib` tree has diverged instead of half-applying the patch. Reinstalling
or upgrading `neuronx-cc` replaces `nkilib` and therefore reverts the patch — re-run
this step after any `neuronx-cc` change.

What the patch contains:

| Fix | Why it is needed |
|---|---|
| qkv SBUF-budget fix (committed upstream as nki-library `356f16a`) | Lets the FP8 qkv kernel fit prefill bucket 2048; without it FP8-native is capped at bucket ≤1024 |
| MLP `dma_transpose` source transpose | The gen3 MLP FP8 path needs the DMA-based transpose; gated by `NKILIB_MLP_BF16_XPOSE_SRC=1` |
| 32-byte-aligned transpose buffers | Alignment requirement of that transpose |
| Activation pre-scale by `gate_up_in_scale` | The scale was being skipped, which is what actually broke full-FP8 MLP output |

Then enable the mode itself in `neuron_config` — see
[FP8-native](#fp8-native-optional). Full FP8 (`fp8:qkv,o_proj,mlp`) additionally
needs `NKILIB_MLP_BF16_XPOSE_SRC=1` exported in the environment;
`fp8:qkv,o_proj` does not.

## Serving

### Online serving (OpenAI-compatible)

The throughput layout is one `TP=8` replica (2 chips); `TP=32` (half the box) is
the low-latency point.

```bash
vllm serve /path/to/Devstral-2-123B-Instruct-2512 \
    --served-model-name Devstral-2-123B-Instruct-2512 \
    --config-format hf \
    --load-format safetensors \
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

Once the server reports `Application startup complete.`:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="Devstral-2-123B-Instruct-2512",
    messages=[{"role": "user",
               "content": "Write a Python function that returns the capital of Japan."}],
    max_tokens=128,
    temperature=0.0,
)
print(response.choices[0].message.content)
```

#### Required flags

| Flag | Reason |
|---|---|
| `--config-format hf` | The repo also ships Mistral-native files; `auto` would force the built-in `MistralForCausalLM` |
| `--load-format safetensors` | Load the HF safetensors copy, not `consolidated-*` |
| `--hf-overrides '{"quantization_config": {}}'` | vLLM rejects the `fp8` quant method; the model's loaders auto-detect FP8 from the checkpoint tensors |
| `--no-enable-prefix-caching` | The stack does not support APC |
| `--max-num-batched-tokens 2048` | Must equal the largest prefill bucket (single-shot prefill) |

### Offline `LLM` API

The shipped example
[`examples/vllm_neuron/models/ministral3/run.py`](../examples/vllm_neuron/models/ministral3/run.py):

```bash
python examples/vllm_neuron/models/ministral3/run.py \
    --model-checkpoint /path/to/Devstral-2-123B-Instruct-2512 \
    --tensor-parallel-size 8
```

The equivalent direct `LLM` call:

```python
import vllm_neuron  # registers ministral3 + the platform + the HF AutoConfig
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Devstral-2-123B-Instruct-2512",
    tensor_parallel_size=8,
    max_model_len=2048,
    max_num_seqs=128,
    max_num_batched_tokens=2048,          # must equal the prefill bucket
    config_format="hf",                   # avoid the MistralForCausalLM override
    load_format="safetensors",
    enable_prefix_caching=False,          # the stack has no APC
    hf_overrides={"quantization_config": {}},
    additional_config={"neuron_config": {
        "quantization": "bf16",
        "num_batched_tokens_buckets": [2048],
        "num_seqs_buckets": [128],
    }},
)

outputs = llm.generate(["The capital of France is "],
                       SamplingParams(max_tokens=64, temperature=0.0))
print(outputs[0].outputs[0].text)
```

#### Bucket sizes

`num_batched_tokens_buckets` lists the discrete padded prefill shapes compiled
into the NEFFs. Keep the configuration single-shot by making the largest bucket
equal `max_model_len`. Each bucket adds compile time, so start with one
(`[2048]`) and add smaller buckets only if you serve short prompts (see
[Multi-bucket prefill](#multi-bucket-prefill-optional)).

### Quantization

The checkpoint is per-tensor static FP8. The compute path is selected with
`quantization` in `neuron_config`:

| Mode | Behavior | Weight HBM |
|---|---|---|
| `bf16` (default) | FP8 weights dequantized to BF16 at load (`weight × weight_scale_inv`) | baseline |
| `fp8:qkv,o_proj` | Attention projections stay FP8 (FP8×FP8 static matmul); MLP stays BF16 | −28.8 GB at TP=32 |
| `fp8:qkv,o_proj,mlp` (or `fp8`) | Full FP8: attention **and** MLP projections in FP8 | ≈ half |

`bf16` is the safe baseline and needs no kernel patch. `fp8:qkv,o_proj` is
BF16-token-faithful (first-token 30/30 on device) and removes the
attention-weight HBM at no measured quality cost. Full FP8 maximizes the HBM and
decode-throughput win and carries the per-tensor-static-FP8 accuracy tradeoff
described in [Verification](#verification).

### FP8-native (optional)

FP8-native runs the dense projections as FP8×FP8 static matmuls instead of
dequantizing to BF16.

**Prerequisite:** [`integration_nkilib.patch`](integration_nkilib.patch) must be
applied to the installed `nkilib` — see
[Step 3 (FP8-native only)](#step-3-fp8-native-only-install-integration_nkilibpatch)
for the apply, verify, and undo commands. Staying on `bf16` needs no patch.

Then set the mode; everything else in the serve / offline command is unchanged:

```jsonc
"neuron_config": {
    "quantization": "fp8:qkv,o_proj,mlp",   // or "fp8:qkv,o_proj" (BF16-faithful)
    "num_batched_tokens_buckets": [2048],
    "num_seqs_buckets": [128]
}
```

Full FP8 (`…,mlp`) additionally needs `NKILIB_MLP_BF16_XPOSE_SRC=1` exported in
the environment (it enables the MLP DMA-transpose path); `fp8:qkv,o_proj` does
not.

To add an FP8 KV cache, pass `--kv-cache-dtype fp8_e4m3`. It runs at scale 1.0
(the checkpoint ships no k/v scales) and is only valid when
`bucket == max_model_len` — a smaller bucket takes the segmented-prefill path,
which does not compile with FP8 KV.

### Whole-box throughput (TP=8 × DP=8)

A single `TP=8` replica is pure configuration. To pack all 16 chips as 8 replicas,
set `--data-parallel-size 8`, which needs two extra launcher settings because vLLM
0.21 spawns fresh subprocesses on Neuron:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=fork    # the DP coordinator child inherits the
                                            # initialized parent instead of re-running
                                            # the ~56 s plugin import (30 s ZMQ timeout)
export NEURON_VISIBLE_DEVICES=$(seq -s, 0 63)   # enumerate ALL 64 logical cores
export VLLM_ENGINE_READY_TIMEOUT_S=2400         # 8 replicas warm up serially

vllm serve /path/to/Devstral-2-123B-Instruct-2512 \
    ... \
    --tensor-parallel-size 8 \
    --data-parallel-size 8 \
    --api-server-count 1        # single front-end; avoids the multi-API-server DP path
```

vLLM 0.21 slices `NEURON_VISIBLE_DEVICES` per DP rank, so the base list must
enumerate all 64 logical cores rather than 16 device indices. The module-scope HF
config class needed to pickle the config across engine-core subprocesses is
already part of the model integration.

### Multi-bucket prefill (optional)

A single `[2048]` prefill bucket pads every short prompt to 2048. Adding smaller
buckets lets the scheduler pick the tightest fit, cutting TTFT on short-input
bursts. Keep the list ascending with the last element equal to `max_model_len`, so
every bucket stays single-shot (and the segmented-prefill path, which does not
compile with FP8 KV, is never taken):

```jsonc
"neuron_config": {
    "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048],
    ...
}
```

Quality is unchanged — this selects which prefill graph runs, not the numerics.
Multi-bucket cold-compiles one graph per bucket, so raise
`VLLM_ENGINE_READY_TIMEOUT_S` (e.g. `3600`) to avoid tripping vLLM's readiness
timeout mid-compile. Measured effect:
[Multi-bucket prefill](#multi-bucket-prefill-dp8-full-fp8--fp8-kv).

## Verification

Verification uses this repository's two model-agnostic tools.

**1. Accuracy — generic logit comparison.** The scripts in
[`examples/vllm_neuron/accuracy/`](../examples/vllm_neuron/accuracy/) compare
HuggingFace CPU goldens (FP32 → BF16) against vLLM-on-Neuron for any text causal
LM:

```bash
# 3-way logit validation (FP32 baseline → BF16 expected → Neuron target)
python examples/vllm_neuron/accuracy/run_logit_validation_offline.py \
    --model /path/to/Devstral-2-123B-Instruct-2512 --tp-size 8

# Same, against a running server over /v1/completions
python examples/vllm_neuron/accuracy/run_logit_validation_online.py \
    --model /path/to/Devstral-2-123B-Instruct-2512 --tp-size 8

# Per-module intermediate-tensor comparison (HF vs Neuron)
python examples/vllm_neuron/accuracy/compare_hf_vs_vllm_neuron.py \
    --model /path/to/Devstral-2-123B-Instruct-2512 --tp-size 8
```

Notes for this model:

- **These scripts have not been run for Devstral-2-123B.** A CPU HuggingFace
  golden at 123B needs a very large host-memory machine; size the comparison host
  accordingly, or start from a reduced-layer configuration.
- `compare_hf_vs_vllm_neuron_with_reconstruction.py` hardcodes a Llama module
  layout — check it against this model's module names before using it.
- `run_encoder_cache_analysis.py` is a vision-encoder tool and does not apply.

**What was verified instead (on device, same build, A/B):**

| Check | Result |
|---|---|
| YaRN RoPE `inv_freq` parity vs `transformers._compute_yarn_parameters` | maxdiff 0.0; `attention_factor` 1.4158883 |
| FP8 weight loaders (sharding / fused-QKV slicing / dequant) numeric check | Pass |
| BF16-dequant, real weights, TP=32 — compile + generate | Pass (128 NEFFs); prefill and decode coherent on counting / factual / story / code prompts |
| `fp8:qkv,o_proj` vs BF16 — first-token agreement | 30/30 (BF16-token-faithful) |
| Full FP8 (`qkv,o_proj,mlp`) — the checkpoint's HF model-card "Tests" prompts | Facts and code correct; occasional per-token deviation, rarely an early derail on long open-ended generation |
| Regression: `llama3` tiny model on the same build | Pass (unaffected) |
| Tiny CPU end-to-end (NKI simulator, TP=1 and TP=2) | Pass |

**On the full-FP8 deviation.** This is inherent to per-tensor static FP8, not a
Trainium or implementation defect. Offline emulation of the exact FP8 math with
the real checkpoint scales matches BF16 at cos ≈ 0.9999 per layer, but ~1%
per-layer error compounds across 88 layers and flips the greedy argmax on a
minority of tokens. The GPU reference deployment runs the *same* per-tensor static
FP8 checkpoint (`weight_block_size=null`, `qscheme_act="TENSOR"`), so a GPU FP8
deployment shows the same class of per-token deviation from BF16. Use
`fp8:qkv,o_proj` (or `bf16`) when byte-faithful output matters; use full FP8 where
the HBM and throughput win outweighs occasional token-level deviation.

**2. Benchmark — `vllm bench serve`.** The built-in, model-agnostic client, run
against a server started as in
[Online serving](#online-serving-openai-compatible):

```bash
vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --host localhost --port 8000 \
    --model Devstral-2-123B-Instruct-2512 \
    --tokenizer /path/to/Devstral-2-123B-Instruct-2512 \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 256 \
    --num-prompts 128 --max-concurrency 128 \
    --ignore-eos --temperature 0 --seed 0 \
    --save-result --result-filename devstral_bench.json
```

`--ignore-eos` fixes the output length so throughput and latency are comparable
across runs. Use `--dataset-name sharegpt` (which flows through the chat template)
when representative content matters.

## Measured performance

All figures below are this port's own measurements on `trn2.48xlarge`, Neuron 2.31
stack, `vllm bench serve` with the random dataset and greedy sampling.

### FP8-native vs BF16-dequant (TP=32, prefill bucket 2048)

| Metric | BF16-dequant | full FP8 (`qkv,o_proj,mlp`) | Δ |
|---|---|---|---|
| weight HBM | 243.6 GB | **121.8 GB** | **−50.0%** |
| peak device HBM | 525.9 GB | 397.6 GB | −24.4% |
| decode throughput | 34.72 tok/s/seq | **41.33 tok/s/seq** | **+19.0%** |
| TTFT (median) | 0.421 s | 0.469 s | +11.6% (prefill tax) |

Serving A/B on the same build (`max_model_len=4096`, `max_num_seqs=2`, in 1024 /
out 256, 20 requests at concurrency 2):

| Metric | BF16-dequant | full FP8 | Δ |
|---|---|---|---|
| benchmark duration | 81.84 s | 70.99 s | **−13%** |
| request throughput | 0.244 req/s | **0.282 req/s** | **+15%** |
| output token throughput | 62.56 tok/s | **72.12 tok/s** | **+15%** |
| total token throughput | 312.55 tok/s | **360.34 tok/s** | **+15%** |
| TTFT mean / P99 | 666 / 889 ms | 720 / 1096 ms | +8% |
| TPOT mean / P99 | 29.5 / 30.3 ms | **25.0 / 25.9 ms** | **−15%** |
| successful / total | 20 / 20 | 20 / 20 | — |

Full FP8 lifts decode throughput at a small prefill cost, consistent with the
halved weight-HBM bandwidth.

### Throughput layout (full FP8 + FP8 KV, in 1024 / out 256)

`max_model_len=2048`, `max_num_seqs=128`, prefill bucket `[2048]`:

| Config | chips | total tok/s | output tok/s | TPOT (mean) | successful / total |
|---|---|---|---|---|---|
| **TP=8 × DP=1** (one replica, 128 prompts @ conc 128) | 2 | **1,644.7** | 328.4 | 245 ms | 128 / 128 |
| **TP=8 × DP=8** (whole box, 512 prompts @ conc 1024) | 16 | **10,240.4** | 2,044.9 | 176 ms | 512 / 512 |

A `trn2.48xlarge` is 64 logical NeuronCores (16 chips × 4 at LNC2) and
`TP × DP == world_size == cores`, so `TP=8` is 8 cores = 2 chips (one replica; 8
replicas fill the box). Per-chip throughput is highest at `TP=8 × DP=1`
(≈822 total tok/s per chip vs ≈640 at DP=8): a dense model's DP replicas share no
collectives, so the DP=8 per-replica drop is host-side contention (CPU,
front-end, host↔device DMA) — replicas spread across nodes with host headroom do
better than a fully packed one.

> **TTFT in these two runs is inflated** (DP=1 mean ~37 s, DP=8 ~19 s) because the
> harness submits every prompt at once — this is a *throughput* regime, not
> steady-state latency. Trainium cannot mix prefill and decode in one step
> (chunked prefill is bs=1 only), so under a burst prefill and decode block each
> other.

### Multi-bucket prefill (DP=8, full FP8 + FP8 KV)

512 prompts at concurrency 1024, baseline `[2048]` vs
`[128, 256, 512, 1024, 2048]`:

| Workload | Metric | baseline | multi-bucket | Δ |
|---|---|---|---|---|
| in 128 / out 128 | median TTFT | 19,198 ms | **4,322 ms** | **4.44× faster** |
| | total token throughput | 2,573 tok/s | **6,061 tok/s** | **2.36×** |
| | mean TPOT | 249.7 ms | **136.2 ms** | 1.83× better |
| in 1024 / out 256 | median TTFT | 19,203 ms | 19,195 ms | 1.00× (neutral) |
| | total token throughput | 10,137 tok/s | 10,219 tok/s | 1.01× |
| | mean TPOT | 175.6 ms | 175.6 ms | 1.00× |

All four runs completed 512/512. The win scales with how much shorter the prompt
is than the single bucket: 16× padding at `in=128` is a large win, 2× padding at
`in=1024` vanishes under saturation. Enable it for short-prompt / high-fan-out
serving; for ~1k-token inputs the single `[2048]` bucket is already near-optimal
and multi-bucket only adds cold-compile time.

## Contents of this bundle

The model is integrated in-tree (committed directly on this branch), so this
top-level `Devstral-2-123B-Instruct-2512/` directory is supplementary: it carries
the source-of-truth README plus the one patch that cannot live in this tree
because it targets a different package. There is no `src/` copy here — that would
duplicate the in-tree package.

```text
Devstral-2-123B-Instruct-2512/
├── README.md                  # This file — the source of truth for the port
└── integration_nkilib.patch   # Kernel fixes for the EXTERNAL nkilib package
                               #   (bundled inside neuronx-cc) that enable FP8-native:
                               #   qkv SBUF budget + MLP dma_transpose / 32B align /
                               #   activation pre-scale — see FP8-native (optional)
```

The bundle ships no `test/` directory: verification uses the two generic tools
described in [Verification](#verification). The tiny CPU end-to-end test and the
bring-up benchmark scripts and result JSONs behind
[Measured performance](#measured-performance) are kept internally, outside this
repository.

### Paths this port touches across the repository

**New — model package** (`vllm_neuron/model/ministral3/`)

```text
vllm_neuron/model/ministral3/
├── __init__.py          # Package exports (Ministral3Config, Ministral3ForCausalLM,
│                        #   Ministral3Attention / MLP / RMSNorm / RotaryEmbedding)
├── README.md            # Module structure (points back at this README)
├── config.py            # Ministral3Config: HF → Neuron config translation (GQA head
│                        #   counts, YaRN RoPE parameters, untied embeddings, the
│                        #   quantization → per-family FP8 mapping)
├── factory.py           # Ministral3ForCausalLM factory: validates the config and
│                        #   instantiates the implementation
├── model.py             # Dense GQA decoder: attention, SwiGLU MLP, RMSNorm, YaRN
│                        #   rotary embedding; BF16-dequant and FP8-native
│                        #   (FP8×FP8 per-tensor static matmul) compute paths
└── weight_loaders.py    # Per-tensor static FP8 (E4M3) weight loaders: shard first,
                         #   then dequantize — or keep FP8 + scalar scale and saturate
                         #   to the Neuron ±240 grid for FP8-native; checkpoint format
                         #   auto-detected from the slice count
```

**New — docs & example**

```text
docs/model-recipes/devstral-2-123b.md            # Model recipe / card (pointer to this README)
docs/tutorials/tutorial-devstral-2-123b.md       # Deployment tutorial (pointer to this README)
examples/vllm_neuron/models/ministral3/run.py    # Offline generation example (TP=8 default)
```

**Modified — shared framework touch-points**

```text
vllm_neuron/model/registry.py    # Register ("Ministral3ForCausalLM", Ministral3ForCausalLM)
vllm_neuron/model/__init__.py    # Add ministral3 to the lazy-import allowlist
vllm_neuron/__init__.py          # _register_ministral3_hf_config(): AutoConfig.register
                                 #   ("ministral3", ...) with a module-scope config class so
                                 #   it pickles across engine-core subprocesses (DP > 1)
vllm_neuron/vllm/platform.py     # Pre-register Ministral3ForCausalLM with a pre-built
                                 #   _ModelInfo (is_text_generation_model=True) at plugin
                                 #   load, avoiding the model-inspection subprocess
docs/model-recipes/index.md      # Add the recipe grid card + toctree entry
docs/tutorials/index.md          # Add the tutorial grid card + toctree entry
```

The root `README.md` Contributed Models table is maintained on the release branch
and taken verbatim from there, so it is not modified on this branch.
