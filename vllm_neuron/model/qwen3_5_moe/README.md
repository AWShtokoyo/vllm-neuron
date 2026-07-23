# qwen3_5_moe (Qwen3.6-35B-A3B, text-only MoE hybrid)

vllm-neuron model implementation for `Qwen/Qwen3.6-35B-A3B` (architecture alias
`Qwen3_5MoeForCausalLM`, `model_type: qwen3_5_moe`).

Qwen3.6-35B-A3B is a **hybrid Mixture-of-Experts** decoder: 40 layers in a 3:1
pattern → **30 GatedDeltaNet (linear-attention) + 10 full-attention** layers, each
routing every token through a 256-expert top-8 MoE (plus a shared expert). The
native checkpoint arch is `Qwen3_5MoeForConditionalGeneration` (multimodal/VLM);
**this port is TEXT-ONLY**.

## Text-only requirement

Two options are MANDATORY on every run (baked into
[`examples/vllm_neuron/models/qwen3_5_moe/run.py`](../../../../examples/vllm_neuron/models/qwen3_5_moe/run.py)):

- `hf_overrides={"architectures":["Qwen3_5MoeForCausalLM"]}` — load the text-only
  class, not the native VLM class.
- `limit_mm_per_prompt={"image":0,"video":0}` — skip the vision tower.

> ⚠️ Do not drop the `Qwen3_5MoeForCausalLM` override. Loading the native
> `Qwen3_5MoeForConditionalGeneration` triggers base vLLM's VLM config hook, which
> forces the SSM cache to fp32 → block_size 384 → compile OOM. The text-only alias
> keeps it bf16 → block 256.

## Architecture

| Parameter | Value |
|-----------|-------|
| Hidden size | 2048 |
| Num hidden layers | 40 (30 GatedDeltaNet + 10 full-attention) |
| Full-attn layers | indices 3, 7, 11, … , 39 |
| Num attention heads (full attn) | 16 |
| Num KV heads (full attn) | 2 (GQA) |
| Head dim (full attn) | 256 |
| Partial rotary factor | 0.25 (64 of 256 dims rotated) |
| Linear num key heads | 16 |
| Linear num value heads | 32 (K repeated 2× to V) |
| Linear key/value head dim | 128 |
| Linear conv kernel dim | 4 |
| Num experts | 256 |
| Experts per token (top-k) | 8 (+ shared expert) |
| MoE intermediate size | 512 |
| Vocab size | 248320 |
| SSM (mamba) dtype | float32 |

### Parallelism

- **TP8 / EP8** — pure expert parallelism, 32 experts per rank. All collectives go
  through one unified world group (the EP-DGE constraint). `tensor_parallel_size`
  must divide the 16 attention heads (max valid = 8).
- Prefill runs sequence-parallel; the GDN recurrent scan all-gathers to full-T then
  reduce-scatters `out_proj` back to SP-local.

### State management

- **KV cache** for the 10 full-attention layers — paged K/V from the block manager.
- **Recurrent + conv state** for the 30 GDN layers — lives in the unified paged
  pool, addressed by the stable page slot (`block_table[:,0]`) and read/written by
  slot-indexed NKI kernels (`gdn_state_update`, `gdn_conv_update`).

The port keeps the vLLM hybrid interfaces (`HasInnerState` / `IsHybrid`,
`get_mamba_state_shape_from_config()`, `get_mamba_state_dtype()`,
`bind_mamba_state()`).

### GDN prefill

Ships the **segmented sequential scan** GDN prefill (`VLLM_GDN_SEQ_NKI=1`), the
device-validated path. For seq ≥ 512 the bounded-graph segmented prefill is
**required** — the unsegmented monolithic prefill trips a scatter/gather OOB at long
sequence. Decode uses the recurrent scan + slot kernels.

## Features

| Category | Feature | Status |
|---|---|---|
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Expert parallelism (EP) | ✅ |
| | Pipeline parallelism (PP) | ❌ |
| **Performance** | Segmented GDN prefill | ✅ |
| | On-device sampling (greedy) | ✅ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (testing) | ✅ |
| **Inputs** | Text | ✅ |
| | Vision (image/video) | ❌ (text-only port) |

## Running

See [`doc/RUN_CONFIGS.md`](doc/RUN_CONFIGS.md) for the full tested TP8/EP8 recipe,
the required environment flags, and the segmented-prefill configuration. A minimal
offline example is
[`examples/vllm_neuron/models/qwen3_5_moe/run.py`](../../../../examples/vllm_neuron/models/qwen3_5_moe/run.py)
(sets the recipe flags via `os.environ.setdefault`, so `python run.py` works
standalone).
