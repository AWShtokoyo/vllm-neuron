# llama-embed-nemotron-8b — artifact bundle

vllm-neuron implementation of
[`nvidia/llama-embed-nemotron-8b`](https://huggingface.co/nvidia/llama-embed-nemotron-8b),
an 8B **bidirectional Llama embedding (pooling) model**: it runs a bidirectional
Llama-3.1-8B backbone, then mask-weighted mean-pools the final hidden states and
L2-normalizes them into a 4096-dim sentence embedding. It is prefill-only (no
decode) and is served with `--runner pooling`.

The model is **natively integrated** into this repository — you do not need to
apply anything to use it:

- Model package: [`vllm_neuron/model/llama_bidirec/`](../vllm_neuron/model/llama_bidirec/)
  (architecture `LlamaBidirectionalModel`, `model_type` `llama_bidirec`).
- Registration and the shared-file touch-points (`model/registry.py`,
  `vllm_neuron/__init__.py`, the scheduler / worker / runner pooling branches, the
  non-causal attention fallback, and the LNC1 codegen flag) are committed directly.

## Documentation

- **Model recipe / card:** [`docs/model-recipes/llama-embed-nemotron-8b.md`](../docs/model-recipes/llama-embed-nemotron-8b.md)
- **Deployment tutorial:** [`docs/tutorials/tutorial-llama-embed-nemotron-8b.md`](../docs/tutorials/tutorial-llama-embed-nemotron-8b.md)
- **Offline example:** [`examples/vllm_neuron/models/llama_bidirec/run.py`](../examples/vllm_neuron/models/llama_bidirec/run.py)

## Contents of this bundle

| Path | Purpose |
|---|---|
| `test/equivalence/` | CPU / on-device / batched embedding-equivalence tests vs the HF reference |
| `test/integration/` | serving + performance sweep scripts and recorded bench results (`results/tpN/`) |
| `integration.patch` | the shared-file edits as a standalone patch, for applying the model on top of a separately-installed `vllm_neuron` (already committed in-tree here; kept for reference / out-of-tree installs) |

## Running the tests

Set a venv and caches via environment variables (the scripts read
`VLLM_NEURON_VENV`, `VLLM_CACHE_ROOT`, `REF`, and `REF_VENV` — all overridable, with
portable `$HOME`-relative defaults):

```bash
# CPU equivalence (fast, no device): full vLLM pooling path vs HF reference
VLLM_NEURON_CPU_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python3 test/equivalence/cpu_equivalence.py

# On-device equivalence (trn2): builds the HF reference on a transformers-4.x venv
# (REF_VENV), then compares on-device on the 2.31 venv.
REF_VENV=$HOME/tf4x_venv bash test/integration/run_ondevice.sh

# Serving + perf (trn2): start the server, then drive the sweep
bash test/integration/run_serve_embed.sh &
python3 test/integration/serve_equivalence.py
TP=2 bash test/integration/run_tp_sweep.sh
```

On the Neuron 2.31 stack (transformers ≥ 5.13) the HF *reference* class must be
built on a **transformers-4.x** venv — the equivalence tests split into a
reference-build stage (CPU, 4.x) and an on-device compare stage. The port itself is
bidirectional and correct on the current stack; only the upstream HF reference class
breaks on transformers ≥ 5.
</content>
