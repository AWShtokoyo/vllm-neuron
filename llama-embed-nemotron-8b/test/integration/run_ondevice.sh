#!/bin/bash
# On-device equivalence (Neuron 2.31 / vLLM 0.21), 2-STAGE. The real 8B llama_bidirec
# model on trn2 (TP=1, runner=pooling) must match the official HF LlamaBidirectionalModel
# reference. Expected: "PHASE 2 ON-DEVICE EQUIVALENCE (2.31): PASS".
#
# WHY TWO STAGES. On the 2.31 stack, transformers is >=5.13, which breaks the HF
# *reference* three ways (the ported vLLM model itself is fine): (a) LlamaModel.forward
# calls create_causal_mask() directly, bypassing the custom _update_causal_mask override
# -> the reference silently runs CAUSAL (cos~0.81); (b) create_bidirectional_mask /
# _prepare_4d_attention_mask no longer exist; (c) the custom config fails rope validation
# at load. So we PRECOMPUTE the reference on a transformers 4.x venv (Stage 1, CPU, no
# Neuron), then compare against the compiled Neuron graph on the 2.31 venv (Stage 2).
#
# VLLM_NEURON_FORCE_LNC1=1 is REQUIRED (Stage 2 only): it only affects neuronx-cc CODEGEN
# (passes --logical-nc-config 1), independent of the RUNTIME logical-core layout. This
# model's graph is all-PyTorch (no NKI kernel); the default LNC=2 auto-partitioner
# mis-shards it and neuronx-cc fails with "[NCC_IXRO002] Undefined SB Memloc". Codegen
# LNC=1 / runtime LNC=2 is the intended, working combo (runtime LNC=2 -> 24GB HBM/logical
# core, needed to fit the ~15GB bf16 weights).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
TEST=test/equivalence/ondevice_equivalence.py
REF="${REF:-$HOME/ref_embeddings.pt}"

# --- Stage 1: build the HF reference on a transformers 4.x venv (CPU, no Neuron) ---
# Point REF_VENV at any transformers-4.x env (required; no default).
REF_VENV="${REF_VENV:?set REF_VENV to a transformers-4.x venv, e.g. REF_VENV=\$HOME/tf4x_venv}"
if [ -f "$REF" ]; then
  echo "### Stage 1: reusing existing reference $REF (delete it to rebuild)"
else
  echo "### Stage 1: building HF reference on transformers 4.x venv: $REF_VENV"
  "$REF_VENV/bin/python3" "$TEST" --make-ref --ref "$REF"
fi

# --- Stage 2: on-device compare on the 2.31 vllm-neuron venv ---
set -x
source "${VLLM_NEURON_VENV:-$HOME/vllm_neuron_231_venv}/bin/activate"
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_SKIP_EFA_AFFINITY=1
# NEFF/NKI compile caches MUST be on a LOCAL fs (FileLock is not NFS-safe).
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/vllm_cache}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# REQUIRED: codegen workaround for the all-PyTorch graph (see README env note).
export VLLM_NEURON_FORCE_LNC1=1
unset VLLM_NEURON_CPU_MODE
python3 "$TEST" --ref "$REF"
