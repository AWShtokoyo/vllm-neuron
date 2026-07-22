#!/bin/bash
# vllm serve for llama_bidirec embedding model (trn2, TP=1, runner=pooling).
# Batching enabled (max_num_seqs=4) so the scheduler packs multiple /v1/embeddings
# requests into one prefill. Mirrors the offline phase2_ondevice_batch config.
source "${VLLM_NEURON_VENV:-$HOME/vllm_neuron_231_venv}/bin/activate"
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_SKIP_EFA_AFFINITY=1
# NEFF/NKI compile caches MUST be on a LOCAL fs (FileLock is not NFS-safe).
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/vllm_cache}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# REQUIRED: codegen workaround for the all-PyTorch graph (see README env note).
export VLLM_NEURON_FORCE_LNC1=1
unset VLLM_NEURON_CPU_MODE
MODEL="${MODEL:-model_hf/llama-embed-nemotron-8b}"
exec vllm serve "$MODEL" \
  --runner pooling \
  --tensor-parallel-size 1 \
  --max-model-len 512 \
  --max-num-seqs 4 \
  --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [256, 512]}}' \
  --port 8000
