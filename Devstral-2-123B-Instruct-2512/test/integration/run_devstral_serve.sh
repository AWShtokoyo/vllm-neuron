#!/bin/bash
# Devstral-2-123B BF16-dequant serve on the public vllm-neuron (Neuron 2.31) stack.
# Simple low-latency entry point: TP=32, single-shot prefill [2048], port 8000.
# This is the DEFAULT path (FP8 checkpoint dequantized to BF16 at load) — needs NO
# integration_nkilib.patch. For the TP=8 throughput config, override TP=8 (see the
# Low-TP throughput config in the README); for FP8-native see run_devstral_serve_fp8.sh.
#
# Config knobs (override on the CLI): TP (default 32), MAX_MODEL_LEN (4096),
# MAX_NUM_SEQS (2), PREFILL_BUCKET (2048), PORT (8000).
source "${VENV:-$HOME/vllm_neuron_231_venv}/bin/activate"
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_SKIP_EFA_AFFINITY=1
export NKI_COMPILE_CACHE_URL="${NKI_COMPILE_CACHE_URL:-$HOME/nki_cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/opt/nvme/vllm_cache_231}"
# Use the compiler-bundled nkilib (shipped in neuronx-cc), the version-matched tree.
export NKILIB_FORCE_BUNDLED_LIBRARY=true
mkdir -p "$NKI_COMPILE_CACHE_URL" "$VLLM_CACHE_ROOT"

# Prefer a local checkpoint if present (offline, faster); else the HF id.
CKPT=mistralai/Devstral-2-123B-Instruct-2512
[ -f /opt/nvme/devstral-ckpt/config.json ] && CKPT=/opt/nvme/devstral-ckpt

TP="${TP:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
PREFILL_BUCKET="${PREFILL_BUCKET:-2048}"
PORT="${PORT:-8000}"

echo "==> BF16-dequant serve: TP=$TP mml=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS bucket=$PREFILL_BUCKET ckpt=$CKPT port=$PORT"

ARTIFACT_DIR="${VLLM_CACHE_ROOT}/neuron_artifacts"; mkdir -p "$ARTIFACT_DIR"; cd "$ARTIFACT_DIR"   # neutral CWD for compiler artifact dumps. Do NOT cd into the vllm_neuron pkg dir: its vllm/ subpackage shadows the real vllm in vLLM's model-inspection subprocess (ModuleNotFoundError: vllm.model_executor).
exec vllm serve "$CKPT" \
  --served-model-name mistralai/Devstral-2-123B-Instruct-2512 \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$PREFILL_BUCKET" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --no-enable-prefix-caching \
  --config-format hf \
  --load-format safetensors \
  --hf-overrides '{"quantization_config": {}}' \
  --additional-config "{\"neuron_config\": {\"on_device_sampling_config\": {\"all_greedy\": \"true\"}, \"num_batched_tokens_buckets\": [${PREFILL_BUCKET}], \"num_seqs_buckets\": [${MAX_NUM_SEQS}]}}" \
  --port "$PORT"
