#!/bin/bash
# Devstral-2-123B FP8-NATIVE serve on the public vllm-neuron (Neuron 2.31) stack
# (no BF16 dequant).
#
# Same stack + work-arounds as run_devstral_serve.sh (FORCE_BUNDLED nkilib,
# single-shot prefill mml==bucket), PLUS the `quantization` neuron_config knob that
# turns on FP8xFP8 STATIC matmuls in the qkv / o_proj / mlp CTE+TKG kernels.
#
# Requires integration_nkilib.patch applied to the BUNDLED nkilib
#   (the nkilib inside neuronx-cc, which `import nkilib` resolves to — see README
#   "Installation" step 3b). FORCE_BUNDLED is set below so that copy is the one used.
#
# QUANT options:
#   fp8:qkv,o_proj       — BF16-token-faithful target (30/30 first-token in beta), no MLP
#   fp8:qkv,o_proj,mlp   — full FP8: weight HBM -50%. NEEDS NKILIB_MLP_BF16_XPOSE_SRC=1
#                          (set automatically below when QUANT contains "mlp").
#
# By default this runs on FREE core indices 16-23 at port 8101 so it can coexist with a
# BF16-dequant server holding indices 0-15 / port 8100 for A/B.
#
# vLLM's multi-process executor REQUIRES NEURON_VISIBLE_DEVICES (NOT
# NEURON_RT_VISIBLE_CORES, which it rejects), and neuron_worker._get_visible_devices()
# asserts len(NEURON_VISIBLE_DEVICES) == num_local_ranks. So the list must have EXACTLY
# TP entries (one core index per rank), not "one device per 4 cores": for TP=8 pass 8
# indices (e.g. "16-23"). Leaving it unset makes vLLM auto-pick 0..TP-1, which collides
# with the BF16 server — so set it explicitly whenever coexisting. Override VISIBLE_DEVICES.
source "${VENV:-$HOME/vllm_neuron_231_venv}/bin/activate"
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_SKIP_EFA_AFFINITY=1
export NKI_COMPILE_CACHE_URL="${NKI_COMPILE_CACHE_URL:-$HOME/nki_cache}"
export VLLM_CACHE_ROOT=/opt/nvme/vllm_cache_231
export NKILIB_FORCE_BUNDLED_LIBRARY=true            # use the neuronx-cc-bundled nkilib
# Multi-bucket prefill (PREFILL_BUCKETS, below) cold-compiles one graph per bucket, so raise
# vLLM's engine-ready timeout. Harmless for single-bucket.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
mkdir -p "$NKI_COMPILE_CACHE_URL" "$VLLM_CACHE_ROOT"

QUANT="${QUANT:-fp8:qkv,o_proj}"
# Full FP8 MLP needs the gen3 DMA-transpose-then-quantize source path (see README FP8-native support).
case "$QUANT" in
  *mlp*) export NKILIB_MLP_BF16_XPOSE_SRC=1 ;;
esac

# Run on free core indices by default so we coexist with the BF16 server (indices 0-15).
# NEURON_VISIBLE_DEVICES must list EXACTLY TP entries (one per rank). TP=8 → "16-23".
VISIBLE_DEVICES="${VISIBLE_DEVICES-16-23}"
[ -n "$VISIBLE_DEVICES" ] && export NEURON_VISIBLE_DEVICES="$VISIBLE_DEVICES"

CKPT=mistralai/Devstral-2-123B-Instruct-2512
[ -f /opt/nvme/devstral-ckpt/config.json ] && CKPT=/opt/nvme/devstral-ckpt

TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
PREFILL_BUCKET="${PREFILL_BUCKET:-2048}"   # mml == bucket → single-shot prefill
# Multi-bucket prefill: opt-in comma list, ascending, last==mml.
# Default = the single PREFILL_BUCKET (unchanged, single-shot). Example:
#   PREFILL_BUCKETS=128,256,512,1024,2048  MAX_MODEL_LEN=2048
# Last bucket must == MAX_MODEL_LEN so every bucket is single-shot and the segmented-prefill
# path (which fails with FP8 KV) is never taken — verified with
# resolve_segmented_prefill_config(2048,2048) -> (None,None).
PREFILL_BUCKETS="${PREFILL_BUCKETS:-$PREFILL_BUCKET}"
MAX_BT="${PREFILL_BUCKETS##*,}"             # last (largest) bucket = max_num_batched_tokens
PORT="${PORT:-8101}"

echo "==> 2.31 FP8-native serve: QUANT=$QUANT NKILIB_MLP_BF16_XPOSE_SRC=${NKILIB_MLP_BF16_XPOSE_SRC:-0}"
echo "    TP=$TP mml=$MAX_MODEL_LEN buckets=[$PREFILL_BUCKETS] seqs=$MAX_NUM_SEQS devices=${NEURON_VISIBLE_DEVICES:-default} port=$PORT ckpt=$CKPT"

ARTIFACT_DIR="${VLLM_CACHE_ROOT}/neuron_artifacts"; mkdir -p "$ARTIFACT_DIR"; cd "$ARTIFACT_DIR"   # neutral CWD for compiler artifact dumps. Do NOT cd into the vllm_neuron pkg dir: its vllm/ subpackage shadows the real vllm in vLLM's model-inspection subprocess (ModuleNotFoundError: vllm.model_executor).
exec vllm serve "$CKPT" \
  --served-model-name mistralai/Devstral-2-123B-Instruct-2512 \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_BT" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --no-enable-prefix-caching \
  --config-format hf \
  --load-format safetensors \
  --hf-overrides '{"quantization_config": {}}' \
  --additional-config "{\"neuron_config\": {\"on_device_sampling_config\": {\"all_greedy\": \"true\"}, \"num_batched_tokens_buckets\": [${PREFILL_BUCKETS}], \"num_seqs_buckets\": [${MAX_NUM_SEQS}], \"quantization\": \"${QUANT}\"}}" \
  --port "$PORT"
