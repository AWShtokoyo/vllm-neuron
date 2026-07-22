#!/bin/bash
# TP=8/DP=1 full-FP8 PERFORMANCE bench — one replica (8 cores / 2 chips).
# Headline: total ~1,640 tok/s, 128 prompts @ conc 128.
# Workload: vllm bench serve, random, in 1024 / out 256, greedy.
set -uo pipefail
source "${VENV:-$HOME/vllm_neuron_231_venv}/bin/activate"
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_SKIP_EFA_AFFINITY=1
export NKI_COMPILE_CACHE_URL="${NKI_COMPILE_CACHE_URL:-$HOME/nki_cache}"
export VLLM_CACHE_ROOT=/opt/nvme/vllm_cache_231
export NKILIB_FORCE_BUNDLED_LIBRARY=true
export NKILIB_MLP_BF16_XPOSE_SRC=1
mkdir -p "$NKI_COMPILE_CACHE_URL" "$VLLM_CACHE_ROOT"
export NEURON_VISIBLE_DEVICES="${VISIBLE_DEVICES:-16,17,18,19,20,21,22,23}"

CKPT=mistralai/Devstral-2-123B-Instruct-2512
[ -f /opt/nvme/devstral-ckpt/config.json ] && CKPT=/opt/nvme/devstral-ckpt
MODEL_ID=mistralai/Devstral-2-123B-Instruct-2512
TP=8; MML=2048; SEQS=128; BUCKET=2048; QUANT="fp8:qkv,o_proj,mlp"; PORT="${PORT:-8108}"
# Multi-bucket prefill: opt-in comma list, ascending, last==MML.
# Default single [2048] preserves the validated config. Set e.g.
# PREFILL_BUCKETS=128,256,512,1024,2048 to let the scheduler pick the tightest prefill fit.
# Stays single-shot (last==mml) so the segmented-prefill path (fails w/ FP8 KV) is NOT triggered.
BUCKETS="${PREFILL_BUCKETS:-$BUCKET}"; MAX_BT="${BUCKETS##*,}"
RESULTS=/tmp/fp8_report/perf_231
mkdir -p "$RESULTS"
ADDCFG="{\"neuron_config\": {\"on_device_sampling_config\": {\"all_greedy\": \"true\"}, \"num_batched_tokens_buckets\": [${BUCKETS}], \"num_seqs_buckets\": [${SEQS}], \"quantization\": \"${QUANT}\"}}"

echo "==> [PERF DP=1] serving TP=8 full-FP8 devices=$NEURON_VISIBLE_DEVICES port=$PORT"
ARTIFACT_DIR="${VLLM_CACHE_ROOT}/neuron_artifacts"; mkdir -p "$ARTIFACT_DIR"; cd "$ARTIFACT_DIR"   # neutral CWD for compiler artifact dumps. Do NOT cd into the vllm_neuron pkg dir: its vllm/ subpackage shadows the real vllm in vLLM's model-inspection subprocess (ModuleNotFoundError: vllm.model_executor).
vllm serve "$CKPT" \
  --served-model-name "$MODEL_ID" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MML" --max-num-batched-tokens "$MAX_BT" --max-num-seqs "$SEQS" \
  --no-enable-prefix-caching --config-format hf --tokenizer-mode mistral \
  --load-format safetensors --hf-overrides '{"quantization_config": {}}' \
  --kv-cache-dtype fp8_e4m3 --additional-config "$ADDCFG" --port "$PORT" \
  > "$RESULTS/serve_dp1.log" 2>&1 &
SPID=$!
cleanup(){ kill "$SPID" 2>/dev/null; for _ in $(seq 1 30); do kill -0 "$SPID" 2>/dev/null||break; sleep 1; done; kill -9 "$SPID" 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "==> waiting for readiness (timeout 2400s)..."
t0=$SECONDS
until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
  kill -0 "$SPID" 2>/dev/null || { echo "!! server died"; tail -40 "$RESULTS/serve_dp1.log"; exit 2; }
  [ $((SECONDS-t0)) -gt 2400 ] && { echo "!! timeout"; tail -40 "$RESULTS/serve_dp1.log"; exit 3; }
  sleep 5
done
echo "==> ready after $((SECONDS-t0))s"

echo "==> vllm bench serve: in=1024 out=256 n=128 conc=128"
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --host localhost --port "$PORT" --model "$MODEL_ID" \
  --tokenizer "$CKPT" --tokenizer-mode mistral \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 128 --max-concurrency 128 \
  --ignore-eos --temperature 0 --seed 0 \
  --save-result --result-dir "$RESULTS" --result-filename bench_dp1.json 2>&1 | tail -40
echo "STAGE_RESULT=DP1_DONE"
