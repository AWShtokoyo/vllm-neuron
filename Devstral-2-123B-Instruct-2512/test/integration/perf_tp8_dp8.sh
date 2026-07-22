#!/bin/bash
# TP=8/DP=8 full-FP8 PERFORMANCE bench — whole box (8 replicas, 16 chips / 64 cores).
# Headline: total ~10,180 tok/s, 512 prompts @ conc 1024.
# DP=8 needs the module-scope Ministral3Config fix (present) + DP CCOM env (from env_dp.sh).
set -uo pipefail
source "${VENV:-$HOME/vllm_neuron_231_venv}/bin/activate"
# DP=8 coordinator fix: force fork. The vllm CLI (cli_env_setup) otherwise sets
# VLLM_WORKER_MULTIPROC_METHOD=spawn, which makes the DPCoordinator child re-run the ~56s
# Neuron plugin import and blow its hard-coded 30s ZMQ-rendezvous timeout ("DP Coordinator
# process failed to report ZMQ addresses"). fork lets the child inherit the initialized parent.
export VLLM_WORKER_MULTIPROC_METHOD=fork
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_SKIP_EFA_AFFINITY=1
export NKI_COMPILE_CACHE_URL="${NKI_COMPILE_CACHE_URL:-$HOME/nki_cache}"
export VLLM_CACHE_ROOT=/opt/nvme/vllm_cache_231
export NKILIB_FORCE_BUNDLED_LIBRARY=true
export NKILIB_MLP_BF16_XPOSE_SRC=1
# DP CCOM workaround + rendezvous (from beta env_dp.sh)
export NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export VLLM_ENGINE_READY_TIMEOUT_S=2400
mkdir -p "$NKI_COMPILE_CACHE_URL" "$VLLM_CACHE_ROOT"
# TP=8 x DP=8 = 64 ranks. vLLM 0.21 SLICES NEURON_VISIBLE_DEVICES per DP-rank as
# [dp_rank*world_size, dp_rank*world_size+local_world_size) (utils.py:get_device_indices),
# so the base list must enumerate ALL 64 LOGICAL CORES (0-63), not 16 device indices.
export NEURON_VISIBLE_DEVICES="${VISIBLE_DEVICES:-$(seq -s, 0 63)}"

CKPT=mistralai/Devstral-2-123B-Instruct-2512
[ -f /opt/nvme/devstral-ckpt/config.json ] && CKPT=/opt/nvme/devstral-ckpt
MODEL_ID=mistralai/Devstral-2-123B-Instruct-2512
TP=8; DP=8; MML=2048; SEQS=128; BUCKET=2048; QUANT="fp8:qkv,o_proj,mlp"; PORT="${PORT:-8108}"
# Multi-bucket prefill: opt-in comma list, ascending, last==MML.
# Default single [2048] preserves the validated config byte-for-byte. Set e.g.
# PREFILL_BUCKETS=128,256,512,1024,2048 to let the scheduler pick the tightest prefill fit.
# Stays SINGLE-SHOT (last bucket == mml) so it does NOT trip the segmented-prefill path
# (which fails with FP8 KV) — verified via resolve_segmented_prefill_config(2048,2048)->None.
BUCKETS="${PREFILL_BUCKETS:-$BUCKET}"; MAX_BT="${BUCKETS##*,}"
RESULTS=/tmp/fp8_report/perf_231; mkdir -p "$RESULTS"
ADDCFG="{\"neuron_config\": {\"on_device_sampling_config\": {\"all_greedy\": \"true\"}, \"num_batched_tokens_buckets\": [${BUCKETS}], \"num_seqs_buckets\": [${SEQS}], \"quantization\": \"${QUANT}\"}}"

echo "==> [PERF DP=8] serving TP=8xDP=8 full-FP8 (whole box, 64 cores) port=$PORT"
ARTIFACT_DIR="${VLLM_CACHE_ROOT}/neuron_artifacts"; mkdir -p "$ARTIFACT_DIR"; cd "$ARTIFACT_DIR"   # neutral CWD for compiler artifact dumps. Do NOT cd into the vllm_neuron pkg dir: its vllm/ subpackage shadows the real vllm in vLLM's model-inspection subprocess (ModuleNotFoundError: vllm.model_executor).
vllm serve "$CKPT" \
  --served-model-name "$MODEL_ID" \
  --tensor-parallel-size "$TP" --data-parallel-size "$DP" \
  --api-server-count 1 \
  --max-model-len "$MML" --max-num-batched-tokens "$MAX_BT" --max-num-seqs "$SEQS" \
  --no-enable-prefix-caching --config-format hf --tokenizer-mode mistral \
  --load-format safetensors --hf-overrides '{"quantization_config": {}}' \
  --kv-cache-dtype fp8_e4m3 --additional-config "$ADDCFG" --port "$PORT" \
  > "$RESULTS/serve_dp8.log" 2>&1 &
SPID=$!
cleanup(){ kill "$SPID" 2>/dev/null; for _ in $(seq 1 40); do kill -0 "$SPID" 2>/dev/null||break; sleep 1; done; kill -9 "$SPID" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "==> waiting for readiness (timeout 2400s)..."
t0=$SECONDS
until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
  kill -0 "$SPID" 2>/dev/null || { echo "!! server died"; tail -50 "$RESULTS/serve_dp8.log"; exit 2; }
  [ $((SECONDS-t0)) -gt 2400 ] && { echo "!! timeout"; tail -50 "$RESULTS/serve_dp8.log"; exit 3; }
  sleep 5
done
echo "==> ready after $((SECONDS-t0))s"

echo "==> vllm bench serve: in=1024 out=256 n=512 conc=1024 (whole box)"
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --host localhost --port "$PORT" --model "$MODEL_ID" \
  --tokenizer "$CKPT" --tokenizer-mode mistral \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 512 --max-concurrency 1024 \
  --ignore-eos --temperature 0 --seed 0 \
  --save-result --result-dir "$RESULTS" --result-filename bench_dp8.json 2>&1 | tail -40
echo "STAGE_RESULT=DP8_DONE"
