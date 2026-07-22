#!/bin/bash
# A/B on-device TTFT benchmark for multi-bucket prefill at TP=8/DP=1.
# DP=1 uses the plain single-front-end launch path (no DP coordinator). It still shows the
# multi-bucket effect: a SHORT-input
# burst where single-[2048] pads every prompt to 2048 (wasted prefill -> long queue) vs
# multi-[128..2048] which picks the tight bucket, draining the queue far faster.
#
# Runs TWO serves back-to-back on cores 0-7:
#   A) baseline  num_batched_tokens_buckets=[2048]
#   B) multibkt  num_batched_tokens_buckets=[128,256,512,1024,2048]
# Same short-input burst against each; compares median/P99 TTFT + throughput.
set -uo pipefail
source "${VENV:-$HOME/vllm_neuron_231_venv}/bin/activate"
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_SKIP_EFA_AFFINITY=1
export NKI_COMPILE_CACHE_URL="${NKI_COMPILE_CACHE_URL:-$HOME/nki_cache}"
export VLLM_CACHE_ROOT=/opt/nvme/vllm_cache_231
export NKILIB_FORCE_BUNDLED_LIBRARY=true
export NKILIB_MLP_BF16_XPOSE_SRC=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
mkdir -p "$NKI_COMPILE_CACHE_URL" "$VLLM_CACHE_ROOT"
export NEURON_VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

CKPT=/opt/nvme/devstral-ckpt
[ -f "$CKPT/config.json" ] || CKPT=mistralai/Devstral-2-123B-Instruct-2512
MODEL_ID=mistralai/Devstral-2-123B-Instruct-2512
TP=8; MML=2048; SEQS=128; QUANT="fp8:qkv,o_proj,mlp"; PORT="${PORT:-8150}"
IN_LEN="${IN_LEN:-128}"; OUT_LEN="${OUT_LEN:-128}"; NPROMPTS="${NPROMPTS:-256}"; CONC="${CONC:-256}"
RESULTS=/tmp/fp8_report/perf_231
mkdir -p "$RESULTS"

run_one(){ # $1=label  $2=buckets-csv
  local label="$1" buckets="$2" maxbt="${2##*,}"
  local addcfg="{\"neuron_config\": {\"on_device_sampling_config\": {\"all_greedy\": \"true\"}, \"num_batched_tokens_buckets\": [${buckets}], \"num_seqs_buckets\": [${SEQS}], \"quantization\": \"${QUANT}\"}}"
  echo "==================================================================="
  echo "==> [$label] serve TP=8/DP=1 buckets=[$buckets] maxbt=$maxbt port=$PORT"
  ARTIFACT_DIR="${VLLM_CACHE_ROOT}/neuron_artifacts"; mkdir -p "$ARTIFACT_DIR"; cd "$ARTIFACT_DIR"   # neutral CWD for compiler artifact dumps. Do NOT cd into the vllm_neuron pkg dir: its vllm/ subpackage shadows the real vllm in vLLM's model-inspection subprocess (ModuleNotFoundError: vllm.model_executor).
  vllm serve "$CKPT" --served-model-name "$MODEL_ID" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MML" --max-num-batched-tokens "$maxbt" --max-num-seqs "$SEQS" \
    --no-enable-prefix-caching --config-format hf --tokenizer-mode mistral \
    --load-format safetensors --hf-overrides '{"quantization_config": {}}' \
    --kv-cache-dtype fp8_e4m3 --additional-config "$addcfg" --port "$PORT" \
    > "$RESULTS/abdp1_${label}_serve.log" 2>&1 &
  local spid=$!
  local t0=$SECONDS
  until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    kill -0 "$spid" 2>/dev/null || { echo "!! [$label] server died"; tail -40 "$RESULTS/abdp1_${label}_serve.log"; return 2; }
    [ $((SECONDS-t0)) -gt "$VLLM_ENGINE_READY_TIMEOUT_S" ] && { echo "!! [$label] timeout"; kill -9 "$spid"; return 3; }
    sleep 10
  done
  echo "==> [$label] ready after $((SECONDS-t0))s. bench: in=$IN_LEN out=$OUT_LEN n=$NPROMPTS conc=$CONC"
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --host localhost --port "$PORT" --model "$MODEL_ID" \
    --tokenizer "$CKPT" --tokenizer-mode mistral \
    --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
    --num-prompts "$NPROMPTS" --max-concurrency "$CONC" \
    --ignore-eos --temperature 0 --seed 0 \
    --save-result --result-dir "$RESULTS" --result-filename "abdp1_${label}.json" 2>&1 | tail -35
  echo "==> [$label] shutting down server"
  kill "$spid" 2>/dev/null; for _ in $(seq 1 40); do kill -0 "$spid" 2>/dev/null||break; sleep 1; done
  kill -9 "$spid" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 8
}

run_one baseline_single "2048"
run_one multibucket "128,256,512,1024,2048"
echo "AB_RESULT=DONE  (results: $RESULTS/abdp1_baseline_single.json vs abdp1_multibucket.json)"
