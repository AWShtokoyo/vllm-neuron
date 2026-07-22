#!/bin/bash
# Serve + perf sweep at a given TP for the llama_bidirec embedding model.
# Usage: TP=2 bash test/integration/run_tp_sweep.sh
# TP=1 needs VLLM_NEURON_FORCE_LNC1=1 (single-core mis-shard dodge). TP>=2 MUST
# drop it (collectives fail to load otherwise) and use the default LNC=2.
set -u
TP="${TP:-1}"
VENV="${VLLM_NEURON_VENV:-$HOME/vllm_neuron_231_venv}"
MODEL="${MODEL:-model_hf/llama-embed-nemotron-8b}"
POOLING_PACK="${VLLM_NEURON_POOLING_PACK:-1}"
OUT="${OUT:-./perf_results_tp${TP}_pack${POOLING_PACK}}"
PORT="${PORT:-8000}"
source "$VENV/bin/activate"
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_SKIP_EFA_AFFINITY=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/vllm_cache_llamaembed_tp${TP}}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_NEURON_POOLING_PACK="$POOLING_PACK"
unset VLLM_NEURON_CPU_MODE
if [ "$TP" = "1" ]; then export VLLM_NEURON_FORCE_LNC1=1; else unset VLLM_NEURON_FORCE_LNC1; fi
mkdir -p "$OUT"

echo "### Serving TP=$TP pack=$POOLING_PACK (LNC1=${VLLM_NEURON_FORCE_LNC1:-unset})"
vllm serve "$MODEL" \
  --runner pooling --tensor-parallel-size "$TP" --max-model-len 512 --max-num-seqs 4 \
  --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [256, 512]}}' \
  --port "$PORT" > "/tmp/serve_tp${TP}.log" 2>&1 &
SERVE_PID=$!
# wait up to ~6 min for readiness (TP>=2 compiles more graphs)
for i in $(seq 1 72); do
  sleep 10
  curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1 && { echo "READY after ~$((i*10))s"; break; }
  if ! kill -0 $SERVE_PID 2>/dev/null; then echo "SERVER DIED"; tail -8 "/tmp/serve_tp${TP}.log"; exit 1; fi
done
curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1 || { echo "NOT READY"; tail -8 "/tmp/serve_tp${TP}.log"; kill -9 $SERVE_PID 2>/dev/null; exit 1; }
grep -h "Max prefills per batch" "/tmp/serve_tp${TP}.log" | tail -1

NUM=64
for ILEN in 128 256 500; do
  for CONC in 1 2 4; do
    echo "=== TP=$TP input_len=$ILEN concurrency=$CONC ==="
    vllm bench serve \
      --backend openai-embeddings --endpoint /v1/embeddings \
      --base-url "http://localhost:${PORT}" --model "$MODEL" \
      --dataset-name random --random-input-len $ILEN --random-output-len 1 \
      --num-prompts $NUM --max-concurrency $CONC \
      --percentile-metrics e2el --metric-percentiles 50,99 \
      --save-result --result-dir "$OUT" --result-filename "bench_i${ILEN}_c${CONC}.json" \
      2>/dev/null | grep -E "Successful|Failed|Request throughput|Total token throughput|Median E2EL|P99 E2EL"
    echo
  done
done
kill -9 $SERVE_PID 2>/dev/null
pkill -9 -f "vllm serve" 2>/dev/null
# The EngineCore subprocess holds the Neuron cores; pkill on "vllm serve" alone
# misses it, leaving cores busy for the next TP run. Kill it explicitly.
pkill -9 -f "EngineCore" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 5
echo "TP=$TP ALL DONE"
