#!/bin/bash
# Embedding perf sweep (mirrors Ministral3 method, adapted for pooling: no TTFT/TPOT).
# Server must be running on :8000. Sweep concurrency x input-len.
source "${VLLM_NEURON_VENV:-$HOME/vllm_neuron_231_venv}/bin/activate"
MODEL="${MODEL:-model_hf/llama-embed-nemotron-8b}"
OUT="${OUT:-./perf_results}"
mkdir -p $OUT
NUM=64   # requests per point (enough to amortize, small enough to be quick)
for ILEN in 128 256 500; do
  for CONC in 1 2 4; do
    echo "=== input_len=$ILEN concurrency=$CONC ==="
    vllm bench serve \
      --backend openai-embeddings --endpoint /v1/embeddings \
      --base-url http://localhost:8000 --model $MODEL \
      --dataset-name random --random-input-len $ILEN --random-output-len 1 \
      --num-prompts $NUM --max-concurrency $CONC \
      --percentile-metrics e2el --metric-percentiles 50,99 \
      --save-result --result-dir $OUT --result-filename bench_i${ILEN}_c${CONC}.json \
      2>/dev/null | grep -E "Successful|Failed|Benchmark duration|Total input tokens|Request throughput|Total token throughput|Mean E2EL|Median E2EL|P99 E2EL"
    echo
  done
done
echo "ALL DONE"
