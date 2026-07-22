# SPDX-License-Identifier: Apache-2.0
"""On-device PERFORMANCE A/B for FP8-native vs BF16-dequant dense projections.

This is the companion to ``run_fp8_ab.py`` (which measures *correctness*:
first-token agreement / perplexity). This script measures *speed & memory* — the
question the whole effort exists to answer: is keeping the dense weights in FP8
(FP8xFP8 matmul) actually faster / lighter than dequantizing them to BF16 at load?

It reports, for ONE quant config:
  * TTFT (prefill latency)            — median over N prompts, fixed prompt length
  * decode throughput (tok/s/seq)     — from a long generation, prefill subtracted
  * end-to-end generate() wall time
  * analytical dense-weight HBM bytes (exact; the deterministic footprint claim)
  * peak device HBM used (neuron-monitor, all 16 devices) as a cross-check

FAIRNESS: bf16 and fp8 MUST run at the SAME --prefill-bucket. fp8:qkv needs a
bucket <= 1024 (SBUF overflow NCC_INKI016 at 2048), so the apples-to-apples A/B
is bucket=1024 for BOTH. Bucketing alone changes latency, so never compare a
bf16@2048 run against an fp8@1024 run.

Usage (trn2 host, venv active, NEURON_PLATFORM_TARGET_OVERRIDE=trn2):

    python run_fp8_perf.py --quant bf16          --prefill-bucket 1024
    python run_fp8_perf.py --quant fp8:qkv,o_proj --prefill-bucket 1024
    # then diff the two JSON outputs (a helper prints a table at the end).
"""

import argparse
import json
import os
import subprocess
import time

# ----- model dims (Devstral-2-123B / Ministral3), for analytical weight HBM -----
HIDDEN = 12288
INTERMEDIATE = 28672
N_KV_HEADS = 8
HEAD_DIM = 128
N_LAYERS = 88
# q: hidden x hidden ; k,v: (kv_heads*head_dim) x hidden ; o: hidden x hidden
KV_DIM = N_KV_HEADS * HEAD_DIM  # 1024
_DENSE_WEIGHTS = {
    "q": HIDDEN * HIDDEN,
    "k": KV_DIM * HIDDEN,
    "v": KV_DIM * HIDDEN,
    "o": HIDDEN * HIDDEN,
    "gate": INTERMEDIATE * HIDDEN,
    "up": INTERMEDIATE * HIDDEN,
    "down": HIDDEN * INTERMEDIATE,
}
_FAMILY_OF = {
    "q": "qkv", "k": "qkv", "v": "qkv", "o": "o_proj",
    "gate": "mlp", "up": "mlp", "down": "mlp",
}


def analytical_weight_hbm(fp8_families: set) -> dict:
    """Total dense projection-weight bytes across all layers (whole model, all
    ranks combined — TP just splits the same total). fp8 family => 1 byte/elem,
    else bf16 => 2 bytes/elem. Scales are scalar per tensor => negligible."""
    bf16_bytes = fp8_bytes = 0
    for proj, n in _DENSE_WEIGHTS.items():
        per_layer = n
        bf16_bytes += per_layer * 2
        fam = _FAMILY_OF[proj]
        fp8_bytes += per_layer * (1 if fam in fp8_families else 2)
    return {
        "all_bf16_GB": round(bf16_bytes * N_LAYERS / 1e9, 2),
        "this_config_GB": round(fp8_bytes * N_LAYERS / 1e9, 2),
        "saved_GB": round((bf16_bytes - fp8_bytes) * N_LAYERS / 1e9, 2),
    }


def sample_peak_hbm(stop_after_s, period=0.5):
    """Run neuron-monitor for stop_after_s seconds, return peak summed device
    HBM-used bytes across all neuron devices (best-effort; 0 if unavailable)."""
    peak = 0
    try:
        proc = subprocess.Popen(
            ["neuron-monitor"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        t_end = time.time() + stop_after_s
        for line in proc.stdout:
            try:
                o = json.loads(line)
            except Exception:
                continue
            total = 0
            for rt in o.get("neuron_runtime_data", []):
                rep = rt.get("report", {})
                mu = rep.get("memory_used", {})
                # neuron_runtime_used_bytes is the per-runtime device HBM total
                v = mu.get("neuron_runtime_used_bytes")
                if isinstance(v, dict):  # sometimes {usage_breakdown:..., "neuron_device": N}
                    v = v.get("neuron_device") or sum(
                        x for x in v.values() if isinstance(x, (int, float))
                    )
                if isinstance(v, (int, float)):
                    total += v
            peak = max(peak, total)
            if time.time() > t_end:
                break
        proc.terminate()
    except Exception:
        pass
    return peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-checkpoint", default="/opt/nvme/devstral-ckpt")
    p.add_argument("--tensor-parallel-size", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--prefill-bucket", type=int, default=1024,
                   help="num_batched_tokens bucket; fp8:qkv needs <=1024. "
                        "Use the SAME value for bf16 and fp8 for a fair A/B.")
    p.add_argument("--quant", default="bf16",
                   help="bf16 | fp8 | fp8:qkv | fp8:qkv,o_proj | ...")
    p.add_argument("--prompt-len", type=int, default=512,
                   help="approx prompt token length for TTFT (padded by repetition)")
    p.add_argument("--decode-tokens", type=int, default=256,
                   help="generated tokens for the decode-throughput measurement")
    p.add_argument("--ttft-trials", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = args.out or f"/opt/nvme/fp8_perf_{args.quant.replace(':', '_').replace(',', '+')}_b{args.prefill_bucket}.json"

    from vllm import LLM, SamplingParams

    fp8_families = set()
    if args.quant and args.quant != "bf16":
        spec = args.quant.split(":", 1)
        if len(spec) == 2:
            fp8_families = {f.strip() for f in spec[1].split(",")}
        else:  # bare "fp8" => all
            fp8_families = {"qkv", "o_proj", "mlp"}

    bucket = args.prefill_bucket
    neuron_config = {
        "on_device_sampling_config": {"all_greedy": "true"},
        "num_batched_tokens_buckets": [bucket],
        "num_seqs_buckets": [2],
    }
    if args.quant and args.quant != "bf16":
        neuron_config["quantization"] = args.quant

    print(f"\n===== PERF quant={args.quant} bucket={bucket} TP={args.tensor_parallel_size} =====", flush=True)

    t_load0 = time.time()
    llm = LLM(
        model=args.model_checkpoint,
        max_model_len=args.max_model_len,
        config_format="hf",
        tokenizer_mode="mistral",
        load_format="safetensors",
        max_num_seqs=2,
        max_num_batched_tokens=bucket,
        enable_prefix_caching=False,
        tensor_parallel_size=args.tensor_parallel_size,
        hf_overrides={"quantization_config": {}},
        additional_config={"neuron_config": neuron_config},
    )
    load_s = time.time() - t_load0

    # A prompt of ~prompt_len tokens (repeat a sentence; exact length not critical,
    # it just has to be IDENTICAL across the bf16 and fp8 runs, and < bucket).
    base = "The quick brown fox jumps over the lazy dog. "
    prompt = (base * ((args.prompt_len // 9) + 1)).strip()

    greedy1 = SamplingParams(max_tokens=1, temperature=0.0)
    greedyN = SamplingParams(max_tokens=args.decode_tokens, temperature=0.0)

    # ---- warmup (triggers graph compile for both the prefill and decode buckets) ----
    print("warmup...", flush=True)
    llm.generate([prompt], greedyN)

    # ---- TTFT: max_tokens=1, median wall time over trials ----
    ttfts = []
    for _ in range(args.ttft_trials):
        t0 = time.time()
        llm.generate([prompt], greedy1)
        ttfts.append(time.time() - t0)
    ttfts.sort()
    ttft_med = ttfts[len(ttfts) // 2]

    # ---- decode throughput: long generation, subtract one TTFT ----
    t0 = time.time()
    out = llm.generate([prompt], greedyN)
    e2e = time.time() - t0
    n_gen = len(out[0].outputs[0].token_ids)
    decode_s = max(e2e - ttft_med, 1e-6)
    decode_tok_s = (n_gen - 1) / decode_s if n_gen > 1 else 0.0

    peak_hbm = sample_peak_hbm(stop_after_s=4)
    wt = analytical_weight_hbm(fp8_families)

    result = {
        "quant": args.quant,
        "fp8_families": sorted(fp8_families),
        "prefill_bucket": bucket,
        "tp": args.tensor_parallel_size,
        "prompt_len_chars": len(prompt),
        "load_s": round(load_s, 1),
        "ttft_s_median": round(ttft_med, 4),
        "ttft_s_all": [round(x, 4) for x in ttfts],
        "decode_tokens": n_gen,
        "e2e_s": round(e2e, 4),
        "decode_tok_s_per_seq": round(decode_tok_s, 2),
        "weight_hbm": wt,
        "peak_device_hbm_GB": round(peak_hbm / 1e9, 2) if peak_hbm else None,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
