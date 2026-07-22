# SPDX-License-Identifier: Apache-2.0
"""Verify `vllm serve /v1/embeddings` matches the HF reference (Neuron 2.31 / vLLM 0.21).
Server must already be running on :8000.

Uses the SAME precomputed reference as the on-device test (Stage 1 of
ondevice_equivalence.py). On the 2.31 stack transformers>=5.13 breaks the
HF *reference* (LlamaModel.forward bypasses the bidirectional-mask override -> the
reference silently runs CAUSAL, cos~0.81), so we do NOT load the official HF class
here; instead we load the .pt built on a transformers-4.x venv and POST the exact same
token ids. Build it first with:
    <tf4.x-venv>/bin/python3 test/equivalence/ondevice_equivalence.py \
        --make-ref --ref $HOME/ref_embeddings.pt
(run_ondevice.sh does this automatically as its Stage 1)."""
import argparse, json, math, urllib.request, itertools
import torch, torch.nn.functional as F

D = "model_hf/llama-embed-nemotron-8b"


def post_embed_ids(token_ids):
    # OpenAI embeddings API accepts pre-tokenized input as a list of ints, matching
    # the exact tokenization the reference was built from (no tokenizer dependence).
    body = json.dumps({"model": D, "input": token_ids}).encode()
    req = urllib.request.Request("http://localhost:8000/v1/embeddings",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    return torch.tensor(d["data"][0]["embedding"], dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="ref_embeddings.pt",
                    help="precomputed reference .pt (built on a transformers 4.x venv)")
    args = ap.parse_args()

    blob = torch.load(args.ref)
    token_ids = blob["token_ids"]
    refs = blob["refs"]
    prompts = blob.get("prompts", [f"prompt{i}" for i in range(len(token_ids))])

    worst = 1.0
    sv = []
    for pr, ids, ref in zip(prompts, token_ids, refs):
        emb = post_embed_ids(ids)
        sv.append(emb)
        cs = F.cosine_similarity(emb, ref, dim=0).item()
        # NaN guard: min(1.0, nan) == 1.0 (nan comparisons are False), which would
        # silently PASS a broken serve output. A NaN MUST fail.
        worst = float("nan") if math.isnan(cs) else min(worst, cs)
        print(f"  '{pr[:38]}...': cos(serve, HF_ref) = {cs:.6f}")

    cross = [F.cosine_similarity(a, b, dim=0).item()
             for a, b in itertools.combinations(sv, 2)]
    print(f"\ncross-prompt cos (serve): {[round(c, 3) for c in cross]}")
    print(f"worst cos (serve, HF_ref) = {worst:.6f} (threshold > 0.99)")
    assert not math.isnan(worst), "serve embedding is NaN (broken graph); reference is fine"
    assert worst > 0.99, f"serve diverges: {worst}"
    print("\nSERVE EQUIVALENCE: PASS")


if __name__ == "__main__":
    main()
