# SPDX-License-Identifier: Apache-2.0
"""Phase 2 ON-DEVICE equivalence (Neuron 2.32 / vLLM 0.24).

The real 8B llama_bidirec model run through the full vLLM Neuron pooling path on
actual Neuron cores must match the official HF LlamaBidirectionalModel reference.
SPLIT into two stages to sidestep transformers>=5 reference-side incompatibilities
on the 2.32 stack:

  Stage 1 (offline, ANY transformers 4.x venv, CPU): build the HF reference
    embeddings + the tokenized prompt_token_ids, save to a .pt file. This is the
    "source of truth"; it does not touch Neuron at all. Run this file's --make-ref
    on a 4.x venv.

  Stage 2 (this file, the 2.32 vllm_neuron venv, on-device): load that .pt,
    feed the SAME prompt_token_ids to the compiled Neuron pooling graph, compare.

Why the split: transformers 5.x (the 2.32 venv) breaks the HF *reference* three
ways — (a) LlamaModel.forward calls create_causal_mask() directly, bypassing the
custom _update_causal_mask override -> reference silently runs CAUSAL (cos~0.81);
(b) create_bidirectional_mask / _prepare_4d_attention_mask no longer exist, so
the old §12 repointing fix doesn't apply; (c) the custom config fails rope-param
validation at load. All three are REFERENCE-side only — the ported vLLM model is
correctly bidirectional (CPU equivalence cos=0.999966). Precomputing the
reference on a 4.x venv keeps the yardstick trustworthy without fighting those.

Run:
  # Stage 1 (once, on a transformers 4.x venv, e.g. vllm_neuron_beta_v2_venv):
  python3 test/equivalence/ondevice_equivalence.py --make-ref \
      --ref /opt/nvme/ref_embeddings.pt
  # Stage 2 (on the 2.32 venv, on trn2; run_ondevice.sh sets the env):
  python3 test/equivalence/ondevice_equivalence.py \
      --ref /opt/nvme/ref_embeddings.pt

Threshold cos > 0.99 (bf16 on-device compute vs the bf16 HF reference)."""

import argparse
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

# Model dir: override with MODEL_DIR (e.g. /opt/nvme/model_hf/llama-embed-nemotron-8b)
# or the HF id "nvidia/llama-embed-nemotron-8b". Defaults to the repo-relative path.
D = os.environ.get("MODEL_DIR", "model_hf/llama-embed-nemotron-8b")

PROMPTS = [
    "The capital of France is Paris.",
    "Machine learning models can generate text.",
    "A quick brown fox jumps over the lazy dog.",
]


def _mean_pool(h, am):
    m = am.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


def _l2(x):
    return F.normalize(x, p=2, dim=-1)


def make_ref(ref_path):
    """Stage 1: compute HF reference embeddings + token ids on CPU (transformers 4.x).

    Saves {"prompts", "token_ids": list[list[int]], "refs": Tensor[N, hidden]}.
    """
    from transformers import AutoTokenizer

    spec = importlib.util.spec_from_file_location(
        "lbm", f"{D}/llama_bidirectional_model.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["lbm"] = m
    spec.loader.exec_module(m)

    import transformers.models.llama.modeling_llama as MLL

    tok = AutoTokenizer.from_pretrained(D)
    print("loading official 8B HF model on CPU (bf16, eager) for reference...")
    model = m.LlamaBidirectionalModel.from_pretrained(
        D, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager"
    ).eval()

    # CRITICAL (§12): the custom class makes the model bidirectional via its
    # _update_causal_mask override, but transformers>=4.5x has LlamaModel.forward
    # call create_causal_mask() DIRECTLY, bypassing that override — so the
    # reference silently runs CAUSAL and collapses to cos~0.81 vs the (correct,
    # bidirectional) device output. Repoint create_causal_mask to a padding-only
    # bidirectional mask for the reference forward (verified: token0 attends to
    # later tokens; cross-prompt cos ~0.28-0.30, matching the design notes).
    def _bidir_ref_forward(ids, am):
        B, T = ids.shape
        dt = next(model.parameters()).dtype

        def _bidir_mask(*a, **k):
            full = torch.zeros(B, 1, T, T, dtype=dt)
            return full.masked_fill((am == 0).view(B, 1, 1, T), float("-inf"))

        orig = MLL.create_causal_mask
        MLL.create_causal_mask = _bidir_mask
        try:
            with torch.no_grad():
                return model(input_ids=ids, attention_mask=am).last_hidden_state
        finally:
            MLL.create_causal_mask = orig

    refs, token_ids = [], []
    for pr in PROMPTS:
        enc = tok(pr, return_tensors="pt")
        ids, am = enc["input_ids"], enc["attention_mask"]
        h = _bidir_ref_forward(ids, am)
        refs.append(_l2(_mean_pool(h, am)).float()[0])
        token_ids.append(ids[0].tolist())

    torch.save(
        {"prompts": PROMPTS, "token_ids": token_ids, "refs": torch.stack(refs)},
        ref_path,
    )
    print(f"saved reference -> {ref_path}  (transformers "
          f"{__import__('transformers').__version__})")


def run_ondevice(ref_path):
    """Stage 2: feed the saved token ids to the compiled Neuron pooling graph."""
    import vllm_neuron  # noqa: F401  (registers llama_bidirec + platform)
    from vllm import LLM

    blob = torch.load(ref_path)
    token_ids = blob["token_ids"]
    refs = blob["refs"]
    prompts = blob.get("prompts", [f"prompt{i}" for i in range(len(token_ids))])

    print("\nbuilding vLLM LLM on Neuron (TP=1, runner=pooling)...")
    # Buckets MUST be > 128 (the TKG/decode-mode threshold in NF.mlp); keep all
    # prefill buckets in CTE mode. Feed prompt_token_ids directly (matches the
    # exact tokenization used for the reference; no tokenizer dependence here).
    llm = LLM(
        model=D,
        runner="pooling",
        max_model_len=512,
        max_num_seqs=1,
        tensor_parallel_size=1,
        additional_config={
            "neuron_config": {"num_batched_tokens_buckets": [256, 512]},
        },
    )

    outs = llm.embed([{"prompt_token_ids": t} for t in token_ids])

    worst = 1.0
    embs = []
    for pr, out, ref in zip(prompts, outs, refs):
        dev_emb = torch.tensor(out.outputs.embedding, dtype=torch.float32)
        embs.append(dev_emb)
        cs = F.cosine_similarity(dev_emb, ref, dim=0).item()
        # NaN guard: Python's min(1.0, nan) returns 1.0 (nan comparisons are
        # False), so a NaN embedding would silently keep worst=1.0 and PASS.
        # A NaN device output (e.g. a broken attention kernel) MUST fail.
        import math
        if math.isnan(cs):
            worst = float("nan")
        else:
            worst = min(worst, cs)
        print(f"  '{pr[:40]}...': cos(device, HF_ref) = {cs:.6f}")

    print(f"\nworst cos = {worst:.6f} (threshold > 0.99)")

    import itertools

    cross = [
        F.cosine_similarity(a, b, dim=0).item()
        for a, b in itertools.combinations(embs, 2)
    ]
    print(f"cross-prompt cos (device): {[round(c, 3) for c in cross]} "
          "(should be < 0.99)")

    import math as _math
    assert not _math.isnan(worst), (
        "on-device embedding is NaN (broken kernel/graph); reference is fine"
    )
    assert worst > 0.99, f"on-device embedding diverges from HF reference: {worst}"
    print("\nPHASE 2 ON-DEVICE EQUIVALENCE (2.32): PASS")
    print("  Real 8B llama_bidirec on Neuron preserves the embedding math.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/opt/nvme/ref_embeddings.pt",
                    help="path to the precomputed reference .pt")
    ap.add_argument("--make-ref", action="store_true",
                    help="Stage 1: build the reference on a transformers 4.x venv")
    args = ap.parse_args()

    if args.make_ref:
        make_ref(args.ref)
    else:
        run_ondevice(args.ref)


if __name__ == "__main__":
    main()
