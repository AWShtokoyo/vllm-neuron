# SPDX-License-Identifier: Apache-2.0
"""Phase 2 BATCH equivalence (multi-sequence prefill).

Goal: confirm per-query key_bounds (model.py) prevents cross-sequence leakage.
When several different-length sentences are embedded in ONE batch, each result
must match (a) the HF reference for that sentence AND (b) the SAME sentence
embedded SOLO. A mismatch would mean a query attended into a neighbor sentence.

Reuses the tiny-weights setup from cpu_equivalence.py.
Threshold cos > 0.99 (bf16 + CPU sim)."""

import json
import os
import tempfile

import torch
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel
import transformers.models.llama.modeling_llama as MLL
from vllm import LLM

TINY = dict(
    vocab_size=256, hidden_size=256, intermediate_size=512,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=64, max_position_embeddings=128, rms_norm_eps=1e-5,
    rope_theta=500000.0,
)
ROPE = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
        "high_freq_factor": 4.0, "original_max_position_embeddings": 64}


def _save_tiny_tokenizer(model_dir, vocab_size=256):
    """Write a minimal WordLevel tokenizer (id i <-> "i") into model_dir.

    vLLM (0.21 onwards, incl. 0.24) builds a pooling IO-processor at LLM() init that requires a
    tokenizer even for embed models fed raw prompt_token_ids (the old
    skip_tokenizer_init=True path no longer works for embed). The tests feed
    token ids directly, so this tokenizer only needs to exist, not be accurate.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {str(i): i for i in range(vocab_size)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="0", pad_token="0",
        bos_token="1", eos_token="2",
    )
    fast.save_pretrained(model_dir)


def build(model_dir):
    cfg = LlamaConfig(**TINY, attention_bias=False, mlp_bias=False,
                      tie_word_embeddings=False, torch_dtype="bfloat16",
                      rope_scaling=ROPE, _attn_implementation="eager")
    torch.manual_seed(0)
    m = LlamaModel(cfg).to(torch.bfloat16)
    m.save_pretrained(model_dir)
    cp = os.path.join(model_dir, "config.json")
    c = json.load(open(cp))
    c["model_type"] = "llama_bidirec"
    c["architectures"] = ["LlamaBidirectionalModel"]
    c["pooling"] = "avg"
    c["rope_scaling"] = ROPE
    json.dump(c, open(cp, "w"))
    _save_tiny_tokenizer(model_dir, TINY["vocab_size"])
    return m, cfg


def hf_ref_embed(model, ids, am, cfg):
    B, T = ids.shape
    dt = next(model.parameters()).dtype

    def patched(*a, **k):
        full = torch.zeros(B, 1, T, T, dtype=dt)
        return full.masked_fill((am == 0).view(B, 1, 1, T), float("-inf"))

    orig = MLL.create_causal_mask
    MLL.create_causal_mask = patched
    try:
        h = model(input_ids=ids, attention_mask=am).last_hidden_state
    finally:
        MLL.create_causal_mask = orig
    mask = am.unsqueeze(-1).to(h.dtype)
    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(pooled.float(), p=2, dim=-1)


def main():
    d = tempfile.mkdtemp()
    model, cfg = build(d)

    # max_num_seqs=4 so multiple sequences can be packed into one prefill batch.
    llm = LLM(model=d, runner="pooling", max_model_len=128, max_num_seqs=4,
              tensor_parallel_size=1, enforce_eager=True,
              additional_config={"neuron_config": {"num_batched_tokens_buckets": [16, 128]}})

    prompts = [
        list(range(1, 11)),          # len 10
        list(range(5, 20)),          # len 15
        [3, 9, 1, 7, 42, 8],         # len 6
        list(range(30, 42)),         # len 12
    ]

    # --- (1) SOLO embeddings (one prompt per call) as the per-sentence baseline ---
    solo = []
    for toks in prompts:
        out = llm.embed([{"prompt_token_ids": toks}])
        solo.append(torch.tensor(out[0].outputs.embedding, dtype=torch.float32))

    # --- (2) BATCH: all prompts in ONE embed() call ---
    batch_out = llm.embed([{"prompt_token_ids": t} for t in prompts])
    batch = [torch.tensor(o.outputs.embedding, dtype=torch.float32) for o in batch_out]

    # --- (3) HF reference per sentence ---
    refs = []
    for toks in prompts:
        ids = torch.tensor([toks], dtype=torch.long)
        am = torch.ones_like(ids)
        refs.append(hf_ref_embed(model, ids, am, cfg)[0])

    worst_bs = 1.0   # batch vs solo  (cross-seq leakage check)
    worst_br = 1.0   # batch vs HF ref
    for i, toks in enumerate(prompts):
        cs_bs = F.cosine_similarity(batch[i], solo[i], dim=0).item()
        cs_br = F.cosine_similarity(batch[i], refs[i], dim=0).item()
        worst_bs = min(worst_bs, cs_bs)
        worst_br = min(worst_br, cs_br)
        print(f"  seq{i} len={len(toks):2d}: cos(batch, solo)={cs_bs:.6f}  cos(batch, HF_ref)={cs_br:.6f}")

    # discriminative sanity: distinct sentences should NOT be identical
    cross = F.cosine_similarity(batch[0], batch[1], dim=0).item()
    print(f"\n  cross-seq cos(batch[0], batch[1]) = {cross:.4f} (should be < ~0.9)")
    print(f"  worst cos(batch, solo)   = {worst_bs:.6f} (threshold > 0.99)")
    print(f"  worst cos(batch, HF_ref) = {worst_br:.6f} (threshold > 0.99)")

    assert worst_bs > 0.99, f"BATCH leaks across sequences vs solo: {worst_bs}"
    assert worst_br > 0.99, f"BATCH diverges from HF reference: {worst_br}"
    print("\nPHASE 2 BATCH EQUIVALENCE: PASS")
    print("  per-query key_bounds prevents cross-sequence leakage in batched prefill.")


if __name__ == "__main__":
    main()
