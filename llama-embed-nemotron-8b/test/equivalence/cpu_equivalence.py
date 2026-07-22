# SPDX-License-Identifier: Apache-2.0
"""Phase 2 CPU equivalence: the embedding returned through the full vLLM Neuron
runner (pooling path) must match the HF reference on the SAME tiny weights.

This closes the loop: Phase 1 proved the math; this proves the framework
integration (runner_type → scheduler → execute_model pooling branch →
pooler_output) preserves it. Threshold cos > 0.99 (bf16 compute + CPU sim)."""

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
    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    head_dim=64, max_position_embeddings=128, rms_norm_eps=1e-5,
    rope_theta=500000.0,
)
ROPE = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
        "high_freq_factor": 4.0, "original_max_position_embeddings": 64}


def _save_tiny_tokenizer(model_dir, vocab_size=256):
    """Write a minimal WordLevel tokenizer (id i <-> "i") into model_dir.

    vLLM 0.21 builds a pooling IO-processor at LLM() init that requires a
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

    llm = LLM(model=d, runner="pooling", max_model_len=128, max_num_seqs=2,
              tensor_parallel_size=1, enforce_eager=True,
              additional_config={"neuron_config": {"num_batched_tokens_buckets": [16, 128]}})

    prompts = [list(range(1, 11)), list(range(5, 20)), [3, 9, 1, 7, 42, 8]]
    worst = 1.0
    for toks in prompts:
        out = llm.embed([{"prompt_token_ids": toks}])
        runner_emb = torch.tensor(out[0].outputs.embedding, dtype=torch.float32)

        ids = torch.tensor([toks], dtype=torch.long)
        am = torch.ones_like(ids)
        ref = hf_ref_embed(model, ids, am, cfg)[0]

        cs = F.cosine_similarity(runner_emb, ref, dim=0).item()
        worst = min(worst, cs)
        print(f"  len={len(toks)}: cos(runner, HF_ref) = {cs:.6f}")

    print(f"\nworst cos = {worst:.6f} (threshold > 0.99)")
    assert worst > 0.99, f"runner embedding diverges from HF reference: {worst}"
    print("PHASE 2 CPU EQUIVALENCE: PASS")
    print("  Full vLLM Neuron pooling path preserves the embedding math.")


if __name__ == "__main__":
    main()
