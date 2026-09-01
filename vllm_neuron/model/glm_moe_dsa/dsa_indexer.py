# SPDX-License-Identifier: Apache-2.0
"""GLM DSA (DeepSeek Sparse Attention) indexer — top-k key selection.

The checkpoint ships a lightweight indexer alongside MLA whose only job is to pick,
for each query, the `index_topk` keys worth attending to. With `index_topk=2048` the
reference model attends to 2,048 rows *regardless of context length*, so the
attention term stops growing past that point.

Reference: `transformers.models.glm_moe_dsa.modeling_glm_moe_dsa.GlmMoeDsaIndexer`.
This is a re-implementation for the Neuron traced path, validated against that
reference on CPU (`equiv_glm_moe_dsa/tests/test_07_dsa_indexer.py`) before any device
compile, because a compile is the expensive step in this project.

FOUR DETAILS THAT A RE-IMPLEMENTATION GETS WRONG IF NOT READ CAREFULLY. Each is
pinned by a test:

1. **`relu` goes before the head-weighted sum**, not after. Moving it changes which
   tokens win. It is invisible on short contexts because when `T <= index_topk`
   every key is selected and only the order differs — the test therefore uses
   `T > index_topk`.
2. **`k_norm` is a LayerNorm with bias**, not the RMSNorm used everywhere else in
   this model. The checkpoint carries `indexer.k_norm.bias` as a separate tensor.
3. **RoPE here is interleaved**, like the rest of GLM but unlike DeepSeek-V3.2,
   and it applies to the first `qk_rope_head_dim` of the indexer head, whose width
   (`index_head_dim=128`) is unrelated to the MLA head dims.
4. **`shared` layers own no weights.** `indexer_types[i]` is `"full"` or `"shared"`;
   a `"shared"` layer reuses the selection computed by the nearest preceding
   `"full"` layer. In the shipped checkpoint that is 21 `full` and 57 `shared`, so
   every layer ends up sparse while only 21 carry indexer weights.

TRACEABILITY. `topk` returns a fixed-width index tensor (`min(index_topk, T)`,
both compile-time constants once the bucket is fixed), so the selection is a static
shape — unlike a data-dependent bound such as `int(x.max().item())`, which cannot
be traced (see the `prior_bucket` comment in model.py).
"""

import torch
import torch.nn.functional as F
from torch import nn

from .config import GlmMoeDsaConfig

# Matches the sentinel the tiled MLA path uses: exp() of this underflows to 0, so a
# masked position contributes nothing without producing the NaN a true -inf would.
MASK_NEG = -30000.0

# Element budget for the per-head-tile score intermediate in `index_scores_segment`.
# 64 Mi fp32 elements = 256 MiB. Sized against ~24 GB per rank of which the weights
# already take ~11.8 GB: 256 MiB is affordable at every configuration, and the untiled
# alternative is 8.0 GiB at the 64K shape (S = Sk = 8192, H = 32).
_INDEX_SCORE_BUDGET_ELEMS = 64 * 1024 * 1024


def _apply_rope_interleaved_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved RoPE on the last dim of `x`, with cos/sin of half that width.

    🔴 The output layout is `cat([even*cos - odd*sin, odd*cos + even*sin], dim=-1)`,
    which is the reference's layout — NOT the `stack(...).flatten(-2)` interleave
    used by `_apply_rotary_emb_interleaved` in model.py. The two produce the same
    *set* of values in a different order, and since q and k are both rotated the dot
    product is unchanged; but the indexer must match the reference elementwise for
    the equivalence test to mean anything, so it follows the reference here.

    Args:
        x: [..., D] with D even; the rotated half is all of D.
        cos, sin: [..., D/2], broadcastable against x[..., 0::2].
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class GlmMoeDsaDsaIndexer(nn.Module):
    """Top-k key selection for one `"full"` layer.

    Weights are replicated per rank rather than sharded: the whole module is
    128×6144 + 4096×2048 + 32×6144 ≈ 9.6 M parameters, negligible against the
    ~11.8 GB/rank the model already carries, and replication avoids a collective on
    the selection path.
    """

    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        # The reference scales by head_dim ** -0.5 inside the score matmul.
        self.softmax_scale = self.head_dim ** -0.5

        self.wq_b_weight = nn.Parameter(
            torch.empty(self.q_lora_rank, self.n_heads * self.head_dim, dtype=self.dtype)
        )
        self.wk_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.head_dim, dtype=self.dtype)
        )
        # LayerNorm, with bias — see detail 2 in the module docstring.
        self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=self.dtype))
        self.k_norm_bias = nn.Parameter(torch.zeros(self.head_dim, dtype=self.dtype))
        self.k_norm_eps = 1e-6
        self.weights_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.n_heads, dtype=self.dtype)
        )

    def index_query(
        self,
        hidden_states: torch.Tensor,   # [B, S, hidden]
        q_resid: torch.Tensor,         # [B, S, q_lora_rank] — q_a_layernorm output
        cos: torch.Tensor,             # [S, rope/2] (this port's layout, not doubled)
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The query half of the score, which does not depend on the key range.

        Split out of `index_scores` so the tiled MLA path can compute it ONCE per
        query tile and then stream key segments past it, instead of needing every
        indexer key resident at once. That is what keeps the selection pass's peak
        memory at `q_tile x segment` rather than `q_tile x context`.

        Returns (q [B, S, H, D], weights [B, S, H]).
        """
        B, S, _ = hidden_states.shape
        H, D = self.n_heads, self.head_dim

        q = torch.matmul(q_resid, self.wq_b_weight).view(B, S, H, D)
        r = self.qk_rope_head_dim
        # cos/sin may arrive as [S, rope/2] (B==1) or [B, S, rope/2]; both have
        # B*S*(rope/2) elements, so one view covers each. 🔴 Not `view(1, S, ...)` --
        # that silently mis-broadcasts every batch beyond the first at B > 1.
        c = cos.view(B, S, 1, -1)
        s = sin.view(B, S, 1, -1)
        q = torch.cat(
            [_apply_rope_interleaved_pairs(q[..., :r], c, s), q[..., r:]], dim=-1
        )
        weights = torch.matmul(
            hidden_states.to(self.weights_proj_weight.dtype), self.weights_proj_weight
        ).float() * (H ** -0.5)                                # [B, S, H]
        return q, weights

    def index_scores_segment(
        self,
        q: torch.Tensor,          # [B, S, H, D] from index_query
        weights: torch.Tensor,    # [B, S, H]    from index_query
        key_seg: torch.Tensor,    # [B, Sk, head_dim] indexer keys for ONE segment
    ) -> torch.Tensor:
        """Scores for one key segment: [B, S, Sk].

        Exactly the tail of `index_scores`, so segment-wise evaluation and
        whole-context evaluation differ only in matmul blocking, not in operations.

        🔴 ACCUMULATED OVER HEAD TILES, because the naive form materialises
        `[B, S, H, Sk]` fp32 before the head-weighted sum collapses it to `[B, S, Sk]`.
        With H=32 that intermediate is 32x the result: at the 64K configuration
        (S = Sk = 8192) it is **8.0 GiB**, against ~24 GB per rank already carrying ~11.8
        GB of weights plus KV. Measured, not estimated: S*H*Sk*4 bytes.

        Head tiling rather than query tiling. Tiling queries would also bound it, but the
        caller re-reads the paged prior cache once per tile, and at S=8192 with a 128-wide
        query tile that is 64 sweeps of the prior — the prior read is the dominant cost of
        the whole path, so bounding memory that way would trade 8 GiB for a 64x slowdown.
        Head tiles are safe because `relu` is elementwise and the head-weighted sum is a
        sum, so partial sums over disjoint head groups are equivalent.

        ⚠️ Equivalent in exact arithmetic, NOT bit-identical in fp32: one 32-head matmul
        reduction and a sequence of partial sums use different summation orders. Measured
        at 5.3e-11 absolute on scores of order 1e-2, i.e. fp32 rounding. What is pinned by
        test is the SELECTION, which is what attention consumes; a tie separated by less
        than fp32 resolution is not a meaningful ordering to preserve, and the tiled MLA
        path already accepts the same class of deviation in its online-softmax accumulator.

        `relu` sits between the two matmuls, so they cannot be fused algebraically -- the
        accumulator `[B, S, Sk]` is the floor, and the tile only bounds what sits on top.
        """
        B, S, H, _ = q.shape
        Sk = key_seg.shape[1]
        k_t = key_seg.float().unsqueeze(1).transpose(-1, -2)      # [B, 1, D, Sk]

        # Largest head tile whose intermediate stays under the budget, at least 1.
        tile = max(1, min(H, _INDEX_SCORE_BUDGET_ELEMS // max(1, S * Sk)))

        out = None
        for h in range(0, H, tile):
            hi = min(h + tile, H)
            s = torch.matmul(q[:, :, h:hi].float(), k_t)          # [B, S, tile, Sk]
            s = F.relu(s * self.softmax_scale)                    # 🔴 relu BEFORE the sum
            part = torch.matmul(
                weights[..., h:hi].unsqueeze(-2), s
            ).squeeze(-2)                                          # [B, S, Sk]
            out = part if out is None else out + part
        return out

    def index_scores(
        self,
        hidden_states: torch.Tensor,   # [B, S, hidden]
        q_resid: torch.Tensor,         # [B, S, q_lora_rank] — q_a_layernorm output
        cos: torch.Tensor,             # [S, rope/2] (this port's layout, not doubled)
        sin: torch.Tensor,
        key_states: torch.Tensor,      # [B, T, head_dim] indexer keys for the full context
        q_positions: torch.Tensor,     # [B, S] absolute position of each query
    ) -> torch.Tensor:
        """The per-key score, before top-k. Returns [B, S, T]."""
        q, weights = self.index_query(hidden_states, q_resid, cos, sin)
        return self.index_scores_segment(q, weights, key_states)

    def compute_keys(
        self,
        hidden_states: torch.Tensor,   # [B, S, hidden]
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Indexer keys for these tokens: [B, S, head_dim]. Cacheable like KV."""
        B, S, _ = hidden_states.shape
        k = torch.matmul(hidden_states, self.wk_weight)
        k = F.layer_norm(k, (self.head_dim,), self.k_norm_weight, self.k_norm_bias, self.k_norm_eps)
        r = self.qk_rope_head_dim
        # See index_query: accepts [S, rope/2] or [B, S, rope/2], never assumes B==1.
        c = cos.view(B, S, -1)
        s = sin.view(B, S, -1)
        return torch.cat([_apply_rope_interleaved_pairs(k[..., :r], c, s), k[..., r:]], dim=-1)

    def select(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_states: torch.Tensor,
        q_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Top-k key indices per query: int32 [B, S, min(index_topk, T)]."""
        idx = self.index_scores(hidden_states, q_resid, cos, sin, key_states, q_positions)
        T = idx.shape[-1]
        key_pos = torch.arange(T, device=idx.device)
        causal = key_pos.view(1, 1, T) > q_positions.unsqueeze(-1)
        idx = idx.masked_fill(causal, float("-inf"))
        k = min(self.index_topk, T)
        return torch.topk(idx, k, dim=-1).indices.to(torch.int32)


def init_running_topk(
    batch: int,
    num_queries: int,
    k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Empty running top-k state: (values [B,S,k] = -inf, indices [B,S,k] = 0).

    Pre-filled to the FULL width `k` so every fold below sees a `k + segment` wide
    concatenation and therefore a static shape, independent of how many segments have
    been swept. The -inf filler is what makes a short context degenerate the same way
    the reference does: with fewer than `k` real keys the leftover entries stay -inf
    and carry index 0, which the tiled path's causal mask then discards. That is the
    same filler behaviour the tiled path's causal mask already discards, not a separate rule.
    """
    return (
        torch.full((batch, num_queries, k), float("-inf"), device=device, dtype=torch.float32),
        torch.zeros((batch, num_queries, k), device=device, dtype=torch.long),
    )


def running_topk_update(
    vals: torch.Tensor,          # [B, S, k]   running best scores
    idx: torch.Tensor,           # [B, S, k]   their GLOBAL key indices
    seg_scores: torch.Tensor,    # [B, S, seg] this segment's index scores
    ks: int,                     # global index of the segment's first key
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold one key segment into a running top-k. Returns the updated (vals, idx).

    WHY THIS EXISTS. The selection has to be made over the WHOLE key range before
    attention runs, but materialising `[B, S, T]` scores costs ~1 GB in fp32 at
    S=4096 / T=64K -- the very thing the tiled MLA path was written to avoid. Folding
    segment by segment holds `[B, S, k + seg]` instead (~42 MB at k=2048, seg=512).

    EXACTNESS. `top-k(A ∪ B) == top-k(top-k(A) ∪ top-k(B))`: a key excluded from a
    part's top-k is beaten by k keys inside that part, so it is beaten by k keys
    overall and cannot be in the global top-k. So the streaming result is the same
    SELECTION as one top-k over the concatenation, not an approximation. Ties may be
    broken differently, which is why the tests compare selected sets rather than
    index order.
    """
    B, S, seg = seg_scores.shape
    seg_idx = torch.arange(ks, ks + seg, device=seg_scores.device, dtype=torch.long)
    seg_idx = seg_idx.view(1, 1, seg).expand(B, S, seg)
    cat_v = torch.cat([vals, seg_scores.to(vals.dtype)], dim=-1)
    cat_i = torch.cat([idx, seg_idx], dim=-1)
    # 🔴 `torch.topk(x, ...)`, NOT `x.topk(...)`. The two spellings do NOT lower the same way on
    # trn2: the METHOD form emits a full-width HLO `sort` (rejected, NCC_EVRF029) while the
    # FUNCTION form lowers. Measured back-to-back on identical shapes, both spellings, 2/2
    # reproducible — including this file's own `[1,1,2560]` fold and the decode `[1,1,2049]`.
    # This is also why MoE routing's `torch.topk(scores, 8)` has shipped on device for months
    # while DSA's selection never compiled: the difference was the spelling, not the width or k.
    new_v, pos = torch.topk(cat_v, vals.shape[-1], dim=-1)
    return new_v, torch.gather(cat_i, -1, pos)


def segment_sparse_mask(
    topk_indices: torch.Tensor,   # [B, S, k] int32, indices into the GLOBAL key range
    ks: int,
    ke: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive DSA term for ONE key segment `[ks, ke)`, from a global selection.

    Returns [B, S, ke-ks]: 0 where the global key index is in the top-k, `MASK_NEG`
    where it is not. This is the form `_mla_attend_tiled` can consume: it already
    builds a per-segment causal/validity `mask_add`, and this term is ADDED to it.

    Causality is deliberately NOT applied here. The tiled path's own mask is the
    single place the causal limit is enforced, so duplicating it would create two
    sources of truth that could drift. Adding both terms lands a doubly-masked
    position at -60000, whose `exp()` underflows to 0 exactly as -30000 does, so the
    sum is harmless.

    🔴 Built by scattering into a TRASH COLUMN, not by `scatter(..., src=in_range)`.
    `topk_indices` legitimately contains duplicates and out-of-segment values: top-k
    over a row whose tail is -inf returns filler, and a global selection mostly
    points outside any given segment. Clamping an out-of-range index into the
    segment and scattering a "not selected" marker there would collide with a real
    "selected" written at the same position by another index, and scatter's result for
    duplicate indices is unspecified — so which one won would be undefined, silently
    opening or closing the wrong key. Routing every out-of-range index to one extra
    column that is then dropped removes the collision instead of racing on it.

    🔴 SCATTERS THE ADDITIVE MASK DIRECTLY, IN `dtype`, WITH NO BOOL INTERMEDIATE.
    The first version built a bool `selected` tensor and scattered `True` into it, then
    turned that into the mask with `torch.where`. That traces and runs fine on CPU but
    **does not lower to XLA**, and the device said so:

        RuntimeError: Error while lowering: [] aten::scatter,
                      xla_shape=pred[1,1,2]{2,1,0}, dynamic_dims: (), dim=2

    (`pred` is XLA's bool; [1,1,2] is B=1, S_decode=1, seg+1=2 — the decode path at
    `num_seqs_buckets=[1]` without speculation.) Scattering into a `dtype`-typed tensor
    avoids the unsupported op entirely, and it also makes duplicate indices harmless by
    VALUE rather than only by construction: every write is the same 0, so their order
    cannot matter.

    ⚠️ `torch.compile(fullgraph=True)` did not catch this. It exercises Dynamo and the
    CPU backend, not the XLA lowering, so it can prove the graph is single and
    shape-static while saying nothing about whether every op has a Neuron lowering. Ops
    added to a traced path still need a device compile to be cleared.
    """
    B, S, _ = topk_indices.shape
    seg = ke - ks
    dev = topk_indices.device
    local = topk_indices.to(torch.long) - ks
    in_range = (local >= 0) & (local < seg)
    local = torch.where(in_range, local, torch.full_like(local, seg))   # trash column
    add = torch.full((B, S, seg + 1), MASK_NEG, dtype=dtype, device=dev)
    add = add.scatter(-1, local, torch.zeros_like(local, dtype=dtype))
    return add[..., :seg]
