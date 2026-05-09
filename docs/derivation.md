# CSA Derivation: Paper §2.3.1 → `src/csa/reference.py`

This document walks through DeepSeek-V4 §2.3.1 (Compressed Sparse Attention)
equation-by-equation and points at the lines in `src/csa/reference.py` that
implement each one. The reference file is intentionally unfused and unoptimized
— it is the **test oracle** that all kernels must match (per `CLAUDE.md`).

The narrative follows the paper exactly: shapes use `n` for sequence length, `d`
for hidden size, `c` for KV head dim, `c_i` for indexer head dim, `n_h` for
core-attention query heads, `n_h_i` for indexer query heads, `d_c` for the
shared low-rank query latent dim, `m` for compression block size, and `k` for
top-k.

## Notation cheatsheet

| Symbol | Meaning | Shape |
| --- | --- | --- |
| `H` (`h`) | input hidden states for one sequence | `[n, d]` |
| `W_a^{KV}, W_b^{KV}` | KV projections (two streams: `a`, `b`) | `[d, c]` |
| `W_a^Z, W_b^Z` | compression-weight projections | `[d, c]` |
| `B^a, B^b` | learnable positional biases | `[m, c]` |
| `C^a, C^b` | per-token KV vectors before compression | `[n, c]` |
| `Z^a, Z^b` | per-token compression-weight vectors | `[n, c]` |
| `C^Comp` | compressed KV entries | `[n/m, c]` |
| `K_I^Comp` | compressed indexer keys (same compressor, dim `c_i`) | `[n/m, c_i]` |
| `c_t^Q` | shared low-rank query latent (eq. 13) | `[d_c]` |
| `q_t^I` (`q_{t,h}^I`) | indexer queries | `[n_h_i, c_i]` |
| `w_t^I` | per-head indexer weight | `[n_h_i]` |
| `I_{t, s}` | index score for query `t` over compressed block `s` | scalar |
| `q_t` | core-attention queries | `[n_h, c]` |
| `o_{t, i}` | per-head core-attention output | `[c]` |

## Step-by-step

### Eqs. 9–10. Per-token KV vectors and compression weights

> $C^a = H \cdot W_a^{KV}, \quad C^b = H \cdot W_b^{KV}$
>
> $Z^a = H \cdot W_a^Z, \quad Z^b = H \cdot W_b^Z$

CSA emits **two streams** of per-token KV vectors and compression weights so
that block `i`'s compressed entry can pull from both block `i` (via the `a`
stream) and block `i-1` (via the `b` stream) — the "overlapped compression"
mentioned right after eq. 12.

Reference: `csa_reference()` body (the four `h @ p.w_*` matmuls just before the
call to `_compress_overlapped`).

### Eqs. 11–12. Overlapped compression of `m` tokens into one entry

> $[S^a_{m i : m(i+1)-1};\, S^b_{m(i-1) : m i - 1}] = \mathrm{Softmax}_{\text{row}} \big([Z^a_{m i : m(i+1)-1} + B^a;\, Z^b_{m(i-1) : m i - 1} + B^b]\big)$
>
> $C^{\text{Comp}}_i = \sum_{j = m i}^{m(i+1)-1} S^a_j \odot C^a_j + \sum_{j = m(i-1)}^{m i - 1} S^b_j \odot C^b_j$

For each output block `i`, we form a `2m × c` matrix of logits by stacking the
`m` rows of `Z^a` from block `i` (plus bias `B^a`) with the `m` rows of `Z^b`
from block `i-1` (plus bias `B^b`). A row-wise softmax over the **2m
dimension** yields per-token, per-channel weights `S^a, S^b`, and the compressed
entry is the Hadamard-weighted sum.

Boundary: when `i = 0`, the previous block does not exist; the paper specifies
padding `Z^b` with `-∞` and `C^b` with zeros, so the softmax over the b-half
collapses to zero and the b-sum vanishes.

Reference: `_compress_overlapped()`.

### Lightning indexer: compressed indexer keys

> "CSA performs the same compression operation used for `C^Comp` to get
> compressed indexer keys `K_I^Comp ∈ R^{n/m × c_i}`."

Same overlapped compressor, different head dim (`c_i` instead of `c`) and a
separate set of projection / bias parameters.

Reference: the second `_compress_overlapped` call in `csa_reference()`.

### Eqs. 13–14. Shared low-rank query latent and indexer queries

> $c_t^Q = h_t \cdot W^{DQ}$
>
> $[q_{t,1}^I; \dots; q_{t, n_h^I}^I] = q_t^I = c_t^Q \cdot W^{IUQ}$

A single low-rank latent `c_t^Q ∈ R^{d_c}` is **shared** between the indexer's
query path (eq. 14) and the core attention's query path (eq. 18). This sharing
is what lets CSA do the indexer compute cheaply.

Reference: `c_q = h @ p.w_dq` and `q_i = (c_q @ p.w_iuq).view(...)`.

### Eqs. 15–16. Index score per (query token, compressed block)

> $w_t^I = h_t \cdot W^w$
>
> $I_{t, s} = \sum_{h=1}^{n_h^I} w_{t,h}^I \cdot \mathrm{ReLU}\big(q_{t, h}^I \cdot K_s^{I,\text{Comp}}\big)$

Note three quirks: (1) the per-head dot product is run through a **ReLU** before
weighting (this is the "lightning" simplification — no softmax over the
compressed-block axis at this stage); (2) the head weights `w_t^I` come from a
**different** projection of `h_t`, not from the shared latent `c_t^Q`; (3) the
score is a scalar, not a vector — one number per (t, s) pair.

Causality (paper text): only blocks `s` with `s < ⌊t / m⌋` are visible, since
block `⌊t / m⌋` overlaps with positions ≥ t.

Reference: `dot = einsum("tnc,sc->tns", q_i, k_i_comp); dot = relu(dot);
i_scores = einsum("tn,tns->ts", w_i, dot)`, followed by causal masking with
`-finfo.min`.

### Eq. 17. Top-k selection

> $\mathcal{C}_t^{\text{SprsComp}} = \{C_s^{\text{Comp}} \mid I_{t, s} \in \text{Top-}k(I_{t, :})\}$

Reference: `torch.topk(i_scores, k=cfg.k, dim=1)`. Tokens with fewer than `k`
visible blocks (early in the sequence) get padded with sentinel index `-1`,
which is filtered out in the per-token attention loop.

### Eq. 18. Core-attention queries from the shared latent

> $[q_{t, 1}; \dots; q_{t, n_h}] = q_t = c_t^Q \cdot W^{UQ}$

Reuse of `c_t^Q` from eq. 13 — no additional projection of `h_t`.

Reference: `q = (c_q @ p.w_uq).view(n, cfg.n_h, cfg.c)`.

### Eq. 19. Multi-Query Attention over selected blocks

> $o_{t, i} = \mathrm{CoreAttn}\big(\text{query} = q_{t, i},\,
>                                     \text{key} = \mathcal{C}_t^{\text{SprsComp}},\,
>                                     \text{value} = \mathcal{C}_t^{\text{SprsComp}}\big)$

A single shared KV (the selected compressed entries) is used as both key and
value for every query head — that's the MQA part. Scaling defaults to
`1 / sqrt(c)`, which is conventional and not specified in the paper at this
step.

Reference: the `for t in range(n)` loop calling `_core_attn_mqa(q=q[t], kv=...)`
in `csa_reference()`.

## What is intentionally **not** here (yet)

- **Grouped output projection** (paper text after eq. 19): a low-rank fold of
  `n_h` head outputs back to hidden dim `d`. Not part of CSA's "core
  attention" surface; will be added when we wire CSA into a Qwen layer.
- **RMSNorm on Q and KV** (§2.3.3 "Query and Key-Value Entry Normalization"):
  optional; can be added without changing the algorithm.
- **Partial RoPE** on the last 64 dims (§2.3.3): optional and orthogonal.
- **Sliding window attention branch** (§2.3.3): a separate, complementary
  attention path; out of scope until `reference.py` matches CSA fixtures.
- **Attention sink** (§2.3.3): trivial extra denominator term; defer until
  fixtures land.

## Testing

The reference is validated bottom-up (per `CLAUDE.md`):

1. Structural invariants (shapes, dtypes, causality, top-k correctness, softmax
   sums-to-one) — `tests/test_reference.py`.
2. Bitwise match against captured tensors from DeepSeek's inference impl —
   *pending fixtures*; see `tests/fixtures/README.md` for the capture
   protocol.
3. Quality (cosine similarity to dense attention on a real LLM) — the spike
   gate; runs end-to-end on a small Qwen.
