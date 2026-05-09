# CSA: paper to code

A walkthrough of DeepSeek-V4 §2.3.1 (Compressed Sparse Attention), with the
relevant lines in `src/csa/reference.py` cited next to each equation.

## Notation

| Symbol               | Meaning                                            | Shape       |
| -------------------- | -------------------------------------------------- | ----------- |
| `H` (`h`)            | input hidden states for one sequence               | `[n, d]`    |
| `W_a^{KV}, W_b^{KV}` | KV projections for the two streams                 | `[d, c]`    |
| `W_a^Z, W_b^Z`       | compression-weight projections                     | `[d, c]`    |
| `B^a, B^b`           | learnable positional biases                        | `[m, c]`    |
| `C^a, C^b`           | per-token KV vectors before compression            | `[n, c]`    |
| `Z^a, Z^b`           | per-token compression-weight vectors               | `[n, c]`    |
| `C^Comp`             | compressed KV entries                              | `[n/m, c]`  |
| `K_I^Comp`           | compressed indexer keys                            | `[n/m, c_i]`|
| `c_t^Q`              | shared low-rank query latent                       | `[d_c]`     |
| `q_t^I`              | indexer queries                                    | `[n_h_i, c_i]` |
| `w_t^I`              | per-head indexer weights                           | `[n_h_i]`   |
| `I_{t, s}`           | indexer score, query token `t` over block `s`      | scalar      |
| `q_t`                | core-attention queries                             | `[n_h, c]`  |
| `o_{t, i}`           | per-head core-attention output                     | `[c]`       |

## Eqs. 9–10. Two streams of per-token KV vectors and weights

`C^a = H W_a^{KV}`, `C^b = H W_b^{KV}`, and likewise for `Z^a`, `Z^b`.

The two streams exist so that block `i`'s compressed entry can pull from both
block `i` (`a` stream) and block `i-1` (`b` stream). This is the "overlapped
compression" mentioned right after eq. 12.

In code: the four `h @ p.w_*` matmuls in `csa_reference()` immediately before
the call to `_compress_overlapped`.

## Eqs. 11–12. Overlapped compression of m tokens into one entry

For each output block `i`, build a `[2m, c]` logits matrix by stacking the `m`
rows of `Z^a` from block `i` (plus bias `B^a`) on top of the `m` rows of `Z^b`
from block `i-1` (plus bias `B^b`). Row-wise softmax over the `2m` axis gives
weights `S^a, S^b`; the compressed entry is the Hadamard-weighted sum.

Boundary case `i = 0`: the previous block does not exist. The paper specifies
padding `Z^b` with `-inf` and `C^b` with zeros, which makes the b-half of the
softmax collapse and the b-sum vanish.

In code: `_compress_overlapped()`.

## Lightning indexer: compressed indexer keys

Same overlapped compressor as above, but with head dim `c_i` and a separate
set of projection / bias parameters. Produces `K_I^Comp` of shape `[n/m, c_i]`.

In code: the second `_compress_overlapped` call in `csa_reference()`.

## Eqs. 13–14. Shared low-rank query latent

`c_t^Q = h_t W^{DQ}`, then `q_t^I = c_t^Q W^{IUQ}` reshaped to `[n_h_i, c_i]`.

The latent `c_t^Q` is shared between the indexer query path (eq. 14) and the
core-attention query path (eq. 18). That sharing is what makes the indexer
cheap.

In code: `c_q = h @ p.w_dq` and `q_i = (c_q @ p.w_iuq).view(...)`.

## Eqs. 15–16. Indexer score per (query token, compressed block)

`w_t^I = h_t W^w`, then

```
I_{t, s} = sum_h w^I_{t, h} * relu(q^I_{t, h} . K^I,Comp_s)
```

Three things to notice: (i) the per-head dot product passes through ReLU
before weighting; (ii) the head weights `w_t^I` come from a separate
projection of `h_t`, not from `c_t^Q`; (iii) the score is a scalar per
`(t, s)`.

Causality (paper text): only blocks `s` with `s < floor(t / m)` are visible,
since block `floor(t / m)` overlaps positions ≥ `t`. In code we mask
non-visible blocks with `-inf` before top-k.

In code: `dot = einsum("tnc,sc->tns", q_i, k_i_comp); dot = relu(dot);
i_scores = einsum("tn,tns->ts", w_i, dot)`.

## Eq. 17. Top-k selection

`torch.topk(i_scores, k=cfg.k, dim=1)`. Tokens early in the sequence have
fewer than `k` visible blocks; their selected indices are padded with the
sentinel `-1` and filtered out before attention.

## Eqs. 18–19. Multi-Query Attention over selected blocks

`q_t = c_t^Q W^{UQ}` reshaped to `[n_h, c]`, then standard scaled-dot-product
attention with the selected compressed entries serving as both keys and
values for every head (the MQA part).

In code: the `for t in range(n)` loop calling `_core_attn_mqa(q=q[t], kv=...)`
in `csa_reference()`. Scaling defaults to `1 / sqrt(c)`.

## Out of scope for the reference

The reference covers the CSA path described in §2.3.1. The following are
mentioned in the paper but not implemented here:

- Grouped output projection (folds `n_h` head outputs back to hidden dim).
- RMSNorm on Q and KV (§2.3.3).
- Partial RoPE on the last 64 dims (§2.3.3).
- Sliding-window attention branch (§2.3.3).
- Attention sink (§2.3.3).
