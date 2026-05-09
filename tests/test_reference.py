"""Invariant tests for src/csa/reference.py.

These tests check structural correctness without DeepSeek inference fixtures:
- output shapes / dtypes
- causality of the lightning indexer (eq. 16) and top-k selector (eq. 17)
- that the compression weights (eq. 11) form a valid softmax distribution
- that top-k actually picks the highest finite scores
- determinism for fixed inputs
- that CSA degenerates to plain MQA over compressed KV when k = n_blk
- shape validation errors

When DeepSeek capture fixtures land in tests/fixtures/, add a fixtures-bitwise
suite alongside this one (per CLAUDE.md testing pyramid).
"""

from __future__ import annotations

import pytest
import torch

from csa import CSAConfig, csa_reference, random_params
from csa.reference import _compress_overlapped, _core_attn_mqa


def test_output_shapes_and_dtypes(small_case):
    out = csa_reference(h=small_case.h, cfg=small_case.cfg, p=small_case.p)
    n, _ = small_case.h.shape
    cfg = small_case.cfg
    n_blk = n // cfg.m

    assert out["c_comp"].shape == (n_blk, cfg.c)
    assert out["k_i_comp"].shape == (n_blk, cfg.c_i)
    assert out["i_scores"].shape == (n, n_blk)
    assert out["topk_idx"].shape == (n, cfg.k)
    assert out["o"].shape == (n, cfg.n_h, cfg.c)

    assert out["c_comp"].dtype == small_case.h.dtype
    assert out["o"].dtype == small_case.h.dtype
    assert out["topk_idx"].dtype == torch.int64


def test_causality_no_future_blocks(small_case):
    """eq. 16 + 17: token t may only attend to blocks s with s < floor(t/m)."""
    out = csa_reference(h=small_case.h, cfg=small_case.cfg, p=small_case.p)
    m = small_case.cfg.m
    n = small_case.h.shape[0]
    blk_of_t = torch.arange(n) // m
    idx = out["topk_idx"]
    valid = idx >= 0
    masked_max = torch.where(valid, idx, torch.full_like(idx, -1))
    for t in range(n):
        cap = int(blk_of_t[t].item())
        chosen = masked_max[t][masked_max[t] >= 0]
        assert (chosen < cap).all(), f"token {t}: selected block >= floor(t/m)={cap}: {chosen.tolist()}"


def test_early_tokens_have_padded_topk(small_case):
    """Tokens with floor(t/m) < k must have (k - floor(t/m)) entries padded with -1."""
    out = csa_reference(h=small_case.h, cfg=small_case.cfg, p=small_case.p)
    m = small_case.cfg.m
    k = small_case.cfg.k
    idx = out["topk_idx"]
    n = small_case.h.shape[0]
    for t in range(n):
        cap = t // m
        valid_count = int((idx[t] >= 0).sum().item())
        assert valid_count == min(cap, k), (
            f"token {t}: expected {min(cap, k)} valid blocks, got {valid_count} (idx={idx[t].tolist()})"
        )


def test_topk_picks_highest_visible_scores(small_case):
    """eq. 17: chosen indices correspond to the top-k visible scores."""
    out = csa_reference(h=small_case.h, cfg=small_case.cfg, p=small_case.p)
    scores = out["i_scores"]
    idx = out["topk_idx"]
    n, n_blk = scores.shape
    k = small_case.cfg.k
    for t in range(n):
        valid_mask = torch.isfinite(scores[t])
        visible = scores[t][valid_mask]
        if visible.numel() == 0:
            assert (idx[t] == -1).all()
            continue
        want_k = min(k, visible.numel())
        expected = torch.topk(visible, k=want_k).values
        chosen_idx = idx[t][idx[t] >= 0]
        got = scores[t][chosen_idx]
        assert torch.allclose(torch.sort(got, descending=True).values, expected), (
            f"token {t}: chose scores {got.tolist()} vs expected top-k {expected.tolist()}"
        )


def test_compression_weights_form_valid_softmax():
    """eq. 11: weights over the 2m row dimension must sum to 1 along that axis."""
    torch.manual_seed(0)
    n, c, m = 12, 5, 3
    c_a = torch.randn(n, c)
    c_b = torch.randn(n, c)
    z_a = torch.randn(n, c)
    z_b = torch.randn(n, c)
    b_a = torch.randn(m, c)
    b_b = torch.randn(m, c)

    n_blk = n // m
    z_a_blk = z_a.view(n_blk, m, c)
    z_b_blk = z_b.view(n_blk, m, c)
    z_b_prev = torch.empty_like(z_b_blk)
    z_b_prev[0] = torch.finfo(z_b.dtype).min
    z_b_prev[1:] = z_b_blk[:-1]
    logits = torch.cat([z_a_blk + b_a, z_b_prev + b_b], dim=1)
    s = torch.softmax(logits, dim=1)

    sums = s.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    out = _compress_overlapped(c_a=c_a, c_b=c_b, z_a=z_a, z_b=z_b, b_a=b_a, b_b=b_b, m=m)
    assert out.shape == (n_blk, c)


def test_first_block_only_uses_a_branch():
    """Per paper text: when i=0 the previous-block (b-branch) is padded with -inf logits and zero values.

    The softmax over [Z^a + B^a; -inf] degenerates to a softmax over Z^a + B^a alone, so the
    first compressed entry must equal the standard softmax-weighted sum over the first m a-tokens.
    """
    torch.manual_seed(7)
    n, c, m = 8, 3, 4
    c_a = torch.randn(n, c)
    c_b = torch.randn(n, c)
    z_a = torch.randn(n, c)
    z_b = torch.randn(n, c)
    b_a = torch.randn(m, c)
    b_b = torch.randn(m, c)

    out = _compress_overlapped(c_a=c_a, c_b=c_b, z_a=z_a, z_b=z_b, b_a=b_a, b_b=b_b, m=m)

    s_a = torch.softmax(z_a[:m] + b_a, dim=0)
    expected_first = (s_a * c_a[:m]).sum(dim=0)
    assert torch.allclose(out[0], expected_first, atol=1e-6)


def test_determinism(small_case):
    a = csa_reference(h=small_case.h, cfg=small_case.cfg, p=small_case.p)
    b = csa_reference(h=small_case.h, cfg=small_case.cfg, p=small_case.p)
    for key in a:
        assert torch.equal(a[key], b[key]), f"non-deterministic output for {key}"


def test_k_equals_n_blocks_matches_dense_mqa_over_compressed(k_equals_blocks_case):
    """When k = n_blk and the indexer is causal, every visible compressed block is selected.

    The expected output for token t is therefore plain MQA over c_comp[:floor(t/m)] using the
    same up-projected query q_t. This pins down eqs. 18–19 independently of the indexer (16–17).
    """
    case = k_equals_blocks_case
    out = csa_reference(h=case.h, cfg=case.cfg, p=case.p)
    cfg = case.cfg
    n = case.h.shape[0]

    c_q = case.h @ case.p.w_dq
    q = (c_q @ case.p.w_uq).view(n, cfg.n_h, cfg.c)
    c_comp = out["c_comp"]

    for t in range(cfg.m, n):
        cap = t // cfg.m
        kv = c_comp[:cap]
        expected = _core_attn_mqa(q=q[t], kv=kv, scale=cfg.scale)
        assert torch.allclose(out["o"][t], expected, atol=1e-5), f"mismatch at t={t}"

    for t in range(cfg.m):
        assert torch.equal(out["o"][t], torch.zeros_like(out["o"][t]))


def test_shape_validation_errors():
    cfg = CSAConfig(m=4, k=2, n_h=2, n_h_i=2, d_c=8, c=4, c_i=2)
    g = torch.Generator(device="cpu").manual_seed(0)
    h = torch.empty(16, 8).normal_(generator=g)
    p = random_params(cfg=cfg, d=8, generator=g)

    with pytest.raises(ValueError, match="rank-2"):
        csa_reference(h=torch.zeros(16), cfg=cfg, p=p)

    cfg_bad = CSAConfig(m=5, k=2, n_h=2, n_h_i=2, d_c=8, c=4, c_i=2)
    p_bad = random_params(cfg=cfg_bad, d=8, generator=g)
    with pytest.raises(ValueError, match="divisible"):
        csa_reference(h=h, cfg=cfg_bad, p=p_bad)

    cfg_overk = CSAConfig(m=4, k=100, n_h=2, n_h_i=2, d_c=8, c=4, c_i=2)
    p_overk = random_params(cfg=cfg_overk, d=8, generator=g)
    with pytest.raises(ValueError, match="cfg.k"):
        csa_reference(h=h, cfg=cfg_overk, p=p_overk)
