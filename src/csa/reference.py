from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class CSAConfig:
    """
    Reference (oracle) implementation of DeepSeek-V4 CSA paper §2.3.1 (eqs. 9–19).

    This file intentionally prioritizes clarity over performance.
    """

    m: int
    k: int
    n_h: int
    n_h_i: int
    d_c: int
    c: int
    c_i: int
    causal: bool = True
    scale: Literal["sqrt_c", "none"] = "sqrt_c"


@dataclass(frozen=True)
class CSAParams:
    # KV compression projections (eqs. 9–10).
    w_a_kv: torch.Tensor  # [d, c]
    w_b_kv: torch.Tensor  # [d, c]
    w_a_z: torch.Tensor  # [d, c]
    w_b_z: torch.Tensor  # [d, c]
    b_a: torch.Tensor  # [m, c]
    b_b: torch.Tensor  # [m, c]

    # Indexer keys compression (same overlapped compressor, but with c_i dim).
    w_a_k_i: torch.Tensor  # [d, c_i]
    w_b_k_i: torch.Tensor  # [d, c_i]
    w_a_z_i: torch.Tensor  # [d, c_i]
    w_b_z_i: torch.Tensor  # [d, c_i]
    b_a_i: torch.Tensor  # [m, c_i]
    b_b_i: torch.Tensor  # [m, c_i]

    # Indexer low-rank query path (eqs. 13–16).
    w_dq: torch.Tensor  # [d, d_c]
    w_iuq: torch.Tensor  # [d_c, c_i * n_h_i]
    w_w: torch.Tensor  # [d, n_h_i]

    # Core attention query up-projection (eq. 18).
    w_uq: torch.Tensor  # [d_c, c * n_h]


def _check_2d(name: str, x: torch.Tensor) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [n, d], got shape={tuple(x.shape)}")


def _check_shape(name: str, x: torch.Tensor, shape: tuple[int | None, ...]) -> None:
    if x.ndim != len(shape):
        raise ValueError(f"{name} must have ndim={len(shape)}, got {x.ndim} (shape={tuple(x.shape)})")
    for i, (got, exp) in enumerate(zip(x.shape, shape, strict=True)):
        if exp is not None and got != exp:
            raise ValueError(f"{name} shape mismatch at dim {i}: got {got}, expected {exp}; shape={tuple(x.shape)}")


def _compress_overlapped(
    *,
    c_a: torch.Tensor,
    c_b: torch.Tensor,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    b_a: torch.Tensor,
    b_b: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """
    CSA overlapped KV compression (eqs. 11–12).

    Inputs:
      - c_a, c_b: [n, c]
      - z_a, z_b: [n, c]
      - b_a, b_b: [m, c]
    Output:
      - c_comp: [n//m, c]

    Notes:
      - Assumes n is divisible by m (as in packed training/inference; paper discusses trailing discard elsewhere).
      - For i=0, the "previous" block is padded with -inf logits and zero values (paper text).
    """
    _check_2d("c_a", c_a)
    _check_2d("c_b", c_b)
    _check_2d("z_a", z_a)
    _check_2d("z_b", z_b)
    _check_2d("b_a", b_a)
    _check_2d("b_b", b_b)
    if c_a.shape != c_b.shape or c_a.shape != z_a.shape or z_a.shape != z_b.shape:
        raise ValueError(
            "c_a, c_b, z_a, z_b must all have identical shape [n, c]; "
            f"got c_a={tuple(c_a.shape)} c_b={tuple(c_b.shape)} z_a={tuple(z_a.shape)} z_b={tuple(z_b.shape)}"
        )
    n, c = c_a.shape
    if b_a.shape != (m, c) or b_b.shape != (m, c):
        raise ValueError(f"b_a and b_b must be [m, c]=[{m}, {c}], got b_a={tuple(b_a.shape)} b_b={tuple(b_b.shape)}")
    if n % m != 0:
        raise ValueError(f"sequence length n={n} must be divisible by m={m} for reference implementation")

    n_blk = n // m
    device = c_a.device
    dtype = c_a.dtype

    # [n_blk, m, c]
    c_a_blk = c_a.view(n_blk, m, c)
    c_b_blk = c_b.view(n_blk, m, c)
    z_a_blk = z_a.view(n_blk, m, c)
    z_b_blk = z_b.view(n_blk, m, c)

    # Previous-block padding for i=0: logits=-inf, values=0.
    neg_inf = torch.finfo(z_b.dtype).min if z_b.dtype.is_floating_point else -1e9
    z_b_prev = torch.empty((n_blk, m, c), device=device, dtype=z_b.dtype)
    c_b_prev = torch.empty((n_blk, m, c), device=device, dtype=dtype)
    z_b_prev[0] = neg_inf
    c_b_prev[0] = 0
    if n_blk > 1:
        z_b_prev[1:] = z_b_blk[:-1]
        c_b_prev[1:] = c_b_blk[:-1]

    # Add positional biases (learnable) then softmax over the 2m "row" dimension (eq. 11).
    # logits: [n_blk, 2m, c]
    logits_a = z_a_blk + b_a.view(1, m, c)
    logits_b = z_b_prev + b_b.view(1, m, c)
    logits = torch.cat([logits_a, logits_b], dim=1)
    s = torch.softmax(logits, dim=1)  # Softmaxrow
    s_a, s_b = s[:, :m, :], s[:, m:, :]

    # Weighted sum with Hadamard product, then sum over tokens (eq. 12).
    c_comp = (s_a * c_a_blk).sum(dim=1) + (s_b * c_b_prev).sum(dim=1)  # [n_blk, c]
    return c_comp


def _core_attn_mqa(
    *,
    q: torch.Tensor,  # [n_h, c]
    kv: torch.Tensor,  # [k, c]
    scale: Literal["sqrt_c", "none"] = "sqrt_c",
) -> torch.Tensor:
    """
    Core attention used in eq. 19, in MQA form (shared key/value).

    Returns:
      - o: [n_h, c]
    """
    _check_shape("q", q, (None, None))
    _check_shape("kv", kv, (None, None))
    if q.shape[1] != kv.shape[1]:
        raise ValueError(f"q and kv must share last dim c; got q={tuple(q.shape)} kv={tuple(kv.shape)}")
    c = q.shape[1]
    logits = q @ kv.T  # [n_h, k]
    if scale == "sqrt_c":
        logits = logits / (c**0.5)
    attn = torch.softmax(logits, dim=-1)
    return attn @ kv  # [n_h, c]


@torch.no_grad()
def csa_reference(
    *,
    h: torch.Tensor,  # [n, d]
    cfg: CSAConfig,
    p: CSAParams,
) -> dict[str, torch.Tensor]:
    """
    CSA reference forward for a single sequence.

    Returns a dict with:
      - c_comp: [n_blk, c] compressed KV entries (eqs. 11–12)
      - k_i_comp: [n_blk, c_i] compressed indexer keys (paper text)
      - i_scores: [n, n_blk] index scores (eq. 16; masked where not visible)
      - topk_idx: [n, k] selected block indices (eq. 17; padded with -1 when fewer visible)
      - o: [n, n_h, c] core attention outputs per head (eq. 19)
    """
    _check_2d("h", h)
    n, d = h.shape
    m = cfg.m
    if n % m != 0:
        raise ValueError(f"n={n} must be divisible by m={m} for reference implementation")
    n_blk = n // m

    # eqs. 9–10
    _check_shape("p.w_a_kv", p.w_a_kv, (d, cfg.c))
    _check_shape("p.w_b_kv", p.w_b_kv, (d, cfg.c))
    _check_shape("p.w_a_z", p.w_a_z, (d, cfg.c))
    _check_shape("p.w_b_z", p.w_b_z, (d, cfg.c))
    c_a = h @ p.w_a_kv
    c_b = h @ p.w_b_kv
    z_a = h @ p.w_a_z
    z_b = h @ p.w_b_z

    c_comp = _compress_overlapped(c_a=c_a, c_b=c_b, z_a=z_a, z_b=z_b, b_a=p.b_a, b_b=p.b_b, m=m)

    # Indexer keys compression (paper: "same compression operation used for C_comp").
    _check_shape("p.w_a_k_i", p.w_a_k_i, (d, cfg.c_i))
    _check_shape("p.w_b_k_i", p.w_b_k_i, (d, cfg.c_i))
    _check_shape("p.w_a_z_i", p.w_a_z_i, (d, cfg.c_i))
    _check_shape("p.w_b_z_i", p.w_b_z_i, (d, cfg.c_i))
    k_i_a = h @ p.w_a_k_i
    k_i_b = h @ p.w_b_k_i
    z_i_a = h @ p.w_a_z_i
    z_i_b = h @ p.w_b_z_i
    k_i_comp = _compress_overlapped(
        c_a=k_i_a, c_b=k_i_b, z_a=z_i_a, z_b=z_i_b, b_a=p.b_a_i, b_b=p.b_b_i, m=m
    )  # [n_blk, c_i]

    # eqs. 13–16: produce indexer queries + weights, then scores against preceding blocks.
    _check_shape("p.w_dq", p.w_dq, (d, cfg.d_c))
    _check_shape("p.w_iuq", p.w_iuq, (cfg.d_c, cfg.c_i * cfg.n_h_i))
    _check_shape("p.w_w", p.w_w, (d, cfg.n_h_i))

    c_q = h @ p.w_dq  # [n, d_c] (eq. 13)
    q_i = (c_q @ p.w_iuq).view(n, cfg.n_h_i, cfg.c_i)  # [n, n_h_i, c_i] (eq. 14)
    w_i = h @ p.w_w  # [n, n_h_i] (eq. 15)

    # Scores I_{t,s} for s < floor(t/m) (eq. 16).
    # dot: [n, n_h_i, n_blk]
    dot = torch.einsum("tnc,sc->tns", q_i, k_i_comp)  # q·K
    dot = torch.relu(dot)
    i_scores = torch.einsum("tn,tns->ts", w_i, dot)  # sum_h w * relu(dot)

    # Mask non-visible blocks.
    if cfg.causal:
        blk_of_t = torch.arange(n, device=h.device, dtype=torch.int64) // m  # floor(t/m)
        s_idx = torch.arange(n_blk, device=h.device, dtype=torch.int64).view(1, n_blk)
        visible = s_idx < blk_of_t.view(n, 1)
        i_scores = i_scores.masked_fill(~visible, torch.finfo(i_scores.dtype).min)

    # eq. 17: top-k selection over I_{t,:}
    k = cfg.k
    if k > n_blk:
        raise ValueError(f"cfg.k={k} must be <= n_blk={n_blk}")
    topk_val, topk_idx = torch.topk(i_scores, k=k, dim=1)  # [n, k]
    # When a token has fewer than k visible blocks (early tokens), topk includes -inf; mark those as invalid.
    valid = torch.isfinite(topk_val)
    topk_idx = torch.where(valid, topk_idx, torch.full_like(topk_idx, -1))

    # eq. 18: core attention queries from shared c_q
    _check_shape("p.w_uq", p.w_uq, (cfg.d_c, cfg.c * cfg.n_h))
    q = (c_q @ p.w_uq).view(n, cfg.n_h, cfg.c)  # [n, n_h, c]

    # eq. 19: MQA over selected compressed KV blocks.
    o = torch.empty((n, cfg.n_h, cfg.c), device=h.device, dtype=h.dtype)
    for t in range(n):
        idx = topk_idx[t]  # [k]
        idx = idx[idx >= 0]
        if idx.numel() == 0:
            o[t].zero_()
            continue
        kv_t = c_comp.index_select(0, idx)  # [k_eff, c]
        o[t] = _core_attn_mqa(q=q[t], kv=kv_t, scale=cfg.scale)

    return {
        "c_comp": c_comp,
        "k_i_comp": k_i_comp,
        "i_scores": i_scores,
        "topk_idx": topk_idx,
        "o": o,
    }

