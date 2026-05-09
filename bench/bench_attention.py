"""CSA vs. dense MQA: the three numbers (memory ratio, tok/s, cosine similarity).

Per CLAUDE.md: do NOT add metrics. Add depth to these three.

Usage:
    python bench/bench_attention.py --n 1024 --d 256 --m 16 --k 8 --device cpu

Notes on interpretation:
- `mem_ratio`  : KV-cache footprint of CSA over dense (for a single layer at sequence
                 length n). Lower is better for CSA; this number does not depend on
                 weight initialization.
- `tok_s`      : tokens / second of the *whole forward* (compression + indexer + MQA)
                 vs. plain dense MQA over the uncompressed sequence. With the
                 reference (pure-PyTorch, unfused) implementation, CSA will lose on
                 wall-clock at small n; CSA is only expected to win once kernels are
                 fused (Phase 1, Triton) and at long context (n >> m * k).
- `cos_sim`    : cosine similarity between CSA's per-token output and the dense
                 baseline's per-token output, with both initialized from the same
                 random projections where shapes match. With random init this number
                 is essentially noise; it only becomes a quality signal when CSA is
                 bolted onto pre-trained LLM weights (the spike gate in CLAUDE.md).
                 We still print it as a smoke check — sudden NaN/zero indicates a bug.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from csa import CSAConfig, csa_reference, random_params


@dataclass
class BenchResult:
    n: int
    d: int
    m: int
    k: int
    n_h: int
    c: int
    mem_ratio: float
    csa_tok_s: float
    dense_tok_s: float
    cos_sim_mean: float
    cos_sim_p10: float


def _dense_mqa_forward(*, h: torch.Tensor, w_kv: torch.Tensor, w_q: torch.Tensor, n_h: int, c: int) -> torch.Tensor:
    """A plain causal MQA baseline over the uncompressed sequence.

    Single shared KV head, n_h query heads, head_dim = c. Used as the dense reference
    for the bench's three numbers. Not optimized.
    """
    n, _ = h.shape
    kv = h @ w_kv  # [n, c]
    q = (h @ w_q).view(n, n_h, c)
    logits = torch.einsum("thc,sc->ths", q, kv) / (c**0.5)
    causal = torch.tril(torch.ones(n, n, device=h.device, dtype=torch.bool))
    logits = logits.masked_fill(~causal.unsqueeze(1), torch.finfo(logits.dtype).min)
    attn = torch.softmax(logits, dim=-1)
    return torch.einsum("ths,sc->thc", attn, kv)


def _kv_bytes_csa(n: int, m: int, c: int, c_i: int, dtype: torch.dtype) -> int:
    """CSA per-layer KV cache: compressed main KV (n/m, c) + indexer keys (n/m, c_i)."""
    elem = torch.tensor([], dtype=dtype).element_size()
    n_blk = n // m
    return elem * n_blk * (c + c_i)


def _kv_bytes_dense(n: int, c: int, dtype: torch.dtype) -> int:
    """Dense MQA per-layer KV cache for a single shared KV head of width c."""
    elem = torch.tensor([], dtype=dtype).element_size()
    return elem * n * c


def _time_forward(fn, *, warmup: int = 1, iters: int = 3, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def run(*, n: int, d: int, m: int, k: int, device: torch.device, dtype: torch.dtype, seed: int) -> BenchResult:
    cfg = CSAConfig(m=m, k=k, n_h=4, n_h_i=2, d_c=64, c=32, c_i=16)
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = torch.empty(n, d).normal_(generator=g).to(device=device, dtype=dtype)
    p = random_params(cfg=cfg, d=d, device=device, dtype=dtype, generator=g)

    w_kv_dense = torch.empty(d, cfg.c, device=device, dtype=dtype).normal_(std=0.02, generator=g)
    w_q_dense = torch.empty(d, cfg.n_h * cfg.c, device=device, dtype=dtype).normal_(std=0.02, generator=g)

    csa_out = csa_reference(h=h, cfg=cfg, p=p)["o"]
    dense_out = _dense_mqa_forward(h=h, w_kv=w_kv_dense, w_q=w_q_dense, n_h=cfg.n_h, c=cfg.c)

    cos = torch.nn.functional.cosine_similarity(
        csa_out.reshape(n, -1),
        dense_out.reshape(n, -1),
        dim=-1,
    )

    mem_csa = _kv_bytes_csa(n=n, m=cfg.m, c=cfg.c, c_i=cfg.c_i, dtype=dtype)
    mem_dense = _kv_bytes_dense(n=n, c=cfg.c, dtype=dtype)

    csa_dt = _time_forward(lambda: csa_reference(h=h, cfg=cfg, p=p), device=device)
    dense_dt = _time_forward(
        lambda: _dense_mqa_forward(h=h, w_kv=w_kv_dense, w_q=w_q_dense, n_h=cfg.n_h, c=cfg.c),
        device=device,
    )

    return BenchResult(
        n=n,
        d=d,
        m=m,
        k=k,
        n_h=cfg.n_h,
        c=cfg.c,
        mem_ratio=mem_csa / mem_dense,
        csa_tok_s=n / csa_dt,
        dense_tok_s=n / dense_dt,
        cos_sim_mean=float(cos.mean().item()),
        cos_sim_p10=float(cos.kthvalue(max(1, n // 10)).values.item()),
    )


def _format(r: BenchResult) -> str:
    return (
        f"shape: n={r.n} d={r.d} | csa: m={r.m} k={r.k} n_h={r.n_h} c={r.c}\n"
        f"  mem_ratio (csa / dense KV) = {r.mem_ratio:.4f}\n"
        f"  tok/s   csa = {r.csa_tok_s:9.1f}   dense = {r.dense_tok_s:9.1f}   "
        f"speedup = {r.csa_tok_s / r.dense_tok_s:.3f}x\n"
        f"  cos_sim (mean / p10) = {r.cos_sim_mean:+.4f} / {r.cos_sim_p10:+.4f}\n"
        f"  note: cos_sim is meaningful only when CSA is bolted onto pretrained weights\n"
        f"        (spike gate, CLAUDE.md). With random init this is essentially noise.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="CSA vs. dense MQA: mem, tok/s, cos sim.")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    result = run(n=args.n, d=args.d, m=args.m, k=args.k, device=device, dtype=dtype, seed=args.seed)
    print(_format(result))


if __name__ == "__main__":
    main()
