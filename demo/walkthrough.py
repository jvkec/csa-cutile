"""v0 demo: run CSA on a small input and write plots to demo/output/.

Produces five PNGs:
  01_input_hidden_states.png    [n, d] heatmap of the input
  02_compressed_kv.png          [n_blk, c] heatmap after eqs. 11-12
  03_indexer_scores.png         [n, n_blk] indexer scores (eq. 16, with causal mask)
  04_sparse_attention_mask.png  [n, n_blk] which blocks each token actually attended to
  05_kv_cache_scaling.png       KV bytes vs sequence length, CSA vs dense MQA

And prints a one-line summary of the three numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from csa import CSAConfig, csa_reference, random_params


def _make_inputs(*, n: int, d: int, cfg: CSAConfig, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = torch.empty(n, d).normal_(generator=g)
    p = random_params(cfg=cfg, d=d, generator=g)
    return h, p


def _plot_heatmap(ax, data: np.ndarray, *, title: str, xlabel: str, ylabel: str, cmap: str = "magma"):
    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


def _save(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def render_plots(*, h, cfg: CSAConfig, out_dir: Path) -> dict:
    p = random_params(cfg=cfg, d=h.shape[1], generator=torch.Generator().manual_seed(0))
    out = csa_reference(h=h, cfg=cfg, p=p)
    n = h.shape[0]
    n_blk = n // cfg.m

    fig, ax = plt.subplots(figsize=(8, 4))
    im = _plot_heatmap(ax, h.numpy(), title=f"Input hidden states  H ∈ R^[{n}, {h.shape[1]}]",
                       xlabel="hidden dim", ylabel="token index")
    fig.colorbar(im, ax=ax, label="value")
    _save(fig, out_dir / "01_input_hidden_states.png")

    fig, ax = plt.subplots(figsize=(8, 3))
    im = _plot_heatmap(ax, out["c_comp"].numpy(),
                       title=f"Compressed KV  C^Comp ∈ R^[{n_blk}, {cfg.c}]   (m={cfg.m}, {n}→{n_blk} entries)",
                       xlabel="head dim c", ylabel="block index")
    fig.colorbar(im, ax=ax, label="value")
    _save(fig, out_dir / "02_compressed_kv.png")

    scores = out["i_scores"].numpy()
    finite_mask = np.isfinite(scores)
    display = np.where(finite_mask, scores, np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(display, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title(f"Lightning indexer scores  I ∈ R^[{n}, {n_blk}]   (white = causally masked)")
    ax.set_xlabel("compressed block s")
    ax.set_ylabel("query token t")
    fig.colorbar(im, ax=ax, label="score")
    _save(fig, out_dir / "03_indexer_scores.png")

    topk_idx = out["topk_idx"]
    mask = np.zeros((n, n_blk), dtype=bool)
    for t in range(n):
        for s in topk_idx[t].tolist():
            if s >= 0:
                mask[t, s] = True
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(mask, aspect="auto", cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(f"Sparse attention pattern  (k={cfg.k} of {n_blk} blocks per token)")
    ax.set_xlabel("compressed block s")
    ax.set_ylabel("query token t")
    _save(fig, out_dir / "04_sparse_attention_mask.png")

    sizes = np.array([64, 128, 256, 512, 1024, 2048, 4096], dtype=np.int64)
    bytes_per = 4
    csa_bytes = (sizes // cfg.m) * (cfg.c + cfg.c_i) * bytes_per
    dense_bytes = sizes * cfg.c * bytes_per
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, dense_bytes / 1024, "o-", label=f"dense MQA  (c={cfg.c})")
    ax.plot(sizes, csa_bytes / 1024, "s-", label=f"CSA  (m={cfg.m}, c={cfg.c}, c_i={cfg.c_i})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("sequence length n")
    ax.set_ylabel("KV cache per layer (KiB, fp32)")
    ax.set_title("KV cache scaling")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out_dir / "05_kv_cache_scaling.png")

    return {
        "n": n,
        "n_blk": n_blk,
        "selected_per_token": int(mask.sum() // n) if n else 0,
        "kv_ratio_at_n": float(csa_bytes[np.argmin(np.abs(sizes - n))] / dense_bytes[np.argmin(np.abs(sizes - n))]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    args = ap.parse_args()

    cfg = CSAConfig(m=args.m, k=args.k, n_h=4, n_h_i=2, d_c=32, c=16, c_i=8)
    h, _ = _make_inputs(n=args.n, d=args.d, cfg=cfg, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"CSA demo  n={args.n}  d={args.d}  m={args.m}  k={args.k}")
    print(f"  blocks={args.n // args.m}  c={cfg.c}  c_i={cfg.c_i}  n_h={cfg.n_h}")
    print()
    summary = render_plots(h=h, cfg=cfg, out_dir=args.out)
    print()
    print(f"  KV ratio at n={args.n}: CSA / dense = {summary['kv_ratio_at_n']:.3f}")
    print(f"  Avg selected blocks per token: {summary['selected_per_token']} (cap = k = {args.k})")


if __name__ == "__main__":
    main()
