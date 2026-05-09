# csa-cutile

DeepSeek-V4's **Compressed Sparse Attention (CSA)** — isolated from the V4
inference bundle, implemented as a drop-in attention module, and (eventually)
shipped as fused Triton kernels and an NVIDIA cuTile port.

> Status: **scoping**. The pure-PyTorch reference (`src/csa/reference.py`,
> derived line-by-line from V4 paper §2.3.1, eqs. 9–19) is in place and
> validated against structural invariants. Kernels and the spike-gate
> end-to-end run come next. See [CLAUDE.md](CLAUDE.md) for the full plan and
> the project's strategic frame.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # or: pip install -r requirements.txt
pytest                           # 9 tests, ~50 ms
python bench/bench_attention.py --n 1024 --d 256 --m 16 --k 8
```

## Layout

```
src/csa/
  reference.py        # CSA from V4 §2.3.1 (eqs. 9–19) — the test oracle
  __init__.py         # public API: CSAConfig, CSAParams, csa_reference, random_params
tests/
  test_reference.py   # shape, causality, top-k, dense-equivalence invariants
  fixtures/README.md  # capture protocol for DeepSeek-inference fixtures (TODO)
bench/
  bench_attention.py  # the only three numbers: mem_ratio, tok/s, cos_sim
docs/
  derivation.md       # paper-to-code map for eqs. 9–19
```

The Triton (`triton_kernels.py`) and cuTile (`cutile_kernels.py`) modules will
be added once `reference.py` matches DeepSeek-inference fixtures bitwise — see
[CLAUDE.md](CLAUDE.md) "Testing pyramid" and "What NOT to do".

## References

- DeepSeek-V4 paper: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf>
- DeepSeek-V4 inference impl (the source we extract CSA from): <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference>
- Project plan and conventions: [CLAUDE.md](CLAUDE.md)
- Math walkthrough: [docs/derivation.md](docs/derivation.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
