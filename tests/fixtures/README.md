# Fixtures

Captured tensors from DeepSeek-V4's reference inference, used to validate
`src/csa/reference.py` bitwise (per CLAUDE.md, the testing pyramid).

## Capture protocol (TODO: stub)

1. Clone the open-source DeepSeek-V4 inference impl
   (https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference).
2. Run a CSA layer at small scale (e.g. n=128, d_model=64, m=4, k=4) on a fixed
   random hidden-state tensor with a fixed seed.
3. Save the following tensors to `tests/fixtures/csa_<config-tag>.pt`:
   - `h`              — input hidden states `[n, d]`
   - `params`         — projection/bias tensors as a flat dict matching `CSAParams`
   - `c_comp`         — compressed KV entries `[n_blk, c]`
   - `k_i_comp`       — compressed indexer keys `[n_blk, c_i]`
   - `i_scores`       — index scores `[n, n_blk]`
   - `topk_idx`       — selected block indices `[n, k]`
   - `o`              — core attention outputs `[n, n_h, c]`
4. Add a corresponding test in `tests/test_reference.py` that loads the fixture
   and asserts `csa_reference(...)` matches each tensor to <1e-6 in FP32.

When fixtures land here, also add `*.pt` exclusions to `.gitignore` if they
exceed the size budget.
