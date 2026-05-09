"""Shared pytest fixtures for the CSA reference tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from csa import CSAConfig, CSAParams, random_params


@dataclass(frozen=True)
class CSACase:
    n: int
    d: int
    cfg: CSAConfig
    h: torch.Tensor
    p: CSAParams


def _make_case(*, n: int, d: int, m: int, k: int, seed: int) -> CSACase:
    cfg = CSAConfig(
        m=m,
        k=k,
        n_h=4,
        n_h_i=2,
        d_c=16,
        c=8,
        c_i=4,
    )
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = torch.empty(n, d).normal_(generator=g)
    p = random_params(cfg=cfg, d=d, generator=g)
    return CSACase(n=n, d=d, cfg=cfg, h=h, p=p)


@pytest.fixture
def small_case() -> CSACase:
    return _make_case(n=32, d=24, m=4, k=2, seed=0)


@pytest.fixture
def k_equals_blocks_case() -> CSACase:
    return _make_case(n=32, d=24, m=4, k=8, seed=1)


@pytest.fixture
def m_equals_one_case() -> CSACase:
    return _make_case(n=16, d=16, m=1, k=16, seed=2)
