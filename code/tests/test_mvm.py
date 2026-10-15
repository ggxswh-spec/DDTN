"""MVM 单元测试：保证附录 G 声称的数值结果可复现。"""

from __future__ import annotations

import numpy as np
import pytest

from mvm import bell_chsh as g2
from mvm import nielsen_ninomiya as g3
from mvm import photon_propagation as g1


# ---------------------------------------------------------------------------
# G.1 光子传播
# ---------------------------------------------------------------------------
def test_wavefront_speed_is_one():
    N = 200
    x = np.arange(N)
    u0 = np.exp(-((x - 50) ** 2) / (2.0 * 6**2))
    u1 = np.roll(u0, 1)
    hist = g1.leapfrog_1d_ring(u0, u1, n_steps=120, c2=1.0)
    c_meas, r2 = g1.measure_wavefront_speed(hist)
    assert abs(c_meas - 1.0) < 1e-6
    assert r2 > 0.9999


def test_dispersion_matches_analytic():
    N = 128
    x = np.arange(N)
    u0 = np.exp(-((x - 32) ** 2) / (2.0 * 5**2))
    u1 = np.roll(u0, 1)
    hist = g1.leapfrog_1d_ring(u0, u1, n_steps=256, c2=1.0)
    _, _, max_err = g1.extract_dispersion_1d(hist, n_modes=32)
    assert max_err < 1e-3


# ---------------------------------------------------------------------------
# G.2 Bell--CHSH
# ---------------------------------------------------------------------------
def test_chsh_reaches_tsirelson():
    S, E = g2.chsh_value()
    assert abs(S - 2 * np.sqrt(2)) < 1e-9
    assert abs(g2.chsh_max_over_signs(E) - 2 * np.sqrt(2)) < 1e-9


def test_no_signaling_marginals():
    rng = np.random.default_rng(0)
    _, _, margin_err = g2.sample_joint(
        (0.0, np.pi / 2), (np.pi / 4, -np.pi / 4), 50_000, rng
    )
    assert margin_err < 0.01


# ---------------------------------------------------------------------------
# G.3 Nielsen--Ninomiya
# ---------------------------------------------------------------------------
def test_naive_doubling_two_zeros():
    zeros, net = g3.count_zero_modes(g3.naive_hamiltonian, n_k=8192)
    # 至少 2 个不同零点（k=0 与 k=±π），净手性为 0
    assert len(zeros) >= 2
    assert abs(net) < 1e-9


def test_wilson_gaps_edge_mode():
    k = np.linspace(-np.pi, np.pi, 1000)
    ev = g3.band_spectrum(lambda kk: g3.wilson_hamiltonian(kk, r=1.0), k)
    # k=π（布里渊区边缘）处的能隙应为 2r = 2.0
    gap_at_pi = float(np.min(np.abs(ev[:, -1])))
    assert gap_at_pi > 1.5


def test_domain_wall_single_zero_mode():
    res = g3.analyze_domain_wall(N=64, m0=1.0, xi=0.5)
    assert abs(res["E0"]) < 1e-8
    assert res["n_zero_modes"] >= 1
    assert res["width"] < 10.0
