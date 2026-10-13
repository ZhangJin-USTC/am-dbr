import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework"))

from core import Step  # noqa: E402
from experimental_methods import (  # noqa: E402
    BinnedWeightedPITECDF,
    OCCCoverageGuard,
    OCCShortestInterval,
)


def identity_step(t, u):
    return Step(t=t, u=float(u), y=float(u), F_t_inv=lambda p: float(p))


def test_binned_weighted_ecdf_matches_histogram_cdf_and_stays_bounded():
    method = BinnedWeightedPITECDF(lam=1.0, bins=4)
    for t, u in enumerate([0.10, 0.20, 0.60, 0.90]):
        method.update(identity_step(t, u))
    assert np.allclose(method.predict_g(np.array([0.0, 0.25, 0.50, 0.75, 1.0])),
                       [0.0, 0.5, 0.5, 0.75, 1.0])

    rng = np.random.default_rng(19)
    for t, u in enumerate(rng.uniform(size=20_000), start=4):
        method.update(identity_step(t, u))
    assert method._mass.shape == (4,)
    grid = np.linspace(0.0, 1.0, 501)
    mapped = method.predict_g(grid)
    assert np.all(np.diff(mapped) >= -1e-12)
    assert mapped[0] == 0.0 and mapped[-1] == 1.0


def test_binned_weighted_ecdf_inverse_produces_ordered_interval():
    method = BinnedWeightedPITECDF(lam=0.95, bins=20)
    for t, u in enumerate(np.linspace(0.15, 0.85, 100)):
        method.update(identity_step(t, u))
    lower, upper = method.predict_interval(lambda p: p, alpha=0.1)
    assert 0.0 < lower <= upper < 1.0
    with pytest.raises(ValueError):
        method.predict_interval(lambda p: p, alpha=1.0)


def test_adaptive_binned_ecdf_has_bounded_state_and_refreshes_lambda():
    method = BinnedWeightedPITECDF(
        bins=16,
        adaptive=True,
        lam_min=0.90,
        lam_max=0.999,
        tau=0.10,
        window=20,
        adapt_every=2,
    )
    stream = np.concatenate([np.linspace(0.05, 0.25, 30), np.linspace(0.75, 0.95, 30)])
    for t, u in enumerate(stream):
        method.update(identity_step(t, u))
    assert method._mass.shape == (16,)
    assert len(method._recent) == 20
    assert method.lam_min <= method._current_lam <= method.lam_max
    assert method._current_lam < method.lam_max
    assert method.name == "DBR-Adaptive"


def test_occ_guard_uses_preupdate_calibrated_pit_scores():
    method = OCCCoverageGuard(lam=0.98, bins=12, score_window=3)
    method.occ._knots_x = np.array([0.0, 1.0])
    method.occ._knots_y = np.array([0.0, 1.0])
    method.occ._g_cache = lambda u: np.asarray(u, dtype=float)
    method.update(identity_step(0, 0.2))
    assert np.allclose(list(method._scores), [0.3])


def test_occ_guard_conformal_rank_and_state_are_bounded():
    method = OCCCoverageGuard(lam=0.985, bins=16, score_window=20)
    rng = np.random.default_rng(31)
    for t, u in enumerate(rng.beta(2.0, 1.5, size=5_000)):
        method.update(identity_step(t, u))
    assert len(method._scores) == 20
    assert method.occ._w.shape == (16,)
    lower, upper = method.predict_interval(lambda p: p, alpha=0.1)
    assert 0.0 < lower <= upper < 1.0
    grid = np.linspace(0.0, 1.0, 101)
    assert np.all(np.diff(method.predict_g(grid)) >= -1e-12)


def test_occ_shortest_interval_is_no_wider_than_equal_tailed_interval():
    method = OCCShortestInterval(lam=0.985, bins=20, tail_grid_size=41)
    method.occ._knots_x = np.array([0.0, 0.25, 0.60, 1.0])
    method.occ._knots_y = np.array([0.0, 0.10, 0.80, 1.0])
    method.occ._g_cache = lambda u: np.interp(
        np.asarray(u, dtype=float), method.occ._knots_x, method.occ._knots_y
    )
    equal_tailed = method.occ.predict_interval(lambda p: p, alpha=0.1)
    shortest = method.predict_interval(lambda p: p, alpha=0.1)
    assert shortest[0] <= shortest[1]
    assert shortest[1] - shortest[0] <= equal_tailed[1] - equal_tailed[0] + 1e-12


def test_occ_shortest_interval_uses_no_growing_state():
    method = OCCShortestInterval(lam=0.98, bins=12, tail_grid_size=11)
    rng = np.random.default_rng(48)
    for t, u in enumerate(rng.uniform(size=10_000)):
        method.update(identity_step(t, u))
    assert method.occ._w.shape == (12,)
    assert not hasattr(method, "_scores")
