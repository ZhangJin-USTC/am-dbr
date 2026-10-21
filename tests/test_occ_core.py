import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework"))

from core import Step  # noqa: E402
from methods import ExactRankOCC, OCC  # noqa: E402


def step(t, u):
    return Step(t=t, u=float(u), y=float(u))


def test_adaptive_state_is_fixed_memory_and_monotone():
    method = OCC(
        adaptive=True,
        bins=16,
        lam_min=0.80,
        lam_max=0.99,
        window=20,
        lambda_path_capacity=7,
    )
    rng = np.random.default_rng(7)
    for t, u in enumerate(rng.uniform(size=10_000)):
        method.predict_g(np.linspace(0.0, 1.0, 31))
        method.update(step(t, u))

    assert not hasattr(method, "_U")
    assert method._w.shape == (16,)
    assert len(method._recent) <= 20
    assert method.lambda_path is not None
    assert len(method.lambda_path) <= 7
    grid = np.linspace(0.0, 1.0, 101)
    g = method.predict_g(grid)
    assert np.all(np.diff(g) >= -1e-12)
    assert np.all((0.0 <= g) & (g <= 1.0))


def test_adaptive_rank_uses_recursive_history_not_current_lambda_power():
    class SequenceOCC(OCC):
        def __init__(self, sequence):
            super().__init__(adaptive=False, bins=10, lam=0.5)
            self.sequence = iter(sequence)

        def _lambda(self, u):  # noqa: ARG002
            return float(next(self.sequence))

    method = SequenceOCC([0.5, 0.9, 0.2])
    method.update(step(0, 0.10))
    method.update(step(1, 0.80))
    # Before the third update, the two bin masses are 0.9 and 1.0.  The
    # current 0.2 must not be used to reconstruct their historical ages.
    expected = 0.9 / (0.9 + 1.0)
    method.update(step(2, 0.20))
    bin_index = int(0.20 * method.B)
    assert np.isclose(method._s[bin_index], expected, atol=1e-12)


def test_fixed_rank_matches_binned_fractional_ecdf():
    method = OCC(adaptive=False, bins=10, lam=0.8)
    method.update(step(0, 0.15))
    method.update(step(1, 0.55))
    # The historical mass is 0.8 for the first point and 1 for the second.
    # At u=.30, the first bin is fully below and the second is above.
    expected = 0.8 / (0.8 + 1.0)
    assert np.isclose(method._rank(0.30), expected, atol=1e-12)


def test_exact_rank_reference_uses_product_weights():
    class SequenceExact(ExactRankOCC):
        def __init__(self, sequence):
            super().__init__(adaptive=False, bins=10, lam=0.5)
            self.sequence = iter(sequence)

        def _lambda(self, u):  # noqa: ARG002
            return float(next(self.sequence))

    method = SequenceExact([0.5, 0.9, 0.2])
    method.update(step(0, 0.10))
    method.update(step(1, 0.80))
    assert np.allclose(method._reference_w, [0.9, 1.0])
    expected = 0.9 / 1.9
    method.update(step(2, 0.20))
    assert np.isclose(method._s[2], expected)
    assert np.allclose(method._reference_w, [0.18, 0.2, 1.0])


def test_occ_interval_uses_piecewise_linear_generalized_inverse():
    method = OCC(lam=0.98, bins=8)
    method._g_cache = lambda u: np.interp(
        np.atleast_1d(u),
        [0.0, 0.25, 0.75, 1.0],
        [0.0, 0.2, 0.8, 1.0],
    )
    method._knots_x = np.array([0.0, 0.25, 0.75, 1.0])
    method._knots_y = np.array([0.0, 0.2, 0.8, 1.0])
    lower, upper = method.predict_interval(lambda p: p, alpha=0.2)
    assert np.isclose(lower, 0.125)
    assert np.isclose(upper, 0.875)


def test_exact_rank_matches_direct_products_random_sequence():
    class SequenceExact(ExactRankOCC):
        def __init__(self, sequence):
            super().__init__(adaptive=False, bins=12, lam=0.9)
            self.sequence = iter(sequence)

        def _lambda(self, u):  # noqa: ARG002
            return float(next(self.sequence))

    rng = np.random.default_rng(20260807)
    values = rng.uniform(size=200)
    lambdas = rng.uniform(0.75, 0.999, size=200)
    method = SequenceExact(lambdas)
    past_values = []
    past_weights = []
    for t, (u, lam) in enumerate(zip(values, lambdas, strict=True)):
        if past_values:
            expected = np.average(np.asarray(past_values) <= u, weights=past_weights)
            assert np.isclose(method._exact_rank(float(u)), expected, atol=1e-14)
        method.update(step(t, u))
        past_weights = [float(lam) * weight for weight in past_weights] + [1.0]
        past_values.append(float(u))
        assert np.allclose(method._reference_w, past_weights, atol=1e-14)


def test_binned_rank_error_is_bounded_by_active_bin_mass():
    rng = np.random.default_rng(17)
    binned = OCC(adaptive=False, bins=24, lam=0.97)
    exact = ExactRankOCC(adaptive=False, bins=24, lam=0.97)
    for t, u in enumerate(rng.beta(0.7, 1.3, size=2_000)):
        total = float(binned._w.sum())
        if total > 0:
            bin_index = min(binned.B - 1, int(float(u) * binned.B))
            active_mass = float(binned._w[bin_index] / total)
            error = abs(binned._rank(float(u)) - exact._exact_rank(float(u)))
            assert error <= active_mass + 1e-12
        current = step(t, u)
        binned.update(current)
        exact.update(current)


def test_generalized_inverse_returns_left_edge_of_plateau():
    method = OCC(lam=0.98, bins=8)
    method._knots_x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    method._knots_y = np.array([0.0, 0.20, 0.20, 0.20, 1.0])
    method._g_cache = lambda u: np.interp(
        np.atleast_1d(u), method._knots_x, method._knots_y
    )
    assert np.isclose(method._inverse_g(0.20), 0.25)
    assert method._inverse_g(0.2000001) > 0.75


def test_fixed_state_dimensions_do_not_grow_with_horizon():
    method = OCC(adaptive=False, bins=20, lam=0.985)
    rng = np.random.default_rng(81)
    for t, u in enumerate(rng.uniform(size=50_000)):
        method.update(step(t, u))
    assert method._w.shape == (20,)
    assert method._s.shape == (20,)
    assert len(method._recent) <= 1
    assert method.lambda_path is not None and len(method.lambda_path) <= 1


def test_occ_rejects_invalid_public_parameters():
    with pytest.raises(ValueError):
        OCC(bins=3)
    with pytest.raises(ValueError):
        OCC(beta=-1.0)
    with pytest.raises(ValueError):
        OCC(tau=1.1)
    with pytest.raises(ValueError):
        OCC(adaptive=True, window=5)
    with pytest.raises(ValueError):
        ExactRankOCC(reference_capacity=0)
    method = OCC(bins=8)
    with pytest.raises(ValueError):
        method.predict_interval(lambda p: p, alpha=0.0)
