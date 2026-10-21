import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework"))

from core import Step  # noqa: E402
from experiment import run_one_method  # noqa: E402
from methods import (  # noqa: E402
    AdaptiveMemoryDBR,
    pit_crps_piecewise_linear,
)


def identity_step(t: int, u: float) -> Step:
    return Step(t=t, u=float(u), y=float(u), F_t_inv=lambda p: float(p))


def test_piecewise_linear_pit_crps_matches_dense_quadrature():
    edges = np.linspace(0.0, 1.0, 13)
    cdf_edges = np.array(
        [0.0, 0.01, 0.04, 0.10, 0.18, 0.29, 0.42, 0.56, 0.70, 0.82, 0.91, 0.97, 1.0]
    )
    grid = np.linspace(0.0, 1.0, 500_001)
    for outcome in (0.0, 0.137, 0.5, 0.903, 1.0):
        predicted = np.interp(grid, edges, cdf_edges)
        target = (grid >= outcome).astype(float)
        numerical = np.trapezoid((predicted - target) ** 2, grid)
        exact = pit_crps_piecewise_linear(edges, cdf_edges, outcome)
        assert np.isclose(exact, numerical, atol=3e-6)


def test_adaptive_memory_map_is_coherent_and_state_is_horizon_independent():
    method = AdaptiveMemoryDBR(
        lambdas=(0.85, 0.95, 0.99, 1.0), bins=24, share=0.02
    )
    rng = np.random.default_rng(20260809)
    for t, u in enumerate(rng.beta(0.7, 1.4, size=20_000)):
        method.update(identity_step(t, u))

    grid = np.linspace(0.0, 1.0, 501)
    mapped = method.predict_g(grid)
    assert mapped[0] == 0.0 and mapped[-1] == 1.0
    assert np.all(np.diff(mapped) >= -1e-12)
    assert np.all((0.0 <= mapped) & (mapped <= 1.0))
    assert len(method._experts) == 4
    assert all(expert._mass.shape == (24,) for expert in method._experts)
    assert method._weights.shape == (5,)
    assert not hasattr(method, "loss_path")


def test_weight_update_is_post_outcome_and_fixed_share_stays_positive():
    method = AdaptiveMemoryDBR(lambdas=(0.8, 0.99), bins=16, share=0.03)
    before = method.predict_g(np.linspace(0.0, 1.0, 101)).copy()
    initial_weights = method._weights.copy()
    outcome = 0.17
    method.update(identity_step(0, outcome))

    identity_crps = outcome**3 / 3.0 + (1.0 - outcome) ** 3 / 3.0
    assert np.allclose(method.last_expert_losses, identity_crps)
    assert np.allclose(method._weights, initial_weights)
    after = method.predict_g(np.linspace(0.0, 1.0, 101))
    assert not np.allclose(after, before)
    assert np.isclose(np.sum(method._weights), 1.0)
    assert np.all(method._weights >= method.share / method._weights.size - 1e-15)


def test_crps_aggregate_satisfies_one_step_mixability_inequality():
    method = AdaptiveMemoryDBR(
        lambdas=(0.75, 0.92, 0.99, 1.0), bins=30, eta=0.5, share=0.01
    )
    rng = np.random.default_rng(73)
    stream = np.concatenate([rng.beta(2, 8, 150), rng.beta(8, 2, 150)])
    for t, u in enumerate(stream):
        weights = method._weights.copy()
        method.update(identity_step(t, u))
        lhs = np.exp(-method.eta * method.last_aggregate_loss)
        rhs = np.sum(weights * np.exp(-method.eta * method.last_expert_losses))
        assert lhs + 1e-13 >= rhs


def test_fixed_share_pathwise_switching_oracle_bound():
    method = AdaptiveMemoryDBR(
        lambdas=(0.70, 0.95, 1.0),
        bins=20,
        eta=0.5,
        share=0.02,
        include_identity=True,
    )
    rng = np.random.default_rng(91)
    stream = np.concatenate(
        [rng.uniform(size=120), rng.beta(10, 2, size=120), rng.uniform(size=120)]
    )
    losses = []
    for t, u in enumerate(stream):
        method.update(identity_step(t, u))
        losses.append(method.last_expert_losses.copy())
    loss_matrix = np.asarray(losses)
    path = np.r_[np.full(120, 3), np.full(120, 0), np.full(120, 3)]
    comparator = float(np.sum(loss_matrix[np.arange(stream.size), path]))
    penalty = method.switching_oracle_penalty(stream.size, switches=2)
    assert method.cumulative_aggregate_loss <= comparator + penalty + 1e-10


def test_adaptive_memory_parameters_are_validated():
    with pytest.raises(ValueError):
        AdaptiveMemoryDBR(lambdas=())
    with pytest.raises(ValueError):
        AdaptiveMemoryDBR(lambdas=(0.9, 0.9))
    with pytest.raises(ValueError):
        AdaptiveMemoryDBR(lambdas=(0.9,), eta=0.51)
    with pytest.raises(ValueError):
        AdaptiveMemoryDBR(lambdas=(0.9,), share=1.0)
    method = AdaptiveMemoryDBR(lambdas=(0.9,), bins=8)
    with pytest.raises(ValueError):
        method.predict_interval(lambda p: p, alpha=0.0)
    with pytest.raises(ValueError):
        method.switching_oracle_penalty(horizon=2, switches=2)


def test_experiment_records_prequential_pit_crps():
    method = AdaptiveMemoryDBR(lambdas=(0.8, 0.99), bins=20, share=0.01)
    online = [identity_step(t, u) for t, u in enumerate((0.2, 0.7, 0.4))]
    result = run_one_method(method, warmup=[], online=online, alphas=(0.1,))
    assert result.pit_crps is not None and result.pit_crps.shape == (3,)
    first_exact = 0.2**3 / 3.0 + 0.8**3 / 3.0
    assert np.isclose(result.pit_crps[0], first_exact, atol=1e-3)
