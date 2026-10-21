import copy
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework"))

from core import Step  # noqa: E402
from experiment import _run_interval_only_multi, run_one_method  # noqa: E402
from methods import DCP, RollingPITECDF  # noqa: E402
from residual_baselines import (  # noqa: E402
    _OrderStatisticTreap,
    ResidualACI,
    ResidualFACI,
    ResidualNExCP,
    ResidualPID,
    ResidualSAOCP,
    ResidualSFOGD,
)


def inv(p):
    p = np.asarray(p, dtype=float)
    value = norm.ppf(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(value) if value.ndim == 0 else value


def step(t, u):
    y = float(norm.ppf(u))
    return Step(t=t, u=float(u), y=y, F_t_inv=inv)


def test_dcp_uses_absolute_pit_score_and_conformal_order_statistic():
    method = DCP(K=20)
    for t, u in enumerate([0.1, 0.5, 0.9]):
        method.update(step(t, u))
    assert np.allclose(list(method._scores), [0.4, 0.0, 0.4])
    lo, hi = method.predict_interval(inv, alpha=0.25)
    assert np.isclose(lo, inv(0.1))
    assert np.isclose(hi, inv(0.9))


def test_rolling_pit_ecdf_matches_manual_ecdf():
    method = RollingPITECDF(K=3)
    for t, u in enumerate([0.2, 0.7, 0.4, 0.9]):
        method.update(step(t, u))
    grid = np.array([0.1, 0.4, 0.8, 1.0])
    # The K=3 state is [0.7, 0.4, 0.9].
    assert np.allclose(method.predict_g(grid), [0.0, 1 / 3, 2 / 3, 1.0])


def test_sfogd_matches_reference_one_step_recurrence():
    method = ResidualSFOGD(alpha_target=0.1, max_scale=np.sqrt(3.0))
    method._update_score(2.0)
    # grad=-coverage=-.9, grad_norm=.81 and scale/sqrt(3)=1.
    assert np.isclose(method.delta, 1.0)
    method._update_score(0.5)
    expected = 1.0 - 0.1 / np.sqrt(0.9**2 + 0.1**2)
    assert np.isclose(method.delta, expected)


def test_faci_and_saocp_are_independent_per_alpha():
    dataset = {
        "warmup": [step(t, u) for t, u in enumerate(np.linspace(0.1, 0.9, 30))],
        "pre_online": [],
        "online": [step(30 + t, u) for t, u in enumerate(np.linspace(0.05, 0.95, 20))],
    }
    for template in [ResidualFACI(), ResidualSAOCP(max_scale=2.0)]:
        before = copy.deepcopy(template)
        result = _run_interval_only_multi(template, dataset, (0.1, 0.2))
        assert set(result.intervals_by_alpha) == {0.1, 0.2}
        assert template.alpha_target == before.alpha_target
        if isinstance(template, ResidualFACI):
            assert np.array_equal(template.alphas, before.alphas)
            assert len(template.scores) == len(before.scores) == 0
        else:
            assert template.t == before.t and template.experts == before.experts
        for lo, hi in result.intervals_by_alpha.values():
            assert np.all(lo <= hi)


def test_interval_runner_calls_method_warmup_before_pre_online():
    class TrackingSFOGD(ResidualSFOGD):
        observed_scales = []

        def warmup(self, warmup_steps):
            super().warmup(warmup_steps)
            type(self).observed_scales.append(self.scale)

    warmup = [step(0, 0.1), step(1, 0.5), step(2, 0.95)]
    dataset = {
        "warmup": warmup,
        "pre_online": [step(3, 0.2)],
        "online": [step(4, 0.8)],
    }
    TrackingSFOGD.observed_scales.clear()
    _run_interval_only_multi(TrackingSFOGD(), dataset, (0.1,))
    expected = max(abs(s.y - inv(0.5)) for s in warmup) * np.sqrt(3.0)
    assert np.allclose(TrackingSFOGD.observed_scales, [expected])


def test_nexcp_matches_reference_weighted_quantile_and_buffer():
    method = ResidualNExCP(alpha_target=0.5, gamma=0.5, max_history=3)
    warmup = [
        Step(t=0, u=0.5, y=0.5, F_t_inv=lambda p: 0.0),
        Step(t=1, u=0.5, y=1.0, F_t_inv=lambda p: 0.0),
        Step(t=2, u=0.5, y=2.0, F_t_inv=lambda p: 0.0),
    ]
    method.warmup(warmup)
    # Scores 0.5, 1, 2 have chronological weights .25, .5, 1; the weighted
    # inverted-CDF median is therefore 2.
    assert np.allclose(method.predict_interval(lambda p: 0.0, alpha=0.5), (-2.0, 2.0))
    method.update(Step(t=3, u=0.5, y=0.25, F_t_inv=lambda p: 0.0))
    assert len(method.scores) == 3
    assert np.allclose(list(method.scores), [1.0, 2.0, 0.25])


def test_pid_residual_is_finite_and_bounded_state():
    method = ResidualPID(alpha_target=0.1, burnin_window=5)
    values = np.linspace(0.05, 0.95, 30)
    method.warmup([step(t, float(u)) for t, u in enumerate(values)])
    lower, upper = method.predict_interval(inv, alpha=0.1)
    assert np.isfinite([lower, upper]).all()
    assert lower <= upper
    assert len(method._scores) == 5
    assert method.name == "PID-residual-projected"
    before = (method._radius, method._q_state, method._error_sum)
    method.update(step(30, 0.9))
    after = (method._radius, method._q_state, method._error_sum)
    assert before != after


def test_projected_pid_binds_at_zero_and_prediction_uses_projected_state():
    method = ResidualPID(
        alpha_target=0.1,
        learning_rate=0.1,
        integral_gain=0.0,
        burnin_window=5,
    )
    method._update_score(0.0)
    assert np.isclose(method._q_state, -0.01)
    assert method._radius == 0.0
    assert np.allclose(method.predict_interval(lambda p: 3.0, alpha=0.1), (3.0, 3.0))
    method._update_score(1.0)
    previous_q = method._q_state
    method._update_score(1.0)
    assert method._q_state > previous_q
    assert method._radius >= 0.0


def test_faci_treap_integration_matches_list_reference_trajectory():
    method = ResidualFACI(alpha_target=0.2, gammas=(0.01, 0.02), I=20)
    scores = []
    alphas = np.full(2, 0.2)
    log_w = np.zeros(2)
    sigma = 1.0 / 40.0
    eta = method.eta
    stream = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.5, 7.0, 1.5]

    for score_value in stream:
        if len(scores) > 5:
            beta = float(np.mean(np.asarray(scores) >= score_value))
            losses = np.maximum(
                0.2 * (beta - alphas),
                0.8 * (alphas - beta),
            )
            wbar = log_w - eta * losses
            shared = logsumexp(wbar)
            log_w = np.logaddexp(
                np.log1p(-sigma) + wbar,
                np.log(sigma / len(alphas)) + shared,
            )
            log_w -= logsumexp(log_w)
            alphas = np.clip(
                alphas + np.asarray([0.01, 0.02]) * (0.2 - (alphas > beta)),
                0.0,
                1.0,
            )
        scores.append(score_value)
        method._update_score(score_value)
        assert np.allclose(method.alphas, alphas)
        assert np.allclose(method.log_w, log_w)
        alpha_bar = float(np.dot(np.exp(log_w - logsumexp(log_w)), alphas))
        expected_radius = np.quantile(scores, 1.0 - alpha_bar, method="inverted_cdf")
        assert np.allclose(method.predict_interval(lambda p: 0.0, 0.2), (-expected_radius, expected_radius))


def test_saocp_expert_creation_and_expiry_trace():
    method = ResidualSAOCP(alpha_target=0.1, max_scale=1.0, lifetime=1)
    expected_keys = [{1}, {1, 2}, {2, 3}, {2, 3, 4}, {4, 5}]
    for score_value, keys in zip([1.0, 0.5, 2.0, 0.25, 1.5], expected_keys):
        method._update_score(score_value)
        assert set(method.experts) == keys
        assert np.isfinite(method._radius())
        for expert in method.experts.values():
            assert expert.age >= 1
            assert np.isfinite([expert.z, expert.wz, expert.grad_norm, expert.yhat]).all()


def test_order_statistic_treap_matches_numpy_ranks_and_inverted_quantiles():
    values = np.array([0.4, 0.1, 0.4, 0.9, 0.2, 0.7, 0.2])
    tree = _OrderStatisticTreap()
    for value in values:
        tree.insert(float(value))
    for probe in [0.0, 0.2, 0.4, 0.65, 1.0]:
        assert np.isclose(tree.tail_fraction(probe), np.mean(values >= probe))
    for q in [0.0, 0.1, 0.5, 0.9, 1.0]:
        expected = np.quantile(values, q, method="inverted_cdf")
        assert np.isclose(tree.inverted_cdf_quantile(q), expected)


def test_only_residual_aci_requires_preupdate_prediction():
    assert ResidualACI.requires_prediction_before_update
    for cls in [ResidualNExCP, ResidualPID, ResidualFACI, ResidualSFOGD, ResidualSAOCP]:
        assert not getattr(cls, "requires_prediction_before_update", False)


def test_generic_runner_respects_prediction_dependent_pre_online_state():
    warmup = [step(0, 0.2), step(1, 0.7)]
    pre_online = [step(2, 0.4)]
    online = [step(3, 0.9)]
    direct = _run_interval_only_multi(
        ResidualACI(),
        {"warmup": warmup, "pre_online": pre_online, "online": online},
        (0.1,),
    )
    generic = run_one_method(ResidualACI(), warmup, online, (0.1,), pre_online)
    assert np.allclose(direct.intervals_by_alpha[0.1][0], generic.intervals_by_alpha[0.1][0])
    assert np.allclose(direct.intervals_by_alpha[0.1][1], generic.intervals_by_alpha[0.1][1])
