"""Regression checks for the independent v1.0 code review."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "framework"), str(ROOT / "scripts"), str(ROOT / "revision_tools")]

from cdf_tools import anchored_cdf_values, empirical_knots, inverse_cdf  # noqa: E402
from core import Step  # noqa: E402
from experiment import run_one_method  # noqa: E402
from methods import AdaptiveMemoryDBR, DirectBinnedRecalibration, OCC  # noqa: E402
import run_revision  # noqa: E402
from run_revision import SOURCE_MANIFEST_FILES, fingerprint_entries  # noqa: E402
import run_state_sensitivity  # noqa: E402


def test_cdf_representation_repairs_ulp_without_erasing_small_slope():
    y = anchored_cdf_values(np.array([0.0, 1.0e-17, 0.2, 1.0 + 2e-16]))
    assert np.all(np.diff(y) >= 0.0)
    assert np.all((0.0 <= y) & (y <= 1.0))
    # The first segment is positive, even though it is far below the old
    # 1e-15 plateau tolerance.
    assert inverse_cdf([0.0, 0.25, 1.0], [0.0, 1.0e-17, 1.0], 0.5e-17) == 0.125


def test_dbr_knots_stay_valid_after_long_rounded_stream():
    method = DirectBinnedRecalibration(lam=0.985, bins=140)
    rng = np.random.default_rng(270919)
    for t, u in enumerate(rng.uniform(0.0, 0.8, size=10_000)):
        method.update(Step(t=t, u=float(u)))
        _, cdf, _ = method._pit_knots()
        assert np.all(np.diff(cdf) >= 0.0)
        assert np.all((cdf >= 0.0) & (cdf <= 1.0))


@pytest.mark.parametrize("method", [
    DirectBinnedRecalibration(lam=.5, bins=4),
    AdaptiveMemoryDBR(lambdas=(.5,), bins=4, include_identity=False),
])
def test_actual_stream_tiny_positive_bin_is_inverted_continuously(method):
    for t, u in enumerate([.1] + [.9] * 55):
        method.update(Step(t=t, u=u))
    _, y, _ = method._pit_knots()
    q = float(y[1] / 2)
    assert 0 < q < 1e-15
    assert method._inverse_g(q) == .125
    assert method.predict_interval(lambda p: p, 2*q)[0] == .125


def test_occ_tiny_positive_cdf_segment_is_not_a_plateau():
    method = OCC(bins=4)
    method._knots_x = np.array([0., .25, .75, 1.])
    method._knots_y = np.array([0., 1e-17, 1., 1.])
    method._g_cache = lambda u: np.interp(u, method._knots_x, method._knots_y)
    assert method._inverse_g(5e-18) == .125
    assert method._inverse_g(1.) == .75


def test_empirical_knots_allow_zero_weights_but_reject_invalid_total():
    x, y, kind = empirical_knots([.1, .5], [0., 1.])
    assert kind == "step"
    assert inverse_cdf(x, y, .1, kind) == .5
    for weights in ([0., 0.], [1., -1.], [1., np.nan]):
        with pytest.raises(ValueError):
            empirical_knots([.1, .5], weights)


def test_am_update_loss_equals_issued_score_exactly():
    method = AdaptiveMemoryDBR(lambdas=(.5, .985, 1.), bins=140,
                              include_identity=False)
    rng = np.random.default_rng(270919)
    for t, u in enumerate(rng.uniform(0, .8, 200)):
        before = method.score_pit(u)
        _, y, _ = method._pit_knots()
        assert y.min() == 0 and y.max() == 1
        assert np.all(np.diff(y) >= 0)
        expert_losses = [expert.score_pit(u) for expert in method._experts]
        method.update(Step(t, u))
        assert method.last_aggregate_loss == before
        np.testing.assert_array_equal(method.last_expert_losses, expert_losses)


def test_am_diagnostics_distinguish_full_history_from_report_window():
    make = lambda t, u: Step(t=t, u=u, y=u, F_t_inv=lambda p: p)
    warmup = [make(0, 0.2), make(1, 0.4)]
    pre_online = [make(2, 0.6)]
    online = [make(3, 0.1), make(4, 0.9)]
    result = run_one_method(
        AdaptiveMemoryDBR(lambdas=(0.8, 1.0), bins=8, eta=0.5, share=0.1),
        warmup, online, alphas=(0.1,), pre_online=pre_online,
    )
    d = result.diagnostics
    assert d["schema_version"] == 2
    assert d["full_history"]["horizon"] == 5
    assert d["report_window"]["horizon"] == 2
    assert d["report_window"]["prior_updates"] == 3
    assert d["report_window"]["scope"].startswith("reported online")
    oracle = AdaptiveMemoryDBR(lambdas=(.8, 1.), bins=8, eta=.5, share=.1)
    for step in warmup + pre_online:
        oracle.update(step)
    np.testing.assert_array_equal(d["report_window"]["initial_expert_weights"], oracle._weights)
    prior_loss = oracle.cumulative_aggregate_loss
    report_expert_losses = np.zeros_like(oracle.cumulative_expert_losses)
    for step in online:
        oracle.update(step)
        report_expert_losses += oracle.last_expert_losses
    np.testing.assert_array_equal(d["report_window"]["cumulative_expert_losses"], report_expert_losses)
    np.testing.assert_array_equal(d["full_history"]["cumulative_expert_losses"], oracle.cumulative_expert_losses)
    assert d["full_history"]["cumulative_aggregate_loss_exact"] == oracle.cumulative_aggregate_loss
    assert d["report_window"]["cumulative_aggregate_loss_exact"] == sum(result.pit_crps)
    assert d["full_history"]["cumulative_aggregate_loss_exact"] == pytest.approx(prior_loss + sum(result.pit_crps))


def test_source_manifest_is_explicit_and_complete_for_revision_tools():
    entries = fingerprint_entries()
    assert set(entries) == set(SOURCE_MANIFEST_FILES)
    assert "revision_tools/run_state_sensitivity.py" in entries
    assert "revision_tools/run_report_window_sensitivity.py" in entries
    assert "reproduction/provenance.py" in entries


def test_manifest_rejects_missing_sources_and_detects_generator_edits(tmp_path, monkeypatch):
    for name in SOURCE_MANIFEST_FILES:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(run_revision, "ROOT", tmp_path)
    initial = run_revision.fingerprint()
    generator = tmp_path / "revision_tools/run_state_sensitivity.py"
    generator.write_bytes(generator.read_bytes() + b"\n# independent edit\n")
    assert run_revision.fingerprint() != initial
    generator.unlink()
    with pytest.raises(FileNotFoundError, match="Required manifest source"):
        run_revision.fingerprint()


def test_state_config_fingerprint_detects_parameter_changes(monkeypatch):
    initial = run_state_sensitivity.config_fingerprint()
    revised = dict(run_state_sensitivity.STATE_CONFIG, seeds=[6100])
    monkeypatch.setattr(run_state_sensitivity, "STATE_CONFIG", revised)
    assert run_state_sensitivity.config_fingerprint() != initial
