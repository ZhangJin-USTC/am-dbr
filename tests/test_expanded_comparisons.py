"""Meaningful checks for the presentation-stage interval score."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision_tools"))
from build_expanded_comparisons import AM, ORDER, SCENARIOS, interval_score, paired_intervals, paired_is90_table  # noqa: E402


def test_interval_score_zero_for_certain_contained_observation():
    # A zero-width interval containing the observation has zero score.
    value = interval_score(np.array([2.0]), np.array([2.0]), np.array([2.0]), alpha=0.1)
    assert np.allclose(value, [0.0])


def test_interval_score_penalises_boundary_misses_with_alpha_weight():
    # A miss of .5 outside a 90% interval receives 2/alpha times the miss.
    value = interval_score(np.array([2.5]), np.array([1.0]), np.array([2.0]), alpha=0.1)
    assert np.allclose(value, [11.0])


def test_interval_score_is_translation_invariant_and_scales_linearly():
    y = np.array([-1.0, 0.5, 4.0])
    lower = np.array([-2.0, 0.0, 3.0])
    upper = np.array([0.0, 1.0, 5.0])
    shifted = interval_score(y + 7.0, lower + 7.0, upper + 7.0)
    original = interval_score(y, lower, upper)
    assert np.allclose(shifted, original)
    scaled = interval_score(2.0 * y, 2.0 * lower, 2.0 * upper)
    assert np.allclose(scaled, 2.0 * original)


def test_interval_score_retains_infinite_inputs_for_a_bad_forecast():
    score = interval_score(np.array([1.0]), np.array([0.0]), np.array([np.inf]))
    assert np.isinf(score[0])


def paired_fixture():
    # Six distinct scenario offsets cancel within each matched comparison.
    # The independent replication unit is the seed, not 48 scenario-seed rows.
    rows = []
    for seed in range(1, 9):
        for scenario_index, scenario in enumerate(SCENARIOS):
            for method in ORDER:
                value = 100 + scenario_index + (0 if method == AM else seed)
                rows.append({"kind": "synthetic", "group": scenario, "seed": seed,
                             "method": method, **{metric: value for metric in
                             ["mean_width", "IS90", "PIT_CRPS", "avg_gap", "rolling_mean_abs_gap"]}})
    return paired_intervals(pd.DataFrame(rows))


def test_paired_interval_uses_eight_seed_means_and_correct_difference_direction():
    paired = paired_fixture()
    overall = paired[(paired.group == "Overall") & (paired.metric == "IS90")]
    assert len(overall) == 14
    assert overall.seed_count.eq(8).all()
    np.testing.assert_allclose(overall.difference, -4.5)
    # t_(.975,7) * sqrt(sample variance 6 / n 8).
    radius = 2.3646242515927844 * np.sqrt(6.0 / 8.0)
    np.testing.assert_allclose(overall.low, -4.5 - radius)
    np.testing.assert_allclose(overall.high, -4.5 + radius)
    rendered = paired_is90_table(paired)
    assert rendered.count(" & 8 ") == 14
    assert "tab:paired-is90" in rendered
    assert "-4.500" in rendered


def test_paired_table_rejects_missing_comparator():
    paired = paired_fixture()
    with pytest.raises(ValueError, match="fourteen comparators"):
        paired_is90_table(paired[paired.right != ORDER[0]])


def test_reporting_rejects_predictions_from_another_code_package(tmp_path):
    import json
    import build_expanded_comparisons as builder
    protocol = builder.main_protocol(builder.SEEDS, list(builder.SCENARIOS), builder.DATASET_NAMES)
    protocol['source_sha256'] = '0' * 64
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    with pytest.raises(ValueError, match='different code'):
        builder.source_authorisation(tmp_path)


def test_reporting_checks_settings_even_when_code_matches(tmp_path):
    import json
    import build_expanded_comparisons as builder
    protocol = builder.main_protocol(builder.SEEDS, list(builder.SCENARIOS), builder.DATASET_NAMES)
    protocol['seeds'] = [9999]
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    with pytest.raises(ValueError, match='Expanded comparison main protocol'):
        builder.source_authorisation(tmp_path)


def test_reporting_accepts_the_complete_current_protocol(tmp_path):
    import json
    import build_expanded_comparisons as builder
    protocol = builder.main_protocol(builder.SEEDS, list(builder.SCENARIOS), builder.DATASET_NAMES)
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    assert builder.source_authorisation(tmp_path) == builder.fingerprint()
