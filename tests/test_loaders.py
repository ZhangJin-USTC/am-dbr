import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real_data"))
from loaders import _autoregressive_frame, _calendar_features, choose_split, data_manifest, fit_base_model  # noqa: E402


def test_split_is_chronological_and_mutable():
    split = choose_split(1000, max_eval=100, train_frac=0.2, warm_frac=0.45, eval_start_frac=0.7)
    assert 0 < split.train_end < split.warm_end <= split.report_start < split.eval_end <= 1000
    split.eval_end -= 1
    assert split.eval_end == 799


def test_autoregressive_frame_drops_only_initial_lags():
    dt = pd.date_range("2020-01-01", periods=20, freq="D")
    dt2, X, y = _autoregressive_frame(dt, np.arange(20.0), (1, 7))
    assert len(y) == 13
    assert len(dt2) == len(X) == len(y)
    assert np.isclose(X.iloc[0]["y_lag_1"], 6.0)
    assert np.isclose(X.iloc[0]["y_lag_7"], 0.0)


def test_model_factory_fits_numeric_features():
    X = pd.DataFrame({"a": [0.0, 1.0, np.nan, 3.0], "b": [1.0, 1.0, 2.0, 2.0]})
    y = np.array([0.0, 1.0, 2.0, 3.0])
    for kind in ("ridge", "hgb"):
        model = fit_base_model(X, y, kind)
        model.fit(X, y)
        pred = model.predict(X)
        assert np.isfinite(pred).all()


def test_calendar_trend_does_not_depend_on_future_dataset_length():
    prefix = pd.date_range("2020-01-01", periods=20, freq="D")
    extended = pd.date_range("2020-01-01", periods=40, freq="D")
    prefix_features = _calendar_features(pd.Series(prefix))
    extended_features = _calendar_features(pd.Series(extended)).iloc[:20]
    assert np.allclose(prefix_features["trend"], extended_features["trend"])


def test_data_manifest_records_relative_path_size_and_hash(tmp_path):
    source = tmp_path / "example.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = data_manifest(tmp_path)
    assert manifest["data_directory"] == tmp_path.name
    assert manifest["files"] == [{
        "path": "example.csv",
        "bytes": source.stat().st_size,
        "sha256": "ea14f99c47575613ab22111122c847728c61007f6bfd7b062d02fcb99df3feb0",
    }]


def test_data_manifest_selects_only_the_requested_dataset_file(tmp_path):
    (tmp_path / "open_meteo_seattle_daily.csv").write_bytes(b"seattle")
    (tmp_path / "open_meteo_phoenix_daily.csv").write_bytes(b"phoenix")
    manifest = data_manifest(tmp_path, dataset="open_meteo_seattle_daily")
    assert manifest["dataset"] == "open_meteo_seattle_daily"
    assert [entry["path"] for entry in manifest["files"]] == ["open_meteo_seattle_daily.csv"]
    assert set(manifest["sources"]) == {"open_meteo_seattle_daily"}
