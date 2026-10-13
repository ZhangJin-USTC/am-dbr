"""Self-contained public real-data loaders for one-step-ahead forecasting.

Each loader returns a chronological feature table whose rows use only calendar
variables known at forecast time and lagged responses.  This avoids the
contemporaneous-response leakage that can arise when all columns of a public
time-series table are passed directly to the base model.
"""

from __future__ import annotations

import io
import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Split:
    train_end: int
    warm_end: int
    report_start: int
    eval_end: int


DATASET_SPLIT_DEFAULTS = {
    "open_meteo_seattle_daily": {"train_frac": 0.20, "warm_frac": 0.45, "eval_start_frac": 0.70},
    "open_meteo_phoenix_daily": {"train_frac": 0.20, "warm_frac": 0.45, "eval_start_frac": 0.70},
    "jena_climate": {"train_frac": 0.20, "warm_frac": 0.45, "eval_start_frac": 0.65},
    "metro_traffic": {"train_frac": 0.20, "warm_frac": 0.45, "eval_start_frac": 0.65},
    "opsd_germany_daily": {"train_frac": 0.20, "warm_frac": 0.45, "eval_start_frac": 0.55},
}

DATA_SOURCE_URLS = {
    "open_meteo_seattle_daily": (
        "https://archive-api.open-meteo.com/v1/archive?latitude=47.61&longitude=-122.33"
        "&start_date=1980-01-01&end_date=2025-12-31&daily=temperature_2m_mean&timezone=UTC"
    ),
    "open_meteo_phoenix_daily": (
        "https://archive-api.open-meteo.com/v1/archive?latitude=33.45&longitude=-112.07"
        "&start_date=1980-01-01&end_date=2025-12-31&daily=temperature_2m_mean&timezone=UTC"
    ),
    "jena_climate": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
    "metro_traffic": "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip",
    "opsd_germany_daily": "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv",
}

DATASET_CACHE_FILES = {
    "open_meteo_seattle_daily": ("open_meteo_seattle_daily.csv",),
    "open_meteo_phoenix_daily": ("open_meteo_phoenix_daily.csv",),
    "jena_climate": ("jena_climate_2009_2016.csv.zip",),
    "metro_traffic": ("metro_interstate_traffic_volume.zip",),
    "opsd_germany_daily": ("opsd_germany_daily.csv",),
}


def choose_split(
    n: int,
    max_eval: int | None,
    train_frac: float,
    warm_frac: float,
    eval_start_frac: float | None = None,
) -> Split:
    if n < 50:
        raise ValueError(f"At least 50 chronological observations are required; received {n}.")
    train_end = max(20, int(np.floor(n * float(train_frac))))
    warm_end = max(train_end + 10, int(np.floor(n * float(warm_frac))))
    report_start = warm_end if eval_start_frac is None else max(warm_end, int(np.floor(n * float(eval_start_frac))))
    eval_end = n if not max_eval or max_eval <= 0 else min(n, report_start + int(max_eval))
    if not (0 < train_end < warm_end <= report_start < eval_end <= n):
        raise ValueError(
            "Invalid chronological split: "
            f"train={train_end}, warm={warm_end}, report={report_start}, end={eval_end}, n={n}."
        )
    return Split(train_end, warm_end, report_start, eval_end)


def fit_base_model(X_train: pd.DataFrame, y_train: np.ndarray, model_type: str):
    del X_train, y_train
    if model_type == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if model_type == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, random_state=0),
        )
    raise ValueError(f"Unknown model_type {model_type!r}; expected 'ridge' or 'hgb'.")


def _download(url: str, path: Path, timeout: int = 180) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    retry = Retry(total=4, connect=4, read=4, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(path)
    return path


def _calendar_features(dt: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(dt, utc=True).dt.tz_localize(None)
    day = dt.dt.dayofyear.to_numpy(dtype=float)
    hour = dt.dt.hour.to_numpy(dtype=float)
    dow = dt.dt.dayofweek.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            # A fixed calendar origin avoids using the final dataset length as
            # future metadata when constructing the current feature vector.
            "trend": (dt - pd.Timestamp("2000-01-01")).dt.total_seconds().to_numpy(dtype=float) / 86_400.0,
            "day_sin": np.sin(2.0 * np.pi * day / 365.25),
            "day_cos": np.cos(2.0 * np.pi * day / 365.25),
            "dow_sin": np.sin(2.0 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2.0 * np.pi * dow / 7.0),
            "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
        }
    )


def _autoregressive_frame(dt, y, lags: tuple[int, ...], frequency="D"):
    """Stable sort, aggregate duplicate timestamps, then construct exact lags.

    A row predicts the mean target at its timestamp using observations strictly
    earlier in local source time. No target is interpolated or forward-filled.
    Missing lag timestamps exclude the row. Gaps in retained evaluation rows
    are therefore permitted and are reported explicitly.
    """
    frame = pd.DataFrame({"time": pd.to_datetime(np.asarray(dt)), "y": np.asarray(y, float)})
    if frame["time"].isna().any() or not np.isfinite(frame["y"].dropna()).all():
        raise ValueError("Invalid timestamp or non-finite target.")
    diagnostics = {"raw_rows": len(frame), "frequency": frequency,
                   "source_time_convention": "source clock (naive timestamps preserved)",
                   "raw_duplicate_rows": int(frame.time.duplicated().sum()),
                   "raw_backwards_steps": int((frame.time.diff() < pd.Timedelta(0)).sum()),
                   "conflicting_duplicate_times": int((frame.groupby("time").y.nunique() > 1).sum()),
                   "duplicate_policy": "mean target at timestamp before any split or lag",
                   "lags": list(lags), "missing_policy": "drop rows missing target or any exact-time lag"}
    series = frame.sort_values("time", kind="stable").groupby("time", sort=True).y.mean()
    times = pd.Series(series.index)
    X = _calendar_features(times)
    delta = pd.tseries.frequencies.to_offset(frequency)
    for lag in lags:
        if lag < 1:
            raise ValueError("All lags must be strictly positive.")
        X[f"y_lag_{lag}"] = series.reindex(series.index-lag*delta).to_numpy()
    valid = X.notna().all(axis=1).to_numpy() & series.notna().to_numpy()
    out_dt = times.loc[valid].reset_index(drop=True)
    out_X = X.loc[valid].reset_index(drop=True)
    diagnostics.update(unique_timestamps=len(series), retained_rows=int(valid.sum()),
                       excluded_rows=int((~valid).sum()),
                       retained_nonunit_gaps=int((out_dt.diff().dropna() !=
                           (pd.Timestamp("2000-01-01") + delta - pd.Timestamp("2000-01-01"))).sum()))
    out_X.attrs["preprocessing"] = diagnostics
    return out_dt, out_X, series.to_numpy()[valid]


PINNED_SHA256 = {
    "open_meteo_seattle_daily.csv": "696af72bb5f1e6f2232d53a4042b6efd02d75fb378d0b50126c618e6ecf7215a",
    "open_meteo_phoenix_daily.csv": "06d399967bcf4a3635556005f615a288b72ee3fb19fc557ed3302931051a05e7",
    "jena_climate_2009_2016.csv.zip": "63d757501e92284a7de7cdbef0337f03b24e13ead7ac2b5b8c86a18d8e38ba5b",
    "metro_interstate_traffic_volume.zip": "b99aeabcbd6cc86f642da3a79d90883425798f58abd3b3302da2fa19dda73768",
    "opsd_germany_daily.csv": "729b025d984497827cb6b89048d7c2c6a551e7aa727bc47ea0b74a2d1e84675d",
}


def verify_pinned_input(path):
    path = Path(path)
    actual = _sha256(path)
    expected = PINNED_SHA256[path.name]
    if actual != expected:
        raise ValueError(f"Raw input hash mismatch for {path.name}: {actual}; expected {expected}. "
                         "Use the frozen source snapshot; do not silently replace the study input.")
    return actual


def _open_meteo(data_dir: Path, city: str, latitude: float, longitude: float):
    cache = data_dir / f"open_meteo_{city}_daily.csv"
    if not cache.exists():
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={latitude}&longitude={longitude}"
            "&start_date=1980-01-01&end_date=2025-12-31"
            "&daily=temperature_2m_mean&timezone=UTC"
        )
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        daily = response.json().get("daily", {})
        frame = pd.DataFrame({"date": daily.get("time", []), "temperature_2m_mean": daily.get("temperature_2m_mean", [])})
        if frame.empty:
            raise RuntimeError(f"Open-Meteo returned no daily observations for {city}.")
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache, index=False)
    verify_pinned_input(cache)
    frame = pd.read_csv(cache)
    dt, X, y = _autoregressive_frame(frame["date"], frame["temperature_2m_mean"], (1, 2, 7, 14, 30))
    return dt, X, y, f"{city.title()} daily mean temperature (one-day-ahead)"


def load_open_meteo_seattle_daily(data_dir: Path):
    return _open_meteo(data_dir, "seattle", 47.61, -122.33)


def load_open_meteo_phoenix_daily(data_dir: Path):
    return _open_meteo(data_dir, "phoenix", 33.45, -112.07)


def load_jena_climate(data_dir: Path):
    archive = _download(
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
        data_dir / "jena_climate_2009_2016.csv.zip",
    )
    verify_pinned_input(archive)
    frame = pd.read_csv(archive, compression="zip")
    dt = pd.to_datetime(frame["Date Time"], format="%d.%m.%Y %H:%M:%S")
    y = frame["T (degC)"].astype(float)
    return (*_autoregressive_frame(dt, y, (1, 6, 12, 144, 1008), frequency="10min"), "Jena temperature (10-minute-ahead)")


def load_metro_traffic(data_dir: Path):
    archive = _download(
        "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip",
        data_dir / "metro_interstate_traffic_volume.zip",
    )
    verify_pinned_input(archive)
    with zipfile.ZipFile(archive) as zf:
        member = next(name for name in zf.namelist() if name.lower().endswith((".csv", ".csv.gz")))
        payload = zf.read(member)
    if member.lower().endswith(".gz"):
        frame = pd.read_csv(io.BytesIO(payload), compression="gzip")
    else:
        frame = pd.read_csv(io.BytesIO(payload))
    dt = pd.to_datetime(frame["date_time"])
    y = frame["traffic_volume"].astype(float)
    return (*_autoregressive_frame(dt, y, (1, 2, 24, 48, 168), frequency="h"), "Metro traffic volume (one-hour-ahead)")


def load_opsd_germany_daily(data_dir: Path):
    path = _download(
        "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv",
        data_dir / "opsd_germany_daily.csv",
    )
    verify_pinned_input(path)
    frame = pd.read_csv(path)
    dt = pd.to_datetime(frame["Date"])
    y = frame["Consumption"].astype(float)
    return (*_autoregressive_frame(dt, y, (1, 2, 7, 14, 28)), "German electricity consumption (one-day-ahead)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_manifest(data_dir: Path, dataset: str | None = None) -> dict:
    """Return portable provenance for the exact cached public inputs used."""
    root = Path(data_dir)
    if dataset is not None and dataset not in DATASET_CACHE_FILES:
        raise KeyError(f"Unknown dataset for provenance: {dataset!r}.")
    relative_paths = (
        DATASET_CACHE_FILES[dataset]
        if dataset is not None
        else tuple(path.relative_to(root).as_posix() for path in sorted(root.rglob("*")) if path.is_file())
    )
    files = []
    for relative in relative_paths:
        path = root / relative
        if path.is_file():
            files.append({
                "path": Path(relative).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            })
    return {
        "data_directory": root.name,
        "dataset": dataset,
        "sources": DATA_SOURCE_URLS if dataset is None else {dataset: DATA_SOURCE_URLS[dataset]},
        "files": files,
    }


LOADERS = {
    "open_meteo_seattle_daily": load_open_meteo_seattle_daily,
    "open_meteo_phoenix_daily": load_open_meteo_phoenix_daily,
    "jena_climate": load_jena_climate,
    "metro_traffic": load_metro_traffic,
    "opsd_germany_daily": load_opsd_germany_daily,
}
