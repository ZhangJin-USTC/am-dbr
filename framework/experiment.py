"""Online experiment loop and data adapter for SOTA OCC comparisons.

The framework runs all methods through the same Step interface:
  - u_t = F_t(y_t), the PIT value under the base predictive CDF;
  - y_t, the realized response in original units;
  - F_t_inv, the base predictive quantile function.

This file reuses the manuscript real-data pipeline to construct Step objects,
then evaluates OCC against explicitly named PIT-map and residual-score
baselines.  Interval-only methods are reported only on interval metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from core import OnlineMethod, Step
from methods import (
    ACI,
    AdaptiveMemoryDBR,
    BaseCDF,
    DCP,
    DirectBinnedRecalibration,
    OCC,
    RollingPITECDF,
    ScalarAdaptiveCP,
    StaticPIT,
    WeightedPITECDF,
    scalar_adaptive_methods,
)
from residual_baselines import residual_baselines
from paper_config import PAPER_CONFIG
import metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "datasets"
DEFAULT_OUT_DIR = ROOT / "outputs" / "sota_occ"
DEFAULT_ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.50)


DATASET_ALIASES = {
    "phoenix": "open_meteo_phoenix_daily",
    "phoenix_weather": "open_meteo_phoenix_daily",
    "gefcom2014_load": "pjme_hourly",
    "pjm_load": "pjme_hourly",
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_pipeline_module():
    """Return the repository real-data module, with long datasets registered."""
    long_candidates = [ROOT / "real_data" / "occ_long_drift_experiments.py"]
    for path in long_candidates:
        if path.exists():
            long_mod = _load_module(path, "occ_existing_long")
            if hasattr(long_mod, "register_long_datasets"):
                long_mod.register_long_datasets()
            setattr(long_mod.occ, "__occ_pipeline_path__", str(path))
            return long_mod.occ

    real_candidates = [ROOT / "real_data" / "occ_real_data_experiments.py"]
    for path in real_candidates:
        if path.exists():
            mod = _load_module(path, "occ_existing_real")
            setattr(mod, "__occ_pipeline_path__", str(path))
            return mod

    loader_path = ROOT / "real_data" / "loaders.py"
    if loader_path.exists():
        mod = _load_module(loader_path, "occ_public_loaders")
        setattr(mod, "__occ_pipeline_path__", str(loader_path))
        return mod

    raise FileNotFoundError("Could not find the existing OCC real-data pipeline.")


def _normal_step(t: int, y: float, mu: float, sigma: float) -> Step:
    u = float(norm.cdf((y - mu) / sigma))

    def F_t_inv(p, _mu=float(mu), _sigma=float(sigma)):
        p_arr = np.asarray(p, dtype=float)
        out = _mu + _sigma * norm.ppf(np.clip(p_arr, 0.0, 1.0))
        return float(out) if np.ndim(p) == 0 else out

    return Step(t=int(t), u=u, y=float(y), F_t_inv=F_t_inv)


def _portable_path_label(path: str | Path) -> str:
    """Avoid machine-specific absolute paths in reproducibility artifacts."""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return candidate.name


def load_dataset(
    name: str,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    max_eval: int = 2500,
    model_type: str = "ridge",
    train_frac: float | None = None,
    warm_frac: float | None = None,
    eval_start_frac: float | None = None,
) -> dict:
    """Construct warmup, pre-online, and reported-online Step streams.

    The split mirrors the manuscript protocol. Warmup initializes methods.
    The optional pre-online segment updates online methods but is not reported;
    this is needed for datasets whose reported window starts after the warmup
    endpoint. Static-PIT ignores these updates by design.
    """
    occ = _load_pipeline_module()
    dataset = DATASET_ALIASES.get(name, name)
    if dataset not in occ.LOADERS:
        raise KeyError(f"Unknown dataset {name!r}. Available keys: {sorted(occ.LOADERS)}")

    dt, X, y, label = occ.LOADERS[dataset](Path(data_dir))
    preprocessing = dict(X.attrs.get("preprocessing", {}))
    dt = pd.Series(pd.to_datetime(dt)).reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=float)

    defaults = occ.DATASET_SPLIT_DEFAULTS.get(dataset, {})
    if train_frac is None:
        train_frac = defaults.get("train_frac", 0.25)
    if warm_frac is None:
        warm_frac = defaults.get("warm_frac", 0.40)
    if eval_start_frac is None:
        eval_start_frac = defaults.get("eval_start_frac", None)
    split = occ.choose_split(
        len(y),
        max_eval=max_eval,
        train_frac=train_frac,
        warm_frac=warm_frac,
        eval_start_frac=eval_start_frac,
    )
    if max_eval is not None and max_eval > 0:
        split.eval_end = min(split.eval_end, split.report_start + int(max_eval))

    X_train = X.iloc[: split.train_end]
    y_train = y[: split.train_end]
    X_all = X.iloc[: split.eval_end]
    model = occ.fit_base_model(X_train, y_train, model_type)
    model.fit(X_train, y_train)
    mu_all = model.predict(X_all)
    sigma = float(np.std(y_train - mu_all[: split.train_end], ddof=1))
    sigma = max(sigma, 1e-3)

    warmup = [_normal_step(t, y[t], mu_all[t], sigma) for t in range(split.train_end, split.warm_end)]
    pre_online = [_normal_step(t, y[t], mu_all[t], sigma) for t in range(split.warm_end, split.report_start)]
    online = [_normal_step(t, y[t], mu_all[t], sigma) for t in range(split.report_start, split.eval_end)]

    cached_data_manifest = (
        occ.data_manifest(Path(data_dir), dataset=dataset) if hasattr(occ, "data_manifest") else None
    )
    return {
        "warmup": warmup,
        "pre_online": pre_online,
        "online": online,
        "meta": {
            "requested_name": name,
            "dataset": dataset,
            "label": label,
            "forecast_task": label,
            "feature_protocol": "calendar variables known at forecast time plus lagged responses only",
            "preprocessing": preprocessing,
            "n_total": int(len(y)),
            "first_timestamp": str(dt.iloc[0]),
            "last_timestamp": str(dt.iloc[-1]),
            "report_first_timestamp": str(dt.iloc[split.report_start]),
            "report_last_timestamp": str(dt.iloc[split.eval_end - 1]),
            "train_end": int(split.train_end),
            "warm_end": int(split.warm_end),
            "report_start": int(split.report_start),
            "eval_end": int(split.eval_end),
            "n_warmup": len(warmup),
            "n_pre_online": len(pre_online),
            "n_online": len(online),
            "sigma": sigma,
            "model_type": model_type,
            "data_dir": _portable_path_label(data_dir),
            "pipeline_module_path": _portable_path_label(
                getattr(occ, "__occ_pipeline_path__", getattr(occ, "__file__", ""))
            ),
            "train_frac": train_frac,
            "warm_frac": warm_frac,
            "eval_start_frac": eval_start_frac,
            "_data_manifest": cached_data_manifest,
        },
    }


@dataclass
class MethodResult:
    name: str
    intervals_by_alpha: dict
    calibrated_pit: np.ndarray | None
    runtime_sec: float
    pit_crps: np.ndarray | None = None
    diagnostics: dict | None = None


def _silent_update(method: OnlineMethod, steps: Sequence[Step], alphas: Sequence[float]) -> None:
    """Advance a method through unreported online steps."""
    for step in steps:
        if getattr(method, "requires_prediction_before_update", False):
            if len(alphas) != 1:
                raise ValueError(
                    "A pre-online interval method with prediction-dependent state "
                    "requires exactly one alpha in run_one_method."
                )
            method.predict_interval(step.F_t_inv, float(alphas[0]))
        method.update(step)


def run_one_method(
    method: OnlineMethod,
    warmup: Sequence[Step],
    online: Sequence[Step],
    alphas: Sequence[float],
    pre_online: Sequence[Step] | None = None,
) -> MethodResult:
    import time

    t0 = time.perf_counter()
    method.warmup(warmup)
    if pre_online:
        _silent_update(method, pre_online, alphas)

    # Warm-up and pre-online updates also incur AM losses.  Capture the actual
    # report-start prior and accumulate report losses separately: subtracting
    # two long cumulative totals would add avoidable cancellation error.
    am_report = None
    if isinstance(method, AdaptiveMemoryDBR):
        am_report = {
            "prior_updates": method._updates,
            "initial_weights": method._weights.copy(),
            "expert_losses": np.zeros_like(method.cumulative_expert_losses),
            "aggregate_loss": 0.0,
        }

    T_online = len(online)
    intervals = {a: (np.zeros(T_online), np.zeros(T_online)) for a in alphas}
    calibrated_pit = np.zeros(T_online)
    pit_crps = np.zeros(T_online)
    produces_g = True
    try:
        method.predict_g(np.array([0.5]))
    except NotImplementedError:
        produces_g = False
        calibrated_pit = None
        pit_crps = None

    for i, step in enumerate(online):
        for alpha in alphas:
            L, U = method.predict_interval(step.F_t_inv, alpha)
            intervals[alpha][0][i] = L
            intervals[alpha][1][i] = U
        if produces_g:
            calibrated_pit[i] = float(method.predict_g(np.array([step.u]))[0])
            pit_crps[i] = method.score_pit(step.u)
        method.update(step)
        if am_report is not None:
            am_report["expert_losses"] += method.last_expert_losses
            am_report["aggregate_loss"] += method.last_aggregate_loss

    diagnostics = None
    if isinstance(method, AdaptiveMemoryDBR):
        def losses(aggregate, experts):
            best = float(np.min(experts))
            return {
                "cumulative_expert_losses": [float(value) for value in experts],
                "cumulative_aggregate_loss_exact": float(aggregate),
                "best_fixed_expert_loss": best,
                "fixed_expert_regret": float(aggregate) - best,
            }

        full_history = losses(method.cumulative_aggregate_loss,
                              method.cumulative_expert_losses)
        full_history.update(horizon=method._updates,
                            scope="all updates, including warmup and pre_online")
        report_window = losses(am_report["aggregate_loss"], am_report["expert_losses"])
        report_window.update(
            horizon=T_online,
            prior_updates=am_report["prior_updates"],
            scope="reported online observations only; expert states are not reset",
            initial_expert_weights=[float(value) for value in am_report["initial_weights"]],
        )
        diagnostics = {
            "schema_version": 2,
            "expert_labels": list(method._expert_labels),
            "final_expert_weights": [float(value) for value in method._weights],
            "full_history": full_history,
            "report_window": report_window,
            # Preserve older callers while making their scope unambiguous.
            "legacy_top_level_scope": "full_history; prefer the explicit nested records",
            **{key: full_history[key] for key in (
                "cumulative_expert_losses", "cumulative_aggregate_loss_exact",
                "best_fixed_expert_loss", "fixed_expert_regret")},
        }
    return MethodResult(
        name=method.name,
        intervals_by_alpha=intervals,
        calibrated_pit=calibrated_pit,
        runtime_sec=time.perf_counter() - t0,
        pit_crps=pit_crps,
        diagnostics=diagnostics,
    )


def _run_aci_multi(template: ACI, dataset: dict, alphas: Sequence[float]) -> MethodResult:
    import copy
    import time

    t0 = time.perf_counter()
    warmup = dataset["warmup"]
    pre_online = dataset.get("pre_online", [])
    online = dataset["online"]
    T = len(online)
    intervals = {}
    for alpha in alphas:
        m = copy.deepcopy(template)
        m.alpha_target = alpha
        m.alpha_t = alpha
        m.warmup(warmup)
        for step in pre_online:
            m.predict_interval(step.F_t_inv, alpha)
            m.update(step)
        L_arr = np.zeros(T)
        U_arr = np.zeros(T)
        for i, step in enumerate(online):
            L, U = m.predict_interval(step.F_t_inv, alpha)
            L_arr[i], U_arr[i] = L, U
            m.update(step)
        intervals[alpha] = (L_arr, U_arr)
    return MethodResult(name="ACI-PIT", intervals_by_alpha=intervals, calibrated_pit=None, runtime_sec=time.perf_counter() - t0)


def _run_interval_only_multi(template: OnlineMethod, dataset: dict, alphas: Sequence[float]) -> MethodResult:
    import copy
    import time

    t0 = time.perf_counter()
    warmup = dataset["warmup"]
    pre_online = dataset.get("pre_online", [])
    online = dataset["online"]
    T = len(online)
    intervals = {}
    display_name = template.name
    for alpha in alphas:
        m = copy.deepcopy(template)
        if hasattr(m, "set_alpha_target"):
            m.set_alpha_target(alpha)
        elif hasattr(m, "alpha_target"):
            m.alpha_target = alpha
        m.warmup(warmup)
        for step in pre_online:
            if getattr(m, "requires_prediction_before_update", False):
                m.predict_interval(step.F_t_inv, alpha)
            m.update(step)
        L_arr = np.zeros(T)
        U_arr = np.zeros(T)
        for i, step in enumerate(online):
            L, U = m.predict_interval(step.F_t_inv, alpha)
            L_arr[i], U_arr[i] = L, U
            m.update(step)
        intervals[alpha] = (L_arr, U_arr)
    return MethodResult(name=display_name, intervals_by_alpha=intervals, calibrated_pit=None, runtime_sec=time.perf_counter() - t0)


def default_methods_factory() -> dict[str, OnlineMethod]:
    fixed_lam = PAPER_CONFIG.fixed_lam
    K = PAPER_CONFIG.baseline_window
    return {
        "Base": BaseCDF(),
        "DBR-Fixed": DirectBinnedRecalibration(
            lam=fixed_lam, bins=PAPER_CONFIG.dbr_bins
        ),
        "OCC-Fixed": OCC(lam=fixed_lam, bins=PAPER_CONFIG.bins),
        "OCC-Adaptive": OCC(
            adaptive=True,
            bins=PAPER_CONFIG.bins,
            lam_min=PAPER_CONFIG.adaptive_lam_min,
            lam_max=PAPER_CONFIG.adaptive_lam_max,
            tau=PAPER_CONFIG.adaptive_tau,
            beta=PAPER_CONFIG.adaptive_beta,
            window=PAPER_CONFIG.adaptive_window,
        ),
        "Static-PIT": StaticPIT(),
        "Rolling-PIT-ECDF": RollingPITECDF(K=K),
        "DCP-score": DCP(K=K),
        "Weighted-PIT-ECDF": WeightedPITECDF(lam=fixed_lam, max_buffer=PAPER_CONFIG.weighted_history),
        **residual_baselines(alpha_target=PAPER_CONFIG.alpha, aci_gamma=PAPER_CONFIG.aci_gamma),
    }


def full_methods_factory(
    include_pit_wrappers: bool = False,
    baseline_path: str | Path | None = None,
) -> dict[str, OnlineMethod]:
    methods = default_methods_factory()
    if include_pit_wrappers:
        methods.update(scalar_adaptive_methods(alpha_target=0.10, baseline_path=baseline_path))
    return methods


def run_comparison(
    dataset: dict,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    methods_factory: Callable[[], dict[str, OnlineMethod]] | None = None,
) -> dict[str, MethodResult]:
    if methods_factory is None:
        methods_factory = default_methods_factory
    methods = methods_factory()

    results = {}
    for name, method in methods.items():
        if isinstance(method, ACI):
            results[name] = _run_aci_multi(method, dataset, alphas)
        elif getattr(method, "interval_only", False):
            results[name] = _run_interval_only_multi(method, dataset, alphas)
        else:
            results[name] = run_one_method(
                method,
                dataset["warmup"],
                dataset["online"],
                alphas,
                pre_online=dataset.get("pre_online", []),
            )
    return results


def summary_table(
    results: dict[str, MethodResult],
    y: np.ndarray,
    alpha_main: float = 0.10,
    pit_metric: str = "ks",
    rolling_window: int | None = None,
    round_digits: int | None = 4,
) -> pd.DataFrame:
    rows = []
    if rolling_window is None:
        rolling_window = min(500, max(50, int(len(y) // 5))) if len(y) else 50
    for name, r in results.items():
        L, U = r.intervals_by_alpha[alpha_main]
        coverage = float(np.mean((L <= y) & (y <= U)))
        cov_err = metrics.marginal_coverage_error(y, L, U, alpha_main)
        width = metrics.interval_width(L, U)
        rolling = metrics.rolling_coverage(y, L, U, window=rolling_window)
        rolling_gap = np.abs(rolling - (1.0 - alpha_main))
        rolling_gap = rolling_gap[np.isfinite(rolling_gap)]
        if r.calibrated_pit is None:
            pit_d = np.nan
        elif pit_metric == "ks":
            pit_d = metrics.pit_ks_distance(r.calibrated_pit)
        else:
            pit_d = metrics.pit_cvm_distance(r.calibrated_pit)
        rows.append(
            {
                "method": name,
                f"cov_err(alpha={alpha_main})": cov_err,
                "coverage": coverage,
                "mean_width": width["mean"],
                "median_width": width["median"],
                "rolling_mean_abs_gap": float(np.mean(rolling_gap)) if rolling_gap.size else np.nan,
                "rolling_max_abs_gap": float(np.max(rolling_gap)) if rolling_gap.size else np.nan,
                "rolling_last_abs_gap": float(rolling_gap[-1]) if rolling_gap.size else np.nan,
                f"PIT_{pit_metric.upper()}": pit_d,
                "PIT_CRPS": float(np.mean(r.pit_crps)) if r.pit_crps is not None else np.nan,
                "runtime_s": r.runtime_sec,
            }
        )
    table = pd.DataFrame(rows).set_index("method")
    return table if round_digits is None else table.round(round_digits)


def multilevel_coverage_table(
    results: dict[str, MethodResult],
    y: np.ndarray,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    round_digits: int | None = 4,
) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        row = {"method": name}
        for alpha in alphas:
            L, U = r.intervals_by_alpha[alpha]
            row[f"err@alpha={alpha}"] = metrics.marginal_coverage_error(y, L, U, alpha)
        row["avg_gap"] = float(np.mean([row[f"err@alpha={a}"] for a in alphas]))
        row["max_gap"] = float(np.max([row[f"err@alpha={a}"] for a in alphas]))
        rows.append(row)
    table = pd.DataFrame(rows).set_index("method")
    return table if round_digits is None else table.round(round_digits)


def results_to_records(dataset: dict, results: dict[str, MethodResult], alphas: Sequence[float]) -> pd.DataFrame:
    y = np.array([s.y for s in dataset["online"]], dtype=float)
    t = np.array([s.t for s in dataset["online"]], dtype=int)
    rows = []
    for name, res in results.items():
        pit_ks = metrics.pit_ks_distance(res.calibrated_pit) if res.calibrated_pit is not None else np.nan
        pit_cvm = metrics.pit_cvm_distance(res.calibrated_pit) if res.calibrated_pit is not None else np.nan
        for alpha in alphas:
            L, U = res.intervals_by_alpha[alpha]
            for i in range(y.size):
                base_center = dataset["online"][i].F_t_inv(0.5)
                rows.append(
                    {
                        "dataset": dataset["meta"]["dataset"],
                        "dataset_label": dataset["meta"]["label"],
                        "time_index": int(t[i]),
                        "method": name,
                        "alpha": float(alpha),
                        "y": float(y[i]),
                        "lower": float(L[i]),
                        "upper": float(U[i]),
                        "covered": float(L[i] <= y[i] <= U[i]),
                        "width": float(U[i] - L[i]),
                        "center_shift": float(0.5 * (L[i] + U[i]) - base_center),
                        "calibrated_pit": float(res.calibrated_pit[i]) if res.calibrated_pit is not None else np.nan,
                        "pit_ks": pit_ks,
                        "pit_cvm": pit_cvm,
                        "pit_crps": float(res.pit_crps[i]) if res.pit_crps is not None else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def run_and_save_dataset(
    name: str,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    max_eval: int = 2500,
    model_type: str = "ridge",
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    methods_factory: Callable[[], dict[str, OnlineMethod]] | None = None,
    train_frac: float | None = None,
    warm_frac: float | None = None,
    eval_start_frac: float | None = None,
    rolling_window: int | None = None,
) -> dict:
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        name,
        data_dir=data_dir,
        max_eval=max_eval,
        model_type=model_type,
        train_frac=train_frac,
        warm_frac=warm_frac,
        eval_start_frac=eval_start_frac,
    )
    results = run_comparison(ds, alphas=alphas, methods_factory=methods_factory)
    y_online = np.array([s.y for s in ds["online"]], dtype=float)
    summary = summary_table(results, y_online, alpha_main=0.10, rolling_window=rolling_window)
    multi = multilevel_coverage_table(results, y_online, alphas=alphas)
    records = results_to_records(ds, results, alphas)
    summary.reset_index().to_csv(out / "summary.csv", index=False)
    multi.reset_index().to_csv(out / "multilevel_coverage.csv", index=False)
    records.to_csv(out / "records.csv", index=False)
    data_manifest = ds["meta"].pop("_data_manifest", None)
    if data_manifest is not None:
        (out / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2), encoding="utf-8")
        ds["meta"]["data_manifest_file"] = "data_manifest.json"
    (out / "metadata.json").write_text(json.dumps(ds["meta"], indent=2), encoding="utf-8")
    return {"dataset": ds, "results": results, "summary": summary, "multi": multi, "records": records, "out": out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["metro_traffic", "beijing_pm25", "jena_climate", "phoenix_weather"])
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-eval", type=int, default=2500)
    parser.add_argument("--model-type", choices=["hgb", "ridge"], default=PAPER_CONFIG.model_type)
    parser.add_argument("--train-frac", type=float, default=None)
    parser.add_argument("--warm-frac", type=float, default=None)
    parser.add_argument("--eval-start-frac", type=float, default=None)
    parser.add_argument("--rolling-window", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for name in args.datasets:
        print(f"Running {name} ...", flush=True)
        run = run_and_save_dataset(
            name,
            out_dir=out,
            data_dir=args.data_dir,
            max_eval=args.max_eval,
            model_type=args.model_type,
            train_frac=args.train_frac,
            warm_frac=args.warm_frac,
            eval_start_frac=args.eval_start_frac,
            rolling_window=args.rolling_window,
        )
        summary = run["summary"].reset_index()
        summary.insert(0, "dataset", name)
        all_summaries.append(summary)
        print(summary.to_string(index=False), flush=True)
    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(out / "all_summary.csv", index=False)


if __name__ == "__main__":
    main()
