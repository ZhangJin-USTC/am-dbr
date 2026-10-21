"""Rerun manuscript real-data windows with the final Fixed/Adaptive OCC rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
REPRODUCTION = ROOT / "reproduction"
for path in [FRAMEWORK, REPRODUCTION]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment import run_and_save_dataset  # noqa: E402
from methods import BaseCDF, DCP, OCC, RollingPITECDF, StaticPIT, WeightedPITECDF  # noqa: E402
from residual_baselines import residual_baselines  # noqa: E402
from paper_config import PAPER_CONFIG  # noqa: E402
from provenance import source_manifest  # noqa: E402


TARGET_ALPHA = 0.10
OUT = ROOT / "outputs" / "realdata_fixed_adaptive_final"
PAPER = ROOT / "paper_ready" / "tables"
FIGDIR = ROOT / "figures"
DATA_DIR = ROOT / "data" / "datasets"

RUNS = [
    {
        "label": "Seattle",
        "name": "open_meteo_seattle_daily",
        "model_type": "ridge",
        "max_eval": 2500,
        "train_frac": 0.20,
        "warm_frac": 0.45,
        "eval_start_frac": 0.70,
        "rolling_window": PAPER_CONFIG.baseline_window,
    },
    {
        "label": "Phoenix",
        "name": "open_meteo_phoenix_daily",
        "model_type": "ridge",
        "max_eval": 2500,
        "train_frac": 0.20,
        "warm_frac": 0.45,
        "eval_start_frac": 0.70,
        "rolling_window": PAPER_CONFIG.baseline_window,
    },
    {
        "label": "Jena",
        "name": "jena_climate",
        "model_type": PAPER_CONFIG.model_type,
        "max_eval": 2500,
        "train_frac": None,
        "warm_frac": None,
        "eval_start_frac": 0.65,
        "rolling_window": PAPER_CONFIG.baseline_window,
    },
    {
        "label": "Metro",
        "name": "metro_traffic",
        "model_type": PAPER_CONFIG.model_type,
        "max_eval": 2500,
        "train_frac": 0.20,
        "warm_frac": 0.45,
        "eval_start_frac": 0.65,
        "rolling_window": PAPER_CONFIG.baseline_window,
    },
    {
        "label": "OPSD",
        "name": "opsd_germany_daily",
        "model_type": "ridge",
        "max_eval": 2000,
        "train_frac": 0.20,
        "warm_frac": 0.45,
        "eval_start_frac": 0.55,
        "rolling_window": PAPER_CONFIG.baseline_window,
    },
]


def methods_factory(include_residual: bool = True) -> dict:
    methods = {
        "Base": BaseCDF(),
        "Static-PIT": StaticPIT(),
        "Rolling-PIT-ECDF": RollingPITECDF(K=PAPER_CONFIG.baseline_window),
        "DCP-score": DCP(K=PAPER_CONFIG.baseline_window),
        "Weighted-PIT-ECDF": WeightedPITECDF(lam=PAPER_CONFIG.fixed_lam, max_buffer=PAPER_CONFIG.weighted_history),
        "OCC-Fixed": OCC(lam=PAPER_CONFIG.fixed_lam, bins=PAPER_CONFIG.bins),
        "OCC-Adaptive": OCC(
            adaptive=True,
            bins=PAPER_CONFIG.bins,
            lam_min=PAPER_CONFIG.adaptive_lam_min,
            lam_max=PAPER_CONFIG.adaptive_lam_max,
            beta=PAPER_CONFIG.adaptive_beta,
            tau=PAPER_CONFIG.adaptive_tau,
            window=PAPER_CONFIG.adaptive_window,
        ),
    }
    if include_residual:
        methods.update(residual_baselines(alpha_target=TARGET_ALPHA, aci_gamma=PAPER_CONFIG.aci_gamma))
    return methods


def run_all(
    selected_labels: set[str] | None = None,
    data_dir: Path = DATA_DIR,
    max_eval_override: int | None = None,
    include_residual: bool = True,
    out_dir: Path = OUT,
    paper_dir: Path = PAPER,
    fig_dir: Path = FIGDIR,
    model_type_override: str | None = None,
    fail_on_partial: bool = True,
) -> pd.DataFrame:
    if fail_on_partial and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Formal real-data output directory is not empty: {out_dir}. "
            "Use a new versioned --out-dir so stale artifacts cannot be mistaken for this run."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    configured_labels = {spec["label"].lower() for spec in RUNS}
    if selected_labels:
        unknown_labels = set(selected_labels) - configured_labels
        if unknown_labels:
            raise ValueError(
                "Unknown real-data labels: " + ", ".join(sorted(unknown_labels))
            )
    requested_specs = [
        spec for spec in RUNS
        if not selected_labels or spec["label"].lower() in selected_labels
    ]
    if not requested_specs:
        raise ValueError("No requested labels matched the configured real-data runs.")
    expected_methods = set(methods_factory(include_residual=include_residual))
    all_rows = []
    manifest = []
    failures = []
    completed_labels = []
    for spec in requested_specs:
        run_name = f"{spec['label'].lower()}_final_fixed_adaptive"
        print(f"Running {run_name}", flush=True)
        try:
            result = run_and_save_dataset(
                spec["name"],
                out_dir=out_dir,
                data_dir=data_dir,
                max_eval=max_eval_override or spec["max_eval"],
                model_type=model_type_override or spec["model_type"],
                alphas=(TARGET_ALPHA,),
                methods_factory=lambda: methods_factory(include_residual=include_residual),
                train_frac=spec["train_frac"],
                warm_frac=spec["warm_frac"],
                eval_start_frac=spec["eval_start_frac"],
                rolling_window=spec["rolling_window"],
            )
        except Exception as exc:
            failures.append({"label": spec["label"], "dataset": spec["name"], "error": repr(exc)})
            print(f"FAILED {run_name}: {exc!r}", flush=True)
            continue
        produced = result["out"]
        target = out_dir / run_name
        if produced != target:
            if target.exists():
                raise FileExistsError(
                    f"Refusing to overwrite an existing run directory: {target}. "
                    "Use a new --out-dir for a formal rerun."
                )
            produced.rename(target)
        summary = pd.read_csv(target / "summary.csv")
        observed_methods = set(summary["method"])
        if observed_methods != expected_methods or len(summary) != len(expected_methods):
            failures.append({
                "label": spec["label"],
                "dataset": spec["name"],
                "error": "method_registry_mismatch",
                "expected_methods": sorted(expected_methods),
                "observed_methods": sorted(observed_methods),
            })
            print(f"FAILED {run_name}: method registry mismatch", flush=True)
            continue
        if summary["method"].duplicated().any():
            failures.append({
                "label": spec["label"],
                "dataset": spec["name"],
                "error": "duplicate_summary_method_rows",
            })
            print(f"FAILED {run_name}: duplicate summary method rows", flush=True)
            continue
        records_path = target / "records.csv"
        multilevel_path = target / "multilevel_coverage.csv"
        metadata_path = target / "metadata.json"
        if not records_path.exists() or not multilevel_path.exists() or not metadata_path.exists():
            failures.append({"label": spec["label"], "dataset": spec["name"], "error": "missing_dataset_artifact"})
            print(f"FAILED {run_name}: missing dataset artifact", flush=True)
            continue
        metadata_check = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = pd.read_csv(records_path)
        expected_n_online = int(metadata_check["n_online"])
        if len(records) != len(expected_methods) * expected_n_online:
            failures.append({
                "label": spec["label"], "dataset": spec["name"],
                "error": "record_row_count_mismatch",
                "expected": len(expected_methods) * expected_n_online,
                "observed": len(records),
            })
            print(f"FAILED {run_name}: record row count mismatch", flush=True)
            continue
        record_methods = set(records["method"])
        if record_methods != expected_methods:
            failures.append({"label": spec["label"], "dataset": spec["name"], "error": "record_method_registry_mismatch"})
            print(f"FAILED {run_name}: record method registry mismatch", flush=True)
            continue
        key_columns = ["method", "alpha", "time_index"]
        if records.duplicated(key_columns).any():
            failures.append({"label": spec["label"], "dataset": spec["name"], "error": "duplicate_record_keys"})
            print(f"FAILED {run_name}: duplicate record keys", flush=True)
            continue
        multilevel_methods = set(pd.read_csv(multilevel_path)["method"])
        if multilevel_methods != expected_methods:
            failures.append({"label": spec["label"], "dataset": spec["name"], "error": "multilevel_method_registry_mismatch"})
            print(f"FAILED {run_name}: multilevel method registry mismatch", flush=True)
            continue
        summary.insert(0, "Data set", spec["label"])
        run_dir_label = target.relative_to(out_dir).as_posix()
        summary.insert(1, "run_dir", run_dir_label)
        all_rows.append(summary)
        meta = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        meta["display_label"] = spec["label"]
        meta["final_occ_rules"] = {
            "fixed_lam": 0.985,
            "adaptive_lam_min": 0.95,
            "adaptive_lam_max": 0.997,
            "adaptive_beta": 16.0,
            "adaptive_tau": 0.18,
            "adaptive_window": 120,
        }
        (target / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        manifest.append(meta)
        completed_labels.append(spec["label"])
    run_manifest = {
        "requested_labels": [spec["label"] for spec in requested_specs],
        "completed_labels": completed_labels,
        "expected_methods": sorted(expected_methods),
        "include_residual": include_residual,
        "model_type_override": model_type_override,
        "source_provenance": source_manifest(ROOT, [
            "framework/experiment.py",
            "framework/methods.py",
            "framework/residual_baselines.py",
            "framework/paper_config.py",
            "real_data/loaders.py",
            "scripts/rerun_main_realdata_fixed_adaptive.py",
        ]),
        "selection_policy": "All requested data sets and pre-specified methods must complete; no result-dependent filtering.",
    }
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if failures or set(completed_labels) != {spec["label"] for spec in requested_specs}:
        run_manifest["status"] = "failed_or_partial"
        (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        (out_dir / "INVALID").write_text(
            "This run is incomplete and must not be used as a paper-facing result.\n",
            encoding="utf-8",
        )
        if fail_on_partial:
            raise RuntimeError("Real-data run is incomplete; no paper-facing table was generated. See failures.json.")
    if not all_rows:
        raise RuntimeError("No real-data run completed; see failures.json.")
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(paper_dir / "realdata_fixed_adaptive_final_summary.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    run_manifest["status"] = "complete" if not failures else "partial_exploratory"
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return out


def build_tables(summary: pd.DataFrame, paper_dir: Path = PAPER) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    occ = summary[summary["method"].isin(["OCC-Fixed", "OCC-Adaptive"])].copy()
    occ["Rule"] = occ["method"].str.replace("OCC-", "", regex=False)
    occ = occ.rename(
        columns={
            "coverage": "Coverage",
            "mean_width": "Width",
            "rolling_mean_abs_gap": "Rolling cov. gap",
        }
    )
    occ[["Data set", "Rule", "Coverage", "Width", "Rolling cov. gap"]].to_csv(
        paper_dir / "realdata_fixed_adaptive_main_table.csv",
        index=False,
    )

    rows = []
    scalar = {"DCP-score", "ACI-residual", "NExCP-residual", "PID-residual-projected", "FACI-residual", "SF-OGD-residual", "SAOCP-residual"}
    dist = {"Base", "Static-PIT", "Rolling-PIT-ECDF", "Weighted-PIT-ECDF"}
    for data_set, sub in summary.groupby("Data set", sort=False):
        for label, methods in [
            ("OCC-Fixed", {"OCC-Fixed"}),
            ("OCC-Adaptive", {"OCC-Adaptive"}),
            ("Descriptive distributional envelope", dist),
            ("Descriptive scalar envelope", scalar),
        ]:
            candidates = sub[sub["method"].isin(methods)].copy()
            near = candidates[candidates["coverage"].between(0.875, 0.925)]
            if near.empty:
                near = candidates.assign(err=(candidates["coverage"] - 0.90).abs()).sort_values(["err", "mean_width"]).head(1)
            else:
                near = near.sort_values(["mean_width", "rolling_mean_abs_gap"]).head(1)
            row = near.iloc[0]
            rows.append(
                {
                    "Data set": data_set,
                    "Benchmark row": label,
                    "Selection status": "post-evaluation descriptive envelope" if "envelope" in label else "pre-specified method",
                    "Method": row["method"],
                    "Coverage": row["coverage"],
                    "Width": row["mean_width"],
                    "Rolling cov. gap": row["rolling_mean_abs_gap"],
                }
            )
    pd.DataFrame(rows).to_csv(paper_dir / "realdata_benchmark_envelope_table.csv", index=False)


def plot_tradeoff(summary: pd.DataFrame, fig_dir: Path = FIGDIR) -> None:
    labels = list(summary["Data set"].drop_duplicates())
    n_panels = max(1, len(labels))
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.0 * n_rows), squeeze=False)
    axes = axes.ravel()
    dist = {"Base", "Static-PIT", "Rolling-PIT-ECDF", "Weighted-PIT-ECDF"}
    scalar = {"DCP-score", "ACI-residual", "NExCP-residual", "PID-residual-projected", "FACI-residual", "SF-OGD-residual", "SAOCP-residual"}
    for ax, label in zip(axes, labels):
        sub = summary[summary["Data set"].eq(label)]
        for _, row in sub.iterrows():
            method = row["method"]
            if method == "OCC-Fixed":
                ax.scatter(row["mean_width"], row["coverage"], s=90, marker="o", color="#1f77b4", label="OCC-Fixed", zorder=4)
            elif method == "OCC-Adaptive":
                ax.scatter(row["mean_width"], row["coverage"], s=90, marker="D", facecolors="white", edgecolors="#60a5fa", linewidths=2.0, label="OCC-Adaptive", zorder=4)
            elif method in dist:
                ax.scatter(row["mean_width"], row["coverage"], s=44, marker="s", color="#6b7280", alpha=0.82, label="Distributional baselines")
            elif method in scalar:
                ax.scatter(row["mean_width"], row["coverage"], s=50, marker="^", color="#d97706", alpha=0.82, label="Scalar interval baselines")
        ax.axhline(0.90, color="#111827", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.set_title(label)
        ax.set_xlabel("Mean interval width")
        ax.set_ylabel("Empirical coverage")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
    for ax in axes[len(labels):]:
        ax.set_visible(False)
    handles, legend_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, item in zip(h, l):
            if item not in legend_labels:
                handles.append(handle)
                legend_labels.append(item)
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.12, wspace=0.25, hspace=0.34)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "realdata_tradeoff_main.png", dpi=230)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=[spec["label"] for spec in RUNS])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--without-residual", action="store_true", help="Run only full-CDF methods for a fast protocol smoke test.")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--paper-dir", type=Path, default=PAPER)
    parser.add_argument("--fig-dir", type=Path, default=FIGDIR)
    parser.add_argument("--model-type", choices=["ridge", "hgb"], default=None)
    parser.add_argument("--allow-partial", action="store_true", help="Permit exploratory partial output; disabled for formal runs.")
    args = parser.parse_args()
    if args.allow_partial:
        args.paper_dir = args.paper_dir / "exploratory_partial"
        args.fig_dir = args.fig_dir / "exploratory_partial"
    selected = {item.lower() for item in args.datasets}
    summary = run_all(
        selected_labels=selected,
        data_dir=args.data_dir,
        max_eval_override=args.max_eval,
        include_residual=not args.without_residual,
        out_dir=args.out_dir,
        paper_dir=args.paper_dir,
        fig_dir=args.fig_dir,
        model_type_override=args.model_type,
        fail_on_partial=not args.allow_partial,
    )
    build_tables(summary, paper_dir=args.paper_dir)
    plot_tradeoff(summary, fig_dir=args.fig_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
