"""Refresh manuscript-facing Fixed/Adaptive OCC empirical assets.

This script is intentionally separate from the earlier broad search scripts.
It uses a single core method set in every displayed setting:

    Base, Static-PIT, Rolling-PIT-ECDF, Weighted-PIT-ECDF, DCP-score,
    OCC-Fixed, OCC-Adaptive, and explicitly labelled PIT-level controllers.

The goal is not to select a different OCC variant per setting.  Instead, it
shows the pre-specified fixed-memory rule and the automatic adaptive-memory
rule side by side, so the manuscript can discuss where each rule is useful.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
REPRODUCTION = ROOT / "reproduction"
for directory in (FRAMEWORK, REPRODUCTION):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from core import Step  # noqa: E402
from experiment import results_to_records, run_comparison, summary_table  # noqa: E402
from methods import (  # noqa: E402
    BaseCDF,
    DCP,
    DirectBinnedRecalibration,
    OCC,
    RollingPITECDF,
    StaticPIT,
    WeightedPITECDF,
)
from residual_baselines import residual_baselines  # noqa: E402
from paper_config import PAPER_CONFIG  # noqa: E402
from provenance import artifact_manifest, source_manifest  # noqa: E402


TARGET_ALPHA = 0.10
OUT = ROOT / "outputs" / "fixed_adaptive_refresh"
FIGDIR = ROOT / "figures"
PAPER = ROOT / "paper_ready" / "tables"
METHOD_ORDER = [
    "Base",
    "Static-PIT",
    "Rolling-PIT-ECDF",
    "DCP-score",
    "Weighted-PIT-ECDF",
    "DBR-Fixed",
    "OCC-Fixed",
    "OCC-Adaptive",
    "ACI-residual-projected",
    "NExCP-residual-empirical",
    "PID-residual-projected",
    "FACI-residual",
    "SF-OGD-residual",
    "SAOCP-residual",
]
PANEL_SCENARIOS = ["Location shift", "Gradual", "Bursty", "Abrupt"]
REPRESENTATIVE_SCENARIOS = ["Location shift", "Gradual", "Bursty"]


@dataclass(frozen=True)
class MethodConfig:
    fixed_lam: float = PAPER_CONFIG.fixed_lam
    adaptive_lam_min: float = PAPER_CONFIG.adaptive_lam_min
    adaptive_lam_max: float = PAPER_CONFIG.adaptive_lam_max
    adaptive_tau: float = PAPER_CONFIG.adaptive_tau
    adaptive_beta: float = PAPER_CONFIG.adaptive_beta
    adaptive_window: int = PAPER_CONFIG.adaptive_window
    dbr_bins: int = PAPER_CONFIG.dbr_bins
    bins: int = PAPER_CONFIG.bins
    baseline_k: int = PAPER_CONFIG.baseline_window
    weighted_history: int = PAPER_CONFIG.weighted_history
    aci_gamma: float = PAPER_CONFIG.aci_gamma


def methods_factory(config: MethodConfig) -> Callable[[], dict]:
    def factory() -> dict:
        methods = {
            "Base": BaseCDF(),
            "Static-PIT": StaticPIT(),
            "Rolling-PIT-ECDF": RollingPITECDF(K=config.baseline_k),
            "DCP-score": DCP(K=config.baseline_k),
            "Weighted-PIT-ECDF": WeightedPITECDF(
                lam=config.fixed_lam, max_buffer=config.weighted_history
            ),
            "DBR-Fixed": DirectBinnedRecalibration(
                lam=config.fixed_lam,
                bins=config.dbr_bins,
            ),
            "OCC-Fixed": OCC(
                lam=config.fixed_lam,
                bins=config.bins,
            ),
            "OCC-Adaptive": OCC(
                adaptive=True,
                bins=config.bins,
                lam_min=config.adaptive_lam_min,
                lam_max=config.adaptive_lam_max,
                tau=config.adaptive_tau,
                beta=config.adaptive_beta,
                window=config.adaptive_window,
            ),
        }
        methods.update(residual_baselines(alpha_target=TARGET_ALPHA, aci_gamma=config.aci_gamma))
        return methods

    return factory


def shift_path(scenario: str, x: np.ndarray) -> np.ndarray:
    if scenario == "no_shift":
        return np.zeros_like(x)
    if scenario == "location":
        return np.full_like(x, 0.85, dtype=float)
    if scenario == "gradual":
        return 1.45 / (1.0 + np.exp(-9.0 * (x - 0.50))) - 0.45
    if scenario == "cycle":
        return 0.55 * np.sin(2.0 * np.pi * 2.3 * x) + 0.45 * np.sin(2.0 * np.pi * 6.0 * x)
    if scenario == "bursty":
        shift = 0.15 * np.sin(2.0 * np.pi * 1.3 * x)
        for center, amp, width in [(0.28, 1.15, 0.030), (0.55, -1.00, 0.040), (0.78, 1.05, 0.028)]:
            shift += amp * np.exp(-0.5 * ((x - center) / width) ** 2)
        return shift
    if scenario == "abrupt":
        return np.select(
            [x < 0.33, x < 0.62, x < 0.82],
            [-0.35, 1.15, -0.15],
            default=0.95,
        )
    raise ValueError(scenario)


def synthetic_dataset(
    scenario: str,
    seed: int,
    n_warmup: int = 300,
    n_online: int = 1200,
) -> dict:
    rng = np.random.default_rng(seed)
    total = n_warmup + n_online
    x = np.linspace(0.0, 1.0, total)
    shift = shift_path(scenario, x)
    scale = 1.0 + 0.12 * np.sin(2.0 * np.pi * x)
    if scenario in {"bursty", "abrupt"}:
        scale += 0.10 * (np.abs(shift) > 0.8)
    if scenario == "location":
        eps = rng.normal(size=total)
    else:
        eps = rng.standard_t(df=6, size=total) / np.sqrt(6.0 / 4.0)
    y = shift + scale * eps
    u = norm.cdf(y)

    def inv(p):
        p_arr = np.asarray(p, dtype=float)
        out = norm.ppf(np.clip(p_arr, 0.0, 1.0))
        return float(out) if np.ndim(p) == 0 else out

    steps = [Step(t=int(i), u=float(u[i]), y=float(y[i]), F_t_inv=inv) for i in range(total)]
    return {
        "warmup": steps[:n_warmup],
        "pre_online": [],
        "online": steps[n_warmup:],
        "meta": {
            "dataset": f"synthetic_{scenario}",
            "label": f"Synthetic {scenario}",
            "scenario": scenario,
            "seed": int(seed),
            "n_warmup": n_warmup,
            "n_online": n_online,
        },
    }


def display_name(scenario: str) -> str:
    return {
        "no_shift": "No shift",
        "location": "Location shift",
        "gradual": "Gradual",
        "cycle": "Cyclic",
        "bursty": "Bursty",
        "abrupt": "Abrupt",
    }[scenario]


def run_synthetic(
    seed_count: int = 12,
    out_dir: Path = OUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = MethodConfig()
    rows = []
    records_list = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for scenario in ["no_shift", "location", "gradual", "cycle", "bursty", "abrupt"]:
        for seed in range(seed_count):
            ds = synthetic_dataset(scenario, seed)
            results = run_comparison(
                ds,
                alphas=(TARGET_ALPHA,),
                methods_factory=methods_factory(config),
            )
            observed_methods = set(results)
            expected_methods = set(METHOD_ORDER)
            if observed_methods != expected_methods:
                raise RuntimeError(
                    f"Incomplete method registry for {scenario=}, {seed=}: "
                    f"expected {sorted(expected_methods)}, got {sorted(observed_methods)}."
                )
            y = np.array([s.y for s in ds["online"]], dtype=float)
            summary = summary_table(results, y, alpha_main=TARGET_ALPHA, rolling_window=250).reset_index()
            summary.insert(0, "seed", seed)
            summary.insert(0, "scenario", display_name(scenario))
            rows.append(summary)
            rec = results_to_records(ds, results, (TARGET_ALPHA,))
            rec.insert(0, "seed", seed)
            rec.insert(0, "scenario", display_name(scenario))
            records_list.append(rec)
    summary_all = pd.concat(rows, ignore_index=True)
    records_all = pd.concat(records_list, ignore_index=True)
    summary_all.to_csv(out_dir / "synthetic_summary_by_seed.csv", index=False)
    records_all.to_csv(out_dir / "synthetic_records.csv", index=False)
    (out_dir / "method_config.json").write_text(json.dumps({
        **asdict(config),
        "seed_count": seed_count,
        "method_order": METHOD_ORDER,
        "selection_policy": "All configured scenarios, seeds and methods are retained without result filtering.",
    }, indent=2), encoding="utf-8")
    return summary_all, records_all


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    agg = (
        summary.groupby(["scenario", "method"], as_index=False)
        .agg(
            coverage=("coverage", "mean"),
            coverage_sd=("coverage", "std"),
            width=("mean_width", "mean"),
            width_sd=("mean_width", "std"),
            rolling_gap=("rolling_mean_abs_gap", "mean"),
            rolling_gap_sd=("rolling_mean_abs_gap", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    for metric in ("coverage", "width", "rolling_gap"):
        agg[f"{metric}_ci95_halfwidth"] = (
            1.96 * agg[f"{metric}_sd"].fillna(0.0) / np.sqrt(agg["n_seeds"])
        )
    agg["method"] = pd.Categorical(agg["method"], METHOD_ORDER, ordered=True)
    agg["scenario"] = pd.Categorical(
        agg["scenario"],
        ["No shift", "Location shift", "Gradual", "Cyclic", "Bursty", "Abrupt"],
        ordered=True,
    )
    return agg.sort_values(["scenario", "method"]).reset_index(drop=True)


def make_main_tables(agg: pd.DataFrame, paper_dir: Path = PAPER) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    main = agg[agg["method"].isin(["OCC-Fixed", "OCC-Adaptive"])].copy()
    main = main.pivot(index="scenario", columns="method", values=["coverage", "width", "rolling_gap"])
    main.columns = [f"{metric}_{method.replace('OCC-', '').lower()}" for metric, method in main.columns]
    main = main.reset_index().rename(
        columns={
            "scenario": "Scenario",
            "coverage_fixed": "Fixed coverage",
            "width_fixed": "Fixed width",
            "rolling_gap_fixed": "Fixed roll. gap",
            "coverage_adaptive": "Adaptive coverage",
            "width_adaptive": "Adaptive width",
            "rolling_gap_adaptive": "Adaptive roll. gap",
        }
    )
    main.to_csv(paper_dir / "fixed_adaptive_main_sim_table.csv", index=False)

    rep = agg[
        agg["scenario"].isin(REPRESENTATIVE_SCENARIOS)
        & agg["method"].isin(METHOD_ORDER)
    ].copy()
    rep = rep.rename(
        columns={
            "scenario": "Scenario",
            "method": "Method",
            "coverage": "Coverage",
            "width": "Width",
            "rolling_gap": "Rolling cov. gap",
        }
    )
    rep.to_csv(paper_dir / "fixed_adaptive_representative_method_table.csv", index=False)

    reduction_rows = []
    scalar = {"ACI-residual-projected", "NExCP-residual-empirical", "PID-residual-projected", "FACI-residual", "SF-OGD-residual", "SAOCP-residual", "DCP-score"}
    dist = {"Base", "Static-PIT", "Rolling-PIT-ECDF", "Weighted-PIT-ECDF"}
    for scenario in PANEL_SCENARIOS:
        sub = agg[agg["scenario"].eq(scenario)]
        for occ in ["OCC-Fixed", "OCC-Adaptive"]:
            occ_row = sub[sub["method"].eq(occ)].iloc[0]
            row = {
                "Scenario": scenario,
                "OCC variant": occ.replace("OCC-", ""),
                "OCC cov.": occ_row["coverage"],
                "OCC width": occ_row["width"],
            }
            for label, methods in [("scalar", scalar), ("dist", dist)]:
                candidates = sub[sub["method"].isin(methods)].copy()
                near = candidates[candidates["coverage"].between(0.875, 0.925)]
                if near.empty:
                    near = candidates.assign(err=(candidates["coverage"] - 0.90).abs()).sort_values(["err", "width"]).head(1)
                else:
                    near = near.sort_values("width").head(1)
                base = near.iloc[0]
                row[f"Descriptive selected {label}"] = base["method"]
                row[f"Descriptive reduction vs {label}"] = 100.0 * (base["width"] - occ_row["width"]) / base["width"]
            reduction_rows.append(row)
    pd.DataFrame(reduction_rows).to_csv(paper_dir / "fixed_adaptive_width_reduction_table.csv", index=False)

    envelope_rows = []
    for scenario in ["No shift", "Location shift", "Gradual", "Cyclic", "Bursty", "Abrupt"]:
        sub = agg[agg["scenario"].eq(scenario)]
        for label, methods in [
            ("OCC-Fixed", {"OCC-Fixed"}),
            ("OCC-Adaptive", {"OCC-Adaptive"}),
            ("Descriptive distributional envelope", dist),
            ("Descriptive scalar envelope", scalar),
        ]:
            candidates = sub[sub["method"].isin(methods)].copy()
            near = candidates[candidates["coverage"].between(0.875, 0.925)]
            if near.empty:
                near = candidates.assign(err=(candidates["coverage"] - 0.90).abs()).sort_values(["err", "width"]).head(1)
            else:
                near = near.sort_values(["width", "rolling_gap"]).head(1)
            row = near.iloc[0]
            envelope_rows.append(
                {
                    "Scenario": scenario,
                    "Benchmark row": label,
                    "Selection status": "post-evaluation descriptive envelope" if "envelope" in label else "pre-specified method",
                    "Method": row["method"],
                    "Coverage": row["coverage"],
                    "Width": row["width"],
                    "Rolling cov. gap": row["rolling_gap"],
                }
            )
    pd.DataFrame(envelope_rows).to_csv(paper_dir / "fixed_adaptive_benchmark_envelope_table.csv", index=False)


def plot_tradeoff(agg: pd.DataFrame, out_path: Path) -> None:
    colors = {
        "OCC-Fixed": "#1f77b4",
        "OCC-Adaptive": "#60a5fa",
        "Distributional": "#6b7280",
        "Scalar": "#d97706",
    }
    markers = {
        "OCC-Fixed": "o",
        "OCC-Adaptive": "o",
        "Distributional": "s",
        "Scalar": "^",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.4))
    axes = axes.ravel()
    for ax, scenario in zip(axes, ["No shift", "Location shift", "Gradual", "Cyclic", "Bursty", "Abrupt"]):
        sub = agg[agg["scenario"].eq(scenario)].copy()
        for _, row in sub.iterrows():
            method = str(row["method"])
            if method in {"OCC-Fixed", "OCC-Adaptive"}:
                label = method
                face = colors[method] if method == "OCC-Fixed" else "white"
                edge = colors[method]
                size = 80
                linewidth = 1.8
            elif method in {"Base", "Static-PIT", "Rolling-PIT-ECDF", "Weighted-PIT-ECDF"}:
                label = "Distributional baselines"
                face = edge = colors["Distributional"]
                size = 42
                linewidth = 0.5
            else:
                label = "Scalar interval baselines"
                face = edge = colors["Scalar"]
                size = 48
                linewidth = 0.5
            ax.scatter(
                row["width"],
                row["coverage"],
                s=size,
                marker="D" if method == "OCC-Adaptive" else markers.get(method, markers["Distributional" if label.startswith("Distributional") else "Scalar"]),
                facecolors=face,
                edgecolors=edge,
                linewidths=linewidth,
                alpha=0.9,
                label=label,
            )
        ax.axhline(0.90, color="#111827", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.set_title(scenario)
        ax.set_xlabel("Mean interval width")
        ax.set_ylabel("Empirical coverage")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.11, wspace=0.24, hspace=0.34)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def plot_width_reduction(path: Path, out_path: Path) -> None:
    table = pd.read_csv(path)
    table["label"] = table["Scenario"] + "\n" + table["OCC variant"]
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(11.0, 4.9))
    ax.bar(x - 0.18, table["Descriptive reduction vs scalar"], width=0.34, color="#d97706", alpha=0.86, label="vs descriptive scalar envelope")
    ax.bar(x + 0.18, table["Descriptive reduction vs dist"], width=0.34, color="#6b7280", alpha=0.86, label="vs descriptive distributional envelope")
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(table["label"], fontsize=8)
    ax.set_ylabel("Width reduction (%)")
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.95, bottom=0.30)
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--paper-dir", type=Path, default=PAPER)
    parser.add_argument("--fig-dir", type=Path, default=FIGDIR)
    args = parser.parse_args()
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    summary, _ = run_synthetic(seed_count=args.seed_count, out_dir=args.out_dir)
    agg = aggregate(summary)
    agg.to_csv(args.paper_dir / "fixed_adaptive_synthetic_aggregated.csv", index=False)
    make_main_tables(agg, paper_dir=args.paper_dir)
    plot_tradeoff(agg, args.fig_dir / "simulation_tradeoff_main.png")
    plot_width_reduction(
        args.paper_dir / "fixed_adaptive_width_reduction_table.csv",
        args.fig_dir / "simulation_width_reduction_main.png",
    )
    provenance = source_manifest(ROOT, [
        "requirements-lock.txt",
        "framework/core.py",
        "framework/experiment.py",
        "framework/methods.py",
        "framework/residual_baselines.py",
        "framework/paper_config.py",
        "scripts/fixed_adaptive_refresh.py",
    ])
    provenance["artifacts"] = artifact_manifest({
        "synthetic_records": args.out_dir / "synthetic_records.csv",
        "synthetic_summary_by_seed": args.out_dir / "synthetic_summary_by_seed.csv",
        "method_config": args.out_dir / "method_config.json",
        "aggregated_table": args.paper_dir / "fixed_adaptive_synthetic_aggregated.csv",
        "tradeoff_figure": args.fig_dir / "simulation_tradeoff_main.png",
    })
    (args.out_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
