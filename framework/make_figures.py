"""Generate diagnostic figures for completed OCC SOTA comparison runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = {
    "Base": "#7f7f7f",
    "BaseCDF": "#7f7f7f",
    "Static-PIT": "#4d4d4d",
    "OCC-Fixed": "#1f77b4",
    "OCC-Adaptive": "#1f77b4",
    "Rolling-IDR": "#17becf",
    "DCP": "#9467bd",
    "Weighted-DCP": "#8c564b",
    "ACI": "#bcbd22",
}


def _method_order(method: str) -> tuple[int, str]:
    order = {
        "Base": 0,
        "BaseCDF": 0,
        "Static-PIT": 1,
        "OCC-Fixed": 2,
        "OCC-Adaptive": 3,
        "Rolling-IDR": 4,
        "DCP": 5,
        "Weighted-DCP": 6,
        "ACI": 7,
    }
    return order.get(method, 99), method


def _rolling(values: pd.Series, window: int) -> pd.Series:
    min_periods = max(20, window // 5)
    return values.rolling(window, min_periods=min_periods).mean()


def create_dataset_figures(dataset_dir: str | Path, alpha: float = 0.10, window: int | None = None) -> list[Path]:
    import matplotlib.pyplot as plt

    root = Path(dataset_dir)
    records = pd.read_csv(root / "records.csv")
    sub = records[records["alpha"].eq(alpha)].copy()
    if sub.empty:
        return []
    if window is None:
        n_time = sub["time_index"].nunique()
        window = min(500, max(50, n_time // 5))
    target = 1.0 - alpha
    label = str(sub["dataset_label"].dropna().iloc[0]) if "dataset_label" in sub else root.name
    methods = sorted(sub["method"].unique(), key=_method_order)
    saved: list[Path] = []

    # Coverage-width tradeoff.
    plot = (
        sub.groupby("method", as_index=False)
        .agg(coverage=("covered", "mean"), width=("width", "mean"))
        .sort_values("method", key=lambda s: s.map(_method_order))
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for _, row in plot.iterrows():
        method = row["method"]
        ax.scatter(
            row["width"],
            row["coverage"],
            s=70 if str(method).startswith("OCC") else 45,
            color=COLORS.get(method, "black"),
            label=method,
        )
        ax.text(row["width"], row["coverage"], "  " + method, va="center", fontsize=8)
    ax.axhline(target, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mean interval width")
    ax.set_ylabel(f"{int(target * 100)}% coverage")
    ax.set_title(f"{label}: coverage-width tradeoff")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = root / "coverage_width.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    saved.append(path)

    # Rolling coverage.
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for method in methods:
        g = sub[sub["method"].eq(method)].sort_values("time_index")
        roll = _rolling(g["covered"], window)
        ax.plot(
            g["time_index"],
            roll,
            label=method,
            color=COLORS.get(method, None),
            linewidth=2.2 if method.startswith("OCC") else 1.3,
            alpha=0.95 if method.startswith("OCC") else 0.80,
        )
    ax.axhline(target, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Time index")
    ax.set_ylabel(f"Rolling {int(target * 100)}% coverage")
    ax.set_title(f"{label}: rolling coverage, window={window}")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(True, alpha=0.20)
    fig.tight_layout()
    path = root / "rolling_coverage.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    saved.append(path)

    # Rolling absolute coverage gap.
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for method in methods:
        g = sub[sub["method"].eq(method)].sort_values("time_index")
        roll = _rolling(g["covered"], window)
        ax.plot(
            g["time_index"],
            (roll - target).abs(),
            label=method,
            color=COLORS.get(method, None),
            linewidth=2.2 if method.startswith("OCC") else 1.3,
            alpha=0.95 if method.startswith("OCC") else 0.80,
        )
    ax.set_xlabel("Time index")
    ax.set_ylabel("Rolling absolute coverage gap")
    ax.set_title(f"{label}: rolling coverage error, window={window}")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(True, alpha=0.20)
    fig.tight_layout()
    path = root / "rolling_abs_gap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    saved.append(path)

    # Center shift tracking for map-producing methods.
    if "center_shift" in sub.columns:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        for method in methods:
            if method == "ACI":
                continue
            g = sub[sub["method"].eq(method)].sort_values("time_index")
            shift = _rolling(g["center_shift"], window)
            ax.plot(
                g["time_index"],
                shift,
                label=method,
                color=COLORS.get(method, None),
                linewidth=2.2 if method.startswith("OCC") else 1.3,
                alpha=0.95 if method.startswith("OCC") else 0.75,
            )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time index")
        ax.set_ylabel("Rolling interval-center shift")
        ax.set_title(f"{label}: calibrated center movement, window={window}")
        ax.legend(ncol=3, fontsize=8, frameon=False)
        ax.grid(True, alpha=0.20)
        fig.tight_layout()
        path = root / "center_shift.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        saved.append(path)

    pd.DataFrame({"figure": [str(p) for p in saved]}).to_csv(root / "figure_manifest.csv", index=False)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--window", type=int, default=None)
    args = parser.parse_args()
    root = Path(args.root)
    if (root / "records.csv").exists():
        create_dataset_figures(root, alpha=args.alpha, window=args.window)
        return
    for child in sorted(root.rglob("records.csv")):
        create_dataset_figures(child.parent, alpha=args.alpha, window=args.window)


if __name__ == "__main__":
    main()
