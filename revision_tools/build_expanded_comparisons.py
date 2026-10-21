"""Compute paper comparisons from checked prediction arrays.

The interval score evaluates the saved predictions without tuning the methods.
The input configuration, code and prediction files must pass integrity checks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("framework", "scripts", "reproduction")]
from adaptive_memory_realdata_study import ALPHAS, DATASETS, methods_factory
from metrics import rolling_coverage
from revision_integrity import completed_payload_names, file_digest, read_json, require_equal, verify_complete
from run_revision import SCENARIOS, expected_cell, fingerprint, main_protocol

AM = "AMDBR-core-noidentity"
DBR = "DBR-Fixed"
ORDER = list(methods_factory())
MAIN = [m for m in ORDER if m not in {"OCC-Fixed", "OCC-Adaptive"}]
FULL_CDF = ["Base", "Static-PIT", "Rolling-PIT-ECDF", "Weighted-PIT-ECDF", DBR, AM]
LABELS = {
    "Base": "Base", "Static-PIT": "Static PIT", "Rolling-PIT-ECDF": "Rolling ECDF",
    "Weighted-PIT-ECDF": "Weighted ECDF", "DCP-score": "DCP score", DBR: "DBR-Fixed",
    "OCC-Fixed": "Rank/PAV fixed", "OCC-Adaptive": "Rank/PAV adaptive", AM: "AM-DBR",
    "ACI-residual-projected": "ACI projected", "NExCP-residual-empirical": "NExCP empirical",
    "PID-residual-projected": "PID projected", "FACI-residual": "FACI",
    "SF-OGD-residual": "SF-OGD", "SAOCP-residual": "SAOCP",
}
SCENARIO_NAMES = dict(zip(SCENARIOS, ["No location shift", "Location shift", "Gradual", "Cyclic", "Bursty", "Abrupt"]))
DATASET_NAMES = [s["label"] for s in DATASETS]
# The source loaders use these response columns, without standardisation.
# Temperature sources specify degrees Celsius; Metro counts vehicles per hour;
# the pinned OPSD daily Consumption series uses GWh.
UNITS = {"Seattle": "degrees C", "Phoenix": "degrees C", "Jena": "degrees C", "Metro": "vehicles/hour", "OPSD": "GWh/day"}
SEEDS = list(range(6100, 6108))
METRICS = ["coverage", "mean_width", "rolling_mean_abs_gap", "avg_gap", "IS90", "PIT_CRPS"]


def close(actual, expected, name):
    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True):
        raise AssertionError(f"{name}: recomputed {actual!r}, recorded {expected!r}")


def interval_score(y, lower, upper, alpha=0.1):
    """Proper central-interval score; infinity is retained, never clipped."""
    return (upper - lower) + (2.0 / alpha) * np.maximum(lower - y, 0.0) + (2.0 / alpha) * np.maximum(y - upper, 0.0)


def source_authorisation(results):
    """Require predictions and configuration from this package."""
    protocol = read_json(results / "protocol.json")
    current = fingerprint()
    if protocol.get("source_sha256") != current:
        raise ValueError("Saved predictions were produced by different code. Run this package in a fresh output directory.")
    expected_protocol = main_protocol(SEEDS, list(SCENARIOS), DATASET_NAMES)
    require_equal(protocol, expected_protocol, "Expanded comparison main protocol")
    return current


def collect(results):
    source_sha = source_authorisation(results)
    hashes = {"protocol.json": file_digest(results / "protocol.json")}
    rows, counts = [], {"cells": 0, "method_runs": 0, "interval_predictions_across_levels": 0}
    cells = [("synthetic", s, seed) for s in SCENARIOS for seed in SEEDS] + [("real", d, -1) for d in DATASET_NAMES]
    for kind, name, seed in cells:
        relative = f"{kind}/{name}_{seed}" if kind == "synthetic" else f"{kind}/{name}"
        dest = results / relative
        receipt_hash = file_digest(dest / "COMPLETE.json")
        expected = expected_cell(kind, name, seed, source_sha=source_sha)
        receipt = verify_complete(dest, expected, completed_payload_names(ORDER))
        hashes[f"{relative}/COMPLETE.json"] = receipt_hash
        for filename, record in receipt["payloads"].items():
            hashes[f"{relative}/{filename}"] = record["sha256"]
        summary = pd.read_csv(dest / "summary.csv")
        if summary.method.duplicated().any() or set(summary.method) != set(ORDER):
            raise ValueError(f"Incomplete or duplicated method registry: {relative}")
        identity_key = "scenario" if kind == "synthetic" else "data_set"
        if set(summary[identity_key]) != {name} or set(summary.seed) != {seed}:
            raise ValueError(f"Summary identity mismatch: {relative}")
        reference_y = reference_t = reference_u = None
        for method in ORDER:
            original = summary.set_index("method").loc[method]
            with np.load(dest / f"{method}.npz", allow_pickle=False) as arrays:
                y, t, u = arrays["y"], arrays["t"], arrays["u"]
                alpha, lower, upper = arrays["alphas"], arrays["lower"], arrays["upper"]
                np.testing.assert_array_equal(alpha, ALPHAS)
                n = len(y)
                if lower.shape != (len(ALPHAS), n) or upper.shape != lower.shape or not n:
                    raise ValueError(f"Invalid interval dimensions: {relative}/{method}")
                if not np.isfinite(y).all() or not np.isfinite(u).all() or np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper):
                    raise ValueError(f"Invalid prediction arrays: {relative}/{method}")
                if reference_y is None:
                    reference_y, reference_t, reference_u = y.copy(), t.copy(), u.copy()
                else:
                    for actual, reference in [(y, reference_y), (t, reference_t), (u, reference_u)]:
                        np.testing.assert_array_equal(actual, reference)
                coverage = ((lower <= y) & (y <= upper)).mean(axis=1)
                error = np.abs(coverage - (1.0 - alpha))
                j = int(np.flatnonzero(alpha == 0.1)[0])
                widths = upper[j] - lower[j]
                rolling = rolling_coverage(y, lower[j], upper[j], window=500)
                local = np.abs(rolling[np.isfinite(rolling)] - 0.9)
                score = interval_score(y, lower[j], upper[j])
                if np.isnan(score).any() or np.any(score < 0.0):
                    raise ValueError(f"Invalid interval scores: {relative}/{method}")
                row = {"kind": kind, "group": name, "seed": seed, "method": method, "n": n,
                       "coverage": float(coverage[j]), "mean_width": float(widths.mean()),
                       "median_width": float(np.median(widths)), "rolling_mean_abs_gap": float(local.mean()),
                       "rolling_max_abs_gap": float(local.max()), "rolling_last_abs_gap": float(local[-1]),
                       "avg_gap": float(error.mean()), "max_gap": float(error.max()), "IS90": float(score.mean()),
                       "PIT_CRPS": float("nan")}
                for i, a in enumerate(alpha):
                    row[f"coverage@alpha={a:g}"] = float(coverage[i])
                    row[f"err@alpha={a:g}"] = float(error[i])
                    close(error[i], original[f"err@alpha={a:g}"], f"{relative}/{method}/error/{a}")
                if "pit_crps" in arrays:
                    pit = arrays["pit_crps"]
                    if pit.shape != (n,) or not np.isfinite(pit).all() or (pit < -1e-12).any() or (pit > 1 + 1e-12).any():
                        raise ValueError(f"Invalid PIT scores: {relative}/{method}")
                    row["PIT_CRPS"] = float(pit.mean())
                for metric in METRICS + ["median_width", "max_gap", "rolling_max_abs_gap", "rolling_last_abs_gap"]:
                    if metric != "IS90":
                        close(row[metric], original[metric], f"{relative}/{method}/{metric}")
                rows.append(row)
                counts["method_runs"] += 1
                counts["interval_predictions_across_levels"] += n * len(ALPHAS)
        counts["cells"] += 1
    if counts["cells"] != 53 or counts["method_runs"] != 795:
        raise ValueError(f"Incomplete expanded comparison: {counts}")
    return pd.DataFrame(rows), hashes, counts


def paired_intervals(frame):
    rows = []
    syn = frame[frame.kind == "synthetic"]
    for group in ["Overall"] + list(SCENARIOS):
        selected = syn if group == "Overall" else syn[syn.group == group]
        for metric in ["mean_width", "IS90", "PIT_CRPS", "avg_gap", "rolling_mean_abs_gap"]:
            means = selected.groupby(["seed", "method"])[metric].mean().unstack("method")
            for other in ORDER:
                if other == AM:
                    continue
                delta = (means[AM] - means[other]).to_numpy()
                if np.isnan(delta).all():
                    continue
                if not np.isfinite(delta).all() or len(delta) != 8:
                    raise ValueError(f"Paired interval requires eight finite seed differences: {group}/{other}/{metric}")
                average = float(delta.mean())
                radius = float(student_t.ppf(0.975, 7) * delta.std(ddof=1) / np.sqrt(8))
                rows.append({"group": group, "metric": metric, "left": AM, "right": other,
                             "difference": average, "low": average - radius, "high": average + radius,
                             "seed_count": 8, "negative_seed_count": int((delta < 0).sum()),
                             "interpretation": "descriptive paired seed interval; no multiplicity adjustment"})
    return pd.DataFrame(rows)


def number(value, digits=3, bold=False):
    if pd.isna(value):
        return "-"
    if np.isposinf(value):
        return r"$\infty$"
    value = f"{value:.{digits}f}"
    return r"\textbf{" + value + "}" if bold else value


def tabular(headers, rows, align=None):
    align = align or "l" + "r" * (len(headers) - 1)
    return (r"\begin{tabular}{" + align + "}\n\\toprule\n" + " & ".join(headers) + r" \\" + "\n\\midrule\n" +
            "\n".join(" & ".join(row) + r" \\" for row in rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table(caption, label, body, small="small"):
    return ("\\clearpage\n\\begin{table}[p]\n\\centering\\" + small + "\n\\setlength{\\tabcolsep}{3.5pt}\n" +
            "\\caption{" + caption + "}\\label{" + label + "}\n" + body + "\\end{table}\n")


def paired_is90_table(paired):
    """Render all comparator intervals from the paired-seed calculation."""
    selected = paired[(paired.group == "Overall") & (paired.metric == "IS90")]
    comparators = [method for method in ORDER if method != AM]
    if (selected.right.duplicated().any() or set(selected.right) != set(comparators)
            or not selected.left.eq(AM).all() or not selected.seed_count.eq(8).all()):
        raise ValueError("Paired IS90 table requires all fourteen comparators and eight paired seeds")
    selected = selected.set_index("right")
    rows = [[LABELS[method]] + [number(selected.loc[method, col]) for col in
            ["difference", "low", "high"]] + [str(int(selected.loc[method, "seed_count"]))]
            for method in comparators]
    caption = (r"Paired simulation differences in the mean 90\% interval score, with AM-DBR minus each comparator. "
               r"Each difference averages the six scenario means within each of eight paired seeds; "
               r"the interval is a 95\% Student interval over the eight seed means. Lower values favour AM-DBR. "
               r"These intervals are descriptive and have no multiplicity adjustment.")
    return table(caption, "tab:paired-is90", tabular(
        ["Comparator", "Difference", r"95\% lower", r"95\% upper", "$n$"], rows), small="footnotesize")


def tables(frame):
    syn = frame[frame.kind == "synthetic"]
    real = frame[frame.kind == "real"]
    overall = syn.groupby("method")[METRICS].mean()
    rows = []
    for method in MAIN:
        x = overall.loc[method]
        rows.append([LABELS[method]] + [number(x[col], 4 if col in {"avg_gap", "rolling_mean_abs_gap", "PIT_CRPS"} else 3) for col in METRICS])
    main = table(r"Simulation means over six scenarios and eight seeds. Coverage, width, local gap and interval score (IS) are for 90\% intervals; multilevel gap averages absolute coverage error over five nominal levels. Lower IS and PIT score are better. A dash denotes that the method does not issue a full CDF.", "tab:sim", tabular(["Method", "Coverage", "Width", "Local gap", "Multi. gap", "IS", "PIT score"], rows))
    values = syn.groupby(["method", "group"]).IS90.mean().unstack("group")
    minima = values.loc[MAIN].min(axis=0)
    rows = [[LABELS[m]] + [number(values.loc[m, s], bold=bool(np.isclose(values.loc[m, s], minima[s], rtol=0, atol=1e-12))) for s in SCENARIOS] for m in MAIN]
    main += table(r"Simulation 90\% interval score by scenario, averaging all eight seeds. Bold marks the lowest observed mean in each column among the thirteen displayed methods; it is not a significance test.", "tab:sim-detail", tabular(["Method", "No shift", "Location", "Gradual", "Cyclic", "Bursty", "Abrupt"], rows))
    values = real.set_index(["group", "method"])
    rows = []
    for dataset in DATASET_NAMES:
        scores = [values.loc[(dataset, m), "PIT_CRPS"] for m in FULL_CDF]
        minimum = min(scores)
        rows.append([dataset] + [number(v, 4, np.isclose(v, minimum, rtol=0, atol=1e-12)) for v in scores])
    main += table("PIT scores for all six primary full-CDF methods in the five real-data main windows. Bold marks the lowest observed score in each row.", "tab:real", tabular(["Dataset", "Base", "Static", "Rolling", "Weighted", "DBR-Fixed", "AM-DBR"], rows))
    minima = {d: min(values.loc[(d, m), "IS90"] for m in MAIN) for d in DATASET_NAMES}
    rows = [[LABELS[m]] + [number(values.loc[(d, m), "IS90"], bold=np.isclose(values.loc[(d, m), "IS90"], minima[d], rtol=0, atol=1e-12)) for d in DATASET_NAMES] for m in MAIN]
    main += table(r"Real-data 90\% interval score for all thirteen primary methods. Each column retains its response units; scores are not averaged across datasets. Bold marks the lowest observed score in a column.", "tab:real-detail", tabular(["Method"] + DATASET_NAMES, rows))
    supplementary = ""
    for kind, groups in [("synthetic", list(SCENARIOS)), ("real", DATASET_NAMES)]:
        for group in groups:
            values = frame[(frame.kind == kind) & (frame.group == group)].groupby("method").mean(numeric_only=True)
            title = SCENARIO_NAMES[group] if kind == "synthetic" else group
            label = f"tab:scenario-{group}" if kind == "synthetic" else f"tab:data_set-{group}"
            rows = [[LABELS[m]] + [number(values.loc[m, metric], 4 if metric == "PIT_CRPS" else 3) for metric in ["coverage", "mean_width", "rolling_mean_abs_gap", "IS90", "PIT_CRPS"]] for m in ORDER]
            body = r"\textbf{Panel A: 90\% intervals and full-distribution score}\par\smallskip" + "\n"
            body += tabular(["Method", "Coverage", "Width", "Local gap", "IS", "PIT score"], rows)
            rows = [[LABELS[m]] + [number(values.loc[m, f"err@alpha={a:g}"], 3) for a in ALPHAS] for m in ORDER]
            body += r"\par\medskip\textbf{Panel B: Absolute coverage error at all five levels}\par\smallskip" + "\n"
            body += tabular(["Method"] + [fr"$e_{{{a:g}}}$" for a in ALPHAS], rows)
            caption = (title + r": all fifteen implementations. IS is the 90\% interval score; local gap is the rolling absolute coverage error with a 500-observation window. " + ("Entries average eight simulation seeds." if kind == "synthetic" else "Entries describe one chronological main window.") + r" PIT scores apply to full-CDF methods; a dash denotes an interval-only output.")
            supplementary += table(caption, label, body, small="footnotesize")
    return main, supplementary


def marker_styles():
    markers = ["o", "s", "D", "^", "v", "P", "X", "<", ">", "h", "p", "8", "*"]
    palette = plt.get_cmap("tab20").colors
    styles = {m: {"color": palette[(i * 2) % 20], "marker": markers[i], "s": 43} for i, m in enumerate(MAIN)}
    styles[AM] = {"color": "#c82627", "marker": "*", "s": 170}
    # Keep Base, DBR-Fixed and AM-DBR visually distinct even when points are
    # close. AM-DBR is the only red star; DBR-Fixed is navy; Base remains the
    # first tab20 colour. Marker shape and numbered labels provide a second
    # cue for grayscale printing.
    styles[DBR] = {"color": "#123b5d", "marker": "o", "s": 62}
    return styles


def add_number_labels(fig, ax, coordinates):
    """Move labels, never data; leader lines preserve exact point positions."""
    renderer = fig.canvas.get_renderer()
    occupied = []
    axes_box = ax.get_window_extent(renderer).padded(-2)
    points = [ax.transData.transform((x, y)) for _, x, y in coordinates]
    candidates = [(dx, dy) for r in (10, 18, 28, 40, 56, 76) for dx, dy in [(r, r), (r, -r), (-r, r), (-r, -r), (r, 0), (-r, 0), (0, r), (0, -r)]]
    # Densest points receive labels first so isolated points do not use their space.
    density_order = sorted(range(len(points)), key=lambda i: sum(np.linalg.norm(points[i] - p) < 35 for p in points), reverse=True)
    for i in density_order:
        method, x, y = coordinates[i]
        label = str(MAIN.index(method) + 1)
        best = None
        for dx, dy in candidates:
            annotation = ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha="center", va="center", fontsize=8.0, fontweight="bold" if method == AM else "normal")
            box = annotation.get_window_extent(renderer).padded(2.2)
            annotation.remove()
            overlaps = sum(box.overlaps(other) for other in occupied)
            hidden_points = sum(box.contains(*p) for p in points)
            outside = not (axes_box.contains(box.x0, box.y0) and axes_box.contains(box.x1, box.y1))
            cost = overlaps * 20000 + hidden_points * 2000 + outside * 100000 + abs(dx) + abs(dy)
            if best is None or cost < best[0]:
                best = (cost, dx, dy, box)
        _, dx, dy, box = best
        occupied.append(box)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha="center", va="center", fontsize=8.0, fontweight="bold" if method == AM else "normal", color="#a51b1c" if method == AM else "#202833", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.88}, arrowprops={"arrowstyle": "-", "color": "#7b838a", "lw": 0.45, "shrinkA": 2, "shrinkB": 3}, zorder=10)


def tradeoff_figure(frame, kind, assets):
    groups = list(SCENARIOS) if kind == "synthetic" else DATASET_NAMES
    styles = marker_styles()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 10.4))
    axes = axes.ravel()
    coordinates_by_axis = []
    for index, (ax, group) in enumerate(zip(axes, groups)):
        values = frame[(frame.kind == kind) & (frame.group == group)].groupby("method")[METRICS].mean()
        coordinates = []
        for method in MAIN:
            x, y = values.loc[method, ["coverage", "mean_width"]]
            if not np.isfinite([x, y]).all():
                raise ValueError(f"Coverage-width point cannot be silently omitted: {kind}/{group}/{method}")
            style = styles[method]
            ax.scatter(x, y, **style, alpha=0.95, linewidths=0.7, edgecolors="white", zorder=5 if method == AM else 3)
            coordinates.append((method, float(x), float(y)))
        xs = [x for _, x, _ in coordinates] + [0.9]
        ys = [y for _, _, y in coordinates]
        dx, dy = max(max(xs) - min(xs), 0.02), max(max(ys) - min(ys), 0.05)
        ax.set_xlim(max(0.0, min(xs) - 0.12 * dx), min(1.0, max(xs) + 0.15 * dx))
        ax.set_ylim(max(0.0, min(ys) - 0.18 * dy), max(ys) + 0.20 * dy)
        ax.axvline(0.9, color="#384451", ls="--", lw=0.85, zorder=1)
        title = SCENARIO_NAMES[group] if kind == "synthetic" else group
        ax.set_title(f"({chr(97 + index)}) {title}", loc="left", fontweight="bold")
        ax.set_xlabel("Empirical coverage (target 0.90)")
        ax.set_ylabel("Mean interval width" + (f" ({UNITS[group]})" if kind == "real" else ""))
        ax.grid(color="#e6eaee", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        coordinates_by_axis.append((ax, coordinates))
    handles = [Line2D([], [], color=styles[m]["color"], marker=styles[m]["marker"], linestyle="None", markersize=8 if m == AM else 6, label=f"{i + 1}. {LABELS[m]}") for i, m in enumerate(MAIN)]
    if kind == "real":
        axes[-1].set_axis_off()
        axes[-1].legend(handles=handles, loc="center left", ncol=1, frameon=False, fontsize=9.1, handletextpad=0.8, labelspacing=0.67)
        fig.subplots_adjust(left=0.105, right=0.98, top=0.97, bottom=0.07, wspace=0.36, hspace=0.43)
    else:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.015), ncol=4, frameon=False, fontsize=8.2, columnspacing=1.2)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.17, wspace=0.33, hspace=0.47)
    fig.canvas.draw()
    for ax, coordinates in coordinates_by_axis:
        add_number_labels(fig, ax, coordinates)
    stem = "synthetic_coverage_width" if kind == "synthetic" else "real_coverage_width"
    fig.savefig(assets / f"{stem}.pdf")
    fig.savefig(assets / f"{stem}.png", dpi=200)
    plt.close(fig)


def narratives(frame):
    syn = frame[frame.kind == "synthetic"].groupby("method")[METRICS].mean()
    real = frame[frame.kind == "real"].set_index(["group", "method"])
    am = syn.loc[AM]
    methods = ["FACI-residual", "SF-OGD-residual", "PID-residual-projected"]
    compared = ", ".join(f"{syn.loc[m, 'IS90']:.3f} for {LABELS[m]}" for m in methods)
    simulation = (fr"Table~\ref{{tab:sim}} compares all thirteen primary implementations on the same reported predictions. AM-DBR has mean 90\% interval score {am.IS90:.3f}, compared with {compared}. Its mean coverage is {am.coverage:.4f} and its mean width is {am.mean_width:.3f}. Figure~\ref{{fig:tradeoff}} shows the coverage-width position of each method in every scenario; Table~\ref{{tab:sim-detail}} reports the corresponding interval scores. The scalar controllers and full-CDF methods occupy different coverage-width positions, especially under location shift and changing location paths. The complete per-level results and paired seed intervals are reported in the supplementary evidence." + "\n")
    wins = {m: sum(real.loc[(d, AM), "IS90"] < real.loc[(d, m), "IS90"] for d in DATASET_NAMES) for m in methods}
    full_min = {d: min(real.loc[(d, m), "PIT_CRPS"] for m in FULL_CDF) for d in DATASET_NAMES}
    n_best = sum(real.loc[(d, AM), "PIT_CRPS"] == full_min[d] for d in DATASET_NAMES)
    comparison = ", ".join(f"{wins[m]} of five versus {LABELS[m]}" for m in methods)
    real_text = (fr"Table~\ref{{tab:real}} compares PIT scores for all six primary full-CDF methods; AM-DBR has the lowest score in {n_best} of the five main windows. Table~\ref{{tab:real-detail}} compares 90\% interval scores across all thirteen primary methods. AM-DBR has the lower interval score in {comparison}. Figure~\ref{{fig:real-tradeoff}} shows coverage and width in each dataset's native units. These dataset-specific comparisons retain the individual chronological windows; widths and interval scores are not pooled across response units." + "\n")
    return simulation, real_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    results, assets = args.results.resolve(), args.assets.resolve()
    frame, hashes, counts = collect(results)
    assets.mkdir(parents=True, exist_ok=True)
    frame.to_csv(assets / "expanded_all_method_metrics.csv", index=False)
    paired = paired_intervals(frame)
    paired.to_csv(assets / "expanded_paired_seed_intervals.csv", index=False)
    for kind in ["synthetic", "real"]:
        frame[frame.kind == kind].groupby(["group", "method"], sort=False)[METRICS].mean().reset_index().to_csv(assets / f"expanded_{kind}_means.csv", index=False)
    main_tex, supp_tex = tables(frame)
    paired_tex = paired_is90_table(paired)
    (assets / "expanded_main_tables.tex").write_text(main_tex, encoding="utf-8")
    (assets / "expanded_paired_is90.tex").write_text(paired_tex, encoding="utf-8")
    (assets / "expanded_supp_tables.tex").write_text(supp_tex + paired_tex, encoding="utf-8")
    for kind in ["synthetic", "real"]:
        tradeoff_figure(frame, kind, assets)
    sim_text, real_text = narratives(frame)
    (assets / "expanded_simulation_narrative.tex").write_text(sim_text, encoding="utf-8")
    (assets / "expanded_real_narrative.tex").write_text(real_text, encoding="utf-8")
    validation = {"passed": True, "schema_version": 1, **counts,
                  "main_methods": MAIN, "supplement_methods": ORDER, "alphas": list(ALPHAS),
                  "array_summary_agreement": True, "full_payload_receipts_verified": True,
                  "identical_response_time_pit_arrays_within_cell": True,
                  "all_original_windows_and_seeds_retained": True,
                  "interval_score": {"alpha": 0.1, "direction": "lower is better", "evaluation_status": "descriptive interval score computed from saved predictions; no retuning", "formula": "upper-lower + 2/alpha * max(lower-y,0) + 2/alpha * max(y-upper,0)"},
                  "local_gap": {"rolling_window": 500, "minimum_periods": 100},
                  "paired_intervals": "eight paired seed means; scenario average within seed for Overall; descriptive unadjusted 95% Student intervals",
                  "source_fingerprint": fingerprint(),
                  "report_builder_sha256": file_digest(Path(__file__)),
                  "input_sha256": hashes,
                  "figure_units": UNITS}
    outputs = [p for p in assets.iterdir() if p.is_file() and (p.name.startswith("expanded_") or p.stem in {"synthetic_coverage_width", "real_coverage_width"}) and p.name != "expanded_validation.json"]
    validation["output_sha256"] = {p.name: file_digest(p) for p in outputs}
    (assets / "expanded_validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"passed": True, **counts, "assets": str(assets), "outputs": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
