"""Locked real-data evaluation for the frozen AM-DBR configuration."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "framework", ROOT / "reproduction", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from experiment import (  # noqa: E402
    load_dataset,
    multilevel_coverage_table,
    results_to_records,
    run_comparison,
    summary_table,
)
from fixed_adaptive_refresh import (  # noqa: E402
    MethodConfig,
    methods_factory as manuscript_methods_factory,
)
from methods import AdaptiveMemoryDBR  # noqa: E402
from paper_config import PAPER_CONFIG  # noqa: E402


DEFAULT_CACHE = ROOT.parent.parent.parent / "2026-08-06" / "03_round3_experiment_validation" / "real_data_cache"
DEFAULT_OUT = ROOT / "outputs" / "adaptive_memory_realdata_study_20260809"
ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.50)
DATASETS = (
    {"label": "Seattle", "name": "open_meteo_seattle_daily", "max_eval": 2500, "train_frac": 0.20, "warm_frac": 0.45, "eval_start_frac": 0.70},
    {"label": "Phoenix", "name": "open_meteo_phoenix_daily", "max_eval": 2500, "train_frac": 0.20, "warm_frac": 0.45, "eval_start_frac": 0.70},
    {"label": "Jena", "name": "jena_climate", "max_eval": 2500, "train_frac": None, "warm_frac": None, "eval_start_frac": 0.65},
    {"label": "Metro", "name": "metro_traffic", "max_eval": 2500, "train_frac": 0.20, "warm_frac": 0.45, "eval_start_frac": 0.65},
    {"label": "OPSD", "name": "opsd_germany_daily", "max_eval": 2000, "train_frac": 0.20, "warm_frac": 0.45, "eval_start_frac": 0.55},
)

FROZEN_CONFIG = {
    "lambdas": list(PAPER_CONFIG.am_lambdas),
    "bins": PAPER_CONFIG.am_bins,
    "eta": PAPER_CONFIG.am_eta,
    "share": PAPER_CONFIG.am_share,
    "include_identity": PAPER_CONFIG.am_include_identity,
}


def methods_factory():
    methods = manuscript_methods_factory(MethodConfig())()
    methods["AMDBR-core-noidentity"] = AdaptiveMemoryDBR(
        lambdas=tuple(FROZEN_CONFIG["lambdas"]),
        bins=FROZEN_CONFIG["bins"],
        eta=FROZEN_CONFIG["eta"],
        share=FROZEN_CONFIG["share"],
        include_identity=FROZEN_CONFIG["include_identity"],
    )
    return methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    requested = set(args.datasets or [item["label"] for item in DATASETS])
    specs = [item for item in DATASETS if item["label"] in requested]
    if {item["label"] for item in specs} != requested:
        raise ValueError(f"Unknown data-set labels: {sorted(requested - {item['label'] for item in specs})}")

    all_summaries = []
    all_records = []
    metadata = []
    for spec in specs:
        dataset = load_dataset(
            spec["name"],
            data_dir=args.data_dir,
            max_eval=spec["max_eval"],
            model_type="ridge",
            train_frac=spec["train_frac"],
            warm_frac=spec["warm_frac"],
            eval_start_frac=spec["eval_start_frac"],
        )
        results = run_comparison(
            dataset,
            alphas=ALPHAS,
            methods_factory=methods_factory,
        )
        y = np.asarray([step.y for step in dataset["online"]], dtype=float)
        summary = summary_table(
            results,
            y,
            alpha_main=0.10,
            rolling_window=500,
            round_digits=None,
        ).reset_index()
        multi = multilevel_coverage_table(
            results, y, alphas=ALPHAS, round_digits=None
        ).reset_index()[["method", "avg_gap", "max_gap"]]
        summary = summary.merge(multi, on="method", how="left")
        summary.insert(0, "data_set", spec["label"])
        all_summaries.append(summary)
        records = results_to_records(dataset, results, (0.10,))
        records.insert(0, "data_set", spec["label"])
        all_records.append(records)
        metadata.append(dataset["meta"])
        print(f"Completed {spec['label']}: {len(y)} online observations", flush=True)

    summary = pd.concat(all_summaries, ignore_index=True)
    summary.to_csv(args.out_dir / "summary_all_methods.csv", index=False)
    summary.groupby("method", observed=True).mean(numeric_only=True).reset_index().to_csv(
        args.out_dir / "summary_overall.csv", index=False
    )
    with gzip.open(args.out_dir / "prequential_records_alpha010.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        pd.concat(all_records, ignore_index=True).to_csv(handle, index=False)
    (args.out_dir / "protocol.json").write_text(
        json.dumps(
            {
                "frozen_configuration": FROZEN_CONFIG,
                "candidate_source": "synthetic validation only; locked real data not used for selection",
                "datasets": list(DATASETS),
                "requested_datasets": [item["label"] for item in specs],
                "alphas": list(ALPHAS),
                "data_cache": str(args.data_dir.name),
                "metadata": metadata,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
