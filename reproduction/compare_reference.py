"""Compare every reported metric against the reference candidate summaries.

Wall-clock runtimes are deliberately excluded. Numeric tolerances accommodate
ordinary floating-point differences, but a changed conclusion is not accepted.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {
    "synthetic": ("main/synthetic_summary.csv", ["scenario", "seed", "method"]),
    "real": ("main/real_summary.csv", ["data_set", "seed", "method"]),
    "state": ("state/summary.csv", ["scenario", "seed", "method"]),
    "windows": ("windows/summary.csv", ["report_start_fraction", "data_set", "seed", "method"]),
}


def compare(actual, expected, keys):
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError("Duplicate experiment identities")
    actual = actual.set_index(keys).sort_index().drop(columns=["runtime_s"], errors="ignore")
    expected = expected.set_index(keys).sort_index().drop(columns=["runtime_s"], errors="ignore")
    if not actual.index.equals(expected.index) or set(actual.columns) != set(expected.columns):
        raise ValueError("Result identities or metric columns differ from the paper reference")
    differences = {}
    for column in expected.columns:
        left, right = actual[column].to_numpy(), expected[column].to_numpy()
        if not np.issubdtype(right.dtype, np.number):
            np.testing.assert_array_equal(left, right)
            continue
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-8, equal_nan=True,
                                   err_msg=f"Reference metric differs: {column}")
        finite = np.isfinite(left) & np.isfinite(right)
        differences[column] = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
    return {"rows": len(actual), "metrics": len(expected.columns), "max_absolute_differences": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.out / ("smoke_reference_comparison.json" if args.smoke else "reference_comparison.json")
    args.out.mkdir(parents=True, exist_ok=True)
    def save_report(report):
        temporary=output.with_suffix('.json.part')
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
    save_report({'status':'running','scope':'smoke' if args.smoke else 'all paper study summaries'})
    reports = {}
    try:
        for name, (path, keys) in STUDIES.items():
            if args.smoke and name != "synthetic":
                continue
            expected = pd.read_csv(ROOT / "reproduction/reference_results" / f"{name}.csv")
            if args.smoke:
                expected = expected[(expected.scenario == "no_shift") & (expected.seed == 6100)]
                path = "smoke/synthetic_summary.csv"
            reports[name] = compare(pd.read_csv(args.out / path), expected, keys)
    except Exception as error:
        save_report({'status':'failed','error':str(error),'completed_studies':reports})
        raise
    report = {"status": "passed", "rtol": 1e-6, "atol": 1e-8,
              "scope": "one synthetic smoke cell" if args.smoke else "all paper study summaries",
              "excluded": ["runtime_s"], "studies": reports}
    save_report(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
