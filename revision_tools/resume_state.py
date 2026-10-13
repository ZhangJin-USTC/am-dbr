"""Resume count-state sensitivity without modifying the frozen study runner.

Completed cells are validated against source/configuration/hash receipts. An interrupted, uncommitted
cell is preserved in a quarantine directory and recomputed in full.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("framework", "scripts", "reproduction", "revision_tools")]
from run_state_sensitivity import (SCENARIOS, STATE_CONFIG, cell, config_fingerprint,
                                  expected_state_cell, fingerprint, state_protocol)
from revision_integrity import completed_payload_names, read_json, require_equal, verify_complete, write_json


def prepare(out):
    expected = state_protocol()
    protocol = out / "protocol.json"
    if out.exists() and any(out.iterdir()):
        if not protocol.exists():
            raise ValueError("State directory has no protocol; use a fresh directory.")
        require_equal(read_json(protocol), expected, "State protocol")
    out.mkdir(parents=True, exist_ok=True)
    write_json(protocol, expected)
    methods = list(STATE_CONFIG["methods"])
    tasks = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for scenario in SCENARIOS:
        for seed in range(6100, 6108):
            dest = out / f"{scenario}_{seed}"
            if (dest / "COMPLETE.json").exists():
                verify_complete(dest, expected_state_cell(scenario, seed), completed_payload_names(methods))
                continue
            if dest.exists() and any(dest.iterdir()):
                # Preserve forensic evidence; never trust or silently delete partial arrays.
                saved = out / "_interrupted" / stamp / dest.name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest), str(saved))
                print(f"Preserved uncommitted cell at {saved}", flush=True)
            tasks.append((scenario, seed, str(out), fingerprint(), config_fingerprint()))
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    tasks = prepare(args.out)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(cell, tasks):
            print("COMPLETE", result, flush=True)
    import pandas as pd
    paths = [args.out / f"{s}_{i}" / "summary.csv" for s in SCENARIOS for i in range(6100, 6108)]
    pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).to_csv(args.out / "summary.csv", index=False)
    print(f"State sensitivity complete: 48 cells; {len(tasks)} computed this invocation.")


if __name__ == "__main__":
    main()
