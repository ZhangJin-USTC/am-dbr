"""Portable entry point for the frozen v1.4.0 paper package.

Uses only the standard library until the environment check passes. Run --help.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def package_release() -> str:
    """Read the public package release instead of duplicating its label."""
    release_path = ROOT / "RELEASE.json"
    if not release_path.is_file():
        return "v1.4.0"
    return json.loads(release_path.read_text(encoding="utf-8"))["release"]
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")
for directory in ("framework", "scripts", "reproduction", "revision_tools"):
    sys.path.insert(0, str(ROOT / directory))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")


def run(arguments, out):
    """Stream output to the terminal and an append-only command log."""
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, *map(str, arguments)]
    print("\nRUN:", subprocess.list2cmdline(command), flush=True)
    with (out / "commands.log").open("a", encoding="utf-8") as log:
        log.write("\n" + time.strftime("%Y-%m-%dT%H:%M:%S%z") + " " + subprocess.list2cmdline(command) + "\n")
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace") as child:
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = child.wait()
        if status:
            raise RuntimeError(f"Command failed (exit {status}); see {out / 'commands.log'}")


def check_environment(out):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12, the tested paper environment. See README.md.")
    versions, mismatch = {}, []
    for line in (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if not requirement:
            continue
        requirement, separator, marker = requirement.partition(";")
        if separator:
            if marker.strip() != 'sys_platform == "win32"':
                raise RuntimeError(f"Unsupported dependency marker: {marker}")
            if sys.platform != "win32":
                continue
        requirement=requirement.strip()
        name, expected = requirement.split("==")
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = "MISSING"
        versions[name] = installed
        if installed != expected:
            mismatch.append(f"{name}: expected {expected}, installed {installed}")
    if mismatch:
        raise RuntimeError("Install requirements-lock.txt in a clean environment:\n" + "\n".join(mismatch))
    from run_revision import fingerprint
    expected = "e74b3647b9c51eb1f8b4fb0e53fd7ec9fd756a198d7f9b49c969222414a9dfed"
    actual = fingerprint()
    if actual != expected:
        raise RuntimeError(f"Core source changed: {actual}; frozen result reuse is not allowed.")
    out.mkdir(parents=True, exist_ok=True)
    record = {"python": sys.version, "platform": platform.platform(),
              "machine": platform.machine(), "dependencies": versions,
              "core_source_sha256": actual, "release": package_release()}
    (out / "environment.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if (ROOT / "PUBLIC_PACKAGE_MANIFEST.json").exists():
        run(["reproduction/verify_package.py"], out)
    print("Environment and frozen core: OK", flush=True)


def check(out):
    check_environment(out)
    run(["-m", "pytest", "tests", "-q"], out)
    run(["scripts/validate_upstream.py", out / "upstream"], out)


def data(args):
    from data_snapshot import fetch_snapshot, import_snapshot, verify_snapshot
    if args.import_data is not None:
        result = import_snapshot(args.import_data.resolve(), args.data_dir)
    elif args.verify_only:
        result = verify_snapshot(args.data_dir)
    else:
        result = fetch_snapshot(args.data_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def report(args):
    check_environment(args.out)
    run(["revision_tools/build_revision_artifacts.py", "--results", args.out / "main",
         "--state", args.out / "state", "--manuscript", args.out / "paper_assets"], args.out)
    run(["revision_tools/validate_report_windows.py", "--windows", args.out / "windows"], args.out)
    run(["revision_tools/build_paper_assets.py", "--results", args.out / "main",
         "--windows", args.out / "windows", "--assets", args.out / "paper_assets"], args.out)
    # Expanded comparison is a presentation-stage audit of the same locked
    # predictions.  It adds the proper 90% interval score and coverage-width
    # panels for all primary methods; it does not rerun or tune a method.
    expanded = ["revision_tools/build_expanded_comparisons.py", "--results", args.out / "main",
                "--assets", args.out / "paper_assets" / "generated"]
    run(expanded, args.out)
    run(["reproduction/compare_reference.py", "--out", args.out], args.out)
    print(f"\nCOMPLETE: paper tables, figures and audit reports are in {args.out / 'paper_assets'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "data", "smoke", "all", "resume", "report"))
    parser.add_argument("--out", type=Path, default=ROOT / "outputs", help="Fresh reproduction directory (default: ./outputs)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data_snapshot")
    parser.add_argument("--workers", type=int, default=1, help="Parallel experiment cells; 1 minimizes memory use")
    parser.add_argument("--import-data", type=Path, help="Import a local directory or archive containing the exact snapshots")
    parser.add_argument("--verify-only", action="store_true", help="For data: verify local snapshots without downloading")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.out = args.out.resolve()
    args.data_dir = args.data_dir.resolve()
    def status(state, error=None):
        args.out.mkdir(parents=True, exist_ok=True)
        record={"status":state,"command":args.command,"release":package_release(),
                "recorded_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if error is not None:record['error']=str(error)
        temporary=args.out/'RUN_STATUS.json.part'
        temporary.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        temporary.replace(args.out/'RUN_STATUS.json')
    status('running')
    try:
        if args.command == "data":
            data(args)
        elif args.command == "check":
            check(args.out)
        elif args.command == "report":
            report(args)
        elif args.command == "smoke":
            check_environment(args.out)
            run(["scripts/run_revision.py", "--out", args.out / "smoke", "--data-dir", args.data_dir,
                 "--workers", args.workers, "--resume", "--datasets", "--scenarios", "no_shift", "--seeds", "6100"], args.out)
            run(["reproduction/compare_reference.py", "--out", args.out, "--smoke"], args.out)
            print("Smoke passed: one full synthetic cell, all 15 methods and five levels; this is not the full paper run.")
        else:
            # Both names are deliberately idempotent; receipts reject changed/mixed evidence.
            check(args.out)
            data(args)
            run(["reproduction/resume_checkpoints.py", "--out", args.out / "main", "--study", "main"], args.out)
            run(["scripts/run_revision.py", "--out", args.out / "main", "--data-dir", args.data_dir,
                 "--workers", args.workers, "--resume"], args.out)
            run(["revision_tools/resume_state.py", "--out", args.out / "state", "--workers", args.workers], args.out)
            run(["reproduction/resume_checkpoints.py", "--out", args.out / "windows", "--study", "windows"], args.out)
            run(["revision_tools/run_report_window_sensitivity.py", "--out", args.out / "windows",
                 "--data-dir", args.data_dir, "--workers", args.workers, "--resume"], args.out)
            report(args)
    except (Exception, KeyboardInterrupt) as error:
        status('failed',error)
        print(f"\nSTOPPED: {error}", file=sys.stderr, flush=True)
        return 1
    status('passed')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
