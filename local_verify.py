"""Run the AM-DBR numerical study from an IDE or notebook.

This module is part of the verified package: do not edit it. Run
make_local_workspace.py to create editable IDE/notebook copies outside the
package. The active interpreter is used by default. See LOCAL_VERIFICATION_zh.txt.
"""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys

# Library defaults; personal settings belong in the external work copy.
STEP = "check"
REPO = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)    # Python 3.12 with requirements-lock.txt installed
RUNS = REPO / "my_verification"
SNAPSHOT = None                # optional Path to an authorized exact-data ZIP/folder
WORKERS = 1


def stream(command, cwd):
    """Stream one child process; stop its worker tree on user interruption."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    options = {"start_new_session": True} if os.name != "nt" else {}
    print("Running:", subprocess.list2cmdline([str(x) for x in command]), flush=True)
    child = subprocess.Popen([str(x) for x in command], cwd=cwd, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", **options)
    try:
        for line in child.stdout:
            print(line, end="", flush=True)
        status = child.wait()
    except KeyboardInterrupt:
        if child.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        print("Stopped. After workers stop, use the same paths and run resume.")
        raise
    finally:
        child.stdout.close()
    if status:
        raise RuntimeError(f"Exit code {status}; inspect the output above and commands.log.")


class LocalVerification:
    """One set of paths shared by preparation, full runs and result inspection."""
    def __init__(self, repo=None, python=None, runs=None, snapshot=None, workers=1):
        self.repo = Path(repo if repo is not None else REPO).expanduser().resolve()
        self.python = Path(python if python is not None else sys.executable).expanduser().resolve()
        self.runs = Path(runs if runs is not None else self.repo / "my_verification").expanduser().resolve()
        self.snapshot = None if snapshot is None else Path(snapshot).expanduser().resolve()
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.workers = workers

    def command(self, step):
        allowed = {"install", "check", "smoke", "data", "all", "resume", "report"}
        if step not in allowed:
            raise ValueError(f"Choose one of {sorted(allowed | {'results'})}")
        if not self.python.is_file() or not (self.repo / "reproduce.py").is_file():
            raise FileNotFoundError("Choose a Python 3.12 executable and the extracted code directory.")
        if step == "install":
            return [self.python, "-m", "pip", "install", "-r", self.repo / "requirements-lock.txt"]
        folder = {"check": "checks", "smoke": "smoke_check"}.get(step, "full_run")
        command = [self.python, "-u", self.repo / "reproduce.py", step,
                   "--out", self.runs / folder, "--data-dir", self.runs / "data_snapshot",
                   "--workers", str(self.workers)]
        if step == "data" and self.snapshot is not None:
            if not self.snapshot.exists():
                raise FileNotFoundError(f"Snapshot not found: {self.snapshot}")
            command += ["--import-data", self.snapshot]
        elif step in {"all", "resume"}:
            command += ["--verify-only"]  # Explicit data preparation precedes full runs.
        return command

    def show_results(self):
        out = self.runs / "full_run"
        status_path, comparison_path = out / "RUN_STATUS.json", out / "reference_comparison.json"
        assets = out / "paper_assets" / "generated"
        expanded_path = assets / "expanded_validation.json"
        if not all(path.is_file() for path in (status_path, comparison_path, expanded_path)):
            print("Full verification is not complete. check/smoke/data alone are insufficient.")
            print("The comparison tables and figures also require expanded_validation.json.")
            print("Expected outputs:", out)
            return False
        status = json.loads(status_path.read_text(encoding="utf-8"))
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        expanded = json.loads(expanded_path.read_text(encoding="utf-8"))
        expected_rows = {"synthetic": 720, "real": 75, "state": 96, "windows": 225}
        studies = comparison.get("studies", {})
        release = json.loads((self.repo / "RELEASE.json").read_text(encoding="utf-8"))["release"]
        complete = (status.get("status") == "passed"
                    and status.get("command") in {"all", "resume", "report"}
                    and status.get("release") == release
                    and comparison.get("status") == "passed"
                    and comparison.get("scope") == "all paper study summaries"
                    and expanded.get("passed") is True
                    and expanded.get("cells") == 53
                    and expanded.get("method_runs") == 795
                    and expanded.get("interval_predictions_across_levels") == 5217000
                    and set(studies) == set(expected_rows)
                    and all(studies[name].get("rows") == count
                            and studies[name].get("metrics") == 16
                            for name, count in expected_rows.items()))
        print("RUN_STATUS:", json.dumps(status, ensure_ascii=False))
        print("Reference comparison:", comparison.get("status"))
        print("Expanded comparison:", expanded.get("passed"),
              "cells:", expanded.get("cells"), "method runs:", expanded.get("method_runs"))
        for name, study in studies.items():
            print(name, "rows:", study.get("rows"), "metrics:", study.get("metrics"))
        print("FULL VERIFICATION PASSED" if complete else "Full verification is not complete.")
        print("Tables and figures:", assets)
        print("Simulation coverage-width figure:", assets / "synthetic_coverage_width.pdf")
        print("Real-data coverage-width figure:", assets / "real_coverage_width.pdf")
        return complete

    def run(self, step):
        if step == "results":
            return self.show_results()
        stream(self.command(step), self.repo)
        if step in {"all", "resume", "report"} and not self.show_results():
            raise RuntimeError("The process exited successfully but the full evidence is incomplete.")


def run_step(step):
    """Compatibility entry point using defaults; prefer LocalVerification in a work copy."""
    return LocalVerification(REPO, PYTHON, RUNS, SNAPSHOT, WORKERS).run(step)


if __name__ == "__main__":
    print("Run make_local_workspace.py first; edit the generated external run_from_IDE.py or notebook.")
