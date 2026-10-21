"""IDE launches must preserve paths and never mistake smoke evidence for full evidence."""
from pathlib import Path
import importlib.util
import json
import hashlib
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("local_verify_public", ROOT / "local_verify.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
workspace_spec = importlib.util.spec_from_file_location("make_local_workspace_public", ROOT / "make_local_workspace.py")
workspace_helper = importlib.util.module_from_spec(workspace_spec)
workspace_spec.loader.exec_module(workspace_helper)


def prepared(tmp_path):
    repo = tmp_path / "code with spaces"
    repo.mkdir()
    (repo / "reproduce.py").write_text("pass")
    (repo / "RELEASE.json").write_text('{"release":"paper-submission"}')
    return helper.LocalVerification(repo, sys.executable, tmp_path / "runs with spaces")


def write_results(launch, command="all", release="paper-submission"):
    out = launch.runs / "full_run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "RUN_STATUS.json").write_text(json.dumps({"status":"passed", "command":command, "release":release}))
    comparison = {"status":"passed", "scope":"all paper study summaries", "studies":
        {name:{"rows":rows,"metrics":16} for name,rows in
         {"synthetic":720,"real":75,"state":96,"windows":225}.items()}}
    (out / "reference_comparison.json").write_text(json.dumps(comparison))
    assets = out / "paper_assets" / "generated"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "expanded_validation.json").write_text(json.dumps({
        "passed": True, "cells": 53, "method_runs": 795,
        "interval_predictions_across_levels": 5217000}))
    return comparison


def test_ide_full_and_resume_keep_the_same_paths_and_require_prepared_data(tmp_path):
    launch = prepared(tmp_path)
    for step in ("all", "resume", "report", "data"):
        command = launch.command(step)
        assert command[0] == Path(sys.executable).resolve()
        assert command[command.index("--out")+1] == launch.runs / "full_run"
        assert command[command.index("--data-dir")+1] == launch.runs / "data_snapshot"
        assert ("--verify-only" in command) == (step in {"all", "resume"})
    for step, folder in (("check","checks"),("smoke","smoke_check")):
        command = launch.command(step)
        assert command[command.index("--out")+1] == launch.runs / folder


def test_snapshot_import_requires_an_existing_path_and_keeps_path_as_one_argument(tmp_path):
    launch = prepared(tmp_path)
    launch.snapshot = tmp_path / "authorized snapshot.zip"
    with pytest.raises(FileNotFoundError):
        launch.command("data")
    launch.snapshot.write_bytes(b"fixture")
    command = launch.command("data")
    assert command[-2:] == ["--import-data", launch.snapshot]


@pytest.mark.parametrize("command,release", [("smoke","paper-submission"),("check","paper-submission"),("all","old-package")])
def test_results_reject_partial_or_stale_release_receipts(tmp_path, command, release):
    launch = prepared(tmp_path)
    write_results(launch, command, release)
    assert not launch.show_results()


def test_results_require_every_study_and_metric(tmp_path):
    launch = prepared(tmp_path)
    comparison = write_results(launch)
    assert launch.show_results()
    comparison["studies"]["windows"]["rows"] = 224
    (launch.runs / "full_run/reference_comparison.json").write_text(json.dumps(comparison))
    assert not launch.show_results()


@pytest.mark.parametrize("field,value", [("passed", False), ("cells", 52),
                         ("method_runs", 794), ("interval_predictions_across_levels", 0)])
def test_results_reject_incomplete_expanded_report(tmp_path, field, value):
    launch = prepared(tmp_path)
    write_results(launch)
    path = launch.runs / "full_run/paper_assets/generated/expanded_validation.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    assert not launch.show_results()


def test_results_reject_missing_expanded_report_even_if_old_references_pass(tmp_path):
    launch = prepared(tmp_path)
    write_results(launch)
    (launch.runs / "full_run/paper_assets/generated/expanded_validation.json").unlink()
    assert not launch.show_results()


def test_full_process_success_without_complete_evidence_is_rejected(tmp_path, monkeypatch):
    launch = prepared(tmp_path)
    monkeypatch.setattr(helper, "stream", lambda *_: None)
    with pytest.raises(RuntimeError, match="evidence is incomplete"):
        launch.run("all")


def test_child_failure_is_not_masked_and_invalid_step_rejected(tmp_path, monkeypatch):
    launch = prepared(tmp_path)
    def fail(*_):
        raise RuntimeError("child failed")
    monkeypatch.setattr(helper, "stream", fail)
    with pytest.raises(RuntimeError, match="child failed"):
        launch.run("smoke")
    with pytest.raises(ValueError):
        launch.command("unknown")


def test_work_copy_can_be_edited_and_executed_without_changing_verified_templates(tmp_path):
    protected = [ROOT / "local_verify.py", ROOT / "AM_DBR_local_verification.ipynb"]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in protected]
    workspace = workspace_helper.create_workspace(tmp_path / "editable space", repo=ROOT)
    script = workspace / "run_from_IDE.py"
    script.write_text(script.read_text(encoding="utf-8").replace('STEP = "check"','STEP = "results"'), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert "Full verification is not complete" in result.stdout
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in protected] == before
    notebook = json.loads((workspace / "AM_DBR_local_verification.ipynb").read_text(encoding="utf-8"))
    config = next(c for c in notebook["cells"] if c["cell_type"] == "code")
    scope = {}
    exec(compile("".join(config["source"]), "work-copy", "exec"), scope)
    assert scope['verify'].repo == ROOT
    assert scope['verify'].runs == workspace / 'runs'


def test_workspace_rejects_package_internal_destination():
    with pytest.raises(ValueError, match="outside"):
        workspace_helper.create_workspace(ROOT / "outputs" / "editable", repo=ROOT)


def test_workspace_never_overwrites_a_users_existing_configuration(tmp_path):
    workspace = workspace_helper.create_workspace(tmp_path / "existing", repo=ROOT)
    script = workspace / "run_from_IDE.py"
    script.write_text("# personal choices", encoding="utf-8")
    with pytest.raises(FileExistsError, match="existing copies"):
        workspace_helper.create_workspace(workspace, repo=ROOT)
    assert script.read_text(encoding="utf-8") == "# personal choices"
