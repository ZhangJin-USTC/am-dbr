"""Portable acquisition and tamper rejection without network access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

CODE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("data_snapshot", CODE / "reproduction/data_snapshot.py")
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


def _spec(name="input.csv", raw=b"exact\r\n"):
    return {"filename": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "download_sources": [{"url": "https://invalid.test/input", "format": "raw"}]}


def test_weather_serialization_is_explicit_and_platform_independent():
    raw = json.dumps({"daily": {"time": ["2000-01-01", "2000-01-02", "2000-01-03"],
                                "temperature_2m_mean": [6.5, 2, None]}}).encode()
    expected = b"date,temperature_2m_mean\r\n2000-01-01,6.5\r\n2000-01-02,2.0\r\n2000-01-03,\r\n"
    assert snapshot._weather_csv(raw) == expected


def test_changed_remote_bytes_cannot_be_installed(monkeypatch, tmp_path):
    manifest = {"files": [_spec()]}
    monkeypatch.setattr(snapshot, "load_manifest", lambda: manifest)
    monkeypatch.setattr(snapshot, "_download", lambda spec, timeout: (b"changed\n", "test"))
    with pytest.raises(RuntimeError, match="Snapshot mismatch"):
        snapshot.fetch_snapshot(tmp_path)
    assert not (tmp_path / "input.csv").exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_bad_cache_is_not_overwritten(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, "load_manifest", lambda: {"files": [_spec()]})
    target = tmp_path / "input.csv"
    target.write_bytes(b"old")
    with pytest.raises(RuntimeError, match="Snapshot mismatch"):
        snapshot.fetch_snapshot(tmp_path)
    assert target.read_bytes() == b"old"


def test_import_reads_only_named_zip_entries(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, "load_manifest", lambda: {"files": [_spec()]})
    archive = tmp_path / "snapshot.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("data_snapshot/input.csv", b"exact\r\n")
        z.writestr("../outside.txt", b"untrusted")
    assert snapshot.import_snapshot(archive, tmp_path / "data")["status"] == "passed"
    assert not (tmp_path / "outside.txt").exists()


def test_snapshot_manifest_matches_scientific_loader_constants():
    import ast
    module = ast.parse((CODE / "real_data/loaders.py").read_text())
    actual = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "PINNED_SHA256" for t in n.targets))
    assert {s["filename"]: s["sha256"] for s in snapshot.load_manifest()["files"]} == actual
