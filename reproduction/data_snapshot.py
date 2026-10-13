"""Fetch/import only the exact five raw inputs used by the paper (stdlib only).

The scientific loaders remain unchanged.  This front door normalizes the
documented weather JSON-to-CSV serialization, including CRLF on every OS.
It never accepts a different data snapshot or silently overwrites a bad cache.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "data_snapshot_manifest.json"


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _record(path, spec):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {spec['filename']}. Run 'python reproduce.py data' or import the exact snapshot.")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != spec["sha256"] or len(raw) != spec["bytes"]:
        raise ValueError(f"Snapshot mismatch: {spec['filename']}; received SHA-256 {actual}, "
                         f"expected {spec['sha256']}. Keep this file separately and import the frozen "
                         "study snapshot; different bytes are never silently accepted or overwritten.")
    return {"filename": spec["filename"], "bytes": len(raw), "sha256": actual}


def verify_snapshot(data_dir):
    """Verify all five files; raises on missing or changed bytes."""
    data_dir = Path(data_dir)
    records = [_record(data_dir / s["filename"], s) for s in load_manifest()["files"]]
    return {"status": "passed", "file_count": len(records), "files": records}


def _weather_csv(raw):
    daily = json.loads(raw)["daily"]
    dates, values = daily["time"], daily["temperature_2m_mean"]
    if not dates or len(dates) != len(values):
        raise ValueError("Malformed weather response: dates and values must have equal nonzero length")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\r\n")
    writer.writerow(["date", "temperature_2m_mean"])
    for date, value in zip(dates, values):
        if value is not None and (isinstance(value, bool) or not math.isfinite(float(value))):
            raise ValueError("Non-finite weather observation")
        writer.writerow([date, "" if value is None else str(float(value))])
    return stream.getvalue().encode("utf-8")


def _download(spec, timeout):
    errors = []
    for source in spec["download_sources"]:
        try:
            request = urllib.request.Request(source["url"], headers={"User-Agent": "OCC-reproduction-v1.0.0-rc.5", "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(32 * 1024 * 1024 + 1)
            if len(raw) > 32 * 1024 * 1024:
                raise ValueError("Unexpectedly large response (limit: 32 MiB)")
            kind = source.get("format", "raw")
            if kind == "weather-json":
                raw = _weather_csv(raw)
            elif kind == "github-content-json":
                document = json.loads(raw)
                if document.get("encoding") != "base64":
                    raise ValueError("GitHub content endpoint did not return base64 bytes")
                raw = base64.b64decode(document["content"], validate=False)
            elif kind != "raw":
                raise ValueError(f"Unsupported manifest format: {kind}")
            actual = hashlib.sha256(raw).hexdigest()
            if actual != spec["sha256"] or len(raw) != spec["bytes"]:
                raise ValueError(f"SHA-256 {actual}; expected {spec['sha256']}. Provider bytes changed.")
            return raw, source["url"]
        except Exception as exc:
            errors.append(f"{source['url']}: {exc}")
    raise RuntimeError(f"Cannot retrieve exact snapshot {spec['filename']}. "
                       "Use offline import of the frozen data files; do not disable verification.\n" + "\n".join(errors))


def _install(raw, destination, spec):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return _record(destination, spec)
    # Unique local staging avoids exposing interrupted or unverified downloads.
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".occ-", delete=False) as handle:
        staged = Path(handle.name)
        handle.write(raw)
    try:
        record = _record(staged, spec)
        staged.replace(destination)
        return record
    finally:
        staged.unlink(missing_ok=True)


def fetch_snapshot(data_dir, timeout=45):
    """Resume missing downloads and verify already present files; fail closed."""
    data_dir = Path(data_dir)
    records, errors = [], []
    for spec in load_manifest()["files"]:
        target = data_dir / spec["filename"]
        try:
            if target.exists():
                record = _record(target, spec)
                record["origin"] = "verified local file"
            else:
                raw, origin = _download(spec, timeout)
                record = _install(raw, target, spec)
                record["origin"] = origin
            records.append(record)
            print(f"VERIFIED {spec['filename']}", flush=True)
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError("\n\n".join(errors))
    return {"status": "passed", "file_count": len(records), "files": records}


def import_snapshot(source, data_dir):
    """Import matching files from a flat directory or zip, then verify all five.

    The archive is never extracted: only named manifest files are read.  A
    four-file public add-on may be imported after the OPSD file was downloaded.
    """
    source, data_dir = Path(source), Path(data_dir)
    manifest = load_manifest()["files"]
    if source.is_dir():
        for spec in manifest:
            path = source / spec["filename"]
            if path.is_file():
                _record(path, spec)
                _install(path.read_bytes(), data_dir / spec["filename"], spec)
    elif source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for spec in manifest:
                matches = [i for i in archive.infolist() if i.filename in
                           (spec["filename"], "data_snapshot/" + spec["filename"])]
                if len(matches) > 1:
                    raise ValueError(f"Ambiguous archive entries for {spec['filename']}")
                if matches:
                    if matches[0].file_size != spec["bytes"]:
                        raise ValueError(f"Unexpected archive member length: {spec['filename']}")
                    _install(archive.read(matches[0]), data_dir / spec["filename"], spec)
    else:
        raise ValueError("--source must be a directory or ZIP snapshot")
    return verify_snapshot(data_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "fetch", "import"))
    parser.add_argument("--data-dir", type=Path, default=Path("data_snapshot"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.action == "import" and args.source is None:
        parser.error("import requires --source")
    try:
        result = (verify_snapshot(args.data_dir) if args.action == "verify" else
                  fetch_snapshot(args.data_dir, args.timeout) if args.action == "fetch" else
                  import_snapshot(args.source, args.data_dir))
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
