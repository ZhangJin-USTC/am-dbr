"""Portable provenance for locked study cells and restartable method outputs.

Receipts detect accidental mixing, truncation and edits; they are not digital
signatures. A receipt is accepted only against a freshly reconstructed expected
source/configuration/identity, never merely against values in its own file.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell_protocol(source_sha, identity, config):
    config = canonical(config)
    return {"schema_version": 1, "source_sha256": source_sha,
            "config_sha256": digest_json(config), "identity": canonical(identity),
            "config": config}


def require_equal(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise ValueError(f"{label}: source/configuration/identity mismatch")


def payload_hashes(folder, names):
    folder = Path(folder)
    result = {}
    for name in sorted(names):
        # Cell receipts never refer outside the cell or recursively follow a
        # path obtained from an untrusted result file.
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid payload name: {name}")
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {"sha256": file_digest(path), "bytes": path.stat().st_size}
    return result


def verify_receipt(folder, receipt_name, expected, required_files):
    receipt = read_json(Path(folder) / receipt_name)
    require_equal(receipt.get("provenance"), expected, receipt_name)
    recorded = receipt.get("payloads", {})
    if set(recorded) != set(required_files):
        raise ValueError(f"{receipt_name}: payload schema mismatch")
    observed = payload_hashes(folder, required_files)
    if recorded != observed:
        changed = [name for name in observed if observed[name] != recorded[name]]
        raise ValueError(f"{receipt_name}: payload hash mismatch: {changed}")
    return receipt


def write_receipt(folder, receipt_name, expected, required_files, **details):
    write_json(Path(folder) / receipt_name,
               {"provenance": expected, "payloads": payload_hashes(folder, required_files),
                **details})


def prepare_cell(folder, expected):
    folder = Path(folder)
    marker = folder / "CELL_PROTOCOL.json"
    if folder.exists() and any(folder.iterdir()):
        if not marker.is_file():
            raise ValueError(f"Unprovenanced or stale cell: {folder}")
        require_equal(read_json(marker), expected, str(marker))
    folder.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        write_json(marker, expected)


def verify_complete(folder, expected, required_files):
    require_equal(read_json(Path(folder) / "CELL_PROTOCOL.json"), expected, "CELL_PROTOCOL")
    return verify_receipt(folder, "COMPLETE.json", expected, required_files)


def method_receipt_name(method):
    return method + "_CHECKPOINT.json"


def method_payload_names(method):
    files = [method + ".csv", method + ".npz"]
    if method == "AMDBR-core-noidentity":
        files.append(method + "_diagnostics.json")
    return files


def completed_payload_names(methods, metadata=True):
    files = ["CELL_PROTOCOL.json", "summary.csv"]
    if metadata:
        files.append("metadata.json")
    for method in methods:
        files += method_payload_names(method) + [method_receipt_name(method)]
    return files


def resume_method(folder, method, expected):
    folder = Path(folder)
    receipt_name = method_receipt_name(method)
    payload_names = method_payload_names(method)
    if (folder / receipt_name).exists():
        verify_receipt(folder, receipt_name, expected, payload_names)
        return True
    if any((folder / name).exists() for name in payload_names):
        raise ValueError(f"Unreceipted or interrupted method payload: {folder}/{method}")
    return False
