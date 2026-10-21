"""Portable hashes and runtime metadata for experiment manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "requests",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(root: Path, relative_paths: Iterable[str]) -> dict:
    root = Path(root).resolve()
    files = []
    for relative in relative_paths:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Required provenance file is missing: {path}")
        files.append({"path": Path(relative).as_posix(), "sha256": sha256_file(path)})
    packages = {}
    for package in DEFAULT_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "source_files": files,
    }


def artifact_manifest(paths: Mapping[str, Path]) -> list[dict]:
    """Hash generated artifacts without embedding machine-specific paths."""
    artifacts = []
    for label, path in sorted(paths.items()):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Required generated artifact is missing: {path}")
        artifacts.append({
            "label": label,
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return artifacts
