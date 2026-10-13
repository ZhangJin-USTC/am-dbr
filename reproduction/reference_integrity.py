"""Check pinned external recurrence files before importing any of them."""
from __future__ import annotations
from pathlib import Path
from revision_integrity import file_digest, read_json

PINNED_REPOSITORY = 'https://github.com/salesforce/online_conformal'
PINNED_COMMIT = '95790a9b9e4061166e6d7a8a13282e019b1a8663'
PINNED_PYTHON_FILES = {f'online_conformal/{name}.py'
                       for name in ('faci','nex_conformal','ogd','saocp','utils')}


def verify_reference(reference_dir):
    root=Path(reference_dir).resolve()
    manifest=read_json(root/'VALIDATION_MANIFEST.json')
    if manifest.get('repository')!=PINNED_REPOSITORY or manifest.get('commit')!=PINNED_COMMIT:
        raise ValueError('Upstream reference repository/commit mismatch')
    required=PINNED_PYTHON_FILES|{'SOURCE.json','LICENSE'}
    files=manifest.get('files',{})
    if set(files)!=required:
        raise ValueError('Upstream validation manifest file schema mismatch')
    for relative,expected in files.items():
        path=root/relative
        if not path.is_file() or file_digest(path)!=expected:
            raise ValueError(f'Upstream reference hash mismatch: {relative}')
    source=read_json(root/'SOURCE.json')
    if (source.get('repository')!=PINNED_REPOSITORY or source.get('commit')!=PINNED_COMMIT
            or set(source.get('files',{}))!=PINNED_PYTHON_FILES):
        raise ValueError('Pinned upstream SOURCE manifest mismatch')
    if any(source['files'][name]!=files[name] for name in PINNED_PYTHON_FILES):
        raise ValueError('Pinned upstream SOURCE file hashes disagree')
    return {'repository':PINNED_REPOSITORY,'commit':PINNED_COMMIT,
            'manifest_sha256':file_digest(root/'VALIDATION_MANIFEST.json'),
            'verified_files':files,'verification':'passed before reference import'}
