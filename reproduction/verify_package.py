"""Verify complete payload bytes and the frozen numerical source manifest."""
from pathlib import Path
import hashlib
import json
import sys

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'PUBLIC_PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
for entry in manifest['files']:
    path=root/entry['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
        raise SystemExit(f"Package file missing or modified: {entry['path']}")
sys.path[:0]=[str(root/p) for p in ('framework','scripts','reproduction','revision_tools')]
from run_revision import fingerprint, fingerprint_entries
source=json.loads((root/'PUBLIC_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
if fingerprint()!=source['source_sha256'] or fingerprint_entries()!=source['source_file_hashes']:
    raise SystemExit('Locked numerical source fingerprint differs')
print(f"Verified {len(manifest['files'])} package files; numerical source {fingerprint()}.")
