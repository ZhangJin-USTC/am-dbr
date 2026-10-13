# Acquire the exact study data

Run `python reproduce.py data`. The default cache is `data_snapshot/`. Five raw
inputs (about 14.8 MB in total) are downloaded and checked against the frozen
SHA-256 manifest. A fresh download of all five sources passed on 2026-09-20
(Asia/Shanghai). No account, API token or proprietary data service was needed.

The download operation uses only Python's standard library. It leaves valid
cached files in place and can be rerun after an interrupted network connection.
The exact weather CSV line endings are generated consistently on Windows,
macOS and Linux. OPSD is downloaded by an immutable Git commit, with a GitHub
content-API route and a raw-file fallback.

For a previously acquired cache:

```text
python reproduction/data_snapshot.py verify --data-dir PATH_TO_DATA
python reproduce.py all --data-dir PATH_TO_DATA
```

For the provided local complete backup or a directory containing the same files:

```text
python reproduction/data_snapshot.py import --source PATH_TO_ZIP_OR_DIRECTORY --data-dir data_snapshot
```

All five files must eventually be present. A ZIP is not blindly extracted: the
importer reads only the five named manifest entries and verifies their sizes and
hashes. The full list of filenames, URLs and hashes is in
`docs/data_snapshot_manifest.json`; see `docs/DATA_LICENSES_AND_ATTRIBUTION.md`
for source-specific attribution and the distinction between the public add-on
and local backup.

If a hash differs, do not change the expected hash or disable the check. Preserve
the differing file separately and obtain the exact archived study snapshot. In
particular, Open-Meteo may revise historical reanalysis responses. Download
success today does not promise permanent byte stability. The optional four-file
CC-BY archival add-on preserves the exact weather/Jena/Metro bytes; it excludes
the OPSD mirror, which remains retrievable by its pinned Git commit.

No raw inputs are included in the public code ZIP. Fetch/import is an explicit
precondition for real-data experiments, so missing inputs cannot silently be
replaced by another dataset or a shortened demonstration.
