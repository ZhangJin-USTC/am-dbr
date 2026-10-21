# Acquire the study data

Run `python reproduce.py data`. The default cache is `data_snapshot/`. The downloader retrieves five public raw inputs (about 14.8 MB in total) and checks them against the SHA-256 values in `docs/data_snapshot_manifest.json`. Raw data are not included in this code package.

The downloader uses Python's standard library. Valid cached files are retained, so an interrupted download can be restarted. Weather CSV serialization uses the same line endings on Windows, macOS and Linux. OPSD is retrieved from the recorded Git commit through the GitHub content API, with a raw-file fallback.

## Use an existing exact-data cache

```text
python reproduction/data_snapshot.py verify --data-dir PATH_TO_DATA
python reproduce.py all --data-dir PATH_TO_DATA
```

If you have lawfully obtained a directory or ZIP containing the study's exact input files, import it with:

```text
python reproduction/data_snapshot.py import --source PATH_TO_ZIP_OR_DIRECTORY --data-dir data_snapshot
```

The importer reads only the five filenames listed in the manifest and verifies their sizes and hashes. All five files must be present before the real-data study starts. See `docs/DATA_LICENSES_AND_ATTRIBUTION.md` for data sources, licence information and attribution.

## If a source has changed

Some providers, particularly the Open-Meteo historical reanalysis API, can revise their responses. If a downloaded file does not match its expected hash, keep it separately and obtain an authorized snapshot matching the recorded filename, size and hash. This code package does not supply a separate raw-data archive. A successful download alone is insufficient for exact reproduction; the inputs must pass the recorded hash checks.

Do not change an expected hash to accept different data when reproducing the paper. The program stops rather than substituting another dataset or a shortened demonstration.
