# Frozen input provenance and attribution

The five inputs are identified by byte length and SHA-256 in
`data_snapshot_manifest.json`. `python reproduce.py data` downloads and verifies
them before any paper experiment. `--data-dir` supports a previously acquired
offline cache. Existing different bytes are never overwritten or accepted.
The public code ZIP does not contain raw inputs.

## Seattle and Phoenix

Weather data by [Open-Meteo.com](https://open-meteo.com/), using its historical
API. These are daily gridded reanalysis temperatures, not station observations.
The API responses for 1980-01-01 through 2025-12-31 are reduced to the date and
daily mean-temperature columns. CSV serialization is UTF-8 without a BOM,
explicit CRLF on every operating system, and Python floating-point text. No
temperature is altered during serialization. Both source queries and exact
resulting byte hashes are in the manifest.

The [provider's licence page](https://open-meteo.com/en/licence) states that API
data are under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The free API has separate noncommercial service terms and request limits
([terms](https://open-meteo.com/en/terms)). The historical endpoint is mutable:
it may eventually return revised data. Verification then stops and an exact
archived snapshot is needed; the tool does not relabel the new values as the
paper's input.

## Jena

Max Planck Institute for Biogeochemistry, Jena weather station, 2009-2016 data,
distributed in the public TensorFlow/Keras archive. The raw archive is unchanged
in this study. The originating institute's
[download page](https://www.bgc-jena.mpg.de/wetter/weather_data.html) specifies
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The study's timestamp
aggregation, missing-lag exclusions and feature construction are downstream
transformations implemented in `real_data/loaders.py`.

## Metro

Hogue, J. (2019), *Metro Interstate Traffic Volume*, UCI Machine Learning
Repository, [doi:10.24432/C5X60B](https://doi.org/10.24432/C5X60B).
The [dataset page](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The raw downloaded archive is unchanged. Duplicate target means and exact-time
lag construction are documented downstream study transformations.

## OPSD: immutable retrieval, separate upstream rights

The input is Jennifer Walker's daily export in `jenfly/opsd`, pinned to commit
`63472081622b39f53ff900b3d69c419c5b50e7aa`, file blob
`312bc7984e453559155b2df404c5f991148c3e08`. The GitHub content API and raw-file
fallback retrieve this same pinned file, never the changing `master` branch.
The file's SHA-256 remains the final acceptance criterion.

The mirror's [preprocessing notebook](https://github.com/jenfly/opsd/blob/63472081622b39f53ff900b3d69c419c5b50e7aa/time-series-preprocessing.ipynb)
uses Open Power System Data's hourly `time_series` release 2018-06-30. It selects
Germany in 2006-2017, renames `DE_load_entsoe_power_statistics` to `Consumption`,
converts MW to GW, and aggregates each UTC day with 24 required observations to
GWh. The paper uses only this consumption target and its lags, not the wind or
solar columns. Attribution: Open Power System Data (2018), *Data Package Time
series*, version 2018-06-30,
[doi:10.25832/time_series/2018-06-30](https://doi.org/10.25832/time_series/2018-06-30),
with primary German load data from ENTSO-E Data Portal and Power Statistics;
daily preprocessing by Jennifer Walker.

The mirror carries an MIT software notice. That notice does not establish that
every underlying source dataset can be relicensed. The upstream
[release README](https://data.open-power-system-data.org/time_series/2018-06-30/README.md)
provides attribution; its machine-readable metadata does not declare a blanket
data licence. OPSD's [legal page](https://open-power-system-data.org/legal/)
explains the separate rights of primary providers. Therefore the public code
release retrieves this already-public file by immutable reference and does not
redistribute it in an archive. This is a factual provenance distinction, not a
new licence for the source data.

## Offline and archival copies

The local delivery contains an optional four-source CC-BY snapshot add-on with
this attribution file and a byte manifest. It excludes OPSD. It may be uploaded
as an accompanying release asset to preserve the mutable weather responses.
The local complete five-file recovery ZIP is marked **LOCAL BACKUP — NOT THE
PUBLIC DATA ADD-ON** because OPSD redistribution terms have not been established.
After acquiring the OPSD file, import the four-source add-on with
`python reproduction/data_snapshot.py import --source PATH_TO_ZIP --data-dir data_snapshot`.
The complete local backup can also be imported for offline verification.

The URLs and licence statements above were checked on 2026-09-20 (Asia/Shanghai).
