# AM-DBR: Adaptive-Memory Direct Binned Recalibration

Paper release: **v1.4.0 / R18**. Upload packaging: **R18-repository-20261013**. The repository identity and publication status are recorded in `REPOSITORY_IDENTITY.json`.

Documentation package: **docs-v1.4.0**. Numerical study source: **code-v1.2.0**, with the v1.4.0 presentation/reporting layer. The numerical algorithms, settings, data snapshots and reference CSVs are unchanged. See `DOCS_RELEASE.json`.

Code for the current research candidate, including every paper method and
baseline. Extract into a new directory; do not mix modules from older versions.
[中文](README_zh.md) · [Methods](docs/methods.md) ·
[Outputs](docs/artifact_map.md) · [Data](docs/data_access.md)

## IDE and notebook entry points

Open `make_local_workspace.py` in Spyder and click Run without editing it. It
creates an editable sibling workspace containing `run_from_IDE.py` and
`AM_DBR_local_verification.ipynb`, with CODE pointing at the original package.
Edit/run those external copies. Do not edit package files or execute/save its
notebook template: all are covered by package integrity hashes. Notebook
autosave affects only the external work copy. The launcher uses a selected
Python 3.12 interpreter and streams output in the IDE.
Follow `LOCAL_VERIFICATION_zh.txt` for environment setup, exact data preparation,
check, smoke, full run, resume and acceptance. Do not upload the personal
workspace, environments, local paths or generated runs.

This release retains the verified rc.5 numerical core and reference metrics;
the v1.4.0 update also makes the method scope explicit in the paper-facing
documentation. Direct binned recalibration (DBR) is treated as an established
PIT-map building block and baseline. The paper's method is AM-DBR: a finite
library of DBR memory scales combined with fixed-share aggregation and analysed
for switching-memory, tracking, and coverage transfer. Rank/PAV implementations
(`OCC-Fixed` and `OCC-Adaptive`) remain runnable as historical structural
comparators/ablations; they are not claimed as the paper's contribution.
No numerical method or API was changed. Presentation generators now omit
rank/PAV from primary tables while preserving its supplementary results.
The locked numerical source fingerprint remains the v1.2.0 fingerprint. The v1.4.0 changes are outside that locked numerical set and concern reporting only. Existing completed outputs from the same v1.2.0 fingerprint can be checked and reported again. Earlier rc.5/v1.1.0 arrays require the separately reviewed historical-evidence workflow and are not accepted by a normal fresh run.

## Expanded empirical comparison (v1.4.0)

`python revision_tools/build_expanded_comparisons.py --results RESULTS --assets ASSETS` writes the expanded report assets after a checked current run. The command verifies all 53 main cells, all 15 methods and all five nominal levels before writing four primary tables, eleven detailed supplementary tables, the paired-seed interval-score table, and two coverage-width figures. `reproduce.py report` calls this generator automatically. The 90% interval score is an additional evaluation of the locked predictions and was not used to tune any method. Historical cached arrays are handled only by the separately reviewed evidence-audit workflow; they are never silently accepted by a normal fresh run.

## Three commands after installing Python 3.12

Windows PowerShell (no activation or execution-policy change needed):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python reproduce.py all --workers 1
```

macOS / Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python reproduce.py all --workers 1
```

`all` verifies code and dependency integrity, runs tests and pinned upstream
recurrence comparisons, obtains and checks exact input snapshots, runs all three
studies, validates saved arrays, generates every current numerical paper table
and figure, then compares all metrics against shipped reference summaries.
The reference environment is Python 3.12.x on Windows x86-64. Commands are
portable; supplied Linux/macOS CI and Docker configurations are not evidence
that those platforms have already executed the full study.

Exact historical inputs are required. Mutable sources may return different data;
a hash mismatch stops execution before the real-data study. See the verified
[data access status](docs/data_access.md). If exact public retrieval fails, obtain
an authorized snapshot and run `reproduce.py data --import-data PATH` before
repeating `all`. Public code alone cannot guarantee future snapshot availability.

No GPU, R, MATLAB or LaTeX is needed to calculate results and render figures.
The full study is a long CPU job. `--workers 1` minimizes parallel memory load;
increase it when RAM permits. It is not a five-minute demo, and machine resources
prevent a literal guarantee for every computer.

## Check, resume and inspect

Using the environment's Python executable above:

```text
python reproduce.py check
python reproduce.py smoke
python reproduce.py data --verify-only
python reproduce.py resume --workers 2
python reproduce.py report
```

`smoke` runs one complete 1,500-observation synthetic cell, all 15 methods and
five levels, against reference metrics; it needs no third-party data. It is not
a full-paper reproduction. `all`/`resume` verify matching completed cells.
Main/window studies preserve uncommitted method files under `_interrupted` and resume
at verified method checkpoints. State sensitivity
preserves an interrupted uncommitted cell under `state/_interrupted`, then
recomputes it. Corrupt committed evidence is rejected. Do not use two writers
on one output directory. `report` validates a full run and regenerates assets.
`--out PATH` selects a separate run and `--data-dir PATH` an exact input cache.

Read `outputs/reference_comparison.json` for numerical acceptance and
`outputs/paper_assets/generated` for TeX tables, narrative numbers, statistics,
and PNG/PDF figures. Tolerances are `rtol=1e-6`, `atol=1e-8`; runtimes are
excluded. Floating-point reproducibility is checked. Byte-identical PDFs,
compressed arrays, timings or results across every CPU/BLAS are not promised.
`outputs/environment.json` and `commands.log` record the run. Require
`outputs/RUN_STATUS.json` to say `passed` for command `all`, `resume` or `report`,
plus a passed full `reference_comparison.json` and
`paper_assets/generated/expanded_validation.json` confirming 53 cells and 795 method runs;
a passed smoke/check is insufficient. `LocalVerification.show_results()` checks all three records.

| Study | Cells | Methods per cell | Levels |
|---|---:|---:|---:|
| Main simulations: 6 scenarios × 8 fixed seeds | 48 | 15 | 5 |
| Main real data: Seattle, Phoenix, Jena, Metro, OPSD | 5 | 15 | 5 |
| Matched count-state sensitivity | 48 | 2 additional settings | 5 |
| Report-start sensitivity: 3 starts × 5 datasets | 15 | 15 | 5 |

The main study stores 5,217,000 interval predictions. Exact noise laws, splits,
names and baseline qualifications are in [methods.md](docs/methods.md). The
method provenance and claim boundary are recorded in
[METHOD_SCOPE_AND_PROVENANCE.md](docs/METHOD_SCOPE_AND_PROVENANCE.md).
Unfavorable results are retained: smaller PIT score need not mean smaller
coverage gap. These are corrective analyses of previously inspected data with
frozen settings, not blinded new experiments or tuning.

`PUBLIC_PACKAGE_MANIFEST.json` records the upload payload hashes;
`PUBLIC_SOURCE_MANIFEST.json` records hashes of 24 locked study source files,
all unchanged from code-v1.2.0. The added v1.4.0 report generator is covered by the
package manifest and the expanded report's provenance record.
`.gitattributes` disables newline conversion. Some retained numerical evidence
records refer to the internal `v1.0.0-rc.1` source fingerprint; that is historical
provenance, not the public package version. The complete package and revised paper
are `v1.4.0`; the numerical study core remains identified as `v1.2.0`. See
`RELEASE.json`.

Optional container (replace the host paths):

```text
docker build -t am-dbr:v1.4.0 .
docker run --rm -v YOUR_OUTPUT_PATH:/work/outputs -v YOUR_DATA_PATH:/work/data_snapshot am-dbr:v1.4.0 all
```

Docker execution was not tested on the Windows review machine. The normal Python
route is verified. Docker Desktop is an additional installation on Windows/macOS.
The project authors must still select a license for original code; see
`LICENSE_STATUS.md`. Upstream reference code retains its Apache-2.0 license
and copyright notices. This is a research candidate, not journal readiness.
