# AM-DBR: Adaptive-Memory Direct Binned Recalibration

This anonymous repository accompanies the manuscript **Adaptive-Memory Online Distributional Recalibration under Distribution Shift**.

The software implements the proposed AM-DBR procedure and the comparison methods used in the manuscript. AM-DBR maintains a finite library of discounted PIT-CDF trackers at different memory scales and combines their maps with fixed-share proper-score aggregation. Direct binned recalibration (DBR) is included as a fixed-memory building block and comparison method. The repository also contains the interval-only and rank/PAV comparison implementations reported in the supplementary material.

## Quick start

The checked reference environment uses Python 3.12 on Windows x86-64. No GPU, R, MATLAB or LaTeX is required to run the numerical studies.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python reproduce.py check
.venv\Scripts\python reproduce.py smoke
```

On macOS or Linux, use Python 3.12 with these commands:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python reproduce.py check
.venv/bin/python reproduce.py smoke
```

`check` verifies the package and environment. `smoke` runs a short complete synthetic study. To reproduce all reported numerical results, use:

```powershell
.venv\Scripts\python reproduce.py all --workers 1
```

On macOS/Linux, use `.venv/bin/python reproduce.py all --workers 1` for the same full study.

The full study is a long CPU computation. It retrieves and checks the required public data snapshots, runs the simulation and real-data studies, validates the stored outputs, and generates the tables and figures used by the manuscript. If a historical data endpoint has changed, use an authorized exact snapshot with `reproduce.py data --import-data PATH` before rerunning the study. The program stops on a data-hash mismatch rather than silently using different inputs.

For an existing completed run, `reproduce.py report` validates the outputs and regenerates the paper assets. `reproduce.py resume` continues an interrupted run from verified checkpoints. Do not let two processes write to the same output directory.

## Where to find results

- `outputs/paper_assets/generated`: tables, statistics, narrative values and figures.
- `outputs/main`: simulation and main real-data results.
- `outputs/state`: matched state-size sensitivity results.
- `outputs/windows`: report-window sensitivity results.
- `outputs/reference_comparison.json`: comparison with the supplied reference summaries.
- `outputs/RUN_STATUS.json`: overall run status.

A complete reproduction requires a passed `RUN_STATUS.json`, a passed reference comparison, and the expanded asset validation record. The supplied reference summaries are for checking the implementation; they do not replace a fresh run when exact input data are available.

## Using an IDE or Jupyter

Run `make_local_workspace.py` from Spyder or another Python IDE. It creates an editable personal workspace and a notebook launcher outside the package directory. Use the generated workspace for notebook execution and saved outputs; keep local environments, downloaded data and generated runs out of the anonymous repository.

## Data and licensing

Raw third-party data are not redistributed. See `docs/data_access.md` and `docs/DATA_LICENSES_AND_ATTRIBUTION.md` for sources, retrieval instructions and attribution requirements. The original project code license must be selected by the authors before a non-anonymous public release. Upstream reference files retain their own license notices.
