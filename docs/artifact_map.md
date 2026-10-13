# v1.4.0 paper-to-output map

Paths are relative to `--out`. Stable LaTeX labels are used because automatic table numbers can change during manuscript editing. `reproduce.py report` generates the original supporting assets and then the expanded v1.4.0 comparison. Use the expanded files for the current main comparisons and the twelve supplementary comparison tables.

| Paper item | Output under `paper_assets/generated` |
|---|---|
| `tab:method-comparison`, comparator qualifications | `method_comparison_table.tex` |
| `tab:sim`, `tab:sim-detail`, `tab:real`, `tab:real-detail` | `expanded_main_tables.tex` (four tables, thirteen primary implementations; PIT columns apply only to full-CDF methods) |
| `tab:state-order` | `detailed_main_tables.tex` (retain only this state-sensitivity table for the current paper) |
| Simulation coverage-width figure, `fig:tradeoff` | `synthetic_coverage_width.pdf` and `.png` |
| Real-data coverage-width figure, `fig:real-tradeoff` | `real_coverage_width.pdf` and `.png` |
| Six `tab:scenario-*` and five `tab:data_set-*` tables, followed by `tab:paired-is90` | `expanded_supp_tables.tex` (twelve tables) |
| `tab:paired-is90` alone | `expanded_paired_is90.tex` (same table as the last table in `expanded_supp_tables.tex`; insert once) |
| Cleaning and main report-window metadata | `supp_tables.tex` (retain the cleaning/window metadata tables, not its earlier comparison tables) |
| `tab:window-sensitivity` | `window_table.tex` |
| Exact synthetic definitions | `generator_details.tex` |
| Results text with calculated numerical summaries | `expanded_simulation_narrative.tex`, `expanded_real_narrative.tex`; the manuscript contains additional interpretation |
| All 795 main method runs, including retrospective 90% interval score | `expanded_all_method_metrics.csv` |
| Paired eight-seed differences, confidence limits, and number of negative seed differences | `expanded_paired_seed_intervals.csv` |
| Per-scenario and per-dataset means | `expanded_synthetic_means.csv`, `expanded_real_means.csv` |
| Additional state/window statistics | `statistics.json` |
| Input/preprocessing metadata | `data_metadata.json` |

The original `main_tables.tex`, earlier comparison portions of `detailed_main_tables.tex`/`supp_tables.tex`, `score_coverage.pdf`, and `real_comparison.pdf` remain for compatibility with previous reports. They are not the authoritative v1.4.0 comparison assets. Do not insert both the earlier and expanded versions of a result table.

`main/artifact_validation.json`, `windows/window_validation.json`, and `paper_assets/presentation_validation.json` record the original array/receipt verification. `paper_assets/generated/expanded_validation.json` records verification of all 53 main cells and 795 method runs, the locked-array interval-score calculation, and hashes of all expanded outputs including `expanded_paired_is90.tex`. `reference_comparison.json` separately compares all original numerical metrics with the supplied reference CSVs. Full acceptance requires passed records and a passed `RUN_STATUS.json` for `all`, `resume`, or `report`; exit code zero alone is insufficient.

The paired interval-score table averages the six scenario differences within each seed, then uses the eight independent seed means for descriptive 95% Student intervals. The retrospective interval score did not drive method tuning. The original reference CSVs are unchanged; the expanded score is checked by rebuilding it from verified saved outcome and endpoint arrays.

Each cell retains outcomes, interval endpoints at all five levels, full-CDF PIT scores where defined, summaries and hash receipts. Real-data widths and interval scores retain each dataset's native response units and are not pooled across datasets. The receipts check consistency; they are not an author's digital signature. Conceptual diagrams, prose and mathematical proofs belong to the same-version manuscript source rather than the empirical generator.

Method provenance is documented in `docs/METHOD_SCOPE_AND_PROVENANCE.md` and its Chinese companion. The numerical core remains code-v1.2.0; the expanded reporting layer is v1.4.0.
