# Paper-to-output map

Paths are relative to `--out`. Stable LaTeX labels are used because automatic table numbers can change during manuscript editing. `reproduce.py report` validates completed study outputs and generates the tables, figures and summaries below. Use `expanded_main_tables.tex` for the four main comparison tables and `expanded_supp_tables.tex` for the twelve supplementary comparison tables.

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

The generator also produces compact supporting summaries (`main_tables.tex`, comparison sections in `detailed_main_tables.tex` and `supp_tables.tex`, `score_coverage.pdf` and `real_comparison.pdf`). The paper comparison tables and coverage-width figures are the files listed above; each result table should be inserted once.

`main/artifact_validation.json`, `windows/window_validation.json` and `paper_assets/presentation_validation.json` check saved arrays, summaries and generated assets. `paper_assets/generated/expanded_validation.json` checks all 53 main-study cells and 795 method runs, recomputes interval scores from outcome and endpoint arrays, and records output checksums. `reference_comparison.json` compares numerical metrics with the reference CSVs. A complete reproduction has passed validation records and a passed `RUN_STATUS.json` for `all`, `resume` or `report`.

The paired interval-score table averages the six scenario differences within each seed, then uses the eight independent seed means for descriptive 95% Student intervals. The retrospective interval score did not drive method tuning. The interval score is checked by rebuilding it from saved outcome and endpoint arrays.

Each cell retains outcomes, interval endpoints at all five levels, full-CDF PIT scores where defined, summaries and hash receipts. Real-data widths and interval scores retain each dataset's native response units and are not pooled across datasets. Checksums verify file consistency. Conceptual diagrams, prose and mathematical proofs are provided in the manuscript source.

Method provenance is documented in `docs/METHOD_SCOPE_AND_PROVENANCE.md` and its Chinese companion.
