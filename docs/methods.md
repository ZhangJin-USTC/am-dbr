# Methods and experiment protocol

Read `METHOD_SCOPE_AND_PROVENANCE.md` before interpreting the labels below:
DBR is an established baseline component, AM-DBR is the paper's specific
proposed procedure, and rank/PAV entries provide supplementary structural comparisons.

`framework/paper_config.py` defines numerical settings. The factory in
`scripts/adaptive_memory_realdata_study.py` is shared by main and window studies.
Predictions precede observing the current outcome. Warmup/pre-report observations
update methods but are excluded from reported metrics.

| Saved method name | Meaning and qualification |
|---|---|
| Base | Original forecast distribution |
| Static-PIT | Fitted once on warmup PITs |
| Rolling-PIT-ECDF | Empirical PIT distribution, window 300 |
| Weighted-PIT-ECDF | Forgetting 0.985, history cap 3000 |
| DCP-score | Rolling central conformity score; interval-only comparator |
| DBR-Fixed | **Discounted fixed-grid PIT-CDF tracker** (B=140, forgetting 0.985) |
| OCC-Fixed | **Two-stage rank/PAV comparator**, B=140, forgetting 0.985 |
| OCC-Adaptive | **Adaptive rank/PAV structural comparator**, B=140, recent window 120, forgetting [0.95,0.997] |
| AMDBR-core-noidentity | **Proposed AM-DBR procedure**: seven DBR experts, B=140 each, lambdas (0.90,0.95,0.98,0.99,0.995, 0.999, 1), learning rate 0.5, share 0.01, no identity expert |
| ACI-residual-projected | Projected ACI variant, gamma 0.005 |
| NExCP-residual-empirical | Empirical weighted quantile, not the test-atom finite-sample variant |
| PID-residual-projected | Projected residual PID variant |
| FACI-residual | Residual FACI recurrence |
| SF-OGD-residual | Residual scale-free OGD recurrence |
| SAOCP-residual | Residual strongly adaptive online conformal recurrence |

FACI, SF-OGD, SAOCP and NExCP are checked against pinned upstream reference
recurrences on five levels and 310 predictions per pair: 20 comparisons and
6200 predictions. This is not replication of every original paper's end-to-end
experiment. Reference code, commit and hashes are in `tests/reference`.

Nominal levels are 0.95, 0.90, 0.80, 0.70 and 0.50. PIT score is the exact integrated
squared PIT-CDF loss, not response-scale CRPS; no full-CDF score is assigned to
interval-only methods. Width refers to 90% intervals. A score improvement does
not automatically improve coverage gaps.

Simulations use six scenarios, seeds 6100-6107, 1500 observations. The constant
location scenario uses Gaussian noise; the other five use standardized t(6).
Seed is the pairing unit across scenarios. Exact paths appear in generated
`generator_details.tex`. There is no parameter search in reproduction.

Real data use chronological sorting, duplicate averaging, exact-timestamp lags
and missing-target/lag exclusion. The ridge forecast is fitted once on training
data. Source-clock conventions are retained. Dataset-specific train, warmup and
report settings are the `DATASETS` tuple; row shifts are not time lags.

Matched count-state sensitivity is a **structural ablation**: it compares OCC70
with DBR140, and AM7x140+7 weights with DBR987, across all 48 scenario/seed
cells. It does not equalize Python memory bytes and is not a claim of universal
rank/PAV inferiority. Report starts 0.75/0.85/0.95 are after cleaning, may overlap
and are not independent replications; the last OPSD window has 218 observations.
