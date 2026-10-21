# Paper names, code labels, and common comparison protocol

The authoritative study registry is `scripts/adaptive_memory_realdata_study.py`
(`methods_factory`); it extends `scripts/fixed_adaptive_refresh.py`. All main
synthetic cells, real-data cells and report-window cells instantiate this same
15-method registry. Parameters below are fixed before these fixed-protocol analyses.

| Paper method / output label | Implementation | Parameter setting / operational scope |
|---|---|---|
| Base | `framework/methods.py:BaseCDF` | Frozen base forecast; identity PIT map |
| Static-PIT | `StaticPIT` | Warm-up anchored rank map; no online updates |
| Rolling-PIT-ECDF | `RollingPITECDF` | 300 PITs; empirical step CDF |
| Weighted-PIT-ECDF | `WeightedPITECDF` | Decay 0.985; at most 3000 PITs |
| DBR-Fixed | `DirectBinnedRecalibration` | **Discounted fixed-grid PIT-CDF tracker.** 140 masses; decay 0.985 |
| Fixed rank/PAV / OCC-Fixed | `OCC` | **Two-stage rank/PAV structural comparator.** 140 bins; decay 0.985; two sufficient-statistic arrays |
| Adaptive rank/PAV / OCC-Adaptive | `OCC` | **Two-stage rank/PAV structural comparator.** 140 bins; logistic decay range [0.95, 0.997], threshold 0.18, slope 16, diagnostic window 120 |
| AM-DBR / AMDBR-core-noidentity | `AdaptiveMemoryDBR` | **Paper's proposed procedure.** Seven DBR experts: 0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1; 140 bins; eta 0.5; fixed share 0.01; no identity expert |
| DCP-score | `DCP` | 300 rolling central PIT scores; interval only; full base support if required rank exceeds sample size |
| ACI-residual-projected | `framework/residual_baselines.py:ResidualACI` | Step 0.005; empirical residual quantile; nonnegative radius projection |
| NExCP-residual-empirical | `ResidualNExCP` | Decay `1-alpha/4`; full residual history; empirical weighted quantile, no infinite test-score atom |
| PID-residual-projected | `ResidualPID` | Learning rate 0.10; saturation 5; integral gain 10; burn-in window 100; no scorecaster |
| FACI-residual | `ResidualFACI` | Rates `0.001*2**k`, k=0,...,7; I=100; scalar residual controller |
| SF-OGD-residual | `ResidualSFOGD` | Warm-up scale `sqrt(3)*max(abs residual)`; scale floor 1e-12 for all-zero warm-up |
| SAOCP-residual | `ResidualSAOCP` | Same warm-up scale; lifetime multiplier 8; scalar residual controller |

The five alpha values are 0.05, 0.10, 0.20, 0.30 and 0.50. Full-CDF methods share
one issued map at every level. Each interval-only method has a separate state
per alpha, initialized and updated with that alpha. For every observation the
forecast is issued before its outcome updates state. PIT-CRPS is computed only
for methods issuing a full CDF; an interval controller's missing score is not
an omitted result. The CDF score and inverse operations use exact knots in
`framework/cdf_tools.py`.

The SF-OGD, SAOCP, FACI and empirical NExCP recurrence checks compare against
unmodified, commit-pinned Salesforce reference source. These checks cover
20 method/alpha combinations and 6200 predictions. They do not assert exact
equivalence to an entire external forecasting pipeline. ACI/PID projections,
the empirical NExCP quantile and zero-scale extension remain explicitly named.

Synthetic data use 6 scenarios, seeds 6100-6107, 300 warm-up and 1200 report
observations. Constant location uses Gaussian noise; the other 5 scenarios use
standardized t6 noise. Real base models are ridge fits on the first 20% of
retained timestamps, with calendar and exact-time target lags only. Calibrators
warm up through 45%, and continue updates until their report starts. The main
report starts at fractions 0.70/0.70/0.65/0.65/0.55 for Seattle/Phoenix/Jena/
Metro/OPSD, with caps 2500/2500/2500/2500/2000. The OPSD actual main report is
1960 observations. Report-start sensitivity fixes 0.75/0.85/0.95 and retains
short tail windows, including OPSD at 0.95 with 218 observations.

State-count sensitivity deliberately changes only comparator resolution:
OCC-Fixed-B70 and DBR-Fixed-B987. It is reported separately, not substituted for
the main 15-method comparison. The known counts 140, 280 and 987 refer to
selected accumulators, not measured resident memory.

Expected numerical summaries are in `reproduction/reference_results/`.
Runtime is excluded from numerical agreement checks because concurrency and
hardware affect it. Numeric tolerances allow small floating-point differences;
matching summaries are neither a new blinded experiment nor a proof of
theoretical assumptions in dependent real data.
