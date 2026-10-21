# Methods and provenance

## Discounted PIT-CDF trackers

Direct binned recalibration (DBR) denotes the discounted fixed-grid PIT-CDF tracker implemented in `framework/methods.py`. It combines exponentially discounted bin masses with continuous piecewise-linear interpolation. Discounted PIT updates, monotone interpolation and online forecast aggregation build on established methods in calibration and forecasting.

`DBR-Fixed` is the fixed-memory comparator in the paper. `DirectBinnedRecalibration` also provides an adaptive-decay option for the supplementary experiments.

## Adaptive-memory aggregation

`AM-DBR` (saved code label `AMDBR-core-noidentity`) combines seven DBR memory scales. It issues one convex PIT-CDF aggregate, scores each expert by PIT-scale CRPS after the outcome arrives, and updates the expert weights by exponential weighting and fixed share. The paper studies this procedure through switching-memory score comparisons, tracking bounds and coverage-transfer results under the stated assumptions.

## Two-stage rank/PAV comparators

`OCC-Fixed` and `OCC-Adaptive` are the saved code labels for the fixed and adaptive rank/PAV comparators in the supplementary material. Each first estimates discounted ranks and then fits a monotone map by weighted isotonic regression. They use the same PIT stream as the direct-CDF methods and provide structural and state-count comparisons.

## Reproduction

The method registry, parameters and chronological evaluation protocol are described in `docs/methods.md` and `docs/METHOD_IMPLEMENTATION_MAP.md`. The reference CSV files provide the expected numerical summaries. `python reproduce.py all` runs the studies and compares its outputs with these summaries. Input-data hashes, source hashes and completed-output checksums support consistent reproduction and safe resumption of the same run.
