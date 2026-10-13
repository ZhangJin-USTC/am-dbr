# Method scope and provenance (v1.2.0)

This note is part of the public reproduction package. It prevents the code
labels from being read as priority claims.

## What is inherited

Direct binned recalibration (DBR) is the discounted fixed-grid PIT-map
construction used as a component and baseline. Related PIT-map updates,
forgetting recursions, monotone interpolation, and online CDF aggregation are
established ingredients in the calibration and forecasting literature. The
repository therefore does not claim that the name DBR, discounted PIT masses,
CRPS aggregation, or fixed-share weighting was invented in this project.

`DBR-Fixed` and `DBR-Adaptive` are included so that the paper's comparisons can
be reproduced. Their implementation is `DirectBinnedRecalibration` in
`framework/methods.py`.

## What the paper proposes

`AM-DBR` (code label `AMDBR-core-noidentity`) is the paper-facing procedure.
It instantiates a finite library of DBR memory scales, issues one convex
PIT-CDF aggregate, updates the weights with PIT-scale CRPS, and applies a
fixed-share transition. The paper's claim concerns this particular
specialization and its switching-memory oracle, tracking, and coverage-transfer
analysis under the stated assumptions. It does not claim priority for the
underlying DBR, CRPS aggregation, or fixed-share ingredients in isolation.

The numerical implementation and public API are unchanged in v1.2.0.
Presentation generators remove rank/PAV from primary tables while retaining
all supplementary results. One generator belongs to the locked study source
set, so `PUBLIC_SOURCE_MANIFEST.json` records a new source fingerprint.
Older checkpoints are not directly resumable with this release; start a fresh
output directory instead of rewriting historical receipts.

## Rank/PAV status

`OCC-Fixed` and `OCC-Adaptive` are retained as historical structural
comparators. They estimate a rank/PAV calibration map and are useful for
diagnostics and the matched state-count sensitivity. They are not part of the
AM-DBR contribution and are not needed to run the proposed method. Their full
definitions and results are retained in the supplementary material; the main
paper reports them only as a short structural ablation/comparator.

## Reproduction interpretation

The shipped CSV/NPZ reference results are inherited verified evidence. A
successful `reproduce.py all` run confirms package integrity, the frozen
protocol, data snapshots and numerical summaries; it does not establish a
priority claim or prove assumptions for dependent real-world streams. The
release label is v1.2.0 while the internal evidence protocol identifier
`1.0.0-rc.1` remains the schema identifier; it does not imply source-hash
compatibility with older releases.
