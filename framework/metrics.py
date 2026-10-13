"""Evaluation metrics for the OCC SOTA comparison framework."""

from __future__ import annotations

import numpy as np


def marginal_coverage_error(y: np.ndarray, L: np.ndarray, U: np.ndarray, alpha: float) -> float:
    """Absolute marginal coverage error at target coverage 1-alpha."""
    y = np.asarray(y, dtype=float)
    L = np.asarray(L, dtype=float)
    U = np.asarray(U, dtype=float)
    covered = (L <= y) & (y <= U)
    return float(abs(np.mean(covered) - (1.0 - alpha)))


def interval_width(L: np.ndarray, U: np.ndarray) -> dict[str, float]:
    """Mean, median, and quantiles of interval width."""
    width = np.asarray(U, dtype=float) - np.asarray(L, dtype=float)
    return {
        "mean": float(np.mean(width)),
        "median": float(np.median(width)),
        "q10": float(np.quantile(width, 0.10)),
        "q90": float(np.quantile(width, 0.90)),
    }


def pit_ks_distance(pit: np.ndarray | None) -> float:
    """Kolmogorov distance between calibrated PIT values and Uniform(0,1)."""
    if pit is None:
        return float("nan")
    u = np.sort(np.asarray(pit, dtype=float))
    u = u[np.isfinite(u)]
    if u.size == 0:
        return float("nan")
    u = np.clip(u, 0.0, 1.0)
    n = u.size
    upper = np.arange(1, n + 1) / n - u
    lower = u - np.arange(0, n) / n
    return float(max(np.max(upper), np.max(lower)))


def pit_cvm_distance(pit: np.ndarray | None) -> float:
    """Cramer-von Mises distance to Uniform(0,1), scaled as a mean square gap."""
    if pit is None:
        return float("nan")
    u = np.sort(np.asarray(pit, dtype=float))
    u = u[np.isfinite(u)]
    if u.size == 0:
        return float("nan")
    u = np.clip(u, 0.0, 1.0)
    n = u.size
    grid = (np.arange(1, n + 1) - 0.5) / n
    return float(np.mean((u - grid) ** 2))


def pit_crps_from_cdf(
    grid: np.ndarray,
    cdf_values: np.ndarray,
    outcome: float,
) -> float:
    """Numerical CRPS for a forecast CDF of the next raw PIT value."""
    x = np.asarray(grid, dtype=float)
    g = np.asarray(cdf_values, dtype=float)
    if x.ndim != 1 or g.ndim != 1 or x.size != g.size or x.size < 2:
        raise ValueError("grid and cdf_values must be one-dimensional and aligned.")
    if not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0.0):
        raise ValueError("grid must be finite and strictly increasing.")
    target = (x >= float(outcome)).astype(float)
    return float(np.trapezoid((g - target) ** 2, x))


def rolling_coverage(
    y: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    window: int = 250,
    min_periods: int | None = None,
) -> np.ndarray:
    """Rolling empirical coverage for interval arrays."""
    y = np.asarray(y, dtype=float)
    covered = ((np.asarray(L) <= y) & (y <= np.asarray(U))).astype(float)
    if min_periods is None:
        min_periods = max(20, window // 5)
    out = np.full_like(covered, np.nan, dtype=float)
    csum = np.r_[0.0, np.cumsum(covered)]
    for i in range(covered.size):
        start = max(0, i + 1 - window)
        count = i + 1 - start
        if count >= min_periods:
            out[i] = (csum[i + 1] - csum[start]) / count
    return out


def multi_alpha_gap(results_by_alpha: dict[float, tuple[np.ndarray, np.ndarray]], y: np.ndarray) -> float:
    """Average absolute coverage gap over all provided alpha levels."""
    gaps = []
    for alpha, (L, U) in results_by_alpha.items():
        gaps.append(marginal_coverage_error(y, L, U, alpha))
    return float(np.mean(gaps)) if gaps else float("nan")
