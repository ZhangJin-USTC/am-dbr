"""Candidate fixed-memory distributional methods under honest evaluation.

These methods are intentionally kept outside the manuscript-facing registry
until a development/validation/locked-test protocol supports their inclusion.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np
from scipy.stats import ks_2samp

from core import OnlineMethod, Step
from methods import DirectBinnedRecalibration, OCC


class BinnedWeightedPITECDF(OnlineMethod):
    """Fixed-memory histogram approximation to a weighted PIT empirical CDF.

    The method stores one exponentially decayed mass per equal-width PIT bin.
    Its piecewise-linear CDF uses cumulative mass at bin edges, so both memory
    and update cost are independent of the stream horizon.
    """

    def __init__(
        self,
        lam: float = 0.985,
        bins: int = 70,
        adaptive: bool = False,
        lam_min: float = 0.95,
        lam_max: float = 0.997,
        beta: float = 16.0,
        tau: float = 0.18,
        window: int = 120,
        adapt_every: int = 10,
    ):
        if not np.isfinite(float(lam)) or not 0.0 < float(lam) <= 1.0:
            raise ValueError("lam must be finite and lie in (0, 1].")
        if int(bins) < 4:
            raise ValueError("bins must be at least 4.")
        if not 0.0 < float(lam_min) <= float(lam_max) <= 1.0:
            raise ValueError("Require 0 < lam_min <= lam_max <= 1.")
        if not np.isfinite(float(beta)) or float(beta) < 0.0:
            raise ValueError("beta must be finite and non-negative.")
        if not 0.0 <= float(tau) <= 1.0:
            raise ValueError("tau must lie in [0, 1].")
        if adaptive and (int(window) < 4 or int(window) % 2):
            raise ValueError("window must be an even integer at least 4.")
        if int(adapt_every) < 1:
            raise ValueError("adapt_every must be positive.")
        self.lam = float(lam)
        self.bins = int(bins)
        self.adaptive = bool(adaptive)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.beta = float(beta)
        self.tau = float(tau)
        self.window = int(window)
        self.adapt_every = int(adapt_every)
        self._recent: deque[float] = deque(maxlen=self.window if adaptive else 1)
        self._current_lam = self.lam_max if adaptive else self.lam
        self._updates = 0
        self._mass = np.zeros(self.bins, dtype=float)
        self._edges = np.linspace(0.0, 1.0, self.bins + 1)
        self._cdf_edges = np.linspace(0.0, 1.0, self.bins + 1)
        self._cache_valid = True

    def update(self, step: Step) -> None:
        u = float(np.clip(step.u, 0.0, 1.0))
        if self.adaptive and len(self._recent) == self.window:
            if self._updates % self.adapt_every == 0:
                values = np.asarray(self._recent, dtype=float)
                half = self.window // 2
                distance = float(
                    ks_2samp(values[:half], values[half:], method="asymp").statistic
                )
                self._current_lam = self.lam_min + (
                    self.lam_max - self.lam_min
                ) / (1.0 + np.exp(self.beta * (distance - self.tau)))
        else:
            self._current_lam = self.lam_max if self.adaptive else self.lam
        self._mass *= self._current_lam
        bin_index = min(self.bins - 1, int(u * self.bins))
        self._mass[bin_index] += 1.0
        self._recent.append(u)
        self._updates += 1
        self._cache_valid = False

    def _refresh(self) -> None:
        total = float(np.sum(self._mass))
        if total <= 1e-15:
            self._cdf_edges = self._edges.copy()
        else:
            self._cdf_edges = np.concatenate(
                ([0.0], np.cumsum(self._mass, dtype=float) / total)
            )
            self._cdf_edges[-1] = 1.0
        self._cache_valid = True

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        if not self._cache_valid:
            self._refresh()
        values = np.asarray(u_grid, dtype=float)
        return np.interp(np.clip(values, 0.0, 1.0), self._edges, self._cdf_edges)

    def _inverse_g(self, target: float) -> float:
        if not self._cache_valid:
            self._refresh()
        q = float(np.clip(target, 0.0, 1.0))
        idx = int(np.searchsorted(self._cdf_edges, q, side="left"))
        if idx <= 0:
            return 0.0
        if idx >= self._cdf_edges.size:
            return 1.0
        x0, x1 = float(self._edges[idx - 1]), float(self._edges[idx])
        y0, y1 = float(self._cdf_edges[idx - 1]), float(self._cdf_edges[idx])
        # Only a zero-width CDF segment is a plateau.  Tiny positive slopes
        # must remain invertible instead of being rounded to the right edge.
        if y1 <= y0:
            return x1
        return float(x0 + (q - y0) * (x1 - x0) / (y1 - y0))

    def predict_interval(
        self,
        F_t_inv: Callable[[float], float],
        alpha: float,
        eps: float = 1e-4,
    ) -> tuple[float, float]:
        if not np.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        lower = float(np.clip(self._inverse_g(float(alpha) / 2.0), eps, 1.0 - eps))
        upper = float(
            np.clip(self._inverse_g(1.0 - float(alpha) / 2.0), eps, 1.0 - eps)
        )
        return F_t_inv(lower), F_t_inv(upper)

    @property
    def name(self) -> str:
        return "Adaptive-Binned-PIT-ECDF" if self.adaptive else "Binned-Weighted-PIT-ECDF"


# Candidate scripts now exercise the formal implementation while retaining the
# historical import name used by their frozen metadata.
BinnedWeightedPITECDF = DirectBinnedRecalibration


class OCCCoverageGuard(OnlineMethod):
    """OCC full-CDF map with a rolling conformal central-interval guard.

    The underlying OCC map remains available through ``predict_g``. For an
    interval request, the guard takes a rolling conformal quantile of the
    prequential scores ``abs(g_t(U_t) - 1/2)`` and inverts the current OCC map
    at the resulting symmetric calibrated-PIT endpoints. This is a candidate
    hybrid, not a claim of exchangeability validity under arbitrary drift.
    """

    def __init__(
        self,
        lam: float = 0.985,
        bins: int = 70,
        score_window: int = 250,
    ):
        if int(score_window) < 2:
            raise ValueError("score_window must be at least 2.")
        self.occ = OCC(lam=lam, bins=bins)
        self.score_window = int(score_window)
        self._scores: deque[float] = deque(maxlen=self.score_window)

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        return self.occ.predict_g(u_grid)

    def _score_quantile(self, alpha: float) -> float:
        if not self._scores:
            return float((1.0 - alpha) / 2.0)
        scores = np.sort(np.asarray(self._scores, dtype=float))
        n = scores.size
        rank = int(np.ceil((n + 1.0) * (1.0 - float(alpha))))
        rank = min(max(rank, 1), n)
        return float(scores[rank - 1])

    def predict_interval(
        self,
        F_t_inv: Callable[[float], float],
        alpha: float,
        eps: float = 1e-4,
    ) -> tuple[float, float]:
        if not np.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        radius = min(0.5, self._score_quantile(float(alpha)))
        lower_level = 0.5 - radius
        upper_level = 0.5 + radius
        lower = float(np.clip(self.occ._inverse_g(lower_level), eps, 1.0 - eps))
        upper = float(np.clip(self.occ._inverse_g(upper_level), eps, 1.0 - eps))
        return F_t_inv(lower), F_t_inv(upper)

    def update(self, step: Step) -> None:
        calibrated_pit = float(self.occ.predict_g(np.asarray([step.u], dtype=float))[0])
        self._scores.append(abs(calibrated_pit - 0.5))
        self.occ.update(step)

    @property
    def name(self) -> str:
        return "OCC-Fixed+Guard"


class OCCShortestInterval(OnlineMethod):
    """Extract the shortest fixed-mass interval from an OCC predictive CDF.

    For coverage level ``1-alpha``, every pair of calibrated-CDF levels
    ``(a, a + 1 - alpha)`` has the requested probability mass. The method
    searches ``a`` in ``[0, alpha]`` and returns the narrowest interval after
    mapping both endpoints through the current OCC inverse and base quantile.
    No outcomes or future observations enter this prediction-time choice.
    """

    def __init__(
        self,
        lam: float = 0.985,
        bins: int = 70,
        tail_grid_size: int = 41,
    ):
        if int(tail_grid_size) < 3:
            raise ValueError("tail_grid_size must be at least 3.")
        self.occ = OCC(lam=lam, bins=bins)
        self.tail_grid_size = int(tail_grid_size)

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        return self.occ.predict_g(u_grid)

    def predict_interval(
        self,
        F_t_inv: Callable[[float], float],
        alpha: float,
        eps: float = 1e-4,
    ) -> tuple[float, float]:
        if not np.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        lower_levels = np.linspace(0.0, float(alpha), self.tail_grid_size)
        upper_levels = lower_levels + 1.0 - float(alpha)
        lower_pit = np.asarray(
            [np.clip(self.occ._inverse_g(level), eps, 1.0 - eps) for level in lower_levels]
        )
        upper_pit = np.asarray(
            [np.clip(self.occ._inverse_g(level), eps, 1.0 - eps) for level in upper_levels]
        )
        lower_y = np.asarray([F_t_inv(float(value)) for value in lower_pit], dtype=float)
        upper_y = np.asarray([F_t_inv(float(value)) for value in upper_pit], dtype=float)
        widths = upper_y - lower_y
        finite = np.isfinite(widths) & (widths >= 0.0)
        if not np.any(finite):
            return self.occ.predict_interval(F_t_inv, float(alpha), eps=eps)
        min_width = float(np.min(widths[finite]))
        tolerance = 1e-12 * max(1.0, abs(min_width))
        near_minimum = np.flatnonzero(finite & (widths <= min_width + tolerance))
        center_index = near_minimum[
            int(np.argmin(np.abs(lower_levels[near_minimum] - float(alpha) / 2.0)))
        ]
        return float(lower_y[center_index]), float(upper_y[center_index])

    def update(self, step: Step) -> None:
        self.occ.update(step)

    @property
    def name(self) -> str:
        return "OCC-Fixed-Shortest"
