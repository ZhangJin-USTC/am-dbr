"""
core.py — base abstractions for online conformal / calibration methods.

Every method implements ``OnlineMethod``. The experiment loop only knows about
this interface, so methods can be swapped without per-method special-casing.

Two modes a method can support (a method may support either or both):
  • ``predict_g(u)``       — returns hat_g_t(u): the PIT recalibration map, a
                            non-decreasing CDF on [0,1]. Methods that
                            recalibrate the predictive distribution provide
                            this (OCC, Static-PIT, and PIT-ECDF baselines).
  • ``predict_interval(F_inv, alpha)``
                          — returns (L_t, U_t): prediction interval at level
                            1-alpha. Methods that directly target intervals
                            provide this (DCP-score and residual conformal
                            baselines). Methods that produce hat_g_t
                            get a default implementation via inversion +
                            base-quantile composition.

Inputs at each step ``t``:
  u_t  = F_t(y_t)             PIT value of the realised observation under the
                              base predictive CDF F_t (you provide this from
                              your model).
  y_t                         optional, the observation itself (used by some
                              methods that look at residuals).
  x_t                         optional, features (used by quantile-regression
                              methods).
  F_t_inv                     inverse base CDF, used to convert calibrated
                              PIT levels back into y-units for intervals.

The expected flow per step:
  1. At time t, the user already knows F_t (from the base model) and x_t.
  2. method.predict_interval(F_t_inv, alpha) gives (L_t, U_t).
  3. Observation y_t arrives; the user computes u_t = F_t(y_t).
  4. method.update(u_t, y_t=y_t, x_t=x_t).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import numpy as np


@dataclass
class Step:
    """One step of the online stream. Convenience container; not all fields
    need to be populated for every method."""
    t: int
    u: float                              # PIT value F_t(y_t) ∈ [0,1]
    y: Optional[float] = None
    x: Optional[np.ndarray] = None
    F_t_inv: Optional[Callable[[float], float]] = None  # base-CDF inverse


class OnlineMethod(ABC):
    """Abstract online conformal / calibration method.

    Implementations should keep their internal state in instance attributes
    and treat ``update`` as the only way to absorb new observations.
    """

    # ------ initialisation ------

    def warmup(self, warmup_steps: Sequence[Step]) -> None:
        """Default implementation: just feed warmup steps through update().

        Override if a different warm-up procedure is needed (e.g., a batch
        fit).
        """
        for s in warmup_steps:
            self.update(s)

    # ------ prediction ------

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        """Return hat_g_t evaluated on ``u_grid``.

        Methods that do not produce a recalibration map should raise
        ``NotImplementedError`` (the default).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not produce a PIT calibration map"
        )

    def predict_interval(
        self,
        F_t_inv: Callable[[float], float],
        alpha: float,
        eps: float = 1e-4,
    ) -> tuple[float, float]:
        """Return (L_t, U_t), the (1-alpha) prediction interval.

        Default implementation: invert hat_g_t at levels alpha/2 and 1-alpha/2
        exactly at the CDF knots, then push through F_t_inv. Methods that produce
        intervals directly (e.g., ACI) should override.
        """
        del eps
        from cdf_tools import inverse_cdf
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        x, y, kind = self._pit_knots()
        return (F_t_inv(inverse_cdf(x, y, alpha/2, kind)),
                F_t_inv(inverse_cdf(x, y, 1-alpha/2, kind)))

    def _pit_knots(self):
        """Return knots, CDF values, and interpolation kind for exact scoring."""
        raise NotImplementedError("This method has no exact CDF representation.")

    def score_pit(self, outcome: float) -> float:
        from cdf_tools import cdf_score
        x, y, kind = self._pit_knots()
        return cdf_score(x, y, outcome, kind)

    # ------ update ------

    @abstractmethod
    def update(self, step: Step) -> None:
        """Incorporate observation at step ``t`` into internal state."""
        ...

    # ------ housekeeping ------

    @property
    def name(self) -> str:
        return self.__class__.__name__


# -----------------------------------------------------------------------------
# Helpers shared across methods
# -----------------------------------------------------------------------------


def isotonic_fit_predict(
    x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit PAV-isotonic on (x, y[, w]) and return a callable predictor.

    Light wrapper around sklearn so callers don't import sklearn directly.
    Inputs assumed to be in [0,1]. Returns a callable that clips inputs.
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(x, y, sample_weight=w)
    return iso.predict


def coverage_indicator(y: float, L: float, U: float) -> int:
    return int(L <= y <= U)
