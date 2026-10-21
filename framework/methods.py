"""
methods.py — implementations of OCC and competing baselines.

Methods covered:
    DirectBinnedRecalibration — fixed-state weighted PIT-CDF tracker
    OCC                    — binned, forgetting-weighted PIT recalibration
    StaticPIT              — isotonic map fit once on warm-up
    RollingPITECDF         — custom rolling PIT empirical CDF
    WeightedPITECDF        — custom finite-buffer weighted PIT empirical CDF
    DCP                    — central PIT-score interval adaptation
    ACI                    — project-specific PIT-level interval wrapper

Residual-score ACI, NExCP, FACI, SF-OGD and SAOCP implementations live in
``residual_baselines.py``.  They are interval-only and are kept separate from
the full-CDF PIT map track.

Each method exposes the OnlineMethod interface from core.py. See README.md for
which methods support which evaluation mode.

The legacy aliases ``RollingIDR`` and ``WeightedDCP`` are retained only for
old notebooks.  Paper-facing registries use the explicit names above and do
not attribute IDR or DCP guarantees to the custom PIT-ECDF ablations.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
from typing import Callable, Optional, Sequence
import math
import numpy as np
from scipy.stats import ks_2samp

from core import OnlineMethod, Step, isotonic_fit_predict
from cdf_tools import anchored_cdf_values, cdf_score, inverse_cdf


Q_GRID = np.linspace(0.001, 0.999, 401)


def load_aci_baselines(path: str | Path | None = None):
    """Load the scalar adaptive conformal baselines used in earlier runs."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.append(Path(__file__).resolve().with_name("ACI_new_baselines.py"))
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("aci_new_baselines_external", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find ACI_new_baselines.py.")


# =============================================================================
# OCC (Online Conformal Calibration) — the paper's method
# =============================================================================


class BaseCDF(OnlineMethod):
    """Uncalibrated base predictive distribution."""

    def _pit_knots(self):
        return np.array([0., 1.]), np.array([0., 1.]), "linear"

    def update(self, step: Step) -> None:
        pass

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(u_grid, dtype=float), 0.0, 1.0)


class ScalarAdaptiveCP(OnlineMethod):
    """Adapter for interval-only adaptive conformal baselines.

    The older ACI-family baselines consume a predictive quantile grid ``(qs,
    qv)`` and expose ``interval(qs, qv)`` / ``observe(...)``. This adapter lets
    them run inside the Step/F_t_inv framework. They do not produce a PIT
    recalibration map, so PIT metrics remain undefined for these methods.
    """

    interval_only = True

    def __init__(
        self,
        method_cls,
        display_name: str,
        alpha_target: float = 0.10,
        kwargs: dict | None = None,
        q_grid: np.ndarray | None = None,
    ) -> None:
        self.method_cls = method_cls
        self.display_name = display_name
        self.alpha_target = float(alpha_target)
        self.kwargs = dict(kwargs or {})
        self.q_grid = np.asarray(Q_GRID if q_grid is None else q_grid, dtype=float)
        self.model = self.method_cls(alpha_target=self.alpha_target, **self.kwargs)
        self._last_interval: tuple[float, float] | None = None
        self._last_qv: np.ndarray | None = None

    def set_alpha_target(self, alpha: float) -> None:
        self.alpha_target = float(alpha)
        self.model = self.method_cls(alpha_target=self.alpha_target, **self.kwargs)
        self._last_interval = None
        self._last_qv = None

    def _quantile_values(self, F_t_inv: Callable[[float], float]) -> np.ndarray:
        return np.asarray(F_t_inv(self.q_grid), dtype=float)

    def predict_interval(
        self,
        F_t_inv: Callable[[float], float],
        alpha: float,
        eps: float = 1e-4,
    ) -> tuple[float, float]:
        if abs(float(alpha) - self.alpha_target) > 1e-12:
            raise ValueError(
                f"{self.display_name} was configured for alpha_target={self.alpha_target} "
                f"but called with alpha={alpha}."
            )
        qv = self._quantile_values(F_t_inv)
        L, U = self.model.interval(self.q_grid, qv)
        self._last_interval = (float(L), float(U))
        self._last_qv = qv
        return self._last_interval

    def update(self, step: Step) -> None:
        if self._last_interval is None or self._last_qv is None or step.y is None:
            return
        L, U = self._last_interval
        self.model.observe(float(step.y), L, U, self.q_grid, self._last_qv)

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.display_name} is an interval-only method.")

    @property
    def name(self) -> str:
        return self.display_name


def scalar_adaptive_methods(
    alpha_target: float = 0.10,
    baseline_path: str | Path | None = None,
    include_aci: bool = False,
) -> dict[str, OnlineMethod]:
    """Return scalar adaptive conformal baselines wrapped as OnlineMethod."""
    baselines = load_aci_baselines(baseline_path)
    gammas = (0.002, 0.004, 0.008, 0.016, 0.032, 0.064)
    methods: dict[str, OnlineMethod] = {}
    if include_aci:
        methods["ACI-PIT-External"] = ScalarAdaptiveCP(
            baselines.ACI,
            "ACI-PIT-External",
            alpha_target=alpha_target,
            kwargs={"gamma": 0.02},
        )
    methods.update(
        {
            "DtACI-PIT": ScalarAdaptiveCP(
                baselines.DtACI,
                "DtACI-PIT",
                alpha_target=alpha_target,
                kwargs={"gammas": gammas},
            ),
            "SF-OGD-PIT": ScalarAdaptiveCP(
                baselines.SFOGD,
                "SF-OGD-PIT",
                alpha_target=alpha_target,
                kwargs={"gamma": 0.08},
            ),
            "SAOCP-PIT": ScalarAdaptiveCP(
                baselines.SAOCP,
                "SAOCP-PIT",
                alpha_target=alpha_target,
            ),
            "EWA-ACI-PIT": ScalarAdaptiveCP(
                baselines.AgACI,
                "EWA-ACI-PIT",
                alpha_target=alpha_target,
                kwargs={"gammas": gammas, "eta": 0.15},
            ),
        }
    )
    return methods


class OCC(OnlineMethod):
    """Online Conformal Calibration with geometric forgetting + bin-isotonic.

    Two memory regimes:
      • fixed lambda (use ``lam=0.99`` etc.)
      • adaptive lambda from a split-window KS statistic (set
        ``adaptive=True``, configure ``D_window`` and ``phi`` per the paper)

    This implementation follows the manuscript experiments: it maintains
    exponentially-forgotten bin sufficient statistics and fits a weighted PAV
    map over bin centers. The adaptive version chooses lambda from a recent
    split-window KS statistic; large recent drift makes lambda smaller.
    """

    def __init__(
        self,
        lam: float = 0.995,
        c_B: float = 1.0,
        bins: int | None = 70,
        adaptive: bool = False,
        lam_min: float = 0.90,
        lam_max: float = 0.997,
        beta: float = 35.0,
        tau: float = 0.18,
        window: int = 120,
        max_history: int | None = None,
        max_buffer: int | None = None,
        record_lambda_path: bool = True,
        lambda_path_capacity: int | None = None,
    ):
        """Create a binned OCC state.

        ``max_history`` and ``max_buffer`` are retained as no-op compatibility
        arguments for old experiment scripts.  The implementation no longer
        stores a PIT history: the rank target is computed from exponentially
        weighted bin masses, and the adaptive KS rule stores only its latest
        ``window`` PIT values.  Thus the OCC state is O(B + W) memory and each
        update is O(B + W) (O(B) for fixed memory).
        """
        numeric_parameters = {
            "lam": lam,
            "lam_min": lam_min,
            "lam_max": lam_max,
            "beta": beta,
            "tau": tau,
        }
        if not all(np.isfinite(float(value)) for value in numeric_parameters.values()):
            raise ValueError("OCC numeric parameters must be finite.")
        if not 0.0 < float(lam) <= 1.0:
            raise ValueError("lam must lie in (0, 1].")
        if not 0.0 < float(lam_min) <= float(lam_max) <= 1.0:
            raise ValueError("Require 0 < lam_min <= lam_max <= 1.")
        if float(beta) < 0.0:
            raise ValueError("beta must be non-negative.")
        if not 0.0 <= float(tau) <= 1.0:
            raise ValueError("tau must lie in [0, 1].")
        if adaptive and (int(window) < 4 or int(window) % 2):
            raise ValueError("window must be an even integer at least 4.")
        if bins is not None and int(bins) < 4:
            raise ValueError("bins must be at least 4 when specified.")
        self.lam = float(lam)
        self.c_B = c_B
        self.bins = bins
        self.adaptive = adaptive
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.beta = beta
        self.tau = tau
        self.window = int(window)
        # Compatibility metadata only; neither parameter controls the state.
        self.max_history = max_history
        self.max_buffer = max_buffer
        self._recent: deque[float] = deque(maxlen=self.window if adaptive else 1)
        self._t = 0
        self._g_cache: Callable[[np.ndarray], np.ndarray] | None = None
        self._knots_x = np.array([0.0, 1.0])
        self._knots_y = np.array([0.0, 1.0])
        B = self.B
        self._centers = (np.arange(B) + 0.5) / B
        self._w = np.zeros(B, dtype=float)
        self._s = np.zeros(B, dtype=float)
        path_capacity = lambda_path_capacity
        if path_capacity is None:
            path_capacity = self.window if adaptive else 1
        self.lambda_path = deque(maxlen=max(1, int(path_capacity))) if record_lambda_path else None

    @property
    def n_eff(self) -> float:
        lam = self.lam_max if self.adaptive else self.lam
        return (1.0 + lam) / max(1e-6, 1.0 - lam)

    @property
    def B(self) -> int:
        if self.bins is not None:
            return max(4, int(self.bins))
        return max(4, int(round(self.c_B * self.n_eff ** (1.0 / 3.0))))

    def _lambda(self, u: float) -> float:
        if not self.adaptive:
            return float(self.lam)
        vals = np.asarray(list(self._recent) + [u], dtype=float)
        vals = vals[-self.window :]
        if vals.size < self.window:
            return float(self.lam_max)
        half = self.window // 2
        d = float(ks_2samp(vals[:half], vals[half:], method="asymp").statistic)
        lam = self.lam_min + (self.lam_max - self.lam_min) / (
            1.0 + math.exp(self.beta * (d - self.tau))
        )
        return float(np.clip(lam, self.lam_min, self.lam_max))

    def _rank(self, u: float) -> float:
        """Return the binned, forgetting-weighted empirical rank of ``u``.

        The bin mass before the current update represents weights
        ``prod_{r=s+1}^{t-1} lambda_r`` for all past observations.  Within the
        current bin we use linear interpolation.  Deterministically, its error
        against an exact weighted empirical rank is at most the current bin's
        normalized empirical mass; converting this to O(1/B) requires a
        separate occupancy argument.  No individual PIT values are stored.
        In particular, no current ``lambda_t`` is raised to historical ages;
        adaptive forgetting is handled solely by the recursive mass update.
        """
        total = float(np.sum(self._w))
        if total <= 1e-12:
            return float(np.clip(u, 0.0, 1.0))
        u = float(np.clip(u, 0.0, 1.0))
        position = min(np.nextafter(float(self.B), 0.0), u * self.B)
        b = min(self.B - 1, int(position))
        within = position - b
        prefix = float(np.sum(self._w[:b]))
        return float(np.clip((prefix + within * self._w[b]) / total, 0.0, 1.0))

    def _refit(self) -> None:
        """Rebuild the calibration map from current sufficient statistics."""
        if np.sum(self._w > 1e-10) < 2:
            self._knots_x = np.array([0.0, 1.0])
            self._knots_y = np.array([0.0, 1.0])
            self._g_cache = lambda u: np.clip(np.atleast_1d(u), 0.0, 1.0)
            return

        seen = self._w > 1e-10
        x = self._centers[seen]
        raw = self._s[seen] / self._w[seen]

        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        # A small floor keeps empty bins from creating arbitrary jumps while
        # letting observed bins dominate the fit.
        iso.fit(x, raw, sample_weight=self._w[seen])
        fitted = iso.predict(x)
        knots_x = np.concatenate([[0.0], x, [1.0]])
        knots_y = np.concatenate([[0.0], fitted, [1.0]])
        knots_y = np.maximum.accumulate(knots_y)  # guarantee monotone
        self._knots_x = knots_x
        self._knots_y = knots_y

        def g(u_grid, _kx=knots_x, _ky=knots_y):
            u_arr = np.atleast_1d(u_grid)
            return np.interp(u_arr, _kx, _ky)

        self._g_cache = g

    def update(self, step: Step) -> None:
        u = float(np.clip(step.u, 0.0, 1.0))
        lam = self._lambda(u)
        v = self._rank(u)
        self._w *= lam
        self._s *= lam
        b = min(self.B - 1, int(u * self.B))
        self._w[b] += 1.0
        self._s[b] += v
        self._recent.append(u)
        self._t = step.t
        if self.lambda_path is not None:
            self.lambda_path.append(lam)
        self._g_cache = None  # invalidate cache; refit on next predict

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        if self._g_cache is None:
            self._refit()
        return self._g_cache(u_grid)

    def _pit_knots(self):
        if self._g_cache is None:
            self._refit()
        return self._knots_x, self._knots_y, "linear"

    def _inverse_g(self, target: float) -> float:
        if self._g_cache is None:
            self._refit()
        return inverse_cdf(self._knots_x, self._knots_y, target)

    def predict_interval(self, F_t_inv, alpha: float, eps: float = 1e-4):
        del eps
        if not np.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        lower = self._inverse_g(float(alpha) / 2.0)
        upper = self._inverse_g(1.0 - float(alpha) / 2.0)
        return F_t_inv(lower), F_t_inv(upper)


class ExactRankOCC(OCC):
    """History-storing OCC reference used only for validation experiments.

    Unlike :class:`OCC`, this class computes the online rank from every stored
    PIT value with recursively updated product weights.  It is O(t) memory and
    O(t) rank work, so it must not be presented as the deployable fixed-memory
    algorithm.  Its purpose is to measure the approximation introduced by the
    binned rank sketch under exactly the same PAV and forgetting rule.
    """

    def __init__(self, *args, reference_capacity: int | None = None, **kwargs):
        if reference_capacity is not None and int(reference_capacity) < 1:
            raise ValueError("reference_capacity must be positive when specified.")
        super().__init__(*args, **kwargs)
        self.reference_capacity = reference_capacity
        self._reference_u: deque[float] = deque(maxlen=reference_capacity)
        self._reference_w: deque[float] = deque(maxlen=reference_capacity)

    def _exact_rank(self, u: float) -> float:
        if not self._reference_u:
            return float(np.clip(u, 0.0, 1.0))
        values = np.asarray(self._reference_u, dtype=float)
        weights = np.asarray(self._reference_w, dtype=float)
        return float(np.sum(weights * (values <= u)) / np.sum(weights))

    def update(self, step: Step) -> None:
        u = float(np.clip(step.u, 0.0, 1.0))
        lam = self._lambda(u)
        v = self._exact_rank(u)

        self._w *= lam
        self._s *= lam
        b = min(self.B - 1, int(u * self.B))
        self._w[b] += 1.0
        self._s[b] += v

        decayed = [lam * weight for weight in self._reference_w]
        self._reference_w.clear()
        self._reference_w.extend(decayed)
        self._reference_u.append(u)
        self._reference_w.append(1.0)

        self._recent.append(u)
        self._t = step.t
        if self.lambda_path is not None:
            self.lambda_path.append(lam)
        self._g_cache = None

    @property
    def name(self) -> str:
        return "OCC-ExactRank-reference"


# =============================================================================
# Static-PIT baseline (fit isotonic once on warmup, never update)
# =============================================================================


class StaticPIT(OnlineMethod):
    """Frozen, endpoint-anchored linear empirical PIT map; ties share ranks.

    Interior knots use the empirical CDF; boundary knots are exactly (0,0)
    and (1,1). Floating-point boundary PITs are smoothed into the edge segments.
    """
    def __init__(self):
        self._knots_x = np.array([0., 1.])
        self._knots_y = np.array([0., 1.])

    def warmup(self, warmup_steps):
        from cdf_tools import empirical_knots
        x, y, _ = empirical_knots([s.u for s in warmup_steps])
        interior = (x > 0) & (x < 1)
        self._knots_x = np.r_[0., x[interior], 1.]
        self._knots_y = np.r_[0., y[interior], 1.]

    def update(self, step):
        pass

    def _pit_knots(self):
        return self._knots_x, self._knots_y, "linear"

    def predict_g(self, u_grid):
        return np.interp(np.atleast_1d(u_grid), self._knots_x, self._knots_y)


# =============================================================================
# Rolling PIT empirical CDF (custom full-CDF baseline)
# =============================================================================


class RollingPITECDF(OnlineMethod):
    """Rolling empirical PIT CDF.

    This is deliberately named as a custom PIT-scale baseline.  It is not the
    covariate-dependent isotonic distributional regression estimator of Henzi,
    Ziegel and Gneiting (2021), which requires a partial-order design.
    """

    def __init__(self, K: int = 500):
        self.K = K
        self._U: deque[float] = deque(maxlen=K)
        self._g_cache: Callable[[np.ndarray], np.ndarray] | None = None

    def update(self, step: Step) -> None:
        self._U.append(float(step.u))
        self._g_cache = None

    def _refit(self):
        from cdf_tools import empirical_knots
        self._knots = empirical_knots(list(self._U), None)
        x, y, kind = self._knots
        def g(u_grid):
            u = np.atleast_1d(u_grid)
            if kind == "linear":
                return np.interp(u, x, y)
            idx = np.searchsorted(x, u, side="right")-1
            return np.where(idx >= 0, y[np.maximum(idx, 0)], 0.)
        self._g_cache = g

    def _pit_knots(self):
        if self._g_cache is None:
            self._refit()
        return self._knots

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        if self._g_cache is None:
            self._refit()
        return self._g_cache(u_grid)

    @property
    def name(self) -> str:
        return "Rolling-PIT-ECDF"


# Backwards-compatible import name for old notebooks.  It is intentionally not
# used by the paper-facing method registry.
RollingIDR = RollingPITECDF


# =============================================================================
# DCP score baseline (Chernozhukov--Wuthrich--Zhu 2021)
# =============================================================================


class DCP(OnlineMethod):
    """Rolling online implementation of the central DCP score rule.

    DCP calibrates ``|U - 1/2|`` and returns the central PIT interval
    ``[1/2-q, 1/2+q]``.  It is an interval-only baseline, not a full PIT map.
    """

    interval_only = True

    def __init__(self, K: int = 500):
        self.K = K
        self._scores: deque[float] = deque(maxlen=K)
        self._last_interval: tuple[float, float] | None = None

    def update(self, step: Step) -> None:
        self._scores.append(abs(float(np.clip(step.u, 0.0, 1.0)) - 0.5))

    def predict_interval(
        self,
        F_t_inv: Callable[[float], float],
        alpha: float,
        eps: float = 1e-4,
    ) -> tuple[float, float]:
        if not self._scores:
            q = 0.5
        else:
            scores = np.sort(np.asarray(self._scores, dtype=float))
            n = scores.size
            rank = int(np.ceil((n + 1.0) * (1.0 - float(alpha))))
            q = 0.5 if rank > n else float(scores[max(rank, 1) - 1])
        lo = float(np.clip(0.5 - q, 0.0, 1.0))
        hi = float(np.clip(0.5 + q, 0.0, 1.0))
        self._last_interval = (float(F_t_inv(lo)), float(F_t_inv(hi)))
        return self._last_interval

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError("DCP is a score-based interval method, not a PIT map.")

    @property
    def name(self) -> str:
        return "DCP-score"


# =============================================================================
# Weighted PIT-ECDF (ablation: OCC without binning / isotonic)
# =============================================================================


class WeightedPITECDF(OnlineMethod):
    """PIT empirical CDF with geometric forgetting and a finite buffer.

    Useful as an ablation isolating what OCC's binning + isotonic adds beyond a
    pure weighted ECDF.  This custom baseline is not distributional conformal
    prediction and does not inherit DCP's validity results.
    """

    def __init__(self, lam: float = 0.995, max_buffer: int = 100_000):
        self.lam = lam
        self._U: deque[float] = deque(maxlen=max_buffer)
        self._g_cache: Callable[[np.ndarray], np.ndarray] | None = None

    def update(self, step: Step) -> None:
        self._U.append(float(step.u))
        self._g_cache = None

    def _refit(self):
        from cdf_tools import empirical_knots
        self._knots = empirical_knots(list(self._U), self.lam ** np.arange(len(self._U)-1, -1, -1))
        x, y, kind = self._knots
        def g(u_grid):
            u = np.atleast_1d(u_grid)
            if kind == "linear":
                return np.interp(u, x, y)
            idx = np.searchsorted(x, u, side="right")-1
            return np.where(idx >= 0, y[np.maximum(idx, 0)], 0.)
        self._g_cache = g

    def _pit_knots(self):
        if self._g_cache is None:
            self._refit()
        return self._knots

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        if self._g_cache is None:
            self._refit()
        return self._g_cache(u_grid)

    @property
    def name(self) -> str:
        return "Weighted-PIT-ECDF"


# =============================================================================
# Direct binned recalibration (DBR)
# =============================================================================


class DirectBinnedRecalibration(OnlineMethod):
    """Fixed-state histogram approximation to a weighted PIT empirical CDF.

    The method stores one exponentially discounted mass per equal-width PIT
    bin. Cumulative masses at bin edges define a monotone piecewise-linear CDF.
    For fixed ``bins`` (and fixed ``window`` in adaptive mode), its state does
    not grow with the stream horizon.
    """

    def __init__(
        self,
        lam: float = 0.985,
        bins: int = 140,
        adaptive: bool = False,
        lam_min: float = 0.95,
        lam_max: float = 0.997,
        beta: float = 16.0,
        tau: float = 0.18,
        window: int = 120,
        adapt_every: int = 1,
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
        self._cdf_edges = self._edges.copy()
        self._cache_valid = True

    def _select_lambda(self, current_u: float) -> None:
        if not self.adaptive:
            self._current_lam = self.lam
            return
        values = np.asarray([*self._recent, current_u], dtype=float)
        values = values[-self.window :]
        if values.size < self.window:
            self._current_lam = self.lam_max
            return
        if self._updates % self.adapt_every:
            return
        half = self.window // 2
        distance = float(
            ks_2samp(values[:half], values[half:], method="asymp").statistic
        )
        self._current_lam = self.lam_min + (
            self.lam_max - self.lam_min
        ) / (1.0 + np.exp(self.beta * (distance - self.tau)))

    def update(self, step: Step) -> None:
        u = float(np.clip(step.u, 0.0, 1.0))
        self._select_lambda(u)
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
            self._cdf_edges = anchored_cdf_values(np.concatenate(
                ([0.0], np.cumsum(self._mass, dtype=float) / total)
            ))
        self._cache_valid = True

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        if not self._cache_valid:
            self._refresh()
        values = np.asarray(u_grid, dtype=float)
        return np.interp(
            np.clip(values, 0.0, 1.0), self._edges, self._cdf_edges
        )

    def _pit_knots(self):
        if not self._cache_valid:
            self._refresh()
        return self._edges, self._cdf_edges, "linear"

    def _inverse_g(self, target: float) -> float:
        if not self._cache_valid:
            self._refresh()
        return inverse_cdf(self._edges, self._cdf_edges, target)

    def predict_interval(
        self,
        F_t_inv: Callable[[float], float],
        alpha: float,
        eps: float = 1e-4,
    ) -> tuple[float, float]:
        if not np.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        lower = self._inverse_g(alpha / 2.0)
        upper = self._inverse_g(1.0 - alpha / 2.0)
        return F_t_inv(lower), F_t_inv(upper)

    @property
    def name(self) -> str:
        return "DBR-Adaptive" if self.adaptive else "DBR-Fixed"


def pit_crps_piecewise_linear(
    edges: np.ndarray,
    cdf_edges: np.ndarray,
    outcome: float,
) -> float:
    """Exact CRPS of a piecewise-linear CDF on ``[0, 1]``.

    The outcome is a raw PIT value. Thus this is the proper score for a
    forecast of the next PIT distribution, rather than outcome-space CRPS.
    """
    x = np.asarray(edges, dtype=float)
    g = np.asarray(cdf_edges, dtype=float)
    if x.ndim != 1 or g.ndim != 1 or x.size != g.size or x.size < 2:
        raise ValueError("edges and cdf_edges must be one-dimensional and aligned.")
    if not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0.0):
        raise ValueError("edges must be finite and strictly increasing.")
    if not np.all(np.isfinite(g)):
        raise ValueError("cdf_edges must be finite.")

    u = float(np.clip(outcome, x[0], x[-1]))

    def segment_integral(width, left, right, target):
        a = left - target
        b = right - target
        return width * (a * a + a * b + b * b) / 3.0

    widths = np.diff(x)
    left_values = g[:-1]
    right_values = g[1:]
    index = int(np.clip(np.searchsorted(x, u, side="right") - 1, 0, x.size - 2))
    loss = float(
        np.sum(
            segment_integral(
                widths[:index], left_values[:index], right_values[:index], 0.0
            )
        )
    )
    loss += float(
        np.sum(
            segment_integral(
                widths[index + 1 :],
                left_values[index + 1 :],
                right_values[index + 1 :],
                1.0,
            )
        )
    )
    x0, x1 = float(x[index]), float(x[index + 1])
    g0, g1 = float(g[index]), float(g[index + 1])
    fraction = (u - x0) / (x1 - x0)
    gu = g0 + fraction * (g1 - g0)
    loss += float(segment_integral(u - x0, g0, gu, 0.0))
    loss += float(segment_integral(x1 - u, gu, g1, 1.0))
    return float(loss)


class AdaptiveMemoryDBR(OnlineMethod):
    """Fixed-state aggregation of DBR experts with different memory scales.

    At round ``t`` the issued PIT CDF is the convex combination of a finite
    library of fixed-forgetting DBR maps. After observing ``U_t``, expert
    weights are updated by exponential weighting under PIT-scale CRPS and a
    fixed-share transition. The current outcome therefore affects only
    round ``t + 1``. For fixed expert count and bin count, state is bounded
    independently of the stream horizon.
    """

    def __init__(
        self,
        lambdas: Sequence[float] = (0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0),
        bins: int = 140,
        eta: float = 0.5,
        share: float = 0.01,
        include_identity: bool = True,
    ) -> None:
        values = tuple(float(value) for value in lambdas)
        if not values:
            raise ValueError("lambdas must contain at least one memory scale.")
        if any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in values):
            raise ValueError("Every forgetting factor must lie in (0, 1].")
        if len(set(values)) != len(values):
            raise ValueError("Forgetting factors must be distinct.")
        if int(bins) < 4:
            raise ValueError("bins must be at least 4.")
        if not np.isfinite(float(eta)) or not 0.0 < float(eta) <= 0.5:
            raise ValueError("eta must lie in (0, 0.5] for the CRPS oracle bound.")
        if not np.isfinite(float(share)) or not 0.0 <= float(share) < 1.0:
            raise ValueError("share must lie in [0, 1).")

        self.lambdas = values
        self.bins = int(bins)
        self.eta = float(eta)
        self.share = float(share)
        self.include_identity = bool(include_identity)
        self._experts = [
            DirectBinnedRecalibration(lam=value, bins=self.bins)
            for value in self.lambdas
        ]
        self._edges = np.linspace(0.0, 1.0, self.bins + 1)
        self._expert_labels = tuple(f"lambda={value:g}" for value in values)
        if self.include_identity:
            self._expert_labels += ("identity",)
        self._weights = np.full(len(self._expert_labels), 1.0 / len(self._expert_labels))
        self._aggregate_cdf_edges = self._edges.copy()
        self._cache_valid = False
        self._updates = 0
        self.last_expert_losses = np.zeros(len(self._expert_labels), dtype=float)
        self.last_aggregate_loss = 0.0
        self.cumulative_expert_losses = np.zeros(len(self._expert_labels), dtype=float)
        self.cumulative_aggregate_loss = 0.0

    def _expert_edge_cdfs(self) -> np.ndarray:
        rows = [expert.predict_g(self._edges) for expert in self._experts]
        if self.include_identity:
            rows.append(self._edges.copy())
        return np.asarray(rows, dtype=float)

    def _refresh(self) -> None:
        expert_cdfs = self._expert_edge_cdfs()
        self._aggregate_cdf_edges = anchored_cdf_values(self._weights @ expert_cdfs)
        self._cache_valid = True

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        if not self._cache_valid:
            self._refresh()
        values = np.asarray(u_grid, dtype=float)
        return np.interp(
            np.clip(values, 0.0, 1.0), self._edges, self._aggregate_cdf_edges
        )

    def _pit_knots(self):
        if not self._cache_valid:
            self._refresh()
        return self._edges, self._aggregate_cdf_edges, "linear"

    def _inverse_g(self, target: float) -> float:
        if not self._cache_valid:
            self._refresh()
        return inverse_cdf(self._edges, self._aggregate_cdf_edges, target)

    def predict_interval(
        self,
        F_t_inv: Callable[[float], float],
        alpha: float,
        eps: float = 1e-4,
    ) -> tuple[float, float]:
        if not np.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        lower = self._inverse_g(alpha / 2.0)
        upper = self._inverse_g(1.0 - alpha / 2.0)
        return F_t_inv(lower), F_t_inv(upper)

    def update(self, step: Step) -> None:
        u = float(np.clip(step.u, 0.0, 1.0))
        expert_cdfs = self._expert_edge_cdfs()
        aggregate_cdf = anchored_cdf_values(self._weights @ expert_cdfs)
        expert_losses = np.asarray(
            [
                cdf_score(self._edges, expert_cdf, u)
                for expert_cdf in expert_cdfs
            ],
            dtype=float,
        )
        aggregate_loss = cdf_score(self._edges, aggregate_cdf, u)

        log_weights = np.log(np.maximum(self._weights, np.finfo(float).tiny))
        log_weights -= self.eta * expert_losses
        log_weights -= float(np.max(log_weights))
        posterior = np.exp(log_weights)
        posterior /= float(np.sum(posterior))
        self._weights = (1.0 - self.share) * posterior + self.share / posterior.size

        for expert in self._experts:
            expert.update(step)
        self.last_expert_losses = expert_losses
        self.last_aggregate_loss = aggregate_loss
        self.cumulative_expert_losses += expert_losses
        self.cumulative_aggregate_loss += aggregate_loss
        self._updates += 1
        self._cache_valid = False

    @property
    def expert_weights(self) -> dict[str, float]:
        return dict(zip(self._expert_labels, self._weights, strict=True))

    def switching_oracle_penalty(self, horizon: int, switches: int) -> float:
        """Fixed-share CRPS penalty for a path with ``switches`` changes."""
        T = int(horizon)
        S = int(switches)
        if T < 1 or S < 0 or S > T - 1:
            raise ValueError("Require horizon >= 1 and 0 <= switches < horizon.")
        K = self._weights.size
        if self.share == 0.0:
            return float(np.log(K) / self.eta) if S == 0 else float("inf")
        complexity = np.log(K)
        complexity += S * np.log(K / self.share)
        complexity += (T - 1 - S) * np.log(1.0 / (1.0 - self.share))
        return float(complexity / self.eta)

    @property
    def name(self) -> str:
        return "AM-DBR"


# Alias used by experiment scripts and saved method labels.
BinnedWeightedPITECDF = DirectBinnedRecalibration


# Backwards-compatible import name for old notebooks.  The paper-facing
# registry uses the explicit custom-baseline name above.
WeightedDCP = WeightedPITECDF


# =============================================================================
# ACI (Gibbs--Candes 2021), Adaptive Conformal Inference
# =============================================================================


class ACI(OnlineMethod):
    """Project-specific PIT-level adaptation using the ACI recurrence.

    State: a single scalar ``alpha_t`` ∈ (0,1). Update rule on observing
    miscoverage err_t ∈ {0, 1}:
        alpha_{t+1} = alpha_t + gamma * (alpha - err_t).
    Targets long-run miscoverage equal to alpha.

    ACI does not natively produce a recalibration map hat_g_t. We provide
    ``predict_interval`` directly. ``predict_g`` is not implemented.
    """

    def __init__(self, alpha_target: float = 0.1, gamma: float = 0.005):
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.alpha_t = alpha_target
        self._last_interval: tuple[float, float] | None = None

    def predict_interval(
        self,
        F_t_inv: Callable[[float], float],
        alpha: float,
        eps: float = 1e-4,
    ) -> tuple[float, float]:
        # ACI ignores the ``alpha`` argument because it adapts its own
        # internal alpha_t to hit the target self.alpha_target. We could allow
        # multiple parallel ACI instances per target; for the comparison,
        # instantiate one ACI per target alpha.
        if alpha != self.alpha_target:
            raise ValueError(
                f"ACI was configured for alpha_target={self.alpha_target} "
                f"but called with alpha={alpha}. Instantiate one ACI per target."
            )
        a = float(np.clip(self.alpha_t, eps, 1.0 - eps))
        L, U = F_t_inv(a / 2.0), F_t_inv(1.0 - a / 2.0)
        self._last_interval = (L, U)
        return L, U

    def update(self, step: Step) -> None:
        if self._last_interval is None or step.y is None:
            return
        L, U = self._last_interval
        err_t = 1.0 if (step.y < L or step.y > U) else 0.0
        self.alpha_t = float(
            np.clip(self.alpha_t + self.gamma * (self.alpha_target - err_t), 1e-4, 0.5)
        )

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ACI is an interval-only method.")

    @property
    def name(self) -> str:
        return "ACI-PIT"
