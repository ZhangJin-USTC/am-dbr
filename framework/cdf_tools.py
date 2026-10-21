"""Exact inverse and integrated squared CDF score on [0, 1].

Knots describe either a continuous piecewise-linear CDF or a right-continuous
step CDF. No quantile grid or tail truncation is used.
"""
from __future__ import annotations
import numpy as np


def inverse_cdf(x, y, q, kind="linear"):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("CDF knots must be one-dimensional arrays of equal length (at least 2).")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
        raise ValueError("CDF knots must be finite with strictly increasing x values.")
    if np.any(np.diff(y) < 0) or np.any((y < 0) | (y > 1)):
        raise ValueError("CDF values must be nondecreasing and lie in [0, 1].")
    if kind not in ("linear", "step"):
        raise ValueError("CDF kind must be 'linear' or 'step'.")
    q = float(q)
    if not 0 <= q <= 1:
        raise ValueError("Quantile level must be in [0, 1].")
    i = min(int(np.searchsorted(y, q, side="left")), len(x) - 1)
    if i == 0 or kind == "step":
        return float(x[i])
    return float(x[i-1] + (q-y[i-1]) * (x[i]-x[i-1]) / (y[i]-y[i-1]))


def anchored_cdf_values(values):
    """Repair rounding only when constructing an anchored monotone CDF.

    Nonnegative bin masses and convex mixtures are nondecreasing in exact
    arithmetic.  Floating-point summation can nevertheless overshoot one by
    an ulp.  Clip and monotonise before setting the two exact anchors so the
    issued representation, its inverse and its score all use a valid CDF.
    This does not impose any positive slope floor.
    """
    y = np.clip(np.maximum.accumulate(np.asarray(values, dtype=float)), 0., 1.)
    y[0], y[-1] = 0., 1.
    return y


def cdf_score(x, y, outcome, kind="linear"):
    """Integral (G(z)-1{outcome<=z})^2 dz, split exactly at outcome."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    u = float(outcome)
    if not 0 <= u <= 1 or not np.isfinite(u):
        raise ValueError("PIT outcome must be finite and in [0, 1].")
    dx = np.diff(x)
    if kind == "step":
        below = np.clip(u-x[:-1], 0, dx)
        return float(np.sum(below*y[:-1]**2 + (dx-below)*(1-y[:-1])**2))
    slope = np.diff(y)/dx
    below = np.clip(u-x[:-1], 0, dx)
    # Primitive of (a+b*z)^2 on each segment, with a different indicator
    # above the outcome. This includes an outcome at either endpoint.
    primitive = lambda a, z: a*a*z + a*slope*z*z + slope*slope*z*z*z/3
    return float(np.sum(primitive(y[:-1], below) +
                        primitive(y[:-1]-1, dx)-primitive(y[:-1]-1, below)))


def empirical_knots(values, weights=None):
    values = np.asarray(values, float)
    if len(values) == 0:
        return np.array([0., 1.]), np.array([0., 1.]), "linear"
    if weights is None:
        weights = np.ones(len(values))
    weights = np.asarray(weights, float)
    if values.ndim != 1 or weights.ndim != 1 or values.size != weights.size:
        raise ValueError("values and weights must be one-dimensional arrays of equal length.")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("values and weights must be finite; weights must be non-negative.")
    x, inv = np.unique(values, return_inverse=True)
    mass = np.bincount(inv, weights=weights)
    total = float(mass.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must have a strictly positive total.")
    # ``cumsum / total`` may exceed one by an ulp for a valid stream.  The
    # explicit monotone repair keeps all downstream interpolants valid while
    # retaining every positive increment, however small.
    y = np.clip(np.maximum.accumulate(np.cumsum(mass, dtype=float) / total), 0., 1.)
    if x[0] > 0:
        x, y = np.r_[0., x], np.r_[0., y]
    if x[-1] < 1:
        x, y = np.r_[x, 1.], np.r_[y, 1.]
    y[-1] = 1.
    return x, y, "step"
