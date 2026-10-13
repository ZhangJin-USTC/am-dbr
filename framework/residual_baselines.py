"""Residual-score online conformal baselines.

These implementations use the same absolute-residual task as the reference
implementations in ``salesforce/online_conformal``.  They are kept separate
from the project-specific PIT-level controllers in ``ACI_new_baselines.py`` so
the paper cannot accidentally attribute residual-method guarantees to a
different probability-level wrapper.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import struct
from typing import Callable, Sequence

import numpy as np
from scipy.special import logsumexp

from core import OnlineMethod, Step


def _center(step: Step) -> float:
    if step.F_t_inv is None:
        raise ValueError("Residual baselines require Step.F_t_inv.")
    return float(step.F_t_inv(0.5))


def _score(step: Step) -> float:
    if step.y is None:
        raise ValueError("Residual baselines require Step.y.")
    return abs(float(step.y) - _center(step))


def _inverted_cdf_quantile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, np.clip(q, 0.0, 1.0), method="inverted_cdf"))


def _weighted_inverted_cdf_quantile(
    values: Sequence[float],
    q: float,
    weights: Sequence[float],
) -> float:
    arr = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.shape != w.shape or np.any(w < 0.0) or not np.any(w > 0.0):
        raise ValueError("Weighted quantiles require matching non-negative weights.")
    order = np.argsort(arr, kind="stable")
    cumulative = np.cumsum(w[order])
    index = int(np.searchsorted(cumulative / cumulative[-1], np.clip(q, 0.0, 1.0)))
    return float(arr[order[min(index, arr.size - 1)]])


def _pinball_loss(y, yhat, q: float):
    return np.maximum(q * (y - yhat), (1.0 - q) * (yhat - y))


def _pinball_grad(y: float, yhat: float, q: float) -> float:
    return float(-q * (y > yhat) + (1.0 - q) * (y < yhat))


@dataclass
class _TreapNode:
    key: float
    priority: int
    count: int = 1
    size: int = 1
    left: "_TreapNode | None" = None
    right: "_TreapNode | None" = None


def _node_size(node: _TreapNode | None) -> int:
    return 0 if node is None else node.size


def _refresh_node(node: _TreapNode) -> None:
    node.size = node.count + _node_size(node.left) + _node_size(node.right)


def _float_priority(value: float) -> int:
    """Stable SplitMix64 priority for an exact order-statistic treap."""
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    bits = (bits + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    bits = ((bits ^ (bits >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    bits = ((bits ^ (bits >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return bits ^ (bits >> 31)


class _OrderStatisticTreap:
    """Exact multiset ranks and order statistics in expected logarithmic time."""

    def __init__(self) -> None:
        self.root: _TreapNode | None = None

    def __len__(self) -> int:
        return _node_size(self.root)

    @staticmethod
    def _rotate_left(node: _TreapNode) -> _TreapNode:
        pivot = node.right
        if pivot is None:
            return node
        node.right = pivot.left
        pivot.left = node
        _refresh_node(node)
        _refresh_node(pivot)
        return pivot

    @staticmethod
    def _rotate_right(node: _TreapNode) -> _TreapNode:
        pivot = node.left
        if pivot is None:
            return node
        node.left = pivot.right
        pivot.right = node
        _refresh_node(node)
        _refresh_node(pivot)
        return pivot

    def _insert(self, node: _TreapNode | None, value: float) -> _TreapNode:
        if node is None:
            return _TreapNode(value, _float_priority(value))
        if value == node.key:
            node.count += 1
        elif value < node.key:
            node.left = self._insert(node.left, value)
            if node.left is not None and node.left.priority < node.priority:
                node = self._rotate_right(node)
        else:
            node.right = self._insert(node.right, value)
            if node.right is not None and node.right.priority < node.priority:
                node = self._rotate_left(node)
        _refresh_node(node)
        return node

    def insert(self, value: float) -> None:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Residual scores must be finite.")
        self.root = self._insert(self.root, value)

    def count_less_than(self, value: float) -> int:
        total = 0
        node = self.root
        while node is not None:
            if value <= node.key:
                node = node.left
            else:
                total += _node_size(node.left) + node.count
                node = node.right
        return total

    def tail_fraction(self, value: float) -> float:
        if len(self) == 0:
            return 0.0
        return float((len(self) - self.count_less_than(value)) / len(self))

    def kth(self, index: int) -> float:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        node = self.root
        while node is not None:
            left_size = _node_size(node.left)
            if index < left_size:
                node = node.left
            elif index < left_size + node.count:
                return node.key
            else:
                index -= left_size + node.count
                node = node.right
        raise RuntimeError("Corrupt order-statistic tree.")

    def inverted_cdf_quantile(self, q: float) -> float:
        if len(self) == 0:
            return 0.0
        q = float(np.clip(q, 0.0, 1.0))
        index = max(0, int(np.ceil(q * len(self))) - 1)
        return self.kth(index)


class ResidualACI(OnlineMethod):
    """ACI recurrence applied to absolute residual conformity scores."""

    interval_only = True
    requires_prediction_before_update = True

    def __init__(self, alpha_target: float = 0.10, gamma: float = 0.005):
        self.alpha_target = float(alpha_target)
        self.gamma = float(gamma)
        self.alpha_t = float(alpha_target)
        self.scores: list[float] = []
        self._score_tree = _OrderStatisticTreap()
        self._last_interval: tuple[float, float] | None = None

    def set_alpha_target(self, alpha: float) -> None:
        self.alpha_target = float(alpha)
        self.alpha_t = float(alpha)
        self._last_interval = None

    def warmup(self, warmup_steps: Sequence[Step]) -> None:
        self.scores = [_score(step) for step in warmup_steps]
        self._score_tree = _OrderStatisticTreap()
        for score in self.scores:
            self._score_tree.insert(score)
        self.alpha_t = self.alpha_target

    def predict_interval(
        self,
        F_t_inv: Callable[[float], float],
        alpha: float,
        eps: float = 1e-4,
    ) -> tuple[float, float]:
        if abs(float(alpha) - self.alpha_target) > 1e-12:
            raise ValueError("ResidualACI must be instantiated separately for each alpha.")
        center = float(F_t_inv(0.5))
        radius = self._score_tree.inverted_cdf_quantile(1.0 - self.alpha_t)
        self._last_interval = (center - radius, center + radius)
        return self._last_interval

    def update(self, step: Step) -> None:
        if self._last_interval is not None and step.y is not None:
            lo, hi = self._last_interval
            err = float(not (lo <= float(step.y) <= hi))
            self.alpha_t = float(np.clip(
                self.alpha_t + self.gamma * (self.alpha_target - err),
                0.0,
                1.0,
            ))
        score = _score(step)
        self.scores.append(score)
        self._score_tree.insert(score)

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ResidualACI is interval-only.")

    @property
    def name(self) -> str:
        return "ACI-residual-projected"


class ResidualNExCP(OnlineMethod):
    """Non-exchangeable conformal interval with geometric residual weights.

    This follows the weighted-quantile rule in Salesforce's
    ``online_conformal.NExConformal``.  The default ``gamma=1-alpha/4`` is the
    reference implementation's choice.  ``max_history=None`` reproduces its
    unbounded residual history; a finite value is an explicitly truncated
    sensitivity variant and must be labelled as such in reports.
    """

    interval_only = True

    def __init__(
        self,
        alpha_target: float = 0.10,
        gamma: float | None = None,
        max_history: int | None = None,
    ):
        self.alpha_target = float(alpha_target)
        self._automatic_gamma = gamma is None
        self.gamma = float(1.0 - self.alpha_target / 4.0 if gamma is None else gamma)
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1].")
        if max_history is not None and int(max_history) < 1:
            raise ValueError("max_history must be positive when specified.")
        self.max_history = None if max_history is None else int(max_history)
        self.scores: list[float] | deque[float] = self._new_score_state()

    def _new_score_state(self) -> list[float] | deque[float]:
        return [] if self.max_history is None else deque(maxlen=self.max_history)

    def set_alpha_target(self, alpha: float) -> None:
        self.alpha_target = float(alpha)
        if self._automatic_gamma:
            self.gamma = 1.0 - self.alpha_target / 4.0

    def warmup(self, warmup_steps: Sequence[Step]) -> None:
        self.scores = self._new_score_state()
        self.scores.extend(_score(step) for step in warmup_steps)

    def predict_interval(
        self,
        F_t_inv: Callable[[float], float],
        alpha: float,
        eps: float = 1e-4,
    ) -> tuple[float, float]:
        del eps
        if abs(float(alpha) - self.alpha_target) > 1e-12:
            raise ValueError("ResidualNExCP must be instantiated separately for each alpha.")
        n = len(self.scores)
        weights = self.gamma ** np.arange(n - 1, -1, -1, dtype=float)
        radius = _weighted_inverted_cdf_quantile(self.scores, 1.0 - self.alpha_target, weights)
        center = float(F_t_inv(0.5))
        return center - radius, center + radius

    def update(self, step: Step) -> None:
        self.scores.append(_score(step))

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ResidualNExCP is interval-only.")

    @property
    def name(self) -> str:
        return "NExCP-residual-empirical" if self.max_history is None else f"NExCP-window-{self.max_history}"


class ResidualPID(OnlineMethod):
    """Nonnegative-radius projection of Conformal PID on absolute residuals.

    The recurrence is the no-scorecaster branch of Angelopoulos, Candès and
    Tibshirani's public implementation, followed after every update by a
    projection of the radius onto ``[0, infinity)``.  The projection makes
    the state and issued interval agree but means that this class is a named
    variant rather than an exact reproduction of the unprojected reference
    recurrence.  ``learning_rate=0.1``, ``Csat=5`` and ``KI=10`` are one
    pre-declared configuration from the authors' examples; no
    evaluation-window selection is performed here.  The class is
    interval-only and does not claim OCC's full-PIT-map semantics.
    """

    interval_only = True

    def __init__(
        self,
        alpha_target: float = 0.10,
        learning_rate: float = 0.10,
        saturation: float = 5.0,
        integral_gain: float = 10.0,
        burnin_window: int = 100,
    ):
        if learning_rate < 0.0 or saturation <= 0.0 or integral_gain < 0.0:
            raise ValueError("PID gains must be non-negative and saturation positive.")
        if int(burnin_window) < 1:
            raise ValueError("burnin_window must be positive.")
        self.alpha_target = float(alpha_target)
        self.learning_rate = float(learning_rate)
        self.saturation = float(saturation)
        self.integral_gain = float(integral_gain)
        self.burnin_window = int(burnin_window)
        self._scores: deque[float] = deque(maxlen=self.burnin_window)
        self._radius = 0.0
        self._q_state = 0.0
        self._error_sum = 0.0
        self._t = 0

    def set_alpha_target(self, alpha: float) -> None:
        self.alpha_target = float(alpha)

    def warmup(self, warmup_steps: Sequence[Step]) -> None:
        self._scores.clear()
        self._radius = 0.0
        self._q_state = 0.0
        self._error_sum = 0.0
        self._t = 0
        for step in warmup_steps:
            self._update_score(_score(step))

    def _integrator(self) -> float:
        if self.integral_gain == 0.0 or self._t <= 0:
            return 0.0
        argument = self._error_sum - self._t * self.alpha_target
        angle = argument * np.log(self._t + 1.0) / (self.saturation * (self._t + 1.0))
        if angle >= np.pi / 2.0:
            return float("inf")
        if angle <= -np.pi / 2.0:
            return float("-inf")
        return float(self.integral_gain * np.tan(angle))

    def _update_score(self, score: float) -> None:
        covered = bool(self._radius >= float(score))
        previous_scores = np.asarray(self._scores, dtype=float)
        if previous_scores.size and self.learning_rate > 0.0:
            scale = float(previous_scores.max() - previous_scores.min())
        else:
            scale = 1.0
        lr_t = self.learning_rate * scale
        gradient = self.alpha_target if covered else -(1.0 - self.alpha_target)
        self._q_state -= lr_t * gradient
        # The public reference integrates the errors available before the
        # current score and uses the resulting q for the next round.
        self._radius = self._q_state + self._integrator()
        self._error_sum += float(not covered)
        self._scores.append(float(score))
        self._t += 1
        self._radius = max(0.0, float(self._radius))

    def predict_interval(
        self,
        F_t_inv: Callable[[float], float],
        alpha: float,
        eps: float = 1e-4,
    ) -> tuple[float, float]:
        del eps
        if abs(float(alpha) - self.alpha_target) > 1e-12:
            raise ValueError("ResidualPID must be instantiated separately for each alpha.")
        radius = float(self._radius)
        center = float(F_t_inv(0.5))
        return center - radius, center + radius

    def update(self, step: Step) -> None:
        self._update_score(_score(step))

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ResidualPID is interval-only.")

    @property
    def name(self) -> str:
        return "PID-residual-projected"


class ResidualFACI(OnlineMethod):
    """FACI/DtACI quantile aggregation on absolute residual scores.

    The recurrence follows ``salesforce/online_conformal/faci.py``: a fixed
    grid of ACI step sizes, exponential aggregation with fixed sharing, and
    the empirical residual-tail probability beta.
    """

    interval_only = True

    def __init__(
        self,
        alpha_target: float = 0.10,
        gammas: Sequence[float] = tuple(0.001 * 2**k for k in range(8)),
        I: int = 100,
    ):
        self.alpha_target = 1.0 - (1.0 - float(alpha_target))
        self.gammas = np.asarray(gammas, dtype=float)
        self.I = int(I)
        self.sigma = 1.0 / (2.0 * self.I)
        self.alphas = np.full(self.gammas.size, self.alpha_target, dtype=float)
        self.log_w = np.zeros(self.gammas.size, dtype=float)
        self.scores = _OrderStatisticTreap()

    @property
    def eta(self) -> float:
        alpha = self.alpha_target
        denom = ((1.0 - alpha) ** 2 * alpha**3 + alpha**2 * (1.0 - alpha) ** 3) / 3.0
        return float(np.sqrt(3.0 / self.I) * np.sqrt((np.log(self.I * self.gammas.size) + 2.0) / denom))

    def set_alpha_target(self, alpha: float) -> None:
        self.alpha_target = 1.0 - (1.0 - float(alpha))
        self.alphas = np.full(self.gammas.size, self.alpha_target, dtype=float)
        self.log_w = np.zeros(self.gammas.size, dtype=float)
        self.scores = _OrderStatisticTreap()

    def _weights(self) -> np.ndarray:
        return np.exp(self.log_w - logsumexp(self.log_w))

    def _update_score(self, score: float) -> None:
        if len(self.scores) > int(np.floor(1.0 / self.alpha_target)):
            beta = self.scores.tail_fraction(score)
            losses = _pinball_loss(beta, self.alphas, self.alpha_target)
            wbar = self.log_w - self.eta * losses
            shared = logsumexp(wbar)
            self.log_w = logsumexp(
                np.vstack((wbar, np.full(self.gammas.size, shared))),
                b=np.vstack((np.full(self.gammas.size, 1.0 - self.sigma), np.full(self.gammas.size, self.sigma / self.gammas.size))),
                axis=0,
            )
            self.log_w -= logsumexp(self.log_w)
            err = self.alphas > beta
            self.alphas = np.clip(
                self.alphas + self.gammas * (self.alpha_target - err.astype(float)),
                0.0,
                1.0,
            )
        self.scores.insert(float(score))

    def warmup(self, warmup_steps: Sequence[Step]) -> None:
        self.scores = _OrderStatisticTreap()
        for step in warmup_steps:
            self._update_score(_score(step))

    def predict_interval(
        self,
        F_t_inv: Callable[[float], float],
        alpha: float,
        eps: float = 1e-4,
    ) -> tuple[float, float]:
        if abs(float(alpha) - self.alpha_target) > 1e-12:
            raise ValueError("ResidualFACI must be instantiated separately for each alpha.")
        alpha_bar = float(np.dot(self._weights(), self.alphas))
        radius = self.scores.inverted_cdf_quantile(1.0 - alpha_bar)
        center = float(F_t_inv(0.5))
        return center - radius, center + radius

    def update(self, step: Step) -> None:
        self._update_score(_score(step))

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ResidualFACI is interval-only.")

    @property
    def name(self) -> str:
        return "FACI-residual"


class ResidualSFOGD(OnlineMethod):
    """Scale-free OGD on the absolute-residual radius."""

    interval_only = True

    def __init__(self, alpha_target: float = 0.10, max_scale: float | None = None):
        self.alpha_target = float(alpha_target)
        self.coverage = 1.0 - self.alpha_target
        self.max_scale = max_scale
        self.scale = 1.0 if max_scale is None else float(max_scale)
        self.delta = 0.0
        self.grad_norm = 0.0

    def set_alpha_target(self, alpha: float) -> None:
        self.alpha_target = float(alpha)
        self.coverage = 1.0 - self.alpha_target
        self.delta = 0.0
        self.grad_norm = 0.0

    def _update_score(self, score: float) -> None:
        grad = _pinball_grad(abs(float(score)), self.delta, self.coverage)
        self.grad_norm += grad**2
        if self.grad_norm > 0:
            self.delta = max(0.0, self.delta - self.scale / np.sqrt(3.0 * self.grad_norm) * grad)

    def warmup(self, warmup_steps: Sequence[Step]) -> None:
        scores = [_score(step) for step in warmup_steps]
        if self.max_scale is None:
            self.scale = 1.0 if not scores else max(max(scores) * np.sqrt(3.0), 1e-12)
        self.delta = 0.0
        self.grad_norm = 0.0
        for score in scores:
            self._update_score(score)

    def predict_interval(self, F_t_inv, alpha: float, eps: float = 1e-4):
        if abs(float(alpha) - self.alpha_target) > 1e-12:
            raise ValueError("ResidualSFOGD must be instantiated separately for each alpha.")
        center = float(F_t_inv(0.5))
        return center - self.delta, center + self.delta

    def update(self, step: Step) -> None:
        self._update_score(_score(step))

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ResidualSFOGD is interval-only.")

    @property
    def name(self) -> str:
        return "SF-OGD-residual"


@dataclass
class _SAExpert:
    start: int
    scale: float
    alpha: float
    yhat: float
    lifetime_mult: int = 8
    z: float = 0.0
    wz: float = 0.0
    age: int = 0
    grad_norm: float = 0.0

    def __post_init__(self):
        t = int(self.start)
        power = 0
        while t % 2 == 0:
            t //= 2
            power += 1
        self.lifetime = self.lifetime_mult * 2**power
        self.base_lr = self.scale / np.sqrt(3.0)

    @property
    def expired(self) -> bool:
        return self.age > self.lifetime

    @property
    def weight(self) -> float:
        return 0.0 if self.age == 0 else self.z / self.age * (1.0 + self.wz)

    def loss(self, score: float) -> float:
        return float(_pinball_loss(score, self.yhat, 1.0 - self.alpha))

    def update(self, score: float, meta_loss: float) -> None:
        old_weight = self.weight
        raw = (meta_loss - self.loss(score)) / self.scale / max(self.alpha, 1.0 - self.alpha)
        gradient = float(np.clip(raw, -1.0 * (old_weight > 0), 1.0))
        self.z += gradient
        self.wz += gradient * old_weight
        self.age += 1
        pinball_gradient = _pinball_grad(score, self.yhat, 1.0 - self.alpha)
        self.grad_norm += pinball_gradient**2
        if self.grad_norm > 0:
            self.yhat = max(0.0, self.yhat - self.base_lr / np.sqrt(self.grad_norm) * pinball_gradient)


class ResidualSAOCP(OnlineMethod):
    """Strongly adaptive online conformal prediction on residual radii."""

    interval_only = True

    def __init__(
        self,
        alpha_target: float = 0.10,
        max_scale: float | None = None,
        lifetime: int = 8,
    ):
        self.alpha_target = float(alpha_target)
        self.coverage = 1.0 - self.alpha_target
        self.max_scale = max_scale
        self.scale = 1.0 if max_scale is None else float(max_scale)
        self.lifetime = int(lifetime)
        self.t = 1
        self.experts: dict[int, _SAExpert] = {}

    def set_alpha_target(self, alpha: float) -> None:
        self.alpha_target = float(alpha)
        self.coverage = 1.0 - self.alpha_target
        self.t = 1
        self.experts = {}

    def _radius(self) -> float:
        # Preserve upstream scalar accumulation order. Tiny rounding differences
        # at a zero coin-betting weight can change later clipping branches.
        prior = {t: 1 / (t**2 * (1 + np.floor(np.log2(t)))) for t in self.experts}
        total = sum(prior.values())
        prior = {t: v / total for t, v in prior.items()}
        weights = {t: prior[t]*max(0, e.weight) for t, e in self.experts.items()}
        total = sum(weights.values())
        weights = {t: v/total for t, v in weights.items()} if total > 0 else prior
        return float(sum(weights[t]*e.yhat for t, e in self.experts.items()))

    def _update_score(self, score: float) -> None:
        initial = self._radius()
        self.experts = {start: expert for start, expert in self.experts.items() if not expert.expired}
        self.experts[self.t] = _SAExpert(
            start=self.t,
            scale=self.scale,
            alpha=1.0-self.coverage,
            yhat=initial,
            lifetime_mult=self.lifetime,
        )
        aggregate = self._radius()
        meta_loss = float(_pinball_loss(score, aggregate, self.coverage))
        for expert in self.experts.values():
            expert.update(score, meta_loss)
        self.t += 1

    def warmup(self, warmup_steps: Sequence[Step]) -> None:
        scores = [_score(step) for step in warmup_steps]
        if self.max_scale is None:
            self.scale = 1.0 if not scores else max(max(scores) * np.sqrt(3.0), 1e-12)
        self.t = 1
        self.experts = {}
        for score in scores:
            self._update_score(score)

    def predict_interval(self, F_t_inv, alpha: float, eps: float = 1e-4):
        if abs(float(alpha) - self.alpha_target) > 1e-12:
            raise ValueError("ResidualSAOCP must be instantiated separately for each alpha.")
        center = float(F_t_inv(0.5))
        radius = self._radius()
        return center - radius, center + radius

    def update(self, step: Step) -> None:
        self._update_score(_score(step))

    def predict_g(self, u_grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ResidualSAOCP is interval-only.")

    @property
    def name(self) -> str:
        return "SAOCP-residual"


def residual_baselines(
    alpha_target: float = 0.10,
    aci_gamma: float = 0.005,
) -> dict[str, OnlineMethod]:
    return {
        "ACI-residual-projected": ResidualACI(alpha_target=alpha_target, gamma=aci_gamma),
        "NExCP-residual-empirical": ResidualNExCP(alpha_target=alpha_target),
        "PID-residual-projected": ResidualPID(alpha_target=alpha_target),
        "FACI-residual": ResidualFACI(alpha_target=alpha_target),
        "SF-OGD-residual": ResidualSFOGD(alpha_target=alpha_target),
        "SAOCP-residual": ResidualSAOCP(alpha_target=alpha_target),
    }
