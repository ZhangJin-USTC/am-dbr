"""
ACI baselines for OCC paper, adapted from the AdaptiveConformal R package
(https://github.com/herbps10/AdaptiveConformal).

Includes:
    ACI    -- Adaptive Conformal Inference (Gibbs & Candes 2021)
    DtACI  -- Dynamically tuned ACI (Gibbs & Candes 2022)
    SFOGD  -- Scale-Free Online Gradient Descent (Orabona & Pal 2018; Bhatnagar 2023)
    SAOCP  -- Strongly Adaptive Online Conformal Prediction (Bhatnagar et al. 2023)
    AgACI  -- Aggregated ACI (Zaffran et al. 2022) using independent bounds aggregation.

Convention used here:
    alpha_target = MISCOVERAGE level (e.g., alpha_target=0.1 -> 90% PI)
    target_cov   = 1 - alpha_target = 0.9
    For symmetric intervals: lo = F^{-1}(alpha_internal/2), hi = F^{-1}(1 - alpha_internal/2)
    where alpha_internal is the time-varying miscoverage estimate.
"""

import numpy as np


# -------------------- shared helpers --------------------

def _as_quantile_arrays(qs, qv):
    qs = np.asarray(qs, dtype=float).reshape(-1)
    qv = np.asarray(qv, dtype=float).reshape(-1)
    if qs.size != qv.size:
        raise ValueError("qs and qv must have the same length.")
    if qs.size < 2:
        raise ValueError("qs and qv must contain at least two points.")
    order = np.argsort(qs)
    return qs[order], qv[order]


def inv_cdf_at(qs, qv, alpha):
    """Piecewise-linear inverse CDF with linear tail extrapolation."""
    qs, qv = _as_quantile_arrays(qs, qv)
    if alpha <= qs[0]:
        slope = (qv[1] - qv[0]) / max(qs[1] - qs[0], 1e-9)
        return float(qv[0] + slope * (alpha - qs[0]))
    if alpha >= qs[-1]:
        slope = (qv[-1] - qv[-2]) / max(qs[-1] - qs[-2], 1e-9)
        return float(qv[-1] + slope * (alpha - qs[-1]))
    return float(np.interp(alpha, qs, qv))


def _cdf_at_value(qs, qv, y):
    """Approximate F(y) by inverting the supplied predictive quantile curve."""
    qs, qv = _as_quantile_arrays(qs, qv)
    order = np.argsort(qv)
    x = qv[order]
    probs = qs[order]
    keep = np.concatenate(([True], np.diff(x) > 1e-12))
    x = x[keep]
    probs = probs[keep]
    if x.size < 2:
        return float(np.clip(probs[0], 1e-4, 1.0 - 1e-4))
    return float(np.clip(np.interp(y, x, probs), 1e-4, 1.0 - 1e-4))


def _interval_from_alpha(qs, qv, alpha_internal):
    """Symmetric interval at miscoverage level alpha_internal."""
    a = float(np.clip(alpha_internal, 1e-3, 1.0 - 1e-3))
    return inv_cdf_at(qs, qv, a / 2.0), inv_cdf_at(qs, qv, 1.0 - a / 2.0)


def _pinball_loss(theta_cov, beta_cov, alpha_target):
    """
    R package's loss_dtaci(alpha=target_cov, beta, theta).
    theta_cov, beta_cov are coverage levels in [0,1].
    """
    diff = beta_cov - theta_cov
    return (1.0 - alpha_target) * diff - min(0.0, diff)


# -------------------- ACI (Gibbs & Candes 2021) --------------------

class ACI:
    """Adaptive Conformal Inference."""
    def __init__(self, alpha_target=0.1, gamma=0.01):
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.alpha_internal = alpha_target

    def interval(self, qs, qv):
        return _interval_from_alpha(qs, qv, self.alpha_internal)

    def observe(self, y, lo, hi, qs=None, qv=None):
        covered = 1.0 if (lo <= y <= hi) else 0.0
        # Equivalent to R's update: alpha_internal decreases (wider interval) if covered == 0
        err = 1.0 - covered
        self.alpha_internal = float(np.clip(
            self.alpha_internal + self.gamma * (self.alpha_target - err),
            1e-3, 1.0 - 1e-3
        ))
        return covered


# -------------------- DtACI (Gibbs & Candes 2022) --------------------

class DtACI:
    """Dynamically tuned ACI."""
    def __init__(self, alpha_target=0.1,
                 gammas=(0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128),
                 I=100, sigma=None, eta=None):
        self.alpha_target = alpha_target
        self.gammas = np.array(gammas, dtype=float)
        self.K = len(gammas)
        self.experts = [ACI(alpha_target, g) for g in self.gammas]
        self.weights = np.full(self.K, 1.0 / self.K)
        self.I = I
        self.sigma = sigma if sigma is not None else 1.0 / (2.0 * I)
        
        if eta is None:
            a = 1.0 - alpha_target
            denom = a ** 2 * (1.0 - a) ** 3 + (1.0 - a) ** 2 * a ** 3
            eta = float(np.sqrt(3.0 / I) * np.sqrt((np.log(self.K * I) + 2.0) / denom))
        self.eta = eta

    def _aggregate_alpha(self):
        alphas = np.array([e.alpha_internal for e in self.experts])
        p = self.weights / self.weights.sum()
        return float(np.clip(np.sum(p * alphas), 1e-3, 1.0 - 1e-3))

    def interval(self, qs, qv):
        a = self._aggregate_alpha()
        return _interval_from_alpha(qs, qv, a)

    def observe(self, y, lo, hi, qs, qv):
        U = _cdf_at_value(qs, qv, y)
        alpha_crit = 2.0 * min(U, 1.0 - U)

        # Each candidate is updated from the interval issued before observing
        # y.  The aggregation loss then uses the candidate states at t+1,
        # matching the update order in the reference DtACI implementation.
        expert_intervals = [e.interval(qs, qv) for e in self.experts]
        for e, (expert_lo, expert_hi) in zip(self.experts, expert_intervals):
            e.observe(y, expert_lo, expert_hi, qs, qv)

        theta_covs = np.array([1.0 - e.alpha_internal for e in self.experts])
        beta_cov = alpha_crit
        losses = np.array([_pinball_loss(tc, beta_cov, self.alpha_target) for tc in theta_covs])

        weights_bar = self.weights * np.exp(-self.eta * losses)
        W_bar = weights_bar.sum()
        if W_bar > 0:
            weights_star = (1.0 - self.sigma) * weights_bar + W_bar * self.sigma / self.K
            self.weights = weights_star / weights_star.sum()


# -------------------- SF-OGD --------------------

class SFOGD:
    """Scale-Free Online Gradient Descent."""
    def __init__(self, alpha_target=0.1, gamma=0.01, theta0=None):
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.theta_cov = float(theta0) if theta0 is not None else (1.0 - alpha_target)
        self.cum_grad_sq = 0.0

    @property
    def alpha_internal(self):
        return float(np.clip(1.0 - self.theta_cov, 1e-3, 1.0 - 1e-3))

    def interval(self, qs, qv):
        return _interval_from_alpha(qs, qv, self.alpha_internal)

    def observe(self, y, lo, hi, qs=None, qv=None):
        # 严格遵守 R 包公式: gradient = covered - target_coverage
        covered = 1.0 if (lo <= y <= hi) else 0.0
        target_cov = 1.0 - self.alpha_target
        gradient = covered - target_cov
        
        self.cum_grad_sq += gradient * gradient
        denom = np.sqrt(max(self.cum_grad_sq, 1e-12))
        
        # 梯度为负（未覆盖）时，减去负数导致 theta_cov 增加 -> alpha_internal 减小 -> 区间变宽
        self.theta_cov = float(max(0.0, self.theta_cov - self.gamma * gradient / denom))


# -------------------- SAOCP --------------------

class _SAOCPExpert:
    """封装了完整 SFOGD 的 SAOCP 局部专家"""
    def __init__(self, alpha_target, gamma, theta0, start_time, lifetime):
        self.sfogd = SFOGD(alpha_target=alpha_target, gamma=gamma, theta0=theta0)
        self.start_time = start_time
        self.lifetime = lifetime
        self.weight = 0.0
        self.sum_g = 0.0
        self.sum_weights_g = 0.0


class SAOCP:
    """Strongly Adaptive Online Conformal Prediction."""
    def __init__(self, alpha_target=0.1, D=1.0, lifetime_mult=8, max_experts=200):
        self.alpha_target = alpha_target
        self.D = D
        self.gamma = D / np.sqrt(3.0)
        self.g_mult = lifetime_mult
        self.max_experts = max_experts
        self.experts = []  
        self.t = 0  
        self._last_alpha_agg = alpha_target
        self._pending_interval = False

    def _lifetime(self, t):
        if t == 0: return self.g_mult
        n = 0
        x = t
        while x % 2 == 0:
            n += 1
            x //= 2
        return self.g_mult * (2 ** n)

    def _aggregate_theta_cov(self):
        if not self.experts:
            return 1.0 - self.alpha_target
        
        # 明确使用专家的绝对启动时间戳 (Absolute start time)
        ts = np.array([e.start_time for e in self.experts], dtype=float)
        ts_safe = np.maximum(ts, 1.0)
        
        prior = ts_safe ** (-2) * (1.0 + np.floor(np.log2(ts_safe))) ** (-1)
        prior = prior / prior.sum()
        
        weights_pos = np.array([max(e.weight, 0.0) for e in self.experts])
        phat = prior * weights_pos
        
        if phat.sum() > 0:
            phat = phat / phat.sum()
        else:
            phat = prior
            
        thetas = np.array([e.sfogd.theta_cov for e in self.experts])
        return float(np.clip(np.sum(phat * thetas), 1e-3, 1.0 - 1e-3))

    def _spawn_expert(self):
        init_theta = self._aggregate_theta_cov() if self.experts else 1.0 - self.alpha_target
        self.t += 1
        e = _SAOCPExpert(
            alpha_target=self.alpha_target,
            gamma=self.gamma,
            theta0=init_theta,
            start_time=self.t,
            lifetime=self._lifetime(self.t)
        )
        self.experts.append(e)
        self.experts = [ex for ex in self.experts if ex.start_time + ex.lifetime >= self.t]
        if len(self.experts) > self.max_experts:
            self.experts = self.experts[-self.max_experts:]

    def _ensure_pending_expert(self):
        if not self._pending_interval:
            self._spawn_expert()
            self._pending_interval = True

    def interval(self, qs, qv):
        self._ensure_pending_expert()
        theta_cov_agg = self._aggregate_theta_cov()
        self._last_alpha_agg = float(np.clip(1.0 - theta_cov_agg, 1e-3, 1.0 - 1e-3))
        return _interval_from_alpha(qs, qv, self._last_alpha_agg)

    def observe(self, y, lo, hi, qs, qv):
        self._ensure_pending_expert()
        U = _cdf_at_value(qs, qv, y)
        alpha_crit = 2.0 * min(U, 1.0 - U)
        beta_cov = alpha_crit
        theta_cov_agg = 1.0 - self._last_alpha_agg
        loss_agg = _pinball_loss(theta_cov_agg, beta_cov, self.alpha_target)

        for e in self.experts:
            theta_cov_i = e.sfogd.theta_cov
            loss_i = _pinball_loss(theta_cov_i, beta_cov, self.alpha_target)
            
            raw = (loss_agg - loss_i) / self.D
            g = float(np.clip(raw, -1.0, 1.0)) if e.weight > 0 else float(np.clip(raw, 0.0, 1.0))
            g = g / max(self.alpha_target, 1.0 - self.alpha_target)
            
            e.sum_g += g
            e.sum_weights_g += e.weight * g
            age = self.t - e.start_time + 1
            e.weight = (1.0 / age) * e.sum_g * (1.0 + e.sum_weights_g)
            
            # 每个专家独立执行内部的 SFOGD 更新
            expert_lo, expert_hi = e.sfogd.interval(qs, qv)
            e.sfogd.observe(y, expert_lo, expert_hi, qs, qv)

        self._pending_interval = False


# -------------------- AgACI --------------------

class AgACI:
    """Aggregated ACI (Independent bounds approach)."""
    def __init__(self, alpha_target=0.1,
                 gammas=(0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128),
                 eta=None):
        self.alpha_target = alpha_target
        self.gammas = np.array(gammas, dtype=float)
        self.K = len(gammas)
        self.experts = [ACI(alpha_target, g) for g in self.gammas]
        self.weights_lo = np.full(self.K, 1.0 / self.K)
        self.weights_hi = np.full(self.K, 1.0 / self.K)
        self.eta = float(eta) if eta is not None else 1.0
        self.tau_lo = alpha_target / 2.0
        self.tau_hi = 1.0 - alpha_target / 2.0

    def interval(self, qs, qv):
        los = np.array([e.interval(qs, qv)[0] for e in self.experts])
        his = np.array([e.interval(qs, qv)[1] for e in self.experts])
        p_lo = self.weights_lo / self.weights_lo.sum()
        p_hi = self.weights_hi / self.weights_hi.sum()
        return float(np.sum(p_lo * los)), float(np.sum(p_hi * his))

    def observe(self, y, lo, hi, qs=None, qv=None):
        if qs is None or qv is None:
            raise ValueError("AgACI.observe requires qs and qv to update expert intervals.")

        los = np.array([e.interval(qs, qv)[0] for e in self.experts])
        his = np.array([e.interval(qs, qv)[1] for e in self.experts])
        
        # 严格使用 Pinball Loss 分别评估上下界
        loss_lo = np.maximum(self.tau_lo * (y - los), (self.tau_lo - 1.0) * (y - los))
        loss_hi = np.maximum(self.tau_hi * (y - his), (self.tau_hi - 1.0) * (y - his))
        
        self.weights_lo *= np.exp(-self.eta * loss_lo)
        self.weights_hi *= np.exp(-self.eta * loss_hi)
        
        if self.weights_lo.sum() > 0: self.weights_lo /= self.weights_lo.sum()
        if self.weights_hi.sum() > 0: self.weights_hi /= self.weights_hi.sum()
        
        for e in self.experts:
            expert_lo, expert_hi = e.interval(qs, qv)
            e.observe(y, expert_lo, expert_hi, qs, qv)
