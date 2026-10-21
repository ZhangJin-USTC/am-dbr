"""Single source of truth for manuscript-facing experiment settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PaperConfig:
    model_type: str = "ridge"
    alpha: float = 0.10
    fixed_lam: float = 0.985
    adaptive_lam_min: float = 0.95
    adaptive_lam_max: float = 0.997
    adaptive_tau: float = 0.18
    adaptive_beta: float = 16.0
    adaptive_window: int = 120
    dbr_bins: int = 140
    bins: int = 140
    baseline_window: int = 300
    aci_gamma: float = 0.005
    weighted_history: int = 3000
    am_lambdas: tuple[float, ...] = (0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0)
    am_bins: int = 140
    am_eta: float = 0.5
    am_share: float = 0.01
    am_include_identity: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


PAPER_CONFIG = PaperConfig()
