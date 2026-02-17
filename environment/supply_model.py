"""
supply_model.py
---------------
Models for supply-side uncertainty: random fill rates, lead times, and
supply disruptions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SupplyModel:
    """Captures fill-rate randomness and supply disruptions.

    Parameters
    ----------
    fill_rate_range : tuple[float, float]
        (min, max) uniform fill-rate applied to each order on arrival.
    disruption_prob : float
        Per-period probability that the supply is **fully** disrupted
        (fill rate forced to 0).  0 = no disruptions.
    jit_lead_time : int
        Fixed lead time for the JIT (fast) source, in periods.
    llt_lead_time : int
        Fixed lead time for the LLT (slow) source, in periods.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    fill_rate_range: tuple[float, float] = (0.8, 1.0)
    disruption_prob: float = 0.0
    jit_lead_time: int = 1
    llt_lead_time: int = 6
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    def realised_fill_rate(self) -> float:
        """Sample a realised fill rate for the current period."""
        if self.disruption_prob > 0 and self._rng.random() < self.disruption_prob:
            return 0.0  # full disruption
        lo, hi = self.fill_rate_range
        return float(self._rng.uniform(lo, hi))

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-seed the internal RNG."""
        self._rng = np.random.default_rng(seed if seed is not None else self.seed)
