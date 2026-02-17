"""
demand_model.py
---------------
Configurable stochastic demand generators for the dual-source inventory
simulation.  Supports Poisson and Normal (clipped to non-negative) demand.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DemandModel:
    """Generates stochastic demand each period.

    Parameters
    ----------
    distribution : str
        ``"poisson"`` or ``"normal"``.
    mean : float
        Mean demand per period.
    std : float
        Standard deviation (only used for ``"normal"``).
    spike_prob : float
        Probability of a demand spike each period (0 = off).
    spike_multiplier : float
        Multiplier applied to the demand when a spike occurs.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    distribution: str = "poisson"
    mean: float = 20.0
    std: float = 5.0
    spike_prob: float = 0.0
    spike_multiplier: float = 3.0
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    def sample(self) -> float:
        """Return a single non-negative demand sample."""
        if self.distribution == "poisson":
            d = float(self._rng.poisson(self.mean))
        elif self.distribution == "normal":
            d = float(max(0.0, self._rng.normal(self.mean, self.std)))
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        # Demand-spike mode
        if self.spike_prob > 0 and self._rng.random() < self.spike_prob:
            d *= self.spike_multiplier

        return d

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-seed the internal RNG."""
        self._rng = np.random.default_rng(seed if seed is not None else self.seed)
