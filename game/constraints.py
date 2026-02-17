"""
constraints.py
--------------
Helper utilities for the *Game Phase* constraint toggles.
Provides a dataclass that holds user-configurable scenario parameters
and a helper function that patches an environment config dict accordingly.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ScenarioConstraints:
    """All tunable game parameters (set via the Streamlit sidebar)."""

    # Cost knobs
    jit_unit_cost: float = 8.0
    llt_unit_cost: float = 4.0
    holding_cost_per_unit: float = 1.0
    shortage_penalty_per_unit: float = 10.0

    # Lead-time / capacity
    llt_lead_time: int = 6
    storage_capacity: float = 200.0

    # Demand
    demand_mean: float = 20.0
    demand_std: float = 5.0
    demand_distribution: str = "poisson"

    # Disruption / spike toggles
    demand_spike_prob: float = 0.0
    demand_spike_multiplier: float = 3.0
    supply_disruption_prob: float = 0.0

    # Capacity mode
    hard_capacity: bool = True

    def to_env_overrides(self) -> dict:
        """Return a dict of keys that should override the base env config."""
        return {
            "jit_unit_cost": self.jit_unit_cost,
            "llt_unit_cost": self.llt_unit_cost,
            "holding_cost_per_unit": self.holding_cost_per_unit,
            "shortage_penalty_per_unit": self.shortage_penalty_per_unit,
            "llt_lead_time": self.llt_lead_time,
            "storage_capacity": self.storage_capacity,
            "demand_mean": self.demand_mean,
            "demand_std": self.demand_std,
            "demand_distribution": self.demand_distribution,
            "demand_spike_prob": self.demand_spike_prob,
            "demand_spike_multiplier": self.demand_spike_multiplier,
            "supply_disruption_prob": self.supply_disruption_prob,
            "hard_capacity": self.hard_capacity,
        }
