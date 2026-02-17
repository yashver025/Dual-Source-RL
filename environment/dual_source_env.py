"""
dual_source_env.py
------------------
Gymnasium-compatible environment for the **dual-source inventory control**
problem with **cost-aware observation** and **domain randomization**.

Observation (Box, 23-dim by default):
    [ inventory_level,                              # 1
      jit_pipeline_0 … jit_pipeline_(slots-1),      # obs_jit_slots  (1)
      llt_pipeline_0 … llt_pipeline_(slots-1),      # obs_llt_slots  (6)
      demand_t-1 … demand_t-W,                      # demand_history  (5)
      time_step,                                     # 1
      capacity_remaining,                            # 1
      prev_jit_order, prev_llt_order,                # 2
      --- scenario parameters (normalised) ---
      jit_unit_cost_norm,                            # 1
      llt_unit_cost_norm,                            # 1
      holding_cost_norm,                             # 1
      shortage_penalty_norm,                         # 1
      storage_capacity_norm,                         # 1
      llt_lead_time_norm ]                           # 1  → total = 23

Action (Box(2)):
    [ jit_order_qty,  llt_order_qty ]   both in [0, 1] → scaled to max_order

Domain randomization:
    When ``cfg["domain_randomization"]`` is True, ``reset()`` samples a
    fresh scenario (costs, capacity, lead time) from the configured ranges
    at the start of every episode, enabling the agent to learn a
    **conditional policy**  π(state, params) → action.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Optional

from environment.demand_model import DemandModel
from environment.supply_model import SupplyModel


class DualSourceEnv(gym.Env):
    """Dual-source inventory control environment (cost-aware)."""

    metadata = {"render_modes": ["human"]}

    # Default configuration ------------------------------------------------
    DEFAULT_CFG = dict(
        # costs
        unit_revenue=12.0,
        jit_unit_cost=8.0,
        llt_unit_cost=4.0,
        holding_cost_per_unit=1.0,
        shortage_penalty_per_unit=10.0,
        capacity_penalty_per_unit=5.0,
        # capacity
        storage_capacity=200.0,
        max_order=50.0,
        # episode
        episode_length=150,
        # demand
        demand_distribution="poisson",
        demand_mean=20.0,
        demand_std=5.0,
        demand_spike_prob=0.0,
        demand_spike_multiplier=3.0,
        # supply
        jit_lead_time=1,
        llt_lead_time=6,
        fill_rate_min=0.8,
        fill_rate_max=1.0,
        supply_disruption_prob=0.0,
        # constraint mode
        hard_capacity=True,
        # history window
        demand_history_len=5,
        # Fixed observation pipeline slots (must match training shape).
        obs_jit_slots=None,   # None → use jit_lead_time
        obs_llt_slots=None,   # None → use llt_lead_time
        # Domain randomization
        domain_randomization=False,
        dr_jit_cost=(2.0, 15.0),
        dr_llt_cost=(1.0, 12.0),
        dr_holding_cost=(0.2, 3.0),
        dr_shortage_penalty=(3.0, 25.0),
        dr_storage_capacity=(80.0, 400.0),
        dr_llt_lead_time=(2, 10),
        dr_demand_mean=(10.0, 40.0),
        # Normalisation constants
        norm_cost_max=25.0,
        norm_capacity_max=500.0,
        norm_lead_time_max=12.0,
        # seed
        seed=42,
    )

    # Number of extra scenario-parameter dimensions appended to obs
    N_SCENARIO_DIMS = 6

    def __init__(self, cfg: dict | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.cfg = {**self.DEFAULT_CFG, **(cfg or {}), **kwargs}

        # Aliases for readability
        self.jit_lt: int = int(self.cfg["jit_lead_time"])
        self.llt_lt: int = int(self.cfg["llt_lead_time"])
        self.max_order: float = float(self.cfg["max_order"])
        self.capacity: float = float(self.cfg["storage_capacity"])
        self.ep_len: int = int(self.cfg["episode_length"])
        self.hist_len: int = int(self.cfg["demand_history_len"])
        self.hard_cap: bool = bool(self.cfg["hard_capacity"])
        self.domain_rand: bool = bool(self.cfg.get("domain_randomization", False))

        # Fixed observation pipeline slots (for stable obs shape)
        self.obs_jit_slots: int = (
            int(self.cfg["obs_jit_slots"])
            if self.cfg.get("obs_jit_slots") is not None
            else self.jit_lt
        )
        self.obs_llt_slots: int = (
            int(self.cfg["obs_llt_slots"])
            if self.cfg.get("obs_llt_slots") is not None
            else self.llt_lt
        )

        # Normalisation constants
        self._norm_cost = float(self.cfg.get("norm_cost_max", 25.0))
        self._norm_cap = float(self.cfg.get("norm_capacity_max", 500.0))
        self._norm_lt = float(self.cfg.get("norm_lead_time_max", 12.0))

        # Current scenario params (may be randomised per episode)
        self._jit_cost = float(self.cfg["jit_unit_cost"])
        self._llt_cost = float(self.cfg["llt_unit_cost"])
        self._hold_cost = float(self.cfg["holding_cost_per_unit"])
        self._short_pen = float(self.cfg["shortage_penalty_per_unit"])
        self._cap_pen = float(self.cfg["capacity_penalty_per_unit"])
        self._revenue = float(self.cfg["unit_revenue"])

        # Sub-models
        self.demand_model = DemandModel(
            distribution=self.cfg["demand_distribution"],
            mean=self.cfg["demand_mean"],
            std=self.cfg["demand_std"],
            spike_prob=self.cfg["demand_spike_prob"],
            spike_multiplier=self.cfg["demand_spike_multiplier"],
            seed=self.cfg["seed"],
        )
        self.supply_model = SupplyModel(
            fill_rate_range=(self.cfg["fill_rate_min"], self.cfg["fill_rate_max"]),
            disruption_prob=self.cfg["supply_disruption_prob"],
            jit_lead_time=self.jit_lt,
            llt_lead_time=self.llt_lt,
            seed=self.cfg["seed"],
        )

        # RNG for domain randomization
        self._dr_rng = np.random.default_rng(self.cfg["seed"])

        # Observation & action spaces
        obs_dim = (
            1                       # inventory
            + self.obs_jit_slots    # JIT pipeline (fixed slots)
            + self.obs_llt_slots    # LLT pipeline (fixed slots)
            + self.hist_len         # demand history
            + 1                     # time step (normalised)
            + 1                     # capacity remaining
            + 2                     # previous actions
            + self.N_SCENARIO_DIMS  # scenario parameters
        )
        high = np.full(obs_dim, 1e4, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state (initialised in reset)
        self.inventory: float = 0.0
        self.jit_pipeline: list[float] = []
        self.llt_pipeline: list[float] = []
        self.demand_history: list[float] = []
        self.t: int = 0
        self.prev_action: np.ndarray = np.zeros(2, dtype=np.float32)

        # Logging accumulators
        self._episode_log: dict[str, list] = {}

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------
    def _randomise_scenario(self) -> None:
        """Sample a fresh scenario from the configured DR ranges."""
        rng = self._dr_rng
        cfg = self.cfg

        self._jit_cost = float(rng.uniform(*cfg["dr_jit_cost"]))
        self._llt_cost = float(rng.uniform(*cfg["dr_llt_cost"]))
        self._hold_cost = float(rng.uniform(*cfg["dr_holding_cost"]))
        self._short_pen = float(rng.uniform(*cfg["dr_shortage_penalty"]))
        self.capacity = float(rng.uniform(*cfg["dr_storage_capacity"]))

        lo_lt, hi_lt = cfg["dr_llt_lead_time"]
        new_llt_lt = int(rng.integers(lo_lt, hi_lt + 1))
        self.llt_lt = new_llt_lt

        new_demand_mean = float(rng.uniform(*cfg["dr_demand_mean"]))
        self.demand_model.mean = new_demand_mean
        self.cfg["demand_mean"] = new_demand_mean

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        s = seed if seed is not None else self.cfg["seed"]
        self.demand_model.reset(s)
        self.supply_model.reset(s)

        # Domain randomization: sample new scenario each episode
        if self.domain_rand:
            self._randomise_scenario()
        else:
            # Use fixed cfg values
            self._jit_cost = float(self.cfg["jit_unit_cost"])
            self._llt_cost = float(self.cfg["llt_unit_cost"])
            self._hold_cost = float(self.cfg["holding_cost_per_unit"])
            self._short_pen = float(self.cfg["shortage_penalty_per_unit"])
            self.capacity = float(self.cfg["storage_capacity"])
            self.llt_lt = int(self.cfg["llt_lead_time"])

        self.inventory = float(self.demand_model.mean * 2)
        self.jit_pipeline = [0.0] * self.jit_lt
        self.llt_pipeline = [0.0] * self.llt_lt
        self.demand_history = [self.demand_model.mean] * self.hist_len
        self.t = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.hard_cap = bool(self.cfg["hard_capacity"])

        self._episode_log = {
            "inventory": [],
            "demand": [],
            "jit_order": [],
            "llt_order": [],
            "reward": [],
            "stockout": [],
            "capacity_violation": [],
            "satisfied": [],
            "total_demand": [],
        }

        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1) Decode action → order quantities
        act = np.clip(action, 0.0, 1.0)
        jit_order = float(act[0] * self.max_order)
        llt_order = float(act[1] * self.max_order)

        # 2) Pipeline arrivals  (FIFO)
        jit_arriving = self.jit_pipeline.pop(0) if self.jit_pipeline else 0.0
        llt_arriving = self.llt_pipeline.pop(0) if self.llt_pipeline else 0.0

        # Apply fill-rate to arrivals
        fill = self.supply_model.realised_fill_rate()
        jit_arriving *= fill
        llt_arriving *= fill

        # 3) Add new orders to back of pipeline
        self.jit_pipeline.append(jit_order)
        self.llt_pipeline.append(llt_order)

        # 4) Inventory update: arrivals
        self.inventory += jit_arriving + llt_arriving

        # 5) Demand realisation
        demand = self.demand_model.sample()

        # 6) Satisfy demand
        satisfied = min(self.inventory, demand)
        shortage = demand - satisfied
        self.inventory -= satisfied

        # 7) Capacity check
        cap_violation = max(0.0, self.inventory - self.capacity)
        if self.hard_cap:
            self.inventory = min(self.inventory, self.capacity)

        # 8) Reward computation  (uses *current* scenario params)
        revenue = satisfied * self._revenue
        jit_cost = jit_order * self._jit_cost
        llt_cost = llt_order * self._llt_cost
        hold_cost = max(self.inventory, 0.0) * self._hold_cost
        short_cost = shortage * self._short_pen
        cap_cost = cap_violation * self._cap_pen

        raw_reward = revenue - jit_cost - llt_cost - hold_cost - short_cost - cap_cost
        reward = raw_reward / self.max_order

        # 9) Logging
        self._episode_log["inventory"].append(self.inventory)
        self._episode_log["demand"].append(demand)
        self._episode_log["jit_order"].append(jit_order)
        self._episode_log["llt_order"].append(llt_order)
        self._episode_log["reward"].append(raw_reward)
        self._episode_log["stockout"].append(1 if shortage > 0 else 0)
        self._episode_log["capacity_violation"].append(1 if cap_violation > 0 else 0)
        self._episode_log["satisfied"].append(satisfied)
        self._episode_log["total_demand"].append(demand)

        # 10) Update history and time
        self.demand_history.pop(0)
        self.demand_history.append(demand)
        self.prev_action = np.array([act[0], act[1]], dtype=np.float32)
        self.t += 1

        terminated = self.t >= self.ep_len
        truncated = False

        info = {
            "revenue": revenue,
            "jit_cost": jit_cost,
            "llt_cost": llt_cost,
            "holding_cost": hold_cost,
            "shortage_cost": short_cost,
            "capacity_cost": cap_cost,
            "demand": demand,
            "shortage": shortage,
            "satisfied": satisfied,
            "cap_violation": cap_violation,
        }
        if terminated:
            info["episode_log"] = self._episode_log

        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _project_pipeline(pipeline: list[float], n_slots: int) -> list[float]:
        """Map a variable-length pipeline to a fixed number of observation slots.

        - If ``len(pipeline) == n_slots`` → identity.
        - If ``len(pipeline) < n_slots``  → zero-pad on the right.
        - If ``len(pipeline) > n_slots``  → evenly distribute entries across slots
          (sum-based bucketing so total in-transit quantity is preserved).
        """
        n = len(pipeline)
        if n == n_slots:
            return list(pipeline)
        if n < n_slots:
            return list(pipeline) + [0.0] * (n_slots - n)
        # n > n_slots: bucket into n_slots bins
        result = [0.0] * n_slots
        for i, val in enumerate(pipeline):
            bucket = int(i * n_slots / n)
            bucket = min(bucket, n_slots - 1)
            result[bucket] += val
        return result

    def _scenario_obs(self) -> list[float]:
        """Return normalised scenario parameters for the observation."""
        return [
            self._jit_cost / self._norm_cost,
            self._llt_cost / self._norm_cost,
            self._hold_cost / self._norm_cost,
            self._short_pen / self._norm_cost,
            self.capacity / self._norm_cap,
            self.llt_lt / self._norm_lt,
        ]

    def _obs(self) -> np.ndarray:
        jit_proj = self._project_pipeline(self.jit_pipeline, self.obs_jit_slots)
        llt_proj = self._project_pipeline(self.llt_pipeline, self.obs_llt_slots)
        # Normalise inventory-related values by max_order for stability
        s = self.max_order  # scaling factor
        obs = np.array(
            [self.inventory / s]
            + [v / s for v in jit_proj]
            + [v / s for v in llt_proj]
            + [d / s for d in self.demand_history]
            + [self.t / self.ep_len]
            + [max(0, self.capacity - self.inventory) / s]
            + list(self.prev_action)
            + self._scenario_obs(),
            dtype=np.float32,
        )
        return obs

    def get_episode_log(self) -> dict[str, list]:
        return self._episode_log
