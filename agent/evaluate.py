"""
evaluate.py
-----------
Evaluate a trained PPO agent across **multiple cost regimes** and compare
against heuristic baselines.  Includes a multi-scenario sweep that tests
generalisation (e.g., JIT cheaper than LLT).

Usage
-----
    python -m agent.evaluate
    python -m agent.evaluate --episodes 20
    python -m agent.evaluate --sweep            # multi-scenario sweep
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.dual_source_env import DualSourceEnv
from configs.default_config import DEFAULT_CONFIG


# ======================================================================
# Baseline policies
# ======================================================================
class BaseStockPolicy:
    """Tailored Base-Surge (TBS) heuristic.

    LLT orders a constant *base* quantity each period.
    JIT 'surges' to cover expected shortfall vs a target.
    """

    def __init__(self, target_inventory: float = 60.0, base_llt: float = 15.0,
                 max_order: float = 50.0):
        self.target = target_inventory
        self.base_llt = base_llt
        self.max_order = max_order

    def predict(self, obs: np.ndarray):
        inventory = obs[0]
        gap = max(0.0, self.target - inventory)
        jit_order = min(gap, self.max_order)
        llt_order = min(self.base_llt, self.max_order)
        # Normalise to [0,1] like the RL agent's action space
        action = np.array([jit_order / self.max_order, llt_order / self.max_order],
                          dtype=np.float32)
        return action, None


# ======================================================================
# Evaluation runner
# ======================================================================
def run_episodes(policy, env: DualSourceEnv, n_episodes: int, seed: int,
                 deterministic: bool = True):
    """Run *n_episodes* and return aggregated metrics."""
    all_metrics: list[dict] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            if hasattr(policy, "predict"):
                action, _ = policy.predict(obs, deterministic=deterministic) if \
                    "deterministic" in policy.predict.__code__.co_varnames else \
                    policy.predict(obs)
            else:
                action = policy(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        log = env.get_episode_log()
        total_demand = sum(log["total_demand"])
        total_satisfied = sum(log["satisfied"])
        metrics = {
            "episode": ep,
            "total_profit": sum(log["reward"]),
            "service_level": total_satisfied / total_demand if total_demand > 0 else 1.0,
            "stockout_freq": sum(log["stockout"]) / len(log["stockout"]),
            "cap_violations": sum(log["capacity_violation"]),
            "avg_inventory": np.mean(log["inventory"]),
            "jit_total": sum(log["jit_order"]),
            "llt_total": sum(log["llt_order"]),
        }
        jit_total = metrics["jit_total"]
        llt_total = metrics["llt_total"]
        total_orders = jit_total + llt_total
        metrics["jit_ratio"] = jit_total / total_orders if total_orders > 0 else 0.0
        all_metrics.append(metrics)

    return pd.DataFrame(all_metrics)


# ======================================================================
# Standard evaluation (single scenario, vs baseline)
# ======================================================================
def evaluate(cfg: dict, n_episodes: int = 10) -> None:
    """Load model, run evaluation, compare vs baseline, save results."""
    os.makedirs(cfg["log_dir"], exist_ok=True)

    eval_cfg = dict(cfg)
    eval_cfg["domain_randomization"] = False
    # Pin pipeline obs slots to training defaults for shape consistency
    eval_cfg["obs_jit_slots"] = cfg.get("jit_lead_time", 1)
    eval_cfg["obs_llt_slots"] = 6
    env = DualSourceEnv(eval_cfg)

    # RL agent
    model_path = cfg["model_save_path"]
    if not os.path.exists(model_path + ".zip"):
        print(f"✗ Trained model not found at {model_path}.zip – run train.py first.")
        return
    model = PPO.load(model_path, env=env)
    print("Evaluating RL agent …")
    rl_df = run_episodes(model, env, n_episodes, cfg["seed"])
    rl_df["policy"] = "PPO"

    # Baseline
    baseline = BaseStockPolicy(
        target_inventory=cfg["demand_mean"] * 3,
        base_llt=cfg["demand_mean"] * 0.7,
        max_order=cfg["max_order"],
    )
    print("Evaluating TBS baseline …")
    bl_df = run_episodes(baseline, env, n_episodes, cfg["seed"])
    bl_df["policy"] = "TBS-Heuristic"

    # Combined
    combined = pd.concat([rl_df, bl_df], ignore_index=True)
    csv_path = os.path.join(cfg["log_dir"], "evaluation_results.csv")
    combined.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}\n")

    # Summary
    summary = combined.groupby("policy").agg(
        profit_mean=("total_profit", "mean"),
        profit_std=("total_profit", "std"),
        service_level=("service_level", "mean"),
        stockout_freq=("stockout_freq", "mean"),
        avg_inventory=("avg_inventory", "mean"),
        jit_ratio=("jit_ratio", "mean"),
    ).round(2)
    print(summary.to_string())
    print()

    # Bar chart comparison
    _plot_comparison(summary, cfg)


# ======================================================================
# Multi-scenario sweep
# ======================================================================
SWEEP_SCENARIOS = [
    # label,            jit_cost, llt_cost, hold, short_pen, capacity, llt_lt
    ("Default",            8.0,     4.0,    1.0,    10.0,    200.0,     6),
    ("JIT cheap",          3.0,     8.0,    1.0,    10.0,    200.0,     6),
    ("JIT=LLT",            6.0,     6.0,    1.0,    10.0,    200.0,     6),
    ("High shortage",      8.0,     4.0,    1.0,    25.0,    200.0,     6),
    ("Low capacity",       8.0,     4.0,    1.0,    10.0,    80.0,      6),
    ("Long LLT (10)",      8.0,     4.0,    1.0,    10.0,    200.0,    10),
    ("Short LLT (2)",      8.0,     4.0,    1.0,    10.0,    200.0,     2),
    ("High holding",       8.0,     4.0,    3.0,    10.0,    200.0,     6),
    ("JIT cheap+high pen", 3.0,    10.0,    1.0,    20.0,    200.0,     6),
    ("All expensive",     12.0,    10.0,    2.5,    20.0,    120.0,     8),
]


def sweep_evaluate(cfg: dict, n_episodes: int = 10) -> None:
    """Evaluate the trained agent across multiple cost regimes."""
    os.makedirs(cfg["log_dir"], exist_ok=True)

    model_path = cfg["model_save_path"]
    if not os.path.exists(model_path + ".zip"):
        print(f"✗ Trained model not found at {model_path}.zip – run train.py first.")
        return

    rows: list[dict] = []
    for label, jit_c, llt_c, hold_c, short_p, cap, llt_lt in SWEEP_SCENARIOS:
        scfg = dict(cfg)
        scfg["domain_randomization"] = False
        scfg["jit_unit_cost"] = jit_c
        scfg["llt_unit_cost"] = llt_c
        scfg["holding_cost_per_unit"] = hold_c
        scfg["shortage_penalty_per_unit"] = short_p
        scfg["storage_capacity"] = cap
        scfg["llt_lead_time"] = llt_lt
        # Pin pipeline obs slots to training defaults for shape consistency
        scfg["obs_jit_slots"] = cfg.get("jit_lead_time", 1)
        scfg["obs_llt_slots"] = 6

        env = DualSourceEnv(scfg)
        model = PPO.load(model_path, env=env)

        df = run_episodes(model, env, n_episodes, cfg["seed"])
        row = {
            "scenario": label,
            "jit_cost": jit_c,
            "llt_cost": llt_c,
            "hold_cost": hold_c,
            "short_pen": short_p,
            "capacity": cap,
            "llt_lt": llt_lt,
            "profit_mean": df["total_profit"].mean(),
            "profit_std": df["total_profit"].std(),
            "service_level": df["service_level"].mean(),
            "jit_ratio": df["jit_ratio"].mean(),
            "avg_inventory": df["avg_inventory"].mean(),
            "stockout_freq": df["stockout_freq"].mean(),
        }
        rows.append(row)
        cheaper = "JIT" if jit_c < llt_c else ("LLT" if llt_c < jit_c else "equal")
        print(
            f"  {label:22s}  profit={row['profit_mean']:+9.0f}  "
            f"SL={row['service_level']:.2f}  JIT%={row['jit_ratio']:.2f}  "
            f"[cheaper={cheaper}]"
        )

    sweep_df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg["log_dir"], "sweep_results.csv")
    sweep_df.to_csv(csv_path, index=False)
    print(f"\n✓ Sweep results saved to {csv_path}")

    _plot_sweep(sweep_df, cfg)


def _plot_sweep(df: pd.DataFrame, cfg: dict) -> None:
    """Generate a grouped bar chart from the sweep results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(df))
    labels = df["scenario"].tolist()

    # Profit
    colors_profit = ["#4CAF50" if p > 0 else "#F44336" for p in df["profit_mean"]]
    axes[0].bar(x, df["profit_mean"], color=colors_profit)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Avg Profit")
    axes[0].set_title("Profit by Scenario")
    axes[0].grid(axis="y", alpha=0.3)

    # Service level
    axes[1].bar(x, df["service_level"], color="#2196F3")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Service Level")
    axes[1].set_title("Service Level by Scenario")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(0.95, ls="--", color="red", alpha=0.4, label="95% target")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    # JIT ratio
    axes[2].bar(x, df["jit_ratio"], color="#FF9800")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("JIT Ratio")
    axes[2].set_title("JIT Usage Ratio by Scenario")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Multi-Scenario Evaluation – Cost-Aware PPO", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(cfg["log_dir"], "sweep_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Sweep chart saved to {path}")


# ======================================================================
# Standard comparison plot
# ======================================================================
def _plot_comparison(summary: pd.DataFrame, cfg: dict) -> None:
    metrics = ["profit_mean", "service_level", "avg_inventory", "jit_ratio"]
    titles = ["Avg Profit", "Service Level", "Avg Inventory", "JIT Ratio"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    colors = ["#2196F3", "#FF9800"]
    for ax, metric, title in zip(axes, metrics, titles):
        vals = summary[metric]
        ax.bar(vals.index, vals.values, color=colors[:len(vals)])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("RL (PPO) vs Tailored Base-Surge Heuristic", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(cfg["log_dir"], "comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Comparison chart saved to {path}")


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--sweep", action="store_true",
                        help="Run multi-scenario sweep evaluation")
    args = parser.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    if args.sweep:
        sweep_evaluate(cfg, n_episodes=args.episodes)
    else:
        evaluate(cfg, n_episodes=args.episodes)
