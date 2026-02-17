"""
visualization.py
----------------
Matplotlib-based plotting helpers used by the Streamlit game dashboard.
All functions return a ``matplotlib.figure.Figure`` so the caller can
simply pass it to ``st.pyplot(fig)``.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_inventory_and_demand(log: dict, capacity: float, steps: int | None = None) -> Figure:
    """Inventory level with demand overlay."""
    n = steps or len(log["inventory"])
    x = range(n)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(list(x), log["inventory"][:n], label="Inventory", color="#2196F3", linewidth=2)
    ax.fill_between(list(x), log["demand"][:n], alpha=0.25, color="#FF9800", label="Demand")
    ax.axhline(capacity, ls="--", color="#F44336", alpha=0.6, label="Capacity")
    ax.set_ylabel("Units")
    ax.set_title("📦  Inventory Level & Demand")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_orders(log: dict, steps: int | None = None) -> Figure:
    """Stacked bar chart of JIT vs LLT orders."""
    n = steps or len(log["jit_order"])
    x = np.arange(n)
    jit = np.array(log["jit_order"][:n])
    llt = np.array(log["llt_order"][:n])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(x, jit, alpha=0.8, label="JIT order", color="#4CAF50", width=1.0)
    ax.bar(x, llt, bottom=jit, alpha=0.8, label="LLT order", color="#FF9800", width=1.0)
    ax.set_ylabel("Qty")
    ax.set_title("🚚  Orders Placed (JIT vs LLT)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_cumulative_profit(log: dict, steps: int | None = None) -> Figure:
    """Cumulative reward (profit) over time."""
    n = steps or len(log["reward"])
    cum = np.cumsum(log["reward"][:n])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(range(n), cum, color="#9C27B0", linewidth=2)
    ax.fill_between(range(n), cum, alpha=0.15, color="#9C27B0")
    ax.set_ylabel("Cumulative Profit")
    ax.set_xlabel("Time Step")
    ax.set_title("💰  Profit Accumulation")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_service_level(log: dict, steps: int | None = None) -> Figure:
    """Rolling service level (fraction of demand satisfied)."""
    n = steps or len(log["satisfied"])
    satisfied = np.array(log["satisfied"][:n])
    demand = np.array(log["total_demand"][:n])
    # Running average
    cum_sat = np.cumsum(satisfied)
    cum_dem = np.cumsum(demand)
    service = np.where(cum_dem > 0, cum_sat / cum_dem, 1.0)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(range(n), service * 100, color="#009688", linewidth=2)
    ax.axhline(95, ls="--", color="#F44336", alpha=0.5, label="95% target")
    ax.set_ylabel("Service Level %")
    ax.set_xlabel("Time Step")
    ax.set_title("📊  Cumulative Service Level")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def compute_summary_metrics(log: dict) -> dict:
    """Return a flat dict of summary KPIs."""
    total_demand = sum(log["total_demand"])
    total_satisfied = sum(log["satisfied"])
    jit_total = sum(log["jit_order"])
    llt_total = sum(log["llt_order"])
    total_orders = jit_total + llt_total
    return {
        "Total Profit": f"{sum(log['reward']):,.1f}",
        "Service Level": f"{(total_satisfied / total_demand * 100) if total_demand else 100:.1f}%",
        "Stockout Periods": f"{sum(log['stockout'])} / {len(log['stockout'])}",
        "Capacity Violations": str(sum(log["capacity_violation"])),
        "Avg Inventory": f"{np.mean(log['inventory']):.1f}",
        "JIT / LLT Ratio": f"{(jit_total / total_orders * 100) if total_orders else 0:.0f}% / "
                           f"{(llt_total / total_orders * 100) if total_orders else 0:.0f}%",
    }
