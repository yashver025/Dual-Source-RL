"""
play_game.py
------------
Streamlit interactive dashboard for the *Game Phase*.

The trained cost-aware PPO agent receives scenario parameters (costs,
capacity, lead time) as part of its observation, so it adapts its ordering
strategy when users change sliders in the sidebar.

Launch with:
    streamlit run game/play_game.py
"""

from __future__ import annotations

import os
import sys
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from stable_baselines3 import PPO

from environment.dual_source_env import DualSourceEnv
from configs.default_config import DEFAULT_CONFIG
from game.constraints import ScenarioConstraints
from game.visualization import (
    plot_inventory_and_demand,
    plot_orders,
    plot_cumulative_profit,
    plot_service_level,
    compute_summary_metrics,
)

# ======================================================================
# Page config
# ======================================================================
st.set_page_config(
    page_title="DualSource-RL  ·  Interactive Game",
    page_icon="🎮",
    layout="wide",
)

# ======================================================================
# Custom CSS for premium feel
# ======================================================================
st.markdown("""
<style>
    /* Dark gradient header */
    .main .block-container { padding-top: 1.5rem; }
    h1 { 
        background: linear-gradient(90deg, #6C63FF, #3F51B5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid rgba(108, 99, 255, 0.2);
    }
    .stMetric label { color: #a0a0c0 !important; font-size: 0.85rem; }
    .stMetric [data-testid="stMetricValue"] { color: #e0e0ff !important; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }
    div[data-testid="stSidebar"] label { color: #c0c0e0 !important; }
    div[data-testid="stSidebar"] .stMarkdown p { color: #a0a0c0; }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# Sidebar – scenario configuration
# ======================================================================
st.sidebar.markdown("## 🎛️  Scenario Controls")

sc = ScenarioConstraints()

st.sidebar.markdown("### 💲 Costs")
sc.jit_unit_cost = st.sidebar.slider("JIT unit cost", 1.0, 20.0, 8.0, 0.5)
sc.llt_unit_cost = st.sidebar.slider("LLT unit cost", 1.0, 15.0, 4.0, 0.5)
sc.holding_cost_per_unit = st.sidebar.slider("Holding cost / unit", 0.1, 5.0, 1.0, 0.1)
sc.shortage_penalty_per_unit = st.sidebar.slider("Shortage penalty / unit", 1.0, 30.0, 10.0, 1.0)

st.sidebar.markdown("### 📦 Capacity & Lead Time")
sc.storage_capacity = st.sidebar.slider("Storage capacity", 50.0, 500.0, 200.0, 10.0)
sc.llt_lead_time = st.sidebar.slider("LLT lead time (periods)", 2, 12, 6)

st.sidebar.markdown("### 📈 Demand")
sc.demand_distribution = st.sidebar.selectbox("Distribution", ["poisson", "normal"])
sc.demand_mean = st.sidebar.slider("Mean demand", 5.0, 60.0, 20.0, 1.0)
if sc.demand_distribution == "normal":
    sc.demand_std = st.sidebar.slider("Demand std dev", 1.0, 20.0, 5.0, 0.5)

st.sidebar.markdown("### ⚡ Disruption Modes")
spike_on = st.sidebar.toggle("Demand spike mode", value=False)
sc.demand_spike_prob = st.sidebar.slider(
    "Spike probability", 0.0, 0.3, 0.05 if spike_on else 0.0, 0.01,
    disabled=not spike_on,
)
sc.demand_spike_multiplier = st.sidebar.slider(
    "Spike multiplier", 1.5, 5.0, 3.0, 0.5, disabled=not spike_on
)
disruption_on = st.sidebar.toggle("Supply disruption mode", value=False)
sc.supply_disruption_prob = st.sidebar.slider(
    "Disruption probability", 0.0, 0.3, 0.05 if disruption_on else 0.0, 0.01,
    disabled=not disruption_on,
)

st.sidebar.markdown("### 🔒 Capacity Constraint")
cap_mode = st.sidebar.radio("Capacity mode", ["Hard limit", "Soft penalty"], horizontal=True)
sc.hard_capacity = cap_mode == "Hard limit"

sim_seed = st.sidebar.number_input("Random seed", value=123, step=1)


# ======================================================================
# Header
# ======================================================================
st.markdown("# 🎮  DualSource-RL  ·  Interactive Simulation")
st.markdown(
    "Configure the scenario in the sidebar, then press **▶ Run Simulation** to "
    "watch the **cost-aware** RL agent adapt its ordering decisions in real time."
)

# Show current cost regime
cheaper = "JIT" if sc.jit_unit_cost < sc.llt_unit_cost else (
    "LLT" if sc.llt_unit_cost < sc.jit_unit_cost else "Equal")
st.info(
    f"💡 **Current regime:** JIT cost = {sc.jit_unit_cost}, "
    f"LLT cost = {sc.llt_unit_cost} → **{cheaper} is cheaper**"
)

# ======================================================================
# Model loading
# ======================================================================
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    DEFAULT_CONFIG["model_save_path"],
)


@st.cache_resource
def load_model(path: str):
    """Load trained PPO model (cached across reruns)."""
    if os.path.exists(path + ".zip"):
        return PPO.load(path)
    return None


model = load_model(MODEL_PATH)
if model is None:
    st.error(
        f"⚠️  Trained model not found at `{MODEL_PATH}.zip`.\n\n"
        "Run training first:\n```bash\npython -m agent.train\n```"
    )
    st.stop()

# ======================================================================
# Run simulation
# ======================================================================
run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)

if run_btn:
    # Build env with user constraints — domain_randomization OFF,
    # but scenario params are still in the obs so the model can condition on them.
    env_cfg = dict(DEFAULT_CONFIG)
    env_cfg.update(sc.to_env_overrides())
    env_cfg["seed"] = int(sim_seed)
    env_cfg["domain_randomization"] = False
    # Pin observation pipeline slots to training defaults (1 JIT + 6 LLT)
    env_cfg["obs_jit_slots"] = DEFAULT_CONFIG["jit_lead_time"]   # 1
    env_cfg["obs_llt_slots"] = 6   # training default LLT slots
    env = DualSourceEnv(env_cfg)

    obs, _ = env.reset(seed=int(sim_seed))

    # Progress bar
    progress = st.progress(0, text="Simulating …")
    ep_len = env_cfg["episode_length"]

    for step in range(ep_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        progress.progress((step + 1) / ep_len, text=f"Step {step + 1} / {ep_len}")
        if terminated or truncated:
            break

    progress.empty()

    log = env.get_episode_log()

    # ── Summary metrics ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊  Performance Summary")
    summary = compute_summary_metrics(log)
    cols = st.columns(len(summary))
    for col, (label, value) in zip(cols, summary.items()):
        col.metric(label, value)

    # ── Charts ───────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_inventory_and_demand(log, env_cfg["storage_capacity"]))
    with c2:
        st.pyplot(plot_orders(log))

    c3, c4 = st.columns(2)
    with c3:
        st.pyplot(plot_cumulative_profit(log))
    with c4:
        st.pyplot(plot_service_level(log))

    # ── Raw data toggle ──────────────────────────────────────────────
    with st.expander("📋  View raw episode data"):
        import pandas as pd
        df = pd.DataFrame({
            "Step": range(len(log["inventory"])),
            "Inventory": log["inventory"],
            "Demand": log["demand"],
            "JIT Order": log["jit_order"],
            "LLT Order": log["llt_order"],
            "Reward": log["reward"],
            "Stockout": log["stockout"],
        })
        st.dataframe(df, use_container_width=True)

    st.success("✅  Simulation complete!")
