# 🎮 DualSource-RL: Reinforcement Learning for Dual Sourcing Inventory Decisions

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/Open%20Source-✓-brightgreen.svg)]()

A fully open-source reinforcement learning system that trains a **dual-sourcing inventory decision agent** and deploys it in an **interactive simulation game** where users can experiment with constraints and observe the model's behaviour in real time.

---

## 📋 Table of Contents

- [Problem Description](#-problem-description)
- [Environment Dynamics](#-environment-dynamics)
- [RL Algorithm](#-rl-algorithm)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Training the Agent](#-training-the-agent)
- [Evaluating & Baselines](#-evaluating--baselines)
- [Launching the Game](#-launching-the-game)
- [Configuration Reference](#-configuration-reference)
- [Example Results](#-example-results)
- [License](#-license)

---

## 🧩 Problem Description

A **dual-source inventory system** replenishes stock from two suppliers:

| Source | Lead Time | Unit Cost | Role |
|--------|-----------|-----------|------|
| **JIT** (Just-In-Time) | 1 period (fast) | Higher | Emergency / surge |
| **LLT** (Long Lead Time) | 4-8 periods (slow) | Lower | Bulk / base |

At every time step the agent must decide **how much to order from each source** while facing:

- **Stochastic demand** (Poisson or Normal)
- **Supply fill-rate uncertainty** (random fraction of orders actually delivered)
- **Storage capacity constraints** (hard limit or soft penalty)
- **Holding cost** for excess inventory
- **Shortage penalty** for unmet demand

The objective is to **maximise total profit** (revenue minus all costs) over a planning horizon.

### Reward Function

```
reward = revenue
       − JIT_ordering_cost
       − LLT_ordering_cost
       − holding_cost
       − shortage_penalty
       − capacity_penalty
```

---

## ⚙️ Environment Dynamics

### State Space

The observation vector includes:

| Component | Dimension |
|-----------|-----------|
| Current inventory level | 1 |
| JIT pipeline (in-transit) | `jit_lead_time` |
| LLT pipeline (in-transit) | `llt_lead_time` |
| Recent demand history | `demand_history_len` |
| Normalised time step | 1 |
| Remaining capacity | 1 |
| Previous actions (JIT, LLT) | 2 |

### Action Space

`Box(2)` — continuous values in `[0, 1]`, scaled to `[0, max_order]`:

- `action[0]` → JIT order quantity
- `action[1]` → LLT order quantity

### Transition Dynamics

1. Orders placed at time *t* enter the pipeline.
2. Pipeline orders from *t − lead_time* arrive (subject to random fill rate).
3. Stochastic demand is realised.
4. Inventory is updated: `inventory += arrivals − satisfied_demand`.
5. Capacity constraints are enforced.

### Episode Length

Configurable; default is **150 time steps**.

---

## 🤖 RL Algorithm

We use **Proximal Policy Optimisation (PPO)** via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/):

- **Policy**: MLP with custom feature extractor (256 → 256 → 128 ReLU)
- **Actor / Critic**: Separate heads (128 → 64)
- **Key hyperparameters**: `lr = 3e-4`, `γ = 0.99`, `clip = 0.2`, `ent_coef = 0.01`

Training typically converges within **100-150k steps** (~700-1000 episodes).

---

## 📂 Project Structure

```
DualSource-RL/
│
├── environment/
│   ├── __init__.py
│   ├── dual_source_env.py    # Gymnasium environment
│   ├── demand_model.py       # Stochastic demand generators
│   └── supply_model.py       # Fill-rate & disruption models
│
├── agent/
│   ├── __init__.py
│   ├── policy_network.py     # Custom SB3 feature extractor
│   ├── train.py              # PPO training script
│   └── evaluate.py           # Evaluation & baseline comparison
│
├── game/
│   ├── __init__.py
│   ├── play_game.py          # Streamlit interactive dashboard
│   ├── constraints.py        # Scenario parameter dataclass
│   └── visualization.py      # Matplotlib plotting helpers
│
├── configs/
│   ├── __init__.py
│   └── default_config.py     # Central configuration
│
├── models/                   # Saved trained models
├── logs/                     # Training logs & plots
├── notebooks/                # Jupyter notebooks (optional)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/DualSource-RL.git
cd DualSource-RL
pip install -r requirements.txt
```

### 2. Train

```bash
python -m agent.train
```

### 3. Play

```bash
streamlit run game/play_game.py
```

---

## 🏋️ Training the Agent

```bash
# Default: 150,000 timesteps
python -m agent.train

# Custom settings
python -m agent.train --timesteps 200000 --seed 123 --lr 1e-4
```

Training produces:

| Artifact | Path |
|----------|------|
| Saved model | `models/dual_source_rl.zip` |
| Learning curve | `logs/learning_curve.png` |
| Evaluation episode | `logs/evaluation_episode.png` |

---

## 📈 Evaluating & Baselines

```bash
# Evaluate RL agent vs Tailored Base-Surge heuristic
python -m agent.evaluate --episodes 20
```

This compares the PPO agent against a **Tailored Base-Surge (TBS)** heuristic:

- **LLT base order**: Constant quantity each period
- **JIT surge**: Covers shortfall vs target inventory level

Output: summary table + `logs/comparison.png` bar chart.

---

## 🎮 Launching the Game

```bash
streamlit run game/play_game.py
```

### Sidebar Controls

| Category | Parameters |
|----------|-----------|
| **Costs** | JIT cost, LLT cost, holding cost, shortage penalty |
| **Capacity** | Storage limit, LLT lead time |
| **Demand** | Distribution, mean, std deviation |
| **Disruptions** | Demand spike mode, supply disruption mode |
| **Constraint Mode** | Hard capacity limit vs soft penalty |

### Dashboard Displays

- 📦 Inventory level & demand over time
- 🚚 JIT vs LLT orders (stacked bar)
- 💰 Cumulative profit curve
- 📊 Running service level
- Summary KPIs (profit, service level, stockouts, etc.)
- Raw episode data table

---

## ⚙️ Configuration Reference

All defaults live in `configs/default_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `unit_revenue` | 12.0 | Revenue per unit sold |
| `jit_unit_cost` | 8.0 | Cost per unit ordered from JIT |
| `llt_unit_cost` | 4.0 | Cost per unit ordered from LLT |
| `holding_cost_per_unit` | 1.0 | Per-unit holding cost per period |
| `shortage_penalty_per_unit` | 10.0 | Penalty per unit of unmet demand |
| `storage_capacity` | 200 | Max inventory capacity |
| `max_order` | 50 | Max order quantity per source |
| `episode_length` | 150 | Steps per episode |
| `demand_mean` | 20 | Mean demand |
| `jit_lead_time` | 1 | JIT delivery lag |
| `llt_lead_time` | 6 | LLT delivery lag |
| `total_timesteps` | 150,000 | Training steps |

---

## 📊 Example Results

After training for 150k steps, typical results:

| Metric | PPO Agent | TBS Heuristic |
|--------|-----------|---------------|
| Avg Profit | ~350-450 | ~200-300 |
| Service Level | 92-97% | 85-92% |
| Stockout Freq | 5-10% | 10-20% |
| JIT Ratio | 30-45% | 40-60% |

The RL agent learns to:
- Use LLT for steady base replenishment
- Reserve JIT for demand surge response
- Balance inventory levels to minimise holding costs
- Adapt ordering to pipeline state

---

## 📄 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — PPO implementation
- [Gymnasium](https://gymnasium.farama.org/) — Environment API
- [Streamlit](https://streamlit.io/) — Interactive dashboard
- [PyTorch](https://pytorch.org/) — Deep learning backend

---

*Built with ❤️ for the operations research & reinforcement learning community.*
