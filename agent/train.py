"""
train.py
--------
Train a PPO agent on the DualSourceEnv **with domain randomization**
so the resulting policy is cost-aware and robust to parameter changes.

Usage
-----
    python -m agent.train                       # defaults (300k steps)
    python -m agent.train --timesteps 500000    # override total steps
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Allow running as  `python -m agent.train`  from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.dual_source_env import DualSourceEnv
from agent.policy_network import DualSourceFeatureExtractor
from configs.default_config import DEFAULT_CONFIG


# ======================================================================
# Callback – logs episode rewards for plotting
# ======================================================================
class RewardLoggerCallback(BaseCallback):
    """Collects episode rewards during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self._current_reward: float = 0.0

    def _on_step(self) -> bool:
        # DummyVecEnv with n_envs=1 → index 0
        self._current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True


# ======================================================================
# Main training routine
# ======================================================================
def train(cfg: dict) -> None:
    """Train the PPO agent with domain randomization and save artifacts."""

    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["model_save_path"]) or "models", exist_ok=True)

    # --- Environment factory (domain randomization ON) ---
    train_cfg = dict(cfg)
    train_cfg["domain_randomization"] = True

    def make_env():
        return DualSourceEnv(train_cfg)

    vec_env = DummyVecEnv([make_env])

    # --- Policy kwargs (custom feature extractor) ---
    policy_kwargs = dict(
        features_extractor_class=DualSourceFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    # --- PPO agent ---
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=cfg["seed"],
    )

    # --- Train ---
    callback = RewardLoggerCallback()
    print(f"\n{'='*60}")
    print(f"  Training PPO (cost-aware, domain-randomised)")
    print(f"  Timesteps: {cfg['total_timesteps']:,}")
    print(f"  Obs dim:   {vec_env.observation_space.shape[0]}")
    print(f"{'='*60}\n")
    model.learn(total_timesteps=cfg["total_timesteps"], callback=callback)

    # --- Save model ---
    save_path = cfg["model_save_path"]
    model.save(save_path)
    print(f"\n✓ Model saved to {save_path}")

    # --- Generate training plots ---
    _plot_learning_curve(callback.episode_rewards, cfg)
    _plot_evaluation_episode(model, cfg)

    print(f"\n✓ Training artifacts saved to  {cfg['log_dir']}")


# ======================================================================
# Plotting helpers
# ======================================================================
def _smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid").tolist()


def _plot_learning_curve(episode_rewards: list[float], cfg: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episode_rewards, alpha=0.3, label="Episode reward")
    smoothed = _smooth(episode_rewards)
    ax.plot(range(len(smoothed)), smoothed, linewidth=2, label="Smoothed (20-ep)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("PPO Learning Curve – Cost-Aware Dual-Source Inventory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(cfg["log_dir"], "learning_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → Learning curve saved to {path}")


def _plot_evaluation_episode(model, cfg: dict) -> None:
    """Run one episode under *default* params with the trained model."""
    eval_cfg = dict(cfg)
    eval_cfg["domain_randomization"] = False      # deterministic scenario
    env = DualSourceEnv(eval_cfg)
    obs, _ = env.reset(seed=cfg["seed"] + 1000)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    log = env.get_episode_log()
    steps = range(len(log["inventory"]))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # -- Inventory --
    axes[0].plot(steps, log["inventory"], label="On-hand inventory", color="#2196F3")
    axes[0].axhline(cfg["storage_capacity"], ls="--", color="red", alpha=0.6, label="Capacity")
    axes[0].fill_between(steps, log["demand"], alpha=0.2, color="orange", label="Demand")
    axes[0].set_ylabel("Units")
    axes[0].set_title("Inventory Level & Demand")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # -- Orders --
    axes[1].bar(steps, log["jit_order"], alpha=0.7, label="JIT order", color="#4CAF50")
    axes[1].bar(steps, log["llt_order"], bottom=log["jit_order"], alpha=0.7,
                label="LLT order", color="#FF9800")
    axes[1].set_ylabel("Order Qty")
    axes[1].set_title("JIT vs LLT Ordering")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # -- Cumulative reward --
    cum_reward = np.cumsum(log["reward"])
    axes[2].plot(steps, cum_reward, color="#9C27B0", linewidth=2)
    axes[2].set_ylabel("Cumulative Reward")
    axes[2].set_xlabel("Time Step")
    axes[2].set_title("Profit Accumulation")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(cfg["log_dir"], "evaluation_episode.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → Evaluation plot saved to {path}")


# ======================================================================
# CLI entry-point
# ======================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train DualSource-RL agent")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = dict(DEFAULT_CONFIG)
    if args.timesteps:
        cfg["total_timesteps"] = args.timesteps
    if args.seed:
        cfg["seed"] = args.seed
    if args.lr:
        cfg["learning_rate"] = args.lr
    train(cfg)
