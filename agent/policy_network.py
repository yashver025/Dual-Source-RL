"""
policy_network.py
-----------------
Custom feature-extractor / policy architecture for the PPO agent.
Uses Stable-Baselines3's ``ActorCriticPolicy`` with a wider MLP
to handle the cost-aware 23-dim observation space.
"""

from __future__ import annotations

import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class DualSourceFeatureExtractor(BaseFeaturesExtractor):
    """Two hidden-layer MLP feature extractor.

    Input (23-dim observation) → 256 → 256 → features_dim (default 128).
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        in_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)
