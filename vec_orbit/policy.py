"""Small actor-critic for vec_orbit: Gaussian policy over 6 raw dims, value baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256, action_dim: int = 6):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi_mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        mean = self.pi_mean(h)
        std = torch.exp(self.log_std.clamp(-5, 2))
        value = self.v(h).squeeze(-1)
        return mean, std, value

    def act(self, obs: torch.Tensor):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        pre = dist.rsample()
        logp = dist.log_prob(pre).sum(-1)
        return pre, logp, value
