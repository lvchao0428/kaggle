"""PyTorch policy/value net for Orbit Wars RL.

Two heads sharing a trunk:
- value head: V(s) used by PPO advantage estimation
- plan-score head: f(s, plan_features) -> scalar in [-1, 1] used at inference
  to rank plans, mirroring v11.NeuralVal.score_modifier semantics.

Architecture is intentionally tiny so that distillation back to NumPy
preserves capacity:

    state(14) ─┐
                ├─► trunk MLP 64 -> 128 -> 64
    plan(17) ──┘             │
                              ├─► value head (1)
                              └─► plan score head (1, tanh)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .feature_extractor import (
    N_FEATURES,
    N_PLAN_FEATURES,
    N_STATE_FEATURES,
)


class PolicyValueNet(nn.Module):
    HIDDEN1 = 128
    HIDDEN2 = 64

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(N_FEATURES, self.HIDDEN1),
            nn.ReLU(),
            nn.Linear(self.HIDDEN1, self.HIDDEN2),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(self.HIDDEN2, 1)
        self.plan_head = nn.Linear(self.HIDDEN2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, N_FEATURES). Returns (value, plan_score) each (B, 1)."""
        h = self.trunk(x)
        return self.value_head(h), torch.tanh(self.plan_head(h))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]

    def plan_score(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[1]


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
