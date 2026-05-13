"""Policy / value networks for v21 RL tiers (lite, pro, ultra)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from tools.v21.feature_extractor_v20 import N_FEATURES, N_PLAN_FEATURES, N_STATE_FEATURES


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PolicyValueNetLite(nn.Module):
    """31 → 192 → 96; ReLU; value + tanh(plan)."""

    H1 = 192
    H2 = 96

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(N_FEATURES, self.H1),
            nn.ReLU(),
            nn.Linear(self.H1, self.H2),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(self.H2, 1)
        self.plan_head = nn.Linear(self.H2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, N_FEATURES)."""
        h = self.trunk(x)
        return self.value_head(h), torch.tanh(self.plan_head(h))

    def forward_plans(self, x_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_k: (B, K, F); same MLP per row; mean pooled value."""
        b, k, f = x_k.shape
        flat = x_k.reshape(b * k, f)
        v_flat, p_flat = self.forward(flat)
        v = v_flat.reshape(b, k).mean(dim=1, keepdim=True)
        plan_scores = p_flat.reshape(b, k)
        return v, plan_scores


class PolicyValueNetPro(nn.Module):
    """Dual-tower on state(14) + plan(17), LN + GELU stack."""

    D_STATE = 64
    D_PLAN = 64
    H1 = 256
    H2 = 256
    H3 = 128

    def __init__(self):
        super().__init__()
        self.state_tower = nn.Sequential(
            nn.Linear(N_STATE_FEATURES, self.D_STATE),
            nn.ReLU(),
        )
        self.plan_tower = nn.Sequential(
            nn.Linear(N_PLAN_FEATURES, self.D_PLAN),
            nn.ReLU(),
        )
        c = self.D_STATE + self.D_PLAN
        self.trunk = nn.Sequential(
            nn.Linear(c, self.H1),
            nn.LayerNorm(self.H1),
            nn.GELU(),
            nn.Linear(self.H1, self.H2),
            nn.LayerNorm(self.H2),
            nn.GELU(),
            nn.Linear(self.H2, self.H3),
            nn.LayerNorm(self.H3),
            nn.GELU(),
        )
        self.value_head = nn.Linear(self.H3, 1)
        self.plan_head = nn.Linear(self.H3, 1)

    def _split(self, x: torch.Tensor):
        return x[..., :N_STATE_FEATURES], x[..., N_STATE_FEATURES:]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s, p = self._split(x)
        h = self.trunk(torch.cat([self.state_tower(s), self.plan_tower(p)], dim=-1))
        return self.value_head(h), torch.tanh(self.plan_head(h))

    def forward_plans(self, x_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_k: (B, K, F)."""
        b, k, f = x_k.shape
        flat = x_k.reshape(b * k, f)
        v_flat, s_flat = self.forward(flat)
        v = v_flat.reshape(b, k).mean(dim=1, keepdim=True)
        plan_scores = s_flat.reshape(b, k)
        return v, plan_scores


class PolicyValueNetUltra(nn.Module):
    """Token = linear(31→d); TransformerEncoder over K plans; pool value."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        nlayers: int = 2,
        dim_ff: int = 512,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.embed = nn.Linear(N_FEATURES, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_ff,
            dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.plan_head = nn.Linear(d_model, 1)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, N_FEATURES) — K=1."""
        return self.forward_plans(x.unsqueeze(1))

    def forward_plans(self, x_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_k: (B, K, F)."""
        h = self.embed(x_k)
        h = self.encoder(h)
        plan_logits = self.plan_head(h).squeeze(-1)
        pooled = h.mean(dim=1)
        v = self.value_head(pooled)
        return v, torch.tanh(plan_logits)


def build_net(tier: str) -> nn.Module:
    t = tier.lower().strip()
    if t == "lite":
        return PolicyValueNetLite()
    if t == "pro":
        return PolicyValueNetPro()
    if t == "ultra":
        return PolicyValueNetUltra()
    raise ValueError(f"unknown tier {tier!r}")
