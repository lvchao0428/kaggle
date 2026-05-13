"""Map policy network outputs to BatchedOrbitEnv step(actions) tensor."""

from __future__ import annotations

import torch


def raw_vec_to_actions(raw: torch.Tensor, p_planets: int, frac_floor: float = 0.02) -> torch.Tensor:
    """
    raw: (B, 6) unconstrained reals from a Gaussian head.
    Returns actions (B, 2, 3): per player [src, dst, frac] (float; env clamps indices).

    src,dst mapped via (tanh(raw_k)+1)/2 * (P-1).
    frac is sigmoid(raw_frac) clipped to [frac_floor, 1].
    """
    pm1 = float(max(p_planets - 1, 1))
    t = torch.tanh(raw)
    s0 = (t[:, 0:1] + 1.0) * 0.5 * pm1
    d0 = (t[:, 1:2] + 1.0) * 0.5 * pm1
    s1 = (t[:, 3:4] + 1.0) * 0.5 * pm1
    d1 = (t[:, 4:5] + 1.0) * 0.5 * pm1
    fr0 = torch.sigmoid(raw[:, 2:3]).clamp(frac_floor, 1.0)
    fr1 = torch.sigmoid(raw[:, 5:6]).clamp(frac_floor, 1.0)

    out = torch.empty(raw.size(0), 2, 3, device=raw.device, dtype=torch.float32)
    out[:, 0, 0] = s0.squeeze(-1)
    out[:, 0, 1] = d0.squeeze(-1)
    out[:, 0, 2] = fr0.squeeze(-1)
    out[:, 1, 0] = s1.squeeze(-1)
    out[:, 1, 1] = d1.squeeze(-1)
    out[:, 1, 2] = fr1.squeeze(-1)
    return out
