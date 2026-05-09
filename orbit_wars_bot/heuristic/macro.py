"""Map low-dimensional RL actions to Orbit Wars move lists (wraps v6 output)."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from orbit_wars_bot.heuristic import v6_wrapped


def apply_macro_to_moves(
    moves: List,
    action_vec: np.ndarray,
    *,
    min_send_frac: float = 0.15,
) -> List:
    """
    action_vec: shape (3,), each in [-1, 1] from Tanh policy.
    - dim0: global ship count scale on every leg
    - dim1: aggression — mix toward sending more per leg (still capped by v6 intent)
    - dim2: minimum fraction floor modifier
    """
    a = np.clip(np.asarray(action_vec, dtype=np.float64).reshape(-1), -1.0, 1.0)
    while a.size < 3:
        a = np.pad(a, (0, 3 - a.size), constant_values=0.0)

    ship_hi = 0.55 + 0.45 * (a[0] + 1.0) * 0.5  # ~[0.55, 1.0]
    agg = 0.85 + 0.15 * (a[1] + 1.0) * 0.5  # ~[0.85, 1.0]
    floor = min_send_frac * (0.5 + 0.5 * (a[2] + 1.0) * 0.5)

    out: List = []
    for m in moves:
        if not m or len(m) < 3:
            continue
        pid, ang, n = int(m[0]), float(m[1]), int(m[2])
        scaled = int(max(1, round(n * ship_hi * agg)))
        min_keep = max(1, int(round(n * floor)))
        scaled = max(scaled, min(min_keep, n))
        if scaled < 1:
            continue
        out.append([pid, ang, scaled])
    return out


def macro_agent(obs: Any, config: Any, action_vec: np.ndarray) -> List:
    base = v6_wrapped.agent(obs, config)
    return apply_macro_to_moves(base, action_vec)
