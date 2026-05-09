"""
Short-horizon forward plumbing (Planet Wars-style rollouts).

First version: geometry-only rollout without cloning the full Kaggle interpreter.
Use `roll_fleet_to_collision` for ETA / aim checks; extend with `Environment.clone`
when you want exact branching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from orbit_wars_bot import geom


@dataclass
class FleetSimState:
    x: float
    y: float
    angle: float
    ships: int


def roll_fleet_to_collision(
    start: FleetSimState,
    planets,
    planet_pos_fn,
    max_steps: int = 120,
    max_speed: float = 6.0,
) -> Optional[Tuple[int, int]]:
    """
    Rough mirror of submission_v6 GameState._predict_fleet_target (discrete steps).
    planets: iterable of objects with id, radius; positions from planet_pos_fn(pid, step).
    Returns (planet_id, arrival_step) or None if dies to sun/boundary first.
    """
    spd0 = geom.fleet_speed(start.ships, max_speed)
    cx, cy = start.x, start.y
    dx = math.cos(start.angle) * spd0
    dy = math.sin(start.angle) * spd0
    for t in range(1, max_steps + 1):
        nx, ny = cx + dx, cy + dy
        if not (0.0 <= nx <= geom.BOARD and 0.0 <= ny <= geom.BOARD):
            return None
        if geom.point_segment_distance(geom.SUN_X, geom.SUN_Y, cx, cy, nx, ny) < geom.SUN_RADIUS:
            return None
        best_pid: Optional[int] = None
        best_d = float("inf")

        for p in planets:
            px, py = planet_pos_fn(p, t)
            d = geom.point_segment_distance(px, py, cx, cy, nx, ny)
            if d < p.radius and d < best_d:
                best_d = d
                best_pid = p.id
        if best_pid is not None:
            return best_pid, t
        cx, cy = nx, ny
    return None


def score_candidate_launches_stub(n_candidates: int) -> List[float]:
    """Placeholder for evaluating multiple move subsets; return uniform scores."""
    return [1.0 / max(1, n_candidates)] * n_candidates
