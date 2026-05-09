"""Official board geometry and fleet-speed helpers (mirrors submission_v6; safe to import without pulling full agent)."""

from __future__ import annotations

import math

SUN_X = SUN_Y = 50.0
SUN_RADIUS = 10.0
BOARD = 100.0
ROTATION_RADIUS_LIMIT = 50.0
DEFAULT_MAX_SHIP_SPEED = 6.0


def fleet_speed(ships: int, max_speed: float = DEFAULT_MAX_SHIP_SPEED) -> float:
    if ships <= 1:
        return 1.0
    spd = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(spd, max_speed)


def point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    abx, aby = bx - ax, by - ay
    l2 = abx * abx + aby * aby
    if l2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / l2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def segment_hits_sun(ax: float, ay: float, bx: float, by: float, margin: float = 1.5) -> bool:
    return point_segment_distance(SUN_X, SUN_Y, ax, ay, bx, by) < SUN_RADIUS + margin
