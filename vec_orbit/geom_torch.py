"""Batched geometry matching orbit_wars_bot.geom (speed, board, sun)."""

from __future__ import annotations

import math
from typing import Tuple

import torch

# Mirrors orbit_wars_bot.geom
SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
BOARD = 100.0
DEFAULT_MAX_SHIP_SPEED = 6.0
SEGMENT_SUN_MARGIN = 1.5
OOB_MARGIN = 0.5


def fleet_speed(ships: torch.Tensor, max_speed: float = DEFAULT_MAX_SHIP_SPEED) -> torch.Tensor:
    """(device-matching) same formula as geom.fleet_speed; ships positive."""
    s = ships.clamp(min=1.0).float()
    spd = 1.0 + (max_speed - 1.0) * (torch.log(s) / math.log(1000.0)).pow(1.5)
    return torch.clamp(spd, max=max_speed)


def point_segment_distance_sq(
    px: torch.Tensor,
    py: torch.Tensor,
    ax: torch.Tensor,
    ay: torch.Tensor,
    bx: torch.Tensor,
    by: torch.Tensor,
) -> torch.Tensor:
    """(E,) squared distance from point (px,py) to segment AB."""
    abx = bx - ax
    aby = by - ay
    l2 = abx * abx + aby * aby + 1e-12
    t = ((px - ax) * abx + (py - ay) * aby) / l2
    t = t.clamp(0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy


def segment_hits_sun(
    ax: torch.Tensor,
    ay: torch.Tensor,
    bx: torch.Tensor,
    by: torch.Tensor,
    margin: float = SEGMENT_SUN_MARGIN,
) -> torch.Tensor:
    """(E,) bool: segment AB passes within SUN_RADIUS + margin of sun."""
    d2 = point_segment_distance_sq(
        torch.full_like(ax, SUN_X),
        torch.full_like(ay, SUN_Y),
        ax,
        ay,
        bx,
        by,
    )
    return d2 < (SUN_RADIUS + margin) ** 2


def out_of_bounds_xy(x: torch.Tensor, y: torch.Tensor, m: float = OOB_MARGIN) -> torch.Tensor:
    """(E,) bool."""
    return (x < m) | (x > BOARD - m) | (y < m) | (y > BOARD - m)
