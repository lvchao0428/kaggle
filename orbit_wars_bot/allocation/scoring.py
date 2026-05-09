"""Planet value / threat hooks for global allocation (Lux/Halite-style scoring)."""

from __future__ import annotations

import math
from typing import Any, Optional, Protocol


class HasPosition(Protocol):
    x: float
    y: float
    ships: int
    production: int
    owner: int


def eta_lower_bound(dist: float, ship_count: int, max_speed: float = 6.0) -> float:
    """Crude turns lower bound using official-ish speed (not exact for tiny fleets)."""
    from orbit_wars_bot.geom import fleet_speed

    spd = fleet_speed(max(1, int(ship_count)), max_speed)
    if spd <= 1e-6:
        return float("inf")
    return dist / spd


def planet_pressure_score(
    p: HasPosition,
    *,
    my_id: int,
    net_threat: int = 0,
) -> float:
    """
    Higher = more valuable to interact with (capture or defend).
    Extend with graph distance / sun penalty in simulation.forward.
    """
    if p.owner == my_id:
        return 0.0
    prod = float(getattr(p, "production", 0))
    ships = float(getattr(p, "ships", 0))
    hostile = 2.0 if p.owner not in (-1, my_id) else 1.0
    threat_term = 0.15 * max(0, net_threat)
    return hostile * (3.0 * prod + 0.08 * ships) - threat_term


def rank_targets(
    planets: list,
    my_id: int,
    incoming_by_player: Optional[dict[int, dict[int, int]]] = None,
    top_k: int = 12,
) -> list:
    """Return up to top_k planets sorted by descending score (not necessarily owned)."""
    inc = incoming_by_player or {}

    def nt(pid: int) -> int:
        m = inc.get(pid, {})
        atk = sum(v for o, v in m.items() if o not in (-1, my_id))
        own = m.get(my_id, 0)
        return atk - own

    scored = []
    for p in planets:
        s = planet_pressure_score(p, my_id=my_id, net_threat=nt(getattr(p, "id", -1)))
        scored.append((s, p))
    scored.sort(key=lambda t: -t[0])
    return [p for _, p in scored[:top_k]]
