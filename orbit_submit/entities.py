from __future__ import annotations

from orbit_submit.constants import *

class Planet:
    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production",
                 "initial_x", "initial_y", "is_comet")

    def __init__(self, id, owner, x, y, radius, ships, production,
                 initial_x=0.0, initial_y=0.0, is_comet=False):
        self.id = id; self.owner = owner
        self.x = x; self.y = y; self.radius = radius
        self.ships = ships; self.production = production
        self.initial_x = initial_x; self.initial_y = initial_y
        self.is_comet = is_comet

    def dist(self, o: "Planet") -> float:
        return math.hypot(self.x - o.x, self.y - o.y)

    def dist_xy(self, x, y) -> float:
        return math.hypot(self.x - x, self.y - y)


class Fleet:
    __slots__ = ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")

    def __init__(self, id, owner, x, y, angle, from_planet_id, ships):
        self.id = id; self.owner = owner; self.x = x; self.y = y
        self.angle = angle; self.from_planet_id = from_planet_id; self.ships = ships


def _combat(owner: int, garrison: int, arrivals: List[Tuple[int, int]]) -> Tuple[int, int]:
    if not arrivals:
        return owner, max(0, int(garrison))
    by_owner: Dict[int, int] = defaultdict(int)
    for o, s in arrivals:
        if s > 0:
            by_owner[int(o)] += int(s)
    if not by_owner:
        return owner, max(0, int(garrison))
    forces = sorted(by_owner.items(), key=lambda kv: kv[1], reverse=True)
    if len(forces) >= 2 and forces[0][1] == forces[1][1]:
        return owner, max(0, int(garrison))
    atk_owner, atk_ships = forces[0]
    second = forces[1][1] if len(forces) >= 2 else 0
    survivor = atk_ships - second
    if survivor <= 0:
        return owner, max(0, int(garrison))
    if atk_owner == owner:
        return owner, max(0, int(garrison) + survivor)
    if survivor > garrison:
        return atk_owner, survivor - int(garrison)
    return owner, int(garrison) - survivor

