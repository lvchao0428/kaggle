"""Fixed-size vectors for RL (macro policies)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from orbit_wars_bot.geom import SUN_X, SUN_Y

MY_PLANET_SLOTS = 6
FEATS_PER_PLANET = 6
GLOBAL_DIM = 9
OBS_DIM = GLOBAL_DIM + MY_PLANET_SLOTS * FEATS_PER_PLANET


def _get(obs: Any, key: str, default=None):
    if hasattr(obs, key):
        return getattr(obs, key)
    if isinstance(obs, dict):
        return obs.get(key, default)
    return default


def featurize(obs: Any, *, max_ship_hint: float = 5000.0) -> np.ndarray:
    player = int(_get(obs, "player", 0) or 0)
    step = int(_get(obs, "step", 0) or 0)
    av = float(_get(obs, "angular_velocity", 0.0) or 0.0)
    planets = _get(obs, "planets", []) or []
    fleets = _get(obs, "fleets", []) or []

    my_p, en_ship, my_ship, neu_ship = [], 0, 0, 0
    my_prod = 0
    for row in planets:
        oid = int(row[1])
        sh = int(row[5])
        pr = int(row[6])
        if oid == player:
            my_p.append(row)
            my_ship += sh
            my_prod += pr
        elif oid == -1:
            neu_ship += sh
        else:
            en_ship += sh

    my_fleet = sum(int(f[6]) for f in fleets if int(f[1]) == player)
    en_fleet = sum(int(f[6]) for f in fleets if int(f[1]) != player and int(f[1]) != -1)

    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    g[0] = step / 500.0
    g[1] = av
    g[2] = math.log1p(len(my_p))
    g[3] = my_ship / max_ship_hint
    g[4] = en_ship / max_ship_hint
    g[5] = neu_ship / max_ship_hint
    g[6] = my_fleet / max_ship_hint
    g[7] = en_fleet / max_ship_hint
    g[8] = my_prod / 25.0

    my_p.sort(key=lambda r: -int(r[6]) * 5 - int(r[5]))
    slots = np.zeros(MY_PLANET_SLOTS * FEATS_PER_PLANET, dtype=np.float32)
    for i, row in enumerate(my_p[:MY_PLANET_SLOTS]):
        off = i * FEATS_PER_PLANET
        x, y = float(row[2]), float(row[3])
        slots[off + 0] = x / 100.0
        slots[off + 1] = y / 100.0
        slots[off + 2] = float(row[4]) / 10.0
        slots[off + 3] = float(row[5]) / max_ship_hint
        slots[off + 4] = float(row[6]) / 5.0
        slots[off + 5] = math.hypot(x - SUN_X, y - SUN_Y) / 70.0

    return np.concatenate([g, slots], axis=0).astype(np.float32)


__all__ = ["OBS_DIM", "featurize"]
