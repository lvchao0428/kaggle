from __future__ import annotations

from orbit_submit.constants import *
from orbit_submit.entities import Planet
from orbit_submit.game_state import GameState
from orbit_submit.kinematics import lead_intercept

def enemy_eta_power(state: GameState, dst: Planet) -> Tuple[int, int]:
    best_eta, best_power = 999, 0
    for e in state.en_pl:
        probe = max(1, min(e.ships, max(5, e.ships * 2 // 3)))
        _, _, eta, _ = lead_intercept(state, e, dst, probe)
        if eta < best_eta:
            best_eta, best_power = eta, e.ships
    return best_eta, best_power
