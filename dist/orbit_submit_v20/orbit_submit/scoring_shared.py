from __future__ import annotations

from orbit_submit.constants import *
from orbit_submit.entities import Planet
from orbit_submit.game_state import GameState
from orbit_submit.snapshot import Snapshot

def approach_bonus(snap: Snapshot, dst: Planet, eta: int) -> float:
    if not snap.state.is_orbiting(dst):
        return 0.0
    cx, cy = snap.centroid
    now_d = math.hypot(dst.x - cx, dst.y - cy)
    fx, fy = snap.state.planet_pos_at(dst, eta)
    fut_d = math.hypot(fx - cx, fy - cy)
    gain = now_d - fut_d
    return max(-28.0, min(34.0, gain * snap.policy.approach_weight + dst.production * 1.2))


def orbit_arc_strategic_score(state: GameState, dst: Planet, eta: int) -> float:
    """Prefer targets that, by arrival time, drift toward our mass / away from enemy.

    Penalizes chasing passing neutrals that rotate into the opponent's arc (user:
    「占了也快进对面半场 = 白给」).
    """
    if dst.is_comet or not state.is_orbiting(dst):
        return 0.0
    if not state.en_pl or not state.my_pl:
        return 0.0
    t = max(1, min(int(eta), 56))
    fx, fy = state.planet_pos_at(dst, t)
    now_d_my = min(dst.dist(m) for m in state.my_pl)
    fut_d_my = min(
        math.hypot(fx - state.planet_pos_at(m, t)[0], fy - state.planet_pos_at(m, t)[1])
        for m in state.my_pl
    )
    now_d_en = min(dst.dist(e) for e in state.en_pl)
    fut_d_en = min(
        math.hypot(fx - state.planet_pos_at(e, t)[0], fy - state.planet_pos_at(e, t)[1])
        for e in state.en_pl
    )
    # Positive: closing on our worlds. Positive toward_en_closing: closing on enemy (bad).
    toward_us = now_d_my - fut_d_my
    toward_en_closing = now_d_en - fut_d_en
    raw = 1.18 * toward_us - 1.52 * max(0.0, toward_en_closing)
    return max(-46.0, min(52.0, raw))


def enemy_eta_power(state: GameState, dst: Planet) -> Tuple[int, int]:
    best_eta, best_power = 999, 0
    for e in state.en_pl:
        probe = max(1, min(e.ships, max(5, e.ships * 2 // 3)))
        _, _, eta, _ = lead_intercept(state, e, dst, probe)
        if eta < best_eta:
            best_eta, best_power = eta, e.ships
    return best_eta, best_power


def recapture_bonus(snap: Snapshot, dst: Planet) -> float:
    if dst.owner not in snap.state.en_ids:
        return 0.0
    cx, cy = snap.centroid
    d = max(1.0, math.hypot(dst.x - cx, dst.y - cy))
    return dst.production * 14.0 / (1.0 + d * 0.04) * snap.policy.recapture_mul


def contest_penalty(state: GameState, dst: Planet) -> float:
    en_inc = state.enemy_incoming(dst.id)
    if en_inc <= 0:
        return 0.0
    return min(40.0, en_inc * 0.6)


def elite_eval(state: GameState) -> float:
    """Static positional eval - reserved for NeuralVal feature engineering."""
    mi = state.my_id
    ms = state.total_ships(mi)
    es = sum(state.total_ships(e) for e in state.en_ids) + 1e-9
    mp = sum(p.production for p in state.my_pl)
    ep = sum(p.production for p in state.en_pl)
    mc = len(state.my_pl)
    ec = len(state.en_pl)
    threat = sum(max(0, state.net_threat(p)) for p in state.my_pl)
    mf = sum(f.ships for f in state.fleets if f.owner == mi)
    ef = sum(f.ships for f in state.fleets if f.owner not in (-1, mi))
    border = sum((35 - m.dist(e)) / 35 * m.production
                 for m in state.my_pl for e in state.en_pl if m.dist(e) < 35)
    ndeny = sum(n.production for n in state.neu_pl
                if any(n.dist(e) < 25 for e in state.en_pl)
                and not any(n.dist(m) < 25 for m in state.my_pl))
    return ((ms - es) + 48.0 * (mp - ep) + 20.0 * (mc - ec)
            - 2.8 * threat + 9.0 * border + 0.45 * (mf - ef) - 12.0 * ndeny)

