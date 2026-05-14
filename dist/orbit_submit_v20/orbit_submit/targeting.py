"""v20 capture heuristics: ``target_score`` + ``regional_capture_adjustment`` (registry hooks)."""

from __future__ import annotations

import math
from typing import Tuple

from orbit_submit.constants import HORIZON_TURNS, SUN_PATH_MARGIN, SUN_RADIUS, SUN_X, SUN_Y
from orbit_submit.entities import Planet
from orbit_submit.game_state import GameState
from orbit_submit.kinematics import (
    capture_need,
    is_sun_belt_planet,
    my_inbound_ships_to,
    point_segment_distance,
    target_state_at,
)
from orbit_submit.regional import RegionalGraph
from orbit_submit.scoring_early import enemy_eta_power
from orbit_submit.scoring_shared import approach_bonus, orbit_arc_strategic_score, recapture_bonus
from orbit_submit.snapshot import Snapshot


def target_score(snap: Snapshot, src: Planet, dst: Planet) -> Tuple[float, int, int]:
    """Score (score, need, eta) — v20: distance-dominant scoring (v19 lineage).

    核心原则（模仿 top 方案）：
    - 近处目标天然占优（eta 是强惩罚因子）
    - 高产值星球有额外加成
    - 路径穿越太阳的目标直接判死（不调 safe_aim，纯几何检查）
    - 不跨越太阳去打远处目标
    """
    state = snap.state
    if dst.owner == state.my_id or src.id == dst.id:
        return -1e18, 0, 0

    need, eta = capture_need(state, src, dst)
    if need <= 0:
        return -1e18, 0, eta

    sun_detour_pen = 0.0
    sun_dist = point_segment_distance(SUN_X, SUN_Y, src.x, src.y, dst.x, dst.y)
    if sun_dist < SUN_RADIUS + SUN_PATH_MARGIN:
        sun_detour_pen = 42.0 + eta * 0.30

    raw_turns = max(1, state.turns_left() - eta)
    turns = min(raw_turns, HORIZON_TURNS)
    if dst.is_comet:
        turns = min(turns, max(0, state.comet_turns_left(dst) - eta), 60)
        if turns <= 8:
            return -1e18, need, eta

    is_neu = dst.owner == -1
    is_en = dst.owner not in (-1, state.my_id)

    if is_neu and dst.production > 0 and dst.ships > dst.production:
        e_eta_first, _ = enemy_eta_power(state, dst)
        if 0 < e_eta_first < eta:
            owner_after, ships_after = target_state_at(state, dst, e_eta_first + 1)
            if owner_after not in (-1, state.my_id):
                snipe_eta = max(eta, e_eta_first + 1)
                snipe_need = ships_after + 8 + min(6, dst.production)
                snipe_need += dst.production * snipe_eta // 5
                if snipe_need > need:
                    need = max(need, snipe_need)
                    eta = snipe_eta

    prod_value = dst.production * min(turns, 30)

    prod_bonus = 0.0
    if dst.production >= 5:
        prod_bonus = 40.0 + dst.production * 5.0
    elif dst.production >= 3:
        prod_bonus = 15.0 + dst.production * 3.0
    elif dst.production >= 1:
        prod_bonus = dst.production * 2.0

    enemy_bonus = 30.0 if is_en else 0.0
    if is_en and src.dist(dst) < 20.0:
        enemy_bonus += 35.0
    comet_bonus = 12.0 if dst.is_comet else 0.0
    rec_bonus = recapture_bonus(snap, dst)
    early_hot_neutral = (
        22.0 if (is_neu and state.phase() == "early" and dst.production >= 4) else 0.0
    )
    neutral_mfg = 0.0
    if is_neu:
        neutral_mfg = 48.0 * float(dst.production * dst.production) / max(1.0, float(need))

    distance_decay = 1.0 / (1.0 + eta * 0.10)

    cost_pen = 0.0
    if eta > 3:
        cost_mul = snap.policy.cost_pen_neutral_mul if is_neu else snap.policy.cost_pen_mul
        cost_pen = cost_mul * need

    snipe_pen = 0.0
    if is_neu:
        e_eta, e_pow = enemy_eta_power(state, dst)
        if e_eta <= eta + 1 and e_pow > max(0, need - 4):
            snipe_pen = 30.0
        elif e_eta <= eta + 2 and e_pow > need + 5:
            snipe_pen = 15.0

    mite_neutral_pen = 0.0
    if is_neu and state.phase() == "early" and dst.production <= 1:
        mite_neutral_pen = 38.0

    orbit_arc = 0.0
    approach_adj = 0.0
    fat_local_neu = 0.0
    finish_neu = 0.0
    if is_neu:
        if state.en_pl:
            orbit_arc = orbit_arc_strategic_score(state, dst, eta)
            approach_adj = 0.48 * approach_bonus(snap, dst, eta)
        if dst.ships >= 38:
            d_anchor = min(dst.dist(m) for m in state.my_pl)
            if d_anchor < 36.0:
                fat_local_neu = 16.0 + (36.0 - d_anchor) * 1.65 + min(22.0, dst.ships * 0.08)
        oth = my_inbound_ships_to(state, dst.id)
        if oth > 0 and oth <= dst.ships + 4:
            gap = (dst.ships + 1) - oth
            if 1 <= gap <= 18:
                finish_neu = 52.0 + (18 - gap) * 1.5

    score = (
        prod_value + prod_bonus + enemy_bonus + comet_bonus + rec_bonus
        + early_hot_neutral + neutral_mfg + fat_local_neu + finish_neu
        + orbit_arc + approach_adj
    ) * distance_decay
    score -= cost_pen + snipe_pen + mite_neutral_pen + sun_detour_pen
    if is_neu and is_sun_belt_planet(state, dst):
        score += 38.0 * distance_decay
    return score, need, eta


def regional_capture_adjustment(
    snap: Snapshot,
    src: Planet,
    dst: Planet,
    regional_graph: RegionalGraph,
    eta: int,
) -> float:
    """Additive regional layer: cohesion bonus or cross-zone expedition tax."""
    state = snap.state
    my_ids = {p.id for p in state.my_pl}
    rid_s = regional_graph.planet_to_region.get(src.id, -1)
    rid_d = regional_graph.planet_to_region.get(dst.id, -1)
    if rid_s < 0 or rid_d < 0:
        return 0.0

    ddist, _ = regional_graph.dijkstra(src.id, dst.id)
    dist_cost = 0.12 * min(ddist, 80.0)

    if rid_s == rid_d:
        my_prod = regional_graph.region_production(rid_d, my_ids)
        bonus = 8.0 + min(22.0, float(my_prod) * 1.55)
        pot = 0.0
        for p in regional_graph.get_planets_in_region(rid_d):
            if p.id == dst.id:
                continue
            if p.owner == -1:
                pot += float(p.production) + 0.5
            elif p.owner in state.en_ids:
                eg = state.effective_garrison(p)
                if eg < p.ships * 0.65:
                    pot += float(p.production) * 0.6
        bonus += min(16.0, pot * 1.8)
        return bonus - dist_cost * 0.35

    cross = 10.0 + 0.045 * float(eta * eta)
    if is_sun_belt_planet(state, dst) and dst.owner == -1:
        cross *= 0.38
    return -cross - dist_cost
