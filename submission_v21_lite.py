"""Orbit Wars v21_lite — thin entry: registry hooks + orbit_submit package."""

from __future__ import annotations

import math
import time
from typing import Tuple

import orbit_submit.registry as registry
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
from orbit_submit.scoring_shared import (
    approach_bonus,
    orbit_arc_strategic_score,
    recapture_bonus,
)
from orbit_submit.snapshot import Snapshot


_NEURAL_WEIGHTS_B64 = "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"  # run tools/distill_to_numpy_v21.py to fill


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

    # 球心连线擦过太阳：启发式上偏难，但弹道可从边缘绕行（见 _emit / safe_aim）。
    # 旧逻辑直接 -1e18 使内环球长期进不了排序；改为扣分，交给真轨迹门。
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

    # Snipe-aware sizing for neutrals
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

    # === 打分（distance-dominant） ===

    # 产值回报：占据后每回合收益 × 收益轮数，但用 min(turns, 30) 压缩远期
    prod_value = dst.production * min(turns, 30)

    # 高产加成：prod>=5 的星球是战略目标
    prod_bonus = 0.0
    if dst.production >= 5:
        prod_bonus = 40.0 + dst.production * 5.0
    elif dst.production >= 3:
        prod_bonus = 15.0 + dst.production * 3.0
    elif dst.production >= 1:
        prod_bonus = dst.production * 2.0

    # 敌方星球额外价值（夺取=削弱对手+增强自己）；贴身弱敌强推（expand 曾漏掉 >20 驻军的邻球）
    enemy_bonus = 30.0 if is_en else 0.0
    if is_en and src.dist(dst) < 20.0:
        enemy_bonus += 35.0
    # FFA 开局：咫尺可吞的弱敌星（同产、兵少）提高排序，避免只顾闷发展
    if (
        is_en
        and state.is_ffa_mode()
        and state.phase() == "early"
    ):
        dm = min((m.dist(dst) for m in state.my_pl), default=999.0)
        if dm <= 44.0:
            eg = float(state.effective_garrison(dst))
            if eg <= 58.0:
                enemy_bonus += 22.0 + max(0.0, 44.0 - dm) * 0.9

    # 终局/贴身「蚊子敌星」（1 产、驻防极低）：清场占点，避免优势局仍不打 +1
    finish_weak_enemy = 0.0
    if is_en:
        eg = float(state.effective_garrison(dst))
        dm_all = min((m.dist(dst) for m in state.my_pl), default=999.0)
        if dm_all < 56.0:
            n_en_worlds = len(state.en_pl)
            if n_en_worlds == 1:
                finish_weak_enemy += (
                    26.0
                    + max(0.0, 55.0 - eg) * 0.42
                    + max(0.0, 52.0 - dm_all) * 0.38
                )
            if dst.production <= 1 and eg <= 36.0:
                finish_weak_enemy = max(
                    finish_weak_enemy,
                    32.0
                    + max(0.0, 32.0 - eg) * 0.95
                    + max(0.0, 48.0 - dm_all) * 0.62,
                )
            elif dst.production <= 2 and eg <= 12.0 and dm_all < 36.0:
                finish_weak_enemy = max(
                    finish_weak_enemy,
                    20.0 + max(0.0, 36.0 - dm_all) * 0.45,
                )
    comet_bonus = 12.0 if dst.is_comet else 0.0
    rec_bonus = recapture_bonus(snap, dst)
    # Early race for big factories: nudge expand ordering toward prod>=4 neutrals.
    early_hot_neutral = 0.0
    if is_neu and state.phase() == "early" and dst.production >= 4:
        early_hot_neutral = 22.0
        if state.is_ffa_mode():
            early_hot_neutral += 20.0
    # Value-per-commitment: prod^2 / need  favors factories over low-prod mites when
    # distances are similar (common case: 20@prod4 vs 14@prod1).
    neutral_mfg = 0.0
    if is_neu:
        neutral_mfg = 48.0 * float(dst.production * dst.production) / max(1.0, float(need))

    # 距离惩罚：eta 的 **强** 衰减——这是与旧版最大的区别
    # eta=5 → 0.67, eta=10 → 0.50, eta=20 → 0.37, eta=30 → 0.31
    distance_decay = 1.0 / (1.0 + eta * 0.10)
    # FFA：进一步惩罚「远征」，保住出生弧附近早占厂、早产兵节奏
    if state.is_ffa_mode():
        ph = state.phase()
        if ph == "early":
            distance_decay *= 1.0 / (1.0 + max(0, eta - 8) * 0.078)
        elif ph == "mid":
            distance_decay *= 1.0 / (1.0 + max(0, eta - 15) * 0.045)

    # 兵力成本：需要的兵越多，性价比越低
    cost_pen = 0.0
    if eta > 3:
        cost_mul = snap.policy.cost_pen_neutral_mul if is_neu else snap.policy.cost_pen_mul
        cost_pen = cost_mul * need

    # Sniping risk
    snipe_pen = 0.0
    if is_neu:
        e_eta, e_pow = enemy_eta_power(state, dst)
        if e_eta <= eta + 1 and e_pow > max(0, need - 4):
            snipe_pen = 30.0
        elif e_eta <= eta + 2 and e_pow > need + 5:
            snipe_pen = 15.0

    # 1 产「蚊子球」：前期压低「远处」排序；贴身/终局不罚，否则会剩 +1 不碰。
    mite_neutral_pen = 0.0
    if is_neu and state.phase() == "early" and dst.production <= 1:
        d_anchor = min((m.dist(dst) for m in state.my_pl), default=999.0)
        if d_anchor >= 46.0:
            mite_neutral_pen = 38.0

    opening_neutral_nudge = 0.0
    if is_neu and state.phase() == "early" and dst.production >= 4 and dst.ships >= 14:
        d_anchor = min((m.dist(dst) for m in state.my_pl), default=999.0)
        if d_anchor < 52.0:
            opening_neutral_nudge = (
                (52.0 - d_anchor) * 0.88
                + float(dst.production) * 5.2
                + min(18.0, max(0.0, float(dst.ships) - 12.0) * 0.22)
            )
            if state.is_ffa_mode():
                opening_neutral_nudge *= 1.2

    # FFA：2–3 产近邻灰也是肉（早占累计产能 + 内环球接力），避免只盯远征
    # 注意：FFA phase 在 progress≳0.36 就进 late，短局 ~80t 起会关掉 early/mid；
    # 若这里也随 phase 关掉，右下「口袋」灰会一直抢不过远征排序。
    mosquito_relay = 0.0
    if is_neu:
        d_an = min((m.dist(dst) for m in state.my_pl), default=999.0)
        if state.is_ffa_mode() and d_an < 52.0 and 2 <= dst.production <= 3:
            ph = state.phase()
            relay_scale = 1.0 if ph != "late" else 0.82
            mosquito_relay = relay_scale * (
                14.0 + (52.0 - d_an) * 0.72 + float(dst.production) * 8.0
            )
            if is_sun_belt_planet(state, dst):
                mosquito_relay += 24.0 * relay_scale
        elif d_an < 46.0 and dst.production == 1 and state.phase() != "early":
            # 中后盘贴身 1 产灰（1v1）：收口占点
            mosquito_relay = 10.0 + (46.0 - d_an) * 0.55
            if state.is_ffa_mode():
                mosquito_relay *= 1.08

    local_expedition_pen = 0.0
    thr_eta = 10 if dst.production >= 3 else 12
    if state.is_ffa_mode() and (is_neu or is_en) and eta > thr_eta:
        dm = min((m.dist(dst) for m in state.my_pl), default=999.0)
        if dm > 44.0:
            idle_n = sum(
                1
                for n in state.neu_pl
                if n.production >= 2
                and min((m.dist(n) for m in state.my_pl), default=999.0) < 56.0
            )
            if idle_n >= 1:
                eta0 = thr_eta - 2
                local_expedition_pen = min(
                    54.0,
                    max(0, eta - eta0)
                    * math.sqrt(float(idle_n))
                    * (1.38 if dst.production >= 3 else 1.12),
                )

    # FFA：己方弧远 + 强敌弧包住目标大肉 → 压低排序（与投资门一起防白送）
    ffa_far_contested_pen = 0.0
    if (
        is_neu
        and state.is_ffa_mode()
        and state.en_pl
        and eta >= 10
        and dst.production >= 2
    ):
        my_dm = min((m.dist(dst) for m in state.my_pl), default=999.0)
        en_dm = min((e.dist(dst) for e in state.en_pl), default=999.0)
        if my_dm >= 38.5 and en_dm + 13.8 < my_dm:
            en_ring = sum(
                float(ep.ships)
                for ep in state.en_pl
                if ep.dist(dst) < min(68.0, my_dm + 22.0)
            )
            e_eta_best, e_src_ships = enemy_eta_power(state, dst)
            need_f = float(need)
            hold_f = float(dst.ships + dst.production * min(max(int(eta), 4), 20))
            loser_race = (
                en_ring >= need_f * 0.80
                or en_ring >= hold_f * 0.88
                or (
                    0 < int(e_eta_best) <= int(eta) + 4
                    and en_ring >= need_f * 0.72
                    and float(e_src_ships) >= need_f * 0.11
                )
            )
            if loser_race:
                gap = max(0.0, my_dm - en_dm)
                ffa_far_contested_pen = min(
                    118.0,
                    38.5
                    + gap * 0.68
                    + max(0, int(eta) - 10) * 2.2
                    + float(dst.production) * 7.8
                    + min(42.0, max(0.0, en_ring - need_f) * 0.09),
                )

    orbit_arc = 0.0
    approach_adj = 0.0
    fat_local_neu = 0.0
    finish_neu = 0.0
    orbiting_tgt = (
        not dst.is_comet and state.is_orbiting(dst) and bool(state.my_pl)
    )

    if is_neu:
        if state.en_pl:
            orbit_arc = orbit_arc_strategic_score(state, dst, eta)
            approach_adj = 0.48 * approach_bonus(snap, dst, eta)
        if dst.ships >= 38:
            d_anchor = min(dst.dist(m) for m in state.my_pl)
            if d_anchor < 36.0:
                # 身边大灰（如 59）：优先吃近处高驻军工厂，少去追「路过」小灰
                fat_local_neu = (
                    16.0 + (36.0 - d_anchor) * 1.65
                    + min(22.0, dst.ships * 0.08)
                )
        elif (
            state.is_ffa_mode()
            and state.phase() == "early"
            and dst.production >= 4
            and dst.ships >= 18
        ):
            d_anchor = min((m.dist(dst) for m in state.my_pl), default=999.0)
            if d_anchor < 44.0:
                fat_local_neu = (
                    9.0 + (44.0 - d_anchor) * 1.15
                    + min(14.0, (dst.ships - 16) * 0.07)
                )
        oth = my_inbound_ships_to(state, dst.id)
        if oth > 0 and oth <= dst.ships + 4:
            # 已有己方舰队在途但未够占领：优先补刀，避免下回合改打远处浪费产兵
            gap = (dst.ships + 1) - oth
            if 1 <= gap <= 18:
                finish_neu = 52.0 + (18 - gap) * 1.5

    elif is_en and orbiting_tgt and state.en_pl:
        # 旋转图：敌对星球同样吃「我方弧 / 敌方弧」漂移（原先仅中立算）
        orbit_arc = orbit_arc_strategic_score(state, dst, eta)
        approach_adj = 0.82 * approach_bonus(snap, dst, eta)

    score = (
        prod_value + prod_bonus + enemy_bonus + finish_weak_enemy
        + comet_bonus + rec_bonus
        + early_hot_neutral + opening_neutral_nudge + mosquito_relay
        + neutral_mfg + fat_local_neu + finish_neu
        + orbit_arc + approach_adj
    ) * distance_decay
    score -= (
        cost_pen + snipe_pen + mite_neutral_pen + sun_detour_pen
        + local_expedition_pen
        + ffa_far_contested_pen
    )
    # 内环球中立：战略优先级（旋转进对手弧前须占）
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

    # FFA：出生弧附近 2–4 产口袋灰若被划到「邻区」，远征税会把排序打到远征后面
    if (
        state.is_ffa_mode()
        and dst.owner == -1
        and 2 <= dst.production <= 4
        and state.my_pl
    ):
        frontier = min(m.dist(dst) for m in state.my_pl)
        if frontier < 46.0:
            dist_cost *= 0.55

    cross = 10.0 + 0.045 * float(eta * eta)
    if is_sun_belt_planet(state, dst) and dst.owner == -1:
        cross *= 0.38
    if state.is_ffa_mode() and dst.production >= 5 and state.my_pl:
        frontier = min(m.dist(dst) for m in state.my_pl)
        if frontier < 32.0:
            cross *= 0.46
    if (
        state.is_ffa_mode()
        and state.phase() == "early"
        and rid_s != rid_d
    ):
        cross *= 1.0 + max(0.0, float(eta - 11)) * 0.058

    if (
        state.is_ffa_mode()
        and dst.owner == -1
        and 2 <= dst.production <= 4
        and state.my_pl
    ):
        frontier = min(m.dist(dst) for m in state.my_pl)
        if frontier < 46.0:
            cross *= 0.42
    return -cross - dist_cost


registry.target_score = target_score
registry.regional_capture_adjustment = regional_capture_adjustment
registry.neural_weights_b64 = _NEURAL_WEIGHTS_B64
registry.arbiter_variant = "v21"

from orbit_submit.engine import DiplomacyEngine, OpponentModel, PlanArbiter
from orbit_submit.neural import NeuralVal
from orbit_submit.policy import PhasePolicy
from orbit_submit.regional import MultiHopPlanner, ProductionTimeline, RegionalGraph as _RG

_GLOBAL_OPP = OpponentModel()
_GLOBAL_NEURAL = NeuralVal()


def agent(obs, config=None):
    """Kaggle-required entry. Returns list of [src_id, angle, ships] moves."""
    global _GLOBAL_OPP, _GLOBAL_NEURAL
    t0 = time.time()
    elapsed = lambda: (time.time() - t0) * 1000.0

    try:
        state = GameState(obs, config, ruleset="v21")
        if not state.my_pl:
            return []

        _GLOBAL_OPP.update(state)
        policy = PhasePolicy.for_state(state)
        snap = Snapshot.build(state, policy)
        diplo = DiplomacyEngine(state, _GLOBAL_OPP)

        regional_graph = None
        multi_hop_planner = None
        try:
            spawn_positions = config.get("spawn_positions", []) if config else []
            regional_graph = _RG(state.planets, spawn_positions)
            timeline = ProductionTimeline(state.planets, set(p.id for p in state.my_pl))
            multi_hop_planner = MultiHopPlanner(regional_graph, timeline)
        except Exception:
            regional_graph = None
            multi_hop_planner = None

        arbiter = PlanArbiter(
            snap,
            diplo,
            _GLOBAL_NEURAL,
            elapsed_ms_fn=elapsed,
            deadline_ms=920.0,
            regional_graph=regional_graph,
            multi_hop_planner=multi_hop_planner,
        )

        arbiter.commit_urgent()
        plans = arbiter.collect_strategic()
        scored = arbiter.score_with_modifiers(plans)
        arbiter.commit_best(scored)
        arbiter.commit_fallback()

        return arbiter.moves
    except Exception:
        return []
