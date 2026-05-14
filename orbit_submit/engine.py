from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from orbit_submit.constants import *
from orbit_submit.entities import Planet, _combat
from orbit_submit.game_state import GameState
from orbit_submit.kinematics import (
    _ray_safe,
    capture_need,
    fleet_speed,
    is_sun_belt_planet,
    launch_hits_target_first,
    launch_origin,
    lead_intercept,
    my_inbound_ships_to,
    neutral_wave_wins,
    point_segment_distance,
    safe_aim,
    segment_hits_sun,
    swept_pair_hit,
    target_state_at,
)
from orbit_submit.policy import PhasePolicy
from orbit_submit.regional import MultiHopPlanner, RegionalGraph, calculate_safe_surplus
from orbit_submit.scoring_early import enemy_eta_power
from orbit_submit.scoring_shared import (
    approach_bonus,
    contest_penalty,
    orbit_arc_strategic_score,
    recapture_bonus,
)
from orbit_submit.snapshot import Snapshot
import orbit_submit.registry as registry
from orbit_submit.neural import NeuralVal


def capture_edge_score(
    snap: Snapshot,
    src: Planet,
    dst: Planet,
    regional_graph: Optional[RegionalGraph] = None,
) -> Tuple[float, int, int]:
    """Unified capture ranking: submission ``registry.target_score`` + optional regional.

    ``orbit_submit.agent`` (and thin repo entries such as ``submission_v20.py``)
    register ``registry.target_score`` / ``regional_capture_adjustment`` before
    planners call this. Keeps a single scoring entry for expand/aggro/counter planners.
    """
    fn = registry.target_score
    if fn is None:
        return -1e18, 0, 0
    base, need, eta = fn(snap, src, dst)
    if regional_graph is None or base <= -1e17:
        return base, need, eta
    adj_fn = registry.regional_capture_adjustment
    if adj_fn is None:
        return base, need, eta
    adj = adj_fn(snap, src, dst, regional_graph, eta)
    return base + adj, need, eta


def target_value_in_region(
    snap: Snapshot,
    src: Planet,
    dst: Planet,
    regional_graph: Optional[RegionalGraph] = None,
) -> float:
    """Same scalar as ``capture_edge_score`` first component (compat / tooling)."""
    sc, _, _ = capture_edge_score(snap, src, dst, regional_graph)
    return sc


# ╔═══ region 6: SimP / SimF / sim_step (forward simulator) ═════════════════╗

class SimP:
    __slots__ = ("id", "owner", "ships", "production")

    def __init__(self, p):
        self.id = p.id; self.owner = p.owner
        self.ships = p.ships; self.production = p.production

    def copy(self) -> "SimP":
        s = SimP.__new__(SimP)
        s.id = self.id; s.owner = self.owner
        s.ships = self.ships; s.production = self.production
        return s


class SimF:
    __slots__ = ("owner", "tid", "ships", "eta")

    def __init__(self, owner, tid, ships, eta):
        self.owner = owner; self.tid = tid; self.ships = ships; self.eta = eta

    def copy(self) -> "SimF":
        return SimF(self.owner, self.tid, self.ships, self.eta)


def clone_sim(state: GameState) -> Tuple[Dict[int, SimP], List[SimF]]:
    planets = {p.id: SimP(p) for p in state.planets}
    fleets: List[SimF] = []
    for f in state.fleets:
        t = state.fleet_target.get(f.id)
        if t:
            fleets.append(SimF(f.owner, t[0], f.ships, t[1]))
    return planets, fleets


def copy_sim(planets: Dict[int, SimP], fleets: List[SimF]):
    return ({pid: p.copy() for pid, p in planets.items()},
            [f.copy() for f in fleets])


def sim_step(planets: Dict[int, SimP], fleets: List[SimF]) -> None:
    for p in planets.values():
        if p.owner >= 0:
            p.ships += p.production
    by_target: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    nxt: List[SimF] = []
    for f in fleets:
        f.eta -= 1
        if f.eta <= 0:
            by_target[f.tid].append((f.owner, f.ships))
        else:
            nxt.append(f)
    for tid, arrivals in by_target.items():
        p = planets.get(tid)
        if p:
            p.owner, p.ships = _combat(p.owner, p.ships, arrivals)
    fleets[:] = nxt


def eval_sim_planets(state: GameState, planets: Dict[int, SimP], fleets: List[SimF]) -> float:
    mi = state.my_id
    ms = sum(p.ships for p in planets.values() if p.owner == mi)
    ms += sum(f.ships for f in fleets if f.owner == mi)
    es = sum(p.ships for p in planets.values() if p.owner not in (-1, mi))
    es += sum(f.ships for f in fleets if f.owner not in (-1, mi))
    mp = sum(p.production for p in planets.values() if p.owner == mi)
    ep = sum(p.production for p in planets.values() if p.owner not in (-1, mi))
    mc = sum(1 for p in planets.values() if p.owner == mi)
    ec = sum(1 for p in planets.values() if p.owner not in (-1, mi))
    return (ms - es) + 48.0 * (mp - ep) + 20.0 * (mc - ec)


# ╔═══ region 7: Plan + planners ════════════════════════════════════════════╗

@dataclass
class Plan:
    actions: List[Tuple[int, int, int]] = field(default_factory=list)
    score: float = 0.0
    tag: str = ""
    urgent: bool = False  # if True, commit before scoring strategic plans


def _prep_sim_own_launches(state: GameState, actions: List[Tuple[int, int, int]],
                           tempo_floor: int) -> Tuple[Dict[int, SimP], List[SimF]]:
    """Clone + apply our launches + tempo_floor idle sim (before eval window)."""
    planets, fleets = clone_sim(state)
    used: Dict[int, int] = defaultdict(int)
    for sid, did, ships in actions:
        sp = state.get(sid); dp = state.get(did); sim_src = planets.get(sid)
        if sp is None or dp is None or sim_src is None or sim_src.owner != state.my_id:
            continue
        send = min(int(ships), max(0, sim_src.ships - ABS_MIN_BATCH - used[sid]))
        if send <= 0:
            continue
        _, eta = safe_aim(state, sp, dp, send)
        sim_src.ships -= send
        used[sid] += send
        fleets.append(SimF(state.my_id, did, send, eta))
    for _ in range(max(0, tempo_floor - 1)):
        sim_step(planets, fleets)
    return planets, fleets


def _rollout_eval(state: GameState, planets: Dict[int, SimP], fleets: List[SimF],
                  steps: int) -> float:
    p, f = copy_sim(planets, fleets)
    for _ in range(steps):
        sim_step(p, f)
    return eval_sim_planets(state, p, f)


def score_plan_actions(state: GameState, actions: List[Tuple[int, int, int]],
                       steps: int = 8, tempo_floor: int = 1) -> float:
    """Score by simulating `steps` turns after applying actions to a clone.

    `tempo_floor` (Bocsimacko `min-turn-to-depart`): run this many *idle*
    sim steps BEFORE the regular eval window, simulating no follow-up fleets
    yet. Plans that depend on chained reinforcements that we have not planned
    for in `actions` get penalised because their targets do not materialise
    within the eval window without those follow-ups. tempo_floor=1 means
    evaluate as-is.
    """
    planets, fleets = _prep_sim_own_launches(state, actions, tempo_floor)
    return _rollout_eval(state, planets, fleets, steps)


def _primary_enemy_id(state: GameState) -> Optional[int]:
    if not state.en_ids:
        return None
    return max(state.en_ids, key=lambda e: state.total_ships(e))


def _try_append_enemy_fleet(
    state: GameState,
    planets: Dict[int, SimP],
    fleets: List[SimF],
    enemy_id: int,
    src_id: int,
    dst_id: int,
    want_ships: int,
) -> bool:
    sp = planets.get(src_id)
    dp = planets.get(dst_id)
    geo_s = state.get(src_id)
    geo_d = state.get(dst_id)
    if sp is None or dp is None or geo_s is None or geo_d is None:
        return False
    if sp.owner != enemy_id:
        return False
    max_send = max(0, sp.ships - ABS_MIN_BATCH)
    send = min(max(int(want_ships), ABS_MIN_BATCH), max_send)
    if send < ABS_MIN_BATCH:
        return False
    _, eta = safe_aim(state, geo_s, geo_d, send)
    sp.ships -= send
    fleets.append(SimF(enemy_id, dst_id, send, eta))
    return True


def _inject_paranoid_profile(
    state: GameState,
    planets: Dict[int, SimP],
    fleets: List[SimF],
    enemy_id: int,
    profile_key: str,
) -> None:
    """Mutate planets/fleets with one pessimistic opponent launch."""
    mi = state.my_id
    enemy_pids = [pid for pid, sp in planets.items()
                  if sp.owner == enemy_id and sp.ships >= ABS_MIN_BATCH * 2]
    my_pids = [pid for pid, sp in planets.items() if sp.owner == mi]
    neu_pids = [pid for pid, sp in planets.items() if sp.owner == -1]
    cx, cy = state.centroid()

    want = ABS_MIN_BATCH * 8

    if profile_key == "nearest_my":
        best = None
        for eid in enemy_pids:
            ge = state.get(eid)
            if ge is None:
                continue
            for mid in my_pids:
                gm = state.get(mid)
                if gm is None:
                    continue
                d = ge.dist(gm)
                if best is None or d < best[0]:
                    best = (d, eid, mid)
        if best:
            _, eid, mid = best
            _try_append_enemy_fleet(state, planets, fleets, enemy_id, eid, mid,
                                    want)

    elif profile_key == "contest_neutral" and neu_pids:
        neu_pids_sorted = sorted(
            neu_pids,
            key=lambda nid: math.hypot(state.get(nid).x - cx, state.get(nid).y - cy)
            if state.get(nid) else 999)
        tgt = None
        for nid in neu_pids_sorted[:6]:
            gn = state.get(nid)
            if gn is None:
                continue
            best_ep = None
            for eid in enemy_pids:
                ge = state.get(eid)
                if ge is None:
                    continue
                d = ge.dist(gn)
                if best_ep is None or d < best_ep[0]:
                    best_ep = (d, eid)
            if best_ep:
                tgt = (best_ep[1], nid)
                break
        if tgt:
            _try_append_enemy_fleet(state, planets, fleets, enemy_id, tgt[0], tgt[1],
                                    want)

    elif profile_key == "strike_high_prod_my" and my_pids:
        mid = max(my_pids, key=lambda pid: state.get(pid).production
                   if state.get(pid) else 0)
        gm = state.get(mid)
        if gm is None:
            return
        best_ep = None
        for eid in enemy_pids:
            ge = state.get(eid)
            if ge is None:
                continue
            d = ge.dist(gm)
            if best_ep is None or d < best_ep[0]:
                best_ep = (d, eid)
        if best_ep:
            _try_append_enemy_fleet(state, planets, fleets, enemy_id,
                                    best_ep[1], mid,
                                    max(want, ABS_MIN_BATCH * 12))


def score_plan_actions_paranoid(
    state: GameState,
    actions: List[Tuple[int, int, int]],
    steps: int,
    tempo_floor: int,
    par_steps: Optional[int] = None,
) -> Tuple[float, float]:
    """Return ``(baseline_sim, pessimistic_sim)``

    Baseline ignores fresh opponent fleets; pessimistic inserts up to three
    single-launch pessimistic envelopes from the strongest enemy, then takes
    the worst ``eval_sim_planets`` among them (and the no-injection path).
    """
    if par_steps is None:
        par_steps = steps
    eid = _primary_enemy_id(state)
    planets0, fleets0 = _prep_sim_own_launches(state, actions, tempo_floor)

    baseline = _rollout_eval(state, planets0, fleets0, steps)
    if eid is None:
        return baseline, baseline

    worst = baseline
    for key in ("nearest_my", "contest_neutral", "strike_high_prod_my"):
        p, f = copy_sim(planets0, fleets0)
        _inject_paranoid_profile(state, p, f, eid, key)
        worst = min(worst, _rollout_eval(state, p, f, par_steps))

    return baseline, worst


def blended_paranoid_sim(
    state: GameState,
    actions: List[Tuple[int, int, int]],
    *,
    steps: int,
    tempo_floor: int,
    par_steps: int,
    blend: float,
) -> float:
    """``base + blend * (pessimistic - base)`` for arbiter blending."""
    b, pe = score_plan_actions_paranoid(state, actions, steps, tempo_floor,
                                        par_steps=par_steps)
    if blend <= 1e-9:
        return b
    return b + blend * (pe - b)


# ── CometEvacPlanner (friendly comets sailing off-map) ────────────────────────

class CometEvacPlanner:
    """Strip stacked ships off *our* comets onto stable planets.

    Ships left parked on departing comets are effectively lost once the rock
    leaves the arena. These moves use urgent commit so Snapshot hoard rules on
    comets cannot block the offload (see ``Snapshot._reserve`` for comets).

    Destination is always the **fastest intercept** among friendly non-comets
    (ETA from ``lead_intercept``, then production / frontline tie-break). When
    the comet is close to path end, ``PlanArbiter._emit`` relaxes the usual
    ``ABS_MIN_BATCH`` source floor so the last 1–7 ships can still launch.
    """

    @staticmethod
    def plan(snap: Snapshot) -> Plan:
        state = snap.state
        actions: List[Tuple[int, int, int]] = []
        score = 0.0
        local_used: Dict[int, int] = defaultdict(int)

        comets = [p for p in state.my_pl if p.is_comet]
        if not comets:
            return Plan([], 0.0, "comet_evac", urgent=True)

        docks = [m for m in state.my_pl if not m.is_comet]
        if not docks:
            return Plan([], 0.0, "comet_evac", urgent=True)

        def nearest_enemy_d(m: Planet) -> float:
            return min((m.dist(e) for e in state.en_pl), default=500.0)

        for c in sorted(comets, key=lambda p: state.comet_turns_left(p)):
            ttl = int(state.comet_turns_left(c))
            stack = int(c.ships)
            used_here = int(snap.used.get(c.id, 0)) + int(local_used.get(c.id, 0))
            liquid = max(0, stack - used_here)
            if liquid <= 0:
                continue
            if ttl >= 170 and liquid <= ABS_MIN_BATCH:
                continue

            # Near end of comet path: try to launch *everything* (incl. last chip).
            panic = (
                ttl <= 38
                or (ttl <= 72 and liquid <= 22)
                or (ttl <= 120 and liquid <= ABS_MIN_BATCH + 8)
            )
            if panic:
                evac = max(1, liquid)
            else:
                if liquid <= ABS_MIN_BATCH:
                    continue
                evac = min(liquid, max(ABS_MIN_BATCH, liquid - ABS_MIN_BATCH))

            probe = max(ABS_MIN_BATCH, min(evac, min(liquid, 220)))
            best_row: Optional[Tuple[int, int, float, Planet]] = None
            for m in docks:
                _, _, eta, _ = lead_intercept(state, c, m, probe)
                ned = nearest_enemy_d(m)
                row = (eta, int(-m.production), float(ned), m)
                if best_row is None or row[:3] < best_row[:3]:
                    best_row = row
            if best_row is None:
                continue
            dst = best_row[3]
            if dst.is_comet or dst.id == c.id:
                continue

            actions.append((c.id, dst.id, evac))
            local_used[c.id] += evac
            score += float(evac) * (8.2 if panic else 3.9)

        return Plan(actions, score, "comet_evac", urgent=True)


# ── DefensePlanner ───────────────────────────────────────────────────────────

class DefensePlanner:
    @staticmethod
    def plan(snap: Snapshot) -> Plan:
        state = snap.state
        actions: List[Tuple[int, int, int]] = []
        score = 0.0
        local_used: Dict[int, int] = defaultdict(int)

        # Combine reactive (net_threat > 0) and proactive (threat_horizon <= 8
        # on high-production planets) targets. Window expanded to 8 in v14.
        reactive = {p for p in state.my_pl if state.net_threat(p) > 0}
        proactive = {state.get(pid) for pid, eta in snap.threat_horizon.items()
                     if eta <= 8}
        proactive = {p for p in proactive if p is not None and p.production >= 3
                     and p not in reactive}

        covered = reactive | proactive
        # FFA: strong enemy worlds adjacent on the orbit arc (little/no inbound fleet) —
        # ``net_threat`` stays 0; inland factories should still reinforce the border.
        ffa_border_rows: List[Tuple[int, Planet]] = []
        if state.is_ffa_mode() and state.en_pl:
            ff_r = 62.0
            for p in state.my_pl:
                if p.is_comet or p in covered:
                    continue
                neighbors = [e for e in state.en_pl if p.dist(e) < ff_r]
                if not neighbors:
                    continue
                press = max(
                    float(state.effective_garrison(ne))
                    + float(ne.production) * 5.5
                    for ne in neighbors)
                ships_f = float(p.ships)
                if press >= ships_f + 8.0 or (
                    press >= 28.0 and press + 10.0 >= ships_f
                ):
                    gap_f = press * 0.54 - ships_f + 22.0
                    need_here = int(min(165, max(float(ABS_MIN_BATCH), gap_f)))
                    ffa_border_rows.append((need_here, p))
            ffa_border_rows.sort(key=lambda r: -r[0])

        for tgt in sorted(reactive, key=lambda p: -state.net_threat(p)):
            threat = state.net_threat(tgt)
            need = threat + max(5, tgt.production * 2)
            helpers = sorted((p for p in state.my_pl if p.id != tgt.id),
                             key=lambda p: p.dist(tgt))
            for src in helpers:
                if need <= 0:
                    break
                avail = max(0, snap.avail(src.id) - local_used[src.id])
                send = min(avail, need)
                if send < ABS_MIN_BATCH:
                    continue
                actions.append((src.id, tgt.id, send))
                local_used[src.id] += send
                need -= send
                score += send * 4.0

        for tgt in sorted(proactive, key=lambda p: -p.production):
            incoming = sum(
                f.ships for f in state.fleets
                if f.owner not in (-1, state.my_id)
                and (tft := state.fleet_target.get(f.id)) is not None
                and tft[0] == tgt.id)
            need = max(incoming - tgt.ships + tgt.production * 3, ABS_MIN_BATCH)
            helpers = sorted((p for p in state.my_pl if p.id != tgt.id),
                             key=lambda p: p.dist(tgt))
            for src in helpers:
                if need <= 0:
                    break
                avail = max(0, snap.avail(src.id) - local_used[src.id])
                send = min(avail, need)
                if send < ABS_MIN_BATCH:
                    continue
                actions.append((src.id, tgt.id, send))
                local_used[src.id] += send
                need -= send
                score += send * 4.0

        for need, tgt in ffa_border_rows:
            helpers = sorted((p for p in state.my_pl if p.id != tgt.id),
                             key=lambda p: p.dist(tgt))
            for src in helpers:
                if need <= 0:
                    break
                avail = max(0, snap.avail(src.id) - local_used[src.id])
                send = min(avail, need)
                if send < ABS_MIN_BATCH:
                    continue
                actions.append((src.id, tgt.id, send))
                local_used[src.id] += send
                need -= send
                score += send * 4.0

        return Plan(actions, score, "defense", urgent=True)


# ── InterceptPlanner ─────────────────────────────────────────────────────────

class InterceptPlanner:
    @staticmethod
    def plan(snap: Snapshot) -> Plan:
        state = snap.state
        actions: List[Tuple[int, int, int]] = []
        score = 0.0
        local_used: Dict[int, int] = defaultdict(int)

        for f in sorted(state.fleets, key=lambda x: -x.ships):
            if f.owner in (-1, state.my_id) or f.ships < 10:
                continue
            target = state.fleet_target.get(f.id)
            if not target:
                continue
            tid, eta_to_planet = target
            dst = state.get(tid)
            if dst is None or dst.owner != state.my_id:
                continue
            helpers = sorted((p for p in state.my_pl if p.id != dst.id),
                             key=lambda p: p.dist(dst))
            need = f.ships + 4
            for src in helpers[:4]:
                avail = max(0, snap.avail(src.id) - local_used[src.id])
                if avail < ABS_MIN_BATCH:
                    continue
                _, _, eta, _ = lead_intercept(state, src, dst, min(avail, need))
                if eta > eta_to_planet + 1:
                    continue
                send = min(avail, need)
                if send < ABS_MIN_BATCH:
                    continue
                actions.append((src.id, dst.id, send))
                local_used[src.id] += send
                need -= send
                score += send * 2.5
                if need <= 0:
                    break
        return Plan(actions, score, "intercept", urgent=True)


# ── Capture-style planner (shared by Expand and Attack) ──────────────────────

def _target_pool(state: GameState, mode: str,
                 diplo: Optional["DiplomacyEngine"] = None) -> List[Planet]:
    if mode == "expand":
        neu = sorted(state.neu_pl, key=lambda p: (-p.production, -p.ships))
        # v20: 旧阈值 ships<=20 会漏掉邻接「26~40兵+3产」等可吃口袋，表现为一格之隔从不打。
        soft_en: List[Planet] = []
        for p in state.en_pl:
            if p.ships <= 20 or p.production <= 2:
                soft_en.append(p)
                continue
            if state.phase() != "early":
                eg = state.effective_garrison(p)
                if eg <= 44 and p.ships <= 52:
                    soft_en.append(p)
        return neu + soft_en
    if mode == "aggro":
        return sorted(state.en_pl,
                      key=lambda p: (state.effective_garrison(p) - p.production * 5,
                                     p.dist_xy(SUN_X, SUN_Y)))
    if mode == "comet":
        return [p for p in state.planets if p.is_comet and p.owner != state.my_id]
    if mode == "counter":
        return [p for p in state.en_pl if state.effective_garrison(p) < p.ships * 0.55]
    if mode == "diplo":
        return state.en_pl
    return state.neu_pl + state.en_pl


def _build_capture_plan(snap: Snapshot, mode: str,
                        diplo: Optional["DiplomacyEngine"] = None,
                        regional_graph: Optional[RegionalGraph] = None,
                        diplo_target: Optional[Planet] = None) -> Plan:
    """Generic build: ranked targets + multi-source ETA-synced batches.
    Does NOT mutate snap.used; the Arbiter does that on commit."""
    state = snap.state
    actions: List[Tuple[int, int, int]] = []
    score = 0.0
    local_used: Dict[int, int] = defaultdict(int)
    target_done: set = set()

    targets = _target_pool(state, mode, diplo)
    if mode == "diplo" and diplo_target is not None:
        targets = [diplo_target] + [p for p in targets if p.id != diplo_target.id]

    # Rank.
    ranked: List[Tuple[float, Planet]] = []
    # Rank (aggro/counter/diplo keep tight floor; expand/balanced need looser floor).
    rank_floor = EXPAND_RANK_SCORE_FLOOR if mode in ("expand", "balanced") else -31.0
    for dst in targets:
        best_sc = -1e18
        for src in state.my_pl:
            sc, _, _ = capture_edge_score(snap, src, dst, regional_graph)
            if mode == "aggro":
                sc += recapture_bonus(snap, dst) * 0.5
            if mode == "counter":
                sc += 38.0 if snap.policy.phase == "mid" else 30.0
            if mode == "diplo":
                sc += 25.0 if dst is diplo_target else 0.0
            if diplo:
                sc += diplo.leader_penalty(dst)
            if sc > best_sc:
                best_sc = sc
        # FFA expand：近处 2+ 产灰星强制挤进排序前列，避免 split_lock 先把出兵星锁去远征
        if (
            mode == "expand"
            and state.is_ffa_mode()
            and dst.owner == -1
            and dst.production >= 2
        ):
            dmin = min((m.dist(dst) for m in state.my_pl), default=999.0)
            if dmin < 50.0:
                best_sc += 44.0
        if (
            mode == "expand"
            and dst.owner not in (-1, state.my_id)
            and dst.production <= 1
            and float(state.effective_garrison(dst)) <= 40.0
        ):
            dmin_en = min((m.dist(dst) for m in state.my_pl), default=999.0)
            if dmin_en < 54.0:
                best_sc += 34.0
        if best_sc > rank_floor:
            ranked.append((best_sc, dst))
    ranked.sort(key=lambda x: -x[0])

    source_dst_lock: Dict[int, int] = {}
    split_lock = (
        mode in ("expand", "balanced")
        and len(state.my_pl) <= ONE_OUTBOUND_DST_PER_SOURCE_UNTIL_N_WORLDS
    )

    for _, dst in ranked[:MAX_TARGETS_PER_PLAN]:
        if len(actions) >= MAX_TOTAL_MOVES or dst.id in target_done:
            continue

        # Defer low-prod neutrals while a prod>=3 factory grey is catchable within a
        # few turns of HQ production — don't burn the first wave on 14@1 when 20@3/4
        # is about to become peelable (same orbit / similar eta).
        if mode == "expand" and dst.owner == -1 and dst.production <= 2:
            touch_d = min((m.dist(dst) for m in state.my_pl), default=999.0)
            skip_defer = (
                (state.is_ffa_mode() and touch_d < 50.0 and dst.production >= 2)
                or (touch_d < 42.0 and dst.production <= 1)
            )
            defer_low_yield = False
            if not skip_defer:
                pool_rem = sum(
                    max(0, snap.avail(p.id) - local_used[p.id]) for p in state.my_pl)
                my_prod_sum = sum(p.production for p in state.my_pl)
                for h in state.neu_pl:
                    if h.production < 3 or h.id == dst.id:
                        continue
                    if not any(
                        point_segment_distance(SUN_X, SUN_Y, sx.x, sx.y, h.x, h.y)
                        >= SUN_RADIUS + SUN_PATH_MARGIN
                        for sx in state.my_pl
                    ):
                        continue
                    min_need = min(
                        capture_need(state, s, h)[0] for s in state.my_pl)
                    # prod==1 mites: wait longer toward factory; prod==2 slightly longer than old slack.
                    if dst.production <= 1:
                        wait_budget = max(my_prod_sum * 6, 20 + my_prod_sum * 2, 24)
                    else:
                        wait_budget = max(my_prod_sum * 4, 12 + my_prod_sum * 2)
                    if pool_rem < min_need <= pool_rem + wait_budget:
                        defer_low_yield = True
                        break
            if defer_low_yield:
                continue

        contributors: List[Tuple[int, Planet, int, int, float]] = []
        en_belt = any(is_sun_belt_planet(state, e) for e in state.en_pl)
        for src in state.my_pl:
            if split_lock:
                locked_d = source_dst_lock.get(src.id)
                if locked_d is not None and locked_d != dst.id:
                    continue
            avail = max(0, snap.avail(src.id) - local_used[src.id])
            # Do not strip inner sun-belt planets for outer “passing” neutrals once
            # the mid-game contest starts (rotation drags these into enemy arcs).
            ffa_pocket_neu = (
                state.is_ffa_mode()
                and dst.owner == -1
                and dst.production >= 2
                and min((m.dist(dst) for m in state.my_pl), default=999.0) < 48.0
            )
            if mode in ("expand", "balanced") and is_sun_belt_planet(state, src):
                if dst.owner == -1 and not is_sun_belt_planet(state, dst):
                    if (
                        len(state.my_pl) >= 2
                        and (en_belt or state.phase() != "early")
                        and not ffa_pocket_neu
                    ):
                        hoard = 14 + src.production * 5
                        if state.phase() in ("mid", "late"):
                            hoard += 8
                        avail = max(0, avail - hoard)
            # v14: lowered threshold to ABS_MIN_BATCH (was max(ABS_MIN_BATCH, prod*2))
            # so all planets with any surplus can join coordinated captures.
            if avail < ABS_MIN_BATCH:
                continue
            need, eta = capture_need(state, src, dst)
            sc, _, _ = capture_edge_score(snap, src, dst, regional_graph)
            contributors.append((eta, src, avail, need, sc))
        if not contributors:
            continue

        contributors.sort(key=lambda x: (x[0], -x[4], -x[2]))
        eta0 = contributors[0][0]
        min_eta = eta0

        staged_ok: List[Tuple[int, int, int]] = []
        group_used: List[Tuple[int, Planet, int, int, float]] = []
        required_ok = 0
        window = SYNC_ETA_WINDOW
        while window <= SYNC_ETA_WINDOW_MAX:
            group = [c for c in contributors if c[0] <= eta0 + window][
                :MAX_SOURCES_PER_TARGET]
            if not group:
                break

            group_eta = max(c[0] for c in group)
            # v16: garrison uses fastest arrival; enemy production uses (group-min) gap.
            if dst.owner == -1 and group_eta > 3 and not snap.is_safe_investment(
                    dst, group_eta):
                slack_pool = sum(
                    max(0, snap.avail(p.id) - local_used[p.id]) for p in state.my_pl)
                brute_need = max(
                    NEUTRAL_BRUTE_SLACK_MIN,
                    int(dst.ships * NEUTRAL_BRUTE_SLACK_MUL + dst.production * 6 + 24),
                )
                if slack_pool < brute_need:
                    break
            owner, garrison = target_state_at(state, dst, min_eta)
            if owner == state.my_id:
                break
            if owner == -1:
                if snap.policy.phase == "early" and min_eta <= 2:
                    neu_pad = 1
                elif snap.policy.phase == "early" and min_eta <= 12:
                    # Match ``capture_need`` early neutral margin (+1), not +2 — otherwise
                    # ``required`` is 22 vs ships+1==21 and solo HQ fails staging one turn early.
                    neu_pad = 1
                else:
                    neu_pad = 3
            else:
                neu_pad = 8 + min(6, dst.production)
            required = garrison + neu_pad
            if owner not in (-1, state.my_id):
                required += dst.production * max(0, group_eta - min_eta) // 3

            # _emit rejects sends < ABS_MIN_BATCH (staging-only required can be 6+1=7).
            required = max(int(required), ABS_MIN_BATCH)

            total_group_avail = sum(c[2] for c in group)
            need_floor = int(max(c[3] for c in group))
            if total_group_avail >= max(required, need_floor):
                required = max(required, min(need_floor, total_group_avail))

            if state.en_pl:
                e_eta_snap, _ = enemy_eta_power(state, dst)
                slack_snipe = int(e_eta_snap) <= 0 or int(e_eta_snap) >= group_eta + 7
                if slack_snipe and snap.is_safe_investment(dst, max(group_eta, 3)):
                    if dst.owner == -1:
                        required += max(8, dst.production * 2 + 10)
                    else:
                        required += max(14, dst.production * 2 + 14)

            if dst.is_comet and state.comet_turns_left(dst) <= group_eta + 5:
                break

            if total_group_avail < required:
                window += max(4, SYNC_ETA_WINDOW)
                continue

            sent = 0
            staged: List[Tuple[int, int, int]] = []
            for _, src, avail, _, _ in group:
                if sent >= required:
                    break
                send = min(avail, required - sent)
                if send < ABS_MIN_BATCH and sent + send < required:
                    continue
                send = max(send, ABS_MIN_BATCH) if send >= ABS_MIN_BATCH else send
                staged.append((src.id, dst.id, send))
                sent += send
            if sent >= required:
                staged_ok = staged
                group_used = group
                required_ok = required
                break

            window += max(4, SYNC_ETA_WINDOW)

        if not staged_ok:
            continue

        for sid, did, send in staged_ok:
            actions.append((sid, did, send))
            local_used[sid] += send
            if split_lock:
                source_dst_lock[sid] = dst.id
        target_done.add(dst.id)
        score += sum(c[4] for c in group_used[:len(staged_ok)]) + required_ok * (
            1.5 if dst.owner != -1 else 0.9)

    return Plan(actions, score, mode)


# ╔═══ region 7b: pragmatic action-space UCB ══════════════════════════════════╗


class _PragmaticNode:
    __slots__ = ("action_idx", "parent", "children", "visits", "value", "_untried")

    def __init__(self, action_idx: Optional[int], parent: Optional["_PragmaticNode"], n_choices: int):
        self.action_idx = action_idx
        self.parent = parent
        self.children: List["_PragmaticNode"] = []
        self.visits = 0
        self.value = 0.0
        self._untried = list(range(n_choices))

    def ucb1(self, c: float = 1.35) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return self.value / self.visits
        return (self.value / self.visits +
                c * math.sqrt(math.log(self.parent.visits) / self.visits))

    def best_child(self, c: float = 1.35) -> "_PragmaticNode":
        return max(self.children, key=lambda n: n.ucb1(c))


def pragmatic_candidate_actions(
    snap: Snapshot,
    regional_graph: Optional[RegionalGraph],
    top_k: int,
) -> List[Tuple[int, int, int]]:
    """Top-K single-edge moves ranked by capture_edge_score."""
    state = snap.state
    pool: List[Planet] = []
    seen: Set[int] = set()
    for p in sorted(state.neu_pl, key=lambda x: (-x.production, -x.ships)):
        if p.id not in seen:
            pool.append(p)
            seen.add(p.id)
    for p in state.en_pl:
        if p.id not in seen and (p.ships <= 28 or p.production <= 3):
            pool.append(p)
            seen.add(p.id)

    ranked_edges: List[Tuple[float, Tuple[int, int, int]]] = []
    for dst in pool[:36]:
        best_sc = -1e18
        best_pack: Optional[Tuple[int, int, int]] = None
        for src in state.my_pl:
            sc, need, _eta = capture_edge_score(snap, src, dst, regional_graph)
            if state.phase() == "late":
                dm = min((m.dist(dst) for m in state.my_pl), default=999.0)
                if dm < 52.0 and dst.owner == -1 and dst.production <= 1:
                    sc += 52.0
                elif (
                    dm < 54.0
                    and dst.owner not in (-1, state.my_id)
                    and dst.production <= 1
                ):
                    if float(state.effective_garrison(dst)) <= 42.0:
                        sc += 58.0
            if sc <= best_sc:
                continue
            avail = snap.avail(src.id)
            if avail < ABS_MIN_BATCH:
                continue
            send = min(avail, max(ABS_MIN_BATCH, min(int(need), avail)))
            if (
                state.is_ffa_mode()
                and avail >= 38
                and send < 22
                and int(need) >= 13
                and dst.owner != -1
            ):
                send = min(avail, 22)
            if send < ABS_MIN_BATCH:
                continue
            best_sc = sc
            best_pack = (src.id, dst.id, send)
        if best_pack is not None and best_sc > EXPAND_RANK_SCORE_FLOOR:
            ranked_edges.append((best_sc, best_pack))

    ranked_edges.sort(key=lambda x: -x[0])
    out: List[Tuple[int, int, int]] = []
    picked: Set[Tuple[int, int]] = set()
    for _sc, triple in ranked_edges:
        sid, did, _ = triple
        if (sid, did) in picked:
            continue
        picked.add((sid, did))
        out.append(triple)
        if len(out) >= top_k:
            break
    return out


class PragmaticActionUCB:
    """UCB1 over top-K atomic moves; rollout via short score_plan_actions."""

    def __init__(
        self,
        state: GameState,
        snap: Snapshot,
        regional_graph: Optional[RegionalGraph],
        candidates: List[Tuple[int, int, int]],
        budget_ms: float,
        max_iters: int,
        rollout_steps: int,
        tempo_floor: int,
    ):
        self.state = state
        self.snap = snap
        self.regional_graph = regional_graph
        self.candidates = candidates
        self.budget_ms = max(0.0, budget_ms)
        self.max_iters = max(0, max_iters)
        self.rollout_steps = max(4, rollout_steps)
        self.tempo_floor = tempo_floor

    def evaluate(self) -> Tuple[Dict[int, float], Optional[int]]:
        n = len(self.candidates)
        if n == 0 or self.budget_ms <= 0 or self.max_iters <= 0:
            return {}, None

        root = _PragmaticNode(action_idx=None, parent=None, n_choices=n)
        deadline = time.time() * 1000.0 + self.budget_ms
        iters = 0
        while iters < self.max_iters and time.time() * 1000.0 < deadline:
            if root._untried:
                idx = root._untried.pop()
                child = _PragmaticNode(action_idx=idx, parent=root, n_choices=0)
                root.children.append(child)
                node = child
            elif root.children:
                node = root.best_child()
            else:
                break

            act = self.candidates[node.action_idx]
            val = score_plan_actions(
                self.state,
                [act],
                steps=self.rollout_steps,
                tempo_floor=self.tempo_floor,
            )
            cur: Optional[_PragmaticNode] = node
            while cur is not None:
                cur.visits += 1
                cur.value += val
                cur = cur.parent
            iters += 1

        out: Dict[int, float] = {}
        best_idx_out: Optional[int] = None
        best_visits = -1
        best_mean = -1e30
        for c in root.children:
            if c.visits <= 0 or c.action_idx is None:
                continue
            mean_v = c.value / c.visits
            out[c.action_idx] = mean_v
            if c.visits > best_visits or (
                c.visits == best_visits and mean_v > best_mean
            ):
                best_visits = c.visits
                best_mean = mean_v
                best_idx_out = c.action_idx

        return out, best_idx_out


class ExpandPlanner:
    """Wraps capture builder for neutral / weak-target modes."""

    @staticmethod
    def plan(snap: Snapshot, mode: str = "expand",
             diplo: Optional["DiplomacyEngine"] = None,
             regional_graph: Optional[RegionalGraph] = None) -> Plan:
        return _build_capture_plan(snap, mode, diplo=diplo, regional_graph=regional_graph)


class AttackPlanner:
    """Wraps capture builder for aggressive enemy-target modes."""

    @staticmethod
    def plan(snap: Snapshot, mode: str = "aggro",
             diplo: Optional["DiplomacyEngine"] = None,
             diplo_target: Optional[Planet] = None,
             regional_graph: Optional[RegionalGraph] = None) -> Plan:
        return _build_capture_plan(snap, mode, diplo=diplo, diplo_target=diplo_target,
                                   regional_graph=regional_graph)


# ── OpponentModel + DiplomacyEngine ──────────────────────────────────────────

class OpponentModel:
    """Cross-turn enemy stats. Shared instance across the agent's lifetime."""

    def __init__(self):
        self.ship_h: Dict[int, List[int]] = defaultdict(list)
        self.planet_h: Dict[int, List[int]] = defaultdict(list)
        self.atk_count: Dict[int, int] = defaultdict(int)

    def update(self, state: GameState):
        for eid in state.en_ids:
            self.ship_h[eid].append(state.total_ships(eid))
            self.planet_h[eid].append(len([p for p in state.planets if p.owner == eid]))
        for f in state.fleets:
            if f.owner in state.en_ids:
                t = state.fleet_target.get(f.id)
                if t:
                    tp = state.get(t[0])
                    if tp and tp.owner == state.my_id:
                        self.atk_count[f.owner] += 1

    def aggression(self, eid: int) -> float:
        n = self.atk_count.get(eid, 0)
        t = max(len(self.ship_h.get(eid, [1])), 1)
        return min(n / t * 6, 1.0)


_GLOBAL_OPP = OpponentModel()


class DiplomacyEngine:
    LEADER = "LEADER"
    MID = "MID"
    WEAK = "WEAK"

    def __init__(self, state: GameState, opp: OpponentModel):
        self.state = state; self.opp = opp

    def power(self, eid: int) -> float:
        s = self.state
        prod = sum(p.production for p in s.planets if p.owner == eid)
        return s.total_ships(eid) + prod * 24.0

    def threat_to_us(self, eid: int) -> float:
        s = self.state
        ep = [p for p in s.planets if p.owner == eid]
        if not ep:
            return 0.0
        prox = sum(1.0 / max(m.dist(e), 1.0) for m in s.my_pl for e in ep)
        aggr = 1.0 + self.opp.aggression(eid)
        return (prox * 38.0 + s.total_ships(eid) * 0.65) * aggr

    def rank(self) -> List[Tuple[str, int, float]]:
        if not self.state.en_ids:
            return []
        scored = [(self.threat_to_us(e), e) for e in self.state.en_ids]
        scored.sort(reverse=True)
        powers = {e: self.power(e) for _, e in scored}
        max_power = max(powers.values()) if powers else 1.0
        result = []
        for thr, eid in scored:
            if len(self.state.en_ids) == 1:
                tag = self.LEADER
            elif powers[eid] >= max_power * 0.75:
                tag = self.LEADER
            elif powers[eid] >= max_power * 0.40:
                tag = self.MID
            else:
                tag = self.WEAK
            result.append((tag, eid, thr))
        return result

    def primary_target(self) -> Optional[Planet]:
        ranked = self.rank()
        if not ranked:
            return None
        leader_ids = [eid for tag, eid, _ in ranked if tag == self.LEADER]
        target_eid = leader_ids[0] if leader_ids else ranked[0][1]
        ep = [p for p in self.state.planets if p.owner == target_eid]
        if not ep:
            return None
        cx, cy = self.state.centroid()
        return min(ep, key=lambda p: self.state.effective_garrison(p)
                                     + math.hypot(p.x - cx, p.y - cy) * 0.38)

    def leader_penalty(self, dst: Planet) -> float:
        ranked = self.rank()
        if len(ranked) <= 1:
            return 0.0
        leaders = {eid for tag, eid, _ in ranked if tag == self.LEADER}
        if dst.owner in leaders:
            return 15.0
        leader_power = sum(self.power(e) for e in leaders)
        our_power = self.state.total_ships(self.state.my_id)
        if leader_power > our_power * 1.2:
            return -20.0
        return 0.0


# ── Redistribution & late dump (fallback fillers) ────────────────────────────

class RedistributionPlanner:
    @staticmethod
    def plan(snap: Snapshot) -> Plan:
        state = snap.state
        rockless = [p for p in state.my_pl if not p.is_comet]
        if len(rockless) < 2 or not state.en_pl:
            return Plan([], 0.0, "redistribute")

        def ned(p):
            return min((p.dist(e) for e in state.en_pl), default=999.0)

        ordered = sorted(rockless, key=ned)
        fc = max(1, min(len(ordered) - 1, len(ordered) // 2 + 1))
        fronts = ordered[:fc]
        en_belt = any(is_sun_belt_planet(state, e) for e in state.en_pl)
        rears = []
        for p in ordered[fc:]:
            if state.net_threat(p) > 0:
                continue
            if is_sun_belt_planet(state, p) and (
                en_belt or state.phase() != "early"
            ):
                # Belt rears are usually reserves for rotation — but huge stacks
                # must still feed fronts when strategic commit is capped.
                stack = max(0, snap.surplus.get(p.id, 0))
                if stack < 52:
                    continue
            rears.append(p)
        thin_front = [
            f
            for f in fronts
            if state.net_threat(f) <= 0
            and int(f.ships) <= ABS_MIN_BATCH * 5 + max(8, int(f.production) * 6)
        ]
        actions: List[Tuple[int, int, int]] = []
        local_used: Dict[int, int] = defaultdict(int)
        for rear in rears[:3]:
            avail = max(0, snap.avail(rear.id) - local_used[rear.id])
            # v17: raise threshold - only redistribute when rear has a meaningful
            # surplus (>= 20 ships). Small trickles just clutter the board.
            min_av = max(20, rear.production * 5)
            if state.is_ffa_mode() and state.phase() == "early":
                nearest_front = min((rear.dist(f) for f in fronts), default=999.0)
                if nearest_front <= 42.0:
                    min_av = max(14, rear.production * 4)
            if avail < min_av:
                continue
            if thin_front:
                dst = min(thin_front, key=lambda ff: rear.dist(ff))
                slice_frac = 0.53
            else:
                dst = min(fronts, key=lambda ff: rear.dist(ff))
                slice_frac = 0.40
            if state.is_ffa_mode() and state.phase() == "early" and rear.dist(dst) > 48.0:
                continue
            pkg_floor = 22 if (state.is_ffa_mode() and len(state.en_ids) >= 2) else 12
            send = max(
                ABS_MIN_BATCH,
                min(avail, max(pkg_floor, max(12, int(avail * slice_frac)))),
            )
            if send >= ABS_MIN_BATCH:
                actions.append((rear.id, dst.id, send))
                local_used[rear.id] += send
        return Plan(actions, float(sum(a[2] for a in actions)) * 0.25, "redistribute")


class UrgentHighProdPlanner:
    """Pre-emptively capture high-production nearby enemy planets BEFORE the
    strategic ranking. Mirrors v10's `build_urgent_highprod_plan`. Activation
    is gated by `policy.urgent_attack_*` so early game stays expansion-only."""

    @staticmethod
    def plan(snap: Snapshot) -> Plan:
        state = snap.state
        policy = snap.policy
        if not state.en_pl or policy.urgent_attack_ratio >= 999.0:
            return Plan([], 0.0, "urgent_hp", urgent=True)

        my_total = state.total_ships(state.my_id)
        if state.is_ffa_mode() and state.en_ids:
            en_cmp = max(state.total_ships(e) for e in state.en_ids)
        else:
            en_cmp = sum(state.total_ships(e) for e in state.en_ids)
        if my_total < en_cmp * policy.urgent_attack_ratio:
            return Plan([], 0.0, "urgent_hp", urgent=True)

        cx, cy = snap.centroid
        near_r = 48.0

        def _urgent_hp_near(p) -> bool:
            d_c = math.hypot(p.x - cx, p.y - cy)
            if d_c < 40.0:
                return True
            if state.is_ffa_mode() and state.my_pl:
                return min(m.dist(p) for m in state.my_pl) < near_r
            return False

        targets = sorted(
            [p for p in state.en_pl
             if p.production >= policy.urgent_attack_min_prod
             and _urgent_hp_near(p)],
            key=lambda p: -(p.production / max(1.0, math.hypot(p.x - cx, p.y - cy))),
        )
        if not targets:
            return Plan([], 0.0, "urgent_hp", urgent=True)

        actions: List[Tuple[int, int, int]] = []
        local_used: Dict[int, int] = defaultdict(int)
        for dst in targets[:3]:
            contributors: List[Tuple[int, Planet, int, int]] = []
            for src in state.my_pl:
                avail = max(0, snap.avail(src.id) - local_used[src.id])
                if avail < ABS_MIN_BATCH:
                    continue
                need, eta = capture_need(state, src, dst)
                contributors.append((eta, src, avail, need))
            if not contributors:
                continue
            contributors.sort(key=lambda x: x[0])
            eta0 = contributors[0][0]
            group = [c for c in contributors
                     if c[0] <= eta0 + SYNC_ETA_WINDOW][:MAX_SOURCES_PER_TARGET]
            group_eta = max(c[0] for c in group)
            min_eta = eta0
            owner, garrison = target_state_at(state, dst, min_eta)
            if owner == state.my_id:
                continue
            required = (garrison + 8 + min(6, dst.production)
                        + dst.production * max(0, group_eta - min_eta) // 3)
            sent = 0
            staged: List[Tuple[int, int, int]] = []
            for _, src, avail, _ in group:
                if sent >= required:
                    break
                send = min(avail, required - sent)
                if send < ABS_MIN_BATCH and sent + send < required:
                    continue
                send = max(send, ABS_MIN_BATCH) if send >= ABS_MIN_BATCH else send
                staged.append((src.id, dst.id, send))
                sent += send
            if sent < required:
                continue
            for sid, did, send in staged:
                actions.append((sid, did, send))
                local_used[sid] += send
        return Plan(actions, float(sum(a[2] for a in actions)) * 1.0,
                    "urgent_hp", urgent=True)


class LateDumpPlanner:
    @staticmethod
    def plan(snap: Snapshot) -> Plan:
        state = snap.state
        if state.phase() != "late" or not state.en_pl:
            return Plan([], 0.0, "late_dump")
        weak = min(state.en_pl,
                   key=lambda p: state.effective_garrison(p) + p.production * 2)
        actions: List[Tuple[int, int, int]] = []
        local_used: Dict[int, int] = defaultdict(int)
        for src in sorted(state.my_pl, key=lambda p: -snap.avail(p.id)):
            avail = max(0, snap.avail(src.id) - local_used[src.id])
            need, _ = capture_need(state, src, weak)
            if avail >= need + ABS_MIN_BATCH:
                send = min(avail, need + 12)
                actions.append((src.id, weak.id, send))
                local_used[src.id] += send
        return Plan(actions, float(sum(a[2] for a in actions)) * 0.5, "late_dump")


# ╔═══ region 8: MCTSEngine - plan-level tree search ════════════════════════╗

class _MCTSNode:
    __slots__ = ("plan_idx", "parent", "children", "visits", "value", "_untried")

    def __init__(self, plan_idx, parent, n_choices):
        self.plan_idx = plan_idx
        self.parent: Optional[_MCTSNode] = parent
        self.children: List[_MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self._untried = list(range(n_choices))

    def ucb1(self, c: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return self.value / self.visits
        return (self.value / self.visits +
                c * math.sqrt(math.log(self.parent.visits) / self.visits))

    def best_child(self, c: float = 1.41) -> "_MCTSNode":
        return max(self.children, key=lambda n: n.ucb1(c))

    def fully_expanded(self) -> bool:
        return len(self._untried) == 0


class MCTSEngine:
    """Plan-level MCTS. Given a list of candidate plans, runs UCB1 over them and
    returns a per-plan bonus reflecting expected outcome above the rest.

    The 'rollout' is the score_plan_actions value applied with `rollout_steps`
    sim turns. We don't expand deeper than 1 level (no branching after a plan
    is chosen) - this keeps cost predictable and the bonus interpretable."""

    def __init__(self, state: GameState, plans: List[Plan],
                 budget_ms: float, max_iters: int,
                 rollout_steps: int = 12):
        self.state = state
        self.plans = plans
        self.budget_ms = max(0.0, budget_ms)
        self.max_iters = max(0, max_iters)
        self.rollout_steps = rollout_steps

    def _rollout(self, plan_idx: int) -> float:
        plan = self.plans[plan_idx]
        return score_plan_actions(self.state, plan.actions, steps=self.rollout_steps)

    def evaluate(self) -> Dict[int, float]:
        """Return {plan_idx: mean_value}. Empty dict if budget == 0 or no plans."""
        n = len(self.plans)
        if n == 0 or self.budget_ms <= 0 or self.max_iters <= 0:
            return {}

        root = _MCTSNode(plan_idx=None, parent=None, n_choices=n)
        deadline = time.time() * 1000.0 + self.budget_ms
        iters = 0
        while iters < self.max_iters and time.time() * 1000.0 < deadline:
            # Selection / expansion (depth 1).
            if root._untried:
                idx = root._untried.pop()
                child = _MCTSNode(plan_idx=idx, parent=root, n_choices=0)
                root.children.append(child)
                node = child
            elif root.children:
                node = root.best_child()
            else:
                break

            val = self._rollout(node.plan_idx)
            cur: Optional[_MCTSNode] = node
            while cur is not None:
                cur.visits += 1
                cur.value += val
                cur = cur.parent
            iters += 1

        out: Dict[int, float] = {}
        for c in root.children:
            if c.visits > 0:
                out[c.plan_idx] = c.value / c.visits
        return out
class PlanArbiter:
    """Single source of decisions. Collects candidate plans, scores them with
    sim + MCTS bonus + Neural modifier, then commits to `moves` honouring the
    Snapshot's per-source surplus."""

    def __init__(self, snap: Snapshot, diplo: DiplomacyEngine,
                 neural: NeuralVal,
                 elapsed_ms_fn,
                 deadline_ms: float = 920.0,
                 regional_graph: Optional[RegionalGraph] = None,
                 multi_hop_planner: Optional[MultiHopPlanner] = None):
        self.snap = snap
        self.policy = snap.policy
        self.diplo = diplo
        self.neural = neural
        self.elapsed_ms = elapsed_ms_fn
        self.deadline_ms = deadline_ms
        self.moves: List[List] = []
        # v20: regional awareness
        self.regional_graph = regional_graph
        self.multi_hop_planner = multi_hop_planner

    # ── 1. urgent: defense + intercept (commit immediately) ──────────────────

    def commit_urgent(self) -> None:
        # Defense + intercept run with `urgent=True` (allowed to dip into reserve).
        for planner in (CometEvacPlanner, DefensePlanner, InterceptPlanner):
            if self._out_of_time(self.deadline_ms - 320):
                return
            self._commit_plan(planner.plan(self.snap), urgent=True)
        # Urgent high-production attacks: phase-gated, NOT urgent w.r.t. reserves
        # (must respect normal surplus to avoid leaving home defenseless).
        if self._out_of_time(self.deadline_ms - 280):
            return
        self._commit_plan(UrgentHighProdPlanner.plan(self.snap), urgent=False)

    # ── 2. strategic: collect, score, commit best ────────────────────────────

    def collect_strategic(self) -> List[Plan]:
        plans: List[Plan] = []
        diplo_tgt = self.diplo.primary_target()

        rg = self.regional_graph  # shorthand, may be None

        # Comet always considered if any non-allied comet exists.
        if any(p.is_comet and p.owner != self.snap.state.my_id
               for p in self.snap.state.planets):
            plans.append(ExpandPlanner.plan(self.snap, "comet", diplo=self.diplo,
                                            regional_graph=rg))

        for mode in self.policy.mode_order:
            if mode == "comet":
                continue  # already handled
            if self._out_of_time(self.deadline_ms - 200):
                break
            if mode in ("expand", "balanced"):
                plans.append(ExpandPlanner.plan(self.snap, mode, diplo=self.diplo,
                                                regional_graph=rg))
            elif mode == "diplo" and diplo_tgt is not None:
                plans.append(AttackPlanner.plan(self.snap, "diplo",
                                                diplo=self.diplo,
                                                diplo_target=diplo_tgt,
                                                regional_graph=rg))
            elif mode in ("aggro", "counter"):
                plans.append(AttackPlanner.plan(self.snap, mode, diplo=self.diplo,
                                                regional_graph=rg))

        # Optional diplo even if not in mode_order (always available).
        if diplo_tgt is not None and not any(p.tag == "diplo" for p in plans):
            plans.append(AttackPlanner.plan(self.snap, "diplo",
                                            diplo=self.diplo,
                                            diplo_target=diplo_tgt,
                                            regional_graph=rg))

        # Drop empty plans.
        return [p for p in plans if p.actions]

    def score_with_modifiers(self, plans: List[Plan]) -> List[Tuple[float, Plan]]:
        if not plans:
            return []

        # 1. Base sim scores (cheap).
        sim_steps = self.policy.sim_steps
        base_rows: List[List] = []
        st = self.snap.state
        for plan in plans:
            sim_b = score_plan_actions(st, plan.actions,
                                      steps=sim_steps,
                                      tempo_floor=self.policy.tempo_floor)
            base_rows.append([plan.score + sim_b, plan, sim_b])

        base_rows.sort(key=lambda r: -r[0])

        # 1b. Optional pessimistic 1-enemy-rollout refinement (timed budget).
        if (
            self.policy.paranoid_score_budget_ms >= 18.0
            and self.policy.paranoid_blend > 1e-6
        ):
            t0_par = self.elapsed_ms()
            lim = max(1, min(self.policy.paranoid_plan_top_k, len(base_rows)))
            pb = float(self.policy.paranoid_score_budget_ms)
            blend = float(self.policy.paranoid_blend)
            psteps = max(4, min(self.policy.paranoid_steps, sim_steps + 8))
            for i in range(lim):
                if self.elapsed_ms() - t0_par >= pb:
                    break
                s_tot, plan, sim_b = base_rows[i]
                _, pessim = score_plan_actions_paranoid(
                    st, plan.actions, sim_steps,
                    tempo_floor=self.policy.tempo_floor,
                    par_steps=psteps)
                adj = sim_b + blend * (pessim - sim_b)
                base_rows[i][0] = plan.score + adj
            base_rows.sort(key=lambda r: -r[0])

        base: List[Tuple[float, Plan]] = [(r[0], r[1]) for r in base_rows]

        # 2. MCTS bonus on top-3.
        top_k = min(3, len(base))
        mcts_bonus: Dict[int, float] = {}
        if top_k > 0 and self.policy.mcts_budget_ms > 0:
            elapsed_now = self.elapsed_ms()
            remaining = max(0.0, self.deadline_ms - elapsed_now - 60.0)
            budget = min(self.policy.mcts_budget_ms, remaining)
            if budget > 30.0:
                top_plans = [p for _, p in base[:top_k]]
                mcts = MCTSEngine(self.snap.state, top_plans,
                                  budget_ms=budget,
                                  max_iters=self.policy.mcts_max_iters,
                                  rollout_steps=max(sim_steps, 10))
                mcts_vals = mcts.evaluate()
                if mcts_vals:
                    # Normalise around 0: subtract mean, scale to ~base score range.
                    mu = sum(mcts_vals.values()) / len(mcts_vals)
                    for idx, v in mcts_vals.items():
                        mcts_bonus[idx] = (v - mu) * 0.5

        prag_bonus: Dict[int, float] = {}
        if self.policy.pragmatic_mcts_budget_ms > 0 and self.policy.pragmatic_mcts_max_iters > 0:
            elapsed_p = self.elapsed_ms()
            rem_p = max(0.0, self.deadline_ms - elapsed_p - 40.0)
            pbudget = min(self.policy.pragmatic_mcts_budget_ms, rem_p)
            if pbudget > 22.0:
                cands = pragmatic_candidate_actions(
                    self.snap,
                    self.regional_graph,
                    self.policy.pragmatic_mcts_top_k,
                )
                if cands:
                    prag = PragmaticActionUCB(
                        self.snap.state,
                        self.snap,
                        self.regional_graph,
                        cands,
                        pbudget,
                        self.policy.pragmatic_mcts_max_iters,
                        self.policy.pragmatic_mcts_rollout_steps,
                        self.policy.tempo_floor,
                    )
                    means, star_idx = prag.evaluate()
                    if means and star_idx is not None:
                        mu_p = sum(means.values()) / len(means)
                        sid_st, did_st, _ = cands[star_idx]
                        spread = (means[star_idx] - mu_p) * 0.42
                        for i, (_, plan) in enumerate(base):
                            if any(a[0] == sid_st and a[1] == did_st for a in plan.actions):
                                prag_bonus[i] = spread

        # 3. Neural modifier (state-only, applied multiplicatively).
        modifier = self.neural.score_modifier(self.snap.state,
                                              self.policy.neural_modifier_strength)

        scored: List[Tuple[float, Plan]] = []
        for i, (s, plan) in enumerate(base):
            bonus = mcts_bonus.get(i, 0.0) + prag_bonus.get(i, 0.0)
            final = s * modifier + bonus
            scored.append((final, plan))
        scored.sort(key=lambda x: -x[0])
        return scored

    def _trim_plan_to_ship_budget(self, plan: Plan, max_total: int) -> Plan:
        """Shrink planned sends to at most ``max_total`` ships (greedy prefix)."""
        if max_total < ABS_MIN_BATCH or not plan.actions:
            return Plan([], plan.score, plan.tag, urgent=plan.urgent)
        remain = int(max_total)
        out: List[Tuple[int, int, int]] = []
        for sid, did, ships in plan.actions:
            if remain < ABS_MIN_BATCH:
                break
            take = min(int(ships), remain)
            if take >= ABS_MIN_BATCH:
                out.append((sid, did, take))
                remain -= take
        return Plan(out, plan.score, plan.tag, urgent=plan.urgent)

    def _strategic_trim_and_gate(
        self, best_score: float, best_plan: Plan
    ) -> Optional[Tuple[float, Plan]]:
        """Regional ship-budget trim + mid/late baseline gate. None = skip candidate."""
        if not best_plan.actions:
            return None

        if (
            self.regional_graph is not None
            and self.snap.policy.phase != "early"
            and best_plan.actions
        ):
            threats = [
                self.regional_graph.region_threat(rid, self.snap.state.en_pl)
                for rid in self.regional_graph.regions
            ]
            max_t = max(threats) if threats else 0.0
            my_prod = float(sum(p.production for p in self.snap.state.my_pl))
            rp = float(self.snap.policy.region_pressure_ratio)
            ssm = float(self.snap.policy.safe_surplus_ship_mult)
            if max_t > my_prod * rp:
                budget = self.snap.calculate_safe_surplus_v20(self.regional_graph)
                limit = int(budget * ssm) + ABS_MIN_BATCH * 4
                n_my = len(self.snap.state.my_pl)
                ta = sum(self.snap.avail(p.id) for p in self.snap.state.my_pl)
                if n_my == 1 and ta >= ABS_MIN_BATCH * 10:
                    limit = max(
                        limit,
                        min(int(ta * 0.34), 240),
                        ABS_MIN_BATCH * 12,
                    )
                ship_sum = sum(a[2] for a in best_plan.actions)
                if ship_sum > limit:
                    trimmed = self._trim_plan_to_ship_budget(best_plan, limit)
                    if not trimmed.actions and n_my == 1 and ta >= ABS_MIN_BATCH * 10:
                        trimmed = self._trim_plan_to_ship_budget(
                            best_plan, max(limit, min(ship_sum, ta // 2))
                        )
                    if not trimmed.actions:
                        return None
                    best_plan = trimmed
                    best_score = trimmed.score

        if self.snap.state.phase() != "early":
            baseline = score_plan_actions(self.snap.state, [],
                                          steps=self.policy.sim_steps,
                                          tempo_floor=self.policy.tempo_floor)
            margin = float(self.snap.policy.baseline_commit_margin)
            commit_bonus = 0.0
            st0 = self.snap.state
            if best_plan.tag in ("expand", "balanced") and best_plan.actions:
                if any(
                    st0.get(a[1]) is not None and st0.get(a[1]).owner == -1
                    for a in best_plan.actions
                ):
                    commit_bonus = 6.0
            total_avail_commit = sum(self.snap.avail(p.id) for p in st0.my_pl)
            lone_hq_snap = (
                len(st0.my_pl) == 1
                and total_avail_commit
                >= max(ABS_MIN_BATCH * 11, int(st0.my_pl[0].ships * 0.14))
                and best_plan.actions
                and best_plan.tag
                in ("expand", "balanced", "aggro", "counter", "comet", "diplo")
            )
            if lone_hq_snap:
                commit_bonus += 24.0
            if (
                best_score + commit_bonus <= baseline + margin
                and not lone_hq_snap
            ):
                return None
        return (best_score, best_plan)

    def commit_best(self, scored: List[Tuple[float, Plan]]) -> None:
        """Commit strategic plans — v21 top-6 + trim/gates + pragmatic; v20 best-only."""
        if registry.arbiter_variant == "v20":
            self._commit_best_v20(scored)
            return
        self._commit_best_v21(scored)

    def _commit_best_v20(self, scored: List[Tuple[float, Plan]]) -> None:
        """v20/0513: single best plan with regional ship cap + baseline gate (no top-6 loop)."""
        if not scored:
            return
        best_score, best_plan = scored[0]

        if (
            self.regional_graph is not None
            and self.snap.policy.phase != "early"
            and best_plan.actions
        ):
            threats = [
                self.regional_graph.region_threat(rid, self.snap.state.en_pl)
                for rid in self.regional_graph.regions
            ]
            max_t = max(threats) if threats else 0.0
            my_prod = float(sum(p.production for p in self.snap.state.my_pl))
            rp = float(self.snap.policy.region_pressure_ratio)
            ssm = float(self.snap.policy.safe_surplus_ship_mult)
            if max_t > my_prod * rp:
                budget = self.snap.calculate_safe_surplus_v20(self.regional_graph)
                limit = int(budget * ssm) + ABS_MIN_BATCH * 4
                ship_sum = sum(a[2] for a in best_plan.actions)
                if ship_sum > limit:
                    trimmed = self._trim_plan_to_ship_budget(best_plan, limit)
                    if not trimmed.actions:
                        return
                    best_plan = trimmed
                    best_score = trimmed.score

        if self.snap.state.phase() != "early":
            baseline = score_plan_actions(
                self.snap.state,
                [],
                steps=self.policy.sim_steps,
                tempo_floor=self.policy.tempo_floor,
            )
            margin = float(self.snap.policy.baseline_commit_margin)
            commit_bonus = 0.0
            st0 = self.snap.state
            if best_plan.tag in ("expand", "balanced") and best_plan.actions:
                if any(
                    st0.get(a[1]) is not None and st0.get(a[1]).owner == -1
                    for a in best_plan.actions
                ):
                    commit_bonus = 6.0
            if best_score + commit_bonus <= baseline + margin:
                return
        self._commit_plan(best_plan, urgent=False)

    def _commit_best_v21(self, scored: List[Tuple[float, Plan]]) -> None:
        """v21_lite: try ranked strategic plans until one survives trim/gates; pragmatic."""
        if not scored:
            return
        n0 = len(self.moves)
        seen_actions: set = set()
        for i in range(min(6, len(scored))):
            sc, raw = scored[i]
            key = tuple(raw.actions)
            if key in seen_actions:
                continue
            seen_actions.add(key)
            prepared = self._strategic_trim_and_gate(sc, raw)
            if prepared is None:
                continue
            _, plan = prepared
            self._commit_plan(plan, urgent=False)
            if len(self.moves) > n0:
                return

        st = self.snap.state
        late_mite_finish = False
        if st.phase() == "late":
            for p in st.planets:
                if p.owner == -1 and p.production <= 1:
                    if min((m.dist(p) for m in st.my_pl), default=999.0) < 52.0:
                        late_mite_finish = True
                        break
                if (
                    p.owner not in (-1, st.my_id)
                    and p.production <= 1
                    and float(st.effective_garrison(p)) <= 42.0
                ):
                    if min((m.dist(p) for m in st.my_pl), default=999.0) < 54.0:
                        late_mite_finish = True
                        break

        if len(self.moves) == n0 and (
            st.phase() == "early"
            or len(st.my_pl) == 1
            or late_mite_finish
        ):
            cands = pragmatic_candidate_actions(
                self.snap, self.regional_graph, top_k=14
            )
            for sid, did, ships in cands:
                if len(self.moves) >= MAX_TOTAL_MOVES:
                    return
                if self._emit(sid, did, ships, urgent=False):
                    return

    # ── 3. fallback: redistribution + late dump ──────────────────────────────

    def commit_fallback(self) -> None:
        for planner in (RedistributionPlanner, LateDumpPlanner):
            if self._out_of_time(self.deadline_ms - 30):
                return
            self._commit_plan(planner.plan(self.snap), urgent=False)

    # ── helpers ──

    def _commit_plan(self, plan: Plan, urgent: bool) -> None:
        for sid, did, ships in plan.actions:
            if len(self.moves) >= MAX_TOTAL_MOVES:
                return
            self._emit(sid, did, ships, urgent=urgent)

    def _emit(self, sid: int, did: int, ships: int, urgent: bool) -> bool:
        snap = self.snap
        state = snap.state
        src = state.get(sid); dst = state.get(did)
        if src is None or dst is None or src.owner != state.my_id:
            return False

        comet_ttl = int(state.comet_turns_left(src)) if src.is_comet else 9999
        liq_src = max(0, int(src.ships) - int(snap.used.get(sid, 0)))
        comet_strip_panic = (
            urgent
            and src.is_comet
            and dst.owner == state.my_id
            and (
                comet_ttl <= 38
                or (comet_ttl <= 72 and liq_src <= 22)
                or (comet_ttl <= 120 and liq_src <= ABS_MIN_BATCH + 8)
            )
        )
        if urgent and not comet_strip_panic:
            cap = max(0, src.ships - ABS_MIN_BATCH - snap.used.get(sid, 0))
        elif comet_strip_panic:
            cap = max(0, int(src.ships) - int(snap.used.get(sid, 0)))
        else:
            cap = snap.avail(sid)
        send_cap = min(int(ships), cap)
        min_emit = 1 if comet_strip_panic else ABS_MIN_BATCH
        if send_cap < min_emit:
            return False

        retry_sends: List[int] = [send_cap]
        for delta in (-8, 8, -16, 16, -4, 12):
            s2 = send_cap + delta
            if s2 >= min_emit and s2 <= cap and s2 not in retry_sends:
                retry_sends.append(s2)
        if comet_strip_panic:
            for s2 in range(min(send_cap - 1, 7), 0, -1):
                if s2 >= min_emit and s2 <= cap and s2 not in retry_sends:
                    retry_sends.append(s2)

        direct_angle = math.atan2(dst.y - src.y, dst.x - src.x)
        chord_clips_sun = (
            point_segment_distance(SUN_X, SUN_Y, src.x, src.y, dst.x, dst.y)
            < SUN_RADIUS + SUN_PATH_MARGIN
        )
        ang_tol = 1.55 if chord_clips_sun else 1.20
        if comet_strip_panic:
            ang_tol += 0.42
        for send in retry_sends:
            angle, eta = safe_aim(state, src, dst, send)
            spd = fleet_speed(send, state.max_speed)
            lx0, ly0 = launch_origin(src, angle)
            if not _ray_safe(lx0, ly0, angle, spd, min_flight=max(1, eta - 1)):
                continue
            angle_diff = abs(angle - direct_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            if angle_diff > ang_tol:
                continue
            lx, ly = launch_origin(src, angle)
            if not launch_hits_target_first(state, lx, ly, angle, send, did,
                                            ignore_planet_id=None):
                continue
            if dst.owner == -1 and not urgent:
                queued = snap.pending_neutral_wave.get(did, 0)
                oth = my_inbound_ships_to(state, did) + queued
                if not neutral_wave_wins(state, dst, eta, send, oth):
                    continue
            self.moves.append([sid, float(angle), int(send)])
            snap.subtract(sid, send)
            if dst.owner == -1:
                snap.pending_neutral_wave[did] = snap.pending_neutral_wave.get(did, 0) + send
            return True
        return False


    def _out_of_time(self, threshold_ms: float) -> bool:
        return self.elapsed_ms() > threshold_ms

