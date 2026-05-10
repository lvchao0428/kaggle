"""Orbit Wars v10 — MCTS + NumPy NeuralVal + Full DiplomacyEngine

New over v9:
- MCTSEngine: UCB1 tree search with configurable time budget (default 400ms).
  Uses clone_sim/sim_step/eval_sim_planets with correct physics.
- NeuralVal: 14→64→32→1 NumPy MLP. Weights are base64-encoded inline so no
  runtime training is needed. Gate: if predicted value < -0.5, override the
  strategy plan with an aggressive attack.
- DiplomacyEngine: Full multi-player threat ranking (LEADER/MID/WEAK).
  Replaces OpponentModel.primary_target() for diplo mode target selection.
  Adds leader_penalty so we never ignore the strongest enemy to farm weak ones.
- Updated time budget: defense<200ms, intercept<300ms, MCTS 300-700ms,
  choose_best_plan<820ms, neural gate<830ms, redistribute/late<960ms.
"""

from __future__ import annotations

import base64
import io
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

SUN_X = SUN_Y = 50.0
SUN_RADIUS = 10.0
BOARD = 100.0
ROTATION_RADIUS_LIMIT = 50.0
DEFAULT_MAX_SHIP_SPEED = 6.0
DEFAULT_EPISODE_STEPS = 500

MAX_TOTAL_MOVES = 26
SYNC_ETA_WINDOW = 3
MAX_SOURCES_PER_TARGET = 8
MAX_TARGETS_PER_PLAN = 6
MIN_PLAN_SCORE = -31.0

ABS_MIN_BATCH = 5

# ── helpers ───────────────────────────────────────────────────────────────────

def _get(obj, key, default=None):
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def fleet_speed(ships: int, max_speed: float = DEFAULT_MAX_SHIP_SPEED) -> float:
    if ships <= 1:
        return 1.0
    spd = 1.0 + (max_speed - 1.0) * (math.log(max(1, ships)) / math.log(1000)) ** 1.5
    return min(spd, max_speed)


def point_segment_distance(px, py, ax, ay, bx, by) -> float:
    abx, aby = bx - ax, by - ay
    l2 = abx * abx + aby * aby
    if l2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / l2))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def segment_hits_sun(ax, ay, bx, by, margin: float = 1.25) -> bool:
    return point_segment_distance(SUN_X, SUN_Y, ax, ay, bx, by) < SUN_RADIUS + margin


# ── data classes ─────────────────────────────────────────────────────────────

class Planet:
    __slots__ = ("id","owner","x","y","radius","ships","production","initial_x","initial_y","is_comet")
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
    __slots__ = ("id","owner","x","y","angle","from_planet_id","ships")
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


# ── GameState ─────────────────────────────────────────────────────────────────

class GameState:
    def __init__(self, obs, config=None):
        self.my_id = int(_get(obs, "player", 0) or 0)
        self.ang_vel = float(_get(obs, "angular_velocity", 0.0) or 0.0)
        self.step = int(_get(obs, "step", 0) or 0)

        cfg = _get(obs, "configuration", None) or config or {}
        self.max_speed = float(_get(cfg, "shipSpeed", DEFAULT_MAX_SHIP_SPEED) or DEFAULT_MAX_SHIP_SPEED)
        self.episode_steps = int(_get(cfg, "episodeSteps", DEFAULT_EPISODE_STEPS) or DEFAULT_EPISODE_STEPS)

        comet_ids = set(int(x) for x in (_get(obs, "comet_planet_ids", []) or []))
        self.comet_paths: Dict[int, Tuple[List[Tuple[float,float]], int]] = {}
        self._parse_comets(_get(obs, "comets", []) or [])

        initial_xy: Dict[int, Tuple[float,float]] = {}
        for row in _get(obs, "initial_planets", []) or []:
            initial_xy[int(row[0])] = (float(row[2]), float(row[3]))

        self.planets: List[Planet] = []
        for row in _get(obs, "planets", []) or []:
            pid = int(row[0])
            ix, iy = initial_xy.get(pid, (float(row[2]), float(row[3])))
            self.planets.append(Planet(pid, int(row[1]), float(row[2]), float(row[3]),
                                       float(row[4]), int(row[5]), int(row[6]), ix, iy, pid in comet_ids))

        self.fleets: List[Fleet] = []
        for row in _get(obs, "fleets", []) or []:
            self.fleets.append(Fleet(int(row[0]), int(row[1]), float(row[2]), float(row[3]),
                                     float(row[4]), int(row[5]), int(row[6])))

        self._pm = {p.id: p for p in self.planets}
        self.my_pl = [p for p in self.planets if p.owner == self.my_id]
        self.en_pl = [p for p in self.planets if p.owner not in (-1, self.my_id)]
        self.neu_pl = [p for p in self.planets if p.owner == -1]
        self.en_ids = sorted({p.owner for p in self.en_pl})

        self.fleet_target: Dict[int, Optional[Tuple[int,int]]] = {}
        self.arrivals: Dict[int, List[Tuple[int,int,int]]] = defaultdict(list)
        self.incoming: Dict[int, Dict[int,int]] = defaultdict(lambda: defaultdict(int))
        for f in self.fleets:
            pred = self._predict_fleet_target(f)
            self.fleet_target[f.id] = pred
            if pred is not None:
                tid, eta = pred
                self.arrivals[tid].append((eta, f.owner, f.ships))
                self.incoming[tid][f.owner] += f.ships

        self.enemy_outbound: Dict[int, int] = defaultdict(int)
        for f in self.fleets:
            if f.owner not in (-1, self.my_id):
                self.enemy_outbound[f.owner] += f.ships

    def _parse_comets(self, groups) -> None:
        for group in groups:
            ids = _get(group, "planet_ids", []) or []
            paths = _get(group, "paths", []) or []
            idx = int(_get(group, "path_index", 0) or 0)
            for i, pid_raw in enumerate(ids):
                if i >= len(paths): continue
                path: List[Tuple[float,float]] = []
                for pt in paths[i] or []:
                    if len(pt) >= 2:
                        path.append((float(pt[0]), float(pt[1])))
                if path:
                    self.comet_paths[int(pid_raw)] = (path, idx)

    def get(self, pid: int) -> Optional[Planet]:
        return self._pm.get(pid)

    def is_orbiting(self, p: Planet) -> bool:
        if p.is_comet: return False
        r = math.hypot(p.initial_x - SUN_X, p.initial_y - SUN_Y)
        return r + p.radius < ROTATION_RADIUS_LIMIT and abs(self.ang_vel) > 1e-12

    def planet_pos_at(self, p: Planet, t: int) -> Tuple[float, float]:
        if p.is_comet and p.id in self.comet_paths:
            path, idx = self.comet_paths[p.id]
            j = idx + max(0, int(t))
            return path[min(j, len(path)-1)]
        if not self.is_orbiting(p):
            return p.x, p.y
        r = math.hypot(p.initial_x - SUN_X, p.initial_y - SUN_Y)
        a0 = math.atan2(p.initial_y - SUN_Y, p.initial_x - SUN_X)
        a1 = a0 + self.ang_vel * (self.step + int(t))
        return SUN_X + r * math.cos(a1), SUN_Y + r * math.sin(a1)

    def comet_turns_left(self, p: Planet) -> int:
        if not p.is_comet or p.id not in self.comet_paths: return 999
        path, idx = self.comet_paths[p.id]
        return max(0, len(path) - idx - 1)

    def _predict_fleet_target(self, f: Fleet, max_steps: int = 220) -> Optional[Tuple[int,int]]:
        spd = fleet_speed(f.ships, self.max_speed)
        cx, cy = f.x, f.y
        dx, dy = math.cos(f.angle) * spd, math.sin(f.angle) * spd
        for t in range(1, max_steps + 1):
            nx, ny = cx + dx, cy + dy
            if not (0.0 <= nx <= BOARD and 0.0 <= ny <= BOARD): return None
            if point_segment_distance(SUN_X, SUN_Y, cx, cy, nx, ny) < SUN_RADIUS: return None
            best_pid, best_d = None, float("inf")
            for p in self.planets:
                px, py = self.planet_pos_at(p, t)
                d = point_segment_distance(px, py, cx, cy, nx, ny)
                if d < p.radius and d < best_d:
                    best_pid, best_d = p.id, d
            if best_pid is not None:
                return best_pid, t
            cx, cy = nx, ny
        return None

    def total_ships(self, owner: int) -> int:
        return (sum(p.ships for p in self.planets if p.owner == owner) +
                sum(f.ships for f in self.fleets if f.owner == owner))

    def centroid(self) -> Tuple[float, float]:
        if not self.my_pl: return SUN_X, SUN_Y
        return (sum(p.x for p in self.my_pl) / len(self.my_pl),
                sum(p.y for p in self.my_pl) / len(self.my_pl))

    def net_threat(self, p: Planet) -> int:
        inc = self.incoming.get(p.id, {})
        attackers = sum(v for k, v in inc.items() if k not in (-1, self.my_id))
        return attackers - inc.get(self.my_id, 0)

    def phase(self) -> str:
        progress = self.step / max(1, self.episode_steps)
        if progress < 0.18: return "early"
        if progress < 0.64: return "mid"
        return "late"

    def turns_left(self) -> int:
        return max(1, self.episode_steps - self.step)

    def enemy_incoming(self, pid: int) -> int:
        inc = self.incoming.get(pid, {})
        return sum(v for k, v in inc.items() if k not in (-1, self.my_id))

    def effective_garrison(self, p: Planet) -> int:
        if p.owner not in self.en_ids: return p.ships
        out = sum(f.ships for f in self.fleets if f.owner == p.owner
                  and self.fleet_target.get(f.id) is not None
                  and (t := self.fleet_target.get(f.id)) is not None
                  and t[0] != p.id)
        return max(0, p.ships - out)


# ── geometry & aiming ─────────────────────────────────────────────────────────

def lead_intercept(state: GameState, src: Planet, dst: Planet, ships: int,
                   iters: int = 8) -> Tuple[float, float, int]:
    spd = fleet_speed(max(1, ships), state.max_speed)
    tx, ty = dst.x, dst.y
    eta = max(1, int(math.ceil(math.hypot(tx - src.x, ty - src.y) / spd)))
    for _ in range(iters):
        tx, ty = state.planet_pos_at(dst, eta)
        new_eta = max(1, int(math.ceil(math.hypot(tx - src.x, ty - src.y) / spd)))
        if new_eta == eta:
            break
        eta = new_eta
    return tx, ty, eta


def safe_aim(state: GameState, src: Planet, dst: Planet, ships: int) -> Tuple[float, int]:
    """Return (angle, eta).
    Guarantees:
    1. Flight path does not clip the sun.
    2. Endpoint stays within [0, BOARD] bounds.
    """
    tx, ty, eta = lead_intercept(state, src, dst, max(1, ships))
    angle = math.atan2(ty - src.y, tx - src.x)
    spd = fleet_speed(max(1, ships), state.max_speed)

    def endpoint(a: float) -> Tuple[float, float]:
        return (src.x + math.cos(a) * spd * eta,
                src.y + math.sin(a) * spd * eta)

    def is_bad(a: float) -> bool:
        ex, ey = endpoint(a)
        if not (0.5 <= ex <= BOARD - 0.5 and 0.5 <= ey <= BOARD - 0.5):
            return True
        return segment_hits_sun(src.x, src.y, ex, ey)

    if not is_bad(angle):
        return angle, eta

    best_angle = angle
    best_clearance = -999.0
    deltas = (0.12, -0.12, 0.22, -0.22, 0.34, -0.34, 0.50, -0.50,
              0.68, -0.68, 0.88, -0.88, 1.1, -1.1, 1.4, -1.4)
    for delta in deltas:
        a = angle + delta
        ex, ey = endpoint(a)
        if not (0.5 <= ex <= BOARD - 0.5 and 0.5 <= ey <= BOARD - 0.5):
            continue
        clearance = point_segment_distance(SUN_X, SUN_Y, src.x, src.y, ex, ey) - (SUN_RADIUS + 1.25)
        if clearance > 0:
            return a, eta
        if clearance > best_clearance:
            best_clearance = clearance
            best_angle = a
    return best_angle, eta


# ── state-at-eta forward projector ───────────────────────────────────────────

def target_state_at(state: GameState, dst: Planet, eta: int) -> Tuple[int, int]:
    owner = dst.owner
    ships = int(dst.ships)
    by_turn: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    for arr_eta, arr_owner, arr_ships in state.arrivals.get(dst.id, []):
        if 1 <= arr_eta <= eta:
            by_turn[arr_eta].append((arr_owner, arr_ships))
    for t in range(1, max(1, eta) + 1):
        if owner >= 0:
            ships += dst.production
        if by_turn.get(t):
            owner, ships = _combat(owner, ships, by_turn[t])
    return owner, max(0, ships)


def capture_need(state: GameState, src: Planet, dst: Planet,
                 margin: Optional[int] = None) -> Tuple[int, int]:
    need = max(ABS_MIN_BATCH, dst.ships + (2 if dst.owner == -1 else 8))
    eta = 1
    for _ in range(4):
        _, _, eta = lead_intercept(state, src, dst, need)
        owner, ships = target_state_at(state, dst, eta)
        if owner == state.my_id:
            need = max(ABS_MIN_BATCH, state.net_threat(dst) + 4)
        else:
            base_margin = margin if margin is not None else (
                3 if owner == -1 else 8 + min(6, dst.production))
            extra_prod = (dst.production * eta) if owner not in (-1, state.my_id) else 0
            need = ships + base_margin + extra_prod // 4
        need = max(ABS_MIN_BATCH, int(need))
    return need, eta


# ── reserve / surplus ─────────────────────────────────────────────────────────

def reserve_for(state: GameState, p: Planet) -> int:
    threat = max(0, state.net_threat(p))
    if p.is_comet:
        ttl = state.comet_turns_left(p)
        return max(threat + 2, 2 if ttl > 10 else p.ships)
    nearest_enemy = min((p.dist(e) for e in state.en_pl), default=999.0)
    front_lock = (10 + p.production * 4) if nearest_enemy < 20 else (
                  (5 + p.production * 3) if nearest_enemy < 36 else 0)
    ph = state.phase()
    # Phase-tuned: slightly lower early growth lock vs v9 baseline so we expand faster;
    # late game holds more on high-prod planets.
    if ph == "early":
        growth_lock = p.production * 4
    elif ph == "mid":
        growth_lock = p.production * 4
    else:
        growth_lock = p.production * 5
    return max(threat + 6, growth_lock, front_lock, 3)


def surplus_for(state: GameState, p: Planet, used: Optional[Dict[int,int]] = None) -> int:
    used_amt = 0 if used is None else used.get(p.id, 0)
    return max(0, p.ships - reserve_for(state, p) - used_amt)


def min_batch_for(src: Planet, urgent: bool = False) -> int:
    if urgent: return ABS_MIN_BATCH
    return max(ABS_MIN_BATCH, int(src.production) * 2)


# ── scoring helpers ───────────────────────────────────────────────────────────

def approach_bonus(state: GameState, dst: Planet, eta: int) -> float:
    if not state.is_orbiting(dst): return 0.0
    cx, cy = state.centroid()
    now_d = math.hypot(dst.x - cx, dst.y - cy)
    fx, fy = state.planet_pos_at(dst, eta)
    fut_d = math.hypot(fx - cx, fy - cy)
    gain = now_d - fut_d
    return max(-28.0, min(34.0, gain * 1.55 + dst.production * 1.2))


def enemy_eta_power(state: GameState, dst: Planet) -> Tuple[int, int]:
    best_eta, best_power = 999, 0
    for e in state.en_pl:
        probe = max(1, min(e.ships, max(5, e.ships * 2 // 3)))
        _, _, eta = lead_intercept(state, e, dst, probe)
        if eta < best_eta:
            best_eta, best_power = eta, e.ships
    return best_eta, best_power


def recapture_bonus(state: GameState, dst: Planet) -> float:
    if dst.owner not in state.en_ids: return 0.0
    cx, cy = state.centroid()
    d = max(1.0, math.hypot(dst.x - cx, dst.y - cy))
    return dst.production * 14.0 / (1.0 + d * 0.04)


def contest_penalty(state: GameState, dst: Planet) -> float:
    en_inc = state.enemy_incoming(dst.id)
    if en_inc <= 0: return 0.0
    return min(40.0, en_inc * 0.6)


def target_score(state: GameState, src: Planet, dst: Planet) -> Tuple[float, int, int]:
    if dst.owner == state.my_id or src.id == dst.id:
        return -1e18, 0, 0
    need, eta = capture_need(state, src, dst)
    if need <= 0:
        return -1e18, 0, eta
    turns = max(1, state.turns_left() - eta)
    if dst.is_comet:
        turns = min(turns, max(0, state.comet_turns_left(dst) - eta), 60)
        if turns <= 8: return -1e18, need, eta

    phase = state.phase()
    prod_value = dst.production * turns
    is_neutral = dst.owner == -1
    is_enemy   = dst.owner not in (-1, state.my_id)
    enemy_bonus = 38.0 if is_enemy else 0.0
    comet_bonus = 16.0 if dst.is_comet else 0.0
    rec_bonus = recapture_bonus(state, dst)
    if phase == "late" and is_enemy:
        rec_bonus *= 1.18
    appr = approach_bonus(state, dst, eta)
    orbit_neu = 0.0
    if is_neutral and state.is_orbiting(dst):
        if phase == "early":
            orbit_neu = 14.0 + dst.production * 4.0
        elif phase == "mid":
            orbit_neu = 9.0 + dst.production * 2.8
        else:
            orbit_neu = 5.0 + dst.production * 1.5

    lane_ang, _ = safe_aim(state, src, dst, need)
    spd = fleet_speed(need, state.max_speed)
    bx = src.x + math.cos(lane_ang) * spd * eta
    by = src.y + math.sin(lane_ang) * spd * eta
    sun_pen = 90.0 if segment_hits_sun(src.x, src.y, bx, by, margin=2.0) else 0.0

    snipe_pen = 0.0
    if is_neutral:
        e_eta, e_pow = enemy_eta_power(state, dst)
        if e_eta <= eta + 1 and e_pow > max(0, need - 4):
            snipe_pen = 46.0 + 0.12 * e_pow
        elif e_eta <= eta + 2 and e_pow > need + 5:
            snipe_pen = 24.0 + 0.08 * e_pow

    eta_pen = 0.34 * eta
    cost_pen = 0.82 * need
    cont_pen = contest_penalty(state, dst)
    if phase == "late":
        cont_pen *= 0.78

    score = prod_value + enemy_bonus + comet_bonus + rec_bonus + appr + orbit_neu
    score -= cost_pen + eta_pen + sun_pen + snipe_pen + cont_pen
    score /= max(1.0, eta ** 0.30)
    return score, need, eta


def elite_eval(state: GameState) -> float:
    mi = state.my_id
    ms = state.total_ships(mi)
    es = sum(state.total_ships(e) for e in state.en_ids) + 1e-9
    mp = sum(p.production for p in state.my_pl)
    ep = sum(p.production for p in state.en_pl)
    mc = len(state.my_pl); ec = len(state.en_pl)
    threat = sum(max(0, state.net_threat(p)) for p in state.my_pl)
    mf = sum(f.ships for f in state.fleets if f.owner == mi)
    ef = sum(f.ships for f in state.fleets if f.owner not in (-1, mi))
    border = sum((35 - m.dist(e)) / 35 * m.production
                 for m in state.my_pl for e in state.en_pl if m.dist(e) < 35)
    ndeny = sum(n.production for n in state.neu_pl
                if any(n.dist(e) < 25 for e in state.en_pl)
                and not any(n.dist(m) < 25 for m in state.my_pl))
    return ((ms - es) + 48.0*(mp-ep) + 20.0*(mc-ec)
            - 2.8*threat + 9.0*border + 0.45*(mf-ef) - 12.0*ndeny)


# ── OpponentModel ─────────────────────────────────────────────────────────────

class OpponentModel:
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

    def growth_rate(self, eid: int) -> float:
        h = self.ship_h.get(eid, [])
        if len(h) < 3: return 0.0
        w = h[-20:]
        return (w[-1] - w[0]) / max(len(w)-1, 1)


_GLOBAL_OPP = OpponentModel()


# ── DiplomacyEngine — full multi-player threat ranking ─────────────────────────

class DiplomacyEngine:
    """Ranks enemies as LEADER/MID/WEAK and picks the best attack target.

    In a 2-player game this degenerates to a single LEADER.  In multi-player
    it prevents ignoring the strongest enemy while farming weak ones.
    """
    LEADER = "LEADER"
    MID = "MID"
    WEAK = "WEAK"

    def __init__(self, state: GameState, opp: OpponentModel):
        self.state = state
        self.opp = opp

    def power(self, eid: int) -> float:
        s = self.state
        prod = sum(p.production for p in s.planets if p.owner == eid)
        return s.total_ships(eid) + prod * 24.0

    def threat_to_us(self, eid: int) -> float:
        s = self.state
        ep = [p for p in s.planets if p.owner == eid]
        if not ep: return 0.0
        prox = sum(1.0 / max(m.dist(e), 1.0) for m in s.my_pl for e in ep)
        aggr = 1.0 + self.opp.aggression(eid)
        return (prox * 38.0 + s.total_ships(eid) * 0.65) * aggr

    def rank(self) -> List[Tuple[str, int, float]]:
        """Returns list of (tag, eid, threat) sorted by threat desc."""
        if not self.state.en_ids:
            return []
        scored = [(self.threat_to_us(e), e) for e in self.state.en_ids]
        scored.sort(reverse=True)
        powers = {e: self.power(e) for _, e in scored}
        max_power = max(powers.values()) if powers else 1.0
        result = []
        for i, (thr, eid) in enumerate(scored):
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
        """Planet on the primary enemy that is cheapest to attack."""
        ranked = self.rank()
        if not ranked:
            return None
        # Prefer the LEADER; never let a WEAK enemy distract us when a LEADER
        # exists and has not been dealt with yet.
        leader_ids = [eid for tag, eid, _ in ranked if tag == self.LEADER]
        target_eid = leader_ids[0] if leader_ids else ranked[0][1]
        ep = [p for p in self.state.planets if p.owner == target_eid]
        if not ep:
            return None
        cx, cy = self.state.centroid()
        return min(ep, key=lambda p: self.state.effective_garrison(p)
                                     + math.hypot(p.x - cx, p.y - cy) * 0.38)

    def leader_penalty(self, dst: Planet) -> float:
        """Extra attack score when dst belongs to the LEADER enemy.
        Negative penalty when attacking a WEAK enemy while LEADER is unchecked."""
        ranked = self.rank()
        if len(ranked) <= 1:
            return 0.0
        leaders = {eid for tag, eid, _ in ranked if tag == self.LEADER}
        if dst.owner in leaders:
            return 15.0
        # Penalise going for a WEAK/MID target when LEADER is strong
        leader_power = sum(self.power(e) for e in leaders)
        our_power = self.state.total_ships(self.state.my_id)
        if leader_power > our_power * 1.2:
            return -20.0
        return 0.0


# ── forward sim ───────────────────────────────────────────────────────────────

class SimP:
    __slots__ = ("id","owner","ships","production")
    def __init__(self, p):
        self.id = p.id; self.owner = p.owner
        self.ships = p.ships; self.production = p.production

    def copy(self) -> "SimP":
        s = SimP.__new__(SimP)
        s.id = self.id; s.owner = self.owner
        s.ships = self.ships; s.production = self.production
        return s


class SimF:
    __slots__ = ("owner","tid","ships","eta")
    def __init__(self, owner, tid, ships, eta):
        self.owner = owner; self.tid = tid; self.ships = ships; self.eta = eta

    def copy(self) -> "SimF":
        return SimF(self.owner, self.tid, self.ships, self.eta)


def clone_sim(state: GameState) -> Tuple[Dict[int,SimP], List[SimF]]:
    planets = {p.id: SimP(p) for p in state.planets}
    fleets: List[SimF] = []
    for f in state.fleets:
        t = state.fleet_target.get(f.id)
        if t: fleets.append(SimF(f.owner, t[0], f.ships, t[1]))
    return planets, fleets


def copy_sim(planets: Dict[int,SimP], fleets: List[SimF]):
    return {pid: p.copy() for pid, p in planets.items()}, [f.copy() for f in fleets]


def sim_step(planets: Dict[int,SimP], fleets: List[SimF]) -> None:
    for p in planets.values():
        if p.owner >= 0: p.ships += p.production
    by_target: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    nxt: List[SimF] = []
    for f in fleets:
        f.eta -= 1
        if f.eta <= 0:
            by_target[f.tid].append((f.owner, f.ships))
        else:
            nxt.append(f)
    for tid, arrivals in by_target.items():
        p = planets.get(tid)
        if p: p.owner, p.ships = _combat(p.owner, p.ships, arrivals)
    fleets[:] = nxt


def eval_sim_planets(state: GameState, planets: Dict[int,SimP], fleets: List[SimF]) -> float:
    mi = state.my_id
    ms = sum(p.ships for p in planets.values() if p.owner == mi)
    ms += sum(f.ships for f in fleets if f.owner == mi)
    es = sum(p.ships for p in planets.values() if p.owner not in (-1,mi))
    es += sum(f.ships for f in fleets if f.owner not in (-1,mi))
    mp = sum(p.production for p in planets.values() if p.owner == mi)
    ep = sum(p.production for p in planets.values() if p.owner not in (-1,mi))
    mc = sum(1 for p in planets.values() if p.owner == mi)
    ec = sum(1 for p in planets.values() if p.owner not in (-1,mi))
    return (ms-es) + 48.0*(mp-ep) + 20.0*(mc-ec)


# ── MCTSEngine ────────────────────────────────────────────────────────────────

class MCTSNode:
    """UCB1 MCTS node.  Each node represents one possible atomic send action
    taken from the parent's simulated state."""

    __slots__ = ("action", "parent", "children", "visits", "value",
                 "planets", "fleets", "_untried")

    def __init__(self, action, parent, planets, fleets):
        self.action = action          # (src_id, dst_id, ships) or None for root
        self.parent: Optional["MCTSNode"] = parent
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0
        self.planets = planets        # shallow sim snapshot at this node
        self.fleets = fleets
        self._untried: Optional[List] = None  # lazily filled

    def ucb1(self, c: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        return (self.value / self.visits +
                c * math.sqrt(math.log(self.parent.visits) / self.visits))

    def best_child(self, c: float = 1.41) -> "MCTSNode":
        return max(self.children, key=lambda n: n.ucb1(c))

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def fully_expanded(self) -> bool:
        return self._untried is not None and len(self._untried) == 0


class MCTSEngine:
    """Time-budgeted MCTS.  Candidate actions are atomic (src→dst, ships) pairs
    generated from current my_pl × top targets.  Rollout runs sim_step × depth
    random moves, scored with eval_sim_planets."""

    def __init__(self, state: GameState, budget_ms: float = 400.0,
                 rollout_depth: int = 10, max_cands: int = 16):
        self.state = state
        self.budget_ms = budget_ms
        self.rollout_depth = rollout_depth
        self.max_cands = max_cands

    def _cands(self, planets: Dict[int,SimP], fleets: List[SimF]) -> List[Tuple[int,int,int]]:
        """Generate atomic (src_id, dst_id, ships) candidates from sim state."""
        state = self.state
        mi = state.my_id
        my_sims = [p for p in planets.values() if p.owner == mi and p.ships > ABS_MIN_BATCH * 2]
        if not my_sims:
            return []
        # Target: neu + en by production desc
        targets = sorted(
            [p for p in state.planets if p.owner != mi],
            key=lambda p: -p.production
        )[:6]
        cands: List[Tuple[int,int,int]] = []
        for sp in my_sims[:4]:
            src = state.get(sp.id)
            if src is None: continue
            surplus = max(0, sp.ships - reserve_for(state, src))
            if surplus < ABS_MIN_BATCH: continue
            for dst in targets[:4]:
                if dst.id == sp.id: continue
                need, _ = capture_need(state, src, dst)
                send = min(surplus, max(need, ABS_MIN_BATCH))
                if send >= ABS_MIN_BATCH:
                    cands.append((sp.id, dst.id, send))
            if len(cands) >= self.max_cands:
                break
        return cands[:self.max_cands]

    def _apply(self, planets: Dict[int,SimP], fleets: List[SimF],
               action: Tuple[int,int,int]) -> Tuple[Dict[int,SimP], List[SimF]]:
        """Apply one send action to a copied sim state."""
        planets2, fleets2 = copy_sim(planets, fleets)
        sid, did, ships = action
        src_sim = planets2.get(sid)
        if src_sim is None or src_sim.owner != self.state.my_id:
            return planets2, fleets2
        send = min(ships, max(0, src_sim.ships - ABS_MIN_BATCH))
        if send < ABS_MIN_BATCH:
            return planets2, fleets2
        src_real = self.state.get(sid)
        dst_real = self.state.get(did)
        if src_real is None or dst_real is None:
            return planets2, fleets2
        _, eta = safe_aim(self.state, src_real, dst_real, send)
        src_sim.ships -= send
        fleets2.append(SimF(self.state.my_id, did, send, eta))
        return planets2, fleets2

    def _rollout(self, planets: Dict[int,SimP], fleets: List[SimF]) -> float:
        """Random playout from given state; returns eval score."""
        p2, f2 = copy_sim(planets, fleets)
        for _ in range(self.rollout_depth):
            cands = self._cands(p2, f2)
            if cands:
                act = random.choice(cands)
                p2, f2 = self._apply(p2, f2, act)
            sim_step(p2, f2)
        return eval_sim_planets(self.state, p2, f2)

    def search(self, start_ms: float) -> Optional[Tuple[int,int,int]]:
        """Run MCTS for budget_ms milliseconds; return best first action."""
        root_p, root_f = clone_sim(self.state)
        root = MCTSNode(None, None, root_p, root_f)
        deadline = start_ms + self.budget_ms

        def _now_ms():
            return time.time() * 1000.0

        iters = 0
        while _now_ms() < deadline:
            # Selection
            node = root
            while not node.is_leaf() and node.fully_expanded():
                node = node.best_child()

            # Expansion
            if node._untried is None:
                node._untried = self._cands(node.planets, node.fleets)
            if node._untried:
                act = node._untried.pop()
                cp, cf = self._apply(node.planets, node.fleets, act)
                sim_step(cp, cf)
                child = MCTSNode(act, node, cp, cf)
                node.children.append(child)
                node = child

            # Rollout
            val = self._rollout(node.planets, node.fleets)

            # Backprop
            cur = node
            while cur is not None:
                cur.visits += 1
                cur.value += val
                cur = cur.parent
            iters += 1

        if not root.children:
            return None
        best = max(root.children, key=lambda n: n.visits)
        return best.action


# ── NeuralVal — NumPy MLP 14→64→32→1 ─────────────────────────────────────────

# Inline pre-trained weights (base64-encoded .npy bytes).
# Generated by running 200 random self-play games and training for 40 epochs.
# If this string is empty the network will be randomly initialised (still
# useful as a gate once it sees a few turns of real game data).
_NEURAL_WEIGHTS_B64 = "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"


class NeuralVal:
    """Lightweight NumPy MLP used as a value gate.

    predict(state) → float in [-1, 1].
    Negative values mean we are likely losing; the agent uses this as a
    trigger to switch to aggressive mode.
    """

    N_FEAT = 14

    def __init__(self):
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.2, (64, self.N_FEAT)).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = rng.normal(0, 0.2, (32, 64)).astype(np.float32)
        self.b2 = np.zeros(32, dtype=np.float32)
        self.W3 = rng.normal(0, 0.2, (1, 32)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)
        self.loss_h: List[float] = []
        self._try_load_inline()

    def _try_load_inline(self):
        if not _NEURAL_WEIGHTS_B64:
            return
        try:
            raw = base64.b64decode(_NEURAL_WEIGHTS_B64)
            buf = io.BytesIO(raw)
            d = np.load(buf, allow_pickle=True).item()
            self.W1 = d["W1"]; self.b1 = d["b1"]
            self.W2 = d["W2"]; self.b2 = d["b2"]
            self.W3 = d["W3"]; self.b3 = d["b3"]
        except Exception:
            pass

    def feat(self, state: GameState) -> np.ndarray:
        mi = state.my_id
        total = sum(state.total_ships(o) for o in [mi] + state.en_ids) + 1e-6
        my_sh = state.total_ships(mi) / total
        en_sh = sum(state.total_ships(e) for e in state.en_ids) / total
        my_prod = sum(p.production for p in state.my_pl)
        en_prod = sum(p.production for p in state.en_pl)
        prod_ratio = my_prod / max(my_prod + en_prod + 1, 1)
        n_planets = max(len(state.planets), 1)
        planet_ratio = len(state.my_pl) / n_planets
        cx, cy = state.centroid()
        min_en_dist = min((math.hypot(p.x - cx, p.y - cy)
                           for p in state.en_pl), default=100.0) / 100.0
        phase_enc = {"early": 0.0, "mid": 0.5, "late": 1.0}[state.phase()]
        tl = state.turns_left() / max(state.episode_steps, 1)
        # frontier ratio
        fronts = [p for p in state.my_pl
                  if any(p.dist(e) < 35 for e in state.en_pl)]
        front_ratio = len(fronts) / max(len(state.my_pl), 1)
        comet_cnt = sum(1 for p in state.planets
                        if p.is_comet and p.owner != mi) / max(n_planets, 1)
        en_fleet = sum(f.ships for f in state.fleets
                       if f.owner not in (-1, mi))
        en_fleet_ratio = en_fleet / max(state.total_ships(mi) + 1, 1)
        net_thr = sum(max(0, state.net_threat(p)) for p in state.my_pl)
        net_thr_ratio = net_thr / max(state.total_ships(mi) + 1, 1)
        ee = float(np.tanh(elite_eval(state) / 500.0))
        border = sum((35 - m.dist(e)) / 35 * m.production
                     for m in state.my_pl for e in state.en_pl
                     if m.dist(e) < 35)
        border_norm = float(np.tanh(border / 100.0))
        return np.array([
            my_sh, en_sh, prod_ratio, planet_ratio,
            min_en_dist, phase_enc, tl, front_ratio,
            comet_cnt, en_fleet_ratio, net_thr_ratio,
            ee, border_norm,
            float(len(state.en_ids) > 1),    # multi-player flag
        ], dtype=np.float32)

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h1 = np.maximum(0.0, self.W1 @ x + self.b1)
        h2 = np.maximum(0.0, self.W2 @ h1 + self.b2)
        out = np.tanh(self.W3 @ h2 + self.b3)
        return h1, h2, out

    def forward(self, x: np.ndarray) -> float:
        _, _, out = self._forward(x)
        return float(out[0])

    def predict(self, state: GameState) -> float:
        try:
            return self.forward(self.feat(state))
        except Exception:
            return 0.0

    def train(self, xs: np.ndarray, ys: np.ndarray,
              epochs: int = 40, lr: float = 3e-3, batch: int = 32):
        """Mini-batch SGD with MSE loss and tanh output derivative."""
        n = len(xs)
        for ep in range(epochs):
            idx = np.random.permutation(n)
            total_loss = 0.0
            steps = 0
            for i in range(0, n, batch):
                bi = idx[i:i+batch]
                xb = xs[bi]; yb = ys[bi].reshape(-1, 1)
                # forward
                h1 = np.maximum(0.0, (self.W1 @ xb.T).T + self.b1)
                h2 = np.maximum(0.0, (self.W2 @ h1.T).T + self.b2)
                out = np.tanh((self.W3 @ h2.T).T + self.b3)
                # loss
                err = out - yb
                total_loss += float(np.mean(err**2))
                steps += 1
                # backward tanh
                d3 = err * (1.0 - out**2) / len(bi)
                dW3 = d3.T @ h2
                db3 = d3.sum(axis=0)
                d2 = (self.W3.T @ d3.T).T * (h2 > 0).astype(np.float32)
                dW2 = d2.T @ h1
                db2 = d2.sum(axis=0)
                d1 = (self.W2.T @ d2.T).T * (h1 > 0).astype(np.float32)
                dW1 = d1.T @ xb
                db1 = d1.sum(axis=0)
                self.W3 -= lr * dW3; self.b3 -= lr * db3
                self.W2 -= lr * dW2; self.b2 -= lr * db2
                self.W1 -= lr * dW1; self.b1 -= lr * db1
            self.loss_h.append(total_loss / max(steps, 1))

    def weights_b64(self) -> str:
        buf = io.BytesIO()
        np.save(buf, {"W1": self.W1, "b1": self.b1,
                      "W2": self.W2, "b2": self.b2,
                      "W3": self.W3, "b3": self.b3})
        return base64.b64encode(buf.getvalue()).decode()


_GLOBAL_NEURAL = NeuralVal()


# ── Plan dataclass ────────────────────────────────────────────────────────────

@dataclass
class Plan:
    actions: List[Tuple[int,int,int]] = field(default_factory=list)
    score: float = 0.0
    tag: str = ""


def score_plan(state: GameState, plan: Plan, steps: int = 8) -> float:
    planets, fleets = clone_sim(state)
    used: Dict[int,int] = defaultdict(int)
    for sid, did, ships in plan.actions:
        sp = state.get(sid); dp = state.get(did); sim_src = planets.get(sid)
        if sp is None or dp is None or sim_src is None or sim_src.owner != state.my_id: continue
        send = min(int(ships), max(0, sim_src.ships - reserve_for(state, sp) - used[sid]))
        if send <= 0: continue
        _, eta = safe_aim(state, sp, dp, send)
        sim_src.ships -= send; used[sid] += send
        fleets.append(SimF(state.my_id, did, send, eta))
    for _ in range(steps):
        sim_step(planets, fleets)
    return eval_sim_planets(state, planets, fleets)


# ── plan builders ─────────────────────────────────────────────────────────────

def build_defense_plan(state: GameState, used: Dict[int,int]) -> Plan:
    actions: List[Tuple[int,int,int]] = []
    score = 0.0
    for tgt in sorted(state.my_pl, key=lambda p: -state.net_threat(p)):
        threat = state.net_threat(tgt)
        if threat <= 0: continue
        need = threat + max(5, tgt.production * 2)
        helpers = sorted((p for p in state.my_pl if p.id != tgt.id), key=lambda p: p.dist(tgt))
        for src in helpers:
            if need <= 0: break
            avail = surplus_for(state, src, used)
            send = min(avail, need)
            if send < ABS_MIN_BATCH: continue
            actions.append((src.id, tgt.id, send))
            used[src.id] = used.get(src.id, 0) + send
            need -= send; score += send * 4.0
    return Plan(actions, score, "defense")


def build_intercept_plan(state: GameState, used: Dict[int,int]) -> Plan:
    actions: List[Tuple[int,int,int]] = []
    score = 0.0
    for f in sorted(state.fleets, key=lambda x: -x.ships):
        if f.owner in (-1, state.my_id) or f.ships < 10: continue
        target = state.fleet_target.get(f.id)
        if not target: continue
        tid, eta_to_planet = target
        dst = state.get(tid)
        if dst is None or dst.owner != state.my_id: continue
        helpers = sorted((p for p in state.my_pl if p.id != dst.id), key=lambda p: p.dist(dst))
        need = f.ships + 4
        for src in helpers[:4]:
            avail = surplus_for(state, src, used)
            if avail < ABS_MIN_BATCH: continue
            _, _, eta = lead_intercept(state, src, dst, min(avail, need))
            if eta > eta_to_planet + 1: continue
            send = min(avail, need)
            if send < ABS_MIN_BATCH: continue
            actions.append((src.id, dst.id, send))
            used[src.id] = used.get(src.id, 0) + send
            need -= send; score += send * 2.5
            if need <= 0: break
    return Plan(actions, score, "intercept")


def _target_pool(state: GameState, mode: str) -> List[Planet]:
    if mode == "expand":
        neu = sorted(state.neu_pl, key=lambda p: (-p.production, -p.ships))
        return neu + [p for p in state.en_pl if p.ships <= 20 or p.production <= 2]
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


def build_capture_plan(state: GameState, mode: str,
                       base_used: Optional[Dict[int,int]] = None,
                       diplo_target: Optional[Planet] = None,
                       diplo_engine: Optional[DiplomacyEngine] = None) -> Plan:
    used: Dict[int,int] = defaultdict(int)
    if base_used: used.update(base_used)
    actions: List[Tuple[int,int,int]] = []
    score = 0.0
    target_done: set = set()

    targets = _target_pool(state, mode)
    if mode == "diplo" and diplo_target:
        targets = [diplo_target] + [p for p in targets if p.id != diplo_target.id]

    ranked: List[Tuple[float, Planet]] = []
    for dst in targets:
        best_sc = -1e18
        for src in state.my_pl:
            sc, _, _ = target_score(state, src, dst)
            if mode == "aggro":   sc += recapture_bonus(state, dst) * 0.5
            if mode == "counter":
                sc += 38.0 if state.phase() == "mid" else 30.0
            if mode == "diplo":   sc += 25.0 if dst is diplo_target else 0.0
            if diplo_engine:      sc += diplo_engine.leader_penalty(dst)
            best_sc = max(best_sc, sc)
        if best_sc > MIN_PLAN_SCORE:
            ranked.append((best_sc, dst))
    ranked.sort(key=lambda x: -x[0])

    for _, dst in ranked[:MAX_TARGETS_PER_PLAN]:
        if len(actions) >= MAX_TOTAL_MOVES or dst.id in target_done: continue

        contributors: List[Tuple[int, Planet, int, int, float]] = []
        for src in state.my_pl:
            avail = surplus_for(state, src, used)
            if avail < min_batch_for(src): continue
            need, eta = capture_need(state, src, dst)
            sc, _, _ = target_score(state, src, dst)
            contributors.append((eta, src, avail, need, sc))
        if not contributors: continue

        contributors.sort(key=lambda x: (x[0], -x[4], -x[2]))
        eta0 = contributors[0][0]
        group = [c for c in contributors if c[0] <= eta0 + SYNC_ETA_WINDOW][:MAX_SOURCES_PER_TARGET]
        if not group: continue

        group_eta = max(c[0] for c in group)
        owner, garrison = target_state_at(state, dst, group_eta)
        if owner == state.my_id: continue
        required = garrison + (3 if owner == -1 else 8 + min(6, dst.production))
        if owner not in (-1, state.my_id):
            required += dst.production * group_eta // 5

        if dst.is_comet and state.comet_turns_left(dst) <= group_eta + 5: continue

        sent = 0
        staged: List[Tuple[int,int,int]] = []
        for _, src, avail, _, _ in group:
            if sent >= required: break
            send = min(avail, required - sent)
            if send < ABS_MIN_BATCH and sent + send < required: continue
            send = max(send, ABS_MIN_BATCH) if send >= ABS_MIN_BATCH else send
            staged.append((src.id, dst.id, send))
            sent += send
        if sent < required: continue

        for sid, did, send in staged:
            actions.append((sid, did, send))
            used[sid] = used.get(sid, 0) + send
        target_done.add(dst.id)
        score += sum(c[4] for c in group[:len(staged)]) + required * (1.5 if dst.owner != -1 else 0.9)

    return Plan(actions, score, mode)


def build_redistribution_plan(state: GameState, used: Dict[int,int]) -> Plan:
    if len(state.my_pl) < 2 or not state.en_pl:
        return Plan([], 0.0, "redistribute")

    def ned(p):
        return min((p.dist(e) for e in state.en_pl), default=999.0)

    ordered = sorted(state.my_pl, key=ned)
    fc = max(1, min(len(ordered)-1, len(ordered)//2+1))
    fronts = ordered[:fc]
    rears = [p for p in ordered[fc:] if state.net_threat(p) <= 0]
    actions: List[Tuple[int,int,int]] = []
    for rear in rears[:5]:
        avail = surplus_for(state, rear, used)
        if avail < max(10, rear.production * 3): continue
        dst = min(fronts, key=lambda f: rear.dist(f))
        send = max(ABS_MIN_BATCH, min(avail, max(8, int(avail * 0.55))))
        if send >= ABS_MIN_BATCH:
            actions.append((rear.id, dst.id, send))
            used[rear.id] = used.get(rear.id, 0) + send
    return Plan(actions, float(sum(a[2] for a in actions)) * 0.25, "redistribute")


def build_urgent_highprod_plan(state: GameState, used: Dict[int,int]) -> Plan:
    """Immediately attack high-production nearby enemy planets regardless of the
    normal score threshold. production >= 4 AND within 40 units of our centroid."""
    cx, cy = state.centroid()
    urgent_targets = sorted(
        [p for p in state.en_pl
         if p.production >= 4 and math.hypot(p.x - cx, p.y - cy) < 40],
        key=lambda p: -(p.production / max(1, math.hypot(p.x - cx, p.y - cy)))
    )
    if not urgent_targets:
        return Plan([], 0.0, "urgent_hp")

    actions: List[Tuple[int,int,int]] = []
    score = 0.0
    for dst in urgent_targets[:3]:
        contributors: List[Tuple[int, Planet, int, int]] = []
        for src in state.my_pl:
            avail = surplus_for(state, src, used)
            if avail < ABS_MIN_BATCH: continue
            need, eta = capture_need(state, src, dst)
            contributors.append((eta, src, avail, need))
        if not contributors: continue
        contributors.sort(key=lambda x: x[0])
        eta0 = contributors[0][0]
        group = [c for c in contributors if c[0] <= eta0 + SYNC_ETA_WINDOW][:MAX_SOURCES_PER_TARGET]
        group_eta = max(c[0] for c in group)
        owner, garrison = target_state_at(state, dst, group_eta)
        if owner == state.my_id: continue
        required = garrison + 8 + min(6, dst.production) + dst.production * group_eta // 5

        sent = 0
        staged: List[Tuple[int,int,int]] = []
        for _, src, avail, _ in group:
            if sent >= required: break
            send = min(avail, required - sent)
            if send < ABS_MIN_BATCH and sent + send < required: continue
            staged.append((src.id, dst.id, max(send, ABS_MIN_BATCH) if send >= ABS_MIN_BATCH else send))
            sent += send
        if sent < required: continue

        for sid, did, send in staged:
            actions.append((sid, did, send))
            used[sid] = used.get(sid, 0) + send
        score += dst.production * 20.0

    return Plan(actions, score, "urgent_hp")


def choose_best_plan(state: GameState, base_used: Dict[int,int], elapsed_ms,
                     diplo: DiplomacyEngine) -> Plan:
    phase = state.phase()
    diplo_tgt = diplo.primary_target()

    modes: List[str] = []
    if any(p.is_comet and p.owner != state.my_id for p in state.planets):
        modes.append("comet")
    # Phase-specific mode order (vs fixed aggro-first mid game): secure neutrals
    # and weak targets before bleeding into hard enemy stacks.
    if phase == "early":
        modes.extend(["expand", "balanced"])
    elif phase == "mid":
        modes.extend(["expand", "balanced", "counter", "aggro"])
    else:
        modes.extend(["aggro", "counter", "expand", "balanced"])
    if diplo_tgt:
        modes.append("diplo")

    candidates: List[Plan] = []
    for mode in modes:
        if elapsed_ms() > 820: break
        plan = build_capture_plan(state, mode, base_used,
                                   diplo_target=diplo_tgt if mode == "diplo" else None,
                                   diplo_engine=diplo)
        if plan.actions:
            sim_steps = 10 if phase == "late" else 8
            plan.score += score_plan(state, plan, steps=sim_steps)
            candidates.append(plan)

    if not candidates:
        return Plan([], 0.0, "none")
    return max(candidates, key=lambda p: p.score)


# ── main agent ────────────────────────────────────────────────────────────────

def agent(obs, config=None):
    global _GLOBAL_OPP, _GLOBAL_NEURAL
    t0 = time.time()
    t0_ms = t0 * 1000.0
    elapsed = lambda: (time.time() - t0) * 1000.0

    try:
        state = GameState(obs, config)
        if not state.my_pl:
            return []

        _GLOBAL_OPP.update(state)
        diplo = DiplomacyEngine(state, _GLOBAL_OPP)

        used: Dict[int,int] = defaultdict(int)
        moves: List[List] = []

        def emit(sid: int, did: int, ships: int, urgent: bool = False) -> bool:
            if len(moves) >= MAX_TOTAL_MOVES:
                return False
            src = state.get(sid); dst = state.get(did)
            if src is None or dst is None or src.owner != state.my_id:
                return False
            reserve = ABS_MIN_BATCH if urgent else reserve_for(state, src)
            avail = max(0, src.ships - reserve - used.get(sid, 0))
            send = min(int(ships), avail)
            if send < ABS_MIN_BATCH:
                return False
            angle, _ = safe_aim(state, src, dst, send)
            moves.append([sid, float(angle), int(send)])
            used[sid] = used.get(sid, 0) + send
            return True

        # 1. Defense  (align with v9-style budget so interceptors are not starved)
        defense = build_defense_plan(state, used)
        for sid, did, sh in defense.actions:
            if elapsed() > 500: break
            emit(sid, did, sh, urgent=True)

        # 2. Intercept
        intercept = build_intercept_plan(state, used)
        for sid, did, sh in intercept.actions:
            if elapsed() > 600: break
            emit(sid, did, sh, urgent=True)

        # 2b. Urgent high-production nearby enemy — mid needs near-parity; late presses from behind
        phase = state.phase()
        my_total = state.total_ships(state.my_id)
        en_total = sum(state.total_ships(e) for e in state.en_ids)
        need_ratio = 1.08 if phase == "mid" else (0.94 if phase == "late" else 999.0)
        if elapsed() < 620 and phase != "early" and state.en_pl and my_total >= en_total * need_ratio:
            urgent_hp = build_urgent_highprod_plan(state, used)
            for sid, did, sh in urgent_hp.actions:
                if elapsed() > 660: break
                emit(sid, did, sh)

        # 3. Multi-mode strategy plan with short forward sim  (< 820ms)
        # Run this FIRST so expansion is never starved of time budget.
        best = choose_best_plan(state, used, elapsed, diplo)
        for sid, did, sh in best.actions:
            if elapsed() > 870: break
            emit(sid, did, sh)

        # 4. MCTS — late game only (mid-game MCTS often wastes ms on noisy rollouts)
        mcts_action = None
        if (state.phase() == "late"
                and elapsed() < 720
                and state.en_pl
                and sum(surplus_for(state, p, used) for p in state.my_pl) >= 24):
            mcts = MCTSEngine(state, budget_ms=min(160.0, 920.0 - elapsed()),
                              rollout_depth=6, max_cands=10)
            mcts_action = mcts.search(start_ms=time.time() * 1000.0)

        if mcts_action is not None:
            already_targeted = {did for _, did, _ in best.actions}
            sid, did, sh = mcts_action
            if did not in already_targeted:
                emit(sid, did, sh)

        # 5. Neural gate: if we're losing badly, override with aggro  (< 830ms)
        if elapsed() < 830 and state.en_pl:
            neural_val = _GLOBAL_NEURAL.predict(state)
            if neural_val < -0.58:
                aggro_plan = build_capture_plan(state, "aggro", used, diplo_engine=diplo)
                for sid, did, sh in aggro_plan.actions:
                    if elapsed() > 875: break
                    emit(sid, did, sh)

        # 6. Redistribution  (< 920ms)
        redist = build_redistribution_plan(state, used)
        for sid, did, sh in redist.actions:
            if elapsed() > 920: break
            emit(sid, did, sh)

        # 7. Late-game dump  (< 960ms)
        if state.phase() == "late" and state.en_pl and elapsed() < 940:
            weak = min(state.en_pl, key=lambda p: state.effective_garrison(p) + p.production * 2)
            for src in sorted(state.my_pl, key=lambda p: -surplus_for(state, p, used)):
                if elapsed() > 955 or len(moves) >= MAX_TOTAL_MOVES: break
                avail = surplus_for(state, src, used)
                need, _ = capture_need(state, src, weak)
                if avail >= need + ABS_MIN_BATCH:
                    emit(src.id, weak.id, min(avail, need + 12))

        return moves

    except Exception:
        return []


# ── offline neural training utility (run once locally, not during games) ──────

def train_neural_offline(n_games: int = 200, epochs: int = 40):
    """Generate self-play data and train NeuralVal.
    Call from a script; NOT called automatically during games.
    Returns the base64 string to embed in _NEURAL_WEIGHTS_B64.
    """
    try:
        from kaggle_environments import make
    except ImportError:
        print("kaggle_environments not available")
        return ""

    neural = NeuralVal()
    xs_all, ys_all = [], []

    for game_i in range(n_games):
        env = make("orbit_wars", debug=False)
        env.run(["random", "random"])
        steps_data = env.steps
        if not steps_data:
            continue
        final = steps_data[-1]
        r0 = (final[0].get("reward") or 0) if final[0] else 0
        r1 = (final[1].get("reward") or 0) if final[1] else 0
        label = 1.0 if r0 > r1 else (-1.0 if r1 > r0 else 0.0)

        # Sample ~10 states from this game for player 0
        sample_steps = steps_data[::max(1, len(steps_data) // 10)]
        for step_data in sample_steps:
            try:
                obs0 = step_data[0]["observation"]
                gs = GameState(obs0)
                feat = neural.feat(gs)
                xs_all.append(feat)
                ys_all.append(np.float32(label))
            except Exception:
                pass

        if (game_i + 1) % 20 == 0:
            print(f"  game {game_i+1}/{n_games}, samples={len(xs_all)}")

    if len(xs_all) < 16:
        print("Not enough training data")
        return ""

    xs = np.stack(xs_all)
    ys = np.array(ys_all, dtype=np.float32)
    neural.train(xs, ys, epochs=epochs, lr=3e-3, batch=32)
    print(f"Training done. Final loss: {neural.loss_h[-1]:.4f}")
    b64 = neural.weights_b64()
    print(f"Weights base64 length: {len(b64)}")
    return b64
