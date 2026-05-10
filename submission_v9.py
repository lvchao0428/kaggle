"""Orbit Wars v9b - official-physics batch allocator + full strategy toolkit.

Key improvements over v9:
- safe_aim: when ALL deltas still hit the sun, pick the deflection angle with
  maximum clearance instead of blindly returning the collision course.
- capture_need iteration includes target production during transit so a neutral
  planet is never under-attacked (the "one wave, no leftover" guarantee).
- Absolute minimum-batch gate is applied at emit() so no 1-3 ship trickle
  ever leaves a planet, even from defense/intercept paths.
- OpponentModel tracks enemy fleet destinations each turn; targets that an
  enemy fleet is heading toward get a contest_penalty so we avoid wasteful
  head-on clashes unless we can decisively win.
- High-production enemy planets recently captured from us get an elevated
  recapture bonus proportional to production * distance_penalty.
- DiplomacyEngine and EliteEval border/neutral-deny scoring from the notebook
  are ported, using correct physics (no notebook's wrong sun radius / speed).
- StrategyEngine counter-attack candidate: attack enemy planets whose garrison
  minus outbound fleets is low.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

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
MIN_PLAN_SCORE = -30.0
ABS_MIN_BATCH = 5   # never send fewer than this many ships in a single order

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

        # enemy outbound: ships each enemy is sending away from their own planets
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

    # Enemy fleet contest: how many enemy ships are currently heading to planet pid?
    def enemy_incoming(self, pid: int) -> int:
        inc = self.incoming.get(pid, {})
        return sum(v for k, v in inc.items() if k not in (-1, self.my_id))

    # Effective garrison of an enemy planet accounting for its outbound fleets
    def effective_garrison(self, p: Planet) -> int:
        if p.owner not in self.en_ids: return p.ships
        out = sum(f.ships for f in self.fleets if f.owner == p.owner
                  and self.fleet_target.get(f.id) is not None
                  and (t := self.fleet_target.get(f.id)) is not None
                  and t[0] != p.id)
        return max(0, p.ships - out)


# ── geometry & aiming ─────────────────────────────────────────────────────────

def lead_intercept(state: GameState, src: Planet, dst: Planet, ships: int,
                   iters: int = 6) -> Tuple[float, float, int]:
    spd = fleet_speed(max(1, ships), state.max_speed)
    tx, ty = dst.x, dst.y
    eta = max(1, int(math.ceil(math.hypot(tx - src.x, ty - src.y) / spd)))
    for _ in range(iters):
        tx, ty = state.planet_pos_at(dst, eta)
        eta = max(1, int(math.ceil(math.hypot(tx - src.x, ty - src.y) / spd)))
    return tx, ty, eta


def safe_aim(state: GameState, src: Planet, dst: Planet, ships: int) -> Tuple[float, int]:
    """Return (angle, eta). Angle is guaranteed to not pass through the sun
    if any deflection avoids it; otherwise the least-bad deflection is returned."""
    tx, ty, eta = lead_intercept(state, src, dst, max(1, ships))
    angle = math.atan2(ty - src.y, tx - src.x)
    spd = fleet_speed(max(1, ships), state.max_speed)

    def endpoint(a):
        return (src.x + math.cos(a) * spd * eta,
                src.y + math.sin(a) * spd * eta)

    bx, by = endpoint(angle)
    if not segment_hits_sun(src.x, src.y, bx, by):
        return angle, eta

    # Try deflections; keep track of the best clearance in case none fully avoids sun
    best_angle = angle
    best_clearance = -999.0
    for delta in (0.12, -0.12, 0.22, -0.22, 0.34, -0.34, 0.50, -0.50, 0.68, -0.68, 0.88, -0.88, 1.1, -1.1):
        a = angle + delta
        bx2, by2 = endpoint(a)
        clearance = point_segment_distance(SUN_X, SUN_Y, src.x, src.y, bx2, by2) - (SUN_RADIUS + 1.25)
        if clearance > 0:
            return a, eta      # first deflection that clears the sun
        if clearance > best_clearance:
            best_clearance = clearance
            best_angle = a
    return best_angle, eta


# ── state-at-eta forward projector ───────────────────────────────────────────

def target_state_at(state: GameState, dst: Planet, eta: int) -> Tuple[int, int]:
    """Simulate known arrivals at dst up to turn eta and return (owner, ships)."""
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
    """How many ships to send from src to guarantee taking dst (3-iteration).
    Includes target production during transit so a single wave always succeeds."""
    # Start with a conservative first estimate
    need = max(ABS_MIN_BATCH, dst.ships + (2 if dst.owner == -1 else 8))
    eta = 1
    for _ in range(4):
        _, _, eta = lead_intercept(state, src, dst, need)
        owner, ships = target_state_at(state, dst, eta)
        if owner == state.my_id:
            # Planet will be ours by then; just need to hold it
            need = max(ABS_MIN_BATCH, state.net_threat(dst) + 4)
        else:
            base_margin = margin if margin is not None else (
                3 if owner == -1 else 8 + min(6, dst.production))
            # Add production earned by target during our transit
            extra_prod = (dst.production * eta) if owner not in (-1, state.my_id) else 0
            need = ships + base_margin + extra_prod // 4   # 1/4 as safety factor
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
    growth_lock = p.production * (5 if state.phase() == "early" else 4)
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
    gain = now_d - fut_d  # positive = planet moving toward our cluster
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
    """Extra score when an enemy has recently taken a high-production planet."""
    if dst.owner not in state.en_ids: return 0.0
    # Bonus scales with production; attenuated by distance from our centroid
    cx, cy = state.centroid()
    d = max(1.0, math.hypot(dst.x - cx, dst.y - cy))
    return dst.production * 14.0 / (1.0 + d * 0.04)


def contest_penalty(state: GameState, dst: Planet) -> float:
    """Penalty for sending to a target that an enemy fleet is also heading to
    (we might fight their fleet mid-capture and waste ships)."""
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

    prod_value = dst.production * turns
    enemy_bonus = 38.0 if dst.owner not in (-1, state.my_id) else 0.0
    comet_bonus = 16.0 if dst.is_comet else 0.0
    rec_bonus = recapture_bonus(state, dst)
    appr = approach_bonus(state, dst, eta)

    lane_ang, _ = safe_aim(state, src, dst, need)
    spd = fleet_speed(need, state.max_speed)
    bx = src.x + math.cos(lane_ang) * spd * eta
    by = src.y + math.sin(lane_ang) * spd * eta
    sun_pen = 90.0 if segment_hits_sun(src.x, src.y, bx, by, margin=2.0) else 0.0

    snipe_pen = 0.0
    if dst.owner == -1:
        e_eta, e_pow = enemy_eta_power(state, dst)
        if e_eta <= eta + 1 and e_pow > max(0, need - 4):
            snipe_pen = 46.0 + 0.12 * e_pow
        elif e_eta <= eta + 2 and e_pow > need + 5:
            snipe_pen = 24.0 + 0.08 * e_pow

    eta_pen = 0.34 * eta
    cost_pen = 0.82 * need
    cont_pen = contest_penalty(state, dst)

    score = prod_value + enemy_bonus + comet_bonus + rec_bonus + appr
    score -= cost_pen + eta_pen + sun_pen + snipe_pen + cont_pen
    score /= max(1.0, eta ** 0.30)
    return score, need, eta


# ── EliteEval: position score from notebook, corrected physics ────────────────

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
    cx, cy = state.centroid()
    border = sum((35 - m.dist(e)) / 35 * m.production
                 for m in state.my_pl for e in state.en_pl if m.dist(e) < 35)
    ndeny = sum(n.production for n in state.neu_pl
                if any(n.dist(e) < 25 for e in state.en_pl)
                and not any(n.dist(m) < 25 for m in state.my_pl))
    return ((ms - es) + 48.0*(mp-ep) + 20.0*(mc-ec)
            - 2.8*threat + 9.0*border + 0.45*(mf-ef) - 12.0*ndeny)


# ── OpponentModel (persistent across turns) ───────────────────────────────────

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

    def threat_score(self, eid: int, state: GameState) -> float:
        ep = [p for p in state.planets if p.owner == eid]
        prox = sum(1.0 / max(m.dist(e), 1) for m in state.my_pl for e in ep)
        sh = state.total_ships(eid)
        aggr = 1.0 + self.aggression(eid)
        return (prox * 38 + sh * 0.65) * aggr

    def primary_target(self, state: GameState) -> Optional[Planet]:
        if not state.en_ids: return None
        best_eid = max(state.en_ids, key=lambda e: self.threat_score(e, state))
        ep = [p for p in state.planets if p.owner == best_eid]
        if not ep: return None
        cx, cy = state.centroid()
        return min(ep, key=lambda p: p.ships + math.hypot(p.x-cx, p.y-cy)*0.38)


_GLOBAL_OPP = OpponentModel()


# ── forward sim ───────────────────────────────────────────────────────────────

class SimP:
    __slots__ = ("id","owner","ships","production")
    def __init__(self, p: Planet):
        self.id = p.id; self.owner = p.owner
        self.ships = p.ships; self.production = p.production


class SimF:
    __slots__ = ("owner","tid","ships","eta")
    def __init__(self, owner, tid, ships, eta):
        self.owner = owner; self.tid = tid; self.ships = ships; self.eta = eta


def clone_sim(state: GameState) -> Tuple[Dict[int,SimP], List[SimF]]:
    planets = {p.id: SimP(p) for p in state.planets}
    fleets: List[SimF] = []
    for f in state.fleets:
        t = state.fleet_target.get(f.id)
        if t: fleets.append(SimF(f.owner, t[0], f.ships, t[1]))
    return planets, fleets


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
    """Reinforce threatened planets to counter incoming enemy fleets."""
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
        return state.neu_pl + [p for p in state.en_pl if p.ships <= 20 or p.production <= 2]
    if mode == "aggro":
        # prioritise enemies with low effective garrison or high production
        return sorted(state.en_pl,
                      key=lambda p: (state.effective_garrison(p) - p.production * 5,
                                     p.dist_xy(SUN_X, SUN_Y)))
    if mode == "comet":
        return [p for p in state.planets if p.is_comet and p.owner != state.my_id]
    if mode == "counter":
        # Enemy planets whose effective garrison is small (they over-extended)
        return [p for p in state.en_pl if state.effective_garrison(p) < p.ships * 0.55]
    if mode == "diplo":
        return state.en_pl
    return state.neu_pl + state.en_pl


def build_capture_plan(state: GameState, mode: str,
                       base_used: Optional[Dict[int,int]] = None,
                       diplo_target: Optional[Planet] = None) -> Plan:
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
            if mode == "counter": sc += 30.0
            if mode == "diplo":   sc += 25.0 if dst is diplo_target else 0.0
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
        # include any production target earns during the longest ETA in the group
        if owner not in (-1, state.my_id):
            required += dst.production * group_eta // 5  # conservative extra buffer

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


def choose_best_plan(state: GameState, base_used: Dict[int,int], elapsed_ms,
                     opp: OpponentModel) -> Plan:
    phase = state.phase()
    diplo_tgt = opp.primary_target(state)

    modes: List[str] = []
    if any(p.is_comet and p.owner != state.my_id for p in state.planets):
        modes.append("comet")
    if phase != "early":
        modes.extend(["aggro", "counter"])
    modes.extend(["expand", "balanced"])
    if diplo_tgt:
        modes.append("diplo")

    candidates: List[Plan] = []
    for mode in modes:
        if elapsed_ms() > 800: break
        plan = build_capture_plan(state, mode, base_used,
                                   diplo_target=diplo_tgt if mode == "diplo" else None)
        if plan.actions:
            plan.score += score_plan(state, plan, steps=8)
            candidates.append(plan)

    if not candidates:
        return Plan([], 0.0, "none")
    return max(candidates, key=lambda p: p.score)


# ── main agent ────────────────────────────────────────────────────────────────

def agent(obs, config=None):
    global _GLOBAL_OPP
    t0 = time.time()
    elapsed = lambda: (time.time() - t0) * 1000.0

    try:
        state = GameState(obs, config)
        if not state.my_pl:
            return []

        _GLOBAL_OPP.update(state)

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

        # 1. Defense
        defense = build_defense_plan(state, used)
        for sid, did, sh in defense.actions:
            if elapsed() > 500: break
            emit(sid, did, sh, urgent=True)

        # 2. Intercept
        intercept = build_intercept_plan(state, used)
        for sid, did, sh in intercept.actions:
            if elapsed() > 600: break
            emit(sid, did, sh, urgent=True)

        # 3. Best multi-mode strategy plan (short sim ranked)
        best = choose_best_plan(state, used, elapsed, _GLOBAL_OPP)
        for sid, did, sh in best.actions:
            if elapsed() > 870: break
            emit(sid, did, sh)

        # 4. Redistribution: surplus rear ships forward
        redist = build_redistribution_plan(state, used)
        for sid, did, sh in redist.actions:
            if elapsed() > 920: break
            emit(sid, did, sh)

        # 5. Late-game dump: send remaining surplus to weakest enemy
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
