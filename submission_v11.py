"""Orbit Wars v11 — Reorganised framework.

Pillars:
- Snapshot:    one-shot per-turn cache of surplus / reserve / threat / centroid.
- PhasePolicy: single table holding all early/mid/late tunables.
- Planners:    Defense / Intercept / Expand / Attack each emit Plan candidates
               WITHOUT mutating shared state.
- PlanArbiter: collects, scores (+ MCTS bonus, * Neural modifier), commits.
- MCTSEngine:  plan-level tree search over the arbiter's top-K plans.
- NeuralVal:   multiplicative score modifier (NEVER overrides a plan).

All sub-systems live in a single file (Kaggle requirement) but are organised
into clearly delimited regions so future passes can replace one without
touching the rest.
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


# ╔═══ region 0: constants & helpers ════════════════════════════════════════╗

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
ABS_MIN_BATCH = 5

# Bocsimacko (2010 Planet Wars champion) caps per-planet scoring at the
# horizon — distant production is heavily discounted instead of accumulated
# linearly to game-end. Combined with the small enemy-ship positional pen,
# this biases the bot toward proximate, certain gains.
HORIZON_TURNS = 60
ENEMY_SHIP_PEN_COEFF = 0.0008  # tiny — only breaks ties


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


# ╔═══ region 1: data classes ═══════════════════════════════════════════════╗

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


# ╔═══ region 2: GameState ══════════════════════════════════════════════════╗

class GameState:
    """Parses obs once. Holds planets/fleets and incoming-fleet metadata."""

    def __init__(self, obs, config=None):
        self.my_id = int(_get(obs, "player", 0) or 0)
        self.ang_vel = float(_get(obs, "angular_velocity", 0.0) or 0.0)
        self.step = int(_get(obs, "step", 0) or 0)

        cfg = _get(obs, "configuration", None) or config or {}
        self.max_speed = float(_get(cfg, "shipSpeed", DEFAULT_MAX_SHIP_SPEED)
                               or DEFAULT_MAX_SHIP_SPEED)
        self.episode_steps = int(_get(cfg, "episodeSteps", DEFAULT_EPISODE_STEPS)
                                 or DEFAULT_EPISODE_STEPS)

        comet_ids = set(int(x) for x in (_get(obs, "comet_planet_ids", []) or []))
        self.comet_paths: Dict[int, Tuple[List[Tuple[float, float]], int]] = {}
        self._parse_comets(_get(obs, "comets", []) or [])

        initial_xy: Dict[int, Tuple[float, float]] = {}
        for row in _get(obs, "initial_planets", []) or []:
            initial_xy[int(row[0])] = (float(row[2]), float(row[3]))

        self.planets: List[Planet] = []
        for row in _get(obs, "planets", []) or []:
            pid = int(row[0])
            ix, iy = initial_xy.get(pid, (float(row[2]), float(row[3])))
            self.planets.append(Planet(pid, int(row[1]), float(row[2]), float(row[3]),
                                       float(row[4]), int(row[5]), int(row[6]),
                                       ix, iy, pid in comet_ids))

        self.fleets: List[Fleet] = []
        for row in _get(obs, "fleets", []) or []:
            self.fleets.append(Fleet(int(row[0]), int(row[1]), float(row[2]),
                                     float(row[3]), float(row[4]),
                                     int(row[5]), int(row[6])))

        self._pm = {p.id: p for p in self.planets}
        self.my_pl = [p for p in self.planets if p.owner == self.my_id]
        self.en_pl = [p for p in self.planets if p.owner not in (-1, self.my_id)]
        self.neu_pl = [p for p in self.planets if p.owner == -1]
        self.en_ids = sorted({p.owner for p in self.en_pl})

        self.fleet_target: Dict[int, Optional[Tuple[int, int]]] = {}
        self.arrivals: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        self.incoming: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for f in self.fleets:
            pred = self._predict_fleet_target(f)
            self.fleet_target[f.id] = pred
            if pred is not None:
                tid, eta = pred
                self.arrivals[tid].append((eta, f.owner, f.ships))
                self.incoming[tid][f.owner] += f.ships

    # ── parsing helpers ──
    def _parse_comets(self, groups) -> None:
        for group in groups:
            ids = _get(group, "planet_ids", []) or []
            paths = _get(group, "paths", []) or []
            idx = int(_get(group, "path_index", 0) or 0)
            for i, pid_raw in enumerate(ids):
                if i >= len(paths):
                    continue
                path: List[Tuple[float, float]] = []
                for pt in paths[i] or []:
                    if len(pt) >= 2:
                        path.append((float(pt[0]), float(pt[1])))
                if path:
                    self.comet_paths[int(pid_raw)] = (path, idx)

    # ── geometry / lookup ──
    def get(self, pid: int) -> Optional[Planet]:
        return self._pm.get(pid)

    def is_orbiting(self, p: Planet) -> bool:
        if p.is_comet:
            return False
        r = math.hypot(p.initial_x - SUN_X, p.initial_y - SUN_Y)
        return r + p.radius < ROTATION_RADIUS_LIMIT and abs(self.ang_vel) > 1e-12

    def planet_pos_at(self, p: Planet, t: int) -> Tuple[float, float]:
        if p.is_comet and p.id in self.comet_paths:
            path, idx = self.comet_paths[p.id]
            j = idx + max(0, int(t))
            return path[min(j, len(path) - 1)]
        if not self.is_orbiting(p):
            return p.x, p.y
        r = math.hypot(p.initial_x - SUN_X, p.initial_y - SUN_Y)
        a0 = math.atan2(p.initial_y - SUN_Y, p.initial_x - SUN_X)
        a1 = a0 + self.ang_vel * (self.step + int(t))
        return SUN_X + r * math.cos(a1), SUN_Y + r * math.sin(a1)

    def comet_turns_left(self, p: Planet) -> int:
        if not p.is_comet or p.id not in self.comet_paths:
            return 999
        path, idx = self.comet_paths[p.id]
        return max(0, len(path) - idx - 1)

    def _predict_fleet_target(self, f: Fleet, max_steps: int = 220) -> Optional[Tuple[int, int]]:
        spd = fleet_speed(f.ships, self.max_speed)
        cx, cy = f.x, f.y
        dx, dy = math.cos(f.angle) * spd, math.sin(f.angle) * spd
        for t in range(1, max_steps + 1):
            nx, ny = cx + dx, cy + dy
            if not (0.0 <= nx <= BOARD and 0.0 <= ny <= BOARD):
                return None
            if point_segment_distance(SUN_X, SUN_Y, cx, cy, nx, ny) < SUN_RADIUS:
                return None
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

    # ── aggregates ──
    def total_ships(self, owner: int) -> int:
        return (sum(p.ships for p in self.planets if p.owner == owner) +
                sum(f.ships for f in self.fleets if f.owner == owner))

    def centroid(self) -> Tuple[float, float]:
        if not self.my_pl:
            return SUN_X, SUN_Y
        return (sum(p.x for p in self.my_pl) / len(self.my_pl),
                sum(p.y for p in self.my_pl) / len(self.my_pl))

    def net_threat(self, p: Planet) -> int:
        inc = self.incoming.get(p.id, {})
        attackers = sum(v for k, v in inc.items() if k not in (-1, self.my_id))
        return attackers - inc.get(self.my_id, 0)

    def phase(self) -> str:
        progress = self.step / max(1, self.episode_steps)
        if progress < 0.18:
            return "early"
        if progress < 0.64:
            return "mid"
        return "late"

    def turns_left(self) -> int:
        return max(1, self.episode_steps - self.step)

    def enemy_incoming(self, pid: int) -> int:
        inc = self.incoming.get(pid, {})
        return sum(v for k, v in inc.items() if k not in (-1, self.my_id))

    def effective_garrison(self, p: Planet) -> int:
        if p.owner not in self.en_ids:
            return p.ships
        out = 0
        for f in self.fleets:
            if f.owner != p.owner:
                continue
            t = self.fleet_target.get(f.id)
            if t is not None and t[0] != p.id:
                out += f.ships
        return max(0, p.ships - out)


# ╔═══ region 3: Snapshot & geometry ════════════════════════════════════════╗

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
    Guarantees: endpoint stays inside [0.5, 99.5] and the segment does not clip
    the sun. Falls back to best-clearance angle if no clean lane is found."""
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
        clearance = (point_segment_distance(SUN_X, SUN_Y, src.x, src.y, ex, ey)
                     - (SUN_RADIUS + 1.25))
        if clearance > 0:
            return a, eta
        if clearance > best_clearance:
            best_clearance = clearance
            best_angle = a
    return best_angle, eta


def target_state_at(state: GameState, dst: Planet, eta: int) -> Tuple[int, int]:
    """Project (owner, ships) of `dst` after `eta` turns including in-flight arrivals."""
    owner = dst.owner
    ships = int(dst.ships)
    by_turn: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
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
    """Iterative estimate of (need, eta) to capture dst from src."""
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


@dataclass
class Snapshot:
    """Per-turn precomputed view. Planners read from here; they never recompute
    surplus / reserve themselves. Mutate by calling `subtract(pid, ships)`
    AFTER an action commits."""

    state: GameState
    policy: "PhasePolicy"

    surplus: Dict[int, int] = field(default_factory=dict)
    reserve: Dict[int, int] = field(default_factory=dict)
    centroid: Tuple[float, float] = (SUN_X, SUN_Y)
    nearest_enemy_dist: Dict[int, float] = field(default_factory=dict)
    used: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    @classmethod
    def build(cls, state: GameState, policy: "PhasePolicy") -> "Snapshot":
        snap = cls(state=state, policy=policy)
        snap.centroid = state.centroid()
        for p in state.my_pl:
            snap.nearest_enemy_dist[p.id] = (
                min((p.dist(e) for e in state.en_pl), default=999.0))
            snap.reserve[p.id] = snap._reserve(p)
            snap.surplus[p.id] = max(0, p.ships - snap.reserve[p.id])
        return snap

    def _reserve(self, p: Planet) -> int:
        s = self.state
        threat = max(0, s.net_threat(p))
        if p.is_comet:
            ttl = s.comet_turns_left(p)
            return max(threat + 2, 2 if ttl > 10 else p.ships)
        ned = self.nearest_enemy_dist.get(p.id, 999.0)
        front_lock = ((10 + p.production * 4) if ned < 20
                      else (5 + p.production * 3) if ned < 36 else 0)
        growth_lock = p.production * self.policy.reserve_growth_mul
        return max(threat + 6, growth_lock, front_lock, 3)

    def avail(self, pid: int) -> int:
        """Surplus minus already-used in this turn."""
        return max(0, self.surplus.get(pid, 0) - self.used.get(pid, 0))

    def subtract(self, pid: int, ships: int) -> None:
        self.used[pid] += int(ships)

    def is_safe_investment(self, dst: Planet, eta: int) -> bool:
        """Bocsimacko `safe-to-invest-p` port (player.lisp:1079).

        Returns True unless this looks like a clearly losing trade:
          - Enemy is much closer AND has overwhelming nearby firepower.
          - Or our friendly planets are under net inbound threat that
            already exceeds our total surplus (i.e. we cannot afford to
            send anything outward without losing a homeworld).

        Conservative — only filters obvious losers, not borderline cases.
        Original Lisp version uses time-vector arrivals; we approximate.
        """
        s = self.state
        # Defensive triage: aggregate friendly inbound threat and surplus.
        net_threat = sum(max(0, s.net_threat(p)) for p in s.my_pl)
        my_total_surplus = sum(self.surplus.values())
        if net_threat > my_total_surplus * 1.10 and my_total_surplus > 0:
            # We are already underwater on defense — no time for expansion.
            return False
        # Enemy proximity / power dominance check.
        my_reach = min((m.dist(dst) for m in s.my_pl), default=999.0)
        en_reach = min((e.dist(dst) for e in s.en_pl), default=999.0)
        if en_reach < my_reach * 0.70:
            en_local = sum(e.ships for e in s.en_pl
                           if e.dist(dst) < my_reach * 1.20)
            if en_local > (dst.ships + dst.production * eta) * 1.6:
                return False
        return True


# ╔═══ region 4: PhasePolicy ════════════════════════════════════════════════╗

# Single source of truth for phase-dependent tuning. Adjusting strategy = edit
# one row.
PHASE_TABLE: Dict[str, Dict[str, object]] = {
    "early": dict(
        reserve_growth_mul=4,
        cost_pen_mul=0.82,
        cost_pen_neutral_mul=0.78,
        urgent_attack_ratio=999.0,           # disable urgent attack early
        urgent_attack_min_prod=99,
        mode_order=["expand", "balanced", "comet"],
        mcts_budget_ms=0,                    # no MCTS early
        mcts_max_iters=0,
        neural_modifier_strength=0.10,
        recapture_mul=1.0,
        approach_weight=1.55,
        sim_steps=8,
        tempo_floor=1,                       # neutral — eval as-is
    ),
    "mid": dict(
        reserve_growth_mul=4,
        cost_pen_mul=0.82,
        cost_pen_neutral_mul=0.74,
        urgent_attack_ratio=1.05,            # only press when ahead
        urgent_attack_min_prod=4,
        mode_order=["expand", "balanced", "counter", "aggro", "comet"],
        mcts_budget_ms=120,
        mcts_max_iters=50,
        neural_modifier_strength=0.12,
        recapture_mul=1.05,
        approach_weight=1.55,
        sim_steps=8,
        tempo_floor=2,                       # Bocsimacko's "min-turn-to-depart=2"
    ),
    "late": dict(
        reserve_growth_mul=5,
        cost_pen_mul=0.78,
        cost_pen_neutral_mul=0.70,
        urgent_attack_ratio=0.90,            # press even at slight deficit
        urgent_attack_min_prod=3,
        mode_order=["aggro", "counter", "expand", "balanced", "comet"],
        mcts_budget_ms=200,
        mcts_max_iters=80,
        neural_modifier_strength=0.15,
        recapture_mul=1.18,
        approach_weight=1.40,
        sim_steps=10,
        tempo_floor=1,                       # late — value every wave
    ),
}


@dataclass
class PhasePolicy:
    """Resolved view of PHASE_TABLE for the current step."""
    phase: str
    reserve_growth_mul: int
    cost_pen_mul: float
    cost_pen_neutral_mul: float
    urgent_attack_ratio: float
    urgent_attack_min_prod: int
    mode_order: List[str]
    mcts_budget_ms: float
    mcts_max_iters: int
    neural_modifier_strength: float
    recapture_mul: float
    approach_weight: float
    sim_steps: int
    tempo_floor: int

    @classmethod
    def for_state(cls, state: GameState) -> "PhasePolicy":
        ph = state.phase()
        row = PHASE_TABLE[ph]
        return cls(
            phase=ph,
            reserve_growth_mul=int(row["reserve_growth_mul"]),
            cost_pen_mul=float(row["cost_pen_mul"]),
            cost_pen_neutral_mul=float(row["cost_pen_neutral_mul"]),
            urgent_attack_ratio=float(row["urgent_attack_ratio"]),
            urgent_attack_min_prod=int(row["urgent_attack_min_prod"]),
            mode_order=list(row["mode_order"]),  # type: ignore[arg-type]
            mcts_budget_ms=float(row["mcts_budget_ms"]),
            mcts_max_iters=int(row["mcts_max_iters"]),
            neural_modifier_strength=float(row["neural_modifier_strength"]),
            recapture_mul=float(row["recapture_mul"]),
            approach_weight=float(row["approach_weight"]),
            sim_steps=int(row["sim_steps"]),
            tempo_floor=int(row["tempo_floor"]),
        )


# ╔═══ region 5: scoring ════════════════════════════════════════════════════╗

def approach_bonus(snap: Snapshot, dst: Planet, eta: int) -> float:
    if not snap.state.is_orbiting(dst):
        return 0.0
    cx, cy = snap.centroid
    now_d = math.hypot(dst.x - cx, dst.y - cy)
    fx, fy = snap.state.planet_pos_at(dst, eta)
    fut_d = math.hypot(fx - cx, fy - cy)
    gain = now_d - fut_d
    return max(-28.0, min(34.0, gain * snap.policy.approach_weight + dst.production * 1.2))


def enemy_eta_power(state: GameState, dst: Planet) -> Tuple[int, int]:
    best_eta, best_power = 999, 0
    for e in state.en_pl:
        probe = max(1, min(e.ships, max(5, e.ships * 2 // 3)))
        _, _, eta = lead_intercept(state, e, dst, probe)
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


def target_score(snap: Snapshot, src: Planet, dst: Planet) -> Tuple[float, int, int]:
    """Score (score, need, eta) of attacking dst from src under snap.policy.
    Returns -1e18 if action is impossible / wasteful."""
    state = snap.state
    if dst.owner == state.my_id or src.id == dst.id:
        return -1e18, 0, 0

    need, eta = capture_need(state, src, dst)
    if need <= 0:
        return -1e18, 0, eta

    # Bocsimacko-style horizon cap: distant production is discounted, not
    # rewarded linearly to game end.
    raw_turns = max(1, state.turns_left() - eta)
    turns = min(raw_turns, HORIZON_TURNS)
    if dst.is_comet:
        turns = min(turns, max(0, state.comet_turns_left(dst) - eta), 60)
        if turns <= 8:
            return -1e18, need, eta

    is_neu = dst.owner == -1
    is_en = dst.owner not in (-1, state.my_id)

    # Bocsimacko snipe-aware sizing: for a neutral that the enemy can grab
    # first, evaluate the post-takeover need (target_state_at at e_eta+1) and
    # use the larger of the two — commit fully to snipe, or skip cleanly.
    if is_neu and dst.production > 0 and dst.ships > dst.production:
        e_eta_first, _ = enemy_eta_power(state, dst)
        if 0 < e_eta_first < eta:
            owner_after, ships_after = target_state_at(state, dst, e_eta_first + 1)
            if owner_after not in (-1, state.my_id):
                # Re-cost as if attacking captured-by-enemy planet at the same eta.
                snipe_eta = max(eta, e_eta_first + 1)
                snipe_need = ships_after + 8 + min(6, dst.production)
                snipe_need += dst.production * snipe_eta // 5
                if snipe_need > need:
                    need = max(need, snipe_need)
                    eta = snipe_eta

    prod_value = dst.production * turns
    enemy_bonus = 38.0 if is_en else 0.0
    comet_bonus = 16.0 if dst.is_comet else 0.0
    rec_bonus = recapture_bonus(snap, dst)
    appr = approach_bonus(snap, dst, eta)

    # Path safety penalty.
    lane_ang, _ = safe_aim(state, src, dst, need)
    spd = fleet_speed(need, state.max_speed)
    bx = src.x + math.cos(lane_ang) * spd * eta
    by = src.y + math.sin(lane_ang) * spd * eta
    sun_pen = 90.0 if segment_hits_sun(src.x, src.y, bx, by, margin=2.0) else 0.0

    # Sniping risk for neutrals.
    snipe_pen = 0.0
    if is_neu:
        e_eta, e_pow = enemy_eta_power(state, dst)
        if e_eta <= eta + 1 and e_pow > max(0, need - 4):
            snipe_pen = 46.0 + 0.12 * e_pow
        elif e_eta <= eta + 2 and e_pow > need + 5:
            snipe_pen = 24.0 + 0.08 * e_pow

    eta_pen = 0.34 * eta
    cost_mul = snap.policy.cost_pen_neutral_mul if is_neu else snap.policy.cost_pen_mul
    cost_pen = cost_mul * need
    cont_pen = contest_penalty(state, dst)

    # Bocsimacko enemy-ship positional micro-penalty: prefer plans that
    # operate near (and threaten) enemy stockpiles. Linearly decayed by ETA
    # so distant pressure counts less, matching `(- *n-turns-till-horizon* i)`.
    en_ship_total = sum(e.ships for e in state.en_pl)
    horizon_decay = max(0, HORIZON_TURNS - eta) / HORIZON_TURNS
    enemy_ship_pen = ENEMY_SHIP_PEN_COEFF * en_ship_total * horizon_decay

    score = prod_value + enemy_bonus + comet_bonus + rec_bonus + appr
    score -= cost_pen + eta_pen + sun_pen + snipe_pen + cont_pen + enemy_ship_pen
    score /= max(1.0, eta ** 0.30)
    return score, need, eta


def elite_eval(state: GameState) -> float:
    """Static positional eval — reserved for NeuralVal feature engineering."""
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
    return ((ms - es) + 48.0 * (mp - ep) + 20.0 * (mc - ec)
            - 2.8 * threat + 9.0 * border + 0.45 * (mf - ef) - 12.0 * ndeny)


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


def score_plan_actions(state: GameState, actions: List[Tuple[int, int, int]],
                       steps: int = 8, tempo_floor: int = 1) -> float:
    """Score by simulating `steps` turns after applying actions to a clone.

    `tempo_floor` (Bocsimacko `min-turn-to-depart`): run this many *idle*
    sim steps BEFORE the regular eval window, simulating "no follow-up
    fleets are dispatched". Plans that depend on chained reinforcements
    that we haven't planned for in `actions` get penalised because their
    targets don't materialise within the eval window without those
    follow-ups. tempo_floor=1 means evaluate as-is.
    """
    planets, fleets = clone_sim(state)
    used: Dict[int, int] = defaultdict(int)
    for sid, did, ships in actions:
        sp = state.get(sid); dp = state.get(did); sim_src = planets.get(sid)
        if sp is None or dp is None or sim_src is None or sim_src.owner != state.my_id:
            continue
        # Re-derive surplus from sim state (Snapshot may have evolved already).
        send = min(int(ships), max(0, sim_src.ships - ABS_MIN_BATCH - used[sid]))
        if send <= 0:
            continue
        _, eta = safe_aim(state, sp, dp, send)
        sim_src.ships -= send
        used[sid] += send
        fleets.append(SimF(state.my_id, did, send, eta))
    # Bocsimacko tempo-floor: extra idle sim steps before scoring window.
    for _ in range(max(0, tempo_floor - 1)):
        sim_step(planets, fleets)
    for _ in range(steps):
        sim_step(planets, fleets)
    return eval_sim_planets(state, planets, fleets)


# ── DefensePlanner ───────────────────────────────────────────────────────────

class DefensePlanner:
    @staticmethod
    def plan(snap: Snapshot) -> Plan:
        state = snap.state
        actions: List[Tuple[int, int, int]] = []
        score = 0.0
        local_used: Dict[int, int] = defaultdict(int)

        for tgt in sorted(state.my_pl, key=lambda p: -state.net_threat(p)):
            threat = state.net_threat(tgt)
            if threat <= 0:
                continue
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
                _, _, eta = lead_intercept(state, src, dst, min(avail, need))
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


def _build_capture_plan(snap: Snapshot, mode: str,
                        diplo: Optional["DiplomacyEngine"] = None,
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
    for dst in targets:
        best_sc = -1e18
        for src in state.my_pl:
            sc, _, _ = target_score(snap, src, dst)
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
        if best_sc > -31.0:
            ranked.append((best_sc, dst))
    ranked.sort(key=lambda x: -x[0])

    for _, dst in ranked[:MAX_TARGETS_PER_PLAN]:
        if len(actions) >= MAX_TOTAL_MOVES or dst.id in target_done:
            continue

        contributors: List[Tuple[int, Planet, int, int, float]] = []
        for src in state.my_pl:
            avail = max(0, snap.avail(src.id) - local_used[src.id])
            if avail < max(ABS_MIN_BATCH, int(src.production) * 2):
                continue
            need, eta = capture_need(state, src, dst)
            sc, _, _ = target_score(snap, src, dst)
            contributors.append((eta, src, avail, need, sc))
        if not contributors:
            continue

        contributors.sort(key=lambda x: (x[0], -x[4], -x[2]))
        eta0 = contributors[0][0]
        group = [c for c in contributors if c[0] <= eta0 + SYNC_ETA_WINDOW][:MAX_SOURCES_PER_TARGET]
        if not group:
            continue

        group_eta = max(c[0] for c in group)
        # Bocsimacko `safe-to-invest-p` gate: skip neutral expansion if
        # defense-vs-counterattack feasibility check fails. Enemy targets
        # bypass — attacking is always potentially worth it.
        if dst.owner == -1 and not snap.is_safe_investment(dst, group_eta):
            continue
        owner, garrison = target_state_at(state, dst, group_eta)
        if owner == state.my_id:
            continue
        required = garrison + (3 if owner == -1 else 8 + min(6, dst.production))
        if owner not in (-1, state.my_id):
            required += dst.production * group_eta // 5

        if dst.is_comet and state.comet_turns_left(dst) <= group_eta + 5:
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
        if sent < required:
            continue

        for sid, did, send in staged:
            actions.append((sid, did, send))
            local_used[sid] += send
        target_done.add(dst.id)
        score += sum(c[4] for c in group[:len(staged)]) + required * (
            1.5 if dst.owner != -1 else 0.9)

    return Plan(actions, score, mode)


class ExpandPlanner:
    """Wraps capture builder for neutral / weak-target modes."""

    @staticmethod
    def plan(snap: Snapshot, mode: str = "expand",
             diplo: Optional["DiplomacyEngine"] = None) -> Plan:
        return _build_capture_plan(snap, mode, diplo=diplo)


class AttackPlanner:
    """Wraps capture builder for aggressive enemy-target modes."""

    @staticmethod
    def plan(snap: Snapshot, mode: str = "aggro",
             diplo: Optional["DiplomacyEngine"] = None,
             diplo_target: Optional[Planet] = None) -> Plan:
        return _build_capture_plan(snap, mode, diplo=diplo, diplo_target=diplo_target)


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
        if len(state.my_pl) < 2 or not state.en_pl:
            return Plan([], 0.0, "redistribute")

        def ned(p):
            return min((p.dist(e) for e in state.en_pl), default=999.0)

        ordered = sorted(state.my_pl, key=ned)
        fc = max(1, min(len(ordered) - 1, len(ordered) // 2 + 1))
        fronts = ordered[:fc]
        rears = [p for p in ordered[fc:] if state.net_threat(p) <= 0]
        actions: List[Tuple[int, int, int]] = []
        local_used: Dict[int, int] = defaultdict(int)
        for rear in rears[:5]:
            avail = max(0, snap.avail(rear.id) - local_used[rear.id])
            if avail < max(10, rear.production * 3):
                continue
            dst = min(fronts, key=lambda f: rear.dist(f))
            send = max(ABS_MIN_BATCH, min(avail, max(8, int(avail * 0.55))))
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
        en_total = sum(state.total_ships(e) for e in state.en_ids)
        if my_total < en_total * policy.urgent_attack_ratio:
            return Plan([], 0.0, "urgent_hp", urgent=True)

        cx, cy = snap.centroid
        targets = sorted(
            [p for p in state.en_pl
             if p.production >= policy.urgent_attack_min_prod
             and math.hypot(p.x - cx, p.y - cy) < 40],
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
            owner, garrison = target_state_at(state, dst, group_eta)
            if owner == state.my_id:
                continue
            required = (garrison + 8 + min(6, dst.production)
                        + dst.production * group_eta // 5)
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


# ╔═══ region 8: MCTSEngine — plan-level tree search ════════════════════════╗

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
    is chosen) — this keeps cost predictable and the bonus interpretable."""

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


# ╔═══ region 9: NeuralVal — multiplicative score modifier ══════════════════╗

# Inline pre-trained weights from v10. Same 14→64→32→1 architecture; lifted
# verbatim because retraining is out-of-scope for v11's framework rebuild.
_NEURAL_WEIGHTS_B64 = ""  # left empty: NeuralVal will use random init (still safe as a small modifier).


class NeuralVal:
    """Score modifier in [-strength, +strength]. NEVER overrides plans; just
    nudges the arbiter's ranking by `(1 + strength * predict)`."""

    N_FEAT = 14

    def __init__(self):
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.2, (64, self.N_FEAT)).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = rng.normal(0, 0.2, (32, 64)).astype(np.float32)
        self.b2 = np.zeros(32, dtype=np.float32)
        self.W3 = rng.normal(0, 0.2, (1, 32)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)
        self._try_load_inline()

    def _try_load_inline(self):
        if not _NEURAL_WEIGHTS_B64:
            return
        try:
            raw = base64.b64decode(_NEURAL_WEIGHTS_B64)
            d = np.load(io.BytesIO(raw), allow_pickle=True).item()
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
            float(len(state.en_ids) > 1),
        ], dtype=np.float32)

    def predict(self, state: GameState) -> float:
        try:
            x = self.feat(state)
            h1 = np.maximum(0.0, self.W1 @ x + self.b1)
            h2 = np.maximum(0.0, self.W2 @ h1 + self.b2)
            out = float(np.tanh(self.W3 @ h2 + self.b3)[0])
            return out
        except Exception:
            return 0.0

    def score_modifier(self, state: GameState, strength: float) -> float:
        """Returns multiplicative factor in [1-strength, 1+strength]."""
        return 1.0 + strength * self.predict(state)


_GLOBAL_NEURAL = NeuralVal()


# ╔═══ region 10: PlanArbiter ═══════════════════════════════════════════════╗

class PlanArbiter:
    """Single source of decisions. Collects candidate plans, scores them with
    sim + MCTS bonus + Neural modifier, then commits to `moves` honouring the
    Snapshot's per-source surplus."""

    def __init__(self, snap: Snapshot, diplo: DiplomacyEngine,
                 neural: NeuralVal,
                 elapsed_ms_fn,
                 deadline_ms: float = 920.0):
        self.snap = snap
        self.policy = snap.policy
        self.diplo = diplo
        self.neural = neural
        self.elapsed_ms = elapsed_ms_fn
        self.deadline_ms = deadline_ms
        self.moves: List[List] = []

    # ── 1. urgent: defense + intercept (commit immediately) ──────────────────

    def commit_urgent(self) -> None:
        # Defense + intercept run with `urgent=True` (allowed to dip into reserve).
        for planner in (DefensePlanner, InterceptPlanner):
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

        # Comet always considered if any non-allied comet exists.
        if any(p.is_comet and p.owner != self.snap.state.my_id
               for p in self.snap.state.planets):
            plans.append(ExpandPlanner.plan(self.snap, "comet", diplo=self.diplo))

        for mode in self.policy.mode_order:
            if mode == "comet":
                continue  # already handled
            if self._out_of_time(self.deadline_ms - 200):
                break
            if mode in ("expand", "balanced"):
                plans.append(ExpandPlanner.plan(self.snap, mode, diplo=self.diplo))
            elif mode == "diplo" and diplo_tgt is not None:
                plans.append(AttackPlanner.plan(self.snap, "diplo",
                                                diplo=self.diplo,
                                                diplo_target=diplo_tgt))
            elif mode in ("aggro", "counter"):
                plans.append(AttackPlanner.plan(self.snap, mode, diplo=self.diplo))

        # Optional diplo even if not in mode_order (always available).
        if diplo_tgt is not None and not any(p.tag == "diplo" for p in plans):
            plans.append(AttackPlanner.plan(self.snap, "diplo",
                                            diplo=self.diplo,
                                            diplo_target=diplo_tgt))

        # Drop empty plans.
        return [p for p in plans if p.actions]

    def score_with_modifiers(self, plans: List[Plan]) -> List[Tuple[float, Plan]]:
        if not plans:
            return []

        # 1. Base sim score.
        sim_steps = self.policy.sim_steps
        base = []
        for plan in plans:
            sim_val = score_plan_actions(self.snap.state, plan.actions,
                                          steps=sim_steps,
                                          tempo_floor=self.policy.tempo_floor)
            base.append((plan.score + sim_val, plan))

        base.sort(key=lambda x: -x[0])

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

        # 3. Neural modifier (state-only, applied multiplicatively).
        modifier = self.neural.score_modifier(self.snap.state,
                                              self.policy.neural_modifier_strength)

        scored: List[Tuple[float, Plan]] = []
        for i, (s, plan) in enumerate(base):
            bonus = mcts_bonus.get(i, 0.0)
            final = s * modifier + bonus
            scored.append((final, plan))
        scored.sort(key=lambda x: -x[0])
        return scored

    def commit_best(self, scored: List[Tuple[float, Plan]]) -> None:
        """Commit the single best strategic plan only (v10 behaviour).
        Lower-ranked plans are discarded; remaining surplus drains via the
        fallback (redistribution + late dump). Greedy multi-plan commits were
        observed to bleed ships into mediocre secondary targets and cost ~30%
        win-rate vs v10."""
        if not scored:
            return
        self._commit_plan(scored[0][1], urgent=False)

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
        # Effective availability = surplus - already-used. For urgent, allow
        # tapping the safety reserve down to ABS_MIN_BATCH.
        if urgent:
            cap = max(0, src.ships - ABS_MIN_BATCH - snap.used.get(sid, 0))
        else:
            cap = snap.avail(sid)
        send = min(int(ships), cap)
        if send < ABS_MIN_BATCH:
            return False
        angle, _ = safe_aim(state, src, dst, send)
        self.moves.append([sid, float(angle), int(send)])
        snap.subtract(sid, send)
        return True

    def _out_of_time(self, threshold_ms: float) -> bool:
        return self.elapsed_ms() > threshold_ms


# ╔═══ region 11: agent() entry ═════════════════════════════════════════════╗

def agent(obs, config=None):
    """Kaggle-required entry. Returns list of [src_id, angle, ships] moves."""
    global _GLOBAL_OPP, _GLOBAL_NEURAL
    t0 = time.time()
    elapsed = lambda: (time.time() - t0) * 1000.0

    try:
        state = GameState(obs, config)
        if not state.my_pl:
            return []

        _GLOBAL_OPP.update(state)
        policy = PhasePolicy.for_state(state)
        snap = Snapshot.build(state, policy)
        diplo = DiplomacyEngine(state, _GLOBAL_OPP)

        arbiter = PlanArbiter(snap, diplo, _GLOBAL_NEURAL,
                              elapsed_ms_fn=elapsed,
                              deadline_ms=920.0)

        # Pipeline.
        arbiter.commit_urgent()
        plans = arbiter.collect_strategic()
        scored = arbiter.score_with_modifiers(plans)
        arbiter.commit_best(scored)
        arbiter.commit_fallback()

        return arbiter.moves
    except Exception:
        return []
