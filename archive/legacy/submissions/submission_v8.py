"""Orbit Wars v8 — 可扩展骨架（Planet Wars 冠军思路迁移 + 本题太阳/连续速度）

通读 planet-wars/src/model.lisp、player.lisp 后的可搬组件（概念级，非 Lisp 翻译）：
- future / arrivals：本题无「未来回合订单」，用 lead_intercept + 在途 incoming 代替离散航程表。
- cumulative-surplus：可安全外派的兵力 = 驻防 − 威胁储备 − 产兵锁（增长惯性），对应 PW 的 surplus 思想。
- step target / 多源：协同进攻仍用 ETA 窗；扩张改为「全局候选 + 约束选解」，避免外层按源贪心。
- 太阳：所有射击经 safe_aim 线段避日（本题相对 2010 的最大差异之一）。

选择策略（非纯贪心扩张）：
1) 生成扩张候选边 (src,dst) 全量评分（含留守机会成本、sniping、太阳罚项）。
2) 全局按分数排序；每条源本回合最多 1 条**扩张**（脉冲），兵力不超过盈余比例 PULSE_EXPAND。
3) 中立若本回合可用兵力达不到全额占领则**不派**（留给后续回合滚兵），减少无脑连发。
4) 防守不全脉冲；协同攻用 PULSE_ATTACK 限单源出力。

入口：agent(obs, config) -> List[[from_planet_id, direction_angle_rad, num_ships]]。
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

SUN_X = SUN_Y = 50.0
SUN_RADIUS = 10.0
BOARD = 100.0
ROTATION_RADIUS_LIMIT = 50.0
DEFAULT_MAX_SHIP_SPEED = 6.0
DEFAULT_EPISODE_STEPS = 500

# PW 迁移 + OW 特化
SYNC_ETA_WINDOW = 2
COORD_MAX_SOURCES_PER_TARGET = 8
PULSE_ATTACK = 0.58
PULSE_EXPAND = 0.44
HOLD_WEIGHT = 0.38
MAX_EDGES_EXPAND_SCAN = 96
MAX_TOTAL_MOVES = 28
REDIST_CAP = 5


def fleet_speed(ships: int, max_speed: float = DEFAULT_MAX_SHIP_SPEED) -> float:
    if ships <= 1:
        return 1.0
    spd = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(spd, max_speed)


def point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    abx, aby = bx - ax, by - ay
    l2 = abx * abx + aby * aby
    if l2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / l2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def segment_hits_sun(ax: float, ay: float, bx: float, by: float, margin: float = 1.5) -> bool:
    return point_segment_distance(SUN_X, SUN_Y, ax, ay, bx, by) < SUN_RADIUS + margin


class Planet:
    __slots__ = (
        "id", "owner", "x", "y", "radius", "ships", "production",
        "initial_x", "initial_y", "is_comet",
    )

    def __init__(
        self, id, owner, x, y, radius, ships, production,
        initial_x=0.0, initial_y=0.0, is_comet=False,
    ):
        self.id = id
        self.owner = owner
        self.x = x
        self.y = y
        self.radius = radius
        self.ships = ships
        self.production = production
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.is_comet = is_comet

    def dist(self, o):
        return math.hypot(self.x - o.x, self.y - o.y)


class Fleet:
    __slots__ = ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")

    def __init__(self, id, owner, x, y, angle, from_planet_id, ships):
        self.id = id
        self.owner = owner
        self.x = x
        self.y = y
        self.angle = angle
        self.from_planet_id = from_planet_id
        self.ships = ships


def _get(obs, key, default=None):
    if hasattr(obs, key):
        return getattr(obs, key)
    if isinstance(obs, dict):
        return obs.get(key, default)
    return default


class GameState:
    def __init__(self, obs, config=None):
        self.my_id: int = int(_get(obs, "player", 0) or 0)
        self.ang_vel: float = float(_get(obs, "angular_velocity", 0.0) or 0.0)
        self.step: int = int(_get(obs, "step", 0) or 0)

        comet_ids = set(_get(obs, "comet_planet_ids", []) or [])
        initial_rows = _get(obs, "initial_planets", []) or []
        initial_xy: Dict[int, Tuple[float, float]] = {}
        for row in initial_rows:
            initial_xy[int(row[0])] = (float(row[2]), float(row[3]))

        self.planets: List[Planet] = []
        for row in _get(obs, "planets", []) or []:
            pid = int(row[0])
            ix, iy = initial_xy.get(pid, (float(row[2]), float(row[3])))
            self.planets.append(
                Planet(
                    id=pid,
                    owner=int(row[1]),
                    x=float(row[2]),
                    y=float(row[3]),
                    radius=float(row[4]),
                    ships=int(row[5]),
                    production=int(row[6]),
                    initial_x=ix,
                    initial_y=iy,
                    is_comet=(pid in comet_ids),
                )
            )

        self.fleets: List[Fleet] = []
        for row in _get(obs, "fleets", []) or []:
            self.fleets.append(
                Fleet(
                    int(row[0]), int(row[1]), float(row[2]), float(row[3]),
                    float(row[4]), int(row[5]), int(row[6]),
                )
            )

        self._pm: Dict[int, Planet] = {p.id: p for p in self.planets}
        self.my_pl = [p for p in self.planets if p.owner == self.my_id]
        self.en_pl = [p for p in self.planets if p.owner not in (-1, self.my_id)]
        self.neu_pl = [p for p in self.planets if p.owner == -1]

        cfg = _get(obs, "configuration", None) or config or {}
        self.max_speed = float(_get(cfg, "shipSpeed", DEFAULT_MAX_SHIP_SPEED) or DEFAULT_MAX_SHIP_SPEED)
        self.episode_steps: int = int(_get(cfg, "episodeSteps", DEFAULT_EPISODE_STEPS) or DEFAULT_EPISODE_STEPS)

        self.fleet_target: Dict[int, Optional[Tuple[int, int]]] = {}
        self.incoming: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for f in self.fleets:
            tid_eta = self._predict_fleet_target(f)
            self.fleet_target[f.id] = tid_eta
            if tid_eta:
                tid, _ = tid_eta
                self.incoming[tid][f.owner] += f.ships

    def is_orbiting(self, p: Planet) -> bool:
        if p.is_comet:
            return False
        r = math.hypot(p.initial_x - SUN_X, p.initial_y - SUN_Y)
        return r + p.radius < ROTATION_RADIUS_LIMIT and abs(self.ang_vel) > 1e-12

    def planet_pos_at(self, p: Planet, t: int) -> Tuple[float, float]:
        if not self.is_orbiting(p):
            return p.x, p.y
        r = math.hypot(p.initial_x - SUN_X, p.initial_y - SUN_Y)
        a0 = math.atan2(p.initial_y - SUN_Y, p.initial_x - SUN_X)
        a1 = a0 + self.ang_vel * (self.step + t)
        return SUN_X + r * math.cos(a1), SUN_Y + r * math.sin(a1)

    def _predict_fleet_target(self, f: Fleet, max_steps: int = 200) -> Optional[Tuple[int, int]]:
        spd = fleet_speed(f.ships, self.max_speed)
        cx, cy = f.x, f.y
        dx, dy = math.cos(f.angle) * spd, math.sin(f.angle) * spd
        for t in range(1, max_steps + 1):
            nx, ny = cx + dx, cy + dy
            if not (0.0 <= nx <= BOARD and 0.0 <= ny <= BOARD):
                return None
            if point_segment_distance(SUN_X, SUN_Y, cx, cy, nx, ny) < SUN_RADIUS:
                return None
            best_pid: Optional[int] = None
            best_d = float("inf")
            for p in self.planets:
                px, py = self.planet_pos_at(p, t)
                d = point_segment_distance(px, py, cx, cy, nx, ny)
                if d < p.radius and d < best_d:
                    best_d = d
                    best_pid = p.id
            if best_pid is not None:
                return best_pid, t
            cx, cy = nx, ny
        return None

    def get(self, pid: int) -> Optional[Planet]:
        return self._pm.get(pid)

    def net_threat(self, p: Planet) -> int:
        inc = self.incoming.get(p.id, {})
        atk = sum(v for k, v in inc.items() if k not in (-1, self.my_id))
        return atk - inc.get(self.my_id, 0)

    def turns_left_game(self) -> int:
        return max(1, self.episode_steps - self.step)


def lead_intercept(
    state: GameState, src: Planet, dst: Planet, ships: int, iters: int = 6
) -> Tuple[float, float, int]:
    spd = fleet_speed(ships, state.max_speed)
    tx, ty = dst.x, dst.y
    eta = max(1, int(math.hypot(tx - src.x, ty - src.y) / spd))
    if state.is_orbiting(dst):
        for _ in range(iters):
            tx, ty = state.planet_pos_at(dst, eta)
            eta = max(1, int(math.hypot(tx - src.x, ty - src.y) / spd))
    return tx, ty, eta


def safe_aim(state: GameState, src: Planet, dst: Planet, ships: int) -> Tuple[float, int]:
    tx, ty, eta = lead_intercept(state, src, dst, ships)
    angle = math.atan2(ty - src.y, tx - src.x)
    spd = fleet_speed(ships, state.max_speed)
    bx = src.x + math.cos(angle) * spd * eta
    by = src.y + math.sin(angle) * spd * eta
    if not segment_hits_sun(src.x, src.y, bx, by):
        return angle, eta
    for delta in (0.10, -0.10, 0.20, -0.20, 0.32, -0.32, 0.46, -0.46, 0.62, -0.62):
        a = angle + delta
        bx = src.x + math.cos(a) * spd * eta
        by = src.y + math.sin(a) * spd * eta
        if not segment_hits_sun(src.x, src.y, bx, by):
            return a, eta
    return angle, eta


def planning_horizon(state: GameState) -> int:
    """PW dynamic horizon 的极简版：夹在短战术与半局长度之间。"""
    raw = state.episode_steps - state.step
    return max(12, min(48, raw))


def hold_value(state: GameState, p: Planet) -> float:
    """己方行星「留住兵力」的机会成本：产量 × 余下回合（粗略 FUTURE _terminal value）。"""
    tl = state.turns_left_game()
    if p.is_comet:
        tl = min(tl, 55)
    return float(p.production) * float(tl)


def cumulative_surplus(state: GameState, p: Planet) -> int:
    """PW cumulative-surplus 简化：可外派船 = 现驻 − 威胁 − 产兵锁。"""
    threat = max(0, state.net_threat(p))
    h = planning_horizon(state)
    growth_lock = p.production * min(8, max(4, h // 5))
    if p.is_comet:
        reserve = max(threat + 2, 4)
    else:
        reserve = max(threat + 5, growth_lock, p.production * 3)
    return max(0, p.ships - reserve)


def _nearest_enemy_eta_ships(state: GameState, dst: Planet) -> Tuple[int, int]:
    if not state.en_pl:
        return 999, 0
    best_e, best_s = 999, 0
    for e in state.en_pl:
        take = max(1, min(e.ships, e.ships * 2 // 3))
        _, _, te = lead_intercept(state, e, dst, take)
        if te < best_e:
            best_e, best_s = te, e.ships
    return best_e, best_s


def snipe_penalty(state: GameState, dst: Planet, my_eta: int, cost: int) -> float:
    if dst.owner != -1:
        return 0.0
    te, pw = _nearest_enemy_eta_ships(state, dst)
    if te <= my_eta + 1 and pw > max(0, cost - 3):
        return 48.0 + 0.17 * float(pw)
    if te <= my_eta + 2 and pw > cost + 5:
        return 24.0 + 0.09 * float(pw)
    return 0.0


def sun_lane_penalty(state: GameState, src: Planet, dst: Planet, ships_try: int) -> float:
    _, _, eta = lead_intercept(state, src, dst, ships_try)
    ang0 = math.atan2(dst.y - src.y, dst.x - src.x)
    spd = fleet_speed(ships_try, state.max_speed)
    bx = src.x + math.cos(ang0) * spd * eta
    by = src.y + math.sin(ang0) * spd * eta
    if segment_hits_sun(src.x, src.y, bx, by, margin=2.0):
        return 95.0
    return 0.0


def edge_capture_score(
    state: GameState, src: Planet, dst: Planet, my_id: int
) -> Tuple[float, int, int]:
    """
    统一「占领/扩张」边际分与整数成本 cost。
    非纯距离：收益/ETA，扣留守机会成本、sniping、太阳。
    """
    if dst.owner == my_id:
        return (-1e9, 0, 0)
    cost = dst.ships + (1 if dst.owner == -1 else 5)
    if cost <= 0:
        return (-1e9, 0, 0)

    ships_move = max(1, cost)
    _, _, eta = lead_intercept(state, src, dst, ships_move)
    tl_after = max(1, state.episode_steps - state.step - eta)
    if dst.is_comet:
        tl_after = min(tl_after, 55)

    future_gain = float(dst.production) * float(tl_after)
    hv = hold_value(state, src)
    opp = HOLD_WEIGHT * hv * (float(cost) / max(1.0, float(src.ships)))

    sn = snipe_penalty(state, dst, eta, cost)
    sunp = sun_lane_penalty(state, src, dst, ships_move)

    travel = 0.22 * float(eta)
    enemy_bonus = 32.0 if dst.owner not in (-1, my_id) else 0.0

    score = future_gain + enemy_bonus - float(cost) - travel - sunp - sn - opp
    score /= max(1.0, float(eta) ** 0.35)
    return score, cost, eta


def _phase(state: GameState) -> str:
    r = state.step / max(1, state.episode_steps)
    if r < 0.18:
        return "early"
    if r < 0.62:
        return "mid"
    return "late"


@dataclass(order=True)
class ExpandCand:
    score: float
    src_id: int
    dst_id: int
    cost: int


def build_expand_candidates(state: GameState) -> List[ExpandCand]:
    targets = list(state.neu_pl)
    targets.extend(p for p in state.en_pl if p.production <= 3 or p.ships < 16)

    cands: List[ExpandCand] = []
    for src in state.my_pl:
        scored_dst: List[Tuple[float, Planet]] = []
        for dst in targets:
            if src.id == dst.id:
                continue
            sc, cost, _ = edge_capture_score(state, src, dst, state.my_id)
            if sc <= 0 or cost <= 0:
                continue
            scored_dst.append((sc, dst))
        scored_dst.sort(key=lambda t: -t[0])
        for sc, dst in scored_dst[:7]:
            __, cost, __ = edge_capture_score(state, src, dst, state.my_id)
            cands.append(ExpandCand(-sc, src.id, dst.id, cost))
    cands.sort()
    return cands[:MAX_EDGES_EXPAND_SCAN]


def defend_edges(state: GameState) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    tmp: Dict[int, int] = defaultdict(int)
    for tgt in state.my_pl:
        th = state.net_threat(tgt)
        if th <= 0:
            continue
        helpers = sorted((p for p in state.my_pl if p.id != tgt.id), key=lambda p: p.dist(tgt))
        need = th + 6
        for src in helpers:
            if need <= 0:
                break
            sur = cumulative_surplus(state, src) - tmp.get(src.id, 0)
            if sur <= 0:
                continue
            send = min(sur, need)
            if send <= 0:
                continue
            out.append((src.id, tgt.id, send))
            tmp[src.id] += send
            need -= send
    return out


def coordinated_edges(state: GameState, used: Dict[int, int]) -> List[Tuple[int, int, int]]:
    if not state.en_pl:
        return []
    tgts = sorted(state.en_pl, key=lambda p: -(p.production * 28 - p.ships))[:2]
    out: List[Tuple[int, int, int]] = []
    for tgt in tgts:
        need = tgt.ships + 11 + max(0, state.net_threat(tgt))
        if tgt.production >= 4:
            need += 7

        contributors = sorted(state.my_pl, key=lambda p: p.dist(tgt))
        cand: List[Tuple[Planet, int, int]] = []
        for src in contributors[:COORD_MAX_SOURCES_PER_TARGET]:
            sur = cumulative_surplus(state, src) - used.get(src.id, 0)
            if sur <= 0:
                continue
            probe = min(sur, max(1, need))
            _, _, et = lead_intercept(state, src, tgt, probe)
            cand.append((src, sur, et))

        if not cand:
            continue
        et0 = min(c[2] for c in cand)
        filt = [(s, a, e) for s, a, e in cand if e <= et0 + SYNC_ETA_WINDOW]

        sent = 0
        for src, sur, _ in filt:
            if sent >= need:
                break
            chunk = min(sur, need - sent)
            pulse = max(1, int(chunk * PULSE_ATTACK))
            chunk = min(chunk, pulse)
            if chunk <= 0:
                continue
            out.append((src.id, tgt.id, chunk))
            used[src.id] = used.get(src.id, 0) + chunk
            sent += chunk

        if sent < tgt.ships + 3:
            for s, d, k in list(out):
                if d == tgt.id:
                    used[s] = max(0, used.get(s, 0) - k)
            out = [x for x in out if x[1] != tgt.id]
    return out


def redistribute_edges(state: GameState, used: Dict[int, int]) -> List[Tuple[int, int, int]]:
    if not state.en_pl or len(state.my_pl) < 2:
        return []

    def ned(p: Planet) -> float:
        return min((p.dist(e) for e in state.en_pl), default=999.0)

    ordered = sorted(state.my_pl, key=ned)
    n = len(ordered)
    cut = max(1, min(n - 1, n // 2 + 1))
    front = {p.id for p in ordered[:cut]}
    backs = [p for p in ordered[cut:] if state.net_threat(p) <= 0]
    fronts = [p for p in state.my_pl if p.id in front]
    if not fronts:
        return []

    out: List[Tuple[int, int, int]] = []
    for rear in backs[:REDIST_CAP]:
        sur = cumulative_surplus(state, rear) - used.get(rear.id, 0)
        if sur < 12:
            continue
        fr = min(fronts, key=lambda f: rear.dist(f))
        send = min(sur, max(8, int(sur * 0.38)))
        if send < 8:
            continue
        out.append((rear.id, fr.id, send))
    return out


def agent(obs, config=None):
    t0 = time.time()
    try:
        state = GameState(obs, config)
        if not state.my_pl:
            return []

        used: Dict[int, int] = defaultdict(int)
        moves: List[List] = []
        phase = _phase(state)

        def emit(sid: int, did: int, ships: int) -> bool:
            src = state.get(sid)
            dst = state.get(did)
            if not src or not dst or src.owner != state.my_id:
                return False
            avail = max(0, src.ships - 1 - used[sid])
            send = min(int(ships), avail)
            if send <= 0:
                return False
            ang, _ = safe_aim(state, src, dst, send)
            moves.append([sid, float(ang), send])
            used[sid] += send
            return True

        for sid, did, sh in defend_edges(state):
            if (time.time() - t0) * 1000 > 620:
                break
            emit(sid, did, sh)

        if phase != "early":
            for sid, did, sh in coordinated_edges(state, dict(used)):
                if (time.time() - t0) * 1000 > 720:
                    break
                emit(sid, did, sh)

        for sid, did, sh in redistribute_edges(state, used):
            if (time.time() - t0) * 1000 > 760:
                break
            emit(sid, did, sh)

        cands = build_expand_candidates(state)
        src_done_expand: set = set()
        for cand in cands:
            if len(moves) >= MAX_TOTAL_MOVES:
                break
            if (time.time() - t0) * 1000 > 840:
                break
            if cand.src_id in src_done_expand:
                continue
            sc = -cand.score
            if sc <= 0:
                continue
            src = state.get(cand.src_id)
            dst = state.get(cand.dst_id)
            if not src or not dst:
                continue
            sur = cumulative_surplus(state, src) - used.get(src.id, 0)
            if sur <= 0:
                continue
            cost = cand.cost
            cap = max(1, int(sur * PULSE_EXPAND))
            if dst.owner == -1:
                if sur < cost:
                    continue
                send = min(cost, cap, sur)
                if send < cost:
                    continue
            else:
                if sur < cost:
                    continue
                send = min(cost, cap, sur)
                if send < cost:
                    continue

            if emit(cand.src_id, cand.dst_id, send):
                src_done_expand.add(cand.src_id)

        if phase == "late" and state.en_pl:
            weak = min(state.en_pl, key=lambda p: p.ships)
            for src in state.my_pl:
                if (time.time() - t0) * 1000 > 900:
                    break
                sur = cumulative_surplus(state, src) - used.get(src.id, 0)
                pulse = max(1, int(sur * 0.72))
                if sur > weak.ships + 8:
                    emit(src.id, weak.id, min(sur, pulse))

        return moves
    except Exception:
        return []

