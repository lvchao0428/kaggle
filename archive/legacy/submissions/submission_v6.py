"""Orbit Wars 启发式 v6 agent（路线 A 阶段 0 修复版）。

针对 ELITE-BOT v5 的 5 处硬伤逐一修复：
1. 太阳半径使用官方 10.0；连续段距离判定避日；
2. 舰队速度用官方公式 1 + (max-1)*(log(ships)/log(1000))**1.5；
3. 每回合多动作输出（多源协同 + 多目标扩张），不再裁成 1-3 条；
4. 不再引入未训练 NEURAL 价值网络做决策；
5. 通过前向直线模拟 + 行星轨道预测重写 fleet 的目标识别（替代角度容差 0.28 rad）。

入口：agent(obs, config) -> List[[from_planet_id, direction_angle_rad, num_ships]]。
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


SUN_X = SUN_Y = 50.0
SUN_RADIUS = 10.0
BOARD = 100.0
ROTATION_RADIUS_LIMIT = 50.0
DEFAULT_MAX_SHIP_SPEED = 6.0
DEFAULT_EPISODE_STEPS = 500


# ---------- geometry / official formulas ----------------------------------------


def fleet_speed(ships: int, max_speed: float = DEFAULT_MAX_SHIP_SPEED) -> float:
    """官方公式：speed = 1 + (max-1) * (log(ships)/log(1000))^1.5。"""
    if ships <= 1:
        return 1.0
    spd = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(spd, max_speed)


def point_segment_distance(px: float, py: float,
                           ax: float, ay: float,
                           bx: float, by: float) -> float:
    """点 P 到线段 AB 的最短距离。"""
    abx, aby = bx - ax, by - ay
    L2 = abx * abx + aby * aby
    if L2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / L2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def segment_hits_sun(ax: float, ay: float, bx: float, by: float,
                     margin: float = 1.5) -> bool:
    return point_segment_distance(SUN_X, SUN_Y, ax, ay, bx, by) < SUN_RADIUS + margin


# ---------- typed observation wrappers ------------------------------------------


class Planet:
    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production",
                 "initial_x", "initial_y", "is_comet")

    def __init__(self, id, owner, x, y, radius, ships, production,
                 initial_x=0.0, initial_y=0.0, is_comet=False):
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
    """高层封装：行星/舰队/玩家/轨道/彗星 + 入侵威胁估算。"""

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
                    id=int(row[0]),
                    owner=int(row[1]),
                    x=float(row[2]),
                    y=float(row[3]),
                    angle=float(row[4]),
                    from_planet_id=int(row[5]),
                    ships=int(row[6]),
                )
            )

        self._pm: Dict[int, Planet] = {p.id: p for p in self.planets}

        self.my_pl = [p for p in self.planets if p.owner == self.my_id]
        self.en_pl = [p for p in self.planets if p.owner not in (-1, self.my_id)]
        self.neu_pl = [p for p in self.planets if p.owner == -1]
        self.en_ids = sorted({p.owner for p in self.en_pl})

        self.max_speed: float = float(_get(_get(obs, "configuration", None) or config or {}, "shipSpeed",
                                           DEFAULT_MAX_SHIP_SPEED) or DEFAULT_MAX_SHIP_SPEED)
        self.episode_steps: int = int(_get(_get(obs, "configuration", None) or config or {}, "episodeSteps",
                                           DEFAULT_EPISODE_STEPS) or DEFAULT_EPISODE_STEPS)

        # 推断目标 + 入侵图
        self.fleet_target: Dict[int, Optional[Tuple[int, int]]] = {}
        self.incoming: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for f in self.fleets:
            tid_eta = self._predict_fleet_target(f)
            self.fleet_target[f.id] = tid_eta
            if tid_eta is not None:
                tid, _eta = tid_eta
                self.incoming[tid][f.owner] += f.ships

    # ---- 行星位置预测：含旋转
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

    # ---- 沿直线模拟舰队，找首先碰撞的星球（连续段判定）
    def _predict_fleet_target(self, f: Fleet, max_steps: int = 200) -> Optional[Tuple[int, int]]:
        spd = fleet_speed(f.ships, self.max_speed)
        cx, cy = f.x, f.y
        dx, dy = math.cos(f.angle) * spd, math.sin(f.angle) * spd
        for t in range(1, max_steps + 1):
            nx, ny = cx + dx, cy + dy
            # 边界 / 太阳 → 销毁
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
        attackers = sum(v for k, v in inc.items() if k not in (-1, self.my_id))
        own_inc = inc.get(self.my_id, 0)
        return attackers - own_inc

    def total_ships(self, owner: int) -> int:
        return (sum(p.ships for p in self.planets if p.owner == owner)
                + sum(f.ships for f in self.fleets if f.owner == owner))


# ---------- aiming with lead + sun avoidance ------------------------------------


def lead_intercept(state: GameState, src: Planet, dst: Planet,
                   ships: int, iters: int = 6) -> Tuple[float, float, int]:
    """返回 (target_x, target_y, eta_turns)。dst 静止时 eta 即直线距离/速度。"""
    spd = fleet_speed(ships, state.max_speed)
    tx, ty = dst.x, dst.y
    eta = max(1, int(math.hypot(tx - src.x, ty - src.y) / spd))
    if state.is_orbiting(dst):
        for _ in range(iters):
            tx, ty = state.planet_pos_at(dst, eta)
            eta = max(1, int(math.hypot(tx - src.x, ty - src.y) / spd))
    return tx, ty, eta


def safe_aim(state: GameState, src: Planet, dst: Planet,
             ships: int) -> Tuple[float, int]:
    """返回 (angle, eta_turns)。若直线穿日，则在 ±0.55 rad 范围内抖动。"""
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
    return angle, eta  # 没找到安全角，原样发（极少见）


# ---------- planning -------------------------------------------------------------


def _budget(state: GameState, p: Planet) -> int:
    """可外出兵力：扣除产量缓冲与净威胁。"""
    threat = max(0, state.net_threat(p))
    reserve = max(p.production * 3, threat + 4) if not p.is_comet else max(1, threat + 2)
    return max(0, p.ships - reserve)


def _phase(state: GameState) -> str:
    progress = state.step / max(1, state.episode_steps)
    if progress < 0.20:
        return "early"
    if progress < 0.65:
        return "mid"
    return "late"


def _expansion_score(state: GameState, src: Planet, dst: Planet) -> float:
    """中立 / 弱敌占领评分：考虑 ROI、距离、避日罚、彗星窗口。"""
    if src.id == dst.id:
        return -1e9
    spd = fleet_speed(max(1, dst.ships + 5), state.max_speed)
    tx, ty, eta = lead_intercept(state, src, dst, dst.ships + 5)
    dist = math.hypot(tx - src.x, ty - src.y)
    cost = dst.ships + 1 if dst.owner == -1 else dst.ships + 5
    if cost <= 0:
        return -1e9
    turns_left = max(1, state.episode_steps - state.step - eta)
    if dst.is_comet:
        turns_left = min(turns_left, 60)
    val = dst.production * turns_left
    bx = src.x + math.cos(math.atan2(ty - src.y, tx - src.x)) * spd * eta
    by = src.y + math.sin(math.atan2(ty - src.y, tx - src.x)) * spd * eta
    sun_pen = 80.0 if segment_hits_sun(src.x, src.y, bx, by, margin=2.0) else 0.0
    enemy_bonus = 35.0 if dst.owner not in (-1, state.my_id) else 0.0
    return (val + enemy_bonus - cost - 0.4 * dist - sun_pen) / max(eta, 1)


def _defend_actions(state: GameState) -> List[Tuple[int, int, int]]:
    """对受威胁星派援军；返回 [(src_id, dst_id, ships)] 列表。"""
    out: List[Tuple[int, int, int]] = []
    for tgt in state.my_pl:
        threat = state.net_threat(tgt)
        if threat <= 0:
            continue
        helpers = sorted(
            (p for p in state.my_pl if p.id != tgt.id),
            key=lambda p: p.dist(tgt),
        )
        need = threat + 6
        for src in helpers:
            if need <= 0:
                break
            avail = _budget(state, src)
            if avail <= 0:
                continue
            send = min(avail, need)
            if send <= 0:
                continue
            out.append((src.id, tgt.id, send))
            need -= send
    return out


def _expansion_actions(state: GameState, used: Dict[int, int]) -> List[Tuple[int, int, int, float]]:
    """对每个我方源星贪心选择若干扩张目标；返回 [(src,dst,ships,score)]。"""
    out: List[Tuple[int, int, int, float]] = []
    targets = state.neu_pl + [
        p for p in state.en_pl
        if p.production <= 2 or p.ships < 12
    ]
    for src in sorted(state.my_pl, key=lambda p: -p.ships):
        bud = _budget(state, src) - used.get(src.id, 0)
        if bud <= 0:
            continue
        ranked = sorted(targets, key=lambda d: -_expansion_score(state, src, d))
        for dst in ranked[:6]:
            if bud <= 0:
                break
            sc = _expansion_score(state, src, dst)
            if sc <= 0:
                continue
            cost = dst.ships + (1 if dst.owner == -1 else 5)
            if cost <= 0 or bud < cost:
                continue
            out.append((src.id, dst.id, cost, sc))
            bud -= cost
    return out


def _coordinated_attack(state: GameState, used: Dict[int, int]) -> List[Tuple[int, int, int]]:
    """中后期：选 1-2 个高产对手星，多源凑足兵力同时打。"""
    if not state.en_pl:
        return []
    sorted_targets = sorted(state.en_pl, key=lambda p: -(p.production * 30 - p.ships))
    out: List[Tuple[int, int, int]] = []
    for tgt in sorted_targets[:2]:
        need = tgt.ships + 12 + (state.net_threat(tgt) if state.net_threat(tgt) > 0 else 0)
        if tgt.production >= 4:
            need += 8
        contributors = sorted(state.my_pl, key=lambda p: p.dist(tgt))
        sent = 0
        for src in contributors:
            if sent >= need:
                break
            avail = max(0, _budget(state, src) - used.get(src.id, 0))
            if avail <= 0:
                continue
            chunk = min(avail, need - sent)
            if chunk <= 0:
                continue
            out.append((src.id, tgt.id, chunk))
            used[src.id] = used.get(src.id, 0) + chunk
            sent += chunk
        # 兵力不足以拿下，回滚以免送菜
        if sent < tgt.ships + 4:
            for s in [m for m in out if m[1] == tgt.id]:
                used[s[0]] = max(0, used.get(s[0], 0) - s[2])
            out = [m for m in out if m[1] != tgt.id]
    return out


# ---------- entrypoint ----------------------------------------------------------


def agent(obs, config=None):
    t0 = time.time()
    try:
        state = GameState(obs, config)
        if not state.my_pl:
            return []

        used: Dict[int, int] = defaultdict(int)
        moves: List[List] = []
        phase = _phase(state)

        def _emit(src_id: int, dst_id: int, ships: int) -> None:
            src = state.get(src_id)
            dst = state.get(dst_id)
            if not src or not dst:
                return
            if src.owner != state.my_id:
                return
            avail = max(0, src.ships - 1 - used[src_id])
            send = min(int(ships), avail)
            if send <= 0:
                return
            ang, _eta = safe_aim(state, src, dst, send)
            moves.append([src_id, float(ang), int(send)])
            used[src_id] += send

        # 1) 防守优先
        for src_id, dst_id, sh in _defend_actions(state):
            if (time.time() - t0) * 1000 > 700:
                break
            _emit(src_id, dst_id, sh)

        # 2) 协同攻击：mid/late 才积极发动；early 主要扩张
        if phase != "early":
            for src_id, dst_id, sh in _coordinated_attack(state, dict(used)):
                if (time.time() - t0) * 1000 > 800:
                    break
                _emit(src_id, dst_id, sh)

        # 3) 扩张：always-on
        for src_id, dst_id, sh, _sc in _expansion_actions(state, dict(used)):
            if (time.time() - t0) * 1000 > 850:
                break
            _emit(src_id, dst_id, sh)

        # 4) 兜底：late 阶段若仍有大量囤积，向最弱敌主动倾泻
        if phase == "late" and state.en_pl:
            weakest = min(state.en_pl, key=lambda p: p.ships)
            for src in state.my_pl:
                avail = max(0, src.ships - 1 - used[src.id])
                if avail > weakest.ships + 6:
                    _emit(src.id, weakest.id, avail)

        return moves
    except Exception:
        return []
