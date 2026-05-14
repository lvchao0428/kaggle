from __future__ import annotations

from orbit_submit.constants import *
from orbit_submit.constants import _get
from orbit_submit.entities import Fleet, Planet, _combat


# ╔═══ region 2: GameState ══════════════════════════════════════════════════╗

class GameState:
    """Parses obs once. Holds planets/fleets and incoming-fleet metadata."""

    def __init__(self, obs, config=None, ruleset: str = "v21"):
        rs = ruleset.strip().lower() if isinstance(ruleset, str) else "v21"
        self.ruleset = rs if rs in ("v20", "v21") else "v21"
        self.my_id = int(_get(obs, "player", 0) or 0)
        self.ang_vel = float(_get(obs, "angular_velocity", 0.0) or 0.0)
        self.step = int(_get(obs, "step", 0) or 0)

        cfg = _get(obs, "configuration", None) or config or {}
        self.max_speed = float(_get(cfg, "shipSpeed", DEFAULT_MAX_SHIP_SPEED)
                               or DEFAULT_MAX_SHIP_SPEED)
        self.episode_steps = int(_get(cfg, "episodeSteps", DEFAULT_EPISODE_STEPS)
                                 or DEFAULT_EPISODE_STEPS)

        if self.ruleset == "v21":
            # Observation may embed a stripped configuration without ``spawn_positions``;
            # runner still passes full ``config`` → merge so FFA gates stay correct.
            sp_raw = _get(cfg, "spawn_positions", None)
            if not isinstance(sp_raw, (list, tuple)) or len(sp_raw) == 0:
                sp_fallback = _get(config, "spawn_positions", None) if config else None
                if isinstance(sp_fallback, (list, tuple)) and len(sp_fallback) > 0:
                    sp_raw = sp_fallback
            if isinstance(sp_raw, (list, tuple)):
                self.spawn_positions: List = list(sp_raw)
            else:
                self.spawn_positions = []
        else:
            self.spawn_positions = []
        self.spawn_count = len(self.spawn_positions)

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

    def planet_motion_segment(self, p: Planet, k: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """(old_pos, new_pos) for planet ``p`` during fleet's k-th move (k>=1), per orbit_wars."""
        if p.is_comet and p.id in self.comet_paths:
            path, idx = self.comet_paths[p.id]
            if k == 1:
                p_old = (p.x, p.y)
                if idx + 1 >= len(path):
                    p_new = p_old
                else:
                    p_new = (float(path[idx + 1][0]), float(path[idx + 1][1]))
                return p_old, p_new
            j0 = min(idx + k - 1, len(path) - 1)
            j1 = min(idx + k, len(path) - 1)
            return ((float(path[j0][0]), float(path[j0][1])),
                    (float(path[j1][0]), float(path[j1][1])))
        if not self.is_orbiting(p):
            xy = (p.x, p.y)
            return xy, xy
        if k == 1:
            p_old = (p.x, p.y)
            r = math.hypot(p.initial_x - SUN_X, p.initial_y - SUN_Y)
            a0 = math.atan2(p.initial_y - SUN_Y, p.initial_x - SUN_X)
            a1 = a0 + self.ang_vel * self.step
            p_new = (SUN_X + r * math.cos(a1), SUN_Y + r * math.sin(a1))
            return p_old, p_new
        r = math.hypot(p.initial_x - SUN_X, p.initial_y - SUN_Y)
        a0 = math.atan2(p.initial_y - SUN_Y, p.initial_x - SUN_X)
        old_i = self.step + k - 2
        new_i = self.step + k - 1
        a_old = a0 + self.ang_vel * old_i
        a_new = a0 + self.ang_vel * new_i
        return ((SUN_X + r * math.cos(a_old), SUN_Y + r * math.sin(a_old)),
                (SUN_X + r * math.cos(a_new), SUN_Y + r * math.sin(a_new)))

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

    def is_ffa_mode(self) -> bool:
        """Four-player FFA: ``spawn_positions`` has 4+ entries on Kaggle.

        Local ``kaggle_environments`` often omits ``spawn_positions``; infer FFA when
        there are **two or more distinct enemy player ids** (4p → 3 opponents).
        """
        if self.ruleset == "v20":
            return False
        if self.spawn_count >= 4:
            return True
        return len(self.en_ids) >= 2

    def is_duel_mode(self) -> bool:
        """Classic 1v1: two spawn slots, or exactly one opponent when spawns missing."""
        if self.ruleset == "v20":
            return len(self.en_ids) == 1
        if self.spawn_count == 2:
            return True
        return len(self.en_ids) == 1

    def phase(self) -> str:
        progress = self.step / max(1, self.episode_steps)
        if self.ruleset == "v20":
            if progress < 0.18:
                return "early"
            if progress < 0.64:
                return "mid"
            return "late"
        # FFA episodes often end ~200–280 steps; stretch late/mid boundaries and
        # add a turns-left cliff so LateDump / late policy row can engage in time.
        if self.is_ffa_mode():
            tl = self.turns_left()
            if tl <= max(90, int(self.episode_steps * 0.22)):
                return "late"
            if progress < 0.14:
                return "early"
            if progress < 0.36:
                return "mid"
            return "late"
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

