from __future__ import annotations

from orbit_submit.constants import *
from orbit_submit.entities import Planet, _combat
from orbit_submit.game_state import GameState

def is_sun_belt_planet(state: GameState, p: Planet) -> bool:
    """Inner rotating ring around the sun (not comets, not static far-outs)."""
    if p.is_comet or not state.is_orbiting(p):
        return False
    r0 = math.hypot(p.initial_x - SUN_X, p.initial_y - SUN_Y)
    return r0 + float(p.radius) <= INNER_SUN_BELT_R


# ╔═══ region 3: Snapshot & geometry ════════════════════════════════════════╗

def lead_intercept(state: GameState, src: Planet, dst: Planet, ships: int,
                   iters: int = 12) -> Tuple[float, float, int, float]:
    """Iteratively converge on intercept of ``dst`` **disk center** from ``src``.

    Each step uses ``launch_origin`` (rim + ENGINE_LAUNCH_PAD) as the true start,
    so bearing/ETA match the engine and moving orbit targets get a proper lead
    (提前量). Orbiting, non-comet targets apply ``ORBIT_AIM_LEAD_STEPS`` to push
    the aim along orbit after convergence (often ~1–2t earlier contact). Returns
    ``(tx, ty, eta, aim_angle)`` where ``aim_angle`` is from rim toward the
    predicted center.
    """
    spd = fleet_speed(max(1, ships), state.max_speed)
    eta = max(1, int(math.ceil(math.hypot(dst.x - src.x, dst.y - src.y) / spd)))
    angle = math.atan2(dst.y - src.y, dst.x - src.x)
    n = max(iters, 10)
    for _ in range(n):
        tx, ty = state.planet_pos_at(dst, eta)
        lx, ly = launch_origin(src, angle)
        dist = math.hypot(tx - lx, ty - ly)
        new_eta = max(1, int(math.ceil(dist / spd))) if dist > 1e-9 else 1
        new_angle = math.atan2(ty - ly, tx - lx)
        if new_eta == eta and abs(new_angle - angle) < 1e-4:
            break
        eta = new_eta
        angle = new_angle
    lead = ORBIT_AIM_LEAD_STEPS if (state.is_orbiting(dst) and not dst.is_comet) else 0
    tx, ty = state.planet_pos_at(dst, eta + lead)
    _MARGIN = 1.0
    tx = max(_MARGIN, min(BOARD - _MARGIN, tx))
    ty = max(_MARGIN, min(BOARD - _MARGIN, ty))
    lx, ly = launch_origin(src, angle)
    angle = math.atan2(ty - ly, tx - lx)
    eta = max(1, int(math.ceil(math.hypot(tx - lx, ty - ly) / spd)))
    return tx, ty, eta, angle


def _ray_safe(sx: float, sy: float, angle: float, spd: float, min_flight: int = 0) -> bool:
    """Check if a ray from (sx,sy) at angle with speed spd is safe.
    - Never enters the sun (entire flight)
    - If min_flight > 0, must stay in-board for at least that many steps
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for e in range(1, 80):
        ex = sx + cos_a * spd * e
        ey = sy + sin_a * spd * e
        if not (0.0 <= ex <= BOARD and 0.0 <= ey <= BOARD):
            if e <= min_flight:
                return False  # exits board before reaching target
            return True
        if math.hypot(ex - SUN_X, ey - SUN_Y) < SUN_RADIUS + 1.0:
            return False
    return True


LAUNCH_SIM_MAX_STEPS = 220
# Must match kaggle_environments/envs/orbit_wars/orbit_wars.py fleet spawn.
ENGINE_LAUNCH_PAD = 0.1
# Discrete ETA + rim spawn tends to under-lead orbiting neutrals; nudge the aim
# point forward along orbit so fleets meet the center ~1–2 steps earlier in practice.
ORBIT_AIM_LEAD_STEPS = 2


def launch_origin(src: Planet, angle: float) -> Tuple[float, float]:
    """Point just outside `src`'s disk on the launch bearing (engine uses radius+0.1)."""
    r = float(src.radius) + ENGINE_LAUNCH_PAD
    return src.x + math.cos(angle) * r, src.y + math.sin(angle) * r


def launch_intercept_step(
    state: GameState,
    src_x: float,
    src_y: float,
    angle: float,
    ships: int,
    target_id: int,
    max_steps: int = LAUNCH_SIM_MAX_STEPS,
    ignore_planet_id: Optional[int] = None,
) -> Optional[int]:
    """First simulation step (1-based) where fleet hits ``target_id``, else None."""
    spd = fleet_speed(max(1, int(ships)), state.max_speed)
    cx, cy = float(src_x), float(src_y)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for k in range(1, max_steps + 1):
        fx0, fy0 = cx, cy
        fx1, fy1 = cx + cos_a * spd, cy + sin_a * spd
        hit_id: Optional[int] = None
        for p in state.planets:
            if ignore_planet_id is not None and p.id == ignore_planet_id:
                continue
            p0, p1 = state.planet_motion_segment(p, k)
            if p0[0] < 0:
                continue
            if swept_pair_hit(fx0, fy0, fx1, fy1,
                              p0[0], p0[1], p1[0], p1[1], p.radius):
                hit_id = p.id
                break
        if hit_id is not None:
            return k if hit_id == target_id else None
        if not (0.0 <= fx1 <= BOARD and 0.0 <= fy1 <= BOARD):
            return None
        if point_segment_distance(SUN_X, SUN_Y, fx0, fy0, fx1, fy1) < SUN_RADIUS:
            return None
        cx, cy = fx1, fy1
    return None


def launch_hits_target_first(
    state: GameState,
    src_x: float,
    src_y: float,
    angle: float,
    ships: int,
    target_id: int,
    max_steps: int = LAUNCH_SIM_MAX_STEPS,
    ignore_planet_id: Optional[int] = None,
) -> bool:
    """True iff the first planetary contact (orbit_wars swept collision order) is target_id.

    Uses the same relative-motion segment test as the official environment, not
    a frozen planet at integer steps.
    """
    return launch_intercept_step(
        state, src_x, src_y, angle, ships, target_id,
        max_steps=max_steps, ignore_planet_id=ignore_planet_id,
    ) is not None


def safe_aim(state: GameState, src: Planet, dst: Planet, ships: int) -> Tuple[float, int]:
    """Return (angle, eta). Pick rim bearing with **earliest** stepped intercept.

    Orbits can admit two families of feasible shots (short meet vs long chase).
    Taking the first passing candidate used to lock in the slow clockwise chase;
    we now minimize intercept time so fleets prefer the short arc (user: CCW vs CW).
    """
    nships = max(1, int(ships))
    spd = fleet_speed(nships, state.max_speed)
    tx, ty, eta_hint, angle0 = lead_intercept(state, src, dst, nships)
    tx = max(1.0, min(BOARD - 1.0, tx))
    ty = max(1.0, min(BOARD - 1.0, ty))
    lx0, ly0 = launch_origin(src, angle0)
    angle0 = math.atan2(ty - ly0, tx - lx0)
    tid = dst.id

    deltas = (0.15, -0.15, 0.30, -0.30, 0.50, -0.50, 0.75, -0.75,
              1.05, -1.05, 1.40, -1.40, 1.80, -1.80, 2.20, -2.20,
              2.60, -2.60, 2.85, -2.85, 3.00, -3.00, 3.02, -3.02,
              3.14, -3.14)
    cands: List[float] = [angle0, angle0 + math.pi]
    for d in deltas:
        cands.append(angle0 + d)

    def best_from_pool(pool: List[float], min_flight: int) -> Optional[Tuple[float, int]]:
        best_s: Optional[int] = None
        best_a: Optional[float] = None
        mf = max(1, min_flight)
        for a in pool:
            lx, ly = launch_origin(src, a)
            if not _ray_safe(lx, ly, a, spd, min_flight=mf):
                continue
            st = launch_intercept_step(
                state, lx, ly, a, nships, tid,
                max_steps=LAUNCH_SIM_MAX_STEPS, ignore_planet_id=None)
            if st is None:
                continue
            if best_s is None or st < best_s:
                best_s, best_a = st, a
        if best_a is None or best_s is None:
            return None
        return best_a, int(best_s)

    got = best_from_pool(cands, 1)
    if got is not None:
        return got

    got = best_from_pool(cands, max(1, eta_hint - 1))
    if got is not None:
        return got

    safe_corners = [(2.0, 2.0), (2.0, 98.0), (98.0, 2.0), (98.0, 98.0)]
    corner = max(safe_corners, key=lambda c: math.hypot(c[0] - src.x, c[1] - src.y))
    ca = math.atan2(corner[1] - src.y, corner[0] - src.x)
    got = best_from_pool([ca], max(1, eta_hint - 1))
    if got is not None:
        return got

    for a in cands:
        lx, ly = launch_origin(src, a)
        if _ray_safe(lx, ly, a, spd, min_flight=max(1, eta_hint - 1)):
            return a, int(eta_hint)
    lx_ca, ly_ca = launch_origin(src, ca)
    if _ray_safe(lx_ca, ly_ca, ca, spd, min_flight=max(1, eta_hint - 1)):
        return ca, int(eta_hint)
    return angle0, int(eta_hint)


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


def my_inbound_ships_to(state: GameState, planet_id: int) -> int:
    """Our fleets already in flight toward ``planet_id`` (obs snapshot)."""
    s = 0
    mi = state.my_id
    for f in state.fleets:
        if f.owner != mi:
            continue
        t = state.fleet_target.get(f.id)
        if t is not None and t[0] == planet_id:
            s += f.ships
    return int(s)


def neutral_wave_wins(state: GameState, dst: Planet, eta: int,
                       send: int, other_my_inbound: int) -> bool:
    """True if ``send`` + inbound allies strictly beats projected grey garrison at eta."""
    if dst.owner != -1:
        return True
    _, gar = target_state_at(state, dst, eta)
    return (send + other_my_inbound) > gar


def capture_need(state: GameState, src: Planet, dst: Planet,
                 margin: Optional[int] = None) -> Tuple[int, int]:
    """Iterative estimate of (need, eta) to capture dst from src."""
    early = state.phase() == "early"
    if dst.owner == -1:
        pad0 = 1 if early else 2
    else:
        pad0 = 8
    need = max(ABS_MIN_BATCH, dst.ships + pad0)
    eta = 1
    for _ in range(4):
        _, _, eta, _ = lead_intercept(state, src, dst, need)
        owner, ships = target_state_at(state, dst, eta)
        if owner == state.my_id:
            need = max(ABS_MIN_BATCH, state.net_threat(dst) + 4)
        else:
            if margin is not None:
                base_margin = margin
            elif owner == -1:
                # Early: tighter neutral padding so first waves leave closer to
                # opponents' razor timings (~ships+1) instead of +3 lag.
                base_margin = 1 if early else 3
            else:
                base_margin = 8 + min(6, dst.production)
            extra_prod = (dst.production * eta) if owner not in (-1, state.my_id) else 0
            need = ships + base_margin + extra_prod // 4
        need = max(ABS_MIN_BATCH, int(need))
    return need, eta
