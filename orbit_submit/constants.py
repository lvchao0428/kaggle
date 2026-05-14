import base64
import io
import math
import os
import random
import time
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from scipy.cluster.hierarchy import fclusterdata as _scipy_fclusterdata
except ImportError:
    _scipy_fclusterdata = None  # type: ignore[misc, assignment]


# Re-export for regional / tests
ORB_STRATEGY_PROFILE: ContextVar[Optional[str]] = ContextVar(
    "ORB_STRATEGY_PROFILE", default=None
)
# ╔═══ region 0: constants & helpers ════════════════════════════════════════╗

SUN_X = SUN_Y = 50.0
SUN_RADIUS = 10.0
# Extra clearance for “direct lane” filters (target_score / _emit) and regional detour heuristic.
SUN_PATH_MARGIN = 3.5
BOARD = 100.0
ROTATION_RADIUS_LIMIT = 50.0
DEFAULT_MAX_SHIP_SPEED = 6.0
DEFAULT_EPISODE_STEPS = 500

MAX_TOTAL_MOVES = 26
# Multi-source capture: start with a tight ETA band around the fastest source; if
# the band cannot meet ``required`` ships, widen up to SYNC_ETA_WINDOW_MAX so a
# fat neutral (e.g. 45-stack) can still receive a delayed third source. Regional
# score was fine — silent failure was purely contributor shortage under narrow sync.
SYNC_ETA_WINDOW = 5
SYNC_ETA_WINDOW_MAX = 26
MAX_SOURCES_PER_TARGET = 8
MAX_TARGETS_PER_PLAN = 3      # 集中力量：最多同时打3个目标
# expand/balanced：己方星球很少时，禁止同一出兵星在本 plan 里打两个不同目标（避免 8+12 拆开浪费）。
ONE_OUTBOUND_DST_PER_SOURCE_UNTIL_N_WORLDS = 3
ABS_MIN_BATCH = 8  # 提高下限减少碎片化小舰队（配合 early 略降 growth_lock）
# Expand / balanced: edge scores below this were dropped from ranked targets — too
# aggressive vs regional cross-zone tax; leaves fat planets idle on contested greys.
EXPAND_RANK_SCORE_FLOOR = -44.0
# Neutral + is_safe_investment False: still try capture if pooled surplus is huge
# (first wave failed vs orange but HQ stack is still rich).
NEUTRAL_BRUTE_SLACK_MUL = 2
NEUTRAL_BRUTE_SLACK_MIN = 52
# First step / solo HQ: cap reserve so opening can peel ~ships+1 vs neutral-20
# (symmetric ladder seeds; see tools/sim_first_turn_opening.py).
OPENING_FIRST_CAPTURE_SEND = 21
# While still one friendly planet and no inbound siege, cap reserve so a ~20-stack
# neutral can be taken as soon as HQ reaches OPENING_FIRST_CAPTURE_SEND (+ batch
# rules), not only on step 0. (Default start is ~10 ships; peeling ramps up later.)
OPENING_SOLO_HQ_RESERVE_LAST_STEP = 48

# Bocsimacko (2010 Planet Wars champion) caps per-planet scoring at the
# horizon - distant production is heavily discounted instead of accumulated
# linearly to game-end. Combined with the small enemy-ship positional pen,
# this biases the bot toward proximate, certain gains.
HORIZON_TURNS = 60
ENEMY_SHIP_PEN_COEFF = 0.0008  # tiny - only breaks ties

# Orbital distance (initial) from sun for “four inner clusters” on symmetric ladder
# maps. Used to garrison and avoid stripping these worlds while they rotate into
# enemy arcs (user: clockwise → need mass, not dribbling to passing neutrals).
INNER_SUN_BELT_R = 33.0


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


def swept_pair_hit(
    ax: float, ay: float, bx: float, by: float,
    p0x: float, p0y: float, p1x: float, p1y: float, r: float,
) -> bool:
    """True iff segment A->B and segment P0->P1 come within r (orbit_wars.py)."""
    d0x, d0y = ax - p0x, ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0

