"""Feature extractors aligned with submission v20 / v21_* (NeuralVal.feat + plan summary).

Loads the target submission module once (GameState, Plan, MAX_TOTAL_MOVES, etc.).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission_resolve import load_submission_module

if TYPE_CHECKING:
    pass

# Default bot for RL stack (can override via load_submission("v21_lite"))
_SUBMISSION_VERSION = "v20"
_MOD = load_submission_module(ROOT, _SUBMISSION_VERSION, "feature_v20_default")
_GAMESTATE = _MOD.GameState
_PLAN = _MOD.Plan
_MAX_MOVES = int(_MOD.MAX_TOTAL_MOVES)
_MAX_TARGETS = int(_MOD.MAX_TARGETS_PER_PLAN)
_GLOBAL_NEURAL = _MOD._GLOBAL_NEURAL
N_STATE_FEATURES = int(_GLOBAL_NEURAL.N_FEAT)

PLAN_TAGS = (
    "defense", "intercept", "urgent_hp", "expand", "balanced",
    "counter", "aggro", "comet", "diplo", "late_dump", "redistribute",
)
_TAG_INDEX = {t: i for i, t in enumerate(PLAN_TAGS)}
N_PLAN_FEATURES = 1 + len(PLAN_TAGS) + 5
N_FEATURES = N_STATE_FEATURES + N_PLAN_FEATURES


def set_submission_version(version: str) -> None:
    """Reload backing submission module (e.g. v20, v21_lite). For tests only."""
    global _MOD, _GAMESTATE, _PLAN, _MAX_MOVES, _MAX_TARGETS, _GLOBAL_NEURAL
    global N_STATE_FEATURES
    _MOD = load_submission_module(ROOT, version, f"feature_{version}")
    _GAMESTATE = _MOD.GameState
    _PLAN = _MOD.Plan
    _MAX_MOVES = int(_MOD.MAX_TOTAL_MOVES)
    _MAX_TARGETS = int(_MOD.MAX_TARGETS_PER_PLAN)
    _GLOBAL_NEURAL = _MOD._GLOBAL_NEURAL
    N_STATE_FEATURES = int(_GLOBAL_NEURAL.N_FEAT)


def gamestate_type():
    return _GAMESTATE


def plan_type():
    return _PLAN


def submission_module():
    return _MOD


def state_features(state) -> np.ndarray:
    return _GLOBAL_NEURAL.feat(state)


def plan_features(plan, state) -> np.ndarray:
    out = np.zeros(N_PLAN_FEATURES, dtype=np.float32)
    out[0] = float(np.tanh(plan.score / 500.0))
    idx = _TAG_INDEX.get(plan.tag, -1)
    if idx >= 0:
        out[1 + idx] = 1.0
    n_actions = len(plan.actions)
    total_sent = sum(a[2] for a in plan.actions)
    targets = {a[1] for a in plan.actions}
    sources = {a[0] for a in plan.actions}
    my_total = max(1, state.total_ships(state.my_id))
    off = 1 + len(PLAN_TAGS)
    out[off + 0] = n_actions / float(_MAX_MOVES)
    out[off + 1] = total_sent / float(my_total)
    out[off + 2] = len(targets) / float(_MAX_TARGETS)
    out[off + 3] = len(sources) / max(1.0, float(len(state.my_pl)))
    out[off + 4] = float(np.tanh(math.sqrt(max(1, n_actions)) / 5.0))
    return out


def combined_features(plan, state) -> np.ndarray:
    return np.concatenate([state_features(state), plan_features(plan, state)], dtype=np.float32)
