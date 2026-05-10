"""Feature extractors for Orbit Wars RL pipeline.

The state-only feature vector matches v11's NeuralVal.feat() exactly so the
distilled student is a drop-in replacement. The plan-conditioned vector is
the state vector concatenated with a plan-summary vector (used during PPO
training so the policy can rank plans).

Imports submission_v11 to reuse `Snapshot.build`, `target_score`, etc.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import submission_v11 as v11  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# State features (14 dims) — identical layout to v11.NeuralVal.feat
# ─────────────────────────────────────────────────────────────────────────────

N_STATE_FEATURES = v11.NeuralVal.N_FEAT  # 14


def state_features(state: "v11.GameState") -> np.ndarray:
    """Returns the 14-dim state feature vector used by v11.NeuralVal."""
    return v11._GLOBAL_NEURAL.feat(state)


# ─────────────────────────────────────────────────────────────────────────────
# Plan-conditioned features (state + plan summary)
# ─────────────────────────────────────────────────────────────────────────────

# Plan summary dims:
#   - plan.score normalized
#   - mode one-hot (8 modes: defense, intercept, urgent_hp, expand, balanced,
#                   counter, aggro, comet, diplo, late_dump, redistribute = 11)
#   - n_actions / 26
#   - total_ships_sent / total_my_ships
#   - n_unique_targets / 6
#   - n_unique_sources / len(my_pl)
#   - mean_eta_normalised (proxy: sqrt(n_actions))
N_PLAN_FEATURES = 1 + 11 + 5  # score + mode_onehot + 5 scalars
N_FEATURES = N_STATE_FEATURES + N_PLAN_FEATURES  # 14 + 17 = 31

PLAN_TAGS = (
    "defense", "intercept", "urgent_hp", "expand", "balanced",
    "counter", "aggro", "comet", "diplo", "late_dump", "redistribute",
)
_TAG_INDEX = {t: i for i, t in enumerate(PLAN_TAGS)}


def plan_features(plan: "v11.Plan", state: "v11.GameState") -> np.ndarray:
    """Returns 17-dim plan-summary vector."""
    out = np.zeros(N_PLAN_FEATURES, dtype=np.float32)
    # score (tanh-normalised so it stays bounded)
    out[0] = float(np.tanh(plan.score / 500.0))
    # mode one-hot
    idx = _TAG_INDEX.get(plan.tag, -1)
    if idx >= 0:
        out[1 + idx] = 1.0
    # scalar summaries
    n_actions = len(plan.actions)
    total_sent = sum(a[2] for a in plan.actions)
    targets = {a[1] for a in plan.actions}
    sources = {a[0] for a in plan.actions}
    my_total = max(1, state.total_ships(state.my_id))
    out[12] = n_actions / float(v11.MAX_TOTAL_MOVES)
    out[13] = total_sent / float(my_total)
    out[14] = len(targets) / float(v11.MAX_TARGETS_PER_PLAN)
    out[15] = len(sources) / max(1.0, float(len(state.my_pl)))
    out[16] = float(np.tanh(math.sqrt(max(1, n_actions)) / 5.0))
    return out


def combined_features(plan: "v11.Plan", state: "v11.GameState") -> np.ndarray:
    """31-dim feature vector = state(14) ++ plan(17)."""
    return np.concatenate([state_features(state), plan_features(plan, state)],
                          dtype=np.float32)
