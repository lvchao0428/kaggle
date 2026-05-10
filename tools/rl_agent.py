"""Trainable Orbit Wars agent for self-play RL.

Wraps v11's PlanArbiter pipeline; the only difference is that plan scoring
goes through a PyTorch policy net (or NumPy reference weights) instead of
v11's static heuristic + NeuralVal modifier. Records per-turn transitions
for PPO.

The class is intentionally callable as `agent(obs, config)` so it can run
inside `kaggle_environments.evaluate(...)` exactly like the static bots.

Phase 3 change: _select_plan now runs a proper softmax over net plan-scores
(rather than combining with heuristic base). The chosen_idx is stored as
the policy target for cross-entropy loss in the learner.

Two callsites:
- rollout_worker.py uses RLAgent(policy_state_dict=..., explore=True, record=True)
- distill check: RLAgent(explore=False, record=False)
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import submission_v11 as v11  # type: ignore
from .feature_extractor import (
    N_FEATURES,
    combined_features,
    state_features,
)


@dataclass
class Transition:
    """One per-turn (state, action) record kept for PPO."""

    obs_feat: np.ndarray            # combined feature of CHOSEN plan, shape (N_FEATURES,)
    plan_feats: np.ndarray          # feats of all candidates, shape (K, N_FEATURES)
    chosen_idx: int                 # Phase 3: index of chosen plan in candidates
    log_prob: float                 # log prob of chosen plan under sampling distribution
    value_pred: float               # net's value estimate of chosen plan
    plan_score_net: float           # net's plan-score (policy logit before softmax)
    step: int
    state_feat: np.ndarray          # state-only 14-dim feature (for distillation later)
    shaped_reward: float = 0.0      # Phase 2: set by rollout_worker after game ends
    oob_penalty: float = 0.0        # v15: modeled straight-line endpoint off-board


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def _oob_penalty_for_plan(state: "v11.GameState", plan: "v11.Plan") -> float:
    """Penalize launches whose constant-speed straight endpoint leaves the board."""
    m = 0.5
    b = float(v11.BOARD)
    pen = 0.0
    for sid, did, ships in plan.actions:
        sp = state.get(sid)
        dp = state.get(did)
        if sp is None or dp is None:
            continue
        n = max(1, int(ships))
        ang, eta = v11.safe_aim(state, sp, dp, n)
        spd = v11.fleet_speed(n, state.max_speed)
        ex = sp.x + math.cos(ang) * spd * eta
        ey = sp.y + math.sin(ang) * spd * eta
        if ex < m or ex > b - m or ey < m or ey > b - m:
            pen -= 0.05
    return pen


class RLAgent:
    """Agent whose plan-ranking head is a tiny NumPy MLP (trainable from
    PyTorch checkpoints). Behaves like a regular Kaggle agent function.

    Phase 3: when weights are present, plan selection is a pure softmax over
    net plan-scores.  When weights are absent (untrained), falls back to
    heuristic base scores for stable play.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, np.ndarray]] = None,
        explore: bool = True,
        record: bool = True,
        temperature: float = 1.0,
        plan_score_strength: float = 1.0,   # Phase 3: 1.0 = full net steering
    ):
        self.weights = weights      # if None: random init, agent ~ v11 baseline
        self.explore = explore
        self.record = record
        self.temperature = max(1e-3, temperature)
        self.plan_score_strength = plan_score_strength
        self.transitions: List[Transition] = []

    # ── public Kaggle entry ──
    def __call__(self, obs, config=None):
        return self._step(obs, config)

    # ── core logic ──
    def _step(self, obs, config) -> List[List]:
        try:
            state = v11.GameState(obs, config)
            if not state.my_pl:
                return []

            v11._GLOBAL_OPP.update(state)
            policy = v11.PhasePolicy.for_state(state)
            snap = v11.Snapshot.build(state, policy)
            diplo = v11.DiplomacyEngine(state, v11._GLOBAL_OPP)

            # Same arbiter pipeline as v11 except score-with-modifiers.
            arbiter = v11.PlanArbiter(snap, diplo, v11._GLOBAL_NEURAL,
                                      elapsed_ms_fn=lambda: 0.0,
                                      deadline_ms=920.0)
            arbiter.commit_urgent()
            plans = arbiter.collect_strategic()
            if not plans:
                arbiter.commit_fallback()
                return arbiter.moves

            # Phase 3: select plan via net softmax (or heuristic fallback).
            chosen_idx, log_prob, value_pred, plan_logit, all_feats = self._select_plan(
                plans, state)
            arbiter.commit_best([(0.0, plans[chosen_idx])])
            arbiter.commit_fallback()

            if self.record:
                oob_pen = _oob_penalty_for_plan(state, plans[chosen_idx])
                self.transitions.append(Transition(
                    obs_feat=all_feats[chosen_idx],
                    plan_feats=all_feats,
                    chosen_idx=chosen_idx,
                    log_prob=float(log_prob),
                    value_pred=float(value_pred),
                    plan_score_net=float(plan_logit),
                    step=state.step,
                    state_feat=state_features(state),
                    oob_penalty=float(oob_pen),
                ))
            return arbiter.moves
        except Exception:
            return []

    def _select_plan(
        self, plans: List["v11.Plan"], state: "v11.GameState",
    ) -> Tuple[int, float, float, float, np.ndarray]:
        """Returns (chosen_idx, log_prob, value_pred, plan_score_net, all_feats).

        Phase 3 logic:
        - If weights are loaded: score plans purely via net plan-scores
          (softmax selection when exploring, argmax when exploiting).
        - If no weights: fall back to v11 heuristic base scores so an
          untrained agent still plays reasonably.
        """
        n = len(plans)
        all_feats = np.stack([combined_features(p, state) for p in plans])  # (K, F)

        if self.weights is not None:
            # Pure net plan-score selection (Phase 3).
            net_scores = self._forward_net_scores(all_feats)   # (K,)
            value_pred = float(self._forward_value(all_feats[0:1])[0])

            if self.explore:
                probs = _softmax(net_scores / self.temperature)
                idx = int(np.random.choice(n, p=probs))
                log_prob = float(np.log(probs[idx] + 1e-9))
            else:
                idx = int(np.argmax(net_scores))
                log_prob = 0.0
            return idx, log_prob, value_pred, float(net_scores[idx]), all_feats

        # Fallback: heuristic base scores (no weights).
        sim_steps = state.phase()
        sim_steps_int = 8 if sim_steps != "late" else 10
        base = np.array([
            p.score + v11.score_plan_actions(state, p.actions, steps=sim_steps_int,
                                             tempo_floor=1)
            for p in plans
        ], dtype=np.float32)

        if self.explore:
            probs = _softmax(base / self.temperature)
            idx = int(np.random.choice(n, p=probs))
            log_prob = float(np.log(probs[idx] + 1e-9))
        else:
            idx = int(np.argmax(base))
            log_prob = 0.0
        return idx, log_prob, 0.0, float(base[idx]), all_feats

    # ── tiny NumPy forward (matches PolicyValueNet shape) ──
    def _forward_trunk(self, x: np.ndarray) -> np.ndarray:
        w = self.weights
        h1 = np.maximum(0.0, x @ w["W1"].T + w["b1"])
        h2 = np.maximum(0.0, h1 @ w["W2"].T + w["b2"])
        return h2

    def _forward_value(self, x: np.ndarray) -> np.ndarray:
        h = self._forward_trunk(x)
        return (h @ self.weights["Wv"].T + self.weights["bv"]).reshape(-1)

    def _forward_net_scores(self, x: np.ndarray) -> np.ndarray:
        h = self._forward_trunk(x)
        out = h @ self.weights["Wp"].T + self.weights["bp"]
        return np.tanh(out).reshape(-1)


def torch_state_dict_to_numpy(sd: Dict) -> Dict[str, np.ndarray]:
    """Map keys from PolicyValueNet.state_dict() to RLAgent weight dict."""
    def t(k):
        return sd[k].detach().cpu().numpy().astype(np.float32)
    return {
        "W1": t("trunk.0.weight"),
        "b1": t("trunk.0.bias"),
        "W2": t("trunk.2.weight"),
        "b2": t("trunk.2.bias"),
        "Wv": t("value_head.weight"),
        "bv": t("value_head.bias"),
        "Wp": t("plan_head.weight"),
        "bp": t("plan_head.bias"),
    }
