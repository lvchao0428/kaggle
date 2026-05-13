"""Trainable Orbit Wars agent on v20 stack (plan-level RL, PyTorch policy)."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v21 import feature_extractor_v20 as fe
from tools.v21.feature_extractor_v20 import (
    combined_features,
    set_submission_version,
    state_features,
)
from tools.v21.nets import build_net
from tools.v21.nets import best_device as net_best_device


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class Transition:
    obs_feat: np.ndarray
    plan_feats: np.ndarray
    chosen_idx: int
    log_prob: float
    value_pred: float
    plan_score_net: float
    step: int
    state_feat: np.ndarray
    shaped_reward: float = 0.0
    oob_penalty: float = 0.0
    # Board scalars at this step (for logging / game_summary; cheap ints)
    my_ships: int = 0
    enemy_ships: int = 0
    n_planets: int = 0
    n_my_planets: int = 0


def _oob_penalty_for_plan(sm, state, plan) -> float:
    m = 0.5
    b = float(sm.BOARD)
    pen = 0.0
    for sid, did, ships in plan.actions:
        sp = state.get(sid)
        dp = state.get(did)
        if sp is None or dp is None:
            continue
        n = max(1, int(ships))
        ang, eta = sm.safe_aim(state, sp, dp, n)
        spd = sm.fleet_speed(n, state.max_speed)
        ex = sp.x + math.cos(ang) * spd * eta
        ey = sp.y + math.sin(ang) * spd * eta
        if ex < m or ex > b - m or ey < m or ey > b - m:
            pen -= 0.05
    return pen


class RLAgentV21:
    def __init__(
        self,
        tier: str = "lite",
        submission_version: str = "v20",
        checkpoint_path: Optional[str] = None,
        explore: bool = True,
        record: bool = True,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.tier = tier.lower().strip()
        self.explore = explore
        self.record = record
        self.temperature = max(1e-3, float(temperature))
        self.transitions: List[Transition] = []

        set_submission_version(submission_version)
        self._sm = fe.submission_module()

        dev = device or net_best_device()
        self.device = dev
        self.net = build_net(self.tier).to(dev)
        self.net.eval()
        self._has_weights = False
        if checkpoint_path and Path(checkpoint_path).is_file():
            try:
                sd = torch.load(checkpoint_path, map_location=dev, weights_only=True)
            except TypeError:
                sd = torch.load(checkpoint_path, map_location=dev)
            self.net.load_state_dict(sd, strict=True)
            self._has_weights = True

    def __call__(self, obs, config=None):
        return self._step(obs, config)

    def _build_regional(self, state, config):
        rg = None
        mh = None
        try:
            spawn = config.get("spawn_positions", []) if config else []
            rg = self._sm.RegionalGraph(state.planets, spawn)
            tl = self._sm.ProductionTimeline(
                state.planets, set(p.id for p in state.my_pl)
            )
            mh = self._sm.MultiHopPlanner(rg, tl)
        except Exception:
            rg, mh = None, None
        return rg, mh

    def _step(self, obs, config) -> List[List]:
        try:
            sm = self._sm
            state = sm.GameState(obs, config)
            if not state.my_pl:
                return []

            sm._GLOBAL_OPP.update(state)
            policy = sm.PhasePolicy.for_state(state)
            snap = sm.Snapshot.build(state, policy)
            diplo = sm.DiplomacyEngine(state, sm._GLOBAL_OPP)
            rg, mh = self._build_regional(state, config or {})

            arbiter = sm.PlanArbiter(
                snap,
                diplo,
                sm._GLOBAL_NEURAL,
                elapsed_ms_fn=lambda: 0.0,
                deadline_ms=920.0,
                regional_graph=rg,
                multi_hop_planner=mh,
            )
            arbiter.commit_urgent()
            plans = arbiter.collect_strategic()
            if not plans:
                arbiter.commit_fallback()
                return arbiter.moves

            chosen_idx, log_prob, value_pred, plan_logit, all_feats = self._select_plan(
                plans, state
            )
            arbiter.commit_best([(0.0, plans[chosen_idx])])
            arbiter.commit_fallback()

            if self.record:
                oob = _oob_penalty_for_plan(sm, state, plans[chosen_idx])
                mi = state.my_id
                my_ships = int(state.total_ships(mi))
                enemy_ships = int(
                    sum(state.total_ships(e) for e in state.en_ids)
                )
                self.transitions.append(
                    Transition(
                        obs_feat=all_feats[chosen_idx],
                        plan_feats=all_feats,
                        chosen_idx=chosen_idx,
                        log_prob=float(log_prob),
                        value_pred=float(value_pred),
                        plan_score_net=float(plan_logit),
                        step=state.step,
                        state_feat=state_features(state),
                        oob_penalty=float(oob),
                        my_ships=my_ships,
                        enemy_ships=enemy_ships,
                        n_planets=int(len(state.planets)),
                        n_my_planets=int(len(state.my_pl)),
                    )
                )
            return arbiter.moves
        except Exception:
            return []

    def _select_plan(self, plans, state):
        n = len(plans)
        sm = self._sm
        all_feats = np.stack([combined_features(p, state) for p in plans])

        if self._has_weights:
            with torch.no_grad():
                xk = (
                    torch.from_numpy(all_feats)
                    .to(self.device, dtype=torch.float32)
                    .unsqueeze(0)
                )
                v, scores_t = self.net.forward_plans(xk)
                net_scores = scores_t.squeeze(0).cpu().numpy()
                value_pred = float(v.item())

            if self.explore:
                probs = _softmax(net_scores / self.temperature)
                idx = int(np.random.choice(n, p=probs))
                log_prob = float(math.log(probs[idx] + 1e-9))
            else:
                idx = int(np.argmax(net_scores))
                log_prob = 0.0
            return idx, log_prob, value_pred, float(net_scores[idx]), all_feats

        sim_ph = state.phase()
        sim_steps_int = 8 if sim_ph != "late" else 10
        base = np.array(
            [
                p.score
                + sm.score_plan_actions(
                    state, p.actions, steps=sim_steps_int, tempo_floor=1
                )
                for p in plans
            ],
            dtype=np.float32,
        )

        if self.explore:
            probs = _softmax(base / self.temperature)
            idx = int(np.random.choice(n, p=probs))
            log_prob = float(math.log(probs[idx] + 1e-9))
        else:
            idx = int(np.argmax(base))
            log_prob = 0.0
        return idx, log_prob, 0.0, float(base[idx]), all_feats


def game_summary_from_agent(agent: RLAgentV21) -> Optional[dict]:
    if not agent.transitions:
        return None
    t = agent.transitions[-1]
    sf = t.state_feat
    return {
        "final_my_ship_ratio": float(sf[0]),
        "final_planet_ratio": float(sf[3]),
        "last_step": int(t.step),
        "n_transitions": len(agent.transitions),
        "final_my_ships": int(t.my_ships),
        "final_enemy_ships": int(t.enemy_ships),
        "final_n_planets": int(t.n_planets),
        "final_n_my_planets": int(t.n_my_planets),
    }
