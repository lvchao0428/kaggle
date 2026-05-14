from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from orbit_submit.constants import ORB_STRATEGY_PROFILE, SUN_X, SUN_Y
from orbit_submit.game_state import GameState

# ╔═══ region 4: PhasePolicy ════════════════════════════════════════════════╗
#
# ── How to iterate parameters (offline → Kaggle) ───────────────────────────
#
# 1) Canonical knobs live only in PHASE_TABLE below (early / mid / late).
#    Typical groups:
#    - Economy vs aggression: reserve_growth_mul, cost_pen_mul,
#      cost_pen_neutral_mul, urgent_attack_ratio, mode_order.
#    - Search cost: mcts_* , pragmatic_mcts_* , neural_modifier_strength ,
#      sim_steps , tempo_floor.
#    - Commit gates (were magic numbers): region_pressure_ratio,
#      safe_surplus_ship_mult — used when regional_graph is active in
#      PlanArbiter.commit_best; baseline_commit_margin vs idle baseline.
#    - Pessimistic 1-enemy-rollout rerank: paranoid_score_budget_ms (early=0),
#      paranoid_plan_top_k , paranoid_steps , paranoid_blend — see
#      score_plan_actions_paranoid + PlanArbiter.score_with_modifiers .
#
# 2) Sweep without editing the table: export ORB_REGION_PRESSURE_RATIO ,
#    ORB_SAFE_SURPLUS_SHIP_MULT , ORB_BASELINE_COMMIT_MARGIN per process ;
#    _merged_phase_row() overlays them onto PHASE_TABLE reads. Useful with
#    tools/sweep_commit_gates.py wrapping scripts/eval_head2head.py .
#
# 3) Scripted eval: scripts/eval_head2head.py --a v20 --b v17 --seeds 0-19 .
#    Style overlays (same file, ContextVar-safe): --a v20@rush ,
#    v20@turtle — ORB_STRATEGY_PROFILE merges _STRATEGY_PROFILE_DELTAS .
#
# 4) Smoked pessimistic math (no episode): python3 tools/paranoid_score.py
#
# 5) RL self-play opponent mix (tools/rollout_worker.py): --opponent-mix
#    "self:0.5,v13:0.3,v20@expand:0.2" — weights normalized; tokens are load
#    paths or ``self``.
#
# 6) Acceptance: prefer 20+ double-seeded games; watch step wall-time (arbiter
#    skips paranoid when over paranoid_score_budget_ms).
#
# Single source of truth for phase-dependent tuning. Adjusting strategy = edit
# one row (or overlay env/profile as above).

PHASE_TABLE: Dict[str, Dict[str, object]] = {
    "early": dict(
        reserve_growth_mul=1,
        cost_pen_mul=0.75,
        cost_pen_neutral_mul=0.60,
        urgent_attack_ratio=2.65,
        urgent_attack_min_prod=3,
        mode_order=["expand", "aggro", "counter", "balanced", "comet"],
        mcts_budget_ms=0,
        mcts_max_iters=0,
        pragmatic_mcts_budget_ms=0,
        pragmatic_mcts_max_iters=0,
        pragmatic_mcts_top_k=6,
        pragmatic_mcts_rollout_steps=12,
        neural_modifier_strength=0.10,
        recapture_mul=1.2,
        approach_weight=1.8,
        sim_steps=8,
        tempo_floor=1,
        region_pressure_ratio=0.72,
        safe_surplus_ship_mult=1.55,
        baseline_commit_margin=0.10,
        paranoid_score_budget_ms=0,
        paranoid_plan_top_k=4,
        paranoid_steps=8,
        paranoid_blend=0.0,
    ),
    "mid": dict(
        reserve_growth_mul=3,
        cost_pen_mul=0.80,
        cost_pen_neutral_mul=0.74,
        urgent_attack_ratio=1.05,
        urgent_attack_min_prod=4,
        mode_order=["aggro", "expand", "counter", "balanced", "comet"],
        mcts_budget_ms=120,
        mcts_max_iters=50,
        pragmatic_mcts_budget_ms=55,
        pragmatic_mcts_max_iters=28,
        pragmatic_mcts_top_k=6,
        pragmatic_mcts_rollout_steps=14,
        neural_modifier_strength=0.12,
        recapture_mul=1.10,
        approach_weight=1.55,
        sim_steps=8,
        tempo_floor=2,
        region_pressure_ratio=0.72,
        safe_surplus_ship_mult=1.55,
        baseline_commit_margin=0.10,
        paranoid_score_budget_ms=40,
        paranoid_plan_top_k=5,
        paranoid_steps=8,
        paranoid_blend=0.42,
    ),
    "late": dict(
        reserve_growth_mul=4,
        cost_pen_mul=0.78,
        cost_pen_neutral_mul=0.70,
        urgent_attack_ratio=0.90,
        urgent_attack_min_prod=3,
        mode_order=["aggro", "counter", "expand", "balanced", "comet"],
        mcts_budget_ms=200,
        mcts_max_iters=80,
        pragmatic_mcts_budget_ms=85,
        pragmatic_mcts_max_iters=36,
        pragmatic_mcts_top_k=8,
        pragmatic_mcts_rollout_steps=16,
        neural_modifier_strength=0.15,
        recapture_mul=1.18,
        approach_weight=1.40,
        sim_steps=10,
        tempo_floor=1,
        region_pressure_ratio=0.72,
        safe_surplus_ship_mult=1.55,
        baseline_commit_margin=0.10,
        paranoid_score_budget_ms=62,
        paranoid_plan_top_k=5,
        paranoid_steps=10,
        paranoid_blend=0.48,
    ),
}

# Optional per-profile overrides for local eval / rollouts via ORB_STRATEGY_PROFILE.
# Naming mirrors Planet-Wars-style archetypes (Java PW bots cannot run here);
# README lists PW ↔ profile mapping.
_STRATEGY_PROFILE_DELTAS: Dict[str, Dict[str, object]] = {
    "turtle": dict(
        reserve_growth_mul_delta=+2,
        urgent_attack_ratio_delta=+0.35,
        cost_pen_mul_delta=+0.06,
        cost_pen_neutral_mul_delta=+0.05,
    ),
    "rush": dict(
        reserve_growth_mul_delta=-1,
        urgent_attack_ratio_delta=-0.20,
        cost_pen_mul_delta=-0.06,
        mode_order_override=["aggro", "counter", "expand", "balanced", "comet"],
    ),
    "expand": dict(
        cost_pen_mul_delta=-0.10,
        cost_pen_neutral_mul_delta=-0.08,
        mode_order_override=["expand", "balanced", "aggro", "counter", "comet"],
    ),
    "greedy_prod": dict(
        urgent_attack_min_prod_delta=-1,
        recapture_mul_delta=+0.12,
        cost_pen_mul_delta=-0.04,
    ),
    "dual": dict(
        urgent_attack_ratio_delta=+0.12,
        mode_order_override=["expand", "aggro", "counter", "balanced", "comet"],
    ),
}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _merged_phase_row(ph: str) -> Dict[str, object]:
    """Merge PHASE_TABLE[ph] with profile deltas and optional env gate floats.

    When set in the environment, ORB_REGION_PRESSURE_RATIO,
    ORB_SAFE_SURPLUS_SHIP_MULT, ORB_BASELINE_COMMIT_MARGIN override the row.
    """
    row: Dict[str, object] = dict(PHASE_TABLE[ph])
    prof = ORB_STRATEGY_PROFILE.get(None)
    if isinstance(prof, str) and prof.strip():
        deltas = _STRATEGY_PROFILE_DELTAS.get(prof.strip().lower())
        if deltas:
            for k, dv in deltas.items():
                if k == "mode_order_override":
                    row["mode_order"] = list(dv)  # type: ignore[arg-type]
                    continue
                if k.endswith("_delta"):
                    base_key = k[:-6]
                    old = row.get(base_key)
                    if old is None:
                        continue
                    if isinstance(old, bool):
                        continue
                    if isinstance(old, int):
                        row[base_key] = int(old) + int(dv)  # type: ignore[arg-type]
                    elif isinstance(old, float):
                        row[base_key] = float(old) + float(dv)  # type: ignore[arg-type]
            rgm = int(row.get("reserve_growth_mul", 2))
            row["reserve_growth_mul"] = max(1, min(8, rgm))
    row["region_pressure_ratio"] = _env_float(
        "ORB_REGION_PRESSURE_RATIO", float(row.get("region_pressure_ratio", 0.72))
    )
    row["safe_surplus_ship_mult"] = _env_float(
        "ORB_SAFE_SURPLUS_SHIP_MULT", float(row.get("safe_surplus_ship_mult", 1.55))
    )
    row["baseline_commit_margin"] = _env_float(
        "ORB_BASELINE_COMMIT_MARGIN", float(row.get("baseline_commit_margin", 0.10))
    )
    return row


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
    pragmatic_mcts_budget_ms: float
    pragmatic_mcts_max_iters: int
    pragmatic_mcts_top_k: int
    pragmatic_mcts_rollout_steps: int
    neural_modifier_strength: float
    recapture_mul: float
    approach_weight: float
    sim_steps: int
    tempo_floor: int
    region_pressure_ratio: float
    safe_surplus_ship_mult: float
    baseline_commit_margin: float
    paranoid_score_budget_ms: float
    paranoid_plan_top_k: int
    paranoid_steps: int
    paranoid_blend: float

    @classmethod
    def for_state(cls, state: GameState) -> "PhasePolicy":
        ph = state.phase()
        row = _merged_phase_row(ph)
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
            pragmatic_mcts_budget_ms=float(row["pragmatic_mcts_budget_ms"]),
            pragmatic_mcts_max_iters=int(row["pragmatic_mcts_max_iters"]),
            pragmatic_mcts_top_k=int(row["pragmatic_mcts_top_k"]),
            pragmatic_mcts_rollout_steps=int(row["pragmatic_mcts_rollout_steps"]),
            neural_modifier_strength=float(row["neural_modifier_strength"]),
            recapture_mul=float(row["recapture_mul"]),
            approach_weight=float(row["approach_weight"]),
            sim_steps=int(row["sim_steps"]),
            tempo_floor=int(row["tempo_floor"]),
            region_pressure_ratio=float(row["region_pressure_ratio"]),
            safe_surplus_ship_mult=float(row["safe_surplus_ship_mult"]),
            baseline_commit_margin=float(row["baseline_commit_margin"]),
            paranoid_score_budget_ms=float(row["paranoid_score_budget_ms"]),
            paranoid_plan_top_k=int(row["paranoid_plan_top_k"]),
            paranoid_steps=int(row["paranoid_steps"]),
            paranoid_blend=float(row["paranoid_blend"]),
        )

