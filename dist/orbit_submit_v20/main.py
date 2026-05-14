"""Orbit Wars v20 — canonical Kaggle entry.

Implementation lives in the ``orbit_submit/`` package (``GameState``, ``PlanArbiter``,
``target_score`` via ``orbit_submit.registry``, etc.). This module re-exports the
same public names older tools/tests used when v20 was a single file.
"""

from __future__ import annotations

# Registry side effects + agent (must import this submodule first).
from orbit_submit.agent import _GLOBAL_NEURAL, _GLOBAL_OPP, agent

from orbit_submit.constants import (
    ABS_MIN_BATCH,
    BOARD,
    DEFAULT_EPISODE_STEPS,
    DEFAULT_MAX_SHIP_SPEED,
    ENEMY_SHIP_PEN_COEFF,
    EXPAND_RANK_SCORE_FLOOR,
    HORIZON_TURNS,
    INNER_SUN_BELT_R,
    MAX_SOURCES_PER_TARGET,
    MAX_TARGETS_PER_PLAN,
    MAX_TOTAL_MOVES,
    NEUTRAL_BRUTE_SLACK_MIN,
    NEUTRAL_BRUTE_SLACK_MUL,
    ONE_OUTBOUND_DST_PER_SOURCE_UNTIL_N_WORLDS,
    OPENING_FIRST_CAPTURE_SEND,
    OPENING_SOLO_HQ_RESERVE_LAST_STEP,
    ORB_STRATEGY_PROFILE,
    ROTATION_RADIUS_LIMIT,
    SUN_PATH_MARGIN,
    SUN_RADIUS,
    SUN_X,
    SUN_Y,
    SYNC_ETA_WINDOW,
    SYNC_ETA_WINDOW_MAX,
    fleet_speed,
    point_segment_distance,
    segment_hits_sun,
    swept_pair_hit,
)
from orbit_submit.entities import Fleet, Planet, _combat
from orbit_submit.game_state import GameState
from orbit_submit.kinematics import (
    ENGINE_LAUNCH_PAD,
    capture_need,
    is_sun_belt_planet,
    launch_hits_target_first,
    launch_intercept_step,
    launch_origin,
    lead_intercept,
    my_inbound_ships_to,
    neutral_wave_wins,
    safe_aim,
    target_state_at,
)
from orbit_submit.engine import (
    AttackPlanner,
    CometEvacPlanner,
    DefensePlanner,
    DiplomacyEngine,
    ExpandPlanner,
    InterceptPlanner,
    LateDumpPlanner,
    MCTSEngine,
    OpponentModel,
    Plan,
    PlanArbiter,
    PragmaticActionUCB,
    RedistributionPlanner,
    UrgentHighProdPlanner,
    blended_paranoid_sim,
    capture_edge_score,
    clone_sim,
    copy_sim,
    eval_sim_planets,
    score_plan_actions,
    score_plan_actions_paranoid,
    sim_step,
    target_value_in_region,
)
from orbit_submit.neural import NeuralVal
from orbit_submit.policy import (
    PHASE_TABLE,
    PhasePolicy,
    _STRATEGY_PROFILE_DELTAS,
    _env_float,
    _merged_phase_row,
)
from orbit_submit.regional import (
    MultiHopPlanner,
    ProductionTimeline,
    Region,
    RegionalGraph,
    Wave,
    calculate_safe_surplus,
)
from orbit_submit.scoring_early import enemy_eta_power
from orbit_submit.scoring_shared import (
    approach_bonus,
    contest_penalty,
    elite_eval,
    orbit_arc_strategic_score,
    recapture_bonus,
)
from orbit_submit.snapshot import Snapshot
from orbit_submit.neural_weights_v20 import NEURAL_WEIGHTS_B64 as _NEURAL_WEIGHTS_B64
from orbit_submit.targeting import regional_capture_adjustment, target_score

__all__ = [
    "ORB_STRATEGY_PROFILE",
    "PHASE_TABLE",
    "PhasePolicy",
    "Plan",
    "PlanArbiter",
    "Snapshot",
    "GameState",
    "agent",
    "target_score",
    "regional_capture_adjustment",
    "capture_edge_score",
    "score_plan_actions",
    "score_plan_actions_paranoid",
    "blended_paranoid_sim",
    "_GLOBAL_OPP",
    "_GLOBAL_NEURAL",
    "target_score",
    "regional_capture_adjustment",
    "_NEURAL_WEIGHTS_B64",
]
