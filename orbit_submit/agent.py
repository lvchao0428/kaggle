"""Kaggle entry for v20: configure ``registry`` hooks then expose ``agent``."""

from __future__ import annotations

import time

import orbit_submit.registry as registry
from orbit_submit import targeting
from orbit_submit.neural_weights_v20 import NEURAL_WEIGHTS_B64

registry.target_score = targeting.target_score
registry.regional_capture_adjustment = targeting.regional_capture_adjustment
registry.neural_weights_b64 = NEURAL_WEIGHTS_B64
registry.arbiter_variant = "v20"

from orbit_submit.engine import (  # noqa: E402
    DiplomacyEngine,
    OpponentModel,
    PlanArbiter,
)
from orbit_submit.game_state import GameState  # noqa: E402
from orbit_submit.neural import NeuralVal  # noqa: E402
from orbit_submit.policy import PhasePolicy  # noqa: E402
from orbit_submit.regional import MultiHopPlanner, ProductionTimeline, RegionalGraph as _RG  # noqa: E402
from orbit_submit.snapshot import Snapshot  # noqa: E402

_GLOBAL_OPP = OpponentModel()
_GLOBAL_NEURAL = NeuralVal()


def agent(obs, config=None):
    """Kaggle-required entry. Returns list of [src_id, angle, ships] moves."""
    global _GLOBAL_OPP, _GLOBAL_NEURAL
    t0 = time.time()
    elapsed = lambda: (time.time() - t0) * 1000.0

    try:
        state = GameState(obs, config, ruleset="v20")
        if not state.my_pl:
            return []

        _GLOBAL_OPP.update(state)
        policy = PhasePolicy.for_state(state)
        snap = Snapshot.build(state, policy)
        diplo = DiplomacyEngine(state, _GLOBAL_OPP)

        regional_graph = None
        multi_hop_planner = None
        try:
            spawn_positions = config.get("spawn_positions", []) if config else []
            regional_graph = _RG(state.planets, spawn_positions)
            timeline = ProductionTimeline(state.planets, set(p.id for p in state.my_pl))
            multi_hop_planner = MultiHopPlanner(regional_graph, timeline)
        except Exception:
            regional_graph = None
            multi_hop_planner = None

        arbiter = PlanArbiter(
            snap,
            diplo,
            _GLOBAL_NEURAL,
            elapsed_ms_fn=elapsed,
            deadline_ms=920.0,
            regional_graph=regional_graph,
            multi_hop_planner=multi_hop_planner,
        )

        arbiter.commit_urgent()
        plans = arbiter.collect_strategic()
        scored = arbiter.score_with_modifiers(plans)
        arbiter.commit_best(scored)
        arbiter.commit_fallback()

        return arbiter.moves
    except Exception:
        return []
