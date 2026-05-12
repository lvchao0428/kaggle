#!/usr/bin/env python3
"""Offline pessimistic rollout scoring (submission_v19 paranoid helpers).

Loads [submission_v19](submission_v19.py) Sim stack and exposes
``score_plan_actions`` / ``score_plan_actions_paranoid`` / ``blended_paranoid_sim``.
Use for diagnosing plan deltas without running full Kaggle episodes.

Examples::

    python3.12 tools/paranoid_score.py --check-import
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Paranoid sim smoke / import check")
    p.add_argument("--check-import", action="store_true")
    args = p.parse_args()

    if not args.check_import:
        args.check_import = True

    import submission_v19 as s19  # noqa: WPS433

    assert hasattr(s19, "score_plan_actions_paranoid")
    assert hasattr(s19, "blended_paranoid_sim")

    gs = getattr(s19, "GameState")

    obs = {
        "player": 0,
        "step": 2,
        "angular_velocity": 0.015,
        "configuration": {"shipSpeed": 6.0, "episodeSteps": 500},
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": [
            [0, 0, 20.0, 50.0],
            [1, -1, 45.0, 50.0],
            [2, 1, 70.0, 52.0],
        ],
        "planets": [
            [0, 0, 20.0, 50.0, 0.5, 40, 3],
            [1, -1, 45.0, 50.0, 0.5, 5, 2],
            [2, 1, 70.0, 52.0, 0.5, 35, 2],
        ],
        "fleets": [],
    }
    state = gs(obs)

    baseline = s19.score_plan_actions(state, [(0, 1, 20)], steps=10, tempo_floor=1)
    b2, pessim = s19.score_plan_actions_paranoid(
        state, [(0, 1, 20)], steps=10, tempo_floor=1, par_steps=10)

    blended = s19.blended_paranoid_sim(
        state, [(0, 1, 20)],
        steps=10, tempo_floor=1, par_steps=10, blend=0.5)

    print("score_plan_actions (baseline-only):", round(baseline, 4))
    print("paranoid (baseline, pessim):", round(b2, 4), round(pessim, 4))
    print("blended 0.5:", round(blended, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
