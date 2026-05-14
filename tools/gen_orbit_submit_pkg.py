#!/usr/bin/env python3
"""One-shot generator: slice submission_v21_lite.py into orbit_submit/*.py.

Run from repo root: python3 tools/gen_orbit_submit_pkg.py
Requires manual follow-up patches: game_state ruleset, registry wiring, arbiter branching.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "submission_v21_lite.py"
OUT = ROOT / "orbit_submit"

# 1-based inclusive line spans from submission_v21_lite.py (see # region markers)
SLICES = {
    "constants.py": (34, 133),  # region 0 SUN constants … swept_pair_hit
    "regional.py": (135, 345),
    "entities.py": (348, 397),
    "game_state.py": (398, 635),  # class GameState ends before is_sun_belt
    "kinematics.py": (637, 909),  # is_sun_belt through capture_need
    "scoring_early.py": (1411, 1418),
    "snapshot.py": (912, 1103),
    "policy.py": (1105, 1367),
    "scoring_shared.py": (1371, 1435),
}

HEAD = '''from __future__ import annotations

'''

IMPORTS_STD = """import base64
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

"""

CONST_EXTRA = """
# Re-export for regional / tests
ORB_STRATEGY_PROFILE: ContextVar[Optional[str]] = ContextVar(
    "ORB_STRATEGY_PROFILE", default=None
)
"""


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "__init__.py").write_text(
        '"""Orbit Wars submission package (split from monolithic submission)."""\n',
        encoding="utf-8",
    )

    for name, (a, b) in SLICES.items():
        chunk = "".join(lines[a - 1 : b])
        path = OUT / name
        if name == "constants.py":
            body = IMPORTS_STD + CONST_EXTRA + chunk
        elif name == "regional.py":
            body = HEAD + "from orbit_submit.constants import *\n\n" + chunk
        elif name == "entities.py":
            body = HEAD + "from orbit_submit.constants import *\n\n" + chunk
        elif name == "game_state.py":
            body = (
                HEAD
                + "from orbit_submit.constants import *\n"
                + "from orbit_submit.entities import Fleet, Planet, _combat\n\n"
                + chunk
            )
        elif name == "kinematics.py":
            body = (
                HEAD
                + "from orbit_submit.constants import *\n"
                + "from orbit_submit.entities import Planet, _combat\n"
                + "from orbit_submit.game_state import GameState\n\n"
                + chunk
            )
        elif name == "scoring_early.py":
            body = (
                HEAD
                + "from orbit_submit.constants import *\n"
                + "from orbit_submit.entities import Planet\n"
                + "from orbit_submit.game_state import GameState\n"
                + "from orbit_submit.kinematics import lead_intercept\n\n"
                + chunk
            )
        elif name == "snapshot.py":
            body = (
                HEAD
                + "from orbit_submit.constants import *\n"
                + "from orbit_submit.entities import Planet\n"
                + "from orbit_submit.game_state import GameState\n"
                + "from orbit_submit.regional import RegionalGraph\n"
                + "from orbit_submit.scoring_early import enemy_eta_power\n\n"
                + chunk
            )
        elif name == "policy.py":
            body = (
                HEAD
                + "from orbit_submit.constants import ORB_STRATEGY_PROFILE, SUN_X, SUN_Y\n"
                + "from orbit_submit.game_state import GameState\n\n"
                + chunk
            )
        elif name == "scoring_shared.py":
            body = (
                HEAD
                + "from orbit_submit.constants import *\n"
                + "from orbit_submit.entities import Planet\n"
                + "from orbit_submit.game_state import GameState\n"
                + "from orbit_submit.snapshot import Snapshot\n\n"
                + chunk
            )
        elif name == "engine.py":
            body = HEAD + "## POPULATED BY PATCH — see manual engine header\n\n" + chunk
        else:
            body = HEAD + chunk
        path.write_text(body, encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")

    # registry stub
    (OUT / "registry.py").write_text(
        '''"""Wire submission-specific hooks (target_score, regional_adj, arbiter variant, neural b64).

Import order:
1. Submission defines target_score / regional_capture_adjustment
2. Submission sets orbit_submit.registry fields
3. Submission imports orbit_submit.engine
"""

from __future__ import annotations

from typing import Any, Callable, Optional

target_score: Optional[Callable[..., Any]] = None
regional_capture_adjustment: Optional[Callable[..., Any]] = None
neural_weights_b64: str = ""
# Which PlanArbiter.commit_best branch to run inside engine.plan_arbiter
arbiter_variant: str = "v21"
'''
    ,
        encoding="utf-8",
    )
    print("Wrote orbit_submit/registry.py")


if __name__ == "__main__":
    main()
