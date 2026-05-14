"""Wire submission-specific hooks (target_score, regional_adj, arbiter variant, neural b64).

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
