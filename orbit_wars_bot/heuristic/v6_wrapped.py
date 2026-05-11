"""Delegate to archived or active submission_v6.agent (stable import path)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, List, Optional

_SUB_V6: Optional[ModuleType] = None


def _load_submission_v6() -> ModuleType:
    global _SUB_V6
    if _SUB_V6 is not None:
        return _SUB_V6
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from submission_resolve import resolve_submission_path

    path = resolve_submission_path(root, "v6")
    spec = importlib.util.spec_from_file_location("submission_v6", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load submission_v6 from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _SUB_V6 = mod
    return mod


def agent(obs: Any, config: Any = None) -> List:
    """Same contract as submission_v6.agent."""
    return _load_submission_v6().agent(obs, config)
