"""Resolve `submission_{version}.py` from active tree or archive (legacy backups)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def resolve_submission_path(root: Path, version: str) -> Path:
    """Return path to ``submission_v{version}.py`` (``version`` like ``v19`` or ``11``)."""
    v = version if version.startswith("v") else f"v{version}"
    fname = f"submission_{v}.py"
    p = root / fname
    if p.is_file():
        return p
    archived = root / "archive" / "legacy" / "submissions" / fname
    if archived.is_file():
        return archived
    raise FileNotFoundError(f"Missing {fname} under {root} or archive/legacy/submissions/")


def load_submission_module(root: Path, version: str, module_tag: str) -> ModuleType:
    """Load submission module via importlib; ``module_tag`` keeps sys.modules keys unique."""
    path = resolve_submission_path(root, version)
    v = version if version.startswith("v") else f"v{version}"
    mod_name = f"submission_{v}_{module_tag}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod
