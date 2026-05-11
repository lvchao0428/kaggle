"""
Kaggle submission helper: re-exports `agent` from submission_v19.py.

Usage:
  - Submit `submission_v19.py` directly as your agent file, **or**
  - Copy this file to `main.py` at bundle root and ensure `submission_v19.py` is alongside it.

The competition expects `main.py` with a callable `agent(obs, config=None)`.
"""

from submission_v19 import agent

__all__ = ["agent"]
