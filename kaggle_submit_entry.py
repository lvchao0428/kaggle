"""
Kaggle submission helper: re-exports the same `agent` as submission_v6.py.

Usage:
  - Submit `submission_v6.py` directly as your agent file, **or**
  - Copy this file to `main.py` at bundle root and ensure `submission_v6.py` is alongside it.

The competition expects `main.py` with a callable `agent(obs, config=None)`.
"""

from submission_v6 import agent

__all__ = ["agent"]
