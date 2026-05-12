"""
Kaggle submission helper: re-exports `agent` from submission_v20.py.

Usage:
  - Submit `submission_v20.py` directly as your agent file, **or**
  - Copy this file to `main.py` at bundle root and ensure `submission_v20.py` is alongside it.

The competition expects `main.py` with a callable `agent(obs, config=None)`.
"""

from submission_v20 import agent

__all__ = ["agent"]
