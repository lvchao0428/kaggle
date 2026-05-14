"""Deprecated shim — use ``submission_v20.py`` (canonical v20 + ``orbit_submit/``).

This filename is historical only; it is **not** a separate bot version.
"""

from submission_v20 import agent

__all__ = ["agent"]
