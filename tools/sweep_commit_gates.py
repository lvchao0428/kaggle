#!/usr/bin/env python3
"""Grid small commit-gate knobs via env (ORB_*) against a fixed opponent.

Runs ``scripts/eval_head2head.py`` in subprocesses so each combo gets a fresh env.

Examples::

    python3.12 tools/sweep_commit_gates.py \\
        --a v19 --b v13 --seeds 0-4 --combos \\
        "ORB_REGION_PRESSURE_RATIO=0.68,ORB_SAFE_SURPLUS_SHIP_MULT=1.55"\\
        "ORB_REGION_PRESSURE_RATIO=0.72,ORB_BASELINE_COMMIT_MARGIN=0.12"

Each combo token is comma-separated KEY=VALUE environment assignments.
Unset keys fall back to PHASE_TABLE defaults unless still set in your shell.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _parse_combo(s: str) -> dict:
    out: dict = {}
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        k, eq, v = part.partition("=")
        if not eq:
            raise ValueError(f"bad kv {part}")
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep ORB_* gates via subprocess eval")
    ap.add_argument("--a", default="v19")
    ap.add_argument("--b", default="v13")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--no-swap", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--combos",
        nargs="+",
        required=True,
        help='Each combo: KEY=VAL,KEY=VAL e.g. ORB_REGION_PRESSURE_RATIO=0.7')
    args = ap.parse_args()

    ev = ROOT / "scripts" / "eval_head2head.py"
    if not ev.is_file():
        print("missing scripts/eval_head2head.py", file=sys.stderr)
        return 1

    baseline_env = dict(os.environ)

    for ci, combo_raw in enumerate(args.combos):
        patch = _parse_combo(combo_raw)
        env = dict(baseline_env)
        env.update(patch)
        cmd = [
            args.python,
            str(ev),
            "--a",
            args.a,
            "--b",
            args.b,
            "--seeds",
            args.seeds,
        ]
        if args.no_swap:
            cmd.append("--no-swap")

        label = "; ".join(f"{k}={v}" for k, v in sorted(patch.items()))
        print(f"\n=== combo {ci + 1}/{len(args.combos)} :: {label} ===")
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
