#!/usr/bin/env python3
"""Orbit Wars: v10 vs v9 双向座位评估（用于迭代调参）。

每 seed 两局：[v10, v9] 与 [v9, v10]，统计 v10 净胜场。

示例::

    python3.12 scripts/eval_v10_vs_v9.py --seeds 0 1 2 3 4
    python3.12 scripts/eval_v10_vs_v9.py --seed-start 0 --seed-end 19

依赖: pip install "kaggle-environments>=1.28.0"（建议 Python 3.11+）
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission_resolve import resolve_submission_path


def _load_agent(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _seed_random_bot(sd: int) -> None:
    random.seed((410224286193 + int(sd) * 2246822519) % (2**63))


def _winner(row):
    if not row or len(row) < 2:
        return None
    r0, r1 = row[0], row[1]
    if r0 is None or r1 is None:
        return None
    if r0 > r1:
        return 0
    if r1 > r0:
        return 1
    return None


def main():
    from kaggle_environments import evaluate

    p = argparse.ArgumentParser(description="v10 vs v9 head-to-head (swapped seats)")
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="Explicit seed list")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seed-end", type=int, default=9, help="Inclusive end if --seeds omitted")
    args = p.parse_args()

    if args.seeds:
        seeds = list(args.seeds)
    else:
        seeds = list(range(args.seed_start, args.seed_end + 1))

    v9_path = resolve_submission_path(ROOT, "v9")
    v10_path = resolve_submission_path(ROOT, "v10")
    v9 = _load_agent("submission_v9_eval", v9_path)
    v10 = _load_agent("submission_v10_eval", v10_path)

    wins = losses = ties = 0
    t0 = time.time()
    for sd in seeds:
        _seed_random_bot(sd)
        ab = evaluate(
            "orbit_wars",
            [lambda o, c, f=v10: f(o, c), lambda o, c, f=v9: f(o, c)],
            configuration={"seed": int(sd)},
            num_episodes=1,
            debug=False,
        )[0]
        ba = evaluate(
            "orbit_wars",
            [lambda o, c, f=v9: f(o, c), lambda o, c, f=v10: f(o, c)],
            configuration={"seed": int(sd)},
            num_episodes=1,
            debug=False,
        )[0]
        w1, w2 = _winner(ab), _winner(ba)
        if w1 == 0:
            wins += 1
        elif w1 == 1:
            losses += 1
        else:
            ties += 1
        if w2 == 1:
            wins += 1
        elif w2 == 0:
            losses += 1
        else:
            ties += 1
        print(f"seed={sd:3d}  [v10,v9]={ab!r}  [v9,v10]={ba!r}")

    n = wins + losses + ties
    pct = (100.0 * wins / n) if n else 0.0
    print()
    print(f"v10 wins={wins}  v9 wins={losses}  ties={ties}  games={n}  v10 win%={pct:.1f}")
    print(f"elapsed {time.time() - t0:.1f}s")

    if pct >= 80.0 and losses <= wins // 4:
        print("target met (>=80% v10 wins).")
        return 0
    if pct >= 100.0 and losses == 0:
        print("perfect sweep.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
