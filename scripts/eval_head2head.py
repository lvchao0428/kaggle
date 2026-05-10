#!/usr/bin/env python3
"""Generic Orbit Wars head-to-head evaluator.

Pits two agent versions against each other (double-seated) or one agent
against the built-in `random` opponent.

Examples::

    # v11 vs v9, 20 seeds, double-seated (40 games total)
    python3.12 scripts/eval_head2head.py --a v11 --b v9 --seeds 0-19

    # v11 vs random, 10 seeds (10 games, A is player 0 only)
    python3.12 scripts/eval_head2head.py --a v11 --b random --seeds 0-9 --no-swap

    # Quick: explicit seed list, default A=v11 B=v9
    python3.12 scripts/eval_head2head.py --seeds 0 1 2 3 4

Each row of output shows the [a,b] (and [b,a]) reward arrays. Per Kaggle
convention 1=win, -1=loss, 0=draw.
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


def _load_agent(version: str):
    """Load `submission_<version>.py` and return its `agent` callable.
    Special string "random" returns the literal string (kaggle-environments
    treats it as a built-in opponent)."""
    if version == "random":
        return "random"
    path = ROOT / f"submission_{version}.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"submission_{version}_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"submission_{version}_eval"] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _seed_random_bot(sd: int) -> None:
    """Built-in `random` bot uses module-level random; reseed for fairness."""
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


def _parse_seeds(args):
    if args.seeds:
        out = []
        for tok in args.seeds:
            tok = tok.strip()
            if "-" in tok and not tok.startswith("-"):
                lo, hi = tok.split("-", 1)
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(tok))
        return out
    return list(range(args.seed_start, args.seed_end + 1))


def main():
    from kaggle_environments import evaluate

    p = argparse.ArgumentParser(description="Orbit Wars head-to-head evaluator")
    p.add_argument("--a", default="v11", help="version label for agent A (e.g. v11, v10, random)")
    p.add_argument("--b", default="v9", help="version label for agent B")
    p.add_argument("--seeds", nargs="*", default=None,
                   help="Seed list. Accepts '0 1 2 3' or '0-9' tokens.")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seed-end", type=int, default=9, help="Inclusive when --seeds omitted")
    p.add_argument("--no-swap", action="store_true",
                   help="Don't swap seats; A is always player 0")
    args = p.parse_args()

    seeds = _parse_seeds(args)

    a = _load_agent(args.a)
    b = _load_agent(args.b)

    a_wins = b_wins = ties = 0
    t0 = time.time()
    for sd in seeds:
        _seed_random_bot(sd)
        ab = evaluate(
            "orbit_wars",
            [a, b] if isinstance(a, str) or isinstance(b, str)
                   else [lambda o, c, f=a: f(o, c), lambda o, c, f=b: f(o, c)],
            configuration={"seed": int(sd)},
            num_episodes=1,
            debug=False,
        )[0]
        w_ab = _winner(ab)
        if w_ab == 0:
            a_wins += 1
        elif w_ab == 1:
            b_wins += 1
        else:
            ties += 1

        ba_str = ""
        if not args.no_swap:
            _seed_random_bot(sd + 9999991)
            ba = evaluate(
                "orbit_wars",
                [b, a] if isinstance(a, str) or isinstance(b, str)
                       else [lambda o, c, f=b: f(o, c), lambda o, c, f=a: f(o, c)],
                configuration={"seed": int(sd)},
                num_episodes=1,
                debug=False,
            )[0]
            w_ba = _winner(ba)
            if w_ba == 1:
                a_wins += 1
            elif w_ba == 0:
                b_wins += 1
            else:
                ties += 1
            ba_str = f"  [{args.b},{args.a}]={ba!r}"

        print(f"seed={sd:3d}  [{args.a},{args.b}]={ab!r}{ba_str}")

    n = a_wins + b_wins + ties
    pct = (100.0 * a_wins / n) if n else 0.0
    print()
    print(f"{args.a} wins={a_wins}  {args.b} wins={b_wins}  ties={ties}  "
          f"games={n}  {args.a} win%={pct:.1f}")
    print(f"elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
