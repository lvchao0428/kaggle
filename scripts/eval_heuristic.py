#!/usr/bin/env python3
"""Evaluate heuristic agents vs random on fixed seeds (needs kaggle-environments)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from kaggle_environments import evaluate

    from orbit_wars_bot.heuristic import v6_wrapped

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes-per-seed", type=int, default=1)
    p.add_argument("--opponent", type=str, default="random")
    args = p.parse_args()

    def ag(obs):
        return v6_wrapped.agent(obs, None)

    wins = 0
    total = 0
    for sd in args.seeds:
        cfg = {"seed": int(sd)}
        rewards = evaluate(
            "orbit_wars",
            [ag, args.opponent],
            configuration=cfg,
            num_episodes=args.episodes_per_seed,
            debug=False,
        )
        for ep in rewards:
            total += 1
            if ep[0] is not None and ep[0] > ep[1]:
                wins += 1
            print(f"seed={sd} rewards={ep}")

    print(f"Heuristic wins {wins}/{total} episodes (player 0 must be higher reward).")


if __name__ == "__main__":
    main()
