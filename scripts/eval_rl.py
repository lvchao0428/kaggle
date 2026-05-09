#!/usr/bin/env python3
"""Compare macro-wrapped policy vs baseline v6 using kaggle evaluate (optional PPO model)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from orbit_wars_bot.heuristic import v6_wrapped
    from orbit_wars_bot.heuristic.macro import apply_macro_to_moves
    from orbit_wars_bot.rl.featurize import featurize

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--opponent", type=str, default="random")
    p.add_argument("--model", type=str, default=None, help="Path to trained PPO (stable-baselines3)")
    args = p.parse_args()

    model = None
    if args.model:
        from stable_baselines3 import PPO

        model = PPO.load(args.model)

    def baseline(obs):
        return v6_wrapped.agent(obs, None)

    class MacroAgent:
        def __call__(self, obs):
            vec = featurize(obs)
            if model is None:
                act = np.zeros(3, dtype=np.float32)
            else:
                act, _ = model.predict(vec, deterministic=True)
                act = np.clip(np.asarray(act, dtype=np.float32), -1.0, 1.0)
            return apply_macro_to_moves(baseline(obs), act)

    macro_agent = MacroAgent()

    from kaggle_environments import evaluate

    for label, agent_fn in [("v6", baseline), ("macro", macro_agent)]:
        print("---", label, "---")
        for sd in args.seeds:
            rew = evaluate(
                "orbit_wars",
                [agent_fn, args.opponent],
                configuration={"seed": int(sd)},
                num_episodes=1,
                debug=False,
            )
            print(f"  seed {sd}: {rew[0]}")


if __name__ == "__main__":
    main()
