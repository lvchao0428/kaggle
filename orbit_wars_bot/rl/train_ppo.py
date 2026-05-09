"""Train a small PPO policy on macro actions (requires requirements-rl.txt)."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from orbit_wars_bot.rl.env import OrbitWarsMacroEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="models/orbit_wars_macro_ppo")
    p.add_argument("--opponent", type=str, default="random")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    env = OrbitWarsMacroEnv(opponent=args.opponent, seed=args.seed)
    ckpt = CheckpointCallback(save_freq=max(args.timesteps // 4, 256), save_path=str(out.parent), name_prefix=out.name)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=args.timesteps, callback=ckpt)
    model.save(str(out))
    print("Saved", out)


if __name__ == "__main__":
    main()
