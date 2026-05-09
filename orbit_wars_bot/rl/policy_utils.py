"""Load a trained PPO and build env-step actions (numpy vec)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_ppo(path: Path, env):
    from stable_baselines3 import PPO

    return PPO.load(str(path), env=env)


def predict_action(model, obs_vec: np.ndarray, deterministic: bool = True) -> np.ndarray:
    act, _ = model.predict(obs_vec, deterministic=deterministic)
    return np.clip(np.asarray(act, dtype=np.float32), -1.0, 1.0)


__all__ = ["load_ppo", "predict_action"]
