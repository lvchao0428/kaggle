"""Gymnasium Env around kaggle_environments `orbit_wars` train() API + macro-v6."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from orbit_wars_bot.heuristic.macro import apply_macro_to_moves
from orbit_wars_bot.heuristic import v6_wrapped
from orbit_wars_bot.rl.featurize import OBS_DIM, featurize


class OrbitWarsMacroEnv(gym.Env):
    """
    One trained slot vs string opponent (e.g. ``random``).
    Action: Box(3,) in [-1, 1] rescaled inside ``apply_macro_to_moves`` on top of v6.
    Observation: fixed tensor from ``featurize``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: str = "random",
        configuration: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.opponent = opponent
        self.base_configuration = dict(configuration or {})
        self._seed = seed
        self._env = None
        self._trainer = None
        self._raw_obs: Any = None
        self._episode_config: Dict[str, Any] = {}

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _make_trainer(self, seed: Optional[int]):
        from kaggle_environments import make

        cfg = dict(self.base_configuration)
        if seed is not None:
            cfg["seed"] = int(seed)
        self._episode_config = cfg
        self._env = make("orbit_wars", configuration=cfg, debug=False)
        return self._env.train([None, self.opponent])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        use_seed = seed if seed is not None else self._seed
        if use_seed is not None:
            use_seed = int(use_seed)
        self._trainer = self._make_trainer(use_seed)
        self._raw_obs = self._trainer.reset()
        return featurize(self._raw_obs), {"raw_obs": self._raw_obs}

    def step(self, action: np.ndarray):
        moves_base = v6_wrapped.agent(self._raw_obs, self._episode_config)
        scaled = apply_macro_to_moves(moves_base, np.asarray(action, dtype=np.float32))
        obs, reward, done, info = self._trainer.step(scaled)
        self._raw_obs = obs
        inf = info if isinstance(info, dict) else {}
        return featurize(obs), float(reward), bool(done), False, inf

    def close(self):
        self._trainer = None
        self._env = None


__all__ = ["OrbitWarsMacroEnv"]
