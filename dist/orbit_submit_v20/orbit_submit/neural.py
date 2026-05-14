from __future__ import annotations

import base64
import io
import math
from typing import TYPE_CHECKING

import numpy as np

from orbit_submit import registry as _reg
from orbit_submit.game_state import GameState
from orbit_submit.scoring_shared import elite_eval

if TYPE_CHECKING:
    pass


class NeuralVal:
    """Score modifier in [-strength, +strength]. NEVER overrides plans; just
    nudges the arbiter's ranking by `(1 + strength * predict)`."""

    N_FEAT = 14

    def __init__(self):
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.2, (64, self.N_FEAT)).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = rng.normal(0, 0.2, (32, 64)).astype(np.float32)
        self.b2 = np.zeros(32, dtype=np.float32)
        self.W3 = rng.normal(0, 0.2, (1, 32)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)
        self._try_load_inline()

    def _try_load_inline(self) -> None:
        wb = _reg.neural_weights_b64
        if not wb:
            return
        try:
            raw = base64.b64decode(wb)
            d = np.load(io.BytesIO(raw), allow_pickle=True).item()
            self.W1 = d["W1"]
            self.b1 = d["b1"]
            self.W2 = d["W2"]
            self.b2 = d["b2"]
            self.W3 = d["W3"]
            self.b3 = d["b3"]
        except Exception:
            pass

    def feat(self, state: GameState) -> np.ndarray:
        mi = state.my_id
        total = sum(state.total_ships(o) for o in [mi] + state.en_ids) + 1e-6
        my_sh = state.total_ships(mi) / total
        en_sh = sum(state.total_ships(e) for e in state.en_ids) / total
        my_prod = sum(p.production for p in state.my_pl)
        en_prod = sum(p.production for p in state.en_pl)
        prod_ratio = my_prod / max(my_prod + en_prod + 1, 1)
        n_planets = max(len(state.planets), 1)
        planet_ratio = len(state.my_pl) / n_planets
        cx, cy = state.centroid()
        min_en_dist = min(
            (math.hypot(p.x - cx, p.y - cy) for p in state.en_pl), default=100.0
        ) / 100.0
        phase_enc = {"early": 0.0, "mid": 0.5, "late": 1.0}[state.phase()]
        tl = state.turns_left() / max(state.episode_steps, 1)
        fronts = [p for p in state.my_pl if any(p.dist(e) < 35 for e in state.en_pl)]
        front_ratio = len(fronts) / max(len(state.my_pl), 1)
        comet_cnt = sum(
            1 for p in state.planets if p.is_comet and p.owner != mi
        ) / max(n_planets, 1)
        en_fleet = sum(f.ships for f in state.fleets if f.owner not in (-1, mi))
        en_fleet_ratio = en_fleet / max(state.total_ships(mi) + 1, 1)
        net_thr = sum(max(0, state.net_threat(p)) for p in state.my_pl)
        net_thr_ratio = net_thr / max(state.total_ships(mi) + 1, 1)
        ee = float(np.tanh(elite_eval(state) / 500.0))
        border = sum(
            (35 - m.dist(e)) / 35 * m.production
            for m in state.my_pl
            for e in state.en_pl
            if m.dist(e) < 35
        )
        border_norm = float(np.tanh(border / 100.0))
        return np.array(
            [
                my_sh,
                en_sh,
                prod_ratio,
                planet_ratio,
                min_en_dist,
                phase_enc,
                tl,
                front_ratio,
                comet_cnt,
                en_fleet_ratio,
                net_thr_ratio,
                ee,
                border_norm,
                float(len(state.en_ids) > 1),
            ],
            dtype=np.float32,
        )

    def predict(self, state: GameState) -> float:
        try:
            x = self.feat(state)
            h1 = np.maximum(0.0, self.W1 @ x + self.b1)
            h2 = np.maximum(0.0, self.W2 @ h1 + self.b2)
            return float(np.tanh(self.W3 @ h2 + self.b3)[0])
        except Exception:
            return 0.0

    def score_modifier(self, state: GameState, strength: float) -> float:
        """Returns multiplicative factor in [1-strength, 1+strength]."""
        return 1.0 + strength * self.predict(state)
