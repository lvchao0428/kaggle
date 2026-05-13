"""CPU layout generation from per-env seeds -> planet tensors on device."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from vec_orbit.geom_torch import BOARD, SUN_X, SUN_Y, SUN_RADIUS

# Owner encoding
OWNER_INVALID = -1
OWNER_NEUTRAL = 0
PLAYER0 = 1
PLAYER1 = 2


def _min_sep_ok(px: float, py: float, xs: np.ndarray, ys: np.ndarray, n: int, sep: float) -> bool:
    if n == 0:
        return True
    d2 = (xs[:n] - px) ** 2 + (ys[:n] - py) ** 2
    return bool(np.min(d2) >= sep * sep)


def _not_in_sun(px: float, py: float, margin: float = SUN_RADIUS + 2.0) -> bool:
    dx = px - SUN_X
    dy = py - SUN_Y
    return dx * dx + dy * dy > margin * margin


def sample_planets_cpu(
    *,
    batch: int,
    max_planets: int,
    seeds: np.ndarray,
    min_sep: float = 6.0,
    max_attempts: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns numpy arrays (B,P) float64/float32/int32:
      px, py, owner (OWNER_*), ships (int), growth (int), valid mask (bool)
    """
    assert seeds.shape == (batch,)
    px = np.zeros((batch, max_planets), dtype=np.float32)
    py = np.zeros((batch, max_planets), dtype=np.float32)
    owner = np.full((batch, max_planets), OWNER_INVALID, dtype=np.int32)
    ships = np.zeros((batch, max_planets), dtype=np.int32)
    growth = np.zeros((batch, max_planets), dtype=np.int32)
    valid = np.zeros((batch, max_planets), dtype=np.bool_)

    for b in range(batch):
        rng = np.random.default_rng(int(seeds[b]) & 0xFFFFFFFF)
        n = int(rng.integers(max(4, max_planets // 2), max_planets + 1))
        n = min(n, max_planets)
        xs = np.zeros((max_planets,), dtype=np.float32)
        ys = np.zeros((max_planets,), dtype=np.float32)
        placed = 0
        for _ in range(max_attempts * n + 100):
            if placed >= n:
                break
            x = float(rng.uniform(2.0, BOARD - 2.0))
            y = float(rng.uniform(2.0, BOARD - 2.0))
            if not _not_in_sun(x, y):
                continue
            if not _min_sep_ok(x, y, xs, ys, placed, min_sep):
                continue
            xs[placed] = x
            ys[placed] = y
            placed += 1
        # fallback: grid jitter if map underfull
        gi = 0
        while placed < n and gi < n * 10:
            gx = 5.0 + (gi % 9) * 10.0 + float(rng.uniform(-1, 1))
            gy = 5.0 + (gi // 9) % 9 * 10.0 + float(rng.uniform(-1, 1))
            gi += 1
            gx = float(np.clip(gx, 2.0, BOARD - 2.0))
            gy = float(np.clip(gy, 2.0, BOARD - 2.0))
            if not _not_in_sun(gx, gy):
                continue
            if not _min_sep_ok(gx, gy, xs, ys, placed, min_sep * 0.7):
                continue
            xs[placed] = gx
            ys[placed] = gy
            placed += 1

        for i in range(placed):
            px[b, i] = xs[i]
            py[b, i] = ys[i]
            growth[b, i] = int(rng.integers(1, 4))
            valid[b, i] = True

        # Split first planets between players; rest neutral
        if placed >= 2:
            i0 = int(rng.integers(0, placed))
            i1 = int(rng.integers(0, placed - 1))
            if i1 >= i0:
                i1 += 1
            owner[b, i0] = PLAYER0
            owner[b, i1] = PLAYER1
            ships[b, i0] = int(rng.integers(20, 80))
            ships[b, i1] = int(rng.integers(20, 80))
        for i in range(placed):
            if owner[b, i] == OWNER_INVALID:
                owner[b, i] = OWNER_NEUTRAL
                ships[b, i] = int(rng.integers(10, 50))

    return px, py, owner, ships, growth, valid


def layouts_to_torch(
    px: np.ndarray,
    py: np.ndarray,
    owner: np.ndarray,
    ships: np.ndarray,
    growth: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    def t(x):
        return torch.from_numpy(np.asarray(x)).to(device=device)

    return (
        t(px),
        t(py),
        t(owner).long(),
        t(ships).float(),
        t(growth).long(),
        t(valid),
    )
