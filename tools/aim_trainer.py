"""Offline aim-accuracy evaluator and trainer for Orbit Wars.

Runs N games with a given bot version, inspects every fleet launched each turn,
and records whether it hits the sun or flies out of bounds. Outputs aggregate
statistics and per-game details so you can measure how well safe_aim works.

Usage::

    python3.12 tools/aim_trainer.py --version v18 --games 20 --seeds 0-19

    # Compare two versions:
    python3.12 tools/aim_trainer.py --version v17 --games 10 --seeds 0-9
    python3.12 tools/aim_trainer.py --version v18 --games 10 --seeds 0-9

Trajectory checks (when the loaded submission exposes ``swept_pair_hit`` and
``GameState.planet_motion_segment``) mirror ``orbit_wars.py``: swept segment
collision vs moving planets, ``launch_origin`` at ``radius + 0.1``, up to 220
steps per fleet.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for part in raw.replace(",", " ").split():
        if "-" in part and not part.startswith("-"):
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


from submission_resolve import resolve_submission_path


def load_submission(version: str):
    """Load submission module (GameState, geometry, agent, …)."""
    path = resolve_submission_path(ROOT, version)
    spec = importlib.util.spec_from_file_location(f"submission_{version}_aim", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"submission_{version}_aim"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_agent(version: str):
    return load_submission(version).agent


def trajectory_oob_before_planet(
    mod,
    state,
    lx: float,
    ly: float,
    angle: float,
    ships: int,
    max_steps: int = 220,
) -> Tuple[bool, bool]:
    """OOB or sun before any planet (engine order: planets, bounds, sun)."""
    spd = mod.fleet_speed(int(ships), state.max_speed)
    cx, cy = lx, ly
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    if hasattr(state, "planet_motion_segment") and hasattr(mod, "swept_pair_hit"):
        for k in range(1, max_steps + 1):
            fx0, fy0 = cx, cy
            fx1, fy1 = cx + cos_a * spd, cy + sin_a * spd
            for p in state.planets:
                p0, p1 = state.planet_motion_segment(p, k)
                if p0[0] < 0:
                    continue
                if mod.swept_pair_hit(
                    fx0, fy0, fx1, fy1,
                    p0[0], p0[1], p1[0], p1[1], p.radius,
                ):
                    return False, False
            if not (0.0 <= fx1 <= mod.BOARD and 0.0 <= fy1 <= mod.BOARD):
                return True, False
            if mod.point_segment_distance(
                mod.SUN_X, mod.SUN_Y, fx0, fy0, fx1, fy1
            ) < mod.SUN_RADIUS:
                return False, True
            cx, cy = fx1, fy1
        return False, False

    for t in range(1, max_steps + 1):
        nx = cx + cos_a * spd
        ny = cy + sin_a * spd
        if nx < 0 or nx > mod.BOARD or ny < 0 or ny > mod.BOARD:
            return True, False
        if mod.point_segment_distance(
            mod.SUN_X, mod.SUN_Y, cx, cy, nx, ny
        ) < mod.SUN_RADIUS:
            return False, True
        for p in state.planets:
            px, py = state.planet_pos_at(p, t)
            if mod.point_segment_distance(px, py, cx, cy, nx, ny) < p.radius:
                return False, False
        cx, cy = nx, ny
    return False, False


def run_one_game(agent_fn, seed: int, mod) -> Dict:
    """Run one game and check every fleet launched for safety."""
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset()

    total_fleets = 0
    oob_fleets = 0
    sun_hit_fleets = 0

    steps_run = 0
    while not env.done:
        obs_list = env.state
        if obs_list is None:
            break

        # Get agent 0's observation
        try:
            player_obs = obs_list[0]["observation"]
        except (KeyError, IndexError, TypeError):
            break

        # Run agent
        try:
            actions = agent_fn(player_obs, env.configuration)
        except Exception:
            actions = []

        if actions:
            try:
                state = mod.GameState(player_obs, env.configuration)
            except Exception:
                state = None
            pm = {p.id: p for p in state.planets} if state is not None else {}

            for move in actions:
                if len(move) >= 3:
                    src_id = int(move[0])
                    angle = float(move[1])
                    ships = int(move[2])
                    total_fleets += 1
                    if state is None or src_id not in pm:
                        continue
                    src_p = pm[src_id]
                    if hasattr(mod, "launch_origin"):
                        lx, ly = mod.launch_origin(src_p, angle)
                    else:
                        r = float(src_p.radius) + 0.1
                        lx = src_p.x + math.cos(angle) * r
                        ly = src_p.y + math.sin(angle) * r
                    oob, sun_hit = trajectory_oob_before_planet(
                        mod, state, lx, ly, angle, ships)
                    if oob:
                        oob_fleets += 1
                    if sun_hit:
                        sun_hit_fleets += 1

        # Step with actions for player 0, random for player 1
        env.step([actions, None])
        steps_run += 1

    # Determine winner
    rewards = [0, 0]
    try:
        for i in range(2):
            st = env.state[i]
            r = st.get("reward", 0) if isinstance(st, dict) else 0
            if r is not None:
                rewards[i] = r
    except Exception:
        pass

    return {
        "seed": seed,
        "steps": steps_run,
        "total_fleets": total_fleets,
        "oob_fleets": oob_fleets,
        "sun_hit_fleets": sun_hit_fleets,
        "oob_rate": oob_fleets / max(1, total_fleets),
        "sun_rate": sun_hit_fleets / max(1, total_fleets),
        "reward": rewards[0],
    }


def main():
    parser = argparse.ArgumentParser(description="Aim accuracy evaluator")
    parser.add_argument("--version", default="v18", help="Bot version to test")
    parser.add_argument("--games", type=int, default=10, help="Number of games")
    parser.add_argument("--seeds", default="0-9", help="Seed range (e.g. 0-9)")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)[:args.games]
    mod = load_submission(args.version)
    agent_fn = mod.agent

    print(f"Aim accuracy test: {args.version}  games={len(seeds)}")
    print("-" * 70)

    totals = {"fleets": 0, "oob": 0, "sun": 0, "games": 0}
    t0 = time.time()

    for seed in seeds:
        result = run_one_game(agent_fn, seed, mod)
        totals["fleets"] += result["total_fleets"]
        totals["oob"] += result["oob_fleets"]
        totals["sun"] += result["sun_hit_fleets"]
        totals["games"] += 1
        oob_pct = result["oob_rate"] * 100
        sun_pct = result["sun_rate"] * 100
        tag = ""
        if result["oob_fleets"] > 0:
            tag += " [OOB!]"
        if result["sun_hit_fleets"] > 0:
            tag += " [SUN!]"
        print(f"  seed={seed:3d}  steps={result['steps']:3d}  "
              f"fleets={result['total_fleets']:3d}  "
              f"oob={result['oob_fleets']:2d}({oob_pct:4.1f}%)  "
              f"sun={result['sun_hit_fleets']:2d}({sun_pct:4.1f}%)  "
              f"reward={result['reward']}{tag}")

    elapsed = time.time() - t0
    n = max(1, totals["fleets"])
    print("-" * 70)
    print(f"TOTAL  games={totals['games']}  fleets={totals['fleets']}  "
          f"oob={totals['oob']}({totals['oob']/n*100:.1f}%)  "
          f"sun={totals['sun']}({totals['sun']/n*100:.1f}%)  "
          f"elapsed={elapsed:.1f}s")

    if totals["oob"] == 0 and totals["sun"] == 0:
        print("\nPERFECT AIM: No OOB or sun-hit fleets detected!")
    else:
        print(f"\nWARNING: {totals['oob']} OOB + {totals['sun']} sun-hit "
              f"fleets out of {totals['fleets']} total.")


if __name__ == "__main__":
    main()
