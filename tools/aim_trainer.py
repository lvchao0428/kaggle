"""Offline aim-accuracy evaluator and trainer for Orbit Wars.

Runs N games with a given bot version, inspects every fleet launched each turn,
and records whether it hits the sun or flies out of bounds. Outputs aggregate
statistics and per-game details so you can measure how well safe_aim works.

Usage::

    python3.12 tools/aim_trainer.py --version v18 --games 20 --seeds 0-19

    # Compare two versions:
    python3.12 tools/aim_trainer.py --version v17 --games 10 --seeds 0-9
    python3.12 tools/aim_trainer.py --version v18 --games 10 --seeds 0-9

The script hooks into the game loop and checks every fleet's trajectory after
the agent produces moves. It does NOT modify the bot -- it's a diagnostic tool
that tells you the empirical OOB and sun-hit rates so you can tune safe_aim
parameters with data instead of guessing.
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

SUN_X = SUN_Y = 50.0
SUN_RADIUS = 10.0
BOARD = 100.0


def _parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for part in raw.replace(",", " ").split():
        if "-" in part and not part.startswith("-"):
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def point_segment_distance(px, py, ax, ay, bx, by) -> float:
    abx, aby = bx - ax, by - ay
    l2 = abx * abx + aby * aby
    if l2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / l2))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return math.hypot(px - proj_x, py - proj_y)


def fleet_speed(ships: int, max_speed: float = 6.0) -> float:
    if ships <= 1:
        return 1.0
    spd = 1.0 + (max_speed - 1.0) * (math.log(max(1, ships)) / math.log(1000)) ** 1.5
    return min(spd, max_speed)


def load_agent(version: str):
    path = ROOT / f"submission_{version}.py"
    spec = importlib.util.spec_from_file_location(f"submission_{version}_aim", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"submission_{version}_aim"] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def check_fleet_safety(src_x: float, src_y: float, angle: float,
                       ships: int, max_speed: float = 6.0,
                       all_planet_positions: List[Tuple[float, float]] = None,
                       ) -> Dict[str, bool]:
    """Check if a fleet's trajectory is safe.

    A fleet is 'OOB' if it exits the board BEFORE reaching any planet
    (within 2 units capture radius). Sun-hit if the flight segment
    passes within SUN_RADIUS of the sun center.
    """
    spd = fleet_speed(ships, max_speed)
    results = {"oob": False, "sun_hit": False}

    # Step along the trajectory and check for OOB/sun/planet arrival.
    for eta in range(1, 80):
        fx = src_x + math.cos(angle) * spd * eta
        fy = src_y + math.sin(angle) * spd * eta

        # Check if fleet reached any planet (capture radius ~2 units).
        if all_planet_positions:
            for px, py in all_planet_positions:
                if math.hypot(fx - px, fy - py) < 3.0:
                    return results  # arrived at planet, safe

        if fx < 0 or fx > BOARD or fy < 0 or fy > BOARD:
            results["oob"] = True
            return results

        # Sun collision along segment from src to current position.
        seg_dist = point_segment_distance(SUN_X, SUN_Y, src_x, src_y, fx, fy)
        if seg_dist < SUN_RADIUS:
            results["sun_hit"] = True
            return results

    return results


def run_one_game(agent_fn, seed: int) -> Dict:
    """Run one game and check every fleet launched for safety."""
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset()

    total_fleets = 0
    oob_fleets = 0
    sun_hit_fleets = 0
    max_speed = 6.0

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

        # Get max_speed from config
        try:
            config = env.configuration
            if hasattr(config, "max_ship_speed"):
                max_speed = config.max_ship_speed
        except Exception:
            pass

        # Run agent
        try:
            actions = agent_fn(player_obs, env.configuration)
        except Exception:
            actions = []

        if actions:
            # Find source planet positions.
            # Planet format: [id, owner, x, y, radius, ships, production]
            planets = {}
            all_positions = []
            raw_planets = player_obs.get("planets", [])
            for p in raw_planets:
                if isinstance(p, (list, tuple)) and len(p) >= 4:
                    planets[p[0]] = (p[2], p[3])
                    all_positions.append((p[2], p[3]))
                elif isinstance(p, dict):
                    planets[p.get("id", -1)] = (p.get("x", 50.0), p.get("y", 50.0))
                    all_positions.append((p.get("x", 50.0), p.get("y", 50.0)))

            for move in actions:
                if len(move) >= 3:
                    src_id = move[0]
                    angle = move[1]
                    ships = move[2]
                    sx, sy = planets.get(src_id, (50.0, 50.0))

                    result = check_fleet_safety(
                        sx, sy, angle, ships, max_speed, all_positions)
                    total_fleets += 1
                    if result["oob"]:
                        oob_fleets += 1
                    if result["sun_hit"]:
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
    agent_fn = load_agent(args.version)

    print(f"Aim accuracy test: {args.version}  games={len(seeds)}")
    print("-" * 70)

    totals = {"fleets": 0, "oob": 0, "sun": 0, "games": 0}
    t0 = time.time()

    for seed in seeds:
        result = run_one_game(agent_fn, seed)
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
