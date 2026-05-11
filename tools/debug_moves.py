"""Debug script to trace moves and check for sun-targeting fleets."""

import importlib.util
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission_resolve import resolve_submission_path

SUN_X = SUN_Y = 50.0
SUN_RADIUS = 10.0


def load_agent(version: str):
    path = resolve_submission_path(ROOT, version)
    spec = importlib.util.spec_from_file_location(f"submission_{version}_debug", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"submission_{version}_debug"] = mod
    spec.loader.exec_module(mod)
    return mod


def is_targeting_sun(src_x, src_y, angle, ships, max_speed=6.0):
    """Check if a fleet is heading toward the sun region."""
    # Simple check: does the ray intersect the sun region within first 50 steps?
    spd = 1.0 + (max_speed - 1.0) * (math.log(max(1, ships)) / math.log(1000)) ** 1.5
    spd = min(spd, max_speed)
    
    for eta in range(1, 50):
        fx = src_x + math.cos(angle) * spd * eta
        fy = src_y + math.sin(angle) * spd * eta
        dist_to_sun = math.hypot(fx - SUN_X, fy - SUN_Y)
        if dist_to_sun < SUN_RADIUS + 2:
            return True, eta, dist_to_sun
    return False, -1, 999.0


def main():
    from kaggle_environments import make
    
    agent_v18 = load_agent("v18").agent
    
    env = make("orbit_wars", configuration={"seed": 0}, debug=False)
    env.reset()
    
    step = 0
    while not env.done and step < 60:
        obs_list = env.state
        if obs_list is None:
            break
        
        player_obs = obs_list[0]["observation"]
        
        try:
            moves = agent_v18(player_obs, env.configuration)
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
        
        if moves:
            planets = {}
            raw_planets = player_obs.get("planets", [])
            for p in raw_planets:
                if isinstance(p, (list, tuple)) and len(p) >= 4:
                    planets[p[0]] = (p[2], p[3], p[6] if len(p) > 6 else 0)
            
            print(f"\n=== STEP {step} ===")
            for move in moves:
                if len(move) >= 3:
                    src_id = move[0]
                    angle = move[1]
                    ships = move[2]
                    if src_id in planets:
                        sx, sy, prod = planets[src_id]
                        is_sun, eta, min_dist = is_targeting_sun(sx, sy, angle, ships)
                        sun_flag = " ⚠️ SUN!" if is_sun else ""
                        print(f"  {src_id} -> angle={angle:.3f} ships={ships}{sun_flag} (min_dist_to_sun={min_dist:.1f})")
        
        env.step([moves, None])
        step += 1


if __name__ == "__main__":
    main()
