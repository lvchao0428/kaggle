"""Check what planet is being targeted at step 49."""

import importlib.util
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(version):
    path = ROOT / f"submission_{version}.py"
    spec = importlib.util.spec_from_file_location(f"submission_{version}_check", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"submission_{version}_check"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    from kaggle_environments import make
    
    mod_v18 = load_module("v18")
    GameState = mod_v18.GameState
    agent = mod_v18.agent
    
    env = make("orbit_wars", configuration={"seed": 0}, debug=False)
    env.reset()
    
    step = 0
    while not env.done and step < 60:
        obs_list = env.state
        if obs_list is None:
            break
        
        player_obs = obs_list[0]["observation"]
        
        if step == 49:
            print(f"\n=== STEP 49 - Looking for planet 12 targets ===\n")
            
            # Parse planets
            planets_by_id = {}
            raw_planets = player_obs.get("planets", [])
            for p in raw_planets:
                if isinstance(p, (list, tuple)) and len(p) >= 4:
                    pid, owner, px, py, radius, ships, prod = p[:7]
                    planets_by_id[pid] = {
                        'owner': owner, 'x': px, 'y': py, 'ships': ships, 'prod': prod
                    }
                    if pid == 12:
                        print(f"Planet 12 (src): pos=({px:.1f}, {py:.1f}) owner={owner} ships={ships}")
            
            # Look for recent arrivals to understand what planet 12 is targeting
            arrivals = player_obs.get("fleets", [])
            print(f"\nRecent fleets from planet 12:")
            for f in arrivals:
                if isinstance(f, (list, tuple)) and len(f) >= 4:
                    src, ships, dest_x, dest_y = f[:4]
                    if src == 12:
                        # Find which planet is near the destination
                        for pid, pinfo in planets_by_id.items():
                            dist = math.hypot(pinfo['x'] - dest_x, pinfo['y'] - dest_y)
                            if dist < 3:
                                print(f"  Fleet to planet {pid}: {ships} ships, dest=({dest_x:.1f}, {dest_y:.1f})")
            
            break
        
        try:
            moves = agent(player_obs, env.configuration)
        except Exception:
            pass
        
        env.step([moves, None])
        step += 1


if __name__ == "__main__":
    main()
