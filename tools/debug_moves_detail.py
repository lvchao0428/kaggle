"""Debug script to inspect step 49 in detail."""

import importlib.util
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUN_X = SUN_Y = 50.0
SUN_RADIUS = 10.0


def load_agent(version: str):
    path = ROOT / f"submission_{version}.py"
    spec = importlib.util.spec_from_file_location(f"submission_{version}_debug", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"submission_{version}_debug"] = mod
    spec.loader.exec_module(mod)
    return mod


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
        
        if step == 49:
            print(f"\n=== STEP 49 DETAIL ===")
            planets = {}
            raw_planets = player_obs.get("planets", [])
            for p in raw_planets:
                if isinstance(p, (list, tuple)) and len(p) >= 4:
                    pid, owner, px, py, radius, ships, prod = p[:7]
                    planets[pid] = (px, py, owner, ships, prod, radius)
                    if pid == 12:
                        print(f"Planet 12: pos=({px:.1f}, {py:.1f}) owner={owner} ships={ships} prod={prod} radius={radius}")
            
            # Get the moves
            try:
                moves = agent_v18(player_obs, env.configuration)
            except Exception as e:
                print(f"Error: {e}")
                return
            
            for move in moves:
                if len(move) >= 3 and move[0] == 12:
                    src_id = move[0]
                    angle = move[1]
                    ships = move[2]
                    sx, sy = planets[src_id][:2]
                    
                    # Trace this fleet for 50 steps
                    spd = 1.0 + (6.0 - 1.0) * (math.log(max(1, ships)) / math.log(1000)) ** 1.5
                    spd = min(spd, 6.0)
                    
                    print(f"\nMove: src=12 angle={angle:.4f} ships={ships} speed={spd:.3f}")
                    print(f"Source position: ({sx:.1f}, {sy:.1f})")
                    print("\nTrajectory:")
                    
                    for eta in range(1, 21):
                        fx = sx + math.cos(angle) * spd * eta
                        fy = sy + math.sin(angle) * spd * eta
                        dist_to_sun = math.hypot(fx - SUN_X, fy - SUN_Y)
                        dist_to_src = math.hypot(fx - sx, fy - sy)
                        if dist_to_sun < SUN_RADIUS + 3:
                            print(f"  eta={eta:2d}: ({fx:6.1f}, {fy:6.1f}) dist_to_sun={dist_to_sun:.1f} ⚠️")
                        else:
                            print(f"  eta={eta:2d}: ({fx:6.1f}, {fy:6.1f}) dist_to_sun={dist_to_sun:.1f}")
            break
        
        try:
            moves = agent_v18(player_obs, env.configuration)
        except Exception:
            pass
        
        env.step([moves, None])
        step += 1


if __name__ == "__main__":
    main()
