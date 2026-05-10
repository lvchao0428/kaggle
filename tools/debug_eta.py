"""Check what eta safe_aim returns."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(version):
    path = ROOT / f"submission_{version}.py"
    spec = importlib.util.spec_from_file_location(f"submit_{version}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"submit_{version}"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    from kaggle_environments import make
    
    mod = load_module("v18")
    
    env = make("orbit_wars", configuration={"seed": 0}, debug=False)
    env.reset()
    
    step = 0
    while not env.done and step < 60:
        obs_list = env.state
        if obs_list is None:
            break
        
        player_obs = obs_list[0]["observation"]
        
        if step == 49:
            print(f"STEP 49: Checking safe_aim return values\n")
            
            # Manually call safe_aim for src=12, checking various targets
            state = mod.GameState(player_obs, env.configuration)
            src_12 = None
            for p in state.my_pl:
                if p.id == 12:
                    src_12 = p
                    break
            
            if src_12:
                print(f"Planet 12 pos: ({src_12.x:.1f}, {src_12.y:.1f})\n")
                
                # Check what target resulted in the -0.0045 angle
                # Try nearby targets
                for dst in state.en_pl[:5]:
                    angle, eta = mod.safe_aim(state, src_12, dst, 5)
                    print(f"  Target planet {dst.id} ({dst.x:.1f}, {dst.y:.1f}): angle={angle:.4f} eta={eta}")
            
            break
        
        try:
            mod.agent(player_obs, env.configuration)
        except:
            pass
        
        env.step([[], None])
        step += 1


if __name__ == "__main__":
    main()
