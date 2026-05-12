#!/usr/bin/env python3
"""本地诊断工具：找出撞日 / 飞出屏幕的根本原因。

用法:
  python3 tools/diagnose_flight.py --seed 0 --a v20 --b v18
  
输出: 每个被发射的舰队的完整轨迹，检查是否会撞日或越界。
"""

import sys
import math
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from submission_v20 import (
    GameState, fleet_speed, point_segment_distance, segment_hits_sun,
    SUN_X, SUN_Y, SUN_RADIUS, BOARD
)

def check_trajectory(src_x, src_y, angle, ships, max_steps=100, name=""):
    """检查一条轨迹是否会撞日或越界。
    
    返回: (is_oob, is_sun_hit, first_problem_step, endpoint_x, endpoint_y)
    """
    spd = fleet_speed(ships)
    
    for step in range(1, max_steps + 1):
        x = src_x + math.cos(angle) * spd * step
        y = src_y + math.sin(angle) * spd * step
        
        # 检查越界
        if not (0 <= x <= BOARD and 0 <= y <= BOARD):
            return True, False, step, x, y
        
        # 检查撞日（轨迹段）
        if segment_hits_sun(src_x, src_y, x, y, margin=1.25):
            return False, True, step, x, y
    
    final_x = src_x + math.cos(angle) * spd * max_steps
    final_y = src_y + math.sin(angle) * spd * max_steps
    return False, False, -1, final_x, final_y


def analyze_from_obs(obs_list):
    """逐帧分析观测，找出问题的舰队。
    
    obs_list: 从回放中提取的 obs 列表
    """
    
    print("\n" + "="*70)
    print("FLIGHT TRAJECTORY ANALYSIS")
    print("="*70)
    
    for step_idx, obs in enumerate(obs_list):
        state = GameState(obs)
        
        # 遍历所有舰队，推断是否是本方发射
        for f in state.fleets:
            if f.owner != state.my_id:
                continue
            
            target_info = state.fleet_target.get(f.id)
            if not target_info:
                continue
            
            target_id, eta_to_planet = target_info
            target = state.get(target_id)
            if not target:
                continue
            
            is_oob, is_sun_hit, problem_step, ex, ey = check_trajectory(
                f.x, f.y, f.angle, f.ships, max_steps=100,
                name=f"Fleet {f.id} from {f.planet_id if hasattr(f, 'planet_id') else '?'}"
            )
            
            if is_oob or is_sun_hit:
                print(f"\n[Step {step_idx}] ⚠️  PROBLEM FLEET")
                print(f"  Fleet ID: {f.id}")
                print(f"  Position: ({f.x:.1f}, {f.y:.1f})")
                print(f"  Angle: {f.angle:.4f} ({math.degrees(f.angle):.1f}°)")
                print(f"  Ships: {f.ships}")
                print(f"  Speed: {fleet_speed(f.ships):.2f}")
                print(f"  Target: Planet {target_id} at ({target.x:.1f}, {target.y:.1f})")
                
                if is_oob:
                    print(f"  ❌ OOB at step {problem_step}: ({ex:.1f}, {ey:.1f})")
                if is_sun_hit:
                    print(f"  ☀️  SUN HIT at step {problem_step}: ({ex:.1f}, {ey:.1f})")
                    sun_dist = point_segment_distance(SUN_X, SUN_Y, f.x, f.y, ex, ey)
                    print(f"  Distance to sun: {sun_dist:.2f} (radius={SUN_RADIUS})")


def main():
    print("本地诊断工具：扫描回放中的轨迹问题")
    print("需要集成 kaggle-environments 或本地回放数据源")
    print("\n当前版本：扫骨架就位，需要与 replay.py 对接")
    

if __name__ == "__main__":
    main()
