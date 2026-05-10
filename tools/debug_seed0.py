#!/usr/bin/env python3
"""诊断脚本：针对 seed 0 的特定舰队，逐步追踪 safe_aim 的决策。"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from submission_v19 import (
    GameState, safe_aim, fleet_speed, point_segment_distance, segment_hits_sun,
    SUN_X, SUN_Y, SUN_RADIUS, BOARD, lead_intercept
)

def diagnose_aim(src_x, src_y, dst_x, dst_y, ships, name=""):
    """诊断一次 safe_aim 调用的完整过程。"""
    
    print(f"\n{'='*70}")
    print(f"SAFE_AIM DIAGNOSIS: {name}")
    print(f"{'='*70}")
    print(f"Source: ({src_x:.1f}, {src_y:.1f})")
    print(f"Target: ({dst_x:.1f}, {dst_y:.1f})")
    print(f"Ships: {ships}")
    
    spd = fleet_speed(ships)
    print(f"Speed: {spd:.3f}")
    
    # Mock state for lead_intercept (需要更多上下文)
    # 这里简化为直线瞄准
    angle_direct = math.atan2(dst_y - src_y, dst_x - src_x)
    print(f"Direct angle: {angle_direct:.4f} ({math.degrees(angle_direct):.1f}°)")
    
    # 检查直线是否撞日
    dist_to_sun = point_segment_distance(SUN_X, SUN_Y, src_x, src_y, dst_x, dst_y)
    print(f"Direct path to target: distance to sun = {dist_to_sun:.2f} (radius={SUN_RADIUS})")
    
    if dist_to_sun < SUN_RADIUS + 3.0:
        print(f"  ⚠️  DIRECT PATH HITS SUN (margin < 3.0)")
    else:
        print(f"  ✓ Direct path OK")
    
    # 检查 eta=1 时的终点
    for eta in [1, 3, 5, 10, 20]:
        ex = src_x + math.cos(angle_direct) * spd * eta
        ey = src_y + math.sin(angle_direct) * spd * eta
        
        in_board = 0 <= ex <= BOARD and 0 <= ey <= BOARD
        sun_dist = point_segment_distance(SUN_X, SUN_Y, src_x, src_y, ex, ey)
        sun_ok = sun_dist > SUN_RADIUS + 4.0
        
        status = "✓" if (in_board and sun_ok) else "✗"
        print(f"  eta={eta:2d}: ({ex:.1f}, {ey:.1f}) board={in_board} sun_dist={sun_dist:.1f} {status}")

def main():
    # 模拟 Step 61 附近的情况
    # 根据截图，蓝色（我方）试图从左中的星球向太阳附近派兵
    
    print("\n[模拟场景：Step 61-78 期间的派兵]")
    print("\n尝试从 (45, 38) 向 (48, 35) 派 9 艘舰")
    diagnose_aim(45, 38, 48, 35, 9, "Attempt 1: Near-sun target")
    
    print("\n尝试从 (35, 42) 向星球 43 派兵")
    # 星球 43 在右上，但轨迹贴太阳
    diagnose_aim(35, 42, 65, 35, 12, "Attempt 2: Arc toward planet 43")

if __name__ == "__main__":
    main()
