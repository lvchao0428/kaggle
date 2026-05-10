"""Inline debugging version of safe_aim to see what's happening."""

import math

SUN_X = SUN_Y = 50.0
SUN_RADIUS = 10.0
BOARD = 100.0


def point_segment_distance(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    l2 = abx * abx + aby * aby
    if l2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / l2))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return math.hypot(px - proj_x, py - proj_y)


# Simulating the problematic case
# src=(16.9, 54.5), angle=-0.0045, speed=1.562, eta (unknown at first)
src_x, src_y = 16.9, 54.5
angle = -0.0045
spd = 1.562

# At step 49, what was the eta returned by safe_aim?
# Let me try to reverse-engineer it from the trajectory
# If dist_to_sun=10.6 at eta=15, what's the endpoint?
fx_15 = src_x + math.cos(angle) * spd * 15
fy_15 = src_y + math.sin(angle) * spd * 15
print(f"At eta=15: endpoint=({fx_15:.1f}, {fy_15:.1f})")

# Check segment distance from src to fx_15,fy_15
seg_dist_15 = point_segment_distance(SUN_X, SUN_Y, src_x, src_y, fx_15, fy_15)
print(f"segment_distance(sun to src-endpoint[15]): {seg_dist_15:.1f}")
print(f"segment_hits_sun(margin=2.0)? {seg_dist_15 < SUN_RADIUS + 2.0}")
print(f"segment_hits_sun(margin=3.0)? {seg_dist_15 < SUN_RADIUS + 3.0}")

#Now let's check what segment_hits_sun is checking in safe_aim
# It should be checking from src to `best_a, best_e` endpoint
# But what IS best_e? Let me check various e values

print("\n\nSearching for best_a, best_e that passes sun_clear with margin=3.0:")
SUN_MARGIN = 3.0

for test_e in [1, 2, 5, 10, 13, 14, 15, 16, 20]:
    ex = src_x + math.cos(angle) * spd * test_e
    ey = src_y + math.sin(angle) * spd * test_e
    seg_dist = point_segment_distance(SUN_X, SUN_Y, src_x, src_y, ex, ey)
    clearance = seg_dist - (SUN_RADIUS + SUN_MARGIN)
    print(f"  e={test_e:2d}: endpoint=({ex:6.1f}, {ey:6.1f}) seg_dist={seg_dist:5.1f} clearance={clearance:6.1f} " + 
          ("✓ CLEAR" if clearance > 0 else "✗ CLIP"))
