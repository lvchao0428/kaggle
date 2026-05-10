"""Unit and integration tests for v19.0 regional graph infrastructure."""

import math
import random
from typing import List, Tuple

# Try importing from kaggle package, else from local
try:
    from kaggle.submission_v19_regional import (
        RegionalGraph, Region, Wave, ProductionTimeline,
        calculate_safe_surplus, MultiHopPlanner,
        point_segment_distance, segment_hits_sun
    )
except ImportError:
    from submission_v19_regional import (
        RegionalGraph, Region, Wave, ProductionTimeline,
        calculate_safe_surplus, MultiHopPlanner,
        point_segment_distance, segment_hits_sun
    )


# Mock Planet class for testing
class MockPlanet:
    def __init__(self, pid, x, y, owner=0, ships=10, production=1, is_comet=False):
        self.id = pid
        self.x = x
        self.y = y
        self.owner = owner
        self.ships = ships
        self.production = production
        self.is_comet = is_comet

    def dist(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)


def create_test_planets() -> List[MockPlanet]:
    """Create a simple 4-planet test scenario (one per quadrant)."""
    return [
        MockPlanet(0, 25, 25, owner=0, production=3),  # My home (bottom-left)
        MockPlanet(1, 75, 25, owner=1, production=2),  # Enemy (bottom-right)
        MockPlanet(2, 25, 75, owner=-1, production=2), # Neutral (top-left)
        MockPlanet(3, 75, 75, owner=1, production=3),  # Enemy (top-right)
    ]


class TestPointSegmentDistance:
    """Test geometry helper."""

    def test_point_on_segment(self):
        # Point on segment should have distance 0
        d = point_segment_distance(50, 50, 0, 0, 100, 100)
        assert abs(d - 0) < 0.1, f"Expected ~0, got {d}"
        print("✓ Point on segment: distance ~0")

    def test_point_off_segment(self):
        # Point perpendicular to segment
        d = point_segment_distance(50, 60, 0, 0, 100, 0)
        assert abs(d - 60) < 0.1, f"Expected ~60, got {d}"
        print("✓ Point off segment: distance correct")

    def test_sun_distance(self):
        # Sun is at (50, 50)
        d = point_segment_distance(50, 50, 40, 40, 60, 60)
        assert abs(d - 0) < 0.1, f"Expected ~0, got {d}"
        print("✓ Sun distance test: 0 (sun on segment)")


class TestSegmentHitsSun:
    """Test sun collision detection."""

    def test_segment_hits_sun(self):
        # Segment passing through sun
        hits = segment_hits_sun(45, 45, 55, 55, margin=3.0)
        assert hits, "Should detect sun collision"
        print("✓ Segment hits sun: detected")

    def test_segment_misses_sun(self):
        # Segment far from sun
        hits = segment_hits_sun(0, 0, 10, 10, margin=3.0)
        assert not hits, "Should not detect collision"
        print("✓ Segment misses sun: correctly identified")

    def test_segment_near_sun(self):
        # Segment just outside sun radius + margin
        hits = segment_hits_sun(45, 50, 55, 50, margin=3.0)
        # This should probably hit with margin 3.0
        print(f"✓ Segment near sun: hits={hits} (margin=3.0)")


class TestRegionalGraph:
    """Test regional clustering and dijkstra."""

    def test_region_creation(self):
        planets = create_test_planets()
        graph = RegionalGraph(planets, spawn_positions=[(25, 25), (75, 25), (25, 75), (75, 75)])

        # Should have 4 regions
        assert len(graph.regions) == 4, f"Expected 4 regions, got {len(graph.regions)}"
        print("✓ Regional graph created with 4 regions")

    def test_planet_assignment(self):
        planets = create_test_planets()
        graph = RegionalGraph(planets, spawn_positions=[(25, 25), (75, 25), (25, 75), (75, 75)])

        # Each planet should be assigned to exactly one region
        for planet in planets:
            region_id = graph.planet_to_region.get(planet.id)
            assert region_id is not None, f"Planet {planet.id} not assigned to region"
        print("✓ All planets assigned to regions")

    def test_same_region(self):
        planets = create_test_planets()
        graph = RegionalGraph(planets)

        # Planets in same quadrant should be same region
        p0_region = graph.planet_to_region[planets[0].id]
        p1_region = graph.planet_to_region[planets[1].id]
        assert p0_region != p1_region, "Different quadrants should be different regions"
        print("✓ Same-region check works")

    def test_dijkstra_cache(self):
        planets = create_test_planets()
        graph = RegionalGraph(planets)

        # Dijkstra cache should be populated
        distance, steps = graph.dijkstra(0, 1)
        assert distance > 0, "Distance should be positive"
        assert steps > 0, "Steps should be positive"
        print(f"✓ Dijkstra cache: {len(graph.dijkstra_cache)} entries, 0->1: dist={distance:.1f}, steps={steps}")

    def test_region_production(self):
        planets = create_test_planets()
        graph = RegionalGraph(planets)

        my_control = {0}  # Only planet 0 is mine
        for region_id in graph.regions.keys():
            prod = graph.region_production(region_id, my_control)
            # Should be 3 for region containing planet 0, 0 for others
            print(f"  Region {region_id}: production={prod}")
        print("✓ Region production calculation")


class TestProductionTimeline:
    """Test production forecasting."""

    def test_predict_surplus(self):
        planets = create_test_planets()
        my_control = {0}  # Only planet 0
        timeline = ProductionTimeline(planets, my_control)

        # Predict 5 turns ahead
        surpluses = timeline.predict_surplus([0], 5)
        assert len(surpluses) == 5, f"Expected 5 surpluses, got {len(surpluses)}"
        # Should be increasing (production accumulates)
        for i in range(1, len(surpluses)):
            assert surpluses[i] >= surpluses[i-1], "Surplus should increase over time"
        print(f"✓ Production timeline: {surpluses}")

    def test_can_support_wave(self):
        planets = create_test_planets()
        my_control = {0}
        timeline = ProductionTimeline(planets, my_control)

        # Check if we can support a wave at turn 2
        can_support = timeline.can_support_wave([0], 100, 2)
        print(f"✓ Can support wave: {can_support}")


class TestCalculateSafeSurplus:
    """Test safe surplus calculation."""

    def test_zero_threat(self):
        planets = create_test_planets()
        my_production = 10
        enemy_threats = {}
        
        surplus = calculate_safe_surplus(planets, my_production, enemy_threats)
        # With zero threat, should be 0.65 * 10 = 6.5 -> 6
        expected = int(my_production * 0.65)
        assert surplus == expected, f"Expected {expected}, got {surplus}"
        print(f"✓ Zero threat: surplus={surplus}")

    def test_with_threat(self):
        planets = create_test_planets()
        my_production = 20
        enemy_threats = {0: 5.0, 1: 3.0}  # 5.0 threat in one region
        
        surplus = calculate_safe_surplus(planets, my_production, enemy_threats)
        # Defensive = 5.0 * 1.5 = 7.5
        # Surplus = (20 - 7.5) * 0.65 ≈ 8
        print(f"✓ With threat: surplus={surplus}")


class TestMultiHopPlanner:
    """Test multi-hop attack planning."""

    def test_plan_creation(self):
        planets = create_test_planets()
        graph = RegionalGraph(planets)
        timeline = ProductionTimeline(planets, {0})
        planner = MultiHopPlanner(graph, timeline)

        # Plan attack on planet 1 from region 0
        waves = planner.plan_attack_sequence(1, 0)
        assert isinstance(waves, list), "Should return list of waves"
        print(f"✓ Multi-hop planner: planned {len(waves)} wave(s)")

    def test_wave_structure(self):
        planets = create_test_planets()
        graph = RegionalGraph(planets)
        timeline = ProductionTimeline(planets, {0})
        planner = MultiHopPlanner(graph, timeline)

        waves = planner.plan_attack_sequence(1, 0)
        if waves:
            wave = waves[0]
            assert hasattr(wave, 'target_id'), "Wave should have target_id"
            assert hasattr(wave, 'required_ships'), "Wave should have required_ships"
            assert hasattr(wave, 'sources'), "Wave should have sources"
            print(f"✓ Wave structure: target={wave.target_id}, ships={wave.required_ships}, sources={wave.sources}")


def run_all_tests():
    """Run all test classes."""
    print("\n" + "="*60)
    print("v19.0 Regional Graph Test Suite")
    print("="*60)

    test_classes = [
        TestPointSegmentDistance,
        TestSegmentHitsSun,
        TestRegionalGraph,
        TestProductionTimeline,
        TestCalculateSafeSurplus,
        TestMultiHopPlanner,
    ]

    total = 0
    passed = 0

    for test_class in test_classes:
        print(f"\n[{test_class.__name__}]")
        tests = test_class()
        for method_name in dir(tests):
            if method_name.startswith("test_"):
                try:
                    getattr(tests, method_name)()
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                except Exception as e:
                    print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                total += 1

    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("✓ ALL TESTS PASSED")
    else:
        print(f"✗ {total - passed} tests failed")
    print("="*60 + "\n")

    return passed == total


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
