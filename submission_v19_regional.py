"""Regional Graph infrastructure for v19.0.

Core concepts:
- RegionalGraph: Clusters planets into 4 regions (based on spawn + geography)
- Dijkstra cache: Precomputed shortest paths avoiding sun
- Multi-hop planning: Sequential captures across regions
- Safe surplus: Balance between offense and defense

This module is independent of v19.py and can be tested in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from scipy.cluster.hierarchy import fclusterdata
import numpy as np

SUN_X = SUN_Y = 50.0
SUN_RADIUS = 10.0
BOARD = 100.0
DEFAULT_MAX_SHIP_SPEED = 6.0


@dataclass
class Region:
    """Represents a spatial region containing planets."""
    id: int
    center: Tuple[float, float]
    my_planets: List[int] = field(default_factory=list)
    enemy_planets: List[int] = field(default_factory=list)
    external_planets: List[int] = field(default_factory=list)
    production_rate: int = 0


@dataclass
class Wave:
    """Represents a single wave of fleet attack."""
    target_id: int
    required_ships: int
    launch_turn: int
    sources: List[int] = field(default_factory=list)
    expected_arrival: int = 0


def point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Closest distance from point (px,py) to line segment (ax,ay)-(bx,by)."""
    abx, aby = bx - ax, by - ay
    l2 = abx * abx + aby * aby
    if l2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / l2))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return math.hypot(px - proj_x, py - proj_y)


def segment_hits_sun(ax: float, ay: float, bx: float, by: float, margin: float = 3.0) -> bool:
    """Check if line segment hits sun (with margin)."""
    return point_segment_distance(SUN_X, SUN_Y, ax, ay, bx, by) < SUN_RADIUS + margin


class RegionalGraph:
    """Geographic clustering and path optimization for 4 regions."""

    def __init__(self, planets: List, spawn_positions: Optional[List[Tuple[float, float]]] = None):
        """Initialize regional clustering.

        Args:
            planets: List of Planet objects with x, y, id, production attributes
            spawn_positions: Optional list of (x, y) spawn points (up to 4)
        """
        self.planets = planets
        self.regions: Dict[int, Region] = {}
        self.planet_to_region: Dict[int, int] = {}
        self.dijkstra_cache: Dict[Tuple[int, int], Tuple[float, int]] = {}

        # Extract coordinates for clustering
        coords = np.array([[p.x, p.y] for p in planets])

        # Cluster into 4 regions using hierarchical clustering
        try:
            cluster_labels = fclusterdata(coords, t=4, criterion='maxclust', method='complete')
            self._build_regions(cluster_labels, spawn_positions)
        except Exception as e:
            # Fallback: simple distance-based clustering
            print(f"Clustering failed: {e}. Using fallback method.")
            self._build_regions_fallback(spawn_positions)

        # Precompute dijkstra cache for all planet pairs
        self._precompute_dijkstra()

    def _build_regions(self, cluster_labels: np.ndarray, spawn_positions: Optional[List] = None):
        """Build regions from cluster labels."""
        # Group planets by cluster
        clusters = {}
        for planet, cluster_id in zip(self.planets, cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(planet)

        # Create 4 regions (take first 4 clusters, combine if needed)
        region_list = []
        for cluster_id in sorted(clusters.keys())[:4]:
            planets_in_cluster = clusters[cluster_id]
            center_x = sum(p.x for p in planets_in_cluster) / len(planets_in_cluster)
            center_y = sum(p.y for p in planets_in_cluster) / len(planets_in_cluster)

            region = Region(
                id=len(region_list),
                center=(center_x, center_y),
            )
            region_list.append(region)
            for p in planets_in_cluster:
                self.planet_to_region[p.id] = region.id

        self.regions = {r.id: r for r in region_list}

    def _build_regions_fallback(self, spawn_positions: Optional[List] = None):
        """Fallback: simple distance-based clustering from spawn points."""
        # If spawn positions provided, use them as region centers
        if spawn_positions and len(spawn_positions) >= 2:
            # Create 4 regions based on spawn position quadrants
            spawn_array = np.array(spawn_positions[:4] if len(spawn_positions) >= 4 else spawn_positions * 2)
        else:
            # Fallback: divide board into 4 quadrants
            spawn_array = np.array([(25, 25), (75, 25), (25, 75), (75, 75)])

        # Assign each planet to nearest spawn
        for planet in self.planets:
            distances = [math.hypot(planet.x - sp[0], planet.y - sp[1]) for sp in spawn_array]
            nearest_region = distances.index(min(distances)) % 4
            self.planet_to_region[planet.id] = nearest_region

        # Create region objects
        for i in range(4):
            self.regions[i] = Region(id=i, center=tuple(spawn_array[i]))

    def _precompute_dijkstra(self):
        """Precompute shortest paths between all planets avoiding sun."""
        planet_ids = [p.id for p in self.planets]
        for src_id in planet_ids:
            for dst_id in planet_ids:
                if src_id != dst_id:
                    distance, steps = self._dijkstra_impl(src_id, dst_id)
                    self.dijkstra_cache[(src_id, dst_id)] = (distance, steps)

    def _dijkstra_impl(self, src_id: int, dst_id: int) -> Tuple[float, int]:
        """Compute shortest path from src to dst avoiding sun.

        Returns: (distance, steps)
        """
        src = next((p for p in self.planets if p.id == src_id), None)
        dst = next((p for p in self.planets if p.id == dst_id), None)
        if not src or not dst:
            return 999.0, 999

        # Simple heuristic: direct distance with sun-avoidance penalty
        direct_dist = math.hypot(src.x - dst.x, src.y - dst.y)

        # Check if direct path hits sun
        if segment_hits_sun(src.x, src.y, dst.x, dst.y, margin=3.0):
            # Add penalty for sun avoidance (arc around sun)
            sun_penalty = SUN_RADIUS * 2
            total_dist = direct_dist + sun_penalty
        else:
            total_dist = direct_dist

        # Estimate steps (assuming average fleet speed ~2.0)
        steps = max(1, int(math.ceil(total_dist / 2.0)))
        return total_dist, steps

    def dijkstra(self, src_id: int, dst_id: int) -> Tuple[float, int]:
        """Get cached shortest path.

        Returns: (distance, steps)
        """
        key = (src_id, dst_id)
        if key in self.dijkstra_cache:
            return self.dijkstra_cache[key]
        return 999.0, 999

    def in_same_region(self, pid1: int, pid2: int) -> bool:
        """Check if two planets are in the same region."""
        return self.planet_to_region.get(pid1, -1) == self.planet_to_region.get(pid2, -1)

    def get_region_by_id(self, region_id: int) -> Optional[Region]:
        """Get region object by ID."""
        return self.regions.get(region_id)

    def get_region_by_planet(self, planet_id: int) -> Optional[Region]:
        """Get region object containing a planet."""
        region_id = self.planet_to_region.get(planet_id, -1)
        return self.regions.get(region_id)

    def region_production(self, region_id: int, my_control: Set[int]) -> int:
        """Sum of production for friendly planets in region."""
        region = self.regions.get(region_id)
        if not region:
            return 0

        production = 0
        for planet in self.planets:
            if (planet.id in my_control and
                self.planet_to_region.get(planet.id) == region_id):
                production += planet.production
        return production

    def region_threat(self, region_id: int, state_by_enemy: Dict) -> float:
        """Estimate threat to a region from enemies.

        Args:
            region_id: target region ID
            state_by_enemy: dict mapping enemy_id -> enemy planets list

        Returns: threat speed (ships per turn)
        """
        region = self.regions.get(region_id)
        if not region:
            return 0.0

        max_threat = 0.0
        for enemy_id, enemy_planets in state_by_enemy.items():
            # Fastest enemy fleet ETA to region center
            min_eta = 999
            for planet in enemy_planets:
                _, eta = self.dijkstra(planet.id, list(self.planets)[0].id)  # Find nearest planet as proxy
                min_eta = min(min_eta, eta)

            # Enemy production rate
            enemy_prod = sum(p.production for p in enemy_planets)
            if min_eta > 0:
                threat_speed = enemy_prod / min_eta
                max_threat = max(max_threat, threat_speed)

        return max_threat

    def get_all_regions(self) -> List[Region]:
        """Return all region objects."""
        return list(self.regions.values())

    def get_planets_in_region(self, region_id: int) -> List:
        """Return all planets in a region."""
        return [p for p in self.planets if self.planet_to_region.get(p.id) == region_id]


class ProductionTimeline:
    """Predict available surplus production over time."""

    def __init__(self, planets: List, my_control: Set[int]):
        """Initialize with planets and my controlled planet IDs."""
        self.planets = planets
        self.my_control = my_control

    def predict_surplus(self, planet_ids: List[int], turns_ahead: int) -> List[int]:
        """Predict available surplus for each turn ahead.

        Args:
            planet_ids: list of planet IDs to monitor
            turns_ahead: how many turns to predict

        Returns: List of available surplus ships per turn
        """
        surplus_per_turn = []
        for turn in range(turns_ahead):
            # Assume production accumulates linearly (simplified)
            production = sum(
                p.production for p in self.planets
                if p.id in planet_ids and p.id in self.my_control
            )
            accumulated = production * (turn + 1)
            # Subtract a small defensive reserve (20%)
            available = int(accumulated * 0.8)
            surplus_per_turn.append(available)
        return surplus_per_turn

    def can_support_wave(self, sources: List[int], required: int, launch_turn: int) -> bool:
        """Check if sources can produce enough ships by launch_turn.

        Args:
            sources: list of source planet IDs
            required: number of ships needed
            launch_turn: when the wave launches

        Returns: True if sufficient production available
        """
        surpluses = self.predict_surplus(sources, launch_turn + 1)
        if launch_turn < len(surpluses):
            return surpluses[launch_turn] >= required
        return False


def calculate_safe_surplus(my_planets: List, my_production: int, enemy_threats: Dict) -> int:
    """Calculate how many ships can safely be used for offense.

    Args:
        my_planets: list of my planets
        my_production: total production rate
        enemy_threats: dict mapping region_id -> threat_speed

    Returns: ships safe to allocate for offense
    """
    # Defensive requirement = max incoming threat * safety margin
    max_threat = max(enemy_threats.values()) if enemy_threats else 0
    defensive_requirement = int(max_threat * 1.5)

    # Safe surplus = my production - defensive requirement
    safe_surplus = my_production - defensive_requirement
    safe_surplus = max(0, int(safe_surplus * 0.65))  # Conservative: use 65% of available

    return safe_surplus


class MultiHopPlanner:
    """Plans sequential captures across multiple hops (intermediate targets)."""

    def __init__(self, regional_graph: RegionalGraph, production_timeline: ProductionTimeline):
        """Initialize with regional graph and production timeline.

        Args:
            regional_graph: RegionalGraph instance
            production_timeline: ProductionTimeline instance
        """
        self.regional_graph = regional_graph
        self.timeline = production_timeline

    def plan_attack_sequence(self, target_id: int, my_region_id: int, 
                            budget_turns: int = 5, max_hops: int = 3) -> List[Wave]:
        """Plan a multi-hop attack sequence to reach target.

        Args:
            target_id: ID of final target planet
            my_region_id: ID of my home region
            budget_turns: maximum turns to plan
            max_hops: maximum number of hops allowed

        Returns: List of Wave objects (empty if not feasible)
        """
        target_planet = next((p for p in self.regional_graph.planets if p.id == target_id), None)
        if not target_planet:
            return []

        # Get dijkstra path from my region to target
        # Find a representative source in my region
        my_sources = [p for p in self.regional_graph.planets 
                     if self.regional_graph.planet_to_region.get(p.id) == my_region_id]
        if not my_sources:
            return []

        source_planet = my_sources[0]
        distance, steps = self.regional_graph.dijkstra(source_planet.id, target_id)

        # For v19.0 MVP, treat this as a single wave (simplified multi-hop)
        # TODO: Implement true multi-hop decomposition in v19.1
        
        waves = []
        wave = Wave(
            target_id=target_id,
            required_ships=int(target_planet.ships + target_planet.production * 2),
            launch_turn=0,
            sources=[s.id for s in my_sources[:3]],
            expected_arrival=steps
        )
        waves.append(wave)

        return waves
