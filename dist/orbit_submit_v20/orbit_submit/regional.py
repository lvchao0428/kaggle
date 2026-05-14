from __future__ import annotations

from orbit_submit.constants import *

# ╔═══ region 0b: v20 regional graph (inlined; no external module) ══════════╗


@dataclass
class Region:
    """Spatial region cluster (centroid + optional bookkeeping fields)."""
    id: int
    center: Tuple[float, float]
    my_planets: List[int] = field(default_factory=list)
    enemy_planets: List[int] = field(default_factory=list)
    external_planets: List[int] = field(default_factory=list)
    production_rate: int = 0


@dataclass
class Wave:
    target_id: int
    required_ships: int
    launch_turn: int
    sources: List[int] = field(default_factory=list)
    expected_arrival: int = 0


class RegionalGraph:
    """Geographic clustering (4 regions) + cached path lengths (sun-aware penalty)."""

    def __init__(self, planets: List, spawn_positions: Optional[List[Tuple[float, float]]] = None):
        self.planets = planets
        self.regions: Dict[int, Region] = {}
        self.planet_to_region: Dict[int, int] = {}
        self.dijkstra_cache: Dict[Tuple[int, int], Tuple[float, int]] = {}

        coords = np.array([[p.x, p.y] for p in planets])
        try:
            if _scipy_fclusterdata is None:
                raise RuntimeError("scipy not available")
            cluster_labels = _scipy_fclusterdata(
                coords, t=4, criterion="maxclust", method="complete"
            )
            self._build_regions(cluster_labels, spawn_positions)
        except Exception:
            self._build_regions_fallback(spawn_positions)

        self._precompute_dijkstra()

    def _build_regions(self, cluster_labels: np.ndarray, spawn_positions: Optional[List] = None):
        clusters: Dict[int, List] = {}
        for planet, cluster_id in zip(self.planets, cluster_labels):
            clusters.setdefault(cluster_id, []).append(planet)

        region_list: List[Region] = []
        for cluster_id in sorted(clusters.keys())[:4]:
            planets_in_cluster = clusters[cluster_id]
            center_x = sum(p.x for p in planets_in_cluster) / len(planets_in_cluster)
            center_y = sum(p.y for p in planets_in_cluster) / len(planets_in_cluster)
            region = Region(id=len(region_list), center=(center_x, center_y))
            region_list.append(region)
            for p in planets_in_cluster:
                self.planet_to_region[p.id] = region.id

        self.regions = {r.id: r for r in region_list}

    def _build_regions_fallback(self, spawn_positions: Optional[List] = None):
        if spawn_positions and len(spawn_positions) >= 2:
            spawn_array = np.array(
                spawn_positions[:4] if len(spawn_positions) >= 4 else spawn_positions * 2
            )
        else:
            spawn_array = np.array([(25, 25), (75, 25), (25, 75), (75, 75)])

        for planet in self.planets:
            distances = [math.hypot(planet.x - sp[0], planet.y - sp[1]) for sp in spawn_array]
            nearest_region = distances.index(min(distances)) % 4
            self.planet_to_region[planet.id] = nearest_region

        for i in range(4):
            self.regions[i] = Region(id=i, center=tuple(spawn_array[i]))

    def _precompute_dijkstra(self):
        planet_ids = [p.id for p in self.planets]
        for src_id in planet_ids:
            for dst_id in planet_ids:
                if src_id != dst_id:
                    distance, steps = self._dijkstra_impl(src_id, dst_id)
                    self.dijkstra_cache[(src_id, dst_id)] = (distance, steps)

    def _dijkstra_impl(self, src_id: int, dst_id: int) -> Tuple[float, int]:
        src = next((p for p in self.planets if p.id == src_id), None)
        dst = next((p for p in self.planets if p.id == dst_id), None)
        if not src or not dst:
            return 999.0, 999

        direct_dist = math.hypot(src.x - dst.x, src.y - dst.y)
        if point_segment_distance(SUN_X, SUN_Y, src.x, src.y, dst.x, dst.y) < SUN_RADIUS + SUN_PATH_MARGIN:
            total_dist = direct_dist + SUN_RADIUS * 2
        else:
            total_dist = direct_dist
        steps = max(1, int(math.ceil(total_dist / 2.0)))
        return total_dist, steps

    def dijkstra(self, src_id: int, dst_id: int) -> Tuple[float, int]:
        key = (src_id, dst_id)
        if key in self.dijkstra_cache:
            return self.dijkstra_cache[key]
        return 999.0, 999

    def in_same_region(self, pid1: int, pid2: int) -> bool:
        return self.planet_to_region.get(pid1, -1) == self.planet_to_region.get(pid2, -1)

    def get_region_by_id(self, region_id: int) -> Optional[Region]:
        return self.regions.get(region_id)

    def get_region_by_planet(self, planet_id: int) -> Optional[Region]:
        region_id = self.planet_to_region.get(planet_id, -1)
        return self.regions.get(region_id)

    def region_production(self, region_id: int, my_control: Set[int]) -> int:
        if region_id not in self.regions:
            return 0
        production = 0
        for planet in self.planets:
            if planet.id in my_control and self.planet_to_region.get(planet.id) == region_id:
                production += planet.production
        return production

    def region_threat(self, region_id: int, enemy_planets: Sequence) -> float:
        region = self.regions.get(region_id)
        if not region:
            return 0.0
        rcx, rcy = region.center
        total = 0.0
        for p in enemy_planets:
            pr = self.planet_to_region.get(p.id, -1)
            prod = float(getattr(p, "production", 0) or 0)
            ships = int(getattr(p, "ships", 0) or 0)
            if pr == region_id:
                total += prod * 2.2 + math.sqrt(max(1, ships)) * 0.25
            else:
                d = math.hypot(p.x - rcx, p.y - rcy)
                if d < 42.0:
                    w = max(0.0, (42.0 - d) / 42.0)
                    total += w * (prod * 1.4 + math.sqrt(max(1, ships)) * 0.12)
        return total

    def get_all_regions(self) -> List[Region]:
        return list(self.regions.values())

    def get_planets_in_region(self, region_id: int) -> List:
        return [p for p in self.planets if self.planet_to_region.get(p.id) == region_id]


class ProductionTimeline:
    def __init__(self, planets: List, my_control: Set[int]):
        self.planets = planets
        self.my_control = my_control

    def predict_surplus(self, planet_ids: List[int], turns_ahead: int) -> List[int]:
        surplus_per_turn: List[int] = []
        for turn in range(turns_ahead):
            production = sum(
                p.production for p in self.planets
                if p.id in planet_ids and p.id in self.my_control
            )
            accumulated = production * (turn + 1)
            available = int(accumulated * 0.8)
            surplus_per_turn.append(available)
        return surplus_per_turn

    def can_support_wave(self, sources: List[int], required: int, launch_turn: int) -> bool:
        surpluses = self.predict_surplus(sources, launch_turn + 1)
        if launch_turn < len(surpluses):
            return surpluses[launch_turn] >= required
        return False


def calculate_safe_surplus(my_planets: List, my_production: int, enemy_threats: Dict) -> int:
    max_threat = max(enemy_threats.values()) if enemy_threats else 0
    defensive_requirement = int(max_threat * 1.5)
    safe_surplus = my_production - defensive_requirement
    return max(0, int(safe_surplus * 0.65))


class MultiHopPlanner:
    def __init__(self, regional_graph: RegionalGraph, production_timeline: ProductionTimeline):
        self.regional_graph = regional_graph
        self.timeline = production_timeline

    def plan_attack_sequence(
        self, target_id: int, my_region_id: int, budget_turns: int = 5, max_hops: int = 3
    ) -> List[Wave]:
        target_planet = next((p for p in self.regional_graph.planets if p.id == target_id), None)
        if not target_planet:
            return []
        my_sources = [
            p for p in self.regional_graph.planets
            if self.regional_graph.planet_to_region.get(p.id) == my_region_id
        ]
        if not my_sources:
            return []
        source_planet = my_sources[0]
        _distance, steps = self.regional_graph.dijkstra(source_planet.id, target_id)
        wave = Wave(
            target_id=target_id,
            required_ships=int(target_planet.ships + target_planet.production * 2),
            launch_turn=0,
            sources=[s.id for s in my_sources[:3]],
            expected_arrival=steps,
        )
        return [wave]


