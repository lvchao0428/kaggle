# Orbit Wars Bot Strategy Roadmap v19+

## Executive Summary

Current bot (v17/v18) suffers from scattered fleet allocation and no regional awareness. Top players use:
1. Regional control (4 zones based on spawn points)
2. Multi-hop sequential captures (2-3 jumps to target)
3. Production-aware timing (waves synchronized with local production)
4. Game-theoretic defense (safe surplus calculation)

v19-v22 roadmap moves from rule-heuristic to graph-theoretic + game-theoretic framework.

---

## Current Architecture Analysis

### What v17/v18 Has
- `DefensePlanner`: reactive defense (checks incoming threats)
- `ExpandPlanner`: greedy target selection (single-hop)
- `AttackPlanner`: high-production prioritization
- `safe_aim`: geometric path finding (avoids sun)
- `SYNC_ETA_WINDOW`: passive wave synchronization (narrow window=3)
- RL infrastructure: present but unused

### What v17/v18 Lacks
- Regional identification: treats all planets independently
- Multi-hop planning: no sequential capture planning
- Production timeline: doesn't model future surplus
- Graph algorithms: no dijkstra for shortest path
- Game theory: no minimax or threat modeling
- Active RL loop: no self-play optimization

---

## v19.0: Regional Graph + Dynamic Weighting + Multi-Hop

### Milestone 1a: Regional Graph Foundation

**New Classes:**
```python
RegionalGraph:
  - __init__(planets, spawn_positions)
    * Cluster planets into 4 regions (k-means or distance-based)
    * Compute dijkstra cache (src→dst with sun-avoidance)
  - dijkstra(src, dst) -> path, distance
  - in_same_region(pid1, pid2) -> bool
  - region_production(region_id, my_control) -> int
  - region_threat(region_id, enemy_ids) -> threat_speed

Region:
  - id: int (0-3)
  - my_planets: List[Planet]
  - enemy_planets: List[Planet]
  - external_planets: List[Planet]
  - center: (x, y)
  - production_rate: int (cumulative)
```

**Key Algorithms:**
- K-means clustering with 4 clusters (based on planet positions, spawn as seeds)
- Dijkstra preprocessing (avoid sun via arc-distance weighting)
- Threat evaluation (fastest enemy fleet ETA to region)

---

### Milestone 1b: Dynamic Target Scoring

**Replaced Functions:**
- `target_score()` → `target_value_in_region(snap, src, dst)`
  
**New Scoring Formula:**
```
base_value = production_of_target * turns_until_mine
path_factor = dijkstra_distance(src, dst)
regional_bonus = 2.0 if same_region else 0.5  # region cohesion
threat_penalty = enemy_eta_to_target / distance_ratio

final_score = base_value * regional_bonus 
              - 0.2 * path_factor 
              - threat_penalty
              - defense_cost_if_needed
```

**Implementation Changes:**
- Remove `approach_bonus()`, `approach_weight` (now in regional_bonus)
- Keep `prod_tier_bonus` but integrate with regional_bonus
- Simplify `eta_pen` (now just path_factor)

---

### Milestone 1c: Multi-Hop Planning

**New Classes:**
```python
MultiHopPlanner:
  - plan_attack_sequence(target, budget_turns=5)
    * Decompose path into hops (intermediate targets)
    * For each hop: calculate capture_need, schedule wave
    * Return: List[Wave] with timing + ship count + target

Wave:
  - target: Planet
  - required_ships: int
  - launch_turn: int
  - sources: List[Planet]  # which planets contribute
  - expected_arrival: int

ProductionTimeline:
  - predict_surplus(planet, turns_ahead) -> List[int]
  - can_support_wave(sources, required, launch_turn) -> bool
```

**Algorithm:**
```
for each target T in priority_order:
  path = regional_graph.dijkstra(my_region, T)
  
  # Decompose into hops (stop at friendly/neutral capture points)
  hops = [
    hop1: my_region → nearest_friendly/neutral,
    hop2: hop1 → next_hop,
    ...
    hopN: → T
  ]
  
  waves = []
  cumulative_time = 0
  for i, hop in enumerate(hops):
    need = capture_need(hop)
    available = production_timeline.predict(my_sources, cumulative_time)
    
    if available >= need * 1.1:  # 10% margin
      wave = Wave(hop, need, cumulative_time)
      waves.append(wave)
      cumulative_time += 3  # min 3 turns between waves
    else:
      break  # can't sustain this path
  
  if len(waves) >= path_length * 0.8:  # 80% completion rate
    add to capture_plans
```

---

### Milestone 1d: Safe Surplus Calculation

**New Function:**
```python
def calculate_safe_surplus(my_region, enemy_threats):
  """
  How many ships can I safely extract for offense?
  
  my_production - defensive_requirement = safe_surplus
  
  defensive_requirement = max_incoming_threat * safety_margin
  max_incoming_threat = fastest_enemy_eta_to_me * enemy_production
  safety_margin = 1.5 (keep 50% extra buffer)
  """
  
  my_prod = sum(p.production for p in my_region)
  
  threats = []
  for enemy_id in enemy_ids:
    enemy_region = regional_graph.get_region_by_owner(enemy_id)
    enemy_eta = regional_graph.dijkstra(enemy_region.center, my_region.center)[1]
    enemy_prod = sum(p.production for p in enemy_region)
    
    threat = enemy_prod / enemy_eta
    threats.append(threat)
  
  max_threat = max(threats) if threats else 0
  
  safe = int(my_prod * 0.65 - max_threat)
  return max(0, safe)
```

---

### Milestone 1e: Planner Integration

**DefensePlanner Update:**
- Input: regional_graph, threat_model
- Logic: defend high-value planets in my_region
- Add: reserve calculation based on safe_surplus

**ExpandPlanner Update:**
- Input: regional_graph, multi_hop_planner
- Logic: call multi_hop_planner for all targets
- Filter by safe_surplus availability

**AttackPlanner Update:**
- Input: regional_graph, game_theory_model
- Logic: prioritize high-production enemy regions
- Reserve based on safe_surplus

**InterceptPlanner:**
- No change (already tactical)

---

## v19.1: Game Theory Layer (Later)

**Not in scope for v19.0, but designed for:**
- ThreatModel class (minimax evaluation)
- Minimax region value (depth=2 lookahead)
- SafeSurplus refinement (account for counter-threat)

---

## v19.2: RL Integration (Later)

**Not in scope for v19.0, but designed for:**
- Self-play collection (v19.0 vs v19.0)
- Learnable parameters: regional_bonus, safe_surplus_ratio, multi_hop_discount
- PPO training loop

---

## v20+: Optimizations (Later)

- Path caching (dijkstra computed once per 10 turns)
- Threat prediction refinement
- RL-learned policy network

---

## Code Architecture

```
submission_v19.py
├─ [Core Physics] (unchanged)
│  ├─ fleet_speed()
│  ├─ lead_intercept()
│  └─ point_segment_distance()
├─ [Regional Graph] (NEW)
│  ├─ RegionalGraph class
│  ├─ Region dataclass
│  └─ dijkstra_precompute()
├─ [Target Evaluation] (REVISED)
│  ├─ target_value_in_region()
│  └─ calculate_safe_surplus()
├─ [Multi-Hop Planning] (NEW)
│  ├─ MultiHopPlanner class
│  ├─ Wave dataclass
│  └─ ProductionTimeline class
├─ [safe_aim] (SIMPLIFIED)
│  └─ safe_aim() [geometry only, no filtering]
├─ [Planners] (REVISED)
│  ├─ DefensePlanner [regional aware]
│  ├─ ExpandPlanner [multi-hop]
│  ├─ AttackPlanner [safe_surplus aware]
│  └─ InterceptPlanner [unchanged]
└─ [PlanArbiter] (REVISED)
   ├─ collect_strategic() [use multi_hop]
   ├─ score_with_modifiers() [use new scoring]
   └─ commit_best() [position gate]
```

---

## Testing Plan

### Phase 1: Smoke Tests
- RegionalGraph clustering: verify 4 regions identified correctly
- Dijkstra cache: compare paths vs direct computation
- target_value_in_region: verify regional_bonus applied

### Phase 2: Unit Tests
- MultiHopPlanner: test 1-hop, 2-hop, 3-hop plans
- ProductionTimeline: predict surplus accuracy
- calculate_safe_surplus: edge cases (no enemies, all enemies)

### Phase 3: Integration Tests
- v19.0 vs random (5 seeds): should win 100%
- v19.0 vs v17 (10 seeds): target >65% (expected 60-75%)
- aim_trainer (5 games): OOB <1%, sun <0.5%

### Phase 4: Analysis
- Visualize regional control over time
- Compare fleet paths (v19 vs v17)
- Measure production-to-offense ratio

---

## Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Dijkstra overhead | Precompute once at game start, cache |
| Multi-hop under-sending | Add 10% safety margin in ProductionTimeline |
| Regional mis-clustering | Use spawn points as seeds, validate manually |
| safe_surplus too conservative | Start with 0.65, tune if too defensive |
| Planner refactor breaks existing | Keep old target_score, shadow-test new one first |

---

## Success Criteria for v19.0

- [ ] Compiles and runs without errors
- [ ] Beats v17 at >60% win rate (10 seeds)
- [ ] OOB rate <1% (aim_trainer 5 games)
- [ ] Sun collision <0.5% (aim_trainer 5 games)
- [ ] No timeout in production games
- [ ] Regional control visible in replays (fleets path-optimized)
- [ ] Multi-hop sequences visible (wave timing)
- [ ] Documentation complete (this file + code comments)

---

## Timeline Estimate

- Milestone 1a (RegionalGraph): 2-3 days
- Milestone 1b (Dynamic Scoring): 1-2 days
- Milestone 1c (Multi-Hop): 2-3 days
- Milestone 1d (Safe Surplus): 1 day
- Milestone 1e (Planner Integration): 2-3 days
- Testing & Tuning: 2-3 days
- **Total: 10-15 days for production-ready v19.0**

---

## Future Directions (v19.1+)

- [ ] Game theory (minimax region value)
- [ ] Threat modeling (enemy counter-attack ETA)
- [ ] RL integration (self-play optimization)
- [ ] Path caching (dijkstra precomputation optimization)
- [ ] Comet prioritization (predict spawn + pre-position)
