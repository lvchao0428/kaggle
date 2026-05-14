from __future__ import annotations

from orbit_submit.constants import *
from orbit_submit.entities import Planet
from orbit_submit.game_state import GameState
from orbit_submit.kinematics import is_sun_belt_planet
from orbit_submit.regional import RegionalGraph
from orbit_submit.scoring_early import enemy_eta_power

@dataclass
class Snapshot:
    """Per-turn precomputed view. Planners read from here; they never recompute
    surplus / reserve themselves. Mutate by calling `subtract(pid, ships)`
    AFTER an action commits."""

    state: GameState
    policy: "PhasePolicy"

    surplus: Dict[int, int] = field(default_factory=dict)
    reserve: Dict[int, int] = field(default_factory=dict)
    centroid: Tuple[float, float] = (SUN_X, SUN_Y)
    nearest_enemy_dist: Dict[int, float] = field(default_factory=dict)
    used: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    # v13: predicted earliest enemy fleet arrival eta per friendly planet id.
    # Only includes planets where an enemy fleet is in-flight and will arrive
    # within THREAT_HORIZON_WINDOW turns.
    threat_horizon: Dict[int, int] = field(default_factory=dict)
    # Same-turn commits toward grey id (obs does not yet list our launched fleets).
    pending_neutral_wave: Dict[int, int] = field(default_factory=dict)

    @classmethod
    def build(cls, state: GameState, policy: "PhasePolicy") -> "Snapshot":
        snap = cls(state=state, policy=policy)
        snap.centroid = state.centroid()
        # v13: build threat_horizon first so _reserve can read it.
        # v14: expanded window to 8 turns (was 3) for earlier reinforcement.
        for f in state.fleets:
            if f.owner in (-1, state.my_id):
                continue
            target = state.fleet_target.get(f.id)
            if not target:
                continue
            tid, eta_to_planet = target
            dst = state.get(tid)
            if dst is None or dst.owner != state.my_id:
                continue
            prev = snap.threat_horizon.get(dst.id, 999)
            if eta_to_planet < prev:
                snap.threat_horizon[dst.id] = eta_to_planet
        for p in state.my_pl:
            snap.nearest_enemy_dist[p.id] = (
                min((p.dist(e) for e in state.en_pl), default=999.0))
            snap.reserve[p.id] = snap._reserve(p)
            snap.surplus[p.id] = max(0, p.ships - snap.reserve[p.id])
        return snap

    def _reserve(self, p: Planet) -> int:
        s = self.state
        threat = max(0, s.net_threat(p))
        if p.is_comet:
            ttl = s.comet_turns_left(p)
            return max(threat + 2, 2 if ttl > 10 else p.ships)
        ned = self.nearest_enemy_dist.get(p.id, 999.0)
        # v20: early phase — relax front_lock so nearby neutrals can be grabbed.
        # Only lock hard when enemy is very close (< 20) or threat is active.
        #
        # FFA: orbiting neighbours often sit ~25–48 away (not inside <36 dual lock);
        # add a wider band so border worlds retain ships vs stationary threats.
        if self.state.phase() == "early":
            if s.is_ffa_mode():
                if ned < 20:
                    front_lock = 8 + p.production * 3
                elif ned < 44:
                    front_lock = 4 + p.production * 2
                else:
                    front_lock = 0
            else:
                front_lock = (8 + p.production * 3) if ned < 20 else 0
        else:
            if s.is_ffa_mode():
                if ned < 22:
                    front_lock = 10 + p.production * 4
                elif ned < 56:
                    front_lock = 6 + p.production * 3
                else:
                    front_lock = 0
            else:
                front_lock = ((10 + p.production * 4) if ned < 20
                              else (5 + p.production * 3) if ned < 36 else 0)
        growth_lock = p.production * self.policy.reserve_growth_mul
        floor_threat_pad = 4 if self.state.phase() == "early" else 6
        floor_idle = 2 if self.state.phase() == "early" else 3
        base = max(threat + floor_threat_pad, growth_lock, front_lock, floor_idle)
        # v13/v14: if an enemy fleet will arrive within 8 turns AND this planet
        # has high production, keep extra buffer to defend it proactively.
        # Window expanded from 3 to 8 in v14 for earlier reinforcement.
        horizon_eta = self.threat_horizon.get(p.id, 999)
        if horizon_eta <= 8 and p.production >= 3:
            # Enemy fleet targeting us soon — keep extra pad, but do NOT lock the
            # entire stack (old ``ships + prod*eta`` made reserve > ships → surplus 0).
            proactive = (threat + p.production * min(horizon_eta, 4)
                         + 6 + max(0, 10 - horizon_eta))
            base = max(base, proactive)

        # FFA: stationary enemy arc — ``net_threat`` may stay 0; hoard on this world so
        # strategic layers cannot peel it toward distant targets every turn.
        if s.is_ffa_mode() and s.en_pl and not p.is_comet:
            ff_r_hold = 60.0
            max_pr = 0.0
            for e in s.en_pl:
                if p.dist(e) >= ff_r_hold:
                    continue
                pr = float(s.effective_garrison(e)) + float(e.production) * 6.0
                max_pr = max(max_pr, pr)
            if max_pr >= 22.0:
                cap_extra = max(0, int(p.ships) - ABS_MIN_BATCH - 1)
                if cap_extra > 0:
                    stationary_hold = int(max_pr * 0.46 + float(p.production) * 3.5 + 12.0)
                    stationary_hold = min(cap_extra, stationary_hold)
                    base = max(base, threat + stationary_hold)

        # Inner sun belt: multi-planet — hoard before rotation carries us into enemy.
        if (
            len(s.my_pl) >= 2
            and is_sun_belt_planet(s, p)
            and not p.is_comet
        ):
            en_belt = any(is_sun_belt_planet(s, e) for e in s.en_pl)
            if en_belt or s.phase() != "early":
                belt_pad = 8 + p.production * 5
                if s.phase() in ("mid", "late"):
                    belt_pad += 8
                base = max(base, threat + belt_pad)

        # Thin uncontested worlds: loosen hoarding so staging rocks can consolidate
        # fleets for one-shot clears (otherwise many low-ship piles waste production tempo).
        if (
            threat == 0
            and not p.is_comet
            and ned >= 56.0
            and len(s.my_pl) >= 2
            and int(p.ships) <= 72
            and int(p.ships) > ABS_MIN_BATCH
        ):
            peel_floor = ABS_MIN_BATCH * 2 + int(p.production) * 3
            if int(p.ships) <= peel_floor + ABS_MIN_BATCH * 6:
                cap_res = max(
                    threat + floor_threat_pad + 3,
                    int(p.ships) - ABS_MIN_BATCH * 2,
                )
                base = min(base, cap_res)

        if (
            self.state.phase() == "early"
            and s.step <= OPENING_SOLO_HQ_RESERVE_LAST_STEP
            and len(s.my_pl) == 1
            and threat == 0
            and not p.is_comet
        ):
            peel = OPENING_FIRST_CAPTURE_SEND
            if p.ships >= peel:
                horizon_eta = self.threat_horizon.get(p.id, 999)
                if horizon_eta <= 8 and p.production >= 3:
                    # Enemy fleet actually inbound: keep slack vs growth_lock / peel math.
                    base = min(base, max(threat + 2, p.ships - peel))
                else:
                    # Race first factory: do not let growth_lock sit on 3–4 ships while
                    # 22-on-HQ cannot peel 21 (user: same turn as screenshot 2).
                    base = min(base, threat + 1)
        return base

    def avail(self, pid: int) -> int:
        """Surplus minus already-used in this turn."""
        return max(0, self.surplus.get(pid, 0) - self.used.get(pid, 0))

    def subtract(self, pid: int, ships: int) -> None:
        self.used[pid] += int(ships)

    def calculate_safe_surplus_v20(self, regional_graph: Optional[RegionalGraph] = None) -> int:
        """Ship budget for strategic commits from current per-planet avail pools."""
        avail_total = sum(self.avail(p.id) for p in self.state.my_pl)
        if avail_total <= ABS_MIN_BATCH:
            return 0
        if not regional_graph:
            return max(ABS_MIN_BATCH, int(avail_total * 0.65))

        threats = [
            regional_graph.region_threat(rid, self.state.en_pl)
            for rid in regional_graph.regions
        ]
        max_t = max(threats) if threats else 0.0
        my_prod = float(sum(p.production for p in self.state.my_pl))
        pressure = max_t / max(my_prod, 1.0)
        pressure = min(1.35, pressure)
        alloc_frac = max(0.45, min(0.82, 0.82 - pressure * 0.28))
        return max(ABS_MIN_BATCH, int(avail_total * alloc_frac))

    def is_safe_investment(self, dst: Planet, eta: int) -> bool:
        """Bocsimacko `safe-to-invest-p` port (player.lisp:1079).

        Returns True unless this looks like a clearly losing trade:
          - Enemy is much closer AND has overwhelming nearby firepower.
          - Or our friendly planets are under net inbound threat that
            already exceeds our total surplus (i.e. we cannot afford to
            send anything outward without losing a homeworld).

        Conservative - only filters obvious losers, not borderline cases.
        Original Lisp version uses time-vector arrivals; we approximate.
        """
        s = self.state
        # Defensive triage: aggregate friendly inbound threat and surplus.
        net_threat = sum(max(0, s.net_threat(p)) for p in s.my_pl)
        my_total_surplus = sum(self.surplus.values())
        if net_threat > my_total_surplus * 1.10 and my_total_surplus > 0:
            # We are already underwater on defense - no time for expansion.
            return False
        # Enemy proximity / power dominance check.
        my_reach = min((m.dist(dst) for m in s.my_pl), default=999.0)
        en_reach = min((e.dist(dst) for e in s.en_pl), default=999.0)
        early = s.phase() == "early"
        reach_factor = 0.56 if early else 0.70
        power_factor = 1.72 if early else 1.60
        if en_reach < my_reach * reach_factor:
            en_local = sum(e.ships for e in s.en_pl
                           if e.dist(dst) < my_reach * 1.20)
            if en_local > (dst.ships + dst.production * eta) * power_factor:
                return False

        # FFA：远征去抢「敌军弧内」大肉（敌明显更近），本地火力环已盖住需求 → 不参与投资
        # （避免 ~step25 从角落扑向蓝方门口 65@+3）
        if (
            dst.owner == -1
            and s.is_ffa_mode()
            and s.en_pl
            and eta >= 10
        ):
            my_d_close = min((m.dist(dst) for m in s.my_pl), default=999.0)
            en_d_close = min((e.dist(dst) for e in s.en_pl), default=999.0)
            if my_d_close > en_d_close + 14:
                horizon = dst.ships + dst.production * min(max(int(eta), 1), 22) // 2 + 26
                en_mass_ring = sum(
                    float(ep.ships)
                    for ep in s.en_pl
                    if ep.dist(dst) < min(72.0, my_d_close + 26.0)
                )
                e_eta_best, _ = enemy_eta_power(s, dst)
                contested = (
                    en_mass_ring >= float(horizon) * 0.84
                    and int(eta) + 4 >= int(e_eta_best)
                )
                greedy_target = dst.production >= 3 or dst.ships >= 52
                if contested and greedy_target:
                    return False
        return True

