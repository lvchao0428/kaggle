"""Batched two-player Orbit-like env; all dynamics on a single torch device."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

import vec_orbit.geom_torch as gt
from vec_orbit.layouts import (
    PLAYER0,
    PLAYER1,
    OWNER_NEUTRAL,
    layouts_to_torch,
    sample_planets_cpu,
)

PLANET_RADIUS = 0.85
INVALID_DEST = -1


class BatchedOrbitEnv:
    """
    Vectorized micro-step simulator:
      - B environments, P planet slots, F in-flight fleet slots.
      - Planets fixed in space for the episode; production each step.
      - Fleets: straight-line toward target; OOB / sun kills; arrival battle.
    """

    def __init__(
        self,
        *,
        batch: int,
        max_planets: int = 12,
        max_fleets: int = 32,
        max_steps: int = 512,
        device: Optional[torch.device] = None,
    ):
        self.batch = batch
        self.P = max_planets
        self.F = max_fleets
        self.max_steps = max_steps
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.px = torch.zeros(batch, max_planets, device=device)
        self.py = torch.zeros(batch, max_planets, device=device)
        self.owner = torch.full((batch, max_planets), -1, dtype=torch.long, device=device)
        self.ships = torch.zeros(batch, max_planets, device=device)
        self.growth = torch.zeros(batch, max_planets, dtype=torch.long, device=device)
        self.valid = torch.zeros(batch, max_planets, dtype=torch.bool, device=device)

        self.fx = torch.zeros(batch, max_fleets, device=device)
        self.fy = torch.zeros(batch, max_fleets, device=device)
        self.fvx = torch.zeros(batch, max_fleets, device=device)
        self.fvy = torch.zeros(batch, max_fleets, device=device)
        self.fships = torch.zeros(batch, max_fleets, device=device)
        self.fowner = torch.zeros(batch, max_fleets, dtype=torch.long, device=device)
        self.fdest = torch.full((batch, max_fleets), INVALID_DEST, dtype=torch.long, device=device)
        self.f_active = torch.zeros(batch, max_fleets, dtype=torch.bool, device=device)

        self.step_count = torch.zeros(batch, dtype=torch.long, device=device)
        self.done = torch.zeros(batch, dtype=torch.bool, device=device)
        self.obs_dim = max_planets * 4 + 6

    def reset(self, seeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """seeds: (B,) long on CPU or GPU; if None, random CPU seeds."""
        if seeds is None:
            seeds_cpu = np.random.randint(0, 2**31 - 1, size=(self.batch,), dtype=np.int64)
        else:
            seeds_cpu = seeds.detach().cpu().numpy().astype(np.int64, copy=False)

        px, py, own, sh, gr, va = sample_planets_cpu(
            batch=self.batch,
            max_planets=self.P,
            seeds=seeds_cpu,
        )
        self.px, self.py, self.owner, self.ships, self.growth, self.valid = layouts_to_torch(
            px, py, own, sh, gr, va, self.device
        )
        self.ships = self.ships.float()

        self.fx.zero_()
        self.fy.zero_()
        self.fvx.zero_()
        self.fvy.zero_()
        self.fships.zero_()
        self.fowner.zero_()
        self.fdest.fill_(INVALID_DEST)
        self.f_active.zero_()

        self.step_count.zero_()
        self.done.zero_()
        return self._obs()

    def _obs(self) -> torch.Tensor:
        B, P = self.batch, self.P
        om = torch.where(
            self.valid,
            self.owner.float() / 2.0,
            torch.full_like(self.owner, -0.5, dtype=torch.float32),
        )
        block = torch.stack(
            [
                self.px / gt.BOARD,
                self.py / gt.BOARD,
                self.ships.clamp(max=500.0) / 500.0,
                om,
            ],
            dim=-1,
        )
        flat = block.reshape(B, -1)

        step_n = (self.step_count.float() / float(self.max_steps)).unsqueeze(1)
        af0 = (self.f_active & (self.fowner == PLAYER0)).float().sum(dim=1, keepdim=True) / float(self.F)
        af1 = (self.f_active & (self.fowner == PLAYER1)).float().sum(dim=1, keepdim=True) / float(self.F)

        mask_p0 = (self.owner == PLAYER0) & self.valid
        mask_p1 = (self.owner == PLAYER1) & self.valid
        mask_n = (self.owner == OWNER_NEUTRAL) & self.valid
        tot0 = (self.ships * mask_p0.float()).sum(dim=1, keepdim=True) / 500.0
        tot1 = (self.ships * mask_p1.float()).sum(dim=1, keepdim=True) / 500.0
        totn = (self.ships * mask_n.float()).sum(dim=1, keepdim=True) / 500.0

        g = torch.cat([step_n, af0, af1, tot0, tot1, totn], dim=1)
        return torch.cat([flat, g], dim=1)

    def _first_free_fleet(self) -> Tuple[torch.Tensor, torch.Tensor]:
        free = ~self.f_active
        has = free.any(dim=1)
        slot = free.long().argmax(dim=1)
        return has, slot

    def _try_launch(
        self,
        player_id: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        frac: torch.Tensor,
    ) -> None:
        B = self.batch
        b = torch.arange(B, device=self.device)
        has, slot = self._first_free_fleet()

        own = self.owner[b, src]
        sh = self.ships[b, src].long()
        ok = (
            has
            & ~self.done
            & self.valid[b, src]
            & self.valid[b, dst]
            & (src != dst)
            & (own == player_id)
            & (frac > 0.02)
            & (sh > 0)
        )

        send = (self.ships[b, src] * frac).long().clamp(min=0)
        send = torch.where((frac > 0.02) & (send == 0) & (sh > 0), torch.ones_like(send), send)
        send = torch.minimum(send, sh)
        ok = ok & (send > 0)

        sx = self.px[b, src]
        sy = self.py[b, src]
        tx = self.px[b, dst]
        ty = self.py[b, dst]
        dx = tx - sx
        dy = ty - sy
        dist = torch.sqrt(dx * dx + dy * dy + 1e-9)
        spd = gt.fleet_speed(send.float())
        ux = dx / dist * spd
        uy = dy / dist * spd

        if not ok.any():
            return

        sl = slot[ok]
        bm = b[ok]
        self.ships[bm, src[ok]] = self.ships[bm, src[ok]] - send[ok].float()
        off = PLANET_RADIUS * 1.1
        self.fx[bm, sl] = sx[ok] + (dx[ok] / dist[ok]) * off
        self.fy[bm, sl] = sy[ok] + (dy[ok] / dist[ok]) * off
        self.fvx[bm, sl] = ux[ok]
        self.fvy[bm, sl] = uy[ok]
        self.fships[bm, sl] = send[ok].float()
        self.fowner[bm, sl] = player_id
        self.fdest[bm, sl] = dst[ok]
        self.f_active[bm, sl] = True

    def _resolve_one_arrival(self, f: int, arrive: torch.Tensor, dest: torch.Tensor) -> None:
        if not arrive.any():
            return
        B = self.batch
        b = torch.arange(B, device=self.device)
        fo = self.fowner[:, f]
        fs = self.fships[:, f]
        po = self.owner[b, dest]

        same = arrive & (fo == po)
        if same.any():
            ix = b[same]
            di = dest[same]
            self.ships[ix, di] = self.ships[ix, di] + fs[same]
            self.f_active[ix, f] = False

        pending = arrive & self.f_active[:, f]
        if not pending.any():
            return
        bm = b[pending]
        d = dest[pending]
        pfo = fo[pending]
        pfs = fs[pending].long()
        ppo = po[pending]
        pps = self.ships[bm, d].long()

        neu = ppo == OWNER_NEUTRAL
        if neu.any():
            ix = bm[neu]
            di = d[neu]
            self.owner[ix, di] = pfo[neu].to(dtype=self.owner.dtype)
            self.ships[ix, di] = pfs[neu].float()
            self.f_active[ix, f] = False

        pending2 = arrive & self.f_active[:, f]
        if not pending2.any():
            return
        bm2 = b[pending2]
        d2 = dest[pending2]
        pfo2 = fo[pending2]
        pfs2 = fs[pending2].long()
        ppo2 = po[pending2]
        pps2 = self.ships[bm2, d2].long()

        enemy = ((pfo2 == PLAYER0) & (ppo2 == PLAYER1)) | ((pfo2 == PLAYER1) & (ppo2 == PLAYER0))
        if not enemy.any():
            return
        ix = bm2[enemy]
        di = d2[enemy]
        atk = pfs2[enemy]
        def_ = pps2[enemy]
        fo_win = pfo2[enemy]
        win = atk > def_
        if win.any():
            self.owner[ix[win], di[win]] = fo_win[win].to(dtype=self.owner.dtype)
            self.ships[ix[win], di[win]] = (atk[win] - def_[win]).float().clamp(min=1.0)
        lose = ~win
        if lose.any():
            self.ships[ix[lose], di[lose]] = (def_[lose] - atk[lose]).float().clamp(min=0.0)
        self.f_active[ix, f] = False

    def _fleet_substep(self) -> None:
        B, P, Fd = self.batch, self.P, self.F
        pr2 = (PLANET_RADIUS * 1.05) ** 2
        b = torch.arange(B, device=self.device)

        for f in range(Fd):
            nx = self.fx[:, f] + self.fvx[:, f]
            ny = self.fy[:, f] + self.fvy[:, f]
            ax, ay = self.fx[:, f], self.fy[:, f]

            seg_sun = gt.segment_hits_sun(ax, ay, nx, ny)
            oob = gt.out_of_bounds_xy(nx, ny)
            bad = seg_sun | oob

            act = self.f_active[:, f] & ~self.done
            dest = self.fdest[:, f].clamp(min=0, max=P - 1)

            px_d = self.px[b, dest]
            py_d = self.py[b, dest]
            dxp = nx - px_d
            dyp = ny - py_d
            dist2 = dxp * dxp + dyp * dyp

            arrive = act & ~bad & (dist2 <= pr2)
            move_ok = act & ~bad & ~arrive

            self.fx[:, f] = torch.where(move_ok, nx, self.fx[:, f])
            self.fy[:, f] = torch.where(move_ok, ny, self.fy[:, f])

            self.f_active[:, f] = self.f_active[:, f] & ~bad

            if arrive.any():
                self._resolve_one_arrival(f, arrive, dest)

    def _production(self) -> None:
        g = self.growth.float()
        self.ships = self.ships + g * self.valid.float()

    def _terminal_reward(self) -> torch.Tensor:
        """Sparse reward for player 0 when episode ends."""
        B = self.batch
        prev = self.done.clone()

        mask_p0 = (self.owner == PLAYER0) & self.valid
        mask_p1 = (self.owner == PLAYER1) & self.valid
        ships0 = (self.ships * mask_p0.float()).sum(dim=1)
        ships1 = (self.ships * mask_p1.float()).sum(dim=1)
        f0 = (self.f_active & (self.fowner == PLAYER0)).float() * self.fships
        f1 = (self.f_active & (self.fowner == PLAYER1)).float() * self.fships
        n0 = (mask_p0.any(dim=1)) | (f0.sum(dim=1) > 1e-6)
        n1 = (mask_p1.any(dim=1)) | (f1.sum(dim=1) > 1e-6)

        timeout = self.step_count >= self.max_steps
        over = ~(n0 & n1) | timeout

        newly = over & ~prev
        rew = torch.zeros(B, device=self.device)
        p0_win = newly & n0 & ~n1
        p1_win = newly & n1 & ~n0
        rew = rew + p0_win.float() * 1.0
        rew = rew - p1_win.float() * 1.0
        self.done = self.done | over
        return rew

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        actions: (B, 2, 3) float — per player [src_idx, dst_idx, frac in (0,1)].
        """
        B = self.batch
        if self.done.all():
            return self._obs(), torch.zeros(B, device=self.device), self.done.clone(), {}

        active = ~self.done
        self.step_count = torch.where(active, self.step_count + 1, self.step_count)

        src0 = actions[:, 0, 0].long().clamp(0, self.P - 1)
        dst0 = actions[:, 0, 1].long().clamp(0, self.P - 1)
        fr0 = actions[:, 0, 2].clamp(0.0, 1.0)
        src1 = actions[:, 1, 0].long().clamp(0, self.P - 1)
        dst1 = actions[:, 1, 1].long().clamp(0, self.P - 1)
        fr1 = actions[:, 1, 2].clamp(0.0, 1.0)

        self._try_launch(PLAYER0, src0, dst0, fr0)
        self._try_launch(PLAYER1, src1, dst1, fr1)
        self._fleet_substep()
        self._production()
        rew = self._terminal_reward()
        return self._obs(), rew, self.done.clone(), {}
