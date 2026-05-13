"""GPU BatchedOrbitEnv rollouts + actor-critic update (reference training pipeline)."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vec_orbit import BatchedOrbitEnv
from vec_orbit.action_utils import raw_vec_to_actions
from vec_orbit.policy import ActorCritic


def _dist_entropy(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Per-batch sum of entropies over action dims (independent Normals)."""
    if std.dim() == 1:
        std = std.view(1, -1).expand_as(mean)
    ent = 0.5 * (1.0 + torch.log(2.0 * math.pi * std * std + 1e-12))
    return ent.sum(dim=-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--horizon", type=int, default=64, help="env steps per optimizer update")
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--planets", type=int, default=12)
    p.add_argument("--fleets", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-every", type=int, default=1, help="print every N updates (0 = only first and last)")
    p.add_argument("--out", type=Path, default=Path("runs/vec_orbit/policy_actor_critic.pth"))
    args = p.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    env = BatchedOrbitEnv(
        batch=args.batch,
        max_planets=args.planets,
        max_fleets=args.fleets,
        max_steps=args.max_steps,
        device=dev,
    )
    net = ActorCritic(env.obs_dim, hidden=args.hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log_cfg = args.log_every

    print(
        "vec_orbit.train_loop | Columns: loss total | L_pi policy term | L_v value MSE sum over horizon | "
        "|gn| grad norm after clip | R_mean/std MC return end of window | V_last critic at last step | "
        "adv_mean mean advantage | H_mean policy entropy | term non-zero sparse rewards in window | "
        "win0/win1 envs with +1/-1 cumulative in window | done_f frac done | |R|/B mean abs terminal | "
        "upd_s wall per update | env_step/s B*H/upd_s | log-every=0 → first+last only",
        flush=True,
    )
    print(f"device={dev} B={args.batch} H={args.horizon} P={args.planets} F={args.fleets} lr={args.lr}", flush=True)

    for u in range(1, args.updates + 1):
        t_up0 = time.perf_counter()
        obs = env.reset()
        logps = []
        vals = []
        rewards = []
        masks = []
        entropies = []

        for _ in range(args.horizon):
            mean, std, val = net.forward(obs)
            dist = torch.distributions.Normal(mean, std)
            pre = dist.rsample()
            logp = dist.log_prob(pre).sum(-1)
            entropies.append(_dist_entropy(mean, std).mean())

            actions = raw_vec_to_actions(pre, env.P)
            next_obs, rew, done, _ = env.step(actions)
            m = (~done).float()
            logps.append(logp)
            vals.append(val)
            rewards.append(rew)
            masks.append(m)
            obs = next_obs

        R = torch.zeros(args.batch, device=dev)
        loss_pi = torch.zeros((), device=dev)
        loss_v = torch.zeros((), device=dev)
        adv_sum = torch.zeros((), device=dev)
        for t in reversed(range(args.horizon)):
            R = rewards[t] + args.gamma * R * masks[t]
            adv = R - vals[t]
            adv_sum = adv_sum + adv.mean()
            loss_pi = loss_pi - (logps[t] * adv.detach()).mean()
            loss_v = loss_v + F.mse_loss(vals[t], R)

        adv_mean = adv_sum / args.horizon
        loss = loss_pi + 0.5 * loss_v
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        dt_up = time.perf_counter() - t_up0
        env_steps = args.horizon * args.batch
        eps = env_steps / max(dt_up, 1e-9)

        with torch.no_grad():
            r_mat = torch.stack(rewards, dim=0)
            term_hits = (r_mat.abs() > 0.5).float().sum().item()
            r_acc = r_mat.sum(dim=0)
            n_p0_win = (r_acc > 0.5).sum().item()
            n_p1_win = (r_acc < -0.5).sum().item()
            done_final = env.done.float().mean().item()
            R_mean = R.mean().item()
            R_std = R.std().item()
            v_pred_last = vals[-1].mean().item()
            rew_per_env = r_acc.abs().sum().item() / max(args.batch, 1)
            ent_mean = sum(entropies) / len(entropies)

        should_log = False
        if log_cfg == 0:
            should_log = u == 1 or u == args.updates
        elif log_cfg > 0:
            should_log = u == 1 or u == args.updates or (u % log_cfg == 0)

        if should_log:
            elapsed = time.perf_counter() - t0
            line = (
                f"[{u:5d}/{args.updates}] "
                f"loss={loss.item():8.4f} "
                f"L_pi={loss_pi.item():8.4f} "
                f"L_v={loss_v.item():8.4f} "
                f"|gn|={float(grad_norm):.3f} "
                f"| "
                f"R_mean={R_mean:+.4f} R_std={R_std:.4f} "
                f"V_last={v_pred_last:+.4f} "
                f"adv_mean={adv_mean.item():+.4f} "
                f"| "
                f"H_mean={ent_mean:.3f} "
                f"| "
                f"term={int(term_hits)} "
                f"win0={n_p0_win} win1={n_p1_win} "
                f"done_f={done_final:.2f} "
                f"|R|/B={rew_per_env:.4f} "
                f"| "
                f"upd_s={dt_up:5.3f}s "
                f"env_step/s={eps:,.0f} "
                f"cum_s={elapsed:7.1f}s"
            )
            print(line)
            if dev.type == "cuda" and (u == 1 or u % 50 == 0 or u == args.updates):
                mib = torch.cuda.max_memory_allocated(dev) / (1024**2)
                print(f"         cuda_peak_alloc_MiB={mib:,.1f}")

    torch.save(net.state_dict(), args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
