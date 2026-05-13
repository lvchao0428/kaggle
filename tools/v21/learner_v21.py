"""PPO-style learner for v21 policy tiers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import msgpack
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v21.nets import build_net, best_device


def load_shards(runs_dir: Path) -> List[Dict]:
    games: List[Dict] = []
    for shard in sorted(runs_dir.glob("shard_w*.msgpack")):
        try:
            with open(shard, "rb") as f:
                payload = msgpack.unpack(f, raw=False)
            games.extend(payload.get("games", []))
            shard.unlink()
        except Exception as e:
            print(f"  failed to load {shard}: {e}", file=sys.stderr)
    return games


def games_to_tensors(
    games: List[Dict],
    device: torch.device,
    gamma: float = 0.997,
    lam: float = 0.95,
):
    feats: List[np.ndarray] = []
    all_plan_feats: List[np.ndarray] = []
    chosen_idxs: List[int] = []
    rets: List[float] = []
    advs: List[float] = []
    old_logp: List[float] = []
    plan_score_old: List[float] = []

    for g in games:
        outcome = float(g["outcome"])
        ts = g["transitions"]
        if not ts:
            continue
        tlen = len(ts)
        rewards = np.zeros(tlen, dtype=np.float32)
        for i, t in enumerate(ts):
            rewards[i] += float(t.get("shaped_reward", 0.0))
            rewards[i] += float(t.get("oob_penalty", 0.0))
        rewards[-1] += outcome

        values = np.array([t["value_pred"] for t in ts], dtype=np.float32)
        adv = np.zeros(tlen, dtype=np.float32)
        gae = 0.0
        next_v = 0.0
        for i in range(tlen - 1, -1, -1):
            delta = rewards[i] + gamma * next_v - values[i]
            gae = delta + gamma * lam * gae
            adv[i] = gae
            next_v = values[i]
        ret = adv + values

        for i, t in enumerate(ts):
            feats.append(np.asarray(t["obs_feat"], dtype=np.float32))
            rets.append(ret[i])
            advs.append(adv[i])
            old_logp.append(t["log_prob"])
            plan_score_old.append(t["plan_score_net"])
            chosen_idxs.append(int(t.get("chosen_idx", 0)))
            all_plan_feats.append(np.asarray(t["plan_feats"], dtype=np.float32))

    if not feats:
        return None

    feats_t = torch.tensor(np.stack(feats), device=device, dtype=torch.float32)
    rets_t = torch.tensor(rets, device=device, dtype=torch.float32)
    advs_t = torch.tensor(advs, device=device, dtype=torch.float32)
    old_logp_t = torch.tensor(old_logp, device=device, dtype=torch.float32)
    old_score_t = torch.tensor(plan_score_old, device=device, dtype=torch.float32)
    chosen_t = torch.tensor(chosen_idxs, device=device, dtype=torch.long)

    if advs_t.numel() > 1:
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-6)

    return feats_t, rets_t, advs_t, old_logp_t, old_score_t, chosen_t, all_plan_feats


def ppo_update(
    net: torch.nn.Module,
    opt: Adam,
    batch,
    clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.005,
    epochs: int = 4,
    minibatch: int = 4096,
) -> Dict[str, float]:
    feats, rets, advs, old_logp, old_score, chosen_idxs, all_plan_feats = batch
    n = feats.size(0)
    device = feats.device
    losses = []

    for ep in range(epochs):
        idx_perm = torch.randperm(n, device=device)
        for start in range(0, n, minibatch):
            mb = idx_perm[start : start + minibatch]
            mb_list = mb.tolist()

            policy_losses = []
            value_losses = []

            for global_i in mb_list:
                pf = torch.tensor(
                    all_plan_feats[global_i], device=device, dtype=torch.float32
                )
                k = pf.size(0)
                xk = pf.unsqueeze(0)
                v_pred, logits = net.forward_plans(xk)
                v_pred = v_pred.squeeze(-1)

                target = min(int(chosen_idxs[global_i].item()), k - 1)
                target_t = torch.tensor(target, device=device, dtype=torch.long)
                ce = F.cross_entropy(logits, target_t.unsqueeze(0))
                adv_val = float(advs[global_i].item())
                if adv_val < 0:
                    policy_losses.append(-adv_val * (-ce))
                else:
                    policy_losses.append(adv_val * ce)

                vi = v_pred.reshape(-1)
                ri = rets[global_i].reshape(-1).to(device)
                value_losses.append(F.mse_loss(vi, ri))

            if not policy_losses:
                continue

            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            entropy = -(old_score[mb] ** 2).mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(
                (
                    float(policy_loss.item()),
                    float(value_loss.item()),
                    float(entropy.item()),
                )
            )

    if not losses:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n": n}
    pl = float(np.mean([l[0] for l in losses]))
    vl = float(np.mean([l[1] for l in losses]))
    el = float(np.mean([l[2] for l in losses]))
    return {"policy_loss": pl, "value_loss": vl, "entropy": el, "n": n}


def save_checkpoint(net: torch.nn.Module, runs_dir: Path, tag: str) -> Path:
    ck = runs_dir / f"policy_{tag}.pth"
    torch.save(net.state_dict(), ck)
    latest = runs_dir / "policy_latest.pth"
    torch.save(net.state_dict(), latest)
    return ck


def main():
    ap = argparse.ArgumentParser(description="v21 PPO learner")
    ap.add_argument("--runs-dir", default="runs/v21_lite")
    ap.add_argument("--tier", default="lite", choices=["lite", "pro", "ultra"])
    ap.add_argument("--updates", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wait-secs", type=float, default=30.0)
    ap.add_argument("--ckpt-every", type=int, default=2)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    device = best_device()
    print(f"v21 learner device: {device}  tier={args.tier}")

    net = build_net(args.tier).to(device)
    opt = Adam(net.parameters(), lr=args.lr)

    latest = runs_dir / "policy_latest.pth"
    if latest.is_file():
        try:
            try:
                net.load_state_dict(torch.load(latest, map_location=device, weights_only=True))
            except TypeError:
                net.load_state_dict(torch.load(latest, map_location=device))
            print(f"  resumed {latest}")
        except Exception as e:
            print(f"  resume failed: {e}")

    for upd in range(args.updates):
        t0 = time.time()
        while True:
            games = load_shards(runs_dir)
            if games:
                break
            if time.time() - t0 > args.wait_secs:
                print(f"  no shards after {args.wait_secs}s, exiting.")
                return
            time.sleep(2.0)

        batch = games_to_tensors(games, device)
        if batch is None:
            continue
        stats = ppo_update(net, opt, batch)
        n_trans = stats["n"]
        meta = {
            "upd": upd + 1,
            "updates": args.updates,
            "tier": args.tier,
            "lr": args.lr,
            "games": len(games),
            "transitions": int(n_trans),
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            "entropy": stats["entropy"],
            "wall_s": time.time() - t0,
        }
        print(
            f"upd {meta['upd']}/{args.updates}  games={meta['games']}  "
            f"tr={n_trans}  pi={stats['policy_loss']:+.4f}  "
            f"v={stats['value_loss']:.4f}  ent={stats['entropy']:+.4f}"
        )
        with open(runs_dir / "last_learner_stats.json", "w") as f:
            json.dump(meta, f, indent=2)

        if (upd + 1) % args.ckpt_every == 0 or upd == args.updates - 1:
            ck = save_checkpoint(net, runs_dir, str(int(time.time())))
            print(f"  saved {ck.name}")


if __name__ == "__main__":
    main()
