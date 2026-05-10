"""PPO learner for Orbit Wars RL.

Reads msgpack shards produced by `rollout_worker.py`, runs PPO updates on
the small PolicyValueNet, and periodically dumps:
- `runs/<exp>/policy_<step>.pth` (full state_dict)
- `runs/<exp>/policy_latest.npz` (numpy weights for rollout workers)

Phase 2 change: `games_to_tensors` now mixes per-step shaped rewards
(from rollout_worker) with the terminal outcome before GAE, giving the
policy denser gradient signal.

Phase 3 change: policy loss is now cross-entropy over the plan index
rather than a tanh-delta PPO surrogate, which is a cleaner signal for
the plan-ranking head.

Usage::

    python3.12 tools/learner.py --runs-dir runs/exp1 --updates 50

The learner consumes ALL shards in the directory at startup (and
re-consumes new ones each iteration). It deletes shards after consuming so
the directory doesn't grow unboundedly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import msgpack
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.policy_torch import PolicyValueNet, best_device


def load_shards(runs_dir: Path) -> List[Dict]:
    """Returns list of game dicts; deletes consumed shard files."""
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


def games_to_tensors(games: List[Dict], device: torch.device,
                     gamma: float = 0.997, lam: float = 0.95,
                     shaped_reward_gamma: float = 0.10):
    """Flatten game transitions to tensors for PPO.

    Phase 2: reward at each step is
        r_t = shaped_reward_t * shaped_reward_gamma + terminal_r (last step only)

    Phase 3: also collect plan_feats (K, F) and chosen_idx for cross-entropy
    policy loss.  Because K can vary per step, we return flat lists and
    reconstruct per-step indexing.
    """
    feats: List[np.ndarray] = []
    all_plan_feats: List[np.ndarray] = []   # list of (K, F) arrays, one per step
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
        T = len(ts)
        # Build per-step rewards: shaped + terminal at last step.
        rewards = np.zeros(T, dtype=np.float32)
        for i, t in enumerate(ts):
            rewards[i] += float(t.get("shaped_reward", 0.0))
            rewards[i] += float(t.get("oob_penalty", 0.0))
        rewards[-1] += outcome  # terminal signal

        values = np.array([t["value_pred"] for t in ts], dtype=np.float32)
        # GAE.
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_v = 0.0
        for i in range(T - 1, -1, -1):
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
            pf = np.asarray(t["plan_feats"], dtype=np.float32)  # (K, F)
            all_plan_feats.append(pf)

    if not feats:
        return None

    feats_t = torch.tensor(np.stack(feats), device=device, dtype=torch.float32)
    rets_t = torch.tensor(rets, device=device, dtype=torch.float32)
    advs_t = torch.tensor(advs, device=device, dtype=torch.float32)
    old_logp_t = torch.tensor(old_logp, device=device, dtype=torch.float32)
    old_score_t = torch.tensor(plan_score_old, device=device, dtype=torch.float32)
    chosen_t = torch.tensor(chosen_idxs, device=device, dtype=torch.long)

    # Standardise advantages.
    if advs_t.numel() > 1:
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-6)

    return feats_t, rets_t, advs_t, old_logp_t, old_score_t, chosen_t, all_plan_feats


def ppo_update(net: PolicyValueNet, opt: Adam, batch,
               clip: float = 0.2, value_coef: float = 0.5,
               entropy_coef: float = 0.005, epochs: int = 4,
               minibatch: int = 4096) -> Dict[str, float]:
    """Phase 3 update: cross-entropy policy loss over plan index.

    For each step we have (plan_feats: K x F, chosen_idx, advantage).
    We score every candidate plan through net.plan_score(), then compute
    cross-entropy between the softmax distribution and the chosen index,
    weighted by the advantage (REINFORCE-style, clipped like PPO).
    """
    feats, rets, advs, old_logp, old_score, chosen_idxs, all_plan_feats = batch
    n = feats.size(0)
    device = feats.device
    losses = []

    for ep in range(epochs):
        idx_perm = torch.randperm(n, device=device)
        for start in range(0, n, minibatch):
            mb = idx_perm[start:start + minibatch]
            mb_list = mb.tolist()

            # Value loss (uses obs_feat of chosen plan).
            v_pred, _ = net(feats[mb])
            v_pred = v_pred.squeeze(-1)
            value_loss = F.mse_loss(v_pred, rets[mb])

            # Policy loss: cross-entropy over plan candidates.
            policy_losses = []
            for local_i, global_i in enumerate(mb_list):
                pf = torch.tensor(
                    all_plan_feats[global_i], device=device, dtype=torch.float32
                )  # (K, F)
                K = pf.size(0)
                _, plan_scores = net(pf)           # (K, 1)
                logits = plan_scores.squeeze(-1)   # (K,)
                target = min(int(chosen_idxs[global_i].item()), K - 1)
                target_t = torch.tensor(target, device=device, dtype=torch.long)
                # Advantage-weighted CE: multiply loss by |adv|, sign handled by detach.
                adv_val = float(advs[global_i].item())
                ce = F.cross_entropy(logits.unsqueeze(0), target_t.unsqueeze(0))
                # For negative advantage, flip: we want to push AWAY from chosen.
                if adv_val < 0:
                    # -adv * ce pushes log-prob of chosen plan down
                    policy_losses.append(-adv_val * (-ce))
                else:
                    policy_losses.append(adv_val * ce)

            if not policy_losses:
                continue
            policy_loss = torch.stack(policy_losses).mean()
            entropy = -(old_score[mb] ** 2).mean()  # encourage diverse plan scoring

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append((float(policy_loss.item()), float(value_loss.item()),
                           float(entropy.item())))

    if not losses:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n": n}
    pl = float(np.mean([l[0] for l in losses]))
    vl = float(np.mean([l[1] for l in losses]))
    el = float(np.mean([l[2] for l in losses]))
    return {"policy_loss": pl, "value_loss": vl, "entropy": el, "n": n}


def save_npz(net: PolicyValueNet, path: Path):
    sd = net.state_dict()
    np.savez(
        path,
        W1=sd["trunk.0.weight"].detach().cpu().numpy(),
        b1=sd["trunk.0.bias"].detach().cpu().numpy(),
        W2=sd["trunk.2.weight"].detach().cpu().numpy(),
        b2=sd["trunk.2.bias"].detach().cpu().numpy(),
        Wv=sd["value_head.weight"].detach().cpu().numpy(),
        bv=sd["value_head.bias"].detach().cpu().numpy(),
        Wp=sd["plan_head.weight"].detach().cpu().numpy(),
        bp=sd["plan_head.bias"].detach().cpu().numpy(),
    )


def main():
    ap = argparse.ArgumentParser(description="Orbit Wars PPO learner")
    ap.add_argument("--runs-dir", default="runs/exp1")
    ap.add_argument("--updates", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wait-secs", type=float, default=30.0,
                    help="Seconds to wait for new shards if none present.")
    ap.add_argument("--ckpt-every", type=int, default=2)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    device = best_device()
    print(f"learner device: {device}")
    net = PolicyValueNet().to(device)
    opt = Adam(net.parameters(), lr=args.lr)

    # Resume if we have an existing checkpoint.
    last_ckpt = sorted(runs_dir.glob("policy_*.pth"))
    if last_ckpt:
        try:
            net.load_state_dict(torch.load(last_ckpt[-1], map_location=device))
            print(f"  resumed from {last_ckpt[-1].name}")
        except Exception as e:
            print(f"  failed to resume: {e}")

    save_npz(net, runs_dir / "policy_latest.npz")

    for upd in range(args.updates):
        # Wait for shards.
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
        print(f"upd {upd+1}/{args.updates}  games={len(games)}  "
              f"transitions={stats['n']}  pi={stats['policy_loss']:+.4f}  "
              f"v={stats['value_loss']:.4f}  ent={stats['entropy']:+.4f}")

        if (upd + 1) % args.ckpt_every == 0 or upd == args.updates - 1:
            ck = runs_dir / f"policy_{int(time.time())}.pth"
            torch.save(net.state_dict(), ck)
            save_npz(net, runs_dir / "policy_latest.npz")
            print(f"  saved {ck.name}  + policy_latest.npz")


if __name__ == "__main__":
    main()
