"""Imitation pre-training for Orbit Wars RL policy.

Trains the PolicyValueNet to imitate the behaviour of a reference bot
(default: v11) through behaviour cloning (BC) on self-play trajectories.
This warm-starts the policy so that early PPO training does not explore
randomly but already plays near the reference bot's level.

Algorithm
---------
1.  Run N_GAMES games of reference_bot vs reference_bot using kaggle_environments.
2.  At each step, ask v11's PlanArbiter for all candidate plans and record
    which plan the reference bot *chose* (heuristic argmax over plan score).
3.  Train PolicyValueNet's plan_head to reproduce that choice (cross-entropy)
    and value_head to predict the final game outcome (MSE).
4.  Save the result as <runs_dir>/pretrained_<timestamp>.pth and
    <runs_dir>/pretrained_latest.npz for rollout_worker to pick up.

Usage::

    # Quick smoke-test (2 games, 5 epochs)
    python3.12 tools/imitation_pretrain.py --games 2 --epochs 5 \\
        --runs-dir runs/pretrain

    # Full warm-start before PPO
    python3.12 tools/imitation_pretrain.py --games 200 --epochs 20 \\
        --runs-dir runs/pretrain --ref-bot v11
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.feature_extractor import combined_features, state_features
import submission_v11 as v11  # type: ignore  # registered by feature_extractor

# ── lazy torch import so CPU-only machines can still import this module ──
import torch
import torch.nn.functional as F
from torch.optim import Adam

from tools.policy_torch import PolicyValueNet, best_device


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory collection
# ─────────────────────────────────────────────────────────────────────────────

class _RecordingAgent:
    """A thin wrapper around v11's decision pipeline that records (plan_feats,
    chosen_idx, state_feat) at every step.  It always picks the v11 heuristic
    argmax so the labels are the expert's preferred plan."""

    def __init__(self):
        self.records: List[Dict] = []  # list of dicts per step

    def __call__(self, obs, config=None):
        try:
            state = v11.GameState(obs, config)
            if not state.my_pl:
                return []

            v11._GLOBAL_OPP.update(state)
            policy = v11.PhasePolicy.for_state(state)
            snap = v11.Snapshot.build(state, policy)
            diplo = v11.DiplomacyEngine(state, v11._GLOBAL_OPP)

            arbiter = v11.PlanArbiter(snap, diplo, v11._GLOBAL_NEURAL,
                                      elapsed_ms_fn=lambda: 0.0,
                                      deadline_ms=920.0)
            arbiter.commit_urgent()
            plans = arbiter.collect_strategic()

            if plans:
                # Heuristic expert: pick plan with highest (score + sim_bonus).
                phase = state.phase()
                sim_k = 8 if phase != "late" else 10
                plan_scores = np.array([
                    p.score + v11.score_plan_actions(state, p.actions,
                                                     steps=sim_k, tempo_floor=1)
                    for p in plans
                ], dtype=np.float32)
                best_idx = int(np.argmax(plan_scores))
                all_feats = np.stack(
                    [combined_features(p, state) for p in plans]
                )  # (K, F)
                self.records.append({
                    "state_feat": state_features(state).tolist(),
                    "plan_feats": all_feats.tolist(),  # (K, F)
                    "chosen_idx": best_idx,
                    "step": state.step,
                })
                arbiter.commit_best([(0.0, plans[best_idx])])

            arbiter.commit_fallback()
            return arbiter.moves
        except Exception:
            return []


def collect_games(n_games: int, ref_bot_version: str, seed_start: int = 0
                  ) -> List[Tuple[List[Dict], float]]:
    """Run n_games and return [(records, outcome), ...] for agent-0."""
    from kaggle_environments import evaluate

    results: List[Tuple[List[Dict], float]] = []
    for g in range(n_games):
        agent0 = _RecordingAgent()
        agent1 = _RecordingAgent()
        seed = seed_start + g
        try:
            rewards = evaluate(
                "orbit_wars",
                [lambda o, c, a=agent0: a(o, c),
                 lambda o, c, a=agent1: a(o, c)],
                configuration={"seed": int(seed)},
                num_episodes=1,
                debug=False,
            )[0]
            r0, r1 = rewards
            if r0 is None or r1 is None or r0 == r1:
                outcome = 0.0
            elif r0 > r1:
                outcome = 1.0
            else:
                outcome = -1.0
            results.append((agent0.records, outcome))
        except Exception as e:
            print(f"  game {g} failed: {e}", file=sys.stderr)
        if (g + 1) % 10 == 0:
            print(f"  collected {g+1}/{n_games} games")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(game_records: List[Tuple[List[Dict], float]]):
    """Flatten records into arrays for training.

    Returns:
        obs_feats  : (N, F) float32  - combined feat of chosen plan
        plan_feats : list of (K, F) arrays, one per step
        chosen_idxs: (N,) int64
        outcomes   : (N,) float32    - game outcome (value target)
    """
    obs_list: List[np.ndarray] = []
    pf_list: List[np.ndarray] = []
    idx_list: List[int] = []
    out_list: List[float] = []

    for records, outcome in game_records:
        for rec in records:
            pf = np.asarray(rec["plan_feats"], dtype=np.float32)  # (K, F)
            ci = int(rec["chosen_idx"])
            obs_list.append(pf[ci])  # chosen plan feat
            pf_list.append(pf)
            idx_list.append(ci)
            out_list.append(outcome)

    if not obs_list:
        return None
    return (np.stack(obs_list),  # (N, F)
            pf_list,             # list of (K, F)
            np.array(idx_list, dtype=np.int64),
            np.array(out_list, dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(net: PolicyValueNet, dataset, device: torch.device,
             epochs: int = 20, batch_size: int = 512,
             lr: float = 1e-3, value_coef: float = 0.5) -> None:
    """Behaviour cloning: cross-entropy plan selection + MSE value prediction."""
    obs_np, pf_list, idx_np, out_np = dataset
    N = obs_np.shape[0]
    opt = Adam(net.parameters(), lr=lr)
    obs_t = torch.tensor(obs_np, device=device, dtype=torch.float32)
    out_t = torch.tensor(out_np, device=device, dtype=torch.float32)

    print(f"  imitation pre-training: N={N}  epochs={epochs}  bs={batch_size}")
    for ep in range(epochs):
        perm = torch.randperm(N).tolist()
        total_pi = 0.0
        total_v = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            mb = perm[start:start + batch_size]
            if not mb:
                continue

            # Value loss (fast: use obs of chosen plan).
            v_pred, _ = net(obs_t[mb])
            v_loss = F.mse_loss(v_pred.squeeze(-1), out_t[mb])

            # Policy loss: cross-entropy over candidates.
            pi_losses = []
            for local_i, global_i in enumerate(mb):
                pf = torch.tensor(pf_list[global_i], device=device,
                                  dtype=torch.float32)  # (K, F)
                K = pf.size(0)
                _, plan_scores = net(pf)            # (K, 1)
                logits = plan_scores.squeeze(-1)    # (K,)
                target = min(int(idx_np[global_i]), K - 1)
                target_t = torch.tensor(target, device=device, dtype=torch.long)
                pi_losses.append(
                    F.cross_entropy(logits.unsqueeze(0), target_t.unsqueeze(0))
                )
            pi_loss = torch.stack(pi_losses).mean()

            loss = pi_loss + value_coef * v_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total_pi += float(pi_loss.item())
            total_v += float(v_loss.item())
            n_batches += 1

        if n_batches:
            print(f"  epoch {ep+1}/{epochs}  "
                  f"pi={total_pi/n_batches:.4f}  "
                  f"v={total_v/n_batches:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers (mirror learner.save_npz)
# ─────────────────────────────────────────────────────────────────────────────

def save_npz(net: PolicyValueNet, path: Path) -> None:
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Imitation pre-training for Orbit Wars RL")
    ap.add_argument("--games", type=int, default=50,
                    help="Number of expert self-play games to collect.")
    ap.add_argument("--epochs", type=int, default=20,
                    help="Number of BC training epochs.")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--runs-dir", default="runs/pretrain",
                    help="Output directory for checkpoints and npz.")
    ap.add_argument("--ref-bot", default="v11",
                    help="Reference bot version for expert demos (ignored; v11 is always used).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    device = best_device()
    print(f"device: {device}")

    print(f"Collecting {args.games} expert games...")
    t0 = time.time()
    game_records = collect_games(args.games, args.ref_bot, seed_start=args.seed)
    print(f"  collected {len(game_records)} games in {time.time()-t0:.1f}s")

    dataset = build_dataset(game_records)
    if dataset is None:
        print("No transitions collected - exiting.")
        return

    net = PolicyValueNet().to(device)
    # Try to resume from an existing pretrained checkpoint.
    existing = sorted(runs_dir.glob("pretrained_*.pth"))
    if existing:
        try:
            net.load_state_dict(torch.load(existing[-1], map_location=device))
            print(f"  resumed from {existing[-1].name}")
        except Exception as e:
            print(f"  could not load {existing[-1].name}: {e}")

    pretrain(net, dataset, device,
             epochs=args.epochs,
             batch_size=args.batch_size,
             lr=args.lr,
             value_coef=args.value_coef)

    ts = int(time.time())
    ckpt_path = runs_dir / f"pretrained_{ts}.pth"
    torch.save(net.state_dict(), ckpt_path)
    npz_path = runs_dir / "pretrained_latest.npz"
    save_npz(net, npz_path)
    # Also write to policy_latest.npz so rollout_worker can pick it up directly.
    save_npz(net, runs_dir / "policy_latest.npz")
    print(f"Saved {ckpt_path.name}  +  {npz_path.name}  +  policy_latest.npz")


if __name__ == "__main__":
    main()
