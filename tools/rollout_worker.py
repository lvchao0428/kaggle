"""Self-play rollout worker for Orbit Wars RL.

Spawns N CPU processes; each plays self-play games with the latest policy
and writes msgpack shards `<runs>/shard_{worker}_{epoch}_{game_idx}.msgpack`
containing flat lists of transitions + per-game outcome.

Phase 2 addition: records mid-game shaped rewards by sampling
eval_sim_planets every DENSE_REWARD_INTERVAL steps and storing the
differential as `shaped_reward` on each transition.  The shaped rewards
are mixed with the terminal outcome in the learner GAE so the policy
gets denser gradient signal.

Usage::

    python3.12 tools/rollout_worker.py --workers 4 --games-per-worker 50 \\
        --runs-dir runs/exp1 --weights runs/exp1/policy_latest.npz \\
        --opponents v9 v10 v11

If --weights is omitted, the policy is randomly initialised (still useful
for smoke testing).
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import msgpack
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.rl_agent import RLAgent  # noqa: E402

# How often (in game steps) to sample a shaped reward.  0 = disabled.
DENSE_REWARD_INTERVAL = 20
# Weight of shaped reward relative to terminal outcome.
DENSE_REWARD_WEIGHT = 0.10


def load_static_agent(version: str):
    """Load submission_<version>.py and return its `agent` callable."""
    if version == "random":
        return "random"
    path = ROOT / f"submission_{version}.py"
    spec = importlib.util.spec_from_file_location(f"submission_{version}_rollout", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"submission_{version}_rollout"] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def load_weights(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    npz = np.load(p)
    return {k: npz[k] for k in npz.files}


def _compute_shaped_rewards(rl_agent: RLAgent, gamma: float = 0.10) -> None:
    """Backfill shaped_reward into already-recorded transitions.

    We sample eval_sim_planets every DENSE_REWARD_INTERVAL steps by looking
    at the `state_feat` stored in each transition.  The shaped reward at step t
    is the change in the normalised score feature (index 0 of state_feat)
    compared to step t-DENSE_REWARD_INTERVAL, scaled by DENSE_REWARD_WEIGHT.

    This function mutates rl_agent.transitions in-place.
    """
    if DENSE_REWARD_INTERVAL <= 0:
        return
    ts = rl_agent.transitions
    if not ts:
        return
    # state_feat[0] is normalised ship-ratio (see feature_extractor).
    for i, t in enumerate(ts):
        if i == 0:
            t.shaped_reward = 0.0
            continue
        prev_i = max(0, i - DENSE_REWARD_INTERVAL)
        if (i - prev_i) >= DENSE_REWARD_INTERVAL:
            delta = float(ts[i].state_feat[0]) - float(ts[prev_i].state_feat[0])
            t.shaped_reward = DENSE_REWARD_WEIGHT * delta
        else:
            t.shaped_reward = 0.0


def play_one_game(rl_agent: RLAgent, opponent, seed: int) -> Dict:
    """Run a single self-play (or vs-static) game. Returns serialisable dict."""
    from kaggle_environments import evaluate
    rl_agent.transitions.clear()
    rewards = evaluate(
        "orbit_wars",
        [lambda o, c, ag=rl_agent: ag(o, c), opponent],
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

    # Backfill shaped rewards after the game is complete.
    _compute_shaped_rewards(rl_agent)

    transitions = [
        {
            "obs_feat": t.obs_feat.tolist(),
            "plan_feats": t.plan_feats.tolist(),
            "chosen_idx": t.chosen_idx,
            "log_prob": t.log_prob,
            "value_pred": t.value_pred,
            "plan_score_net": t.plan_score_net,
            "step": t.step,
            "state_feat": t.state_feat.tolist(),
            # Phase 2: dense shaped reward for this step.
            "shaped_reward": getattr(t, "shaped_reward", 0.0),
        }
        for t in rl_agent.transitions
    ]
    return {
        "outcome": outcome,
        "rewards": list(rewards),
        "transitions": transitions,
    }


def worker_main(args, worker_id: int):
    np.random.seed(int(time.time() * 1000 + worker_id) & 0xFFFFFFFF)
    random.seed(int(time.time() * 1000 + worker_id) & 0xFFFFFFFF)
    weights = load_weights(args.weights)
    out_dir = Path(args.runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Opponent pool: latest weights + named static bots.
    opp_versions = list(args.opponents) if args.opponents else ["v11"]
    static_opponents = [load_static_agent(v) for v in opp_versions]
    weights_for_self = weights  # used when sampling 'self' opponent

    rl_agent = RLAgent(weights=weights, explore=True, record=True,
                       temperature=args.temperature)

    games_dump: List[Dict] = []
    t0 = time.time()
    for g in range(args.games_per_worker):
        # Opponent: 50% self (most recent weights), else random pick from pool.
        if random.random() < 0.5 and weights_for_self is not None:
            opp = RLAgent(weights=weights_for_self, explore=True,
                          record=False, temperature=args.temperature)
        else:
            opp = random.choice(static_opponents)
        seed = random.randint(0, 1_000_000)
        try:
            game = play_one_game(rl_agent, opp, seed)
            games_dump.append(game)
        except Exception as e:
            print(f"[w{worker_id}] game {g} failed: {e}", file=sys.stderr)

    shard_path = out_dir / f"shard_w{worker_id}_{int(time.time())}.msgpack"
    with open(shard_path, "wb") as f:
        msgpack.pack({"games": games_dump}, f, use_bin_type=True)
    print(f"[w{worker_id}] wrote {shard_path}  "
          f"games={len(games_dump)}  elapsed={time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="Orbit Wars self-play rollout workers")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games-per-worker", type=int, default=20)
    ap.add_argument("--runs-dir", default="runs/exp1")
    ap.add_argument("--weights", default=None,
                    help="Path to .npz with RLAgent weights")
    ap.add_argument("--opponents", nargs="*", default=["v11", "v10", "v9"])
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    if args.workers <= 1:
        worker_main(args, worker_id=0)
        return

    procs = []
    for wid in range(args.workers):
        p = mp.Process(target=worker_main, args=(args, wid), daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
