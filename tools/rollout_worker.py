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

    python3.12 tools/rollout_worker.py --games-per-worker 20 \\
        --opponent-mix "self:0.5,v13:0.35,v19@rush:0.15"

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

from submission_resolve import resolve_submission_path  # noqa: E402
from tools.rl_agent import RLAgent  # noqa: E402

# How often (in game steps) to sample a shaped reward.  0 = disabled.
DENSE_REWARD_INTERVAL = 20
# Weight of shaped reward relative to terminal outcome.
DENSE_REWARD_WEIGHT = 0.10


def load_static_agent(version: str):
    """Load submission_<version>.py and return its ``agent``.

    Supports ``v19@turtle``-style suffixes wired to submission_v19
    ``ORB_STRATEGY_PROFILE`` (parallel safe per process).
    """
    if version.strip().lower() == "random":
        return "random"
    raw = version.strip()
    profile = ""
    if "@" in raw:
        base, _, suf = raw.partition("@")
        base = base.strip()
        profile = suf.strip()
    else:
        base = raw
    path = resolve_submission_path(ROOT, base)
    mod_tag = f"{base.replace('/', '_')}__{profile}".replace("/", "_") if profile else base.replace("/", "_")
    mod_key = f"submission_rollout_{mod_tag}".replace("/", "_").replace("\\", "_")
    spec = importlib.util.spec_from_file_location(mod_key, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    spec.loader.exec_module(mod)
    agent_fn = mod.agent
    cv = getattr(mod, "ORB_STRATEGY_PROFILE", None)
    if profile and cv is not None:

        def wrapped(obs, cfg=None, _agent=agent_fn, _pf=profile, _cv=cv):
            t = _cv.set(_pf)
            try:
                return _agent(obs, cfg)
            finally:
                _cv.reset(t)
        return wrapped
    return agent_fn


def _parse_opponent_mix(s: str) -> List[tuple]:
    pairs: List[tuple] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        tok, sep, wt = chunk.rpartition(":")
        if not sep:
            raise ValueError(f"bad opponent-mix token {chunk}")
        pairs.append((tok.strip(), float(wt.strip())))
    return pairs


def _normalize_pairs(pairs: List[tuple]) -> List[tuple]:
    total = sum(w for _, w in pairs if w > 0)
    if total <= 0:
        raise ValueError("opponent_mix weights sum to zero")
    return [(tok, max(1e-6, w / total)) for tok, w in pairs]


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
            "oob_penalty": getattr(t, "oob_penalty", 0.0),
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

    mix_entries: Optional[List[tuple]] = None
    static_cache: Dict[str, object] = {}
    if getattr(args, "opponent_mix", None):
        mix_entries = _normalize_pairs(_parse_opponent_mix(args.opponent_mix))

    # Opponent pool: latest weights + named static bots (legacy path).
    opp_versions = list(args.opponents) if args.opponents else ["v11"]
    static_opponents = [load_static_agent(v) for v in opp_versions]
    weights_for_self = weights  # used when sampling 'self' opponent

    def resolve_mix_opponent(tok: str):
        kt = tok.strip().lower()
        if kt == "self":
            if weights_for_self is None:
                raise RuntimeError(
                    "--opponent-mix references self but --weights is missing")
            return RLAgent(
                weights=weights_for_self,
                explore=True,
                record=False,
                temperature=args.temperature,
            )
        key = tok.strip()
        if key not in static_cache:
            static_cache[key] = load_static_agent(key)
        return static_cache[key]

    rl_agent = RLAgent(weights=weights, explore=True, record=True,
                       temperature=args.temperature)

    games_dump: List[Dict] = []
    t0 = time.time()
    for g in range(args.games_per_worker):
        if mix_entries is not None:
            acc = 0.0
            r = random.random()
            chosen_tok = mix_entries[-1][0]
            for tok, w in mix_entries:
                acc += w
                if r <= acc:
                    chosen_tok = tok
                    break
            opp = resolve_mix_opponent(chosen_tok)
        elif random.random() < 0.5 and weights_for_self is not None:
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
    ap.add_argument(
        "--opponent-mix",
        default=None,
        help=(
            "Comma-separated tok:w pairs, weights auto-normalised, "
            'e.g. "self:0.5,v13:0.3,v19@turtle:0.2"; overrides legacy 50/50 mix.'))
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
