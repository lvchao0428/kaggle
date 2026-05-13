"""v21 rollout workers: RLAgentV21 + optional opponent mix (PyTorch .pth policy)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import msgpack
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission_resolve import resolve_submission_path
from tools.v21.rl_agent_v21 import RLAgentV21, game_summary_from_agent

try:
    from tools.rollout_worker import DENSE_REWARD_INTERVAL, DENSE_REWARD_WEIGHT
except Exception:  # pragma: no cover
    DENSE_REWARD_INTERVAL = 20
    DENSE_REWARD_WEIGHT = 0.10


def load_static_agent(version: str):
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
    mod_key = f"submission_rollout_v21_{mod_tag}".replace("/", "_").replace("\\", "_")
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


def _opp_kind(opp: object, mix_token: Optional[str] = None) -> str:
    if opp == "random":
        return "random"
    if isinstance(opp, RLAgentV21):
        return "trainable_policy"
    if mix_token:
        return f"mix:{mix_token}"
    return "submission"


def _outcome_label(outcome: float) -> str:
    if outcome > 0:
        return "win"
    if outcome < 0:
        return "loss"
    return "draw"


def _append_progress_jsonl(out_dir: Path, worker_id: int, record: Dict) -> None:
    path = out_dir / f"rollout_progress_w{worker_id}.jsonl"
    try:
        with open(path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _emit_game_progress(
    *,
    out_dir: Path,
    worker_id: int,
    game_idx: int,
    games_total: int,
    seed: int,
    game: Dict,
    opp_kind: str,
    quiet: bool,
    t_elapsed: float,
) -> None:
    if quiet:
        return
    gs = game.get("game_summary") or {}
    outcome = float(game.get("outcome", 0.0))
    rw = list(game.get("rewards", []))
    ntr = len(game.get("transitions", []))
    ms = int(gs.get("final_my_ships", 0) or 0)
    es = int(gs.get("final_enemy_ships", 0) or 0)
    np_ = int(gs.get("final_n_planets", 0) or 0)
    nmy = int(gs.get("final_n_my_planets", 0) or 0)
    print(
        f"[w{worker_id}] game {game_idx + 1}/{games_total} "
        f"+{t_elapsed:.0f}s "
        f"seed={seed} {_outcome_label(outcome)} "
        f"r={rw} "
        f"ship_r={float(gs.get('final_my_ship_ratio', 0.0)):.3f} "
        f"planet_r={float(gs.get('final_planet_ratio', 0.0)):.3f} "
        f"ships_mine={ms} ships_en={es} planets={np_} mine={nmy} "
        f"last_step={int(gs.get('last_step', 0))} "
        f"transitions={ntr} opp={opp_kind}",
        flush=True,
    )
    _append_progress_jsonl(
        out_dir,
        worker_id,
        {
            "ts": time.time(),
            "worker": worker_id,
            "game_idx": game_idx,
            "games_total": games_total,
            "seed": seed,
            "outcome": outcome,
            "outcome_label": _outcome_label(outcome),
            "rewards": rw,
            "final_my_ship_ratio": gs.get("final_my_ship_ratio"),
            "final_planet_ratio": gs.get("final_planet_ratio"),
            "final_my_ships": gs.get("final_my_ships"),
            "final_enemy_ships": gs.get("final_enemy_ships"),
            "final_n_planets": gs.get("final_n_planets"),
            "final_n_my_planets": gs.get("final_n_my_planets"),
            "last_step": gs.get("last_step"),
            "transitions": ntr,
            "opp_kind": opp_kind,
        },
    )


def _compute_shaped_rewards(rl_agent: RLAgentV21) -> None:
    if DENSE_REWARD_INTERVAL <= 0:
        return
    ts = rl_agent.transitions
    if not ts:
        return
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


def play_one_game(rl_agent: RLAgentV21, opponent, seed: int) -> Dict:
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
            "shaped_reward": getattr(t, "shaped_reward", 0.0),
            "oob_penalty": getattr(t, "oob_penalty", 0.0),
        }
        for t in rl_agent.transitions
    ]
    gsum = game_summary_from_agent(rl_agent)
    if gsum:
        gsum["outcome"] = outcome
        gsum["rewards"] = list(rewards)
    out = {
        "outcome": outcome,
        "rewards": list(rewards),
        "transitions": transitions,
        "game_summary": gsum or {},
    }
    return out


def _checkpoint_path(args) -> Optional[str]:
    if getattr(args, "checkpoint", None) and Path(args.checkpoint).is_file():
        return str(Path(args.checkpoint).resolve())
    fallback = Path(args.runs_dir) / "policy_latest.pth"
    if fallback.is_file():
        return str(fallback.resolve())
    return None


def _make_agent(args, ckpt: Optional[str], record: bool) -> RLAgentV21:
    dev = torch.device("cpu")
    if getattr(args, "device", None) == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    return RLAgentV21(
        tier=args.tier,
        submission_version=args.submission,
        checkpoint_path=ckpt,
        explore=True,
        record=record,
        temperature=args.temperature,
        device=dev,
    )


def worker_main(args, worker_id: int):
    np.random.seed(int(time.time() * 1000 + worker_id) & 0xFFFFFFFF)
    random.seed(int(time.time() * 1000 + worker_id) & 0xFFFFFFFF)
    out_dir = Path(args.runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = _checkpoint_path(args)
    mix_entries: Optional[List[tuple]] = None
    static_cache: Dict[str, object] = {}
    if getattr(args, "opponent_mix", None):
        mix_entries = _normalize_pairs(_parse_opponent_mix(args.opponent_mix))

    opp_versions = list(args.opponents) if args.opponents else ["v20", "v19"]
    static_opponents = [load_static_agent(v) for v in opp_versions]

    def resolve_mix_opponent(tok: str):
        kt = tok.strip().lower()
        if kt == "self":
            if ckpt is None:
                raise RuntimeError(
                    "--opponent-mix self needs a trained policy_latest.pth or --checkpoint"
                )
            return _make_agent(args, ckpt, record=False)
        key = tok.strip()
        if key not in static_cache:
            static_cache[key] = load_static_agent(key)
        return static_cache[key]

    rl_agent = _make_agent(args, ckpt, record=True)

    quiet = bool(getattr(args, "quiet_rollout", False))
    gmax = args.games_per_worker
    if not quiet:
        print(
            f"[w{worker_id}] rollout_start games={gmax} tier={args.tier} "
            f"submission={args.submission} policy_ckpt={'yes' if ckpt else 'no'}",
            flush=True,
        )

    games_dump: List[Dict] = []
    t0 = time.time()
    for g in range(gmax):
        mix_pick: Optional[str] = None
        if mix_entries is not None:
            acc = 0.0
            r = random.random()
            chosen_tok = mix_entries[-1][0]
            for tok, w in mix_entries:
                acc += w
                if r <= acc:
                    chosen_tok = tok
                    break
            mix_pick = chosen_tok
            opp = resolve_mix_opponent(chosen_tok)
        elif ckpt and random.random() < 0.5:
            opp = _make_agent(args, ckpt, record=False)
        else:
            opp = random.choice(static_opponents)
        seed = random.randint(0, 1_000_000)
        try:
            game = play_one_game(rl_agent, opp, seed)
            games_dump.append(game)
            _emit_game_progress(
                out_dir=out_dir,
                worker_id=worker_id,
                game_idx=g,
                games_total=gmax,
                seed=seed,
                game=game,
                opp_kind=_opp_kind(opp, mix_pick),
                quiet=quiet,
                t_elapsed=time.time() - t0,
            )
        except Exception as e:
            print(f"[w{worker_id}] game {g} failed: {e}", file=sys.stderr)

    shard_path = out_dir / f"shard_w{worker_id}_{int(time.time())}.msgpack"
    with open(shard_path, "wb") as f:
        msgpack.pack({"games": games_dump}, f, use_bin_type=True)
    print(
        f"[w{worker_id}] wrote {shard_path}  games={len(games_dump)}  "
        f"elapsed={time.time()-t0:.1f}s"
    )


def main():
    ap = argparse.ArgumentParser(description="v21 self-play rollout workers")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games-per-worker", type=int, default=20)
    ap.add_argument("--runs-dir", default="runs/v21_lite")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="policy .pth (default: runs-dir/policy_latest.pth if present)",
    )
    ap.add_argument("--tier", default="lite", choices=["lite", "pro", "ultra"])
    ap.add_argument("--submission", default="v20", help="submission_vXX stem, e.g. v20, v21_lite")
    ap.add_argument("--opponents", nargs="*", default=None)
    ap.add_argument("--opponent-mix", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--quiet-rollout",
        action="store_true",
        help="Do not print per-game progress lines or rollout_progress_w*.jsonl lines",
    )
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
