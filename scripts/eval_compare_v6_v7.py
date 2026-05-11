#!/usr/bin/env python3
"""本地对比 v6 / v7 / v8（无需提交 Kaggle）。

- 默认：各方案作为 player 0 对 built-in 对手（默认 random），统计胜率。
- ``--head2head``：额外做 v7 对 v8 双向座位（每 seed 两局），更贴近真实强度对比。
- 内置 ``random`` 对手使用全局 ``random``；脚本在每一局前对全局 RNG 重播种，避免
  「先跑完 v6 再跑 v7」时跨版本对比不公平。

依赖: pip install "kaggle-environments>=1.28.0"
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission_resolve import resolve_submission_path


def _load_agent(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(str(file_path))
    mod = importlib.util.module_from_spec(spec)
    # Required so @dataclass(order=True) can resolve cls.__module__ during exec.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _seed_orbit_wars_random_bot(sd: int) -> None:
    """orbit_wars built-in ``random`` opponent uses module-level ``random`` (not
    ``configuration.seed``). Without reseeding, running many ``evaluate`` calls
    in one process makes later agents see a different RNG stream for the same
    ``seed`` than earlier agents — unfair when comparing v6 / v7 / v8.

    Reseed once per episode so the random bot's stream restarts predictably per
    ``sd``. (After turn 1, streams still diverge if agents choose different
    moves; that is expected.)
    """
    random.seed((410224286193 + int(sd) * 2246822519) % (2**63))


def _winner(rewards_row):
    if not rewards_row or len(rewards_row) < 2:
        return None
    r0, r1 = rewards_row[0], rewards_row[1]
    if r0 is None or r1 is None:
        return None
    if r0 > r1:
        return 0
    if r1 > r0:
        return 1
    return None


def main():
    from kaggle_environments import evaluate

    p = argparse.ArgumentParser(description="Local Orbit Wars agent comparison")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    p.add_argument("--opponent", type=str, default="random")
    p.add_argument(
        "--head2head",
        action="store_true",
        help="Also run v7 vs v8 with swapped seats each seed (2 games per seed)",
    )
    p.add_argument(
        "--versions",
        type=str,
        default="6,7,8",
        help="Comma list for vs-random phase (e.g. 7,8)",
    )
    p.add_argument(
        "--skip-random",
        action="store_true",
        help="Only run head2head (requires --head2head)",
    )
    args = p.parse_args()

    agents = {}
    for v in [x.strip() for x in args.versions.split(",") if x.strip()]:
        try:
            path = resolve_submission_path(ROOT, f"v{v}")
        except FileNotFoundError:
            print(f"Skip v{v}: missing submission_v{v}.py", file=sys.stderr)
            continue
        agents[f"v{v}"] = _load_agent(f"sub_v{v}", path)

    if args.head2head:
        if "v7" not in agents or "v8" not in agents:
            ag7 = _load_agent("sub_v7", resolve_submission_path(ROOT, "v7"))
            ag8 = _load_agent("sub_v8", resolve_submission_path(ROOT, "v8"))
        else:
            ag7, ag8 = agents["v7"], agents["v8"]

        v7_games = 0
        v8_games = 0
        print("=== head2head v7 vs v8 (two seatings per seed) ===")
        for sd in args.seeds:
            # Agent.act passes (observation, configuration). A 2-arg lambda would bind
            # configuration to the defaulted slot and call it as the inner agent.
            r_ab = evaluate(
                "orbit_wars",
                [
                    lambda obs, cfg, f=ag7: f(obs, cfg),
                    lambda obs, cfg, f=ag8: f(obs, cfg),
                ],
                configuration={"seed": int(sd)},
                num_episodes=1,
                debug=False,
            )[0]
            r_ba = evaluate(
                "orbit_wars",
                [
                    lambda obs, cfg, f=ag8: f(obs, cfg),
                    lambda obs, cfg, f=ag7: f(obs, cfg),
                ],
                configuration={"seed": int(sd)},
                num_episodes=1,
                debug=False,
            )[0]
            w_ab = _winner(r_ab)
            w_ba = _winner(r_ba)
            if w_ab == 0:
                v7_games += 1
            elif w_ab == 1:
                v8_games += 1
            if w_ba == 0:
                v8_games += 1
            elif w_ba == 1:
                v7_games += 1
            la = "v7" if w_ab == 0 else "v8" if w_ab == 1 else "tie?"
            lb = "v8" if w_ba == 0 else "v7" if w_ba == 1 else "tie?"
            print(f"  seed={sd}  [v7,v8]={r_ab} -> {la} | [v8,v7]={r_ba} -> {lb}")
        tot = v7_games + v8_games
        print(f"\nv7 wins {v7_games}/{tot} | v8 wins {v8_games}/{tot}\n")

    if args.skip_random and not args.head2head:
        print("Nothing to do: use --head2head or omit --skip-random", file=sys.stderr)
        sys.exit(1)

    if not args.skip_random:
        print(f"=== vs opponent={args.opponent!r} (you = player 0) ===")
        for name, ag in agents.items():
            wins = 0
            n = 0
            for sd in args.seeds:
                if args.opponent == "random":
                    _seed_orbit_wars_random_bot(sd)
                rew = evaluate(
                    "orbit_wars",
                    [lambda obs, cfg, f=ag: f(obs, cfg), args.opponent],
                    configuration={"seed": int(sd)},
                    num_episodes=1,
                    debug=False,
                )
                n += 1
                r0, r1 = rew[0][0], rew[0][1]
                if r0 is not None and r1 is not None and r0 > r1:
                    wins += 1
                print(f"  {name} seed={sd} -> {rew[0]}")
            print(f"{name}: wins {wins}/{n} as player 0\n")


if __name__ == "__main__":
    main()
