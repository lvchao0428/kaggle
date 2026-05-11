#!/usr/bin/env python3
"""Orbit Wars — 本地对局可视化回放生成器

运行一局完整对战（或指定步数），生成可在浏览器中播放的 HTML 回放文件。

用法::

    # 最简：v13 vs v12，seed=42，生成 replay.html 后自动打开浏览器
    python3.12 scripts/replay.py

    # 指定版本 / seed / 步数
    python3.12 scripts/replay.py --a v13 --b v11 --seed 7 --steps 200

    # 生成但不自动打开
    python3.12 scripts/replay.py --a v13 --b v9 --seed 0 --no-open

    # vs 内置 random
    python3.12 scripts/replay.py --a v13 --b random --seed 42

    # 与 eval_head2head 相同 seed 列表，生成多个 HTML，默认用浏览器打开最后一场
    python3.12 scripts/replay.py --a v15 --b v14 --seeds 0-9

    # 多场只打开第一场
    python3.12 scripts/replay.py --a v15 --b v14 --seeds 0-9 --open-first

输出的 HTML 文件用标准浏览器打开即可看到：
  - 动画回放（可播放 / 暂停 / 拖进度条）
  - 每回合行星状态、舰队轨迹
  - 双方分数曲线
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from submission_resolve import resolve_submission_path


def _load_agent(version: str):
    if version == "random":
        return "random"
    path = resolve_submission_path(ROOT, version)
    spec = importlib.util.spec_from_file_location(f"sub_{version}_replay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"sub_{version}_replay"] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _make_callable(agent):
    """Wrap agent so it can be passed to kaggle_environments."""
    if isinstance(agent, str):
        return agent
    return lambda obs, cfg, _a=agent: _a(obs, cfg)


def _parse_seeds(seed: int, seeds_arg: list[str] | None) -> list[int]:
    """Single --seed or multiple --seeds tokens (same as eval_head2head: '0-9' or '0 1 2')."""
    if seeds_arg:
        out: list[int] = []
        for tok in seeds_arg:
            tok = tok.strip()
            if "-" in tok and not tok.startswith("-"):
                lo, hi = tok.split("-", 1)
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(tok))
        return out
    return [seed]


def main():
    p = argparse.ArgumentParser(description="Orbit Wars local replay generator")
    p.add_argument("--a", default="v13", help="Agent A version (e.g. v13)")
    p.add_argument("--b", default="v12", help="Agent B version (e.g. v12, random)")
    p.add_argument("--seed", type=int, default=42,
                   help="Game seed (ignored if --seeds is set)")
    p.add_argument("--seeds", nargs="*", default=None,
                   help="Multiple seeds, e.g. --seeds 0-9 or --seeds 0 1 2. Generates one HTML per seed; opens the last unless --open-first.")
    p.add_argument("--steps", type=int, default=500,
                   help="Max steps to simulate (default 500 = full game)")
    p.add_argument("--out", default=None,
                   help="Output HTML path (only valid with a single seed; default: replays/<a>_vs_<b>_seed<seed>.html)")
    p.add_argument("--no-open", action="store_true",
                   help="Don't auto-open in browser after generation")
    p.add_argument("--open-first", action="store_true",
                   help="With multiple --seeds, open the first replay instead of the last")
    args = p.parse_args()

    seed_list = _parse_seeds(args.seed, args.seeds)
    if len(seed_list) > 1 and args.out:
        print("--out is only allowed with a single seed; ignoring --out", file=sys.stderr)

    from kaggle_environments import make

    print(f"Loading agents: {args.a} vs {args.b} ...")
    agent_a = _load_agent(args.a)
    agent_b = _load_agent(args.b)

    last_out: Path | None = None
    for i, sd in enumerate(seed_list):
        print(f"Running game  seed={sd}  max_steps={args.steps} ...")
        env = make("orbit_wars", debug=False, configuration={"seed": int(sd)})
        env.reset()

        t0 = time.time()
        step = 0
        while not env.done and step < args.steps:
            obs_a = env.state[0].observation
            obs_b = env.state[1].observation
            cfg = env.configuration

            act_a = agent_a(obs_a, cfg) if not isinstance(agent_a, str) else None
            act_b = agent_b(obs_b, cfg) if not isinstance(agent_b, str) else None

            actions = []
            if isinstance(agent_a, str):
                actions.append(agent_a)
            else:
                actions.append(act_a)
            if isinstance(agent_b, str):
                actions.append(agent_b)
            else:
                actions.append(act_b)

            env.step(actions)
            step += 1
            if step % 100 == 0:
                r = [s.reward for s in env.state]
                print(f"  step {step}/{args.steps}  scores={r}  elapsed={time.time()-t0:.1f}s")

        elapsed = time.time() - t0
        final_rewards = [s.reward for s in env.state]
        winner = args.a if final_rewards[0] > final_rewards[1] else (
            args.b if final_rewards[1] > final_rewards[0] else "tie"
        )
        print(f"Game done  seed={sd}  steps={step}  rewards={final_rewards}  "
              f"winner={winner}  {elapsed:.1f}s")

        # Render HTML
        print("Rendering HTML ...")
        html = env.render(mode="html")

        # Determine output path
        if args.out and len(seed_list) == 1:
            out_path = Path(args.out)
        else:
            replay_dir = ROOT / "replays"
            replay_dir.mkdir(exist_ok=True)
            out_path = replay_dir / f"{args.a}_vs_{args.b}_seed{sd}.html"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        print(f"Saved  {out_path}  ({len(html)//1024} KB)")
        last_out = out_path

    if not last_out:
        return 1

    if not args.no_open:
        to_open = last_out
        if args.open_first and len(seed_list) > 1:
            replay_dir = ROOT / "replays"
            replay_dir.mkdir(exist_ok=True)
            first_sd = seed_list[0]
            candidate = replay_dir / f"{args.a}_vs_{args.b}_seed{first_sd}.html"
            if candidate.is_file():
                to_open = candidate
        print(f"Opening in browser: {to_open}")
        _open_browser(to_open)

    return 0


def _open_browser(path: Path) -> None:
    """Cross-platform browser open."""
    url = path.resolve().as_uri()
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", url])
    else:
        import webbrowser
        webbrowser.open(url)


if __name__ == "__main__":
    raise SystemExit(main())
