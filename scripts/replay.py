#!/usr/bin/env python3
"""Orbit Wars — 本地对局可视化回放生成器

运行一局完整对战（或指定步数），生成可在浏览器中播放的 HTML 回放文件。

用法::

    # 最简：v13 vs v12，seed=42，生成 replay.html 后自动打开浏览器
    python3.12 scripts/replay.py

    # 四人混战 FFA（默认 v20 + 3×random，只跑一局、生成 HTML，比 4×v20 评估快很多）
    python3 scripts/replay.py --ffa --a v20 --seed 1619309859

    # FFA：你是 v20，另外三家都用 v19（比 4×random 更「正常」，仍比 4×v20 轻）
    python3 scripts/replay.py --ffa --a v20 --b v19 --c v19 --d v19 --seed 1619309859

    # 四人全部指定（会慢：四席都是重 bot 时）
    python3 scripts/replay.py --ffa --a v20 --b v19 --c random --d random --seed 0

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
  - 两方或四方分数曲线（取决于 1v1 / --ffa）
"""

from __future__ import annotations

import argparse
import importlib.util
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


def _slug_version(v: str) -> str:
    return v.replace("@", "_at_").replace("/", "_").replace(".", "_")


def _run_until_done(env, agents: list, max_steps: int) -> int:
    """Step environment with ``len(agents)`` parallel players. Returns steps executed."""
    n = len(agents)
    step = 0
    t0 = time.time()
    while not env.done and step < max_steps:
        cfg = env.configuration
        actions = []
        for i in range(n):
            ag = agents[i]
            obs = env.state[i].observation
            if isinstance(ag, str):
                actions.append(ag)
            else:
                actions.append(ag(obs, cfg))
        env.step(actions)
        step += 1
        if step % 100 == 0:
            r = [s.reward for s in env.state]
            print(f"  step {step}/{max_steps}  scores={r}  elapsed={time.time()-t0:.1f}s")
    return step


def main():
    p = argparse.ArgumentParser(description="Orbit Wars local replay generator")
    p.add_argument(
        "--ffa",
        action="store_true",
        help="Four-player FFA HTML replay. Default seats 1–3: random (seat 0 = --a).",
    )
    p.add_argument("--a", default="v13", help="Agent A / seat 0 (e.g. v20 with --ffa)")
    p.add_argument(
        "--b",
        default=None,
        help="Agent B / seat 1 (1v1 default v12 if omitted; --ffa default random)",
    )
    p.add_argument("--c", default=None, help="Seat 2 (--ffa only; default random)")
    p.add_argument("--d", default=None, help="Seat 3 (--ffa only; default random)")
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

    if args.ffa:
        lineup_labels = [
            args.a,
            args.b or "random",
            args.c or "random",
            args.d or "random",
        ]
    else:
        lineup_labels = [args.a, args.b or "v12"]

    seed_list = _parse_seeds(args.seed, args.seeds)
    if len(seed_list) > 1 and args.out:
        print("--out is only allowed with a single seed; ignoring --out", file=sys.stderr)

    from kaggle_environments import make

    print(f"Loading agents: {' | '.join(lineup_labels)} ({'FFA' if args.ffa else '1v1'}) ...")
    agents = [_load_agent(lab) for lab in lineup_labels]

    last_out: Path | None = None
    for i, sd in enumerate(seed_list):
        print(f"Running game  seed={sd}  max_steps={args.steps} ...")
        env = make("orbit_wars", debug=False, configuration={"seed": int(sd)})
        env.reset(len(agents))

        t0 = time.time()
        step = _run_until_done(env, agents, args.steps)

        elapsed = time.time() - t0
        final_rewards = [s.reward for s in env.state]
        if len(agents) == 2:
            winner = (
                lineup_labels[0]
                if final_rewards[0] > final_rewards[1]
                else (
                    lineup_labels[1]
                    if final_rewards[1] > final_rewards[0]
                    else "tie"
                )
            )
        else:
            mx = max(final_rewards)
            win_ix = [i for i, r in enumerate(final_rewards) if r == mx]
            winner = (
                ", ".join(f"seat{j}={lineup_labels[j]}" for j in win_ix)
                if len(win_ix) < len(final_rewards)
                else "tie"
            )
        print(
            f"Game done  seed={sd}  steps={step}  rewards={final_rewards}  "
            f"winner={winner}  {elapsed:.1f}s"
        )

        # Render HTML
        print("Rendering HTML ...")
        html = env.render(mode="html")

        # Determine output path
        if args.out and len(seed_list) == 1:
            out_path = Path(args.out)
        else:
            replay_dir = ROOT / "replays"
            replay_dir.mkdir(exist_ok=True)
            if args.ffa:
                parts = [_slug_version(x) for x in lineup_labels]
                out_path = replay_dir / f"ffa_{'_'.join(parts)}_seed{sd}.html"
            else:
                out_path = replay_dir / f"{args.a}_vs_{args.b or 'v12'}_seed{sd}.html"

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
            if args.ffa:
                parts = [_slug_version(x) for x in lineup_labels]
                candidate = replay_dir / f"ffa_{'_'.join(parts)}_seed{first_sd}.html"
            else:
                candidate = replay_dir / f"{args.a}_vs_{args.b or 'v12'}_seed{first_sd}.html"
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
