#!/usr/bin/env python3
"""Four-player Orbit Wars FFA evaluator (same agent in all seats).

Requires kaggle-environments. Typical use:

    python3 scripts/eval_ffa_4p.py --version v20 --seeds 1619309859 0 1

First startup can look "stuck": importing kaggle_environments loads many games.
Each seed then runs a **full** episode with 4× heavyweight agents (e.g. v20);
that often takes **many minutes** — progress lines print on env step advances
For a **single browser replay** without 4× heavy agents, use (v20 + 3× builtin
random):

    python3 scripts/replay.py --ffa --a v20 --seed 1619309859

    python3 scripts/eval_ffa_4p.py --version v20 --seeds 1619309859 --progress-every 10
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission_resolve import resolve_submission_path


def _parse_seeds(args):
    out: list[int] = []
    for tok in args.seeds:
        tok = tok.strip()
        if "-" in tok and not tok.startswith("-"):
            lo, hi = tok.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    return out


def _load_agent(version: str):
    path = resolve_submission_path(ROOT, version)
    mod_tag = version.replace("@", "_at_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(
        f"submission_{mod_tag}_ffa_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"submission_{mod_tag}_ffa_eval"] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _obs_step(obs) -> int:
    """Current environment step from a Kaggle observation dict."""
    if isinstance(obs, dict):
        return int(obs.get("step", 0) or 0)
    return int(getattr(obs, "step", 0) or 0)


def _wrap_agent_with_progress(agent_fn, seat: int, prog: dict, every: int, verbose_first: int):
    """Print env step when seat 0 sees a new global step (four agent calls per step)."""
    def wrapped(obs, config=None):
        if seat != 0:
            return agent_fn(obs, config)
        st = _obs_step(obs)
        if st != prog["last_step"]:
            prog["last_step"] = st
            if st <= verbose_first or (every > 0 and st % every == 0):
                elapsed = time.time() - float(prog["t0"])
                print(
                    f"  orbit_wars step={st}  ({elapsed:.0f}s since seed start)",
                    flush=True,
                )
        return agent_fn(obs, config)

    return wrapped


def main() -> int:
    p = argparse.ArgumentParser(description="Orbit Wars 4p FFA evaluate")
    p.add_argument("--version", default="v20", help="submission label, e.g. v20")
    p.add_argument(
        "--seeds",
        nargs="+",
        default=["1619309859", "0", "1", "2"],
        help="Seed list; supports '0-3' style tokens",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=25,
        metavar="N",
        help="Print env step every N steps (via seat 0). Use 0 for no periodic lines after --progress-first.",
    )
    p.add_argument(
        "--progress-first",
        type=int,
        default=12,
        metavar="M",
        help="Always print each of the first M env steps",
    )
    args = p.parse_args()
    seeds = _parse_seeds(args)

    print(
        "Note: cabt loader warnings on macOS are usually harmless.\n"
        "Importing kaggle_environments can take a while; then each 4×v20 seed can run many minutes.\n",
        flush=True,
    )

    from kaggle_environments import evaluate

    print("Loading submission agent…", flush=True)
    ag = _load_agent(args.version)

    t0_all = time.time()
    for sd in seeds:
        prog = {"last_step": -1, "t0": time.time()}
        wrapped = [
            _wrap_agent_with_progress(
                ag,
                seat,
                prog,
                every=args.progress_every,
                verbose_first=args.progress_first,
            )
            for seat in range(4)
        ]
        print(f"seed={sd}: starting evaluate (4× {args.version}) …", flush=True)
        random.seed((410224286193 + int(sd) * 2246822519) % (2**63))
        t_seed = time.time()
        row = evaluate(
            "orbit_wars",
            [lambda o, c, fn=w: fn(o, c) for w in wrapped],
            configuration={"seed": int(sd)},
            num_episodes=1,
            debug=False,
        )[0]
        print(
            f"seed={sd} rewards={row!r}  (seed wall {time.time() - t_seed:.1f}s)",
            flush=True,
        )
    print(f"total elapsed {time.time() - t0_all:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
