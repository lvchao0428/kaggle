#!/usr/bin/env python3
"""Opening peel probe: v20 should send ~OPENING_FIRST_CAPTURE_SEND to a factory neutral.

合成观测（HQ=27、厂 20@prod4）用于锁「削顶 reserve + 一波抢厂」逻辑。
**真实对局**同一 (seed, premoves) 下 HQ 可能远小于 27（兵已派出、负战损、多星分流），
不能与 synthetic 的预期兵力类比。

用法::

    python3.12 tools/sim_first_turn_opening.py --synthetic -q
    python3.12 tools/sim_first_turn_opening.py --seed 0 --premoves 17 -b v18 -q
    python3.12 tools/sim_first_turn_opening.py --seed 0 -b v18 --scan-premoves 0-200 -q

``-q`` 只抬高 Python logging；cabt dlopen 仍会打印到 stderr，可 ``2>/dev/null | grep -v cabt``。

退出码: 0=PASS, 1=FAIL 或扫描无 PASS, 2=SKIP（单点模式下兵力不足）
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import random
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _quiet_loggers() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    for name in (
        "kaggle_environments",
        "kaggle_environments.envs",
        "kaggle_environments.envs.open_spiel_env",
        "kaggle_environments.envs.open_spiel_env.open_spiel_env",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def _load_v20():
    spec = importlib.util.spec_from_file_location("v20sim", ROOT / "submission_v20.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v20sim"] = mod
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def _seed_random_bot(sd: int) -> None:
    random.seed((410224286193 + int(sd) * 2246822519) % (2**63))


def _load_opponent(version: str):
    if version == "random":
        return "random"
    from submission_resolve import resolve_submission_path

    path = resolve_submission_path(ROOT, version)
    tag = f"oppsim_{version}"
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[tag] = mod
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod.agent


def _parse_int_range(spec: str) -> range:
    spec = spec.strip()
    if "-" in spec and not spec.startswith("-"):
        lo_s, hi_s = spec.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        if hi < lo:
            lo, hi = hi, lo
        return range(lo, hi + 1)
    v = int(spec)
    return range(v, v + 1)


def _resolve_hit_planet(mod, state, src_id: int, angle: float, ships: int):
    src = state.get(src_id)
    if src is None:
        return None
    lx, ly = mod.launch_origin(src, float(angle))
    for dst in state.planets:
        if dst.id == src_id:
            continue
        if mod.launch_hits_target_first(
            state,
            lx,
            ly,
            float(angle),
            int(ships),
            dst.id,
            ignore_planet_id=None,
        ):
            return dst
    return None


def _synthetic_observation() -> dict[str, Any]:
    return {
        "player": 0,
        "step": 17,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
        "initial_planets": [
            [101, 0, 12.0, 50.0],
            [102, -1, 12.0, 76.0],
        ],
        "planets": [
            [101, 0, 12.0, 50.0, 3.0, 27, 4],
            [102, -1, 12.0, 76.0, 3.0, 20, 4],
        ],
        "fleets": [],
        "configuration": {"shipSpeed": 6.0, "episodeSteps": 500},
    }


def _check_moves(
    mod,
    state,
    moves: list,
    min_send: int,
    neutral_ships: int,
    tol: int,
) -> tuple[bool, str]:
    if not moves:
        return False, "agent returned no moves"

    for m in moves:
        if len(m) < 3:
            continue
        src_id, angle, ships = int(m[0]), float(m[1]), int(m[2])
        if ships < min_send:
            continue
        dst = _resolve_hit_planet(mod, state, src_id, angle, ships)
        if dst is None:
            continue
        if dst.owner != -1:
            continue
        if abs(dst.ships - int(neutral_ships)) > int(tol):
            continue
        return True, (
            f"peel={ships} -> neutral id={dst.id} ships={dst.ships} prod={dst.production}"
        )
    return False, "no move met min_send + neutral ~target garrison + geometry hit"


def _estimate_need_minmax_neutral_band(
    mod,
    state,
    neutral_ships: int,
    tol: int,
) -> tuple[int | None, int | None]:
    lo: int | None = None
    hi: int | None = None
    for dst in state.neu_pl:
        if abs(dst.ships - int(neutral_ships)) > int(tol):
            continue
        for src in state.my_pl:
            need, _ = mod.capture_need(state, src, dst)
            lo = need if lo is None or need < lo else lo
            hi = need if hi is None or need > hi else hi
    return lo, hi


def _run_premoves_env(
    mod,
    *,
    seed: int,
    premoves: int,
    opponent_agent: Any,
    episode_steps: int | None,
):
    from kaggle_environments import make

    cfg_kw: dict = {"seed": int(seed)}
    if episode_steps is not None:
        cfg_kw["episodeSteps"] = int(episode_steps)

    env = make("orbit_wars", debug=False, configuration=cfg_kw)
    env.reset()
    cfg = env.configuration
    act_a: Callable = mod.agent

    for i in range(max(0, premoves)):
        obs_a = env.state[0].observation
        obs_b = env.state[1].observation
        if opponent_agent == "random":
            _seed_random_bot(int(seed) + i * 10007)
        a_moves = act_a(obs_a, cfg)
        if opponent_agent == "random":
            b_moves = opponent_agent
        else:
            b_moves = opponent_agent(obs_b, cfg)
        env.step([a_moves, b_moves])

    obs = env.state[0].observation
    state = mod.GameState(obs, cfg)
    policy = mod.PhasePolicy.for_state(state)
    snap = mod.Snapshot.build(state, policy)
    return obs, cfg, state, snap


def _probe_state(
    mod,
    args: argparse.Namespace,
    obs: Any,
    cfg: Any,
    state: Any,
    snap: Any,
    *,
    seed: int,
    premoves: int,
    verbose: bool,
) -> tuple[int, str]:
    hq = max(state.my_pl, key=lambda p: p.ships)
    if verbose:
        print(
            f"seed={seed} premoves={premoves}  step={state.step}  phase={state.phase()}"
        )
        print(
            f"  HQ id={hq.id} ships={hq.ships} prod={hq.production}  "
            f"reserve={snap.reserve[hq.id]} surplus={snap.surplus[hq.id]}"
        )

    if args.min_hq_ships and hq.ships < args.min_hq_ships:
        return 2, (
            f"SKIP: HQ ships {hq.ships} < --min-hq-ships {args.min_hq_ships}"
        )

    if hq.ships < args.min_send:
        return 2, (
            f"SKIP: HQ ships {hq.ships} < min-send {args.min_send} — "
            f"synthetic 是人工 27 兵；真盘请用 --scan-premoves 找 HQ≥min-send 的时刻"
        )

    if verbose:
        for n in sorted(state.neu_pl, key=lambda x: (-x.production, -x.ships))[:10]:
            print(
                f"  neu id={n.id} ships={n.ships} prod={n.production} "
                f"pos=({n.x:.1f},{n.y:.1f})"
            )

    moves = mod.agent(obs, cfg)
    if verbose:
        print(f"  moves ({len(moves)}): {moves!r}")
        print(
            f"  constants OPENING_SEND={mod.OPENING_FIRST_CAPTURE_SEND} "
            f"SOLO_LAST={mod.OPENING_SOLO_HQ_RESERVE_LAST_STEP}"
        )

    ok, detail = _check_moves(
        mod, state, moves, args.min_send, args.neutral_ships, args.tol
    )
    if ok:
        return 0, detail
    if not moves or "no moves" in detail:
        lo, hi = _estimate_need_minmax_neutral_band(
            mod, state, args.neutral_ships, args.tol
        )
        max_1 = max((snap.surplus.get(p.id, 0) for p in state.my_pl), default=0)
        tot = sum(snap.surplus.get(p.id, 0) for p in state.my_pl)
        if lo is not None and hi is not None:
            detail += (
                f" | hint: max_single_surplus={max_1} sum_surplus={tot} "
                f"capture_need range for ~{args.neutral_ships}±{args.tol} neu: {lo}..{hi} "
                f"(planner may pick high end / multi-source / gates)"
            )
            if (
                not moves
                and max_1 >= hi
                and hi == lo
            ):
                detail += (
                    "; parity: heuristic need==surplus but peel may require +1 "
                    "after intercept iteration / _emit retry — or plan blocked by sim gate"
                )
    return 1, detail


def main() -> int:
    p = argparse.ArgumentParser(description="Opening 21-ship factory grab probe")
    p.add_argument("--synthetic", action="store_true", help="toy GameState only (no env)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--premoves", type=int, default=0, help="full steps before probe")
    p.add_argument("-b", "--opponent", default="random", help="random | v18 | …")
    p.add_argument("--episode-steps", type=int, default=None)
    p.add_argument("--min-send", type=int, default=21)
    p.add_argument("--neutral-ships", type=int, default=20)
    p.add_argument("--tol", type=int, default=2)
    p.add_argument(
        "--min-hq-ships",
        type=int,
        default=0,
        help="SKIP if max friendly ships below this (0=disabled)",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="less logging (does not silence cabt stderr)",
    )
    p.add_argument(
        "--scan-premoves",
        metavar="LO-HI",
        default=None,
        help="for each premoves in range on --seed(s), run probe when HQ>=min-send",
    )
    p.add_argument(
        "--scan-seeds",
        metavar="LO-HI",
        default=None,
        help="outer loop over seeds (with --scan-premoves)",
    )
    args = p.parse_args()

    if args.quiet:
        _quiet_loggers()

    mod = _load_v20()

    if args.synthetic:
        raw = _synthetic_observation()
        cfg = raw["configuration"]
        state = mod.GameState(raw, cfg)
        policy = mod.PhasePolicy.for_state(state)
        snap = mod.Snapshot.build(state, policy)
        hq = state.my_pl[0]
        print(
            f"SYNTHETIC  step={state.step}  HQ id={hq.id} ships={hq.ships} prod={hq.production}"
        )
        print(
            f"  reserve={snap.reserve.get(hq.id)} surplus={snap.surplus.get(hq.id)}  "
            f"OPENING_SEND={mod.OPENING_FIRST_CAPTURE_SEND} "
            f"SOLO_RESERVE_LAST={mod.OPENING_SOLO_HQ_RESERVE_LAST_STEP}"
        )
        moves = mod.agent(raw, cfg)
        print(f"  moves ({len(moves)}): {moves!r}")
        ok, detail = _check_moves(
            mod, state, moves, args.min_send, args.neutral_ships, args.tol
        )
        if ok:
            print(f"PASS  {detail}")
            return 0
        print(f"FAIL  {detail}")
        return 1

    opponent = _load_opponent(args.opponent)

    if args.scan_premoves or args.scan_seeds:
        seeds = (
            _parse_int_range(args.scan_seeds)
            if args.scan_seeds
            else range(int(args.seed), int(args.seed) + 1)
        )
        pms = (
            _parse_int_range(args.scan_premoves)
            if args.scan_premoves
            else range(int(args.premoves), int(args.premoves) + 1)
        )

        saw_ledger = False
        for sd in seeds:
            for pm in pms:
                obs, cfg, state, snap = _run_premoves_env(
                    mod,
                    seed=sd,
                    premoves=pm,
                    opponent_agent=opponent,
                    episode_steps=args.episode_steps,
                )
                hq = max(state.my_pl, key=lambda p: p.ships)
                if hq.ships < args.min_send:
                    continue
                saw_ledger = True
                code, detail = _probe_state(
                    mod,
                    args,
                    obs,
                    cfg,
                    state,
                    snap,
                    seed=sd,
                    premoves=pm,
                    verbose=not args.quiet,
                )
                if args.quiet:
                    st = (
                        "PASS"
                        if code == 0
                        else ("SKIP" if code == 2 else "FAIL")
                    )
                    print(
                        f"seed={sd} premoves={pm} HQ={hq.ships} "
                        f"surplus={snap.surplus[hq.id]}  {st}  {detail}"
                    )
                elif code != 0:
                    print(f"\n{'FAIL' if code == 1 else 'SKIP'}  {detail}")
                if code == 0:
                    if not args.quiet:
                        print(f"\nPASS  {detail}")
                    return 0

        if not saw_ledger:
            print(
                "Scan: no (seed, premoves) produced HQ ships >= min-send.\n"
                "  Widen --scan-premoves / --scan-seeds, or use --synthetic for logic."
            )
            return 1
        return 1

    obs, cfg, state, snap = _run_premoves_env(
        mod,
        seed=args.seed,
        premoves=args.premoves,
        opponent_agent=opponent,
        episode_steps=args.episode_steps,
    )
    code, detail = _probe_state(
        mod,
        args,
        obs,
        cfg,
        state,
        snap,
        seed=args.seed,
        premoves=args.premoves,
        verbose=True,
    )
    if code == 0:
        print(f"\nPASS  {detail}")
    elif code == 1:
        print(f"\nFAIL  {detail}")
    else:
        print(f"\n{detail}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
