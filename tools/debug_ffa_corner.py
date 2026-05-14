#!/usr/bin/env python3
"""FFA 单 seed 调试：复现 replay 环境，打印右下角口袋灰星打分、expand/aggro 方案排序与 baseline gate。

用法（与 replay 一致）::

    cd kaggle && python3 tools/debug_ffa_corner.py --seed 1619309859 --step 122

默认会 **屏蔽 OpenSpiel 的 INFO**（``logging.disable(INFO)``），并在导入 ``kaggle_environments.make``、
以及 ``make()+reset()`` 时用 Python 层 ``redirect_stderr`` 吞常见噪声。
若仍看到 ``Loading environment cabt failed``，多为 **C 扩展直接写 fd2**，Python 无法拦截，可：
``python3 tools/debug_ffa_corner.py ... 2>/dev/null`` 或修/reinstall 与架构匹配的 ``kaggle_environments``。

依赖 kaggle_environments。每一步用 seat0 的 v20 更新 submission 内 ``_GLOBAL_OPP``，使 Diplomacy 与实战一致。
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission_resolve import resolve_submission_path


def _quiet_kaggle_loggers() -> None:
    """Best-effort: kaggle/spiel loggers vary by package version."""
    root = logging.getLogger()
    root.setLevel(logging.ERROR)
    logging.disable(logging.INFO)
    for name in (
        "kaggle_environments",
        "kaggle_environments.envs.open_spiel",
        "kaggle_environments.envs.open_spiel_env",
        "kaggle_environments.envs.open_spiel_env.open_spiel_env",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
        logging.getLogger(name).propagate = False


def _infer_move_target(
    s20,
    state,
    src_id: int,
    angle: float,
    ships: int,
):
    """First planet hit by synthetic fleet (``launch_origin`` + ``_predict_fleet_target``)."""
    src = state.get(src_id)
    if src is None:
        return None, None, None
    lx, ly = s20.launch_origin(src, angle)
    f = s20.Fleet(9_000_001, state.my_id, lx, ly, angle, src_id, ships)
    pred = state._predict_fleet_target(f, max_steps=220)
    if pred is None:
        return None, None, None
    pid, eta = pred[0], pred[1]
    tgt = state.get(pid)
    return pid, eta, tgt


def _load_agent(version: str):
    if version == "random":
        return "random"
    path = resolve_submission_path(ROOT, version)
    spec = importlib.util.spec_from_file_location(f"sub_{version}_dbg", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"sub_{version}_dbg"] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def main() -> int:
    p = argparse.ArgumentParser(description="FFA corner-neutral debug for submission_v20")
    p.add_argument("--seed", type=int, default=1619309859)
    p.add_argument("--step", type=int, default=122, help="Pause and introspect after this many env steps")
    p.add_argument("--min-x", type=float, default=52.0, help="Bottom-right filter: planet.x >= this")
    p.add_argument("--min-y", type=float, default=52.0, help="Bottom-right filter: planet.y >= this")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show kaggle/OpenSpiel INFO and stderr from importing kaggle_environments",
    )
    args = p.parse_args()

    if not args.verbose:
        logging.basicConfig(level=logging.ERROR, force=True)
        _quiet_kaggle_loggers()
    else:
        logging.basicConfig(level=logging.INFO, force=True)

    import submission_v20 as s20

    if not args.verbose:
        _stderr_backup = io.StringIO()
        with contextlib.redirect_stderr(_stderr_backup):
            from kaggle_environments import make
    else:
        from kaggle_environments import make

    ret = 0
    try:
        ret = _run_body(args, s20, make)
    finally:
        if not args.verbose:
            logging.disable(logging.NOTSET)
    return ret


def _run_body(args, s20, make) -> int:
    lineup = ["v20", "v19", "v19", "v19"]
    agents = [_load_agent(lab) for lab in lineup]

    if not args.verbose:
        with contextlib.redirect_stderr(io.StringIO()):
            env = make("orbit_wars", debug=False, configuration={"seed": int(args.seed)})
            env.reset(len(agents))
    else:
        env = make("orbit_wars", debug=False, configuration={"seed": int(args.seed)})
        env.reset(len(agents))

    cfg = env.configuration
    step = 0
    while not env.done and step < args.step:
        actions: list = []
        for i in range(4):
            ag = agents[i]
            obs = env.state[i].observation
            actions.append(ag(obs, cfg) if not isinstance(ag, str) else ag)
        env.step(actions)
        st = s20.GameState(env.state[0].observation, cfg)
        s20._GLOBAL_OPP.update(st)
        step += 1

    obs0 = env.state[0].observation
    state = s20.GameState(obs0, cfg)
    policy = s20.PhasePolicy.for_state(state)
    snap = s20.Snapshot.build(state, policy)
    diplo = s20.DiplomacyEngine(state, s20._GLOBAL_OPP)

    spawn_positions = cfg.get("spawn_positions", []) if cfg else []
    try:
        rg = s20.RegionalGraph(state.planets, spawn_positions)
    except Exception:
        rg = None

    print("=== FFA corner debug ===")
    print(f"seed={args.seed}  env_step={step}  episode_steps={state.episode_steps}")
    print(
        f"phase={state.phase()}  turns_left={state.turns_left()}  "
        f"spawn_positions_len={state.spawn_count}  is_ffa_mode={state.is_ffa_mode()} "
        f"en_ids={len(state.en_ids)}"
    )
    print(f"policy.mode_order={policy.mode_order}")
    print(f"my_id={state.my_id}  my worlds={len(state.my_pl)}  neutrals={len(state.neu_pl)}")
    print()

    # Pocket neutrals (default: bottom-right quadrant)
    pocket = [
        n for n in state.neu_pl
        if n.x >= args.min_x and n.y >= args.min_y
    ]
    pocket.sort(key=lambda q: (-q.production, q.ships))
    print(f"Neutrals in x>={args.min_x}, y>={args.min_y}: {len(pocket)}")
    for n in pocket[:12]:
        dm = min(m.dist(n) for m in state.my_pl)
        best_sc = -1e18
        best_src = None
        for m in state.my_pl:
            sc, need, eta = s20.capture_edge_score(snap, m, n, rg)
            if sc > best_sc:
                best_sc, best_src = sc, m
        print(
            f"  id={n.id:2d}  ships={n.ships:3d}  prod={n.production}  "
            f"pos=({n.x:.1f},{n.y:.1f})  d_min={dm:.1f}  best_edge_sc={best_sc:.2f}  "
            f"best_src={getattr(best_src, 'id', None)}"
        )
    print()

    # Top-ranked expand targets (same idea as _build_capture_plan ranking)
    targets = s20._target_pool(state, "expand", None)
    ranked: list[tuple[float, object]] = []
    floor = s20.EXPAND_RANK_SCORE_FLOOR
    for dst in targets:
        best_sc = -1e18
        for src in state.my_pl:
            sc, _, _ = s20.capture_edge_score(snap, src, dst, rg)
            if sc > best_sc:
                best_sc = sc
        if (
            state.is_ffa_mode()
            and dst.owner == -1
            and dst.production >= 2
        ):
            dmin = min((m.dist(dst) for m in state.my_pl), default=999.0)
            if dmin < 50.0:
                best_sc += 44.0
        if best_sc > floor:
            ranked.append((best_sc, dst))
    ranked.sort(key=lambda x: -x[0])
    print("Top 15 expand targets (after FFA pocket rank boost):")
    for sc, dst in ranked[:15]:
        mark = " *" if dst.owner == -1 and dst.x >= args.min_x and dst.y >= args.min_y else ""
        print(
            f"  sc={sc:8.2f}  id={dst.id:2d}  own={dst.owner:2d}  "
            f"ships={dst.ships:3d}  prod={dst.production}  "
            f"pos=({dst.x:.0f},{dst.y:.0f}){mark}"
        )
    print()

    # Arbiter strategic stack (after urgent), without final emit
    st2 = s20.GameState(obs0, cfg)
    policy2 = s20.PhasePolicy.for_state(st2)
    snap2 = s20.Snapshot.build(st2, policy2)
    diplo2 = s20.DiplomacyEngine(st2, s20._GLOBAL_OPP)
    rg2 = rg
    neural = s20._GLOBAL_NEURAL
    arb = s20.PlanArbiter(
        snap2,
        diplo2,
        neural,
        elapsed_ms_fn=lambda: 0.0,
        deadline_ms=920.0,
        regional_graph=rg2,
        multi_hop_planner=None,
    )
    arb.commit_urgent()
    plans = arb.collect_strategic()
    scored = arb.score_with_modifiers(plans)
    print(f"Strategic plans (after urgent commits): {len(plans)}  scored: {len(scored)}")
    for i, (comb, pl) in enumerate(scored[:8]):
        ship_sum = sum(a[2] for a in pl.actions)
        heads = pl.actions[:4]
        print(
            f"  #{i+1}  comb={comb:10.2f}  tag={pl.tag:10s}  "
            f"raw_sc={pl.score:8.2f}  ships={ship_sum:4d}  moves={len(pl.actions)}  head={heads}"
        )

    if scored:
        best_sc, best_pl = scored[0]
        baseline = s20.score_plan_actions(
            st2,
            [],
            steps=policy2.sim_steps,
            tempo_floor=policy2.tempo_floor,
        )
        margin = float(policy2.baseline_commit_margin)
        commit_bonus = 0.0
        if best_pl.tag in ("expand", "balanced") and best_pl.actions:
            if any(
                st2.get(a[1]) is not None and st2.get(a[1]).owner == -1
                for a in best_pl.actions
            ):
                commit_bonus = 6.0
        would_commit = st2.phase() == "early" or (best_sc + commit_bonus > baseline + margin)
        print()
        print("commit_best gate (late/mid):")
        print(f"  baseline(empty sim)={baseline:.4f}  margin={margin}  commit_bonus={commit_bonus}")
        print(f"  best combined={best_sc:.4f}  early_always={st2.phase() == 'early'}")
        print(f"  => would_commit_strategic={would_commit}  (if False, only urgent+fallback fire)")

    moves = s20.agent(obs0, cfg)
    print()
    print(f"Full agent moves this turn: {len(moves)}")
    for mv in moves[:30]:
        sid, ang, ships = mv[0], mv[1], mv[2]
        src = state.get(sid)
        pid, eta, tgt = _infer_move_target(s20, state, sid, ang, ships)
        spos = f"src_pos=({src.x:.1f},{src.y:.1f})" if src else "src_pos=?"
        prod = src.production if src else -1
        extra = ""
        if tgt is not None:
            who = "neu" if tgt.owner == -1 else f"p{tgt.owner}"
            extra = f"  -> dst={pid} eta~{eta}  {who} ships={tgt.ships} prod={tgt.production} pos=({tgt.x:.0f},{tgt.y:.0f})"
        elif pid is None:
            extra = "  -> (no planet hit in sweep / OOB or sun)"
        print(
            f"  src={sid}  ships={ships:3d}  angle={ang:.4f}  "
            f"{spos}  prod={prod}{extra}"
        )
    if len(moves) > 30:
        print(f"  ... ({len(moves) - 30} more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
