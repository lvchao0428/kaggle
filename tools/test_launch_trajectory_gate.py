#!/usr/bin/env python3
"""Offline smoke tests for v20 OOB / launch trajectory gate.

Run from repo ``kaggle/`` (same directory as ``submission_v20.py``):

  python3 tools/test_launch_trajectory_gate.py

This is **not** a game replay: it builds a toy ``GameState`` and checks that
``launch_hits_target_first`` and ``PlanArbiter._emit`` behave as expected **without**
installing ``kaggle_environments``.  Full-game OOB stats: ``tools/aim_trainer.py``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import submission_v20 as v20


def _banner(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg)


def _minimal_state():
    """Static lane y=30, two planets; avoids sun; good for cheap geometry checks."""
    obs = {
        "player": 0,
        "step": 0,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
        "initial_planets": [
            [0, 0, 5.0, 30.0],
            [1, -1, 80.0, 30.0],
        ],
        "planets": [
            [0, 0, 5.0, 30.0, 3.0, 100, 5],
            [1, -1, 80.0, 30.0, 3.0, 10, 2],
        ],
        "fleets": [],
        "configuration": {"shipSpeed": 6.0, "episodeSteps": 500},
    }
    return v20.GameState(obs, obs["configuration"])


def _run_case(quiet: bool, title: str, explain: str, fn) -> None:
    if not quiet:
        print(f"  [{title}]")
        print(f"      {explain}")
    fn()
    if not quiet:
        print("      → pass")
        print()


def test_hits_intended_target() -> None:
    st = _minimal_state()
    sx, sy = 5.0, 30.0
    ang = 0.0
    assert v20.launch_hits_target_first(st, sx, sy, ang, 50, 1, ignore_planet_id=0)


def test_wrong_target_id_fails() -> None:
    st = _minimal_state()
    sx, sy = 5.0, 30.0
    ang = 0.0
    assert not v20.launch_hits_target_first(st, sx, sy, ang, 50, 0, ignore_planet_id=0)


def test_oob_before_any_target() -> None:
    st = _minimal_state()
    sx, sy = 5.0, 30.0
    ang = -math.pi / 2
    assert not v20.launch_hits_target_first(st, sx, sy, ang, 20, 1, ignore_planet_id=0)


_emit_debug: tuple | None = None


def test_emit_accepts_good_launch() -> None:
    global _emit_debug
    st = _minimal_state()
    pol = v20.PhasePolicy.for_state(st)
    snap = v20.Snapshot.build(st, pol)
    arb = v20.PlanArbiter(
        snap,
        v20.DiplomacyEngine(st, v20._GLOBAL_OPP),
        v20._GLOBAL_NEURAL,
        elapsed_ms_fn=lambda: 0.0,
        deadline_ms=10000.0,
        regional_graph=None,
        multi_hop_planner=None,
    )
    ok = arb._emit(0, 1, 40, urgent=False)
    assert ok
    assert len(arb.moves) == 1
    sid, angle, ships = arb.moves[0]
    assert sid == 0 and ships > 0
    lx, ly = v20.launch_origin(st.get(0), angle)
    assert v20.launch_hits_target_first(st, lx, ly, angle, ships, 1,
                                       ignore_planet_id=None)
    _emit_debug = (angle, ships, lx, ly)


def main() -> None:
    global _emit_debug
    ap = argparse.ArgumentParser(
        description="Smoke-test v20 launch / OOB gate without kaggle_environments.",
    )
    ap.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="only print final status line (or nothing on success with exit 0)",
    )
    args = ap.parse_args()
    quiet: bool = args.quiet
    _emit_debug = None  # reset module-level between runs

    if not quiet:
        print("test_launch_trajectory_gate — v20 launch / OOB gate smoke tests")
        print("=" * 60)
        print(
            "用途: 在**不跑 orbit_wars** 的前提下，用极简地图验证submission_v20里\n"
            "  • launch_hits_target_first（swept 碰撞 + 首碰目标是否为目标星）\n"
            "  • launch_origin（引擎一致 radius+0.1）\n"
            "  • PlanArbiter._emit 最终门闩是否放行一条合法发往行星 1 的 move\n"
            "完整对局统计请用: python3 tools/aim_trainer.py --version v20 ...\n"
        )

    cases = [
        (
            "hit_target",
            "水平向东：首碰行星 1（忽略源星 0 重叠），应通过。",
            test_hits_intended_target,
        ),
        (
            "wrong_target",
            "同样弹道但声称目标为行星 0：首碰仍是 1，应拒绝。",
            test_wrong_target_id_fails,
        ),
        (
            "oob_miss",
            "垂直向北飞出棋盘：在撞到目标前应失败。",
            test_oob_before_any_target,
        ),
        (
            "emit_pipeline",
            "走 _emit(0→1, 40)：应产生 1 条 move，且从 launch_origin 仿真仍首碰 1。",
            test_emit_accepts_good_launch,
        ),
    ]

    for title, explain, fn in cases:
        _run_case(quiet, title, explain, fn)

    if not quiet and _emit_debug is not None:
        ang, ships, lx, ly = _emit_debug
        deg = math.degrees(ang)
        print("  [_emit 样例输出]")
        print(f"      angle={ang:.4f} rad ({deg:.1f}°), ships={ships}, "
              f"launch_origin=({lx:.2f}, {ly:.2f})")
        print()

    msg = "全部通过 (4 cases)。改 safe_aim / launch_hits / swept 后可再跑本脚本做冒烟。"
    if quiet:
        print(msg)
    else:
        print("=" * 60)
        print(msg)


if __name__ == "__main__":
    main()
