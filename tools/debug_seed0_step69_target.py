#!/usr/bin/env python3
"""One-off: why v20 skips the big corner neutral on seed 0 ~step 69."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission_resolve import resolve_submission_path


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def main():
    v20 = _load("v20dbg", ROOT / "submission_v20.py")
    v18 = _load("v18dbg", resolve_submission_path(ROOT, "v18"))

    stop = 69
    env = make("orbit_wars", debug=False, configuration={"seed": 0})
    env.reset()
    step = 0
    while step < stop and not env.done:
        cfg = env.configuration
        env.step(
            [
                v20.agent(env.state[0].observation, cfg),
                v18.agent(env.state[1].observation, cfg),
            ]
        )
        step += 1

    obs = env.state[0].observation
    cfg = env.configuration
    st = v20.GameState(obs, cfg)
    pol = v20.PhasePolicy.for_state(st)
    snap = v20.Snapshot.build(st, pol)

    cfg_d = dict(cfg) if hasattr(cfg, "keys") else {}
    spawn = cfg_d.get("spawn_positions", [])
    rg = v20.RegionalGraph(st.planets, list(spawn) if spawn else None)

    others = [p for p in st.neu_pl if abs(p.ships - 45) < 1]
    dst = min(others, key=lambda p: p.x + p.y)

    print(f"step={st.step} phase={st.phase()} my_id={st.my_id}")
    print(
        "target",
        f"id={dst.id} pos=({dst.x:.1f},{dst.y:.1f}) prod={dst.production} ships={dst.ships}",
    )

    for src in st.my_pl:
        base, need, eta = v20.target_score(snap, src, dst)
        adj = v20.regional_capture_adjustment(snap, src, dst, rg, eta)
        sc, _, _ = v20.capture_edge_score(snap, src, dst, rg)
        sun = v20.point_segment_distance(
            v20.SUN_X, v20.SUN_Y, src.x, src.y, dst.x, dst.y
        )
        rs, rd = rg.planet_to_region.get(src.id), rg.planet_to_region.get(dst.id)
        print(
            f"  src={src.id} reg={rs} dst_reg={rd} avail={snap.avail(src.id)} "
            f"eta={eta} need={need} base={base:.1f} adj={adj:.1f} edge={sc:.1f} "
            f"sun_chord={sun:.2f}"
        )

    best = max(
        (v20.capture_edge_score(snap, s, dst, rg)[0] for s in st.my_pl), default=-999.0
    )
    print(f"BEST capture_edge={best:.2f}  ranked_gate(best>-31)={best > -31.0}")

    plan = v20._build_capture_plan(snap, "expand", regional_graph=rg)
    print("expand plan dst ids:", sorted({a[1] for a in plan.actions}))

    ranked: list[tuple[float, int, int, int]] = []
    for p in st.neu_pl:
        best_sc = max(
            (v20.capture_edge_score(snap, s, p, rg)[0] for s in st.my_pl),
            default=-1e18,
        )
        if best_sc > -31.0:
            ranked.append((best_sc, p.id, p.ships, p.production))
    ranked.sort(reverse=True)
    print("top neutrals:", ranked[:12])
    idx = next((i for i, t in enumerate(ranked) if t[1] == dst.id), None)
    print(f"target rank among neutrals passing -31 gate: {idx}")


if __name__ == "__main__":
    main()
