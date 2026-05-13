#!/usr/bin/env python3
"""Orchestrate v21 rollout + learner with rich logging (JSONL + console + file)."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def _run_nv_smi() -> Optional[Dict[str, Any]]:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        parts = r.stdout.strip().split(",")
        if len(parts) < 3:
            return None
        return {
            "gpu_util_pct": float(parts[0].strip() or 0),
            "gpu_mem_used_mb": float(parts[1].strip() or 0),
            "gpu_mem_total_mb": float(parts[2].strip() or 0),
        }
    except Exception:
        return None


def _cpu_mem_sample() -> Dict[str, float]:
    out: Dict[str, float] = {}
    if psutil is None:
        return out
    out["cpu_pct"] = float(psutil.cpu_percent(interval=0.15))
    vm = psutil.virtual_memory()
    out["ram_used_pct"] = float(vm.percent)
    out["ram_used_gb"] = round(vm.used / (1024**3), 3)
    return out


def _log_resources(logger: logging.Logger, *, phase: str, iter_idx: int) -> None:
    """Sample CPU/RAM (psutil) and GPU (nvidia-smi); log one INFO line to train.log + stdout."""
    parts = [f"iter={iter_idx}", phase]
    cm = _cpu_mem_sample()
    if cm:
        parts.append(f"cpu={cm['cpu_pct']:.1f}%")
        parts.append(f"ram={cm['ram_used_gb']:.2f}GiB({cm['ram_used_pct']:.0f}%)")
    else:
        parts.append("cpu_ram=n/a(install_psutil)")
    gpu = _run_nv_smi()
    if gpu:
        parts.append(
            f"gpu_util={gpu['gpu_util_pct']:.0f}% "
            f"gpu_mem={gpu['gpu_mem_used_mb']:.0f}/{gpu['gpu_mem_total_mb']:.0f}MiB"
        )
    else:
        parts.append("gpu=n/a(no_nvidia-smi_or_no_gpu)")
    logger.info("resources | %s", " | ".join(parts))


def _aggregate_game_summaries(shard_games: List[Dict]) -> Dict[str, float]:
    ratios = []
    planets = []
    steps = []
    for g in shard_games:
        gs = g.get("game_summary") or {}
        if "final_my_ship_ratio" in gs:
            ratios.append(float(gs["final_my_ship_ratio"]))
        if "final_planet_ratio" in gs:
            planets.append(float(gs["final_planet_ratio"]))
        if "last_step" in gs:
            steps.append(float(gs["last_step"]))
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "mean_final_my_ship_ratio": mean(ratios),
        "mean_final_planet_ratio": mean(planets),
        "mean_last_step": mean(steps),
    }


def setup_logging(runs_dir: Path) -> logging.Logger:
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / "train.log"
    logger = logging.getLogger("v21_supervisor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    ap = argparse.ArgumentParser(description="v21 train supervisor")
    ap.add_argument("--runs-dir", default="runs/v21_lite")
    ap.add_argument("--tier", default="lite", choices=["lite", "pro", "ultra"])
    ap.add_argument("--submission", default="v20")
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games-per-worker", type=int, default=15)
    ap.add_argument("--learner-updates", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wait-secs", type=float, default=120.0)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--opponents", nargs="*", default=None)
    ap.add_argument("--opponent-mix", default=None)
    ap.add_argument("--rollout-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--quiet-rollout",
        action="store_true",
        help="Pass to rollout_worker: no per-game stdout/jsonl",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    logger = setup_logging(runs_dir)
    metrics_path = runs_dir / "metrics.jsonl"
    state_path = runs_dir / "supervisor_state.json"

    state: Dict[str, Any] = {"cumulative_train_sec": 0.0}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            pass
    cumulative_base = float(state.get("cumulative_train_sec", 0.0))

    _log_resources(logger, phase="supervisor_start", iter_idx=0)

    for it in range(1, args.iterations + 1):
        iter_t0 = time.time()
        logger.info(
            "=== iter %s/%s rollout tier=%s submission=%s workers=%s games/worker=%s ===",
            it,
            args.iterations,
            args.tier,
            args.submission,
            args.workers,
            args.games_per_worker,
        )
        _log_resources(logger, phase="before_rollout", iter_idx=it)
        logger.info(
            "Rollout ~%s games total (workers=%s × games/worker=%s). "
            "Per-game lines → this log / nohup log; machine lines → %s/rollout_progress_w*.jsonl",
            args.workers * args.games_per_worker,
            args.workers,
            args.games_per_worker,
            runs_dir,
        )
        rcmd = [
            args.python,
            str(ROOT / "tools" / "v21" / "rollout_worker_v21.py"),
            "--runs-dir",
            str(runs_dir),
            "--workers",
            str(args.workers),
            "--games-per-worker",
            str(args.games_per_worker),
            "--tier",
            args.tier,
            "--submission",
            args.submission,
            "--device",
            args.rollout_device,
        ]
        if args.opponent_mix:
            rcmd.extend(["--opponent-mix", args.opponent_mix])
        elif args.opponents:
            rcmd.append("--opponents")
            rcmd.extend(args.opponents)
        if args.quiet_rollout:
            rcmd.append("--quiet-rollout")
        r = subprocess.run(rcmd, cwd=str(ROOT))
        if r.returncode != 0:
            logger.error("rollout failed exit=%s", r.returncode)
            sys.exit(r.returncode)

        _log_resources(logger, phase="after_rollout", iter_idx=it)

        # Collect shard stats before learner eats them
        shard_games: List[Dict] = []
        for shard in sorted(runs_dir.glob("shard_w*.msgpack")):
            try:
                import msgpack
                with open(shard, "rb") as f:
                    payload = msgpack.unpack(f, raw=False)
                shard_games.extend(payload.get("games", []))
            except Exception as e:
                logger.warning("peek shard %s: %s", shard, e)
        gs_agg = _aggregate_game_summaries(shard_games)

        logger.info(
            "Learner step: consuming shards in %s (this prints upd/... lines to stdout)",
            runs_dir,
        )
        lcmd = [
            args.python,
            str(ROOT / "tools" / "v21" / "learner_v21.py"),
            "--runs-dir",
            str(runs_dir),
            "--tier",
            args.tier,
            "--updates",
            str(args.learner_updates),
            "--lr",
            str(args.lr),
            "--wait-secs",
            str(args.wait_secs),
        ]
        r = subprocess.run(lcmd, cwd=str(ROOT))
        if r.returncode != 0:
            logger.error("learner failed exit=%s", r.returncode)
            sys.exit(r.returncode)

        _log_resources(logger, phase="after_learner", iter_idx=it)

        learner_stats_path = runs_dir / "last_learner_stats.json"
        lstats: Dict[str, Any] = {}
        if learner_stats_path.is_file():
            try:
                lstats = json.loads(learner_stats_path.read_text())
            except Exception:
                pass

        iter_wall = time.time() - iter_t0
        cumulative = cumulative_base + iter_wall
        cumulative_base = cumulative

        row: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "iter": it,
            "iter_wall_sec": round(iter_wall, 3),
            "cumulative_train_sec": round(cumulative, 3),
            "tier": args.tier,
            "submission": args.submission,
            "lr": args.lr,
            "games_this_iter": lstats.get("games"),
            "transitions": lstats.get("transitions"),
            "policy_loss": lstats.get("policy_loss"),
            "value_loss": lstats.get("value_loss"),
            "entropy": lstats.get("entropy"),
            "cumulative_train_sec": round(cumulative, 3),
        }
        row.update(_cpu_mem_sample())
        gpu = _run_nv_smi()
        if gpu:
            row.update(gpu)
        row.update(gs_agg)

        with open(metrics_path, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")

        state_path.write_text(
            json.dumps({"cumulative_train_sec": cumulative, "last_row": row}, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "iter %s done wall=%.1fs cum_train=%.1fs games=%s tr=%s pi=%s v=%s "
            "ship_ratio_mean=%.3f planet_ratio_mean=%.3f cpu=%s gpu=%s",
            it,
            iter_wall,
            cumulative,
            row.get("games_this_iter"),
            row.get("transitions"),
            row.get("policy_loss"),
            row.get("value_loss"),
            row.get("mean_final_my_ship_ratio", 0.0),
            row.get("mean_final_planet_ratio", 0.0),
            row.get("cpu_pct", "na"),
            row.get("gpu_util_pct", "na"),
        )

    logger.info("=== supervisor finished %s iters ===", args.iterations)


if __name__ == "__main__":
    main()
