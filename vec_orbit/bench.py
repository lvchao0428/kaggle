"""Throughput benchmark for vec_orbit.BatchedOrbitEnv (vectorized steps on GPU/CPU)."""

from __future__ import annotations

import argparse
import time

import torch

from vec_orbit import BatchedOrbitEnv


def _mem_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--planets", type=int, default=12)
    p.add_argument("--fleets", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu | cuda | cuda:0 (default: cuda if available else cpu)",
    )
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    if args.device:
        dev = torch.device(args.device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    env = BatchedOrbitEnv(
        batch=args.batch,
        max_planets=args.planets,
        max_fleets=args.fleets,
        max_steps=args.max_steps,
        device=dev,
    )
    B = args.batch
    actions = torch.rand(B, 2, 3, device=dev, dtype=torch.float32)

    for _ in range(args.warmup):
        env.reset()
        for _ in range(min(5, args.steps)):
            env.step(actions)

    env.reset()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    n_steps = 0
    for _ in range(args.steps):
        env.step(actions)
        n_steps += 1
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0

    sps = n_steps / elapsed if elapsed > 0 else 0.0
    env_sps = sps * B
    peak = _mem_mb(dev)
    print(f"device={dev} batch={B} planets={args.planets} fleets={args.fleets} rollout_steps={n_steps}")
    print(f"wall_sec={elapsed:.4f}  steps/s={sps:,.0f}  env_steps/s={env_sps:,.0f}")
    if dev.type == "cuda":
        print(f"cuda_peak_mem_MiB={peak:,.1f}")


if __name__ == "__main__":
    main()
