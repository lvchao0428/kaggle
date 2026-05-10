"""Distill the trained PolicyValueNet's value head into a v11-compatible
NeuralVal NumPy MLP (14 -> 64 -> 32 -> 1, tanh output).

This way v12 only needs to swap `_NEURAL_WEIGHTS_B64` — no other code change.

Usage::

    python3.12 tools/distill_to_numpy.py \\
        --checkpoint runs/exp1/policy_<latest>.pth \\
        --shards-dir runs/exp1 \\
        --out-b64 runs/exp1/v12_neural_b64.txt \\
        --epochs 80
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import List

import msgpack
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.feature_extractor import N_STATE_FEATURES
from tools.policy_torch import PolicyValueNet, best_device


# Same shape as v11.NeuralVal: 14 -> 64 -> 32 -> 1 with ReLU + tanh.
class StudentMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_STATE_FEATURES, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return torch.tanh(self.fc3(h2))


def collect_state_targets(shards_dir: Path, teacher: PolicyValueNet,
                           device: torch.device) -> tuple:
    """Read every shard (without deleting) for state features + teacher V(state)
    targets. Skip shards that are too small."""
    xs: List[np.ndarray] = []
    for shard in sorted(shards_dir.glob("shard_w*.msgpack")):
        try:
            with open(shard, "rb") as f:
                payload = msgpack.unpack(f, raw=False)
        except Exception as e:
            print(f"  skipped {shard}: {e}")
            continue
        for game in payload.get("games", []):
            for t in game["transitions"]:
                xs.append(np.asarray(t["state_feat"], dtype=np.float32))
    if not xs:
        return None, None
    X = np.stack(xs)  # (N, 14)
    X_t = torch.tensor(X, device=device, dtype=torch.float32)
    # Teacher targets: pad plan-feature dims with zeros, take value head.
    pad = torch.zeros(X_t.size(0), 17, device=device, dtype=torch.float32)
    feats31 = torch.cat([X_t, pad], dim=1)
    with torch.no_grad():
        v, _ = teacher(feats31)
    Y = v.squeeze(-1)
    return X_t, Y


def encode_b64(student: StudentMLP) -> str:
    """Match v11's expected format: dict[str, ndarray] under np.save."""
    sd = student.state_dict()
    payload = {
        "W1": sd["fc1.weight"].detach().cpu().numpy().astype(np.float32),
        "b1": sd["fc1.bias"].detach().cpu().numpy().astype(np.float32),
        "W2": sd["fc2.weight"].detach().cpu().numpy().astype(np.float32),
        "b2": sd["fc2.bias"].detach().cpu().numpy().astype(np.float32),
        "W3": sd["fc3.weight"].detach().cpu().numpy().astype(np.float32),
        "b3": sd["fc3.bias"].detach().cpu().numpy().astype(np.float32),
    }
    buf = io.BytesIO()
    np.save(buf, payload)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shards-dir", required=True)
    ap.add_argument("--out-b64", default="v12_neural_b64.txt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = best_device()
    print(f"distill device: {device}")

    teacher = PolicyValueNet().to(device)
    teacher.load_state_dict(torch.load(args.checkpoint, map_location=device))
    teacher.eval()

    X, Y = collect_state_targets(Path(args.shards_dir), teacher, device)
    if X is None:
        print("No transitions found in shards-dir.")
        return
    print(f"Distillation set: {X.size(0)} samples")

    student = StudentMLP().to(device)
    opt = Adam(student.parameters(), lr=args.lr)
    n = X.size(0)
    for ep in range(args.epochs):
        idx = torch.randperm(n, device=device)
        total = 0.0
        steps = 0
        for s in range(0, n, args.batch):
            mb = idx[s:s + args.batch]
            pred = student(X[mb]).squeeze(-1)
            loss = ((pred - Y[mb]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep+1:3d}/{args.epochs}  loss={total/max(steps,1):.4f}")

    b64 = encode_b64(student)
    out_path = Path(args.out_b64)
    out_path.write_text(b64)
    print(f"Wrote {out_path}  ({len(b64)} chars)")


if __name__ == "__main__":
    main()
