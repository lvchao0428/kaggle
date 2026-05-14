"""Distill PolicyValueNet value head into target submission NeuralVal (variable width)."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import sys
from pathlib import Path
from typing import List, Tuple

import msgpack
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v21.feature_extractor_v20 import N_FEATURES, N_STATE_FEATURES
from tools.v21.nets import build_net, best_device


def _load_target_shapes(target_py: Path) -> Tuple[int, int, int]:
    mod_key = f"target_nv_sub__{target_py.stem}"
    spec = importlib.util.spec_from_file_location(mod_key, target_py)
    if spec is None or spec.loader is None:
        raise ImportError(target_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    spec.loader.exec_module(mod)
    nv = mod.NeuralVal()
    h1, nf = nv.W1.shape
    h2, _ = nv.W2.shape
    if nf != nv.N_FEAT:
        raise ValueError("NeuralVal N_FEAT mismatch")
    return int(nf), int(h1), int(h2)


class StudentMLP(nn.Module):
    def __init__(self, n_feat: int, h1: int, h2: int):
        super().__init__()
        self.fc1 = nn.Linear(n_feat, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return torch.tanh(self.fc3(h2))


def collect_state_targets(
    shards_dir: Path, teacher: torch.nn.Module, device: torch.device, plan_pad: int
):
    xs: List[np.ndarray] = []
    shard_paths = sorted(shards_dir.glob("shard_w*.msgpack"))
    arch = shards_dir / "shard_archive"
    if arch.is_dir():
        shard_paths.extend(sorted(arch.glob("shard_w*.msgpack")))
    shard_paths = sorted(set(shard_paths))
    for shard in shard_paths:
        try:
            with open(shard, "rb") as f:
                payload = msgpack.unpack(f, raw=False)
        except Exception as e:
            print(f"  skip {shard}: {e}")
            continue
        for game in payload.get("games", []):
            for t in game["transitions"]:
                xs.append(np.asarray(t["state_feat"], dtype=np.float32))
    if not xs:
        return None, None
    X = np.stack(xs)
    X_t = torch.tensor(X, device=device, dtype=torch.float32)
    pad = torch.zeros(X_t.size(0), plan_pad, device=device, dtype=torch.float32)
    need = N_FEATURES - N_STATE_FEATURES
    if plan_pad != need:
        raise ValueError(f"plan_pad {plan_pad} != expected {need}")
    feats_full = torch.cat([X_t, pad], dim=1)
    with torch.no_grad():
        v, _ = teacher(feats_full)
    Y = v.squeeze(-1)
    return X_t, Y


def encode_b64(student: StudentMLP) -> str:
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
    need_pad = N_FEATURES - N_STATE_FEATURES

    ap = argparse.ArgumentParser(description="Distill into v21_* NeuralVal")
    ap.add_argument("--checkpoint", required=True, help="Teacher policy .pth")
    ap.add_argument("--teacher-tier", default="lite", choices=["lite", "pro", "ultra"])
    ap.add_argument("--target-submission", required=True, type=Path)
    ap.add_argument("--shards-dir", required=True, type=Path)
    ap.add_argument("--out-b64", type=Path, default=Path("neural_weights.b64.txt"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = best_device()
    nf, h1, h2 = _load_target_shapes(args.target_submission)
    if nf != N_STATE_FEATURES:
        raise SystemExit(f"Target N_FEAT {nf} != pipeline {N_STATE_FEATURES}")

    teacher = build_net(args.teacher_tier).to(device)
    try:
        teacher.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    except TypeError:
        teacher.load_state_dict(torch.load(args.checkpoint, map_location=device))
    teacher.eval()

    X, Y = collect_state_targets(args.shards_dir, teacher, device, need_pad)
    if X is None:
        print("No samples in shards-dir.")
        return
    print(f"students: {X.size(0)}  target MLP {nf}->{h1}->{h2}->1")

    student = StudentMLP(nf, h1, h2).to(device)
    opt = Adam(student.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        perm = torch.randperm(X.size(0), device=device)
        loss_acc = 0.0
        for start in range(0, X.size(0), args.batch):
            idx = perm[start : start + args.batch]
            pred = student(X[idx]).squeeze(-1)
            loss = torch.mean((pred - Y[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_acc += float(loss.item())
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}/{args.epochs} mse~{loss_acc:.6f}")

    b64 = encode_b64(student)
    args.out_b64.write_text(b64, encoding="utf-8")
    print(f"Wrote {args.out_b64} — paste into target _NEURAL_WEIGHTS_B64")


if __name__ == "__main__":
    main()
