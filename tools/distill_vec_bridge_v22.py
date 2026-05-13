"""Distill vec_orbit ActorCritic value + learned encoder(state_feat→obs) into v22 NeuralVal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import msgpack
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.distill_to_numpy_v21 import StudentMLP, encode_b64, _load_target_shapes
from tools.v21.feature_extractor_v20 import N_STATE_FEATURES
from tools.v21.nets import best_device
from vec_orbit.policy import ActorCritic


def load_vec_actor_critic(path: Path, device: torch.device) -> tuple[ActorCritic, int, int]:
    try:
        sd = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location=device)
    obs_dim = int(sd["trunk.0.weight"].shape[1])
    hidden = int(sd["trunk.0.weight"].shape[0])
    net = ActorCritic(obs_dim, hidden=hidden).to(device)
    net.load_state_dict(sd)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, obs_dim, hidden


def vec_value_head(net: ActorCritic, obs: torch.Tensor) -> torch.Tensor:
    h = net.trunk(obs)
    return net.v(h).squeeze(-1)


class StateToObsEncoder(nn.Module):
    """Map real-game state_feat (14) to vec_orbit observation space."""

    def __init__(self, n_in: int, n_out: int, enc_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collect_state_feats(shards_dir: Path) -> torch.Tensor:
    rows: list[np.ndarray] = []
    for shard in sorted(shards_dir.glob("shard_w*.msgpack")):
        try:
            with open(shard, "rb") as f:
                payload = msgpack.unpack(f, raw=False)
        except OSError:
            continue
        for game in payload.get("games", []):
            for t in game.get("transitions", []):
                rows.append(np.asarray(t["state_feat"], dtype=np.float32))
    if not rows:
        raise SystemExit(f"No transitions in {shards_dir}/shard_w*.msgpack")
    return torch.tensor(np.stack(rows, axis=0), dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vec-checkpoint", type=Path, required=True, help="vec_orbit train_loop .pth")
    ap.add_argument("--shards-dir", type=Path, required=True, help="Directory with shard_w*.msgpack (real env)")
    ap.add_argument("--target-submission", type=Path, required=True)
    ap.add_argument("--out-b64", type=Path, default=Path("neural_weights_v22.b64.txt"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--enc-hidden", type=int, default=256)
    args = ap.parse_args()

    if not args.target_submission.name.startswith("submission_v22"):
        print("warning: v22 pipeline expects submission_v22_*.py", file=sys.stderr)

    device = best_device()
    vec_net, obs_dim, _hidden = load_vec_actor_critic(args.vec_checkpoint, device)

    nf, h1, h2 = _load_target_shapes(args.target_submission)
    if nf != N_STATE_FEATURES:
        raise SystemExit(f"Target N_FEAT {nf} != {N_STATE_FEATURES}")

    X_all = collect_state_feats(args.shards_dir).to(device)
    n = X_all.size(0)
    print(f"samples={n}  vec_obs_dim={obs_dim}  student {nf}->{h1}->{h2}->1")

    enc = StateToObsEncoder(N_STATE_FEATURES, obs_dim, args.enc_hidden).to(device)
    student = StudentMLP(nf, h1, h2).to(device)
    opt = Adam(list(enc.parameters()) + list(student.parameters()), lr=args.lr)

    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        loss_acc = 0.0
        for start in range(0, n, args.batch):
            idx = perm[start : start + args.batch]
            x = X_all[idx]
            obs_hat = enc(x)
            target = vec_value_head(vec_net, obs_hat)
            pred = student(x).squeeze(-1)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(student.parameters()), 1.0)
            opt.step()
            loss_acc += float(loss.item())
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{args.epochs} mse~{loss_acc:.6f}")

    b64 = encode_b64(student)
    args.out_b64.write_text(b64, encoding="utf-8")
    print(f"Wrote {args.out_b64} — paste into {args.target_submission} _NEURAL_WEIGHTS_B64")


if __name__ == "__main__":
    main()
