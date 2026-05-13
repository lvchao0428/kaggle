# vec_orbit — GPU-batched Orbit-style simulator (experimental)

This package is **not** a bit-accurate reimplementation of Kaggle `orbit_wars`. It is a **small, vectorized** two-player toy that reuses board geometry and fleet-speed scaling from [`orbit_wars_bot/geom.py`](../orbit_wars_bot/geom.py) so you can stress **batched `step` throughput on CUDA**.

- **v22（可提交桥接）**：`submission_v22_*` + `tools/distill_vec_bridge_v22.py` + 真环境 `shard_w*.msgpack`，一键 **`scripts/train_v22_submit.sh`**（见 [PIPELINE.md §6](PIPELINE.md)）。

**Full I/O, tensor layout, and submission paths (v21 vs v22):** see **[PIPELINE.md](PIPELINE.md)**.

## What is implemented (v1)

- **B** independent games in lockstep; state is tensors on one `torch.device`.
- **P** planet slots, **F** in-flight fleet slots; CPU `reset` samples layouts from per-env seeds (NumPy), then dynamics run on GPU/CPU.
- Planets are **fixed** for the episode; **production** each step; **fleets** fly straight to a target planet; **sun / board edge** kills; **arrival**: reinforce, neutral capture, or battle.
- **Sparse reward** for player 0: `+1` / `-1` on terminal elimination; `0` on timeout or draw.
- **Training reference:** [`train_loop.py`](train_loop.py) + [`policy.py`](policy.py) + [`action_utils.py`](action_utils.py).

## What is not implemented

Kaggle fidelity: 4-player FFA, comets, `PlanArbiter`, diplomacy, orbital motion, exact scoring, etc.

## Benchmark

From repo root:

```bash
python -m vec_orbit.bench --batch 8192 --steps 200 --device cuda
python -m vec_orbit.bench --batch 1024 --steps 100 --device cpu
```

`env_steps/s` = `batch * steps/s`.

## Train a toy policy (GPU)

**一键脚本**（仓库根目录；日志进 `logs/`，权重进 `runs/vec_orbit/`，可用环境变量覆盖默认值）：

```bash
chmod +x scripts/train_vec_orbit.sh
./scripts/train_vec_orbit.sh
# 例如：BATCH=8192 UPDATES=800 HORIZON=64 DEVICE=cuda SEED=42 ./scripts/train_vec_orbit.sh
```

直接调模块（列说明见运行首行打印）：

```bash
python -m vec_orbit.train_loop --device cuda --batch 4096 --horizon 64 --updates 500 \
  --log-every 1 --seed 0 --out runs/vec_orbit/policy_actor_critic.pth
```

This checkpoint is **not** the v21 submission format; see [PIPELINE.md](PIPELINE.md) for the path to `_NEURAL_WEIGHTS_B64`.

## Programmatic use

```python
import torch
from vec_orbit import BatchedOrbitEnv
from vec_orbit.policy import ActorCritic
from vec_orbit.action_utils import raw_vec_to_actions

env = BatchedOrbitEnv(batch=4096, max_planets=12, max_fleets=32, device=torch.device("cuda"))
obs = env.reset()
net = ActorCritic(env.obs_dim).to(env.device)
pre, _, _ = net.act(obs)
actions = raw_vec_to_actions(pre, env.P)
obs, reward_p0, done, info = env.step(actions)
```

## Submission file generation (Kaggle-faithful)

Use **real** `orbit_wars` rollouts + v21 learner + distill — summarized in [PIPELINE.md §5](PIPELINE.md).
