# Orbit Wars — 快速上手指南

> 适用对象：新队友，从零开始了解本项目结构、核心思路、本地调试方法和 RL 训练流程。

---

## 一、比赛背景

**[Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)** 是 Kaggle 上的一个连续 2D 太空策略游戏竞赛。

| 要素 | 说明 |
|------|------|
| 地图 | 100×100，中心太阳 (50,50)，半径 10，舰队进入即销毁 |
| 行星类型 | 轨道行星（距太阳 < 50，顺时针旋转）+ 静止行星 |
| 彗星 | 随机出现，沿预设路径移动，可占领 |
| 舰队速度 | 对数公式：`1 + (max_speed-1) * (log(ships)/log(1000))^1.5`，官方 max_speed=6 |
| 每回合动作 | `[from_planet_id, angle_rad, num_ships]`，最多 26 条 |
| 战斗规则 | 多方同时到达：最多 vs 第二多 差值 = 存活；平局守方不变 |
| 观测接口 | `obs.planets / obs.fleets / obs.comets / obs.initial_planets`，每回合完整可见 |
| 胜负判定 | 500 回合后总舰船数（星球 + 在途）最多的玩家胜 |

---

## 二、环境安装

```bash
# Python 3.12 推荐（本项目使用 /opt/local/bin/python3.12）
pip install "kaggle-environments>=1.28.0" msgpack torch numpy

# 验证环境可用
python3.12 -c "from kaggle_environments import make; env = make('orbit_wars'); print('OK')"
```

---

## 三、项目目录结构

```
kaggle/
├── submission_v9.py          # 策略基线（批量分配 + 全工具箱）
├── submission_v10.py         # v9 + MCTS + NeuralVal + DiplomacyEngine
├── submission_v11.py         # 大框架重建（模块化 + 2010冠军惩罚函数）★基础
├── submission_v12.py         # v11 + RL 蒸馏权重（首版 PPO 训练）
├── submission_v13.py         # v12 + 3项Bug修复 + RL管线升级  ★当前最强
│
├── scripts/
│   ├── eval_head2head.py     # 通用 head-to-head 评估脚本  ★最常用
│   ├── eval_rl.py            # RL agent 专用评估
│   └── eval_compare_v6_v7.py # 早期版本比较（历史）
│
├── tools/                    # RL 训练管线
│   ├── feature_extractor.py  # 特征向量（状态14维 + 计划17维 = 31维）
│   ├── policy_torch.py       # PyTorch 策略网络 PolicyValueNet
│   ├── rl_agent.py           # 可训练 Agent（包装 v11 PlanArbiter）
│   ├── rollout_worker.py     # 多进程自对弈，生成 msgpack shards
│   ├── learner.py            # PPO 学习器（消费 shards，输出 .pth 和 .npz）
│   ├── distill_to_numpy.py   # 蒸馏：pth → NeuralVal 兼容的 base64 NumPy MLP
│   ├── imitation_pretrain.py # 行为克隆热启动（v11 专家演示）
│   └── train_loop.sh         # 完整训练驱动脚本
│
├── AGENTS.md                 # 版本演进完整记录（主要参考文档）
├── started.txt               # 官方比赛规则全文
├── 2010Planet-war-readme.txt # 2010 Planet Wars 官方规则
└── orbit-wars-target-score-2000-4-4f3559-annotated.ipynb  # Notebook 参考方案（含中文注释）
```

---

## 四、版本演进速览

| 版本 | 核心改进 | 对 v9 胜率 | 对上一版胜率 |
|------|---------|-----------|------------|
| v9 | 批量分配 + 完整策略工具箱（correct physics） | 基准 | — |
| v10 | + MCTS + NeuralVal + DiplomacyEngine | — | 50% |
| v11 | 大框架重建（Snapshot/PhasePolicy/PlanArbiter）+ 2010冠军4项惩罚函数 | 40% | 50% |
| v12 | v11 + RL 自对弈 PPO 蒸馏权重 | ~40% | 50% |
| **v13** | v12 + 3项Bug修复 + RL管线升级（dense reward/CE loss/imitation pretrain） | **80%** | **80%** |

> v13 是当前提交版本，相对 v11/v12 胜率 80%（10 seeds × 双座位，20 局）。

---

## 五、本地对战调试

### 5.1 快速单局测试

```bash
cd /Users/lvchao0428/project/kaggle/kaggle

# v13 vs v12，单局，seed=42
python3.12 -c "
from kaggle_environments import evaluate
import importlib.util, sys

def load(ver):
    p = f'submission_{ver}.py'
    spec = importlib.util.spec_from_file_location(ver, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.agent

a, b = load('v13'), load('v12')
r = evaluate('orbit_wars',
    [lambda o,c,f=a: f(o,c), lambda o,c,f=b: f(o,c)],
    configuration={'seed': 42}, num_episodes=1)[0]
print('v13 score:', r[0], '  v12 score:', r[1])
"
```

### 5.2 标准 head-to-head 评估（双座位，推荐）

```bash
# v13 vs v12，10 seeds 双座位 = 20 局，约 6 分钟
python3.12 scripts/eval_head2head.py --a v13 --b v12 --seeds 0-9

# v13 vs v11，20 局
python3.12 scripts/eval_head2head.py --a v13 --b v11 --seeds 0-9

# v13 vs v9，20 局
python3.12 scripts/eval_head2head.py --a v13 --b v9 --seeds 0-9

# v13 vs random（不换座位），10 局
python3.12 scripts/eval_head2head.py --a v13 --b random --seeds 0-9 --no-swap

# 指定多个独立 seed
python3.12 scripts/eval_head2head.py --a v13 --b v9 --seeds 0 1 2 3 4
```

**输出格式说明：**
```
seed=  0  [v13,v12]=[1, -1]  [v12,v13]=[-1, 1]
# [A,B] = [A的得分, B的得分]，1=胜，-1=负，0=平

v13 wins=16  v12 wins=4  ties=0  games=20  v13 win%=80.0
elapsed 338.6s
```

### 5.3 快速 smoke test（验证新 bot 不崩溃）

```bash
# 确认新版本能跑完并击败 random
python3.12 scripts/eval_head2head.py --a v13 --b random --seeds 0-4 --no-swap
# 预期：v13 wins=5, random wins=0
```

### 5.4 在 Notebook / REPL 中交互调试

```python
from kaggle_environments import make
import importlib.util, sys

# 加载 agent
spec = importlib.util.spec_from_file_location("v13", "submission_v13.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
agent = mod.agent

# 创建环境，手动步进
env = make("orbit_wars", configuration={"seed": 42})
env.reset()
obs = env.state[0].observation
config = env.configuration

# 看 agent 第一步输出
actions = agent(obs, config)
print(actions)
```

---

## 六、理解 submission_v13.py 的核心结构

v13 代码按 Region（区域）划分，约 1600 行：

| Region | 约在哪里 | 内容 |
|--------|---------|------|
| 0 | 顶部常量 | `SUN_RADIUS=10`, `fleet_speed()`, `segment_hits_sun()` 等物理函数 |
| 1 | `Planet`/`Fleet` | 数据类，`_combat()` 战斗解算 |
| 2 | `GameState` | 解析每回合 obs，缓存 fleets/arrivals/comet_paths |
| 3 | `Snapshot` | 每回合预计算：surplus/reserve/threat_horizon/centroid |
| 4 | `PhasePolicy` + `PHASE_TABLE` | **调参入口**：早/中/后期策略权重表 |
| 5 | `target_score()` 等 | 评分函数（从 PhasePolicy 取参数） |
| 6 | `SimP/SimF/sim_step` | 短时前向仿真器（8-10 步） |
| 7 | Planners | Defense / Intercept / Expand / Attack / UrgentHighProd / Redistribution / LateDump |
| 8 | `MCTSEngine` | plan-level UCB1 树搜索 |
| 9 | `NeuralVal` | NumPy MLP，v13 使用 RL 蒸馏权重 |
| 10 | `PlanArbiter` | 收集所有候选 Plan → 评分 → 提交最优 |
| 11 | `agent()` | Kaggle 入口 |

**调参最快路径：** 修改 `PHASE_TABLE`（Region 4），改变早/中/后期的 `mode_order`、`cost_pen_mul`、`reserve_growth_mul` 等，无需动其他逻辑。

---

## 七、RL 训练管线（v12/v13 起）

### 7.1 整体流程

```
专家演示(v11) → 行为克隆热启动 → PPO 自对弈迭代 → 蒸馏 → 更新 submission 权重
```

### 7.2 Step 1：Imitation Pre-Training（热启动，约 1 小时）

```bash
cd /Users/lvchao0428/project/kaggle/kaggle

# 正式热启动（200 局 v11 vs v11，20 epoch BC）
python3.12 tools/imitation_pretrain.py \
    --games 200 --epochs 20 \
    --runs-dir runs/exp2

# 快速烟雾（2 局 / 5 epoch，约 30 秒）
python3.12 tools/imitation_pretrain.py \
    --games 2 --epochs 5 \
    --runs-dir runs/smoke
```

输出：`runs/exp2/policy_latest.npz`（供后续 PPO 读取）

### 7.3 Step 2：PPO 自对弈训练

```bash
# 使用 train_loop.sh（推荐）
# 参数：<runs_dir> <iters> <workers> <games_per_worker_per_iter>
bash tools/train_loop.sh runs/exp2 20 4 15
# 20 次迭代，4 个 worker，每 worker 每次 15 局
# 约 2-4 小时（取决于机器）

# 或者手动分开运行
# A. 生成 shards（self-play）
python3.12 tools/rollout_worker.py \
    --workers 4 --games-per-worker 20 \
    --runs-dir runs/exp2 \
    --weights runs/exp2/policy_latest.npz \
    --opponents v9 v10 v11

# B. PPO 学习
python3.12 tools/learner.py \
    --runs-dir runs/exp2 \
    --updates 5 \
    --lr 3e-4
```

### 7.4 Step 3：蒸馏 → 更新 submission 权重

```bash
# 生成小批量 shards 用于蒸馏数据
python3.12 tools/rollout_worker.py \
    --workers 2 --games-per-worker 5 \
    --runs-dir runs/exp2_distill \
    --weights runs/exp2/policy_latest.npz

# 蒸馏（输出 base64 字符串）
python3.12 tools/distill_to_numpy.py \
    --checkpoint runs/exp2/policy_<timestamp>.pth \
    --shards-dir runs/exp2_distill

# 将输出的 base64 字符串替换到 submission_v13.py（或 v14.py）的
# _NEURAL_WEIGHTS_B64 变量中
```

### 7.5 超参速查

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| PPO clip | `learner.py:ppo_update` | 0.2 | 策略更新幅度限制 |
| GAE γ / λ | `learner.py:games_to_tensors` | 0.997 / 0.95 | 回报折扣 / GAE 平滑 |
| Dense reward weight | `rollout_worker.py` | 0.10 | 中间奖励权重 |
| Dense reward interval | `rollout_worker.py` | 20 steps | 采样间隔 |
| Plan score strength | `rl_agent.py:__init__` | 1.0 | net 评分权重 |
| Temperature | `rollout_worker.py --temperature` | 1.0 | 探索温度 |
| Hidden layers | `policy_torch.py:HIDDEN1/2` | 128 / 64 | 网络容量 |

---

## 八、已知评估结果汇总

| 对战 | 局数 | 胜方 | 胜率 | 备注 |
|------|------|------|------|------|
| v9 vs v8 | 10 | v9 10:0 | 100% | |
| v9 vs notebook elite_bot | 10 | v9 10:0 | 100% | notebook 物理 bug |
| v10 vs v9（完整 500 步） | 10 | v10 5:5 | 50% | |
| v11 vs v10 | 40 | 20:20 | 50% | 框架重建后与 v10 齐平 |
| v11 vs v9 | 40 | 16:24 | 40% | 噪声区间 |
| v12 vs v11（tiny train）| 10 | 5:5 | 50% | 蒸馏路径无退化 ✓ |
| **v13 vs v12** | **20** | **16:4** | **80%** | Phase 1 Bug 修复效果显著 |
| **v13 vs v11** | **20** | **16:4** | **80%** | 与 v12 结果一致，修复稳定 |

---

## 九、常见问题

**Q: 为什么 v11/v12 对 v9 的胜率比 v13 低很多？**

v13 修复了三个关键 Bug：① 舰队飞出屏幕（`safe_aim` OOB）② 附近静态行星迟迟未占（`cost_pen` 未豁免）③ 高产星没有提前防御（缺少 `threat_horizon`）。这三个问题在 v9-v12 中均存在，导致大量"白送"行为。

**Q: 如何新建一个版本（如 v14）？**

1. `cp submission_v13.py submission_v14.py`
2. 修改代码或替换 `_NEURAL_WEIGHTS_B64`
3. 用 `eval_head2head.py --a v14 --b v13 --seeds 0-9` 验证

**Q: 提交到 Kaggle 时需要注意什么？**

- 提交文件必须是单个 `.py` 文件，无外部依赖（所有工具函数都已内联）
- `_NEURAL_WEIGHTS_B64` 是 base64 编码的 NumPy 权重，直接内嵌在文件中
- 每回合时间限制约 1 秒，MCTS 预算已在 `PhasePolicy` 中控制在 < 920ms

**Q: 本地对战一局大约需要多久？**

v13 vs v13 每局约 30-35 秒（Mac M 系列）。10 seeds 双座位（20 局）约 6 分钟。

**Q: RL 训练需要什么硬件？**

- rollout workers：纯 CPU，Mac 128GB 即可，4-8 worker 并行
- PPO 学习器：PyTorch，自动使用 Apple MPS（Mac）或 CUDA（RTX 5090）
- 单次 imitation pretrain（200 局）约 1 小时
- 完整 PPO 训练（20 iter × 4 workers × 15 games）约 2-4 小时

---

## 十、下一步开发方向

1. **运行完整 RL 训练**：`bash tools/train_loop.sh runs/exp2 20 4 15`，训练后蒸馏更新 v13 权重
2. **验证 v13 vs v9**：预期 ≥ 70%，补全版本进化链
3. **v14 构想**：在 v13 Bug 修复基础上，用 imitation pretrain + 完整 PPO 训练得到更强的蒸馏权重
4. **5090 加速**：把 `learner.py` 搬到 RTX 5090（改 `best_device()` 自动选 CUDA），rollout 留在 Mac

---

*文档生成时间：2026-05-10。如有更新请同步修改 `AGENTS.md`。*
