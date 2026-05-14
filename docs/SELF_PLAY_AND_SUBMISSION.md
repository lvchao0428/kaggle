# Orbit Wars：自对弈、评测脚本与 Kaggle 打包

本文说明本仓库里与「自我对弈 / 对手池 / 调参评测」相关的操作，以及如何把 **`submission_v20.py`**（薄 Kaggle 入口）+ **`orbit_submit/`**（v20 实现）打成最终上传包（`scripts/package_submission.sh` 或 `tools/package_orbit_submission.py`）。架构说明另见 [`docs/ARCHITECTURE_submission_v20_zh.md`](ARCHITECTURE_submission_v20_zh.md)（中文）/ [`docs/ARCHITECTURE_submission_v20.md`](ARCHITECTURE_submission_v20.md)（English）。

**版本关系**：**`submission_v19.py`** 保留为 **v19 基线/对照**（你当前目录中多为昨日/归档快照，用于与 v20 A/B）；**新调参、打包、随文示例默认均以 v20 为准**。Head-to-head 可随时写 `--a v20 --b v19` 做对比。

---

## 1. 两套「自对弈」别混了

| 类型 | 做什么 | 典型入口 |
|------|--------|----------|
| **规则 Bot 评测** | 两个 **静态** `submission_v*.py`（或内置 `random`）对打，算胜率 | `scripts/eval_head2head.py` |
| **RL 自博弈（训练）** | `RLAgent`（`tools/rl_agent.py`，当前 **包装 submission_v11**）与对手（自重 / 静态 bot / 混合）对局，写 **msgpack shard** → `learner.py` 更新权重 | `tools/rollout_worker.py` + `tools/learner.py`（或 `tools/train_loop.sh`） |

`submission_v20.py` 是 **薄入口**：`import orbit_submit.agent` 注册 `registry` 钩子并导出 `agent`；**实现**在 `orbit_submit/`（与 **§8.1** 的多文件打包约定一致）。RL 训练的 `PolicyValueNet` 目前仍挂在 **v11 的 PlanArbiter** 上；蒸馏出的 NumPy MLP **形状须与目标文件的 `NeuralVal` 一致**（常为 `14→64→32→1`），把生成的 base64 **替换** `orbit_submit/neural_weights_v20.py` 里的 `NEURAL_WEIGHTS_B64`（或同步写 `registry.neural_weights_b64`）即可。要让 **v20**（或对照用的 v19）吃进这套权重，目标文件里仍需 **同名 `NeuralVal` 接口且特征维一致**——否则要自己改蒸馏脚本与接线。本文以 **v20 评测 + 打包**为主；RL 的原理与命令在 **§6**。

---

## 2. 环境准备

- **推荐解释器**：仓库约定用 Python 3.12（与 `kaggle-environments` 一致；不要用错 conda 下的 3.13 若你遇到无关的 `cabt`/`.so` 加载报错）。
- **安装**：`python3.12 -m pip install -U "kaggle-environments>=1.28.0"`
- **`submission_v20.py` + `orbit_submit/`**（与同族 v19 单文件栈不同）：`RegionalGraph` 依赖 **`scipy`**（`fclusterdata`），本地与 Kaggle 运行环境需能安装 scipy。

---

## 3. Head-to-head：两个静态 Bot 对打（不落训练数据）

**双座位**（默认）：每个 seed 打两盘（交换先后/座位标签），减少对称图上的抽样偏置。

```bash
# 例：v20 vs v19 或 v17，种子 0–19（共 40 局）
python3.12 scripts/eval_head2head.py --a v20 --b v19 --seeds 0-19
python3.12 scripts/eval_head2head.py --a v20 --b v17 --seeds 0-19
```

**单座位**（例如对 `random`）：

```bash
python3.12 scripts/eval_head2head.py --a v20 --b random --seeds 0-9 --no-swap
```

### 3.1 本地风格 profile（`v20@rush` 等）

仅影响 **对应版本入口**（如 **`submission_v20.py`** 所 re-export 的 `ORB_STRATEGY_PROFILE`）：通过 `ORB_STRATEGY_PROFILE`（在脚本里用 ContextVar 包住 `agent`）合并 `_STRATEGY_PROFILE_DELTAS`（定义在 `orbit_submit/policy.py`）。Kaggle 正式提交 **不要** 带 `@`，这是本地/消融用。

```bash
python3.12 scripts/eval_head2head.py --a v20@rush --b v17 --seeds 0-9
python3.12 scripts/eval_head2head.py --a v20@turtle --b v19 --seeds 0-9
```

profile 名：`turtle`、`rush`、`expand`、`greedy_prod`、`dual`（见 **`orbit_submit/policy.py`** 中 `_STRATEGY_PROFILE_DELTAS`；若 **`submission_v19.py`** 保留相同机制，亦可使用 `v19@rush` 等）。

---

## 4. 悲观前向仿真冒烟（不开整局）

不写 shard、不走 learner，只验证「我方计划 + 简化对手一手」分值是否合理：

```bash
python3.12 tools/paranoid_score.py --check-import
```

实现 lives in **`orbit_submit/engine.py`** 等模块（`score_plan_actions_paranoid` 等由 **`submission_v20.py` re-export**；`tools/paranoid_score.py` 默认 `import submission_v20`）。

---

## 5. Commit 门控小网格（环境变量覆盖）

**不改编译**，用进程级环境变量覆盖 `PHASE_TABLE` 里已与 `PlanArbiter.commit_best` 绑定的三项门控（见 `_merged_phase_row`）：

- `ORB_REGION_PRESSURE_RATIO`
- `ORB_SAFE_SURPLUS_SHIP_MULT`
- `ORB_BASELINE_COMMIT_MARGIN`

```bash
python3.12 tools/sweep_commit_gates.py \
  --a v20 --b v17 --seeds 0-4 \
  --combos \
  "ORB_REGION_PRESSURE_RATIO=0.68,ORB_SAFE_SURPLUS_SHIP_MULT=1.55" \
  "ORB_REGION_PRESSURE_RATIO=0.72,ORB_BASELINE_COMMIT_MARGIN=0.12"
```

每一条 combo 会起一个子进程跑 `eval_head2head.py`；挑出胜率稳定的组合后，**把数字写回** **`orbit_submit/policy.py`** 的 `PHASE_TABLE` 对应字段，而不是长期靠 env。

---

## 6. RL 自对弈（训练侧）

本节的 **策略训练** 与 **Kaggle 提交 bot** 是两层东西：`RLAgent` 仍走 **v11 的 `PlanArbiter` 全栈**（生成多条候选 `Plan`），只在「选哪一条计划」上用小型神经网络替换/加强启发式；蒸馏产物默认对齐 **v11 系 `NeuralVal`（仅状态 14 维）**。**v20 / v19** 若未保留相同形状与接线，不能直接「换手权重」，需要改代码或单独训一条与目标 `submission` 对齐的特征管线。

### 6.1 为什么这样做会有效

1. **回合级决策但终局才判胜负**：纯启发式很难在 500 步里全局一致地分配「机会成本」；PPO + GAE 用 **终局 ±1** 给出长期方向，避免每一步手写权重互相打架。
2. **在候选计划上做策略，而不是在原始动作空间**：`PlanArbiter` 已经把防御/扩张/MCTS 等压成少量带标签的 `Plan`（见 `feature_extractor.PLAN_TAGS`）。策略网络只做 **K 选 1 的 softmax**，搜索空间小、和现有 bot 对齐，冷启动可以 **无权重时回落到 v11 启发式**（`rl_agent.py`）。
3. **自对弈 + 固定对手池**：与「当前的自己」对打会不断暴露 exploit；混入 **v9/v10/v11、静态 v20/v19 或 profile**（`--opponents` / `--opponent-mix`）能缓解策略塌缩、贴近真实 submission 分布。
4. **密集塑形奖励**：仅用终局信号，方差很大；`rollout_worker` 在局末回填 `shaped_reward`（每隔 `DENSE_REWARD_INTERVAL` 步，用 `state_feat[0]` 的差分 × `DENSE_REWARD_WEIGHT`），再与终局胜负一起进 GAE，让中段局势变化也能推梯度。
5. **越界惩罚进 shard**：`oob_penalty` 与塑形、终局一同写入 `rewards`，减轻「直线飞出棋盘」的可行解。

以上几点都不改变「物理与规则在引擎里为真」这一前提，只是把 **计划排序** 这一块变成可学习的。

### 6.2 模型与数据流（训练时 vs 蒸馏进提交）

**训练态 `PolicyValueNet`**（`tools/policy_torch.py`）：

- **输入**：单条向量 **31 维** = `N_STATE_FEATURES`（14，与 `v11.NeuralVal.feat` 一致）+ `plan_features`（17：归一化计划分、11 维 mode one-hot、若干标量摘要）。由 `combined_features(plan, state)` 拼成，每条候选计划一条 31 维向量。
- **共享 trunk**：`Linear(31 → 128) → ReLU → Linear(128 → 64) → ReLU`。
- **两个头**：
  - **value**：`Linear(64 → 1)`，供 GAE / 值函数回归；
  - **plan score**：`Linear(64 → 1)` 再过 **tanh**，对 **同一状态** 下 K 个候选分别前向，得到 K 个标量，**softmax（带 `temperature`）** 采样计划索引；`learner` 里用 **advantage 加权的交叉熵** 更新（见 `ppo_update`）。

**局内推理**：`RLAgent` 用 NumPy 复现同一结构（权重来自 `policy_latest.npz`），与 PyTorch 导出一致。

**蒸馏进提交**（`tools/distill_to_numpy.py`，可选）：教师是整个 `PolicyValueNet`；把状态 14 维后面 **补 17 维零** 喂给教师，取 **value head** 作为回归目标，训练 **学生 `StudentMLP`：`14 → 64 → 32 → 1`，tanh**，与 **`v11.NeuralVal`** 形状一致，产出 base64 写入 **`orbit_submit/neural_weights_v20.py`** 的 `NEURAL_WEIGHTS_B64`（`submission_v20.py` 通过 `registry` 与 re-export 的 `_NEURAL_WEIGHTS_B64` 与之对齐）。也就是说：**提交包里运行的仍是启发式栈 + 小价值网络修饰**；完整 31 维 plan head **不** 随默认蒸馏进 Kaggle 包。

### 6.3 训练中 / 训练后常改的超参数

下列在代码里多为 **常量或默认 CLI**；效果好坏高度依赖试算，没有一劳永逸默认值。

| 类别 | 位置 | 常见调法 |
|------|------|-----------|
| **Rollout 吞吐** | `rollout_worker` CLI | `--workers`、`--games-per-worker`：总样本量；`train_loop.sh` 里每轮各跑一轮。 |
| **探索** | `--temperature`（默认 1.0） | 高于 1 → softmax 更平、更随机；低于 1 → 更贪。影响 `log_prob` 与数据多样性。 |
| **对手分布** | `--opponents`、`--opponent-mix` | 提高对 strong baseline 比例可压过拟合弱对手；`self` 需配合 `--weights`。 |
| **密集奖励** | `rollout_worker` 顶部常量 | `DENSE_REWARD_INTERVAL`（默认 20）、`DENSE_REWARD_WEIGHT`（默认 0.10）：越大/step 越密，可能与终局信号比例失配。 |
| **PPO / GAE** | `learner.games_to_tensors` / `ppo_update` **默认参数** | `gamma=0.997`、`lam=0.95`；`clip=0.2`、`value_coef=0.5`、`entropy_coef=0.005`、`epochs=4`、`minibatch=4096`、**梯度裁剪 1.0**。当前 CLI **只暴露** `--lr`、`--updates`、`--wait-secs`、`--ckpt-every`；改 PPO 内部超参需编辑 `learner.py` 或加 CLI。 |
| **学习率** | `learner.py --lr`（默认 `3e-4`） | 不稳定时降低；太慢时小幅上调。 |
| **BC 热身** | `imitation_pretrain.py` | `--games`、`--epochs`、`--runs-dir`：减少 RL 冷启动浪费。 |
| **蒸馏** | `distill_to_numpy.py` | `--epochs`（默认 80）、`--lr`、`--batch`：影响学生与教师 value 的贴合度。 |

`train_loop.sh` 每轮调用 `learner.py --updates 1`：长跑时可改为一次消费多 shard 的更大 `--updates`，与 rollout 节奏一起调。

---

### 6.4 单次 rollout（多进程写 shard）

```bash
python3.12 tools/rollout_worker.py \
  --workers 4 \
  --games-per-worker 20 \
  --runs-dir runs/exp1 \
  --weights runs/exp1/policy_latest.npz \
  --opponents v11 v10 v9
```

- `--weights` 缺省：随机初始化策略（烟测）。
- 默认对手逻辑：**约 50%** 与 **当前 weights 的 RLAgent 自对**（若 weights 存在），**否则**在 `--opponents` 列表里随机选一个 **静态** `submission_v*.py`。
- **`--opponent-mix`**：显式加权混合（权重会归一化），**覆盖**上述 50/50 逻辑，例如：

```bash
python3.12 tools/rollout_worker.py \
  --workers 2 \
  --games-per-worker 10 \
  --runs-dir runs/exp_mix \
  --weights runs/exp_mix/policy_latest.npz \
  --opponent-mix "self:0.5,v13:0.35,v20@rush:0.15"
```

- token：`self`（需 `--weights`）、`random`、任意可解析的 **`submission_vXX`**、`v20@rush`、`v19@turtle` 等同理。

### 6.5 交替 rollout + learner

```bash
bash tools/train_loop.sh runs/exp1 6 4 15
# 参数：<runs_dir> <iters> <workers> <games_per_worker_per_iter>
```

脚本内硬编码 `tools/rollout_worker.py` + `tools/learner.py`；Mac 上默认 `PY=/opt/local/bin/python3.12`（可按机器改脚本顶部）。

### 6.6 Learner 单独跑（已有一批 shard）

```bash
python3.12 tools/learner.py --runs-dir runs/exp1 --updates 50
```

Learner 会消费 `runs_dir` 下 `shard_w*.msgpack`，读完后 **删除 shard**（避免目录无限涨）。

### 6.7 蒸馏进提交（可选、管线外一步）

用训练好的 **`PolicyValueNet` checkpoint**，在 **`runs/exp/...`** 里已有 rollout **shard**（供采状态样本）的前提下，把学生 MLP（与目标 `submission` 里的 `NeuralVal` **同拓扑**）压成 base64：

```bash
python3.12 tools/distill_to_numpy.py \
  --checkpoint runs/exp1/policy_<timestamp>.pth \
  --shards-dir runs/exp1 \
  --out-b64 runs/exp1/neural_weights.b64.txt \
  --epochs 80 --lr 1e-3 --batch 512
```

产出文件是纯文本 base64：**整体替换** **`orbit_submit/neural_weights_v20.py`** 中 `NEURAL_WEIGHTS_B64 = "..."` 的字符串（并确认 `orbit_submit/agent.py` 仍从该模块载入 `registry.neural_weights_b64`）。**改权重前务必核对** `orbit_submit/neural.py` 里 `NeuralVal` **层形状**是否与 `tools/distill_to_numpy.py` 里 `StudentMLP` 一致；不一致则要先改蒸馏脚本里的学生网络再训。

可选 **BC 热身**（减少 RL 冷启动），与 PPO **共用同一 `runs-dir`**，再跑 `policy_latest.npz` 给 rollout 读：

```bash
python3.12 tools/imitation_pretrain.py \
  --games 50 --epochs 20 --runs-dir runs/exp1 --ref-bot v11
# 再上 §6.5 train_loop，或按需加大 --games
```

额外 CLI 详见：`python3.12 tools/distill_to_numpy.py --help`、`python3.12 tools/imitation_pretrain.py --help`。

---

## 7. 打包成 Kaggle 最终提交

仓库提供两套常见产物：

- **`scripts/package_submission.sh`**：**单文件** `main.py` + `dist/submission.tar.gz`（适合只上传一个 `.py` 的比赛流程）。
- **`tools/package_orbit_submission.py --version v20`**：`main.py`（由 `submission_v20.py` 复制）+ **`orbit_submit/`** 目录打进 zip（与当前默认 v20 布局一致）。

```bash
./scripts/package_submission.sh submission_v20.py
# 或多文件包：
python3.12 tools/package_orbit_submission.py --version v20
```

产物：

- `dist/main.py` — 与源文件内容一致（文件名符合 many competitions 的 `main.py` 约定）。
- `dist/submission.tar.gz` — 含 `main.py`，供网页 / CLI 上传。

**建议 Kaggle CLI（需已接受比赛规则并完成认证）：**

```bash
kaggle competitions submit orbit-wars -f dist/submission.tar.gz -m "v20 heuristic"
```

上传前确认：若走 **多文件 zip**，解压后应含 `main.py` 与 `orbit_submit/`；**含 scipy** 时 Kaggle 镜像能装依赖，且单步耗时尚未大量超时（**v20** 含区域聚类 + 可选悲观/MCTS，注意 wall time；调 `orbit_submit/policy.py` 的 `PHASE_TABLE` 与 `deadline_ms`）。

---

## 8. 附录（本文内闭环）

### 8.1 Kaggle 提交约定（与评测脚本一致）

- **入口**：`agent(obs, config=None) → list[list]`，每项为 **`[from_planet_id, angle_rad, num_ships]`**，每回合至多 **26** 条。
- **默认 v20 布局**：`main.py`（薄入口）+ **`orbit_submit/`** 包（`tools/package_orbit_submission.py`）；也可用 `package_submission.sh` 打成**单文件** `main.py`（若你自行内联/合并）。
- **不应**在提交里 `import` 本仓库未随包上传的模块（例如 `orbit_wars_bot`、`tools`）；RL 训练的 PyTorch **仅在线下**。
- **神经网络位**：权重默认在 **`orbit_submit/neural_weights_v20.py`**；`NeuralVal` 定义见 **`orbit_submit/neural.py`**。
- **墙钟**：超重搜索（大区聚类、悲观仿真、MCTS）需在 **`orbit_submit/policy.py`** 的 `PHASE_TABLE` 与 `PlanArbiter` 的 `deadline_ms` 等与线上时间对齐。

### 8.2 游戏规则摘要（读至此处理解评测与 bot 语境即可）

以下为 **极简**摘录，够用理解本文里的「舰队 / 棋盘 / 胜负」用词。

- **场地**：连续 **100×100**；太阳中心约 **(50,50)**，半径 **10**，舰队轨迹碰到太阳即灭。
- **时长**：默认 **500** 回合。
- **胜负**：终局己方 **舰船总数**（在星球驻军 + 在途舰队）**多者胜**。
- **星球**：离散目标点，有驻军与产兵；一类绕日公转（由观测里的角速度推演），远处可为静止星。
- **舰队**：出发前定 **方向角**，之后 **匀速直线**，速度随舰容 **对数** 上浮（参见环境默认公式）；出界线段亦会灭舰队。
- **动作**：只能从未方星球派出，运力不超过当场驻军。

评测与环境与 **Kaggle `orbit-wars`** 对齐时，观测字段以官方说明为准。

### 8.3 代码入口速查（不依赖外链）

| 需求 | 代码位置 |
|------|----------|
| 双 bot 胜率 | `scripts/eval_head2head.py` |
| 打 tar 包 | `scripts/package_submission.sh`（默认 `submission_v20.py`） |
| 网页式 re-export | `kaggle_submit_entry.py` → `from submission_v20 import agent` |
| 版本索引（默认提交） | `VERSIONS.md`（与 `docs/` 同级） |
| v20（默认）/ v19（对照）profile、相位表 | **`orbit_submit/policy.py`**（v20）/ `submission_v19.py`（若仍为单文件栈） |
| RL rollout / learner | `tools/rollout_worker.py`、`tools/learner.py`、`tools/train_loop.sh` |
| 特征与候选计划标签 | `tools/feature_extractor.py` |
