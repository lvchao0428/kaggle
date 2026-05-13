# Orbit Wars v21：Lite / Pro / Ultra 训练栈说明

本文档描述 **`tools/v21/`** 下的 RL 管线（基于 **v20 单文件栈**）、三档 **policy 网络**、**训练数据与目标**、**提交物 `submission_v21_*.py`** 的生成与蒸馏，以及 **后台训练与日志** 操作。

**本地解释器**：训练脚本默认 **`PY=python3.13`**；Kaggle 评测环境与文档其它处仍多以 **3.12** 为参照，本地用 3.13 一般可行，依赖装在同一解释器下即可（`python3.13 -m pip install -r requirements.txt`）。

---

## 1. 产物一览

| 产物 | 说明 |
|------|------|
| [`submission_v21_lite.py`](submission_v21_lite.py) | 由 v20 生成；`NeuralVal` **64→32→1**；` _NEURAL_WEIGHTS_B64` 为空，需蒸馏后粘贴 |
| [`submission_v21_pro.py`](submission_v21_pro.py) | **128→64→1** |
| [`submission_v21_ultra.py`](submission_v21_ultra.py) | **192→96→1**（单步更重，建议 teacher 或 distill 到 lite/pro 再交） |
| [`tools/gen_v21_submissions.py`](tools/gen_v21_submissions.py) | 从 `submission_v20.py` 重新生成上述三者（v20 大改后必须重跑） |

---

## 2. 模型结构（训练侧）

三档网络均由 [`tools/v21/nets.py`](tools/v21/nets.py) 的 `build_net(tier)` 构造，输入为 **单步候选计划的组合特征**。

### 2.1 公共输入维度

- **状态** `state_feat`：**14 维**，与 v20 `NeuralVal.feat(state)` 完全一致（兵力占比、产能比、星球占比、阶段等）。  
- **计划摘要** `plan_features`：**17 维** = `tanh(score/500)` + 11 维 mode one-hot + 5 个归一化标量（出兵条数、目标数、ETA 代理等）。详见 [`tools/v21/feature_extractor_v20.py`](tools/v21/feature_extractor_v20.py)。  
- **拼接** `combined_features`：**31 维** = 14 + 17，对 **每个候选 Plan** 各算一条（同一状态下 14 维状态部分重复）。

### 2.2 Lite（≈1h 档）

- **PolicyValueNetLite**：`31 → 192 → 96`（ReLU）→ `value_head(96→1)` + `plan_head(96→1)`（plan 输出经 **tanh**）。  
- **跨候选**：无交互；`forward_plans` 对 K 行独立过 trunk，**value 取 K 个 value 的均值**（与 learner 一致）。

### 2.3 Pro（≈6h 档）

- **PolicyValueNetPro**：状态塔 `14→64`、计划塔 `17→64`，拼接 **128** 后 `256→256→128`，每层 **LayerNorm + GELU**；双 head 同上。  
- **跨候选**：与 Lite 相同（每行独立 trunk，value 平均）。

### 2.4 Ultra（≈24h 档）

- **PolicyValueNetUltra**：每条候选 `31 → d_model(128)` 为 token；**2 层 `TransformerEncoder`**（`nhead=4`, `dim_ff=512`, `norm_first`, GELU）；**plan 头**逐 token `d→1` 再 tanh；**value** 为 token **均值池化** 后 `d→1`。

---

## 3. 训练数据如何生成

1. **策略对象**：[`tools/v21/rl_agent_v21.py`](tools/v21/rl_agent_v21.py) 中 **`RLAgentV21`**：与 v20 `agent()` 一致构建 `RegionalGraph` / `PlanArbiter`，`collect_strategic()` 得候选 **`Plan` 列表**，再用当前 **policy** 在候选上做 **softmax 采样**（explore）或 argmax。  
2. **对手**：[`tools/v21/rollout_worker_v21.py`](tools/v21/rollout_worker_v21.py) 支持静态 `submission_v*`（含 `v20@rush` 等）、`random`、以及 **50% 自对**（若存在 `runs/.../policy_latest.pth`）。  
3. **每步记录**（与旧 v11 rollout 对齐并兼容 learner）：  
   - `obs_feat`：选中计划的 31 维向量  
   - `plan_feats`：**(K, 31)** 全部候选  
   - `chosen_idx`, `log_prob`, `value_pred`, `plan_score_net`, `step`, `state_feat`(14), `oob_penalty`  
   - 局末后填 **`shaped_reward`**（`state_feat[0]` 船舶占比的间隔差分 × 权重）  
4. **落盘**：worker 写 **`shard_w*.msgpack`**；[`tools/v21/learner_v21.py`](tools/v21/learner_v21.py) 消费后 **删除 shard**。  
5. **局摘要 `game_summary`**：`final_my_ship_ratio`、`final_planet_ratio`（来自 14 维特征）、`last_step`、`n_transitions`；监督进程会写入 **metrics.jsonl** 的均值。

---

## 4. 训练目标与 reward

- **回报**：与 [`tools/learner.py`](tools/learner.py) 相同逻辑：每步 `shaped_reward + oob_penalty`，**最后一步 + 终局 outcome（±1）**；**GAE(γ=0.997, λ=0.95)** 得 `advantages` / `returns`。  
- **价值损失**：对每条样本，用 **`forward_plans(1,K,F)`** 的 **value 向量** 与对应 **`returns`** 做 **MSE**。  
- **策略损失**：对 **plan 索引** 做 **advantage 加权的交叉熵**（正 advantage 拉高选中项，负的压）。  
- **熵 bonus**：沿用对 `plan_score_net` 的弱正则（与旧 learner 同风格）。

---

## 5. Checkpoint 与蒸馏

- **RL checkpoint**：`runs/<exp>/policy_latest.pth`（完整 `state_dict`）；可选带时间戳 `policy_<ts>.pth`。  
- **rollout 加载**：优先 `--checkpoint`，否则 `runs-dir/policy_latest.pth`；**无权重** 时走 **启发式 baseline**（仍可产 shard）。  
- **蒸馏进提交**：[`tools/distill_to_numpy_v21.py`](tools/distill_to_numpy_v21.py)  
  - `--teacher-tier` + `--checkpoint`：教师为对应 **PolicyValueNet**  
  - `--target-submission submission_v21_pro.py`：自动读取目标 **`NeuralVal`** 的 **h1/h2** 宽度  
  - `--shards-dir`：用 transition 里的 **`state_feat`** 回归教师的 **value**（对 state 填零 pad 到 31 维再过教师）  
  - 输出 **`--out-b64`** 文本，**替换**目标文件里 **`_NEURAL_WEIGHTS_B64 = "..."`**

---

## 6. 后台训练与日志

### 6.1 一键后台（推荐）

```bash
chmod +x scripts/train_v21_lite.sh scripts/train_v21_pro.sh scripts/train_v21_ultra.sh
./scripts/train_v21_lite.sh   # 或 pro / ultra
```

- 标准输出写入 **`logs/v21_<tier>_<timestamp>.log`**；打印后台 **PID**。  
- 进程内另有 **`runs/<exp>/train.log`（轮转 FileHandler）**。

### 6.2 监督进程 CLI

```bash
python3.13 tools/v21/train_supervisor.py \
  --runs-dir runs/v21_lite \
  --tier lite \
  --submission v20 \
  --iterations 6 \
  --workers 4 \
  --games-per-worker 15 \
  --lr 3e-4
```

**写出：**

- `runs/<exp>/metrics.jsonl`：每轮一行 JSON（**lr、policy/value loss、entropy、games、transitions、CPU%、内存、nvidia-smi 利用率/显存、局摘要均值、累计训练秒**）  
- `runs/<exp>/supervisor_state.json`：累计 Wall 时间与最后一行摘要  

### 6.3 单跑 rollout / learner

```bash
python3.13 tools/v21/rollout_worker_v21.py \
  --runs-dir runs/v21_lite --workers 4 --games-per-worker 10 \
  --tier lite --submission v20 --opponents v20 v19

python3.13 tools/v21/learner_v21.py \
  --runs-dir runs/v21_lite --tier lite --updates 5 --wait-secs 60
```

---

## 7. 维护 v20 → v21 文件

 whenever `submission_v20.py` 大幅更新：

```bash
python3.13 tools/gen_v21_submissions.py
python3.13 -m py_compile submission_v21_lite.py submission_v21_pro.py submission_v21_ultra.py
```

---

## 8. 评测

```bash
python3.13 scripts/eval_head2head.py --a v21_lite --b v20 --seeds 0-4
```

（需已蒸馏填权或接受随机 `NeuralVal` 修饰。）

---

## 9. 依赖

- **`psutil`**（CPU/内存采样）：已加入 [`requirements.txt`](requirements.txt)。  
- **GPU 统计**：调用本机 **`nvidia-smi`**；无 GPU 时对应字段省略。  
- **kaggle-environments**、**torch**、**msgpack**、**scipy** 等同仓库原有 RL 环境。

---

## 10. 常见问题

**`ModuleNotFoundError: No module named 'msgpack'`**（或缺少 `torch`）  
当前使用的解释器（默认脚本为 **`python3.13`**）未安装依赖时，在仓库根目录执行：

```bash
python3.13 -m pip install -r requirements.txt
# 或最小：python3.13 -m pip install -U msgpack torch kaggle-environments
```

训练脚本会在启动 **nohup 前** 做一次 `import msgpack, torch` 检查；若失败会直接报错并提示上述命令。  
若本机没有 `python3.13`、或你想用其它解释器（如 **3.12**、conda 的 `python`），可覆盖：

```bash
PY=python3.12 ./scripts/train_v21_lite.sh
PY=python ./scripts/train_v21_lite.sh
```

---

## 11. 实时监控对弈进度

Rollout 默认在 **每局结束** 打印一行，并追加 **JSON**（便于 `jq` 等解析）：

- **人类可读行**（出现在 nohup 的 `logs/*.log`、`runs/<exp>/train.log` 所镜像的 stdout）：例如  
  `[w2] game 4/12 +120s seed=... win r=[1, -1] ship_r=0.51 planet_r=0.12 ships_mine=8000 ships_en=12000 planets=24 mine=8 last_step=500 transitions=420 opp=submission`  
  字段含义：`w*`=worker；`game i/n`；`+秒`=该 worker 从本轮 rollout 开始的累计时间；`seed`；胜负；`r=` 环境原始 reward；`ship_r` / `planet_r` 为终局 **兵力占比 / 占星球比例**（来自 14 维 `state_feat`）；**`ships_mine` / `ships_en`** 为终局 **我方总兵力 / 所有敌方兵力之和**（整数）；**`planets` / `mine`** 为 **地图上星球总数 / 我方占领数**；`last_step`；本局 **transition 条数**；`opp=` 对手类别（`submission`、`trainable_policy`、`random`、`mix:v20` 等）。
- **`runs/<exp>/rollout_progress_w0.jsonl`** … **`w{k}.jsonl`**：每行一条与上对应的 JSON；多进程各写自己的文件，避免锁。

**Supervisor** 在启动时、每轮 **rollout 开始前 / 结束后 / learner 结束后** 各打一行 **`resources | ...`**（`psutil` 的 CPU/RAM + `nvidia-smi` 的 GPU 利用率与显存）；末尾 **`iter N done ...`** 与 **`metrics.jsonl`** 里也会带当轮的 `cpu_pct` / `gpu_util_pct` 等。若长时间只看到 OpenSpiel 与逐局行、没有 `resources`，请确认跟踪的是 **`runs/<exp>/train.log` 或包含 supervisor 的 nohup 日志**（rollout 子进程的 stdout 不会包含 supervisor 的机器采样）。

**推荐**另开终端：

```bash
chmod +x scripts/watch_v21_training.sh
./scripts/watch_v21_training.sh runs/v21_lite
```

若需安静模式（不写逐局 jsonl、不刷屏），对 `train_supervisor.py` 传入 **`--quiet-rollout`**。
