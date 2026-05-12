# Orbit Wars：瞄准 / OOB 辅助工具说明

面向 [submission_v20.py](../submission_v20.py) 中与「飞出棋盘（OOB）」「撞日」相关的本地验证。工作目录一般为仓库内的 **`kaggle/`**（与 `submission_*.py` 同级）。

---

## 1. `aim_trainer.py`

**作用**：用 `kaggle_environments` 的 **`orbit_wars`** 跑若干局，统计**己方**每回合 `agent` 返回的每条发船指令，在**不与官方环境逐步交互**的前提下，用与 bot 对齐的几何模型重放轨迹，统计：

- **OOB**：在**尚未与任意行星发生命中**之前，舰队中心飞出 `[0, 100] × [0, 100]`（与 `orbit_wars.py` 边界判定一致）。
- **Sun**：本段位移穿过太阳半径（与环境里对舰段与太阳的距离判定一致）。

**几何模型（与 v19 对齐）**：若所加载的 `submission_{version}.py` 提供 `swept_pair_hit` 与 `GameState.planet_motion_segment`，则使用与环境相同的 **swept 段–段碰撞**（舰队一步位移 vs 行星本 tick 的 `old_pos → new_pos`），发射起点为 `launch_origin`（`radius + 0.1`）。旧版提交缺少上述符号时，脚本会退回旧的离散近似。

**依赖**：已安装且注册 **`orbit_wars`** 的 `kaggle_environments`（需与本地 Python 版本匹配）。

**用法**：

```bash
cd kaggle

# 默认 v18、10 局、种子 0–9
python3.12 tools/aim_trainer.py

# 指定版本与种子范围（与 eval 一样可用 0-9、0,1,2 等）
python3.12 tools/aim_trainer.py --version v19 --games 20 --seeds 0-19
```

**参数**：

| 参数 | 含义 | 默认 |
|------|------|------|
| `--version` | 要测试的脚本版本，对应 `submission_{version}.py` | `v18` |
| `--games` | 最多跑多少局（会截取 `--seeds` 列表前 N 个） | `10` |
| `--seeds` | 种子列表，如 `0-9`、`0,2,4` | `0-9` |

**输出**：每行一局：`fleets` 为统计的己方发船条数；`oob` / `sun` 为计数及占该局舰队的比例；最后为合计 OOB / sun 占比。若全程为 0，会打印 `PERFECT AIM`。

**说明**：第二路玩家由环境默认处理（与 `replay.py` / head2head 的对局配置可能不同），数值宜用作**回归对比**而非与某一场 HTML 回放逐帧等价。

---

## 2. `test_launch_trajectory_gate.py`

**作用**：**不依赖** `kaggle_environments`，用极简构造的 `GameState` 做单元级校验，覆盖：

- `launch_hits_target_first`（首碰目标是否与 `did` 一致）；
- 明显 OOB 轨迹应失败；
- `PlanArbiter._emit` 在好场景下能通过门闩并产生合法 move。

**用法**：

```bash
cd kaggle
python3 tools/test_launch_trajectory_gate.py
```

默认会打印：**脚本用途**、四个用例各在测什么、每步 `→ pass`，以及一次 `_emit` 的示例 `angle / ships / launch_origin`。仅需一行结果时：`python3 tools/test_launch_trajectory_gate.py -q`。

成功退出码为 `0`（`-q` 时只打印收尾一句）。

**适用场景**：CI / 无 orbit_wars 安装的环境；改 `safe_aim`、`launch_hits_target_first`、`launch_origin` 或 `_emit` 后快速冒烟。

---

## 相关代码入口

| 文件 | 说明 |
|------|------|
| [submission_v20.py](../submission_v20.py) | `swept_pair_hit`、`planet_motion_segment`、`launch_hits_target_first`、`ENGINE_LAUNCH_PAD` |
| [tools/aim_trainer.py](aim_trainer.py) | 对局级 OOB/sun 统计 |
| [tools/test_launch_trajectory_gate.py](test_launch_trajectory_gate.py) | 离线单元测试 |
