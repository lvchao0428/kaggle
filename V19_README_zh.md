# Orbit Wars v19.0 中文说明（区域图与多跳规划）

## 1. 这是什么

v19 在 **v17 策略管线**（分阶段、`Snapshot` 储备、`safe_aim`/`PlanArbiter` 等）之上，增加了 **区域图（RegionalGraph）**、**区域加权的 `target_value_in_region`**、**多跳规划骨架（MultiHopPlanner）** 与 **安全盈余估算（`calculate_safe_surplus`）**，用于减少「漫无目的分兵」、强调同区域连片扩张。

并行地，发射安全 **必须与 v18 对齐**：先前若 `submission_v19.py` 仍沿用旧版 `safe_aim` + 仅检查终点，会出现 **中段掠过太阳却仍发出** 的情况；现已 **合并 v18 的兜底航向与 `_emit` 轨迹采样**，见下文「冲向太阳」。

## 2. 文件分工

| 文件 | 作用 |
|------|------|
| `submission_v19.py` | Kaggle 入口 `agent`、游戏状态、`PlanArbiter`、出牌与安全门 |
| `submission_v19_regional.py` | 区域聚类、路径缓存（启发式）、`ProductionTimeline`、`MultiHopPlanner`、安全盈余函数 |
| `test_v19_regional.py` | 区域模块单元测试（本地） |

## 3. 区域图在做什么（简要）

- 用星球坐标做 **4 类聚类**，得到 `planet_id → region_id`。
- `dijkstra(src, dst)`：当前实现是 **带绕日惩罚的启发式距离 + 步数估计**，不是完整图论最短路；用于相对排序与成本，不替代物理上的 `safe_aim`。
- `target_value_in_region`：同区目标乘性加成、跨区打折；**实际选目标仍以 `target_score` 为主路径**（见代码中的 `_build_capture_plan`），区域分是扩展层，可按需再接到排序上。

## 4. 常见问题（与你截图相关）

### 4.1 「前期是不是太保守？21 就能打，为什么等到 23？」

**主要是算法里的「占据余量」而不是模拟器问题。**

在 `_build_capture_plan` 里，对 **中立星** 在「最早到达时间 `min_eta`」下的需求大致是：

`required = 到达时守军 garrison + neu_pad`

历史上固定 `neu_pad = 3`，所以 **20 守军 → 常规划到 23**。若你认为 **21 对 20 已够**（取决于规则里平局/先后手细节），这属于 **故意多带船**，避免轨道移动、敌方插队和产兵导致失手。

**v19 调整（仅前期 + 近距中立）**：

- `early` 且 `min_eta ≤ 2`：`neu_pad = 1`
- `early` 且 `min_eta ≤ 5`：`neu_pad = 2`
- 其它情况仍 `neu_pad = 3`

这样在「开局、很近的中立」上会 **更早派出足够占点的批次**；远目标或中后期仍偏稳。

另：**储备（reserve）** 由 `Snapshot._reserve` 与阶段表 `reserve_growth_mul` 决定，也会推迟「可动用 surplus」；若仍偏保守，可再单独调 `PhasePolicy` 里 early 的 `reserve_growth_mul`，与上面余量是两件事。

### 4.2 「第二截图里仍朝太阳飞，是模拟命令错了还是航向错了？」

在 Kaggle / `kaggle-environments` 里，你方提交的是 **`[planet_id, angle, ship_count]`**；引擎按 **匀速直线** 飞。若视觉上 **直指太阳中心**，说明 **我们给出的 `angle` 与（或）通过安全门时用的几何不一致**。

根因（已修）包括两类：

1. **`safe_aim` 在找不到完全净空角时，曾直接返回「best 候选」**，该候选 **仍可能切太阳**（尤其 `eta` 缩短后）。
2. **`_emit` 只检查 `eta` 终点**，**中段** `src → 中间点` 可能已进入太阳半径，仍会通过。

**修复（与 v18 一致）：**

- `safe_aim`：**较大太阳余量**、**最后再验证 `best`；不行则引向最远安全角并重选 `eta`**。
- `_emit`：在 `eta` 内多点采样（如 1,3,5,10,…）做 **越界与日心距离** 检查；任一点失败则 **本动作不发**。

因此：**这不是「模拟器把你的命令改成了朝太阳」**；而是 **旧的航向 + 偏弱的安全门**。若回放仍出现异常，再走 `tools/debug_moves_detail.py` 等对具体 `step`/舰队追一条轨迹复核。

## 5. 本地自检

```bash
# 区域模块测试
python3 kaggle/test_v19_regional.py

# 语法检查
python3 -m py_compile kaggle/submission_v19.py kaggle/submission_v19_regional.py
```

若本地 `replay.py` 报 `orbit_wars` 环境不可用，多半是 **当前 Python 环境里未注册该自定义环境**，与 bot 逻辑无关；仍以平台或完整依赖环境为准。

## 6. 与路线图的关系

整体设计仍见仓库根目录 `STRATEGY_v19_ROADMAP.md`。**真正多跳（中间星落点分解、波浪时刻表）** 在 v19.0 里多为骨架；后续 v19.1+ 可把 `MultiHopPlanner` 与 `_build_capture_plan` 更深地接在一起。

---

*文档随 `submission_v19.py` 中安全与前期余量逻辑更新；若你改动了 `capture_need` 与 `required` 两套口径，请同步检查是否一致。*
