# 串讲稿：v20（模块化后）— `submission_v20.py` + `orbit_submit/`

## 0. 开场：三件事先说清

1. **Kaggle 上你提交的“v20”** = 根目录 [`submission_v20.py`](../submission_v20.py)（薄入口）+ 同包里的 [`orbit_submit/`](../orbit_submit/)（全部实现）。**没有**第二个平行实现。
2. **`submission_v20_0513.py`** 只是历史文件名下的 **deprecated shim**（`from submission_v20 import agent`），**不要**把它当成“0513 特供版 bot”。
3. **深度技术索引**（模块表、`PHASE_TABLE`、打包、改哪里）以 [`docs/ARCHITECTURE_submission_v20_zh.md`](ARCHITECTURE_submission_v20_zh.md)（中文）/ [`docs/ARCHITECTURE_submission_v20.md`](ARCHITECTURE_submission_v20.md)（English）为准；本文是**按对局时间线**的串讲。

---

## 1. v20 整体策略总览

v20 在每一回合里，把「**这片战区值不值得押注**」「**这条扩张边该不该排进计划**」「**高压下还敢不敢提交大开销**」绑在同一条链路上；并在 **开局留守**、**扩张顺序**、**多源攻坚** 三条线上给了明确的节奏与形状。下面按能力层与行为层拆开说（**不写与其它 submission 版本的逐项对照**；常数名与实现以 `orbit_submit/` 及 [`docs/ARCHITECTURE_submission_v20_zh.md`](ARCHITECTURE_submission_v20_zh.md) 为准）。

### 1.1 区域价值与仲裁骨架

- **战区划分**：全图行星做空间聚类（目标约四区），再配合太阳感知的图最短路，得到远征成本与「哪片更值得经营」的语境。
- **统一边分**：每条 `(己方源, 目标)` 先算距离主导的 **`target_score`**；在区域图可用时叠加 **`regional_capture_adjustment`**——同区抬高「抱团」与高潜力（本区产、区内中立或可 snipe 的敌星潜力）；跨区付远征税与图距离成本。二者在 **`capture_edge_score`** 合成，作为捕获计划与 Top-K 的**单一口径**，避免多套打分各说各话。
- **战略预算与提交门控**：`Snapshot` 在总 `avail` 之上，用各区的 **`region_threat`** 等压力折算 **安全可押的战略盈余**（`calculate_safe_surplus_v20`），与 **`PlanArbiter.commit_best`** 的门控（如 `region_pressure_ratio`、`safe_surplus_ship_mult`、`baseline_commit_margin`）联动：战区压力大时，自动收紧或放弃大开销计划的提交。
- **多波次扩展钩**：`ProductionTimeline` 与 `MultiHopPlanner` 提供产线 surplus 预测与多跳 **`Wave`** 脚手架；对局内参与度可深可浅，但设计意图是：**区域态势 → 边分 → 安全盈余 → 仲裁** 始终挂在同一套 `RegionalGraph` 上，避免状态分裂。

### 1.2 开局节奏（单基地 HQ）

在 **early**、**仅一颗己方星**、**净威胁可忽略**、**非彗星** 等条件下，v20 用 **`OPENING_FIRST_CAPTURE_SEND`**、**`OPENING_SOLO_HQ_RESERVE_LAST_STEP`** 等常数约束第一波「剥厂」留守：兵力达到 peel 门槛、且未被近距离围城式威胁（`threat_horizon`）锁死高产守家加码时，把留守压到 **`threat + 1`**，尽快派出足量兵力吃掉第一波厂星；若敌舰已在 **短 horizon（如 8 步内）** 扑向本星且对方产地集中，则走更保守的 **`ships - peel` / 原留守逻辑**，优先保 HQ。

### 1.3 扩张形状：先厂后「小中立」

- **低产中立 defer**：在 **expand** 模式下，若在太阳安全路径上存在 **更高产**（尤其 **prod≥3**）的厂类目标，且其 `capture_need` 落在 **`wait_budget`** 内，则暂缓啃近处 **低产中立（尤其 prod≤2）**，避免节奏被大量小点拖散。
- **打分侧**：在 **early、中立、prod≤1** 时，**`target_score`** 对极小中立追加 **`mite_neutral_pen`**，从全局排序上压低「过早捡芝麻」的收益。

### 1.4 大中立与多源同步

对 **大中立 / 胖灰星**，v20 使用 **`SYNC_ETA_WINDOW`** 至 **`SYNC_ETA_WINDOW_MAX`** 的渐进 ETA 带，让多颗己方源在放宽的时间窗内凑兵，提高「**多源同步落地**」计划的可行率；与 **`Snapshot.is_safe_investment`**、**`capture_need`**、前向 **`target_state_at`** 一起在 `_build_capture_plan` 里闭合（串讲第 5 节会再走一遍流水线语境）。

---

## 2. 从一局的一步讲起：`agent` 在干什么？

入口：[`orbit_submit/agent.py`](../orbit_submit/agent.py) 里的 `agent(obs, config)`。

1. **`GameState(obs, config, ruleset="v20")`**  
   解析星球/舰队/彗星、预测舰队目标、维护 `incoming` / `arrivals`。`ruleset="v20"` 关闭 FFA 专用的一些推断分支，行为对齐旧单文件 v20 线。
2. **`PhasePolicy.for_state(state)`**  
   读 [`orbit_submit/policy.py`](../orbit_submit/policy.py) 里的 `PHASE_TABLE`，得到本步的策略参数对象（预算、门控、`mode_order` 等）。本地可用 `ORB_STRATEGY_PROFILE` 叠 `_STRATEGY_PROFILE_DELTAS`；可用环境变量覆盖 commit 门控三项（见架构文档）。
3. **`Snapshot.build(state, policy)`**  
   算每颗己方星的 `reserve` / `surplus` / `avail`，后面所有 planner 与 `PlanArbiter._emit` 都依赖这个“流动性视图”。
4. **区域图（可选）**  
   用 `spawn_positions`（若有）构造 `RegionalGraph` + `ProductionTimeline` + `MultiHopPlanner`。失败则置 `None`，核心打分仍可走 `target_score`。
5. **`PlanArbiter`**（[`orbit_submit/engine.py`](../orbit_submit/engine.py)）  
   流水线：`commit_urgent` → `collect_strategic` → `score_with_modifiers` → `commit_best` → `commit_fallback`，最后返回 `moves`。

---

## 3. 分阶段策略：`PHASE_TABLE` 怎么读？

把 `policy.py` 里 `PHASE_TABLE` 的三行（`early` / `mid` / `late`）理解成**三张旋钮面板**即可：

- **经济 vs 侵略**：`reserve_growth_mul`、`cost_pen_mul`、`urgent_attack_ratio`、`mode_order`。
- **搜索成本**：`mcts_*`、`pragmatic_mcts_*`、`sim_steps`、`tempo_floor`。
- **提交门控**（有区域图时更明显）：`region_pressure_ratio`、`safe_surplus_ship_mult`、`baseline_commit_margin`。
- **悲观重排**：`paranoid_score_budget_ms`、`paranoid_blend` 等，与 `score_plan_actions_paranoid` 相连。

`GameState.phase()` 决定用哪一行；v20 ruleset 下用步数比例切 early/mid/late（见 `game_state.py`）。

---

## 4. 打分：从“想打谁”到 `capture_edge_score`

1. **`target_score(snap, src, dst)`**（[`orbit_submit/targeting.py`](../orbit_submit/targeting.py)）  
   距离主导启发式：eta 强惩罚、产值与工厂加成、太阳弦路径扣分（交给 `_emit` 真轨迹门）、中立 snipe 感知等。
2. **`registry`**（[`orbit_submit/registry.py`](../orbit_submit/registry.py)）  
   `agent` 模块在 import 时把 `target_score` / `regional_capture_adjustment` / `neural_weights_b64` / `arbiter_variant` 写进 registry，再 import `engine`，这样 `capture_edge_score` 里能调到钩子。
3. **`capture_edge_score`**（`engine.py`）  
   `target_score` + 可选 `regional_capture_adjustment`（同区加成 / 跨区远征税）。

串讲提示：对比旧单文件时，可以把 **`targeting.py` 当成以前 region 5 里 `target_score` 那一坨**；`engine.capture_edge_score` 是统一入口。

---

## 5. 规划与仿真：`_build_capture_plan` 在做什么？

仍在 **`engine.py`**：对每种 `mode`（`expand` / `balanced` / `aggro` / …）：

1. **`_target_pool`** 列出候选目标星球。
2. 对每个 `dst`，对所有己方 `src` 调 `capture_edge_score` 排名。
3. **多源 ETA 窗口**（`SYNC_ETA_WINDOW` … `SYNC_ETA_WINDOW_MAX`）尝试凑齐 `required` 兵力；与 `Snapshot.is_safe_investment`、`capture_need`、`target_state_at` 强耦合。
4. 产出 `Plan(actions, score, tag)`；`PlanArbiter` 再对多个 plan 做仿真分与修饰。

**注意：** 无敌人局面下，不应再叠加“防 snipe 的额外 required 缓冲”；实现上已对 `state.en_pl` 做了守卫，避免训练/合成图上的假 snipe 把 required 撑爆（与单文件快照行为一致）。

---

## 6. 修饰器：MCTS、Pragmatic UCB、`NeuralVal`

- **MCTS / Pragmatic**：预算来自 `PHASE_TABLE`；在 `PlanArbiter.score_with_modifiers` 里把 bonus 加到仿真分上。
- **`NeuralVal`**（[`orbit_submit/neural.py`](../orbit_submit/neural.py)）：读 `registry.neural_weights_b64`（默认内容在 [`neural_weights_v20.py`](../orbit_submit/neural_weights_v20.py)），对状态特征做一个小 MLP，输出**乘性**修正系数，**不**覆盖启发式排序，只 nudge。

---

## 7. 打包与依赖

- **命令**：`python3.12 tools/package_orbit_submission.py --version v20`  
  生成 `main.py`（由 `submission_v20.py` 复制）+ `orbit_submit/` 目录。
- **scipy**：`RegionalGraph` 聚类优先 scipy；无 scipy 时走 fallback（见 `regional.py`）。

---

## 8. 与工具链的衔接

- **悲观分冒烟**：`tools/paranoid_score.py` 仍 `import submission_v20`（薄入口 re-export 了 `score_plan_actions*`）。
- **发射门闩**：`tools/test_launch_trajectory_gate.py` 依赖 `PlanArbiter` / `launch_hits_target_first` 等 re-export。
- **区域图单测**：`test_v19_regional.py` 从 `submission_v20` import `RegionalGraph` 等。
- **v21/v22 生成脚本**：从 [`tools/templates/v20_monolith_for_v21_codegen.py`](../tools/templates/v20_monolith_for_v21_codegen.py) 取 **NeuralVal 形状** 等可 patch 的单文件快照；与线上薄入口 **分离**，避免生成器去 patch 3000 行动态入口。

---

## 9. Q&A（课堂常见问题）

**Q：改一个数最快走哪？**  
**A：** 相位相关 → `orbit_submit/policy.py`；打分形状 → `orbit_submit/targeting.py`；提交门控/arbiter → `orbit_submit/engine.py`。

**Q：`submission_v20_0513.py` 还要保留吗？**  
**A：** 仅为兼容旧引用；新代码请只依赖 `submission_v20.py`。

**Q：如何自证没改挂？**  
**A：** `tools/paranoid_score.py --check-import`、`tools/test_launch_trajectory_gate.py`、`tools/sim_first_turn_opening.py --synthetic`，再加小规模 `eval_head2head`。
