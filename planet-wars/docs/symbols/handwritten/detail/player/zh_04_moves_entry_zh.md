# `player.lisp` 卷四：步进目标、Move 生成与 `compute-orders`（约 L681–1176）

上一级：[player_lisp_zh.md](../player_lisp_zh.md)

---

## 目标与ownership 扫描

| 符号 | L | 说明 |
|------|---|------|
| **`compute-step-target`** | **681** | 综合 `future`、`full-attack-future`、各方 surplus，算出 **单星单步「希望达到的兵力/权属」向量**（贪心扩张的打分输入）。 |
| **`arrivals-of-player`** | **704** | 取指定玩家在该星的到达向量片段。 |
| **`find-first-ownership-change`** | **711** | 未来首次易主回合。 |
| **`find-first-possible-takeover-opportunity`** | **729** | 结合 surplus 推断「何时有机会拿下」。 |
| **`maybe-take-over-and-defend`** | **771** | 判断是否值得在该星执行占领+防守型调度。 |
| **`find-neutral-steal`** | **832** | 窃取中立机会的启发式检索。 |

---

## Step ↔ Move

| 符号 | L | 说明 |
|------|---|------|
| **`find-step`** | **843** | 从累计 surplus 与一个 **step-target** 导出 **单笔或小型 order 组合**（可能为空）。 |
| **`find-steps`** | **881** | 扩展到多 arrivals-needed **链式**补足。 |
| **`takeablep`** | **890** | 用 **full-attack-future** 快速判定「举国能否拿下」。 |
| **`generate-candidate-steps`** | **893** | 对局面枚举所有可行 **step + 分数**。 |
| **`planets-involved-in-move`** | **961** | 提取 move 中出现的 planet 集合。 |
| **`valid-move-p`** | **968** | 合法性：**非负运力、不超 surplus、与未来一致**（具体条件见源码）。 |

---

## 排序与生成 API

| 符号 | L | 说明 |
|------|---|------|
| **`generate-moves-from-steps`** | **977** | steps → 合成的 **向量 move**（多笔 order）。 |
| **`score-and-sort-moves`** | **993** | 对每个 move **`with-orders` 仿真**后用 **`eval*`**（或附带 full-attack）打分并排序。 |
| **`generate-and-score-moves`** | **1004** | 高层：**generate-candidate-steps** → **`generate-moves-from-steps`** → **sort**；供 **1-ply 贪心**或 **α–β list-actions** 调用。 |

---

## Bot 与入口

| 符号 | L | 说明 |
|------|---|------|
| **`bocsimacko`** | **1114** | 槽位 **`turn`**（引擎步计数）、**`timeout`**（默认 **0.8s**）。 |
| **`with-game`** | **1121** | 绑定 **`*n-turns-left*`**、读取 **`read-game`**、算 **`horizon`** 并可 **`truncate-game`**；最外层 **`with-orders`** 初始化缓存栈。 |
| **`compute-orders`** `(bocsimacko)` | **1150** | 外层 **`with-best-move-on-timeout`**；`**<3` 己方行星** → **`alpha-beta* :widths (4 4 4 4)`**；否则 **`generate-and-score-moves`** 取 **最高分首着**。返回值 **`values move score`** 供 `play` 写单回合订单。 |
| **`compute-orders :after`** | **1175** | `incf (turn bocsimacko)`：**引擎每调用一次递增**。 |

---

## 参考

- [alpha_beta_lisp_zh.md](../alpha_beta_lisp_zh.md)
