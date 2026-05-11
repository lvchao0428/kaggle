# `model.lisp` — 中文逐符号说明（`src/model.lisp`）

自动符号表：[src__model.md](../../by-file/src__model.md)。

`eval-when` 包裹的常量（`**L3–14**`）在「顶层括号深度≠0」、`gen_lisp_symbol_docs.py` **不会枚举**；此处手写补全：`*max-n-initial-ships*`、`*max-growth*`、`*max-n-planets*`、`*max-n-turns*`、`*n-players*`；运行时由 `compute-orders`/`with-game` 绑定的 `*n-turns-till-horizon*`、`*n-turns-left-in-game*`。

---

## 类型与常量

| 符号 | 说明 |
|------|------|
| `ship-count` / `ship-count-vector` | 兵力计数整数类型与简单向量类型；尺度由最大船数推导。 |
| `player` / `player-vector` | 玩家枚举型（含中立 0）与向量。 |
| `make-count-vector` / `make-player-vector` | 分配计数/玩家向量，初值为 0。 |

---

## 类：`planet`

| 槽位 | 说明 |
|------|------|
| `id` | 星球编号（载入后填入）。 |
| `owner` | 所有者：0 中立；1、2 为双方。 |
| `n-ships` | 当前地面兵力。 |
| `x` / `y` | 坐标。 |
| `growth` | 每回合增产。 |
| `arrivals-1` / `arrivals-2` | 按「相对当前回合偏移」索引的 **`ship-count-vector`**：某时刻到达的船数。**双方不能合一成带符号向量**，因为中立星上的会战需要两方独立到账数。 |
| `departures-1` / `departures-2` | 同理：计划在未来某回合从本星**出发**的船数（多拍调度核心）。 |
| `neighbours` | `((turns-to-travel planet...) ...)` **升序**按航程分组。 |
| `turns-to-neighbours` | 向量：到各 `planet` id 的航程（可由几何预填）。 |

`print-object`（**L90**）：调试打印简要字段。

---

## 类：`game`

| 槽位 | 说明 |
|------|------|
| `planets` | 星球向量。 |
| `n-ships-beyond-1` / `n-ships-beyond-2` | 盘面截断地平线之后仍在飞的船数统计（记在 game 头上）。 |
| `caches-and-moves` | 与仿真/挂单栈配合的缓存（**accessor**）。 |

---

## 类：`order`

| 槽位 | 说明 |
|------|------|
| `source` / `destination` | 源/宿 **planet**。 |
| `owner` | 默认 1（己方）。 |
| `n-ships` | 派出数量。 |
| `turn` | **相对回合偏移**：引擎当回合执行的为 **`turn=0`**；未来执行的为正整数。 |

`print-object`（**L99**）：打印船数、owner、两端 id、回合。

---

## 比较与航程

| 符号 | 说明 |
|------|------|
| `arrival-turn` | `turn + turns-to-travel(source, dest)`。 |
| `order=` / `move=` | 单笔/整单 move（`order` 列表）相等。 |
| `current-order-p` | owner 存在且 **`turn=0`**（引擎本拍执行）。 |

---

## 邻居遍历宏

| 符号 | 说明 |
|------|------|
| `do-neighbours` | 绑定 `(turns-to-travel neighbours)`，**跳过自环**（0 航程组）。正向遍历 neighbors 列表。 |
| `do-neighbours/reverse` | 同上但 **reverse** 邻接分组顺序。 |

---

## 几何与战斗

| 符号 | 说明 |
|------|------|
| `planet-id` | planet 或 id 统一成 id。 |
| `turns-to-travel*` | 欧氏距离 **ceiling** = 整数航程。 |
| `turns-to-travel` | 查预计算向量 `turns-to-neighbours`。 |
| `count-ships-for-battle` | 多方舰队合并为计数向量。 |
| `resolve-battle` | 返回战胜方索引与剩余船数（**三方 neutral + p1 + p2** 规则实现）。 |
| `player-multiplier` | 1→+1，2→-1，0→0（用于带符号累加等）。 |
| `opponent` | 1↔2。 |

---

## `*turn-adjustment*`（**L205**）

到达桶索引的 **全局微调**（默认 0）；`execute-order` 写到达槽时用 `+ turn turns-to-travel *turn-adjustment*`。

---

## 执行/撤销订单与错误协议

| 符号 | 说明 |
|------|------|
| `execute-order` | **可变**：按 owner 增加 `departures-*` 于 `turn`，增加对端 `arrivals-*` 于 `turn+travel`；若桶下溢则 **`error 'future-impossible`**。 |
| `undo-order` | 同函数 `undo t` 反号。 |
| `execute-orders` / `undo-orders` | `map` 单笔。 |

---

## 挂单栈与缓存

| 符号 | 说明 |
|------|------|
| `move-and-stuff` | 槽位 **`move`**（order 列表）、**`cache`**（每星一条关联表）。 |
| `*moves*` | 当前 **动态**链：最近 `with-orders` 压入的 `move-and-stuff`  cons 栈。 |
| `orders-since` | 从栈顶向下直到某 `move-and-stuff` 为止收集 move。 |
| `lookup-cached-stuff` | 在某星 cache 里按键找值，并返回是否命中及之后订单前缀。 |
| `set-cached-stuff` | 向**当前栈顶**该星 cache 推入 `(key . values)`。 |
| `with-orders` | **核心宏**：`execute-orders` → 压新层 `*moves*` → body（SBCL 下常无中断地跑）→ **`unwind-protect` 必 `undo-orders`**。 |

---

## 截断地平线

| 符号 | 说明 |
|------|------|
| `truncate-planet` | 把四向量的长度裁到 `n+1`，**复制**原数据前缀。 |
| `truncate-game` | 对所有 planet 调用。配合 `horizon` 缩短仿真长度。 |

---

## 参考

- [CHAMPION_MODULES_zh.md](../CHAMPION_MODULES_zh.md)
- [io_lisp_zh.md](io_lisp_zh.md)（读入后填充 `planet`/`game`）
