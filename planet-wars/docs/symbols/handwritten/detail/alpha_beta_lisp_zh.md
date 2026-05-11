# `alpha-beta.lisp` — 中文逐符号说明（`src/alpha-beta.lisp`）

自动符号表：[src__alpha-beta.md](../../by-file/src__alpha-beta.md)。

极小极大 **`alpha-beta`** 脚手架，与 **`player.lisp`** 的 **`evaluate/full-attack`**、`generate-and-score-moves` 串联；开局 **`n-friendly-planets < 3`** 时用 **`alpha-beta*`** 固定宽度 **`(4 4 4 4)`**。

---

## 核心递归

| 符号 | 说明 |
|------|------|
| `alpha-beta` | 泛型 **α–β**：深度、α/β、当前 `game`、`player`、`move`。叶或截断调用 **`maybe-evaluate-state`**。 |

---

## 轮次与挂单

| 符号 | 说明 |
|------|------|
| `player-to-move` | 由深度奇偶断言当前轮到哪一方（断言式剪枝）。 |
| `*widths*` | **分支宽度**列表（每层截断多少个 action）；NIL 则用默认推导。 |
| `*deferred-orders*` | 动态保存「延后执行」的一串 order，仿真时与 **`with-undeferred-orders`** 配合，保证状态与挂单一致。 |
| `split-past-and-future-orders` | 把挂单拆成已到拍 / 未到拍两段。 |

---

## 宏与安全仿真

| 符号 | 说明 |
|------|------|
| `with-undeferred-orders` | 在给定挂单上下文中跑 body，`unwind` 清理。 |

---

## 估价与分枝

| 符号 | 说明 |
|------|------|
| `maybe-evaluate-state` | 叶节点：构造对手 **full-attack surplus**、`eval*`、`evaluate/full-attack` → 分值。 |
| `subseq*` | 安全 `subseq`（越界容错）。 |
| `list-actions` | 调用 **`generate-and-score-moves`**，按 **`*widths*` 当前深度**截取前 K 个 **move**。 |
| `call-with-action` | 在某个 **单层 move**（order 向量）上做 **`with-orders`** 递归下一层 **`alpha-beta`**。 |

---

## 调试与包装

| 符号 | 说明 |
|------|------|
| `trace-alpha-beta` | 调试用递归包装，可打追踪。 |
| `alpha-beta*` | 外层入口：**bind `*depth*`**，`split-past-and-future-orders`，启动根层 `alpha-beta`。 |

---

## 参考

- [player/zh_04_moves_entry_zh.md](player/zh_04_moves_entry_zh.md)（`compute-orders` 里如何调用 `alpha-beta*`）
