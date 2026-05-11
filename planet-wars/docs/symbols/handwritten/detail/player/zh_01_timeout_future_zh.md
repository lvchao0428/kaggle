# `player.lisp` 卷一：超时钩子与 Future（L3–171）

上一级：[player_lisp_zh.md](../player_lisp_zh.md)

---

## 全局调试/搜索深度变量

| 符号 | L | 说明 |
|------|---|------|
| `*depth*` | **3** | α–β 或生成阶段当前递归深度；**0** 表示根层着法。 |
| `*turn*` | **4** | 本局相对回合索引（与 `bocsimacko` 槽位 `turn` 概念不同：后者为引擎已走步数）。 |
| `(declaim (type ... *depth* *turn*))` | **5** | 16 位无符号类型声明。 |

---

## 超时兜底

| 符号 | L | 说明 |
|------|---|------|
| `*best-move-so-far*` | **11** | `cons move score`：**根层已评估到的最优着法**。 |
| `register-evaluated-move` | **13** | 仅当 **`*depth*=0`** 且分数更优时更新 `*best-move-so-far*`。 |
| `with-best-move-on-timeout` | **19–26** `#+allegro` | 包在 **`mp:with-timeout`**：`timeout` 秒到时若已有 best 则 **`car`**，否则空列表。 |
| `with-best-move-on-timeout` | **29–38** `#+sbcl` | **`handler-case`** + **`sb-ext:with-timeout`**：`timeout` condition 同样返回 best 或 `()`。 |

---

## `future` 类

| 符号 | L | 说明 |
|------|---|------|
| **`future`** 槽位：`planet`、`owners`、`n-ships`、`balance` | **46–53** | 单颗星在裁剪地平线内按回合的归属/兵力向量；**`balance`**：攻中立过程双方损耗差（己方视角）。 |
| `print-object` (`future`) | **58** | 打印 id、首尾 owner、balance。 |

| 符号 | L | 说明 |
|------|---|------|
| `first-owner` | **70** | `(elt (owners future) *turn*)`：当前拍起算的首属主。 |
| `last-owner` | **73** | 地平线末属主。 |

---

## 条件与核心仿真

| 符号 | L | 说明 |
|------|---|------|
| **`future-impossible`** | **78** | `execute-order` 与 future 推演不一致时由 **`cerror "Continue"`** 恢复的控制流错误。 |
| **`compute-future*`** | **96** | 给定 **arrivals/departures** 四向量与可选 **`resolve-battle-fn`**，逐回合 **到港→写 owners/n-ships→离港+增长**；非法离港触发 **`future-impossible`**。返回值：`owners`、`n-shipss`、`balance`。 |
| **`compute-future`** | **167** | 封装：用星球当前快照调 `compute-future*` 并 **`make-instance 'future`**。 |

---

下一卷：[zh_02_surplus_attack_zh.md](zh_02_surplus_attack_zh.md)
