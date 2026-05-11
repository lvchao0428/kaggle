# `player.lisp` 卷三：打分、节拍与安全投资地平线（约 L503–1110）

上一级：[player_lisp_zh.md](../player_lisp_zh.md)

---

## 价值与全盘评估

| 符号 | L | 说明 |
|------|---|------|
| **`score`** | **503** | 单颗星的 **`future` 向量**打分：加权 **产量差**与 **敌方船惩罚**（带 horizon 衰减等），随代码内注释的实现细节为准。 |
| **`evaluate/full-attack`** | **544** | 在 **举国突击**假定下对各星 `evaluate-planet`，合成局面值。 |
| **`evaluate-planet`** | **567** | 单星：neutral 与中立 steal、敌我占有分支；中立时接 **`candidate-min-turns-to-arrive`** snipe 档位。 |

---

## Snipe / 节拍

| 符号 | L | 说明 |
|------|---|------|
| **`first-non-neutral-turn`** | **591** | 某颗 `future` 首次出现非中立属主的回合索引。 |
| **`candidate-min-turns-to-arrive`** | **595** | 估算敌我「最早能落地参与争夺」的 **分档 ETA**（冠军帖文强调的 snipe-aware 中立估值核心）。 |
| **`evaluate-non-neutral-planet`** | **610** | 非中立星上的专用分支（攻守、夺回等）。 |
| **`eval*`** | **642** | 全盘 future 数组上的聚合评估入口（常与 `evaluate/full-attack` / α–β叶子对接）。 |

---

## 安全边际与投资闸门

| 符号 | L | 说明 |
|------|---|------|
| **`safety-margin`** | **1025** | 对玩家 1 各星构造「相对最低安全阈值」的余量向量，用于判断是否敢投资中立。 |
| **`safety-margin!`**（带 `!`） | **1052** | 破坏性/缓存版：可把 **累计可能到达**编入 margin 更新。 |
| **`n-turns-to-break-even`** | **1069** | 收复中立需要多少回合回本（产能相关）。 |
| **`safe-to-invest-p`** | **1079** | 给定 margin、中立兵力、敌方距离、`growth`，是否允许把该中立纳入投资候选。 |

---

## 动态地平线

| 符号 | L | 说明 |
|------|---|------|
| **`horizon`** | **1092** | 默认下限 **30**，但可 **伸长**到「第三便宜的安全中立」回本回合；再回到 `truncate-game` 裁短向量（参见 **`with-game`**）。 |

---

下一卷：[zh_04_moves_entry_zh.md](zh_04_moves_entry_zh.md)
