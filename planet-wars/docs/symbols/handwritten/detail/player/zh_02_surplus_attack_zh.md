# `player.lisp` 卷二：Surplus 与全军增援 Future（约 L174–501）

上一级：[player_lisp_zh.md](../player_lisp_zh.md)

---

## Surplus（可安全派发）

| 符号 | L | 说明 |
|------|---|------|
| **`cumulative-surplus`** | **191** | 从地平线末反向扫：双方「还能派多少而不失星/不把失败提前」的合成向量（**己方正、敌方负**，互斥至多一方非零）。 |
| **`uncumulate-surplus!`** / **`uncumulate-surplus`** | **241–255** | 差分 / 惰性展开：「每回合**增量**可派」。 |
| **`surplus`** | **263** | 从 cumulative 中取 **当前回合**起可用的双方 surplus 语义包装。 |

---

## 全军突击 Future

| 符号 | L | 说明 |
|------|---|------|
| **`full-attack-future`** | **273** | 继承 **`future`**，语义上假设 **所有己方 surplus** 可被调度去增援一颗目标星。 |

| 符号 | L | 说明 |
|------|---|------|
| **`compute-full-attack-future`** | **290** | 对每个相关星算 surplus→把 surplus **灌进到达桶**构造临时 **`arrivals-*`**→`compute-future*`。 |
| **`add-surplus-into-arrivals`** | **325** | 把某星某方的 surplus **按 ETA 叠加**到目标星的到达向量。 |
| **`compute-full-attack-arrivals`** | **378** | 汇总「全员增援模式下」目标的到达序列。 |
| **`update-full-attack-arrivals`** | **407** | 在已有到达基底上微调（用于迭代/局部修正）。 |
| **`make-full-attack-arrivals`** | **452** | 构造整套用于 full-attack 评估的 arrivals 快照。 |

这些函数与 **`evaluate/full-attack`**、`takeablep` 一脉相承：判断「举国之力能否抢了/保住某星」。

---

下一卷：[zh_03_eval_horizon_zh.md](zh_03_eval_horizon_zh.md)
