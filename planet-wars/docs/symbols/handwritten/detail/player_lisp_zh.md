# `player.lisp` — 手写中文索引（`src/player.lisp`）

自动符号表（含行号）：[src__player.md](../../by-file/src__player.md)。

本篇按专题**分卷**，每卷覆盖一段连续 **`def*`**；条目中的 **「Lxxxx」均指 `player.lisp` 源码行号**。

| 卷 | 文件 | 主要范围（约） |
|----|------|----------------|
| 1 — 超时与 Future | [player/zh_01_timeout_future_zh.md](player/zh_01_timeout_future_zh.md) | `*depth*` / `with-best-move-on-timeout`、`future`、`compute-future*` |
| 2 — Surplus 与全军突击 | [player/zh_02_surplus_attack_zh.md](player/zh_02_surplus_attack_zh.md) | `cumulative-surplus`、`surplus`、`full-attack-*` |
| 3 — 评估与地平线 | [player/zh_03_eval_horizon_zh.md](player/zh_03_eval_horizon_zh.md) | `score`、`evaluate*`、`safe-to-invest-p`、`horizon` |
| 4 — 走子生成与入口 | [player/zh_04_moves_entry_zh.md](player/zh_04_moves_entry_zh.md) | `find-step*`、`generate-*`、`bocsimacko`、`compute-orders` |

---

## `#+nil` 与文末块注释

- 文末 **`#| ... |#`**（约 **L1181** 起）为开发注释，**不参与**自动生成索引的「顶层 def*」语义。
- Allegro/SBCL 各一份 **`with-best-move-on-timeout`**（**L19** vs **L29**）；实际编译只有一种实现生效。

---

## 参考

- 模块鸟瞰：[CHAMPION_MODULES_zh.md](../CHAMPION_MODULES_zh.md)
- α–β 衔接：[alpha_beta_lisp_zh.md](../alpha_beta_lisp_zh.md)
