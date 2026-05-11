# 冠军 Bot 核心模块中文综述

本文 **手写维护**；与「按文件自动生成的符号表」互补。生成表见 [`by-file/`](../by-file/) 与 [`INDEX.md`](../INDEX.md)。

**逐符号手写中文版**（按文件/分卷，重跑生成器不会覆盖）：[`detail/`](detail/) 目录。

---

## `package.lisp`（`src/package.lisp`）

- **作用**：定义 `planet-wars` 包、**导入/导出**符号表，使 `model` / `io` / `player` 等文件共享同一命名空间而不循环 `import`。
- **在整盘棋里**：无运行时决策，仅编译期/加载期配置。

**关键符号**：`defpackage`、`in-package`。细目见 [src__package.md](../by-file/src__package.md)。**逐符号中文**：[detail/package_lisp_zh.md](detail/package_lisp_zh.md)。

---

## `timer.lisp`（`src/timer.lisp`，`#+sbcl`）

- **作用**：在 **SBCL** 下调度/包装 **墙钟超时**（与 `with-timeout` 一类宏配合），保证 `compute-orders` 在官方 0.8s 限制内总能返回已登记的 **best move so far**。
- **在整盘棋里**：防止搜索写满时间片被平台杀进程；**Allegro** 分支在 `player.lisp` 里用 `mp:with-timeout` 等价实现。

详见 [src__timer.md](../by-file/src__timer.md)。**逐符号中文**：[detail/timer_lisp_zh.md](detail/timer_lisp_zh.md)。

---

## `model.lisp`

- **作用**：**游戏世界数据结构**与**只读几何/邻接**遍历宏。
- **核心类型**：
  - **`planet`**：`owner`、`n-ships`、`growth`、双方按回合的 **`arrivals-1/2`** / **`departures-1/2`**（与引擎协议对齐的桶）、**`neighbours`**（按 `turns-to-travel` 分组的邻居列表）。
  - **`game`**：所有 `planet` 向量、终局后仍飞在路上的船统计。
  - **`order`**：`source`/`destination`/`n-ships`/执行 **`turn`**（可到未来回合），这是 Planet Wars 「多拍调度」的核心。
- **在整盘棋里**：一切 **future/surplus** 推演、`step`/`move` 合法性，都在此类型上调 `owner`、`arrivals-of-player`、**`do-neighbours`** 等完成。

符号级清单（自动生成）：[src__model.md](../by-file/src__model.md)。**逐符号中文**：[detail/model_lisp_zh.md](detail/model_lisp_zh.md)。

---

## `io.lisp`（`src/io.lisp`）

- **作用**：与 **引擎文本协议** 对话——**读入一整帧 `game`**、写出当前选手的 **`order`** 列表、`go`/`ready` 等握手行。
- **在整盘棋里**：边界上把「stdin 字节」变成 **`game` 对象**供 `compute-orders`；把 **Lisp move** 变成引擎可执行的文本。

详见 [src__io.md](../by-file/src__io.md)。**逐符号中文**：[detail/io_lisp_zh.md](detail/io_lisp_zh.md)。

---

## `play.lisp`（`src/play.lisp`）

- **作用**：封装 **挂订单上下文**——`with-orders` 建立「本回合考虑的出发/取消」、`with-game`/`read-game`，以及 **planet 在给定全局 move 向量下的一致性更新**。
- **在整盘棋里**：搜索树或 α–β试探时需要在 **克隆的可变状态**里反复 **提交/回放**候选 move；本章提供 **缓存与数据结构 `caches-and-moves`**。

详见 [src__play.md](../by-file/src__play.md)。**逐符号中文**：[detail/play_lisp_zh.md](detail/play_lisp_zh.md)。

---

## `player.lisp`（`src/player.lisp`）

Bocsimackó **全部棋力**：模拟、打分、出兵。

### 子系统划分

| 模块 | Lisp 前缀 / 代表作 | 棋理 |
|------|-------------------|------|
| Timeout 兜底 | `*best-move-so-far*`、`register-evaluated-move`、`with-best-move-on-timeout` | 时钟到点仍返回当前最优 |
| Future | `future` 类、`compute-future*`、`compute-future` | 在固定到达/出动桶下推演 **归属 + 兵力** 时间向量 |
| Surplus | `cumulative-surplus`、`surplus` | 「还能安全派出多少」按回合扫回朔 |
| Full attack | `full-attack-future`、`compute-full-attack-future`、`*-arrivals` 成套 | **全员 surplus 增援一颗星** 的抵达假设 |
| 评估 | `score`、`evaluate/full-attack`、`evaluate-planet`、`candidate-min-turns-to-arrive`、`eval*` | **产能差 + 占中损益 + snipe 分档到达 + tempo（min-turn-to-depart）** |
| Horizon / 安全投资 | `safety-margin`、`safe-to-invest-p`、`horizon` | **缩短 lookahead**；过滤明显亏中立投资 |
| 走子生成 | `compute-step-target`、`find-step*`、`generate-moves-*` | **step 目标向量 → steps → sorted moves** |
| 入口 | `bocsimacko`、`compute-orders` | 开局 **`<3` 己方星**：`alpha-beta*` `width (4 4 4 4)`；否则 **1-ply 最高分 move** |

逐符号自动生成表：[src__player.md](../by-file/src__player.md)。**手写逐符号中文（分卷）**：[detail/player_lisp_zh.md](detail/player_lisp_zh.md)。

---

## `alpha-beta.lisp`

- **作用**：**浅层极小极大搜索**脚手架：`alpha-beta` 泛型递归、`maybe-evaluate-state` 叶子调用 **`evaluate/full-attack`**、`list-actions` 从 `generate-and-score-moves` **截断宽度 `*widths*`**、`call-with-action` 在当前 **模拟 game**上挂单层 move。
- **在整盘棋里**：只对 **前两步左右**的对手响应做 pessimistic lookahead；宽度 **4** 控制分支因子。

详见 [src__alpha-beta.md](../by-file/src__alpha-beta.md)。**逐符号中文**：[detail/alpha_beta_lisp_zh.md](detail/alpha_beta_lisp_zh.md)。

---

## `MyBot.lisp` / `RunMyBot.lisp` / `ProxyBot.lisp` / `setup.lisp`

- **`MyBot.lisp`**：单文件提交入口，一般 **载入系统 + 导出引擎期待的函数名**。  
- **`RunMyBot.lisp`**：**本地 SBCL `--script` / image** 启动器。  
- **`ProxyBot.lisp`**：**TCP** 对战入口包装。  
- **`setup.lisp`**：镜像内部 **Quicklisp-less** 环境下的 **ASDF**/路径初始化。

自动生成索引见 [`INDEX.md`](../INDEX.md)。**根入口中文说明**：[detail/entry_lisp_zh.md](detail/entry_lisp_zh.md)。
