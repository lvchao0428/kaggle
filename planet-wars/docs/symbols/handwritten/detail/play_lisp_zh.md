# `play.lisp` — 中文逐符号说明（`src/play.lisp`）

自动符号表：[src__play.md](../../by-file/src__play.md)。

---

## `main`（**L4**）

Kaggle **`MyBot` 二进制**的顶层入口：`with-reckless-exit`、`with-errors-logged`（失败可退出进程）→ 调用 **`play`**。

---

## `compute-orders`（**L10，`defgeneric`**）

多 **method** 的泛型钩子：`(compute-orders bot input)`。**`player.lisp`** 上对 **`bocsimacko`** 特化返回 **order 列表**（可为未来回合挂单的 `order`）。

---

## `play`（**L12**）

| 要点 | 说明 |
|------|------|
| 默认 **`player`** | `(make-instance 'bocsimacko)` |
| IO | `*standard-input*` / `*standard-output*` |
| 主循环 | 有字符可读则每 **`turn`**：`compute-orders` → 拆分 **`current-order-p`** → 仅写 **`turn=0`** 订单；写 **`go`** 并 **`force-output`**。 |
| 日志 | `pw-util:logmsg` 打 turn 与订单。 |

此处 **不产生**超时；超时在 **`bocsimacko` 的 `compute-orders` method**（`player.lisp`）里由宏嵌套。

---

## `start-server-for-proxy-bot`（**L29**）

| 要点 | 说明 |
|------|------|
| 监听 | Allegro：`"localhost"`；SBCL：`#(127 0 0 1)`；端口 **41807**，`reuse-address`。 |
| 每连接 | 新线程：`play` **同一 stream** 作 input/output，`player-class` 默认可仍为 `bocsimacko`。 |
| `one-shot` | 若真，接单一次后退出循环。 |

与 [`ProxyBot.lisp`](../../../ProxyBot.lisp)、[`src/proxy-bot/README.md`](../../../src/proxy-bot/README.md) 配合做本地 TCP 对战。

---

## 参考

- [package_lisp_zh.md](package_lisp_zh.md)（导出 `play`、`start-server-for-proxy-bot`）
- [player_lisp_zh.md](player_lisp_zh.md)
