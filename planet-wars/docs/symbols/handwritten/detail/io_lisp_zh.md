# `io.lisp` — 中文逐符号说明（`src/io.lisp`）

自动符号表：[src__io.md](../../by-file/src__io.md)。

与 **引擎文本协议** 对接：stdin 上的 `P`/`F` 行 → **`game`**；stdout 上写 **`order`** 列表（及由 `play.lisp` 写的 `go`）。

---

## 行星行

| 符号 | 说明 |
|------|------|
| `parse-planet` | 解析 `P x y owner n-ships growth`，构造 **`planet`**，四向 `arrivals-*`/`departures-*` 长度为 **`1+ *n-turns-till-horizon*`**（须已绑定）。 |
| `read-planet` | `read-line` + `parse-planet`。 |
| `write-planet` | 输出一行 `P`（调试用/非主循环路径）。 |

---

## 舰队行

| 符号 | 说明 |
|------|------|
| `parse-fleet` | 解析 `F` 行；按 **owner、船数、源/宿 id、航程、剩余回合** 把在途船写入 **目标星** 的 `arrivals-*` 桶，或计入 **game** 的 `n-ships-beyond-*`（若剩余回合超出地平线）。内含与 `turns-to-travel` 一致性 **warn**。 |

---

## 辅助

| 符号 | 说明 |
|------|------|
| `group-by-turn` | 按 `turn-fn` 把列表聚成相邻同 turn 组。 |
| `group-planets` | 对单星与其它所有星算 `turns-to-travel*`，排序后 `group-by-turn` → **`neighbours` 格式**。 |

---

## 整帧读入

| 符号 | 说明 |
|------|------|
| `read-game` | **两阶段**：先读到空行/`go` 之前读完所有 **`P`**，赋 `id`、填 **`turns-to-neighbours`** 与 **`neighbours`**；再读 **`F`** 行直到遇 `go`；返回 **`game`**。 |

依赖动态变量：**`*n-turns-till-horizon*`**（与 `parse-planet`、`parse-fleet` 长度一致）。

---

## 写出订单

| 符号 | 说明 |
|------|------|
| `write-order` | `source-id dest-id n-ships` 一行三个整数（引擎格式）。 |
| `write-orders` | 对列表逐笔 `write-order`。 |

---

## 参考

- [model_lisp_zh.md](model_lisp_zh.md)
- [play_lisp_zh.md](play_lisp_zh.md)
