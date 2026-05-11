# `package.lisp` — 逐符号中文说明（`src/package.lisp`）

自动符号表：[src__package.md](../../by-file/src__package.md)。

---

## `defpackage :planet-wars`

| 子句 | 说明 |
|------|------|
| `:nicknames :pw :pwbot` | 短包名；`MyBot.lisp` 等用 `pwbot::main` 入口。 |
| `:use :cl #+sbcl :sb-bsd-sockets :pw-util` | 基础 Common Lisp；**SBCL** 下再 use 套接字实现；始终 use 本项目的 `planet-wars-util`。 |
| `:export` | 仅导出两个符号给「外部世界 / 其他系统」：见下表。 |

---

## 导出符号

| 符号 | 引擎/脚本侧用途 |
|------|----------------|
| **`play`** | 主循环：读 stdin 局面、每回合调用 `compute-orders`、写 `order` 行与 `go`。定义在 [`play.lisp`](../../../src/play.lisp)；`RunMyBot.lisp` 直接 `(pw:play)`。 |
| **`start-server-for-proxy-bot`** | 在 **localhost:41807** 监听 TCP，每连接一个新线程里跑 `play`（可换 `player-class`）。供代理 Bot 与本地引擎桥接。见 [`play.lisp`](../../../src/play.lisp) **L29**。 |

---

## 未导出但同包内的模块

`model` / `io` / `player` / `alpha-beta` 等均在 `:planet-wars` 包内 **`in-package :planet-wars`**，**无需**从 `package.lisp` 再 export；比赛提交只要求能从 `play` 与入口加载链触达 `compute-orders`。

---

## 参考

- [CHAMPION_MODULES_zh.md](../CHAMPION_MODULES_zh.md)
- [src/README.md](../../../src/README.md)
