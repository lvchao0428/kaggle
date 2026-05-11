# 仓库根 Lisp 入口 — 中文说明

自动符号索引（若有对应 `.md`）：[INDEX.md](../../INDEX.md)。

这些文件 **`in-package :cl-user`** 或 ASDF **之外**的职责是：**加载 ASDF → 导出可执行镜像 / 单次 REPL play**。

---

## `setup.lisp`

| 内容 | 说明 |
|------|------|
| 定位 `planet-wars` 根路径 | `(pathname-directory *load-truename*)` |
| 扫描 `**/*.asd` | 填入 **`asdf:*central-registry*`**（竞赛环境无 Quicklisp 时使用）。 |

不向 `:planet-wars` 导出符号。

---

## `MyBot.lisp`

| 内容 | 说明 |
|------|------|
| 竞赛可见 | 服务器用它识别 **Common Lisp** 提交。 |
| **`load setup.lisp`** | 再接 **`asdf:oos 'load-op :planet-wars`**。 |
| **stderr→stdout** | 避免编译守护进程误判失败。 |
| 错误时 **`handler-bind`** 打印系统信息与目录。 |
| `parse-config-line` / `path-to-lisp` | 读 **`config`** 里 Lisp 可执行路径。 |
| **`dump`** | Allegro：**dumplisp** + 写出 shell wrapper；SBCL：**`save-lisp-and-die`**，toplevel **`pwbot::main`**（见 `planet-wars-util`/`play.lisp` 链）。 |

---

## `RunMyBot.lisp`

| 内容 | 说明 |
|------|------|
| 用途 | **不导出镜像**的快速回归：`bin/run-bot.sh` 等用这个文件。 |
| IO | **`standard-output`** 重定向到 **`*error-output*`**，stdin 仍可被引擎读。 |
| 流程 | `setup` → `require :planet-wars` → **`(pw:play)`**。 |

---

## `ProxyBot.lisp`

| 内容 | 说明 |
|------|------|
| 加载 | `:proxy-bot` 系统（监听/代理）；同样避免污染 stdout。 |
| **`dump`** | toplevel 设为 **`pw-proxy-bot:proxy`**（与 **`start-server-for-proxy-bot`** / 镜像配套）。 |

---

## 与本目录其它文档的关系

- 运行态规则与数据结构 → [model_lisp_zh.md](model_lisp_zh.md)、[play_lisp_zh.md](play_lisp_zh.md)
- 包导出 → [package_lisp_zh.md](package_lisp_zh.md)
