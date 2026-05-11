# `planet-wars/src/usocket-0.4.1/`

[**usocket 0.4.1**](https://common-lisp.net/project/usocket/)：**跨实现 BSD 风格 socket API**。

本 Bot 仅在 **TCP 代理 / 远端对弈**时需要；普通 `PlayGame.jar` 本地 stdin/stdout 模式可以不经过 usocket。

| 子目录 | 说明 |
|--------|------|
| [`backend/`](backend/README.md) | 各 Lisp 实现的底层后端（ACL、SBCL、CMU…）。 |
| [`doc/`](doc/README.md) | 简短说明与设计笔记。 |
| [`notes/`](notes/README.md) | 备忘。 |
| [`test/`](test/README.md) | **上游单元测试**：不参与比赛推理，已从符号生成脚本排除。 |

**符号文档**：[`../../docs/symbols/INDEX.md`](../../docs/symbols/INDEX.md) 列出本目录下全部 `.lisp`（不含 `test/`）。
