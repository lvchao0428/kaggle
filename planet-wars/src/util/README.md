# `planet-wars/src/util/` — `planet-wars-util`

ASDF 中声明的 `:planet-wars-util` 子系统，为本 Bot 提供与 **地图 / 数据结构**相关的通用小工具（独立于 `planet-wars` 包名空间可避免循环依赖）。

| 文件 | 作用 |
|------|------|
| [`package.lisp`](package.lisp) | `planet-wars-util` 包的 `defpackage`、`in-package`。 |
| [`util.lisp`](util.lisp) | B 树、`btree`、`btree-insert` / `btree-insert-string` / ` btree-search` / ` btree-search-string` / ` btree-as-alist`、`btree-remove` ，以及哈希表包装与邻接预处理等——供 **建树、按前缀查地图元数据**之用（参见符号表）。 |

**符号自动生成**：[文档](../../docs/symbols/by-file/src__util__util.md) · [package](../../docs/symbols/by-file/src__util__package.md)
