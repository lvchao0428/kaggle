# 手写综述（不会被生成器覆盖）

- **[CHAMPION_MODULES_zh.md](CHAMPION_MODULES_zh.md)** — `package`/`timer`/`model`/`io`/`play`/`player`/`alpha-beta` 及入口文件的**模块级中文**说明。
- **[detail/](detail/)** — **逐符号**手写中文：`timer`/`package`/`model`/`io`/`play`/`alpha-beta`/`entry`/`player`（player 多分卷）。与 [`../by-file/`](../by-file/) 自动生成表双向链接。
- **[中文注释 Lisp 副本](../../translated-sources/)** — `docs/translated-sources/` 下与 `src/` 同构的 `.lisp`（仅增 `;;;`/`;;`，非注释行与原版一致）；**不用于提交**。

自动生成的 **`../by-file/*.md`** 若重跑 `scripts/gen_lisp_symbol_docs.py` 会被覆盖；**本目录（含 `detail/`）除外**。

**说明**：模块级读 `CHAMPION_MODULES_zh.md`，逐符号读 `detail/`，想「开着同名文件对照」用 `translated-sources/`；三者均不修改上游 `src/*.lisp`。
