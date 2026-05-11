# `planet-wars/src/` — 源代码总索引

[Gábor Melis](https://github.com/melisgl/planet-wars) 的冠军 Bot 主逻辑与本仓库 **ASD F 捆绑的依赖**。

## ASDF 载入顺序（参见 [`planet-wars.asd`](../planet-wars.asd)）

`serial t` **串行**编译，顺序即依赖顺序：

```
#+SBCL timer.lisp           ; SBCL 定时器钩子
package.lisp → model.lisp → io.lisp → play.lisp → player.lisp → alpha-beta.lisp
```

系统级依赖：`parse-number`、`split-sequence`、`usocket`、`alexandria`、本项目 `planet-wars-util`（[`util/` 子包](util/README.md)）。

```mermaid
flowchart LR
  PKG[package]
  MD[model]
  IO[io]
  PL[play]
  PW[player]
  AB[alpha-beta]
  PKG --> MD --> IO --> PL --> PW --> AB
```

## 冠军逻辑文件（逐项符号表）

手写 **逐符号中文** 与 **带中文注释的同构 `.lisp`**：[`../docs/symbols/handwritten/detail/`](../docs/symbols/handwritten/detail/) · [`../docs/translated-sources/`](../docs/translated-sources/)

| 文件 | 综述与符号索引 |
|------|----------------|
| [`player.lisp`](player.lisp) | [自动生成](../docs/symbols/by-file/src__player.md) · [逐符号中文（分卷）](../docs/symbols/handwritten/detail/player_lisp_zh.md) · [综述](../docs/symbols/handwritten/CHAMPION_MODULES_zh.md) |
| [`model.lisp`](model.lisp) | [自动生成](../docs/symbols/by-file/src__model.md) · [detail/model_lisp_zh.md](../docs/symbols/handwritten/detail/model_lisp_zh.md) · 同上 |
| [`alpha-beta.lisp`](alpha-beta.lisp) | [自动生成](../docs/symbols/by-file/src__alpha-beta.md) · [detail/alpha_beta_lisp_zh.md](../docs/symbols/handwritten/detail/alpha_beta_lisp_zh.md) · 同上 |
| [`io.lisp`](io.lisp) | [自动生成](../docs/symbols/by-file/src__io.md) · [detail/io_lisp_zh.md](../docs/symbols/handwritten/detail/io_lisp_zh.md) |
| [`play.lisp`](play.lisp) | [自动生成](../docs/symbols/by-file/src__play.md) · [detail/play_lisp_zh.md](../docs/symbols/handwritten/detail/play_lisp_zh.md) |
| [`timer.lisp`](timer.lisp) | [自动生成](../docs/symbols/by-file/src__timer.md) · [detail/timer_lisp_zh.md](../docs/symbols/handwritten/detail/timer_lisp_zh.md)（仅 SBCL 编译） |
| [`package.lisp`](package.lisp) | [自动生成](../docs/symbols/by-file/src__package.md) · [detail/package_lisp_zh.md](../docs/symbols/handwritten/detail/package_lisp_zh.md) |

## 子目录

| 目录 | README |
|------|--------|
| [`alexandria/`](alexandria/README.md) | 第三方 Lisp 便携库 |
| [`parse-number/`](parse-number/README.md) | 数字解析库 |
| [`split-sequence/`](split-sequence/README.md) | 序列拆分库 |
| [`usocket-0.4.1/`](usocket-0.4.1/README.md) | 套接字抽象 |
| [`util/`](util/README.md) | `planet-wars-util` ASDF 子系统 |
| [`proxy-bot/`](proxy-bot/README.md) | TCP 对战代理 |

## 全量 Lisp 自动生成索引

- [`../docs/symbols/INDEX.md`](../docs/symbols/INDEX.md) — 每一 `.lisp` → 一页 Markdown。

重跑：`python3 scripts/gen_lisp_symbol_docs.py`（需在 `planet-wars/` 下或带正确路径）。
