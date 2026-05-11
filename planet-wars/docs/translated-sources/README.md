# `translated-sources/` — 中文注释 Lisp 源码副本

本目录下的 `.lisp` 是 **`planet-wars/src/`** 与仓库根 **`MyBot.lisp`、`RunMyBot.lisp`、`ProxyBot.lisp`、`setup.lisp`** 的**对照读本**：在**不改动任何非注释行**的前提下，用 `;;;` / `;;` 追加中文说明。用语可与 [`../symbols/handwritten/detail/`](../symbols/handwritten/detail/) 对照。

## 用途与免责

- **仅供阅读理解**，与手写综述 [`../symbols/handwritten/detail/`](../symbols/handwritten/detail/) 互为补充。
- **竞赛提交 / 评测 / 镜像打包**请以原版为准；不要 `load` 本目录顶替 `src/`（路径、`in-package`、`#+sbcl` 等与原版一致但未保证在此目录树下可编译）。
- **不含** Alexandria、usocket、parse-number、split-sequence、`src/util`、`src/proxy-bot` 等依赖库源码。

## 目录结构（镜像）

| 副本 | 原版 |
|------|------|
| [`src/package.lisp`](src/package.lisp) | [`../../src/package.lisp`](../../src/package.lisp) |
| [`src/model.lisp`](src/model.lisp) | [`../../src/model.lisp`](../../src/model.lisp) |
| [`src/io.lisp`](src/io.lisp) | [`../../src/io.lisp`](../../src/io.lisp) |
| [`src/play.lisp`](src/play.lisp) | [`../../src/play.lisp`](../../src/play.lisp) |
| [`src/player.lisp`](src/player.lisp) | [`../../src/player.lisp`](../../src/player.lisp) |
| [`src/alpha-beta.lisp`](src/alpha-beta.lisp) | [`../../src/alpha-beta.lisp`](../../src/alpha-beta.lisp) |
| [`src/timer.lisp`](src/timer.lisp) | [`../../src/timer.lisp`](../../src/timer.lisp)（SBCL） |
| [`MyBot.lisp`](MyBot.lisp) | [`../../MyBot.lisp`](../../MyBot.lisp) |
| [`RunMyBot.lisp`](RunMyBot.lisp) | [`../../RunMyBot.lisp`](../../RunMyBot.lisp) |
| [`ProxyBot.lisp`](ProxyBot.lisp) | [`../../ProxyBot.lisp`](../../ProxyBot.lisp) |
| [`setup.lisp`](setup.lisp) | [`../../setup.lisp`](../../setup.lisp) |

## 维护

原版变更后，应对照 diff 同步副本：非注释差异必须反映到副本的非注释部分；注释可另行人工增补。
