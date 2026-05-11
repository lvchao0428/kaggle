# `planet-wars/bin/`

本地启动 Lisp、跑图、打 TCP 的可执行脚本。路径均相对 `planet-wars/` 根目录调用。

| 文件 | 作用 |
|------|------|
| [`run-lisp.sh`](run-lisp.sh) | 读取上级 `config` 里的 `LISP=sbcl|…`；若为 **sbcl** 转调 `run-sbcl.sh`，否则把 `--load` / `--eval` 转成 **Allegro** 的 `run-acl.sh` 参数。 |
| [`run-sbcl.sh`](run-sbcl.sh) | SBCL 包装：设置 `SBCL_HOME`、堆大小，执行 `sbcl`（见脚本内具体选项）。 |
| [`run-acl.sh`](run-acl.sh) | Allegro CL 包装（竞赛时代常用商业实现）。 |
| [`run-bot.sh`](run-bot.sh) | 单次启动冠军 Bot 进程的薄封装（具体参数见文件）。 |
| [`run-proxy-bot.sh`](run-proxy-bot.sh) | 启动 **TCP 代理 Bot**（[`../src/proxy-bot`](../src/proxy-bot/README.md)），便于接测试服。 |
| [`play-tcp`](play-tcp) | 与远程/本地 TCP 对局相关的启动器（读脚本内用法）。 |
| [`play-zeroviz`](play-zeroviz) | 无可视化或与zeroviz兼容的对局模式（读脚本内注释）。 |
| [`make-submission.sh`](make-submission.sh) | **不要单独手跑**：由 `make submission` 调用；把指定目录下 `.lisp` `.asd` 与 `version` 打成 zip。 |
| [`on-all-maps.sh`](on-all-maps.sh) | 在 [`../maps`](../maps/README.md) 上批量评测/遍历（参见脚本注释）。 |
| [`stats-so-far.sh`](stats-so-far.sh) | 汇总统计类辅助脚本（见文件内注释）。 |

依赖：上级目录的 **`config`** 文件（通常由 `./configure` 或手工生成）指明 SBCL/Lisp 可执行路径与 `LISP` 变量。
