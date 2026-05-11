# `planet-wars/tools/`

官方 **Java** 套件中的可执行 jar（二进制，本仓库不写内部 API 详解）。

| 文件 | 作用 |
|------|------|
| `PlayGame.jar` | **对局引擎 / 判罚**：给定双方 bot 命令行、地图路径，跑出胜负与回放数据（竞赛标准流程）。 |
| `ShowGame.jar` | **可视化回放**：读取对局日志或中间格式，图形展示星球与舰队动向（用法见上游文档或 `java -jar ShowGame.jar -help` 类选项）。 |

与 Lisp Bot 的常见关系：Makefile / `run-bot.sh` / `bin/play-*` 通过 `PlayGame.jar` 调起 Lisp 进程的 stdin/stdout 协议（见当年 starter pack 说明）。
