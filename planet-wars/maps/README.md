# `planet-wars/maps/`

收录 **Google AI Challenge 2010 Planet Wars** 时代的官方/决赛用 **地图文本**（大量 `map_finals-*.txt`）。

## 文件性质

- 同一格式的 **平面图描述**：ASCII 文本，_engine / PlayGame.jar_ 可读。
- 命名如 `map_finals-1_01.txt`、`map_finals-2_69.txt` 表示不同决赛阶段与编号；**内容结构相同**，仅星球坐标/出兵/归属等不同。

## 行格式示例（抽样 [`map_finals-1_01.txt`](map_finals-1_01.txt)）

- `P x y owner ships growth`：一颗行星——坐标、所有者（0 中立 / 1、2 玩家）、当前兵力、每秒增长（产能）。
- 其他首字母行类型以当年 **Engine 文档**为准（本项目以 `tools/PlayGame.jar` 所用格式为真）。

## 不需要逐文件写 README 的原因

文件数量多且 **schema 同质**；开发与阅读时任选 1～2 张对照即可。若需要完整列表：`ls *.txt | wc -l`。

## 相关脚本

[`../bin/on-all-maps.sh`](../bin/on-all-maps.sh) 可对整目录批量评测。
