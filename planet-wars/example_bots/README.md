# `planet-wars/example_bots/`

竞赛 starter pack 自带的 **示例 Java Bot**（源码 + 已编译 `.jar`），用于验证引擎与 Lisp bot 的强度基线。

| 文件 | 说明 |
|------|------|
| `Planet.java` `Fleet.java` | 星球 / 舰队数据结构的 Java 建模。 |
| `PlanetWars.java` | 与引擎交互的框架类（读写回合、下发指令）。 |
| `RandomBot.java` / `.jar` | 随机合法出兵。 |
| `ExpandBot.java` | 朴素扩张示例。 |
| `BullyBot.java` / `.jar` | 侵略型示例逻辑。 |
| `DualBot.java` / `.jar` | 双策略或分阶段示例（见源码）。 |
| `RageBot.java` / `.jar` | 另一种激进脚本。 |
| `ProspectorBot.java` / `.jar` | 资源/探索向示例 Bot。 |

**无 `.java` 仅有 `.jar` 的包**：不向文档中捏造未公开的方法表；若要分析行为请 **反编译** 或以 `PlayGame.jar` 对局观察。
