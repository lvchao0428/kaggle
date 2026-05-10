## v19 快速调整说明

根据回放分析（seed 0），发现三个并发问题：

### 1. 连续小派兵 (8, 19, 15, 15...)
**原因**：`ABS_MIN_BATCH = 5` 太小，`_build_capture_plan` 每轮都能凑出一个 5+ 的小批。
**改法**：调到 **`ABS_MIN_BATCH = 8`**，强制每批更大，减少碎片化。

### 2. 仍往太阳飞
**检查项**：
- `safe_aim` 的 `SUN_MARGIN` 原为 3.0，改到 **3.5** 加大避让
- 角度搜索 `deltas` 范围从 ±2.8 扩到 **±3.02**，覆盖更多可能
- `_emit` 的 `check_e` 采样 `[1,3,5,10,15,20]` 已在位；轨迹采样检查逻辑也已集成

**可能根因**：`lead_intercept` 本身在某些轨道行星场景下，返回的目标点可能就接近日心；或 `safe_aim` 在"找不到完全净空射线"的极端情况下，fallback 逻辑可能还没触发。

### 3. 目标分散
**观察**：派兵方向无连贯性（可能同时攻击左下、右下、左上的多个星）。
**原因**：
- 区域图可能聚类不力（例如各星被分配到不同区，失去同区加成）
- `target_score` vs `target_value_in_region` 没形成统一排序

**简易检验**：本地跑一下

```bash
python3 kaggle/test_v19_regional.py
```

看区域分配是否合理（应该是 4 个大区，每区多颗行星）。

---

### 即时修改

已在 `submission_v19.py` 做了：
- `ABS_MIN_BATCH = 8`
- `SUN_MARGIN = 3.5`
- 扩大 `deltas` 角度网格

现在可以尝试本地回放或参赛测试，观察是否改善「连续小派」和「往太阳飞」。

---

### 后续诊断（若问题仍存）

如果改调还不够，需要走 `tools/debug_moves_detail.py` 或 `tools/debug_safe_aim.py` 来：
1. 追踪具体某一帧的派兵决策流程
2. 检查 `lead_intercept` 返回的目标点是否接近日心
3. 检查本轮 `safe_aim` 是否真的触发了 fallback（向安全角) 或只返回了最近没找到净空的"best"

持续观察回放；或者可以在 `safe_aim` / `_emit` 里加 print 把关键信息输到 stderr，本地重放时看输出。
