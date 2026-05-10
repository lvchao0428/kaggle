## v19 第二轮加固方案（针对 seed 0 的精度问题）

### 问题诊断

从你的截图看，Step 61 和 78 都有舰队轨迹贴太阳：

1. **Step 61**：多个小舰队（9, 9, 12）试图从左侧派出，明显方向指向太阳边缘
2. **Step 78**：更大批次（8）从左上派出，同样贴日

这不是"即将冲出"，而是 **角度搜索的粒度不足** — 即使有 44 个候选角度，仍可能在太阳周围有"盲角"（dense 的角度都被太阳卡住）。

### 改动内容

#### 1. 角度搜索密度大幅提升
- **之前**：固定 44 个角度（不均匀，粒度最大 0.5）
- **现在**：生成 0 到 3.2 弧度、粒度 **0.02** 的所有候选（共 ~320 个角度）
  ```python
  for i in range(int(3.2 / 0.02) + 1):
      delta = i * 0.02
      deltas.append(delta)
      if delta > 0:
          deltas.append(-delta)
  ```
  这样能覆盖 0.02 弧度（约 1.1°）的任何盲角

#### 2. Fallback 角落扩大
- 从 `1.5` 扩到 `2.0`，确保角落本身完全在安全区

#### 3. _emit 中的轨迹检查保持

### 预期效果

- **角度精度**：从 ~1 度→ 0.02 度级别，基本覆盖所有可能的避让方向
- **Fallback 更激进**：如果目标确实被太阳卡死，会自动转向最远安全角落
- **零容忍**：任何一步违规都 reject，**宁可不动，不能撞日或越界**

### 是否真的运行最新代码

由于你用的是 `python3.12 scripts/replay.py --a v19 --b v18`，需要确认：

1. **replay.py 是否正确 load v19**（检查 `--a v19` 映射到哪个文件）
2. **是否有本地 cache 的旧版本**（可能被 Python 的 `.pyc` 缓存干扰）

建议做一次 **完全清理再试**：

```bash
# 清除编译缓存
find /Users/lvchao0428/project/kaggle -name "*.pyc" -delete
find /Users/lvchao0428/project/kaggle -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# 重新运行（强制 Python 重新编译）
python3.12 scripts/replay.py --a v19 --b v18 --seed 0 --no-open
```

然后看 Step 61/78 是否还会撞日或飞出屏幕。

---

**如果问题仍存**，说明根源在别处（如 `lead_intercept` 的目标点计算本身就有问题），需要用 `tools/debug_seed0.py` 逐步追踪。
