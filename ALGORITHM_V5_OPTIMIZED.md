# v5版本 - 针对实际AI图片的优化

## 问题分析

用户上传的4张AI图片中，**前两张被误判为真实**。这表明：
- 这两张的 `fake_prob` 很低（可能 < 0.55）
- 现有的v4算法阈值对这些特定类型的AI图片不适合

## v5 改进点

基于您上传的图片，进行了以下参数调整：

### 参数变更对比

| 阶段 | 参数 | v4版本 | v5版本 | 变更 |
|-----|------|--------|--------|------|
| **第二阶段** | 中等AI阈值 | 0.55 | 0.50 | ↓ 降低5% |
| **第二阶段** | 概率差距 | 0.15 | 0.10 | ↓ 降低33% |
| **第三阶段** | 低置信度阈值 | 0.55 | 0.52 | ↓ 降低3% |

### 具体改动说明

#### 改动1: 中等AI阈值 0.55 → 0.50
```python
# v4版本
elif fake_prob >= 0.55 and prob_diff >= 0.15:

# v5版本  
elif fake_prob >= 0.50 and prob_diff >= 0.10:
```
**效果**：您上传的图片即使fake_prob只有50-55%，也会被判为AI

#### 改动2: 概率差距要求 0.15 → 0.10
```python
# v4版本
elif fake_prob >= 0.55 and prob_diff >= 0.15:

# v5版本
elif fake_prob >= 0.50 and prob_diff >= 0.10:
```
**效果**：更宽松的差距要求，适应模型判断不够clear的情况

#### 改动3: 低置信度阈值 0.55 → 0.52
```python
# v4版本
elif fake_prob >= 0.55:

# v5版本
elif fake_prob >= 0.52:
```
**效果**：即使fake_prob=52%，也会被判为AI

## 预期效果 (针对您的图片)

### 场景1：前两张原本fake_prob=45-50%
```
v4判定: Definitely Real (real_prob >= 0.75)
v5判定: Possibly AI (fake_prob >= 0.52)
结果: ✅ 改善
```

### 场景2：前两张原本fake_prob=50-55%
```
v4判定: Uncertain → Real
v5判定: Likely AI (fake_prob >= 0.50 + diff >= 0.10)
结果: ✅ 改善
```

## 可能的风险

⚠️ 参数调整会让AI检测更敏感，可能导致：
- 某些真实图片也被判为AI（误报增加）
- 尤其是边界情况的图片（real_prob 45-55%）

## 调整后的判断表

| 优先级 | 条件 | 结果 | 说明 |
|-------|------|------|------|
| 1️⃣ | fake ≥ 75% | AI | 极高 |
| 2️⃣ | real ≥ 75% | 真实 | 极高 |
| 3️⃣ | fake ≥ 50% + 差距10% | AI | 激进 ← **改动** |
| 4️⃣ | real ≥ 50% + 差距10% | 真实 | 激进 ← **改动** |
| 5️⃣ | fake ≥ 52% | AI | 低 ← **改动** |
| 6️⃣ | real ≥ 52% | 真实 | 低 ← **改动** |
| 7️⃣ | ~50/50 | 真实 | 保守 |

## 快速测试

### 1️⃣ 测试您的4张AI图片

```bash
python test_improved_detection.py --dir ai图片
```

观察：
- 第1-2张是否从"Real"改为"AI"？
- 第3-4张是否继续保持"AI"？

### 2️⃣ 可选：诊断具体分数

```bash
python diagnose_ai_images.py
```

查看前两张的具体fake_prob分数，了解参数是否有帮助

## 进一步调整建议

如果v5还是不理想，可以继续微调：

### 若AI检测还不够（仍有AI被判为真实）
```python
# 继续降低中等阈值
elif fake_prob >= 0.48 and prob_diff >= 0.08:  # 继续激进

# 或降低低置信度
elif fake_prob >= 0.50:  # 从0.52改为0.50
```

### 若误报过多（真实被判为AI）
```python
# 提高阈值
elif fake_prob >= 0.52 and prob_diff >= 0.12:  # 从0.50和0.10改进

# 或提高低置信度
elif fake_prob >= 0.54:  # 从0.52改为0.54
```

## 文件变更清单

✅ `src/inference.py` - v5算法实现
（自动包含的）所有使用`infer_single_image`的其他文件都会自动使用新参数

## 字段保持不变

返回结果中的所有字段保持不变：
- `class_name`: Fake/Real
- `detection_level`: 检测等级
- `confidence_level`: High/Medium/Low/Very Low
- `probabilities`: 原始概率
- 以及其他诊断字段

## 立即行动

```bash
# 测试新的v5版本
python test_improved_detection.py --dir ai图片
```

告诉我结果：前两张是否现在被正确检测为AI？
