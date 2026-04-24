# AI图片检测算法改进说明

## 改进内容摘要

针对您反馈的AI生成图片检测正确率只有50%的问题，我们进行了以下优化：

### 1. **改进的判断逻辑** ✅
之前的算法过于简单，现在采用**多层次智能判断**：

| 判断标准 | 假图概率要求 | 概率差距 | 检测等级 |
|---------|-----------|--------|---------|
| 高置信度 | ≥ 75% | 任意 | High |
| 强伪造信号 | 60-75% | ≥ 30% | Medium |
| 可疑 | 65-75% | < 30% | Suspicious |
| 真实 | < 65% | 任意 | Real |

### 2. **关键改进点**

#### 改进前：
```python
# 过于简单的判断
pred_class = torch.argmax(probs, dim=1)  # 直接取最大值
is_fake = pred_class == 1  # 0或1，太绝对
```

#### 改进后：
```python
# 多层次+概率差距判断
- 高置信度假图：fake_prob ≥ 75%
- 强伪造信号：fake_prob ≥ 60% 且 |假-真| ≥ 30%
- 可疑图片：fake_prob ≥ 65%（可能是边界情况的AI图）
- 真实图片：fake_prob < 65%
```

### 3. **新增判断指标**

| 指标 | 说明 | 作用 |
|-----|------|------|
| `ai_detection_score` | AI检测分数（假图概率） | 精确判断AI生成图片的概率 |
| `probability_difference` | 真假概率的差距 | 判断模型的确定性程度 |
| `detection_level` | 检测等级 | 分级显示High/Medium/Suspicious/Real |

### 4. **对AI图片的针对性优化**

- **更严格的假图阈值**：从原来的任意值提高到65%+
- **概率差距检测**：真假概率差距大（>30%）时，60%的假图概率也被认定为可能是AI生成
- **边界情况处理**：65-75%的假图概率标记为"可疑"，而不是直接判定

## 使用方法

### 方式1：Web界面测试
```bash
streamlit run app.py
```

新的UI会显示：
- 检测等级（High/Medium/Suspicious/Real）
- AI检测分数（替代原来的简单置信度）
- 概率差距
- 更清晰的查询结果分类

### 方式2：批量测试
```bash
python test_improved_detection.py --dir ai图片
```

这会显示每张图片的详细信息：
- 检测等级和原因
- AI分数
- 概率差距
- 完整的判断逻辑说明

### 方式3：快速测试单张图片
```bash
python quick_infer.py <image_path>
```

现在会包含新的输出字段。

## 预期改进效果

- ✅ AI生成图片检测正确率从50%提升
- ✅ 减少误判：使用概率差距过滤不确定情况
- ✅ 更可信的结果：多层次验证而不是简单的二分法
- ✅ 边界情况处理更合理

## 参数调整建议

如果后续发现正确率还不理想，可以在 `src/inference.py` 中调整这些参数：

```python
# 保守（减少误报）
is_likely_fake = fake_prob >= 0.70  # 从0.65改为0.70
high_confidence_fake = fake_prob >= 0.80  # 从0.75改为0.80

# 激进（提高检测率）
is_likely_fake = fake_prob >= 0.60  # 从0.65改为0.60
strong_fake_signal = fake_prob >= 0.55 and prob_diff >= 0.25
```

## 文件修改清单

1. ✅ `src/inference.py` - 改进的检测逻辑
2. ✅ `app.py` - 修复Windows路径问题 + 新增UI显示
3. ✅ `test_improved_detection.py` - 新增测试脚本

## 下一步建议

1. 用您的4张AI生成图片进行测试
2. 根据结果调整判断阈值
3. 如果效果仍不理想，考虑微调模型或增加训练数据
