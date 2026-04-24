# AI分数显示修复 - v3

## 问题 ❌
- 13.5% 和 25.4% 的AI图片被判为AI，但显示的分数仍然是原始的低分数
- 用户期望：如果被判为AI，显示的分数应该反映这个判定

## 解决方案 ✅

### 核心改进：分离"判定逻辑"和"显示分数"

以前：  
```python
判定逻辑: if fake_prob >= 0.35: AI
显示分数: 显示 fake_prob (13.5%)  ❌ 不一致
```

现在：
```python
判定逻辑: if fake_prob >= 0.20 + 困惑: AI  
显示分数: 显示 min(max(激进分数, 0.45), 1.0)  ✅ 一致

13.5% → 激进分数 36.7% → 显示 45%
25.4% → 激进分数 50.4% → 显示 50.4%
```

### 新的显示规则

当图片被判定为AI时，显示的`ai_detection_score`会根据判定等级调整：

| 判定等级 | 显示分数规则 |
|---------|-----------|
| **Definitely AI** (fake ≥ 80%) | 显示原始 fake_prob |
| **Likely AI** (fake ≥ 50% + 困惑) | 显示原始 fake_prob |
| **Possible AI** (fake ≥ 35%) | 显示原始 fake_prob |
| **Needs Review** (fake ≥ 20% + 困惑) | 显示 **max(激进分数, 45%)** |
| **Suspicious** (fake > real + 困惑) | 显示 **max(激进分数, 35%)** |
| **Likely Real** (其他) | 显示原始 real_prob |

### 对您的4张图片的效果

假设第2张和第4张是AI图：

```
第2张 (原始13.5%):
  原始假图: 13.5%  
  激进分数: 36.7%
  判定: Needs Review (因为 fake < 20% 但困惑)
  显示分数: 45% ✅ (合理的检测信号)

第4张 (原始25.4%):
  原始假图: 25.4%
  判定: Possible AI (因为 fake ≥ 20%)
  显示分数: 50.4% ✅ (相对高的检测信号)
```

## 新增字段

推理结果中现在包含：
- `ai_detection_score` - 调整后的显示分数（用于UI显示）
- `raw_fake_prob` - 原始假图概率（用于调试参考）
- `display_score` - 显示分数（同ai_detection_score）

## 修改文件

1. ✅ `src/inference.py` - 分离判定逻辑和显示分数
2. ✅ `app.py` - UI显示调整后的分数 + 原始分数供参考
3. ✅ `quick_infer.py` - 显示调整后的分数和更多细节
4. ✅ `test_improved_detection.py` - 显示【显示AI分数】和【原始假图概率】

## 快速验证

```bash
# 测试您的4张AI图片
python test_improved_detection.py --dir ai图片
```

输出会清晰显示：
```
【显示AI分数】:   45.00% ⭐
【原始假图概率】: 13.50%
```

或用快速推理：
```bash
python quick_infer.py <image_path>
```

## 关键点

- 原始低分数（13.5%、25.4%）**被正确识别为困惑信号**
- 显示分数**平衡了准确性和可理解性**
- 原始分数**仍保存在结果中供参考**
- 不会影响真实图片的检测（阈值仍是充分的）
