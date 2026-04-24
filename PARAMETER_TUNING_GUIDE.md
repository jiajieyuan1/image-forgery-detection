# v4算法参数调整指南

## 快速参数位置

文件：`src/inference.py`
方法：`infer_single_image()`

搜索这些关键行：
```python
# 第一阶段：高置信度判断
if fake_prob >= 0.75:

# 第二阶段：中等置信度判断
elif fake_prob >= 0.55 and prob_diff >= 0.15:
```

## 参数调整表

### 推荐的调整方案

根据**测试结果不理想的方向**选择调整：

#### 方案A：AI漏报太多（AI被判为真实）

**当前问题**：
```
真实图片4张判对3-4张 ✓
AI图片4张判对0-1张 ❌ (漏报严重)
```

**调整步骤**：

1. **第一步** - 降低中等阈值
```python
# 改这行
elif fake_prob >= 0.55 and prob_diff >= 0.15:  # 旧
elif fake_prob >= 0.50 and prob_diff >= 0.15:  # 新（降低5%）
```

2. **第二步** - 降低差距要求
```python
# 改这行
elif fake_prob >= 0.55 and prob_diff >= 0.15:  # 旧
elif fake_prob >= 0.55 and prob_diff >= 0.10:  # 新（降低5%）
```

3. **第三步** - 降低高置信度阈值（最极端）
```python
# 改这行
if fake_prob >= 0.75:    # 旧
if fake_prob >= 0.70:    # 新（激进一点）
```

**调整表**：
| 尝试 | 改动 | 期望变化 |
|-----|-----|--------|
| 1次 | 中等阈值: 0.55→0.50 | AI检测↑ 真实不变 |
| 2次 | 差距: 0.15→0.10 | AI检测↑ 误报↑ |
| 3次 | 高阈值: 0.75→0.70 | AI检测↑↑ 误报↑↑ |

#### 方案B：真实误报太多（真实被判为AI）

**当前问题**：
```
AI图片4张判对3-4张 ✓
真实图片4张判对0-1张 ❌ (误报严重)
```

**调整步骤**：

1. **第一步** - 提高中等阈值
```python
# 改这行
elif fake_prob >= 0.55 and prob_diff >= 0.15:  # 旧
elif fake_prob >= 0.60 and prob_diff >= 0.15:  # 新（提高5%）
```

2. **第二步** - 提高差距要求
```python
# 改这行
elif fake_prob >= 0.55 and prob_diff >= 0.15:  # 旧
elif fake_prob >= 0.55 and prob_diff >= 0.20:  # 新（提高5%）
```

3. **第三步** - 提高高置信度阈值（最保守）
```python
# 改这行
if fake_prob >= 0.75:    # 旧
if fake_prob >= 0.80:    # 新（保守一点）
```

**调整表**：
| 尝试 | 改动 | 期望变化 |
|-----|-----|--------|
| 1次 | 中等阈值: 0.55→0.60 | 真实保护↑ AI检测↓ |
| 2次 | 差距: 0.15→0.20 | 真实保护↑↑ AI检测↓ |
| 3次 | 高阈值: 0.75→0.80 | 真实保护↑↑↑ 严格许多 |

#### 方案C：两边都有误判

**当前问题**：
```
AI图片4张判对2张 (50%)
真实图片4张判对2张 (50%)
```

**调整步骤**：
这种情况意味着模型本身可能不足以分辨。但可以尝试：

```python
# 变更第一套值
if fake_prob >= 0.72:    # 从0.75改为0.72
elif real_prob >= 0.72:

# 变更第二套值
elif fake_prob >= 0.52 and prob_diff >= 0.14:  # 从0.55和0.15改
elif real_prob >= 0.52 and prob_diff >= 0.14:
```

这样会稍微平衡一点，但预期还是50-50。

## 完整代码片段参考

### 当前v4的完整代码（第一阶段到第三阶段）

```python
# 第一阶段：高置信度判断（保留原算法的高置信部分）
if fake_prob >= 0.75:
    # 极高置信度AI
    final_class = 1
    display_score = fake_prob
    detection_level = "Definitely AI"
    reason = "Very high fake probability (>75%)"
    confidence_level = "High"
    
elif real_prob >= 0.75:
    # 极高置信度真实
    final_class = 0
    display_score = real_prob
    detection_level = "Definitely Real"
    reason = "Very high real probability (>75%)"
    confidence_level = "High"

# 第二阶段：中等置信度判断 - 使用概率差距
elif fake_prob >= 0.55 and prob_diff >= 0.15:  # ← 这两个参数可改
    # 中度AI信号 + 足够的差距
    final_class = 1
    display_score = fake_prob
    detection_level = "Likely AI"
    reason = f"Moderate AI signal (55-75%, diff={prob_diff:.1%})"
    confidence_level = "Medium"
    
elif real_prob >= 0.55 and prob_diff >= 0.15:  # ← 这两个参数可改
    # 中度真实信号 + 足够的差距
    final_class = 0
    display_score = real_prob
    detection_level = "Likely Real"
    reason = f"Moderate real signal (55-75%, diff={prob_diff:.1%})"
    confidence_level = "Medium"

# 第三阶段：低置信度判断 - 非常conservative
elif fake_prob >= 0.55:
    # 略高于50%的AI信号，但不够clear
    final_class = 1
    display_score = fake_prob
    detection_level = "Possibly AI"
    reason = f"Weak AI signal (51-55%, uncertain)"
    confidence_level = "Low"
    
elif real_prob >= 0.55:
    # 略高于50%的真实信号，但不够clear
    final_class = 0
    display_score = real_prob
    detection_level = "Possibly Real"
    reason = f"Weak real signal (51-55%, uncertain)"
    confidence_level = "Low"

else:
    # 完全无法分辨（50/50 或接近）- 默认判为真实（保守）
    final_class = 0
    display_score = real_prob
    detection_level = "Uncertain"
    reason = "Model cannot distinguish (near 50/50)"
    confidence_level = "Very Low"
```

## 调参流程建议

### 第1次测试（基线，不改参数）
```bash
python test_improved_detection.py --dir ai图片
```
→ 记录准确率和误判方向

### 第2次测试（一个参数）
- 根据误判方向选择方案
- **只改一个参数**（中等阈值 OR 差距要求）

```bash
python test_improved_detection.py --dir ai图片
```
→ 观察变化，如果改善继续，否则还原

### 第3次测试（累积改动）
如果第2次有改善，可以再调一个参数

### 第4+次测试（微调）
根据反馈继续微调

## 需要帮助时

告诉我您当前的情况：
1. **AI图片**准确率是多少%
2. **真实图片**准确率是多少%
3. **具体是什么情况**：
   - "AI都被判为真实"
   - "真实都被判为AI"
   - "都是50/50"

然后我可以给出具体的参数建议！
