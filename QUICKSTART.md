# 快速使用指南

## 📚 项目概述

这是一个基于深度学习的图像伪造检测系统，支持检测：
- 🤖 AI生成的图像 (DeepFake, DALL-E等)
- 🖌️ 图像篡改/修复
- 📐 图像拼接
- 📊 压缩伪迹

## 🚀 5分钟快速开始

### 1️⃣ 环境准备 (第一次使用时)

```bash
# 已经完成！依赖已安装
# 演示数据集已生成在 data/raw/
```

### 2️⃣ 运行系统

选择下面任一方式：

#### 方式A：Web UI (推荐) ⭐

```bash
streamlit run app.py
```

然后在浏览器打开 `http://localhost:8501`

✨ **功能:**
- 上传或拖拽图像
- 实时检测结果
- 批量处理
- 详细分析

#### 方式B：命令行快速推理

```bash
python quick_infer.py path/to/image.jpg
```

#### 方式C：Python脚本

```python
from src.models import ModelFactory
from src.inference import InferenceEngine
import torch

# 加载模型
model = ModelFactory.create_model('efficientnet')
engine = InferenceEngine(model)

# 推理
result = engine.infer_single_image('test.jpg')
print(f"结果: {result['class_name']} (置信度: {result['confidence']:.1%})")
```

#### 方式D：REST API

```bash
# 启动服务
python src/api.py

# 在另一个终端调用
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

## 📊 演示与测试

### 运行完整演示

```bash
python demo.py
```

这会展示系统的所有功能，包括：
- 模型加载
- 数据预处理
- 单张推理
- 批量处理
- 配置信息

### 查看演示数据

```bash
# 已生成的演示数据位置
data/raw/real/    # 30张真实图像
data/raw/fake/    # 30张伪造图像
```

## 🎓 完整工作流

### 📈 完整的训练管道

```bash
# 1. 准备你的数据
# 确保目录结构为:
# data/real/  (真实图像)
# data/fake/  (伪造图像)

# 2. 训练模型
python train.py \
    --model efficientnet \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --data-dir ./data

# 3. 启动Web UI
streamlit run app.py

# 4. 上传图像进行检测
```

### 🔧 自定义配置

编辑 `config.py` 调整参数：

```python
# 数据配置
IMAGE_SIZE = (256, 256)         # 输入大小
BATCH_SIZE = 32                  # 批大小

# 模型配置
MODEL_ARCHITECTURE = {
    'name': 'efficientnet',      # resnet50, efficientnet, xception
    'pretrained': True
}

# 训练配置
TRAINING = {
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'optimizer': 'adamw'          # adam, adamw, sgd
}
```

## 🎯 常见任务

### ✅ 检测单张图像

```bash
python quick_infer.py my_image.jpg
```

### ✅ 批量检测图像

```python
from src.inference import InferenceEngine

results = engine.infer_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
for r in results:
    print(f"{r['image_path']}: {r['class_name']}")
```

### ✅ 处理大图像

```python
from src.inference import TilingInference

tiling = TilingInference(model, tile_size=256)
result = tiling.infer_large_image('very_large_image.jpg')
```

### ✅ 模型集成

```python
from src.models import ModelFactory

# 创建多个模型集成
ensemble = ModelFactory.create_ensemble(
    ['resnet50', 'efficientnet', 'xception']
)
```

## 📋 文件说明

| 文件 | 说明 |
|-----|------|
| `train.py` | 模型训练脚本 |
| `app.py` | Streamlit Web UI |
| `src/api.py` | Flask REST API |
| `quick_infer.py` | 快速推理脚本 |
| `demo.py` | 完整演示脚本 |
| `config.py` | 项目配置文件 |
| `create_demo_data.py` | 生成演示数据 |

## 🐛 故障排除

### 问题1: 模型文件不存在

```
错误: 模型文件不存在于 models/
```

**解决:** 训练模型
```bash
python train.py --epochs 100
```

### 问题2: CUDA内存不足

```
RuntimeError: CUDA out of memory
```

**解决:**
```bash
# 减小批大小
python train.py --batch-size 16

# 或使用CPU
# 编辑 config.py，设置 device='cpu'
```

### 问题3: 图像无法加载

```
ValueError: 无法加载图像
```

**解决:**
- 检查文件格式（JPG, PNG, BMP, WebP）
- 检查文件路径
- 检查文件是否损坏

## 💡 性能优化建议

### 🚀 加快推理速度

```python
# 1. 使用GPU
engine = InferenceEngine(model, device='cuda')

# 2. 批量推理（更高效）
results = engine.infer_batch(image_list)

# 3. 小模型（ResNet50 比 Xception 快）
model = ModelFactory.create_model('resnet50')
```

### 📈 提高准确率

```python
# 1. 使用模型集成
ensemble = ModelFactory.create_ensemble(['resnet50', 'efficientnet'])

# 2. 增加训练数据
# 3. 更长的训练时间
python train.py --epochs 200

# 4. 使用更大的模型
model = ModelFactory.create_model('xception')
```

## 📞 获取帮助

1. 查看 [README.md](README.md) 了解详细文档
2. 运行 `python demo.py` 查看示例
3. 检查 `config.py` 中的配置选项

## ✨ 项目亮点

- ✅ **开箱即用**: 包含演示数据和预配置
- ✅ **多种部署**: Web UI、API、命令行
- ✅ **高精度**: 基于EfficientNet的深度学习模型
- ✅ **易于扩展**: 模块化设计，便于定制
- ✅ **完整文档**: 详细的使用指南和代码注释

---

**开始使用:**
```bash
streamlit run app.py
```

🎉 开始检测伪造图像吧！
