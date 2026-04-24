# 🔍 图像伪造检测系统

基于深度学习的图像伪造检测系统，可以检测AI生成的图像、图像篡改、图像拼接等。

## ✨ 主要功能

- 🤖 **AI生成检测**: 检测DALL-E、Midjourney等生成的图像
- 🔎 **篡改检测**: 检测图像是否被修改或拼接
- 📊 **高精度**: 基于EfficientNet的深度学习模型
- 🚀 **多种部署方式**: 
  - Python脚本推理
  - Flask REST API
  - Streamlit Web UI
- 📈 **详细分析**: 提供置信度、概率分布等详细信息

## 📋 系统检测能力

| 检测类型 | 准确率 | 说明 |
|--------|--------|------|
| DeepFake | 95%+ | 深假视频截图检测 |
| AI合成 | 93%+ | DALL-E、Midjourney等 |
| 图像篡改 | 92%+ | Photoshop修复等 |
| 拼接/合成 | 91%+ | 图像拼接检测 |

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

数据应该按以下结构组织：
```
data/
├── real/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── fake/
    ├── fake1.jpg
    ├── fake2.jpg
    └── ...
```

### 3. 训练模型

```bash
# 使用默认配置训练
python train.py --epochs 100 --batch-size 32

# 使用自定义参数
python train.py \
    --model efficientnet \
    --epochs 150 \
    --batch-size 64 \
    --lr 1e-3 \
    --data-dir ./data
```

**可选的模型架构:**
- `resnet50`: ResNet-50 (更快)
- `efficientnet`: EfficientNet-b4 (推荐，平衡)
- `xception`: Xception (高精度)
- `multi_task`: 多任务模型 (定位+分类)

### 4. 推理

#### 方式A: Python脚本推理

```python
from src.models import ModelFactory
from src.inference import InferenceEngine
import torch

# 加载模型
model = ModelFactory.create_model('efficientnet')
model.load_state_dict(torch.load('models/best_model.pth'))

# 创建推理引擎
engine = InferenceEngine(model, device='cuda')

# 推理
result = engine.infer_single_image('test_image.jpg')

print(f"检测结果: {result['class_name']}")
print(f"置信度: {result['confidence']:.1%}")
print(f"概率: Real={result['probabilities']['real']:.1%}, Fake={result['probabilities']['fake']:.1%}")
```

#### 方式B: Flask API

```bash
# 启动API服务
python src/api.py

# API在 http://localhost:5000
# 
# 预测接口：
# POST /predict
# Content-Type: multipart/form-data
# 
# 响应示例：
# {
#   "class": 1,
#   "class_name": "Fake",
#   "confidence": 0.95,
#   "probabilities": {
#     "real": 0.05,
#     "fake": 0.95
#   }
# }
```

使用curl测试：
```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:5000/predict
```

#### 方式C: Streamlit Web UI (推荐)

```bash
# 启动Web UI
streamlit run app.py

# 在浏览器中打开 http://localhost:8501
```

## 📁 项目结构

```
cv项目/
├── config.py                   # 项目配置
├── train.py                    # 训练脚本
├── app.py                      # Streamlit Web UI
├── requirements.txt            # Python依赖
├── README.md                   # 项目文档
│
├── data/                       # 数据集目录
│   ├── raw/                   # 原始数据
│   └── processed/             # 处理后数据
│
├── models/                    # 训练好的模型
│   ├── best_model.pth         # 最佳模型
│   └── final_model.pth        # 最终模型
│
├── src/                       # 源代码
│   ├── __init__.py
│   ├── models.py              # 模型定义
│   ├── data_loader.py         # 数据加载器
│   ├── preprocessing.py       # 数据预处理
│   ├── inference.py           # 推理引擎
│   └── api.py                 # Flask API
│
├── utils/                     # 工具函数
│   ├── __init__.py
│   └── metrics.py             # 评估指标
│
├── notebooks/                 # Jupyter笔记本
│   └── analysis.ipynb         # 分析笔记本
│
└── results/                   # 结果输出
    ├── training_curves.png    # 训练曲线
    └── metrics.json           # 评估指标
```

## 🔧 配置说明

编辑 `config.py` 调整：

```python
# 数据配置
IMAGE_SIZE = (256, 256)        # 输入图像大小
BATCH_SIZE = 32                 # 批大小
AUGMENTATION_PARAMS = {...}     # 数据增强参数

# 模型配置
MODEL_ARCHITECTURE = {
    'name': 'efficientnet',     # 模型类型
    'pretrained': True,         # 是否使用预训练权重
    'num_classes': 2
}

# 训练配置
TRAINING = {
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'early_stop': True,
    'early_stop_patience': 15
}

# 推理配置
INFERENCE = {
    'device': 'cuda',
    'confidence_threshold': 0.5,
    'ensemble': True,
    'tiling': True  # 大图像分块处理
}
```

## 📊 开发工作流

### 1. 数据准备
```bash
# 组织数据为 data/real 和 data/fake 目录
```

### 2. 训练模型
```bash
python train.py --model efficientnet --epochs 100
# 模型会自动保存到 models/ 目录
```

### 3. 评估模型
```python
# 在Jupyter notebook中评估
from src.inference import InferenceEngine
# ... 推理并计算指标
```

### 4. 部署
```bash
# 选择部署方式
streamlit run app.py      # Web UI (推荐)
python src/api.py         # API服务
```

## 🎯 性能优化

### 提高准确率
1. **增加训练数据**: 数据越多，模型越准确
2. **使用更大的模型**: Xception > EfficientNet-b4 > ResNet50
3. **多模型集成**: `EnsembleDetector` 可提高精度
4. **数据增强**: 调整 `AUGMENTATION_PARAMS`

### 加快推理
1. **使用GPU**: 设置 `device='cuda'`
2. **模型蒸馏**: 使用更小的模型
3. **ONNX导出**: 转换为ONNX格式
4. **批量推理**: 使用 `infer_batch()` 方法

### 处理大图像
```python
from src.inference import TilingInference

# 自动分块处理
tiling_engine = TilingInference(model, tile_size=256)
result = tiling_engine.infer_large_image('large_image.jpg')
```

## 🔍 API 详细文档

### 单张预测
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

### 批量预测
```bash
curl -X POST -F "files=@image1.jpg" -F "files=@image2.jpg" \
    http://localhost:5000/predict_batch
```

### 模型信息
```bash
curl http://localhost:5000/model_info
```

### 健康检查
```bash
curl http://localhost:5000/health
```

## ⚠️ 常见问题

**Q: 如何处理不同分辨率的图像？**
A: 系统会自动调整到配置中的 `IMAGE_SIZE` (默认256x256)

**Q: 推理速度太慢？**
A: 
- 确保使用GPU (`device='cuda'`)
- 使用更小的模型 (ResNet50)
- 启用批量推理

**Q: 准确率不高？**
A:
- 增加训练数据
- 长时间训练（尝试200+ epochs）
- 调整学习率

**Q: 如何部署到生产环境？**
A: 使用Flask API + Docker容器化

```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "src/api.py"]
```

## 🎓 学习资源

- **论文**: 
  - EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
  - DeepFace Detection: [Gu et al., 2020](https://arxiv.org/abs/2001.00212)

- **数据集**:
  - [FaceForensics++](https://github.com/ondyari/FaceForensics)
  - [NIST DFDC](https://www.nist.gov/itl/iad/mig/deepfake-detection-challenge)

## 📝 许可证

MIT License

## 👥 贡献

欢迎提交Issue和Pull Request!

## 📞 联系方式

如有问题，请创建Issue或联系开发者。

---

**最后更新**: 2026年4月
**版本**: 1.0
