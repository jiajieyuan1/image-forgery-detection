# 图像伪造检测项目 - 开发指南

## 项目概述
这是一个基于深度学习的图像伪造检测系统，可以检测以下类型的图像问题：
- AI 生成的图像 (DeepFake, DALL-E等)
- 图像篡改/修复 (Inpainting)
- 图像拼接 (Splicing)
- 压缩伪迹

## 项目结构
```
cv项目/
├── data/                  # 数据集存放
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后数据
├── models/               # 训练好的模型
├── src/                  # 源代码
│   ├── data_loader.py   # 数据加载器
│   ├── models.py        # 模型定义
│   ├── preprocessing.py # 数据预处理
│   ├── inference.py     # 推理引擎
│   └── api.py          # Flask API
├── utils/                # 工具函数
│   ├── viz.py           # 可视化
│   └── metrics.py       # 评估指标
├── notebooks/            # 数据分析笔记本
├── app.py               # Streamlit应用
├── train.py             # 训练脚本
└── config.py            # 配置文件
```

## 快速开始

1. 环境设置：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行训练：
   ```bash
   python train.py --config config.py --epochs 100
   ```

3. 运行推理API：
   ```bash
   python src/api.py
   ```

4. 启动Web UI：
   ```bash
   streamlit run app.py
   ```

## 主要功能
- ✅ 多种检测模型 (ResNet, EfficientNet, Vision Transformer)
- ✅ 数据预处理和增强
- ✅ 模型训练和评估
- ✅ 推理API (Flask)
- ✅ Web UI (Streamlit)
- ✅ 实时检测和可视化
