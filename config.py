"""
图像伪造检测项目配置文件
"""

import os
from pathlib import Path

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# 确保目录存在
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== 数据配置 ====================
# 数据集相关
DATASET_SPLIT = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

# 图像处理
IMAGE_SIZE = (256, 256)  # 输入图像大小
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True

# 数据增强
AUGMENTATION_PARAMS = {
    'horizontal_flip': 0.5,
    'vertical_flip': 0.2,
    'rotation': 30,
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
    'noise_std': 0.02
}

# ==================== 模型配置 ====================
# 选择模型架构
MODEL_ARCHITECTURE = {
    'name': 'efficientnet',  # 可选: resnet50, efficientnet, vit, xception
    'pretrained': True,
    'num_classes': 2,  # Real: 0, Fake: 1
    'dropout_rate': 0.3
}

# 特定模型配置
MODEL_CONFIG = {
    'efficientnet': {
        'version': 'b4',  # b0-b7
        'input_channels': 3,
        'output_features': 1280
    },
    'resnet50': {
        'in_channels': 3,
        'num_blocks': [3, 4, 6, 3],
        'num_classes': 2
    },
    'xception': {
        'num_classes': 2
    },
    'vit': {
        'image_size': 256,
        'patch_size': 16,
        'num_classes': 2,
        'num_heads': 12,
        'num_layers': 12,
        'hidden_dim': 768
    }
}

# ==================== 训练配置 ====================
TRAINING = {
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'optimizer': 'adamw',  # adam, adamw, sgd
    'scheduler': 'cosine',  # cosine, step, exponential
    'warmup_epochs': 5,
    
    # 损失函数
    'loss_function': 'cross_entropy',  # cross_entropy, focal, ohem
    'focal_loss_alpha': 0.25,
    'focal_loss_gamma': 2.0,
    
    # 早停
    'early_stop': True,
    'early_stop_patience': 15,
    'early_stop_min_delta': 1e-4,
    
    # 日志
    'log_interval': 10,
    'save_interval': 5,
    'save_best_only': True
}

# ==================== 增强的检测头部 ====================
DETECTION_HEAD = {
    'attention_module': 'cbam',  # 'cbam', 'se', None
    'num_layers': 2,
    'hidden_dim': 512,
    'dropout': 0.3
}

# ==================== 多任务学习 ====================
MULTI_TASK = {
    'enabled': True,
    'tasks': ['classification', 'localization', 'confidence'],
    'task_weights': {
        'classification': 1.0,
        'localization': 0.3,
        'confidence': 0.2
    }
}

# ==================== 推理配置 ====================
INFERENCE = {
    'device': 'cuda',  # cuda, cpu
    'confidence_threshold': 0.5,
    'ensemble': True,  # 使用多个模型集成
    'ensemble_strategy': 'voting',  # voting, average
    'num_ensembles': 3,
    'tiling': True,  # 大图像分块处理
    'tile_size': 256,
    'tile_overlap': 0.2
}

# ==================== 评估指标 ====================
METRICS = {
    'include_auc': True,
    'include_confusion_matrix': True,
    'include_per_class_metrics': True,
    'calculate_visualization': True
}

# ==================== 实验跟踪 ====================
EXPERIMENT = {
    'use_tensorboard': True,
    'use_wandb': False,  # Weights & Biases
    'save_config': True,
    'save_model_checkpoint': True
}

# ==================== 数据集配置 ====================
DATASETS = {
    'deepfake': {
        'name': 'DeepFaceLab / FaceForensics++',
        'num_classes': 2,
        'supported_formats': ['mp4', 'jpg', 'png']
    },
    'manipulation': {
        'name': 'COCO-Manipulation / SpliceOut',
        'num_classes': 2,
        'supported_formats': ['jpg', 'png', 'tiff']
    },
    'synthetic': {
        'name': 'AI-Generated (DALL-E, Midjourney, etc)',
        'num_classes': 2,
        'supported_formats': ['jpg', 'png', 'webp']
    }
}

# ==================== 后处理 ====================
POSTPROCESSING = {
    'apply_crf': False,  # 条件随机场平滑
    'morphological_ops': True,
    'min_component_size': 50,  # 最小连通分量大小
    'confidence_smoothing': True
}

# ==================== 部署配置 ====================
DEPLOYMENT = {
    'flask_host': '0.0.0.0',
    'flask_port': 5000,
    'streamlit_theme': 'dark',
    'max_upload_size_mb': 20,
    'supported_formats': ['jpg', 'jpeg', 'png', 'webp', 'bmp']
}

# ==================== 调试配置 ====================
DEBUG = {
    'verbose': True,
    'save_intermediate_results': False,
    'visualize_augmentation': False,
    'profile_inference': False
}
