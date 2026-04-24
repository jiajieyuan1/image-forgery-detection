"""
完整演示脚本
展示项目的所有功能
"""

import sys
from pathlib import Path
import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ModelFactory
from src.data_loader import DatasetBuilder
from src.preprocessing import DataAugmentation
from src.inference import InferenceEngine
from config import *


def demo_01_basic_setup():
    """演示1: 基本设置和模型创建"""
    print("\n" + "="*60)
    print("演示 1: 基本设置和模型创建")
    print("="*60)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"运行设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 创建模型
    print("\n可用的模型:")
    models = ['resnet50', 'efficientnet', 'xception']
    for i, name in enumerate(models, 1):
        print(f"  {i}. {name}")
    
    # 创建EfficientNet模型
    print("\n创建 EfficientNet 模型...")
    model = ModelFactory.create_model('efficientnet', num_classes=2)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ 模型参数总数: {total_params:,}")
    print(f"✓ 可训练参数: {trainable_params:,}")
    
    return model, device


def demo_02_data_loading():
    """演示2: 数据加载"""
    print("\n" + "="*60)
    print("演示 2: 数据加载和预处理")
    print("="*60)
    
    print(f"\n数据目录: {DATA_DIR}")
    
    # 检查数据集结构
    real_dir = DATA_DIR / 'raw' / 'real'
    fake_dir = DATA_DIR / 'raw' / 'fake'
    
    if real_dir.exists() and fake_dir.exists():
        real_count = len(list(real_dir.glob('*.jpg')))
        fake_count = len(list(fake_dir.glob('*.jpg')))
        
        print(f"✓ 真实图像: {real_count} 张")
        print(f"✓ 伪造图像: {fake_count} 张")
        print(f"✓ 总图像数: {real_count + fake_count} 张")
    else:
        print("✗ 数据集目录不存在，请先运行: python create_demo_data.py")
        return None
    
    # 展示数据增强
    print("\n数据增强配置:")
    for key, value in AUGMENTATION_PARAMS.items():
        print(f"  {key}: {value}")
    
    return real_count, fake_count


def demo_03_inference():
    """演示3: 推理"""
    print("\n" + "="*60)
    print("演示 3: 单张图像推理")
    print("="*60)
    
    # 创建临时测试图像
    from PIL import Image
    
    # 创建简单的测试图像
    test_img = np.random.randn(256, 256, 3) * 50 + 128
    test_img = np.clip(test_img, 0, 255).astype(np.uint8)
    
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(test_img)
    
    temp_path = Path('/tmp') / 'test_inference.jpg'
    pil_img.save(str(temp_path))
    
    print(f"\n创建测试图像: {temp_path}")
    
    # 创建模型和推理引擎
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelFactory.create_model('efficientnet', num_classes=2, pretrained=True)
    
    # 模型移至设备
    model = model.to(device)
    
    print(f"模型已移至: {device}")
    
    # 创建推理引擎
    engine = InferenceEngine(model, device=device, confidence_threshold=0.5)
    
    print("✓ 推理引擎已创建")
    
    # 执行推理
    print("\n执行推理...")
    result = engine.infer_single_image(str(temp_path))
    
    print("\n推理结果:")
    print(f"  分类: {result['class_name']}")
    print(f"  置信度: {result['confidence']:.1%}")
    print(f"  真实概率: {result['probabilities']['real']:.1%}")
    print(f"  伪造概率: {result['probabilities']['fake']:.1%}")
    
    return result


def demo_04_batch_processing():
    """演示4: 批量处理"""
    print("\n" + "="*60)
    print("演示 4: 批量图像处理")
    print("="*60)
    
    # 创建多个测试图像
    print("创建 5 张测试图像...")
    
    temp_paths = []
    for i in range(5):
        if i < 3:
            # 真实类型
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            for x in range(256):
                img[:, x] = [x, 128, 255-x]
        else:
            # 伪造类型
            img = np.random.randn(256, 256, 3) * 50 + 128
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        from PIL import Image
        pil_img = Image.fromarray(img)
        temp_path = Path('/tmp') / f'batch_test_{i}.jpg'
        pil_img.save(str(temp_path))
        temp_paths.append(temp_path)
    
    print(f"✓ 创建 {len(temp_paths)} 张测试图像")
    
    # 批处理
    print("\n执行批处理...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelFactory.create_model('efficientnet', num_classes=2, pretrained=True)
    model = model.to(device)
    
    engine = InferenceEngine(model, device=device)
    results = engine.infer_batch(temp_paths)
    
    print("\n批处理结果:")
    fake_count = sum(1 for r in results if r['is_fake'])
    real_count = len(results) - fake_count
    
    print(f"  真实图像: {real_count} 张")
    print(f"  伪造图像: {fake_count} 张")
    print(f"  伪造率: {fake_count/len(results):.1%}")
    
    # 显示每个结果
    print("\n详细结果:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['class_name']:5s} (置信度: {result['confidence']:.1%})")


def demo_05_configuration():
    """演示5: 配置信息"""
    print("\n" + "="*60)
    print("演示 5: 项目配置")
    print("="*60)
    
    print("\n数据配置:")
    print(f"  图像大小: {IMAGE_SIZE}")
    print(f"  批大小: {BATCH_SIZE}")
    print(f"  数据集划分: train={DATASET_SPLIT['train']}, val={DATASET_SPLIT['val']}, test={DATASET_SPLIT['test']}")
    
    print("\n模型配置:")
    print(f"  模型类型: {MODEL_ARCHITECTURE['name']}")
    print(f"  预训练: {MODEL_ARCHITECTURE['pretrained']}")
    print(f"  类别数: {MODEL_ARCHITECTURE['num_classes']}")
    
    print("\n训练配置:")
    print(f"  轮数: {TRAINING['num_epochs']}")
    print(f"  学习率: {TRAINING['learning_rate']}")
    print(f"  优化器: {TRAINING['optimizer']}")
    print(f"  早停: {TRAINING['early_stop']}")
    
    print("\n推理配置:")
    print(f"  设备: {INFERENCE['device']}")
    print(f"  置信度阈值: {INFERENCE['confidence_threshold']}")
    print(f"  集成: {INFERENCE['ensemble']}")
    
    print("\n路径配置:")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  模型目录: {MODELS_DIR}")
    print(f"  结果目录: {RESULTS_DIR}")


def demo_06_model_comparison():
    """演示6: 模型对比"""
    print("\n" + "="*60)
    print("演示 6: 模型架构对比")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_info = {
        'resnet50': '标准ResNet-50，快速推理',
        'efficientnet': 'EfficientNet-b4，速度和精度平衡',
        'xception': 'Xception，高精度'
    }
    
    print("\n可用模型:")
    for name, description in models_info.items():
        try:
            model = ModelFactory.create_model(name, num_classes=2, pretrained=True)
            model = model.to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            
            print(f"\n{name}:")
            print(f"  描述: {description}")
            print(f"  参数数: {total_params:,}")
            
        except Exception as e:
            print(f"\n{name}: (错误: {e})")


def main():
    """运行所有演示"""
    print("\n" + "="*60)
    print("图像伪造检测系统 - 完整演示")
    print("="*60)
    
    # 演示1: 基本设置
    model, device = demo_01_basic_setup()
    
    # 演示2: 数据加载
    demo_02_data_loading()
    
    # 演示3: 推理
    demo_03_inference()
    
    # 演示4: 批量处理
    demo_04_batch_processing()
    
    # 演示5: 配置信息
    demo_05_configuration()
    
    # 演示6: 模型对比
    demo_06_model_comparison()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    
    print("""
下一步:
1. 训练模型: python train.py --epochs 100
2. 启动Web UI: streamlit run app.py
3. 启动API: python src/api.py
4. 快速推理: python quick_infer.py <image_path>

详细文档请参考 README.md
    """)


if __name__ == '__main__':
    main()
