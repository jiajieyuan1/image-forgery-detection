"""
快速推理脚本
用于快速测试模型
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ModelFactory
from src.inference import InferenceEngine
from config import MODELS_DIR


def quick_infer(image_path):
    """快速推理"""
    # 加载模型
    model_path = MODELS_DIR / 'best_model.pth'
    if not model_path.exists():
        model_path = MODELS_DIR / 'final_model.pth'
    
    if not model_path.exists():
        print(f"错误: 模型文件不存在于 {MODELS_DIR}")
        print("请先运行: python train.py")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建模型并加载权重
    model = ModelFactory.create_model('efficientnet', num_classes=2, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 创建推理引擎
    engine = InferenceEngine(model, device=device)
    
    # 推理
    print(f"处理图像: {image_path}")
    result = engine.infer_single_image(image_path)
    
    # 打印结果
    print("\n" + "="*70)
    print("检测结果 (v4平衡版本)")
    print("="*70)
    print(f"【检测结果】: {result['class_name']}")
    print(f"【检测等级】: {result['detection_level']}")
    print(f"【置信度等级】: {result.get('confidence_level', 'N/A')}")
    print(f"【检测原因】: {result['detection_reason']}")
    print("-"*70)
    print(f"真实概率: {result['probabilities']['real']:.2%}")
    print(f"假图概率: {result['probabilities']['fake']:.2%}")
    print(f"概率差距: {result['probability_difference']:.2%}")
    print(f"假/真比例: {result.get('fake_real_ratio', 'N/A'):.4f}")
    print(f"概率熵: {result.get('entropy', 'N/A'):.4f} (> 0.69 = 不确定)")
    print(f"是伪造: {result['is_fake']}")
    print(f"高置信度: {'✓ 是' if result['is_confident'] else '✗ 否'}")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='快速推理')
    parser.add_argument('image', help='图像路径')
    
    args = parser.parse_args()
    quick_infer(args.image)
