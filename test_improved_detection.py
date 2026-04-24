"""
改进的检测算法测试脚本 v2
用于验证激进的AI图片检测逻辑
"""

import torch
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ModelFactory
from src.inference import InferenceEngine
from config import MODELS_DIR


def test_detection(image_dir="ai图片"):
    """测试检测结果"""
    model_path = MODELS_DIR / 'best_model.pth'
    if not model_path.exists():
        model_path = MODELS_DIR / 'final_model.pth'
    
    if not model_path.exists():
        print(f"错误: 模型文件不存在于 {MODELS_DIR}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")
    
    # 创建模型并加载权重
    model = ModelFactory.create_model('efficientnet', num_classes=2, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 创建推理引擎
    engine = InferenceEngine(model, device=device)
    
    # 扫描图像文件夹
    image_folder = Path(image_dir)
    if not image_folder.exists():
        print(f"文件夹 {image_dir} 不存在")
        return
    
    image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
    
    if not image_files:
        print(f"在 {image_dir} 中未找到图像文件")
        return
    
    print(f"发现 {len(image_files)} 张图像\n")
    print("=" * 100)
    print("激进的AI图片检测结果 (v2)")
    print("=" * 100)
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] 📄 {image_path.name}")
        print("-" * 100)
        
        try:
            result = engine.infer_single_image(str(image_path))
            
            # 打印详细结果
            print(f"  【检测结果】:     {result['class_name']}")
            print(f"  【检测等级】:     {result['detection_level']}")
            print(f"  【置信度等级】:   {result.get('confidence_level', 'N/A')}")
            print(f"  【检测原因】:     {result['detection_reason']}")
            print(f"  " + "-" * 80)
            print(f"  真实概率:         {result['probabilities']['real']:.2%}")
            print(f"  假图概率:         {result['probabilities']['fake']:.2%}")
            print(f"  概率差距:         {result['probability_difference']:.2%}")
            print(f"  假/真比例:        {result.get('fake_real_ratio', 'N/A'):.3f}")
            print(f"  概率熵:           {result.get('entropy', 'N/A'):.4f}")
            print(f"  " + "-" * 80)
            print(f"  显示AI分数:       {result['ai_detection_score']:.1%}")
            print(f"  高置信度:         {'✓ 是' if result['is_confident'] else '✗ 否'}")
            
            # 结果判断
            if result['is_fake']:
                print(f"\n  ⚠️  结论: 【判定为AI生成图片】")
            else:
                print(f"\n  ✓  结论: 【判定为真实照片】")
                
        except Exception as e:
            print(f"  错误: {e}")
    
    print("\n" + "=" * 100)
    print("测试完成!")
    print("=" * 100)
    
    # 显示判断标准说明
    print("\n" + "=" * 100)
    print("判断标准说明 (激进算法 v2)")
    print("=" * 100)
    print("""
【Definitely AI】           fake_prob >= 80%
【Likely AI】               fake_prob >= 50% 且模型困惑 (差距 < 15%)
【Possible AI】             fake_prob >= 35% (激进判断)
【Needs Review】            fake_prob >= 20% 且模型困惑 (建议人工审核)
【Suspicious】              fake > real 且模型困惑
【Likely Real】             fake_prob < 20% 且模型清楚

模型困惑判断: 当真实和假图概率差距 < 15% 时，模型不确定，可能是AI
    """)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='测试激进的检测算法v2')
    parser.add_argument('--dir', default='ai图片', help='图像文件夹路径')
    
    args = parser.parse_args()
    test_detection(args.dir)
