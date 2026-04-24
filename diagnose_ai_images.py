"""
诊断脚本 - 针对用户上传的AI图片分析
用于理解为什么前两张被误判为真实
"""

import torch
import sys
from pathlib import Path
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ModelFactory
from src.inference import InferenceEngine
from config import MODELS_DIR


def diagnose_images():
    """诊断图片检测分数"""
    model_path = MODELS_DIR / 'best_model.pth'
    if not model_path.exists():
        model_path = MODELS_DIR / 'final_model.pth'
    
    if not model_path.exists():
        print(f"错误: 模型文件不存在于 {MODELS_DIR}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")
    
    # 创建模型
    model = ModelFactory.create_model('efficientnet', num_classes=2, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 创建推理引擎
    engine = InferenceEngine(model, device=device)
    
    # 测试图片
    test_images = [
        ('ai图片\\第一张.jpg', '第一张（彩色人物，像素化）'),
        ('ai图片\\第二张.jpg', '第二张（油画风格背景）'),
        ('ai图片\\第三张.jpg', '第三张（机器人头部）'),
        ('ai图片\\第四张.jpg', '第四张（重复人物头像）'),
    ]
    
    print("=" * 100)
    print("AI图片诊断结果 - 分析每张图片的raw分数")
    print("=" * 100)
    
    for img_path, description in test_images:
        full_path = Path(img_path)
        
        if not full_path.exists():
            print(f"\n⚠️ {description}: 文件未找到 ({img_path})")
            continue
        
        print(f"\n{'='*100}")
        print(f"📸 {description}")
        print(f"路径: {full_path}")
        print(f"{'='*100}")
        
        try:
            result = engine.infer_single_image(str(full_path))
            
            real_prob = result['probabilities']['real']
            fake_prob = result['probabilities']['fake']
            prob_diff = result['probability_difference']
            
            print(f"\n【原始模型输出】")
            print(f"  真实概率:     {real_prob:.4f} ({real_prob*100:.2f}%)")
            print(f"  假图概率:     {fake_prob:.4f} ({fake_prob*100:.2f}%)")
            print(f"  概率差距:     {prob_diff:.4f} ({prob_diff*100:.2f}%)")
            print(f"  假/真比例:    {fake_prob/real_prob if real_prob > 0 else float('inf'):.4f}")
            print(f"  概率熵:       {result.get('entropy', 0):.4f}")
            
            print(f"\n【v4版本判定】")
            print(f"  检测结果:     {result['class_name']}")
            print(f"  检测等级:     {result['detection_level']}")
            print(f"  置信度等级:   {result.get('confidence_level', 'N/A')}")
            print(f"  判断原因:     {result['detection_reason']}")
            
            # 分析为什么会误判
            print(f"\n【误判分析】(如果是AI但判为真实)")
            if fake_prob < 0.55:
                print(f"  ➜ 问题: fake_prob = {fake_prob:.2%} < 55%")
                print(f"  ➜ 解决: 需要降低中等阈值或增加AI检测信号")
            if prob_diff < 0.15:
                print(f"  ➜ 问题: 概率差距 = {prob_diff:.2%} < 15%")
                print(f"  ➜ 解决: 需要降低差距要求")
            if real_prob > 0.8:
                print(f"  ➜ 问题: real_prob = {real_prob:.2%} > 80%")
                print(f"  ➜ 解决: 模型本身给真实概率过高，难以改进")
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")
    
    print(f"\n{'='*100}")
    print("诊断完成")
    print(f"{'='*100}")
    
    # 建议
    print("\n📋 【改进建议】\n")
    print("""
根据诊断结果，您可以选择以下调整方案：

方案1: 激进检测（如果fake_prob 40-55%）
  - 降低中等阈值: 0.55 → 0.48
  - 降低差距要求: 0.15 → 0.10
  文件: src/inference.py 第53行和54行

方案2: 超级激进（如果fake_prob 30-40%）
  - 降低中等阈值: 0.55 → 0.45
  - 降低差距要求: 0.15 → 0.08
  - 降低低置信度: 0.55 → 0.48
  
方案3: 极端激进（如果fake_prob < 30%）
  - 可能需要完全修改策略，添加其他AI检测特征
  - 或考虑重新训练模型

建议先尝试方案1，然后根据结果调整。
    """)


if __name__ == '__main__':
    diagnose_images()
