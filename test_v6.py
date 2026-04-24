"""
v6版本测试脚本 - 快速测试激进判断算法
测试 ai图片 文件夹中的所有图像
"""

import torch
import sys
from pathlib import Path
from tabulate import tabulate

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ModelFactory
from src.inference import InferenceEngine
from config import MODELS_DIR


def test_v6_aggressive():
    """测试v6激进算法"""
    
    # 检查模型存在
    model_path = MODELS_DIR / 'best_model.pth'
    if not model_path.exists():
        model_path = MODELS_DIR / 'final_model.pth'
    
    if not model_path.exists():
        print(f"❌ 错误: 模型文件不存在")
        return
    
    # 设备和模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ 设备: {device}")
    
    model = ModelFactory.create_model('efficientnet', num_classes=2, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    engine = InferenceEngine(model, device=device)
    print(f"✓ 模型已加载\n")
    
    # 获取图像文件夹
    ai_images_dir = Path(__file__).parent / 'ai图片'
    if not ai_images_dir.exists():
        print(f"❌ 找不到 ai图片 文件夹")
        return
    
    # 获取所有图像
    image_files = sorted([
        f for f in ai_images_dir.glob('*')
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    ])
    
    if not image_files:
        print(f"❌ ai图片 文件夹中没有图像")
        return
    
    print(f"📊 找到 {len(image_files)} 张图像\n")
    
    # 测试每张图像
    results = []
    ai_count = 0
    
    print("=" * 140)
    print("v9 激进得分转换版本 - 检测结果（显示原始得分 vs 调整后得分）")
    print("=" * 140)
    
    for idx, image_path in enumerate(image_files, 1):
        try:
            result = engine.infer_single_image(str(image_path))
            
            # 统计AI检测
            if result['class'] == 1:
                ai_count += 1
            
            # 提取关键信息
            results.append({
                '序号': idx,
                '文件': image_path.name[:35],
                '结果': result['class_name'],
                '置信度': result['confidence_level'],
                '等级': result['detection_level'],
                'Raw_Fake%': f"{result['raw_fake_prob']:.1%}",
                '调整后Fake%': f"{result['adjusted_fake_score']:.1%}",
                '差异': f"{(result['adjusted_fake_score']-result['raw_fake_prob'])*100:+.1f}%",
                '熵值': f"{result.get('entropy', 0):.3f}",
                'Ratio': f"{result.get('fake_real_ratio', 0):.2f}",
            })
            
        except Exception as e:
            print(f"❌ 处理 {image_path.name} 时出错: {e}")
            continue
    
    # 打印表格
    print(tabulate(results, headers='keys', tablefmt='grid'))
    
    print("\n" + "=" * 140)
    print(f"📈 统计结果 (v9版本)")
    print("=" * 140)
    print(f"总检测数: {len(image_files)} 张")
    print(f"判为AI的图片: {ai_count} 张")
    print(f"判为Real的图片: {len(image_files) - ai_count} 张")
    
    # AI检测率
    if len(image_files) > 0:
        ai_rate = (ai_count / len(image_files)) * 100
        print(f"\n✨ AI检测率: {ai_rate:.1f}%")
        
        if ai_rate == 100:
            print("🎉 完美! v6 版本成功检测所有AI图片!")
        elif ai_rate >= 75:
            print(f"✅ 达标! v9检测率 >= 75%")
        else:
            print(f"⚠️  检测率低于75%, 需要调整参数")
    
    print("\n" + "=" * 140)


if __name__ == '__main__':
    test_v6_aggressive()
