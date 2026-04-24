"""
创建演示数据集
"""

import numpy as np
import cv2
from pathlib import Path
import sys
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, RAW_DATA_DIR


def create_demo_dataset(num_samples_per_class=20):
    """创建演示数据集"""
    
    real_dir = RAW_DATA_DIR / 'real'
    fake_dir = RAW_DATA_DIR / 'fake'
    
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    print("创建演示数据集...")
    
    # 创建真实图像（自然图像特征）
    print(f"生成 {num_samples_per_class} 张真实图像...")
    for i in range(num_samples_per_class):
        # 生成自然的梯度图像
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        for x in range(256):
            for y in range(256):
                # 创建自然的颜色变化
                img[y, x, 0] = int((x / 256) * 255)  # R
                img[y, x, 1] = int((y / 256) * 255)  # G
                img[y, x, 2] = int(((x + y) / 512) * 255)  # B
        
        # 添加一些噪声变得更自然
        noise = np.random.randn(*img.shape) * 10
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        
        path = real_dir / f'real_{i:04d}.jpg'
        # 使用 PIL 而不是 cv2，因为 OpenCV 在中文路径下会失败
        Image.fromarray(img, 'RGB').save(str(path))
    
    # 创建伪造图像（人工噪声图像）
    print(f"生成 {num_samples_per_class} 张伪造图像...")
    for i in range(num_samples_per_class):
        # 生成高频噪声（假图像特征）
        img = np.random.randn(256, 256, 3) * 40 + 128
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        path = fake_dir / f'fake_{i:04d}.jpg'
        # 使用 PIL 而不是 cv2，因为 OpenCV 在中文路径下会失败
        Image.fromarray(img, 'RGB').save(str(path))
    
    print(f"✓ 演示数据集创建完成！")
    print(f"  真实图像: {real_dir}")
    print(f"  伪造图像: {fake_dir}")


if __name__ == '__main__':
    create_demo_dataset(num_samples_per_class=30)
