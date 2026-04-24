"""
数据预处理模块
包括图像加载、标准化、增强等功能
"""

import cv2
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, image_size=(256, 256), normalize=True):
        self.image_size = image_size
        self.normalize = normalize
        
        # 标准化参数（ImageNet）
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def load_image(self, image_path):
        """加载图像"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法加载图像：{image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def resize(self, img, size=None):
        """调整大小"""
        if size is None:
            size = self.image_size
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(self, img):
        """标准化图像"""
        if self.normalize:
            img = img.astype(np.float32) / 255.0
            for i in range(3):
                img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]
        return img
    
    def preprocess(self, image_path, return_numpy=False):
        """完整预处理流程"""
        img = self.load_image(image_path)
        img = self.resize(img)
        img = self.normalize_image(img)
        
        if not return_numpy:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        return img


class DataAugmentation:
    """数据增强"""
    
    def __init__(self, image_size=(256, 256), augment_params=None):
        self.image_size = image_size
        self.augment_params = augment_params or {}
        
        # 训练增强
        self.train_transform = A.Compose([
            A.Resize(*image_size),
            A.HorizontalFlip(p=self.augment_params.get('horizontal_flip', 0.5)),
            A.VerticalFlip(p=self.augment_params.get('vertical_flip', 0.2)),
            A.Rotate(limit=self.augment_params.get('rotation', 30), p=0.5),
            A.GaussNoise(p=0.3),
            A.ColorJitter(
                brightness=self.augment_params.get('brightness', 0.2),
                contrast=self.augment_params.get('contrast', 0.2),
                saturation=self.augment_params.get('saturation', 0.2),
                p=0.5
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # 验证增强（无随机变换）
        self.val_transform = A.Compose([
            A.Resize(*image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def get_train_transform(self):
        """获取训练变换"""
        return self.train_transform
    
    def get_val_transform(self):
        """获取验证变换"""
        return self.val_transform


class ManipulationDetectionPreprocessor:
    """篡改检测专用预处理器"""
    
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def extract_rgb_channels(self, img):
        """提取RGB通道"""
        return img[:, :, :3]
    
    def compute_dct_coefficients(self, img, block_size=8):
        """计算DCT系数（JPEG伪迹检测）"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # 分块计算DCT
        dct_blocks = []
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                if block.shape == (block_size, block_size):
                    dct = cv2.dct(block.astype(np.float32))
                    dct_blocks.append(dct)
        
        return np.array(dct_blocks) if dct_blocks else None
    
    def compute_noise_residual(self, img):
        """计算噪声残差"""
        # Laplacian高通滤波
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        residual = cv2.filter2D(img, -1, kernel)
        return residual
    
    def extract_manipulation_features(self, img):
        """提取篡改特征"""
        features = {}
        
        # 噪声分析
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        features['noise_variance'] = np.var(gray)
        
        # 边界特征
        features['edge_strength'] = np.sum(cv2.Canny(gray, 100, 200)) / (gray.shape[0] * gray.shape[1])
        
        # 颜色一致性
        features['color_std'] = np.std(img, axis=(0, 1))
        
        return features


class SynthenticImageDetector:
    """合成图像检测预处理"""
    
    @staticmethod
    def analyze_frequency_spectrum(img):
        """分析频率谱"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        return magnitude_spectrum
    
    @staticmethod
    def compute_phase_consistency(img, num_scales=3):
        """计算相位一致性（自然图像特征）"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32)
        
        phase_consistency = []
        
        for scale in range(1, num_scales + 1):
            # 多尺度分析
            kernel = cv2.getGaussianKernel(2**scale + 1, 1.0)
            blurred = cv2.filter2D(gray, -1, kernel)
            diff = np.abs(gray - blurred)
            phase_consistency.append(np.mean(diff))
        
        return np.array(phase_consistency)
    
    @staticmethod
    def extract_texture_features(img, num_bins=256):
        """提取纹理特征"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 直方图
        hist = cv2.calcHist([gray], [0], None, [num_bins], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # LBP（局部二进制模式）
        lbp_hist, _ = np.histogram(gray.flatten(), bins=num_bins, range=(0, 256))
        lbp_hist = lbp_hist / lbp_hist.sum()
        
        return hist, lbp_hist


def batch_preprocess_images(image_dir, image_size=(256, 256), output_dir=None):
    """批量预处理图像"""
    preprocessor = ImagePreprocessor(image_size=image_size)
    image_dir = Path(image_dir)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_images = []
    
    for image_path in image_dir.glob('**/*.jpg') | image_dir.glob('**/*.png'):
        try:
            img = preprocessor.preprocess(image_path, return_numpy=True)
            
            if output_dir:
                output_path = output_dir / image_path.name
                np.save(output_path, img)
            
            processed_images.append(img)
        except Exception as e:
            print(f"处理图像失败 {image_path}: {e}")
    
    return processed_images
