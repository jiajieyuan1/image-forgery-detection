"""
数据加载器模块
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from .preprocessing import DataAugmentation, ImagePreprocessor


class ForgeriesDataset(Dataset):
    """图像伪造检测数据集"""
    
    def __init__(self, image_paths, labels, transform=None, augment=True):
        """
        Args:
            image_paths: 图像路径列表
            labels: 标签列表 (0: real, 1: fake)
            transform: 图像变换
            augment: 是否使用数据增强
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.preprocessor = ImagePreprocessor()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = str(self.image_paths[idx])
        label = self.labels[idx]
        
        # 加载图像
        img = cv2.imread(image_path)
        if img is None:
            # 返回占位符
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        if self.transform is not None:
            img = self.transform(image=img)['image']
        
        return img, torch.tensor(label, dtype=torch.long)


class MultiSourceDataset(Dataset):
    """多源数据集（DeepFake、篡改、合成等）"""
    
    def __init__(self, data_sources, transform=None):
        """
        Args:
            data_sources: dict，包含不同来源的数据
                {
                    'deepfake': {'images': [...], 'labels': [...]},
                    'manipulation': {'images': [...], 'labels': [...]},
                    'synthetic': {'images': [...], 'labels': [...]}
                }
            transform: 图像变换
        """
        self.data_sources = data_sources
        self.transform = transform
        self.all_images = []
        self.all_labels = []
        self.source_types = []
        
        # 合并所有数据源
        source_type_map = {'deepfake': 0, 'manipulation': 1, 'synthetic': 2}
        
        for source_name, source_data in data_sources.items():
            self.all_images.extend(source_data['images'])
            self.all_labels.extend(source_data['labels'])
            self.source_types.extend([source_type_map.get(source_name, 0)] * len(source_data['labels']))
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.all_labels[idx]
        source_type = self.source_types[idx]
        
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long),
            'source': torch.tensor(source_type, dtype=torch.long)
        }


class DatasetBuilder:
    """数据集构建器"""
    
    def __init__(self, data_dir, image_size=(256, 256), augment_params=None):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment_params = augment_params or {}
        self.augmentation = DataAugmentation(image_size, augment_params)
    
    def create_dataset_from_directory(self, batch_size=32, num_workers=4):
        """从目录结构创建数据集"""
        image_paths = []
        labels = []
        
        # 假设目录结构: data_dir/real /* 和 data_dir/fake /*
        real_dir = self.data_dir / 'real'
        fake_dir = self.data_dir / 'fake'
        
        if real_dir.exists():
            for img_path in list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png')):
                image_paths.append(img_path)
                labels.append(0)  # Real
        
        if fake_dir.exists():
            for img_path in list(fake_dir.glob('*.jpg')) + list(fake_dir.glob('*.png')):
                image_paths.append(img_path)
                labels.append(1)  # Fake
        
        if not image_paths:
            raise ValueError(f"未找到图像文件于 {self.data_dir}")
        
        # 划分数据集
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        # 创建数据集
        train_dataset = ForgeriesDataset(
            train_paths, train_labels,
            transform=self.augmentation.get_train_transform()
        )
        
        val_dataset = ForgeriesDataset(
            val_paths, val_labels,
            transform=self.augmentation.get_val_transform()
        )
        
        test_dataset = ForgeriesDataset(
            test_paths, test_labels,
            transform=self.augmentation.get_val_transform()
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def create_multi_source_dataset(self, sources: dict, batch_size=32, num_workers=4):
        """创建多源数据集"""
        dataset = MultiSourceDataset(
            sources,
            transform=self.augmentation.get_train_transform()
        )
        
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader


def create_demo_dataset(num_samples=100):
    """创建演示数据集"""
    images = []
    labels = []
    
    for i in range(num_samples):
        # 生成随机图像
        if i < num_samples // 2:
            # Real images (Gaussian noise)
            img = np.random.randn(256, 256, 3) * 30 + 128
            label = 0
        else:
            # Fake images (不同的模式)
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            img[:, :, 0] = np.sin(np.linspace(0, 4*np.pi, 256))[:, np.newaxis] * 127 + 128
            label = 1
        
        img = np.clip(img, 0, 255).astype(np.uint8)
        images.append(img)
        labels.append(label)
    
    return images, labels
