"""
训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import argparse
from pathlib import Path
import sys
import time
from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ModelFactory
from src.data_loader import DatasetBuilder, ForgeriesDataset
from src.preprocessing import DataAugmentation
from config import *
from utils.metrics import MetricsCalculator, Visualization


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=1e-4, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
            # 保存最佳模型
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
        
        return False


class Trainer:
    """训练器"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        if config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        else:  # SGD
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay'],
                momentum=0.9
            )
        
        # 学习率调度器
        if config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['num_epochs'],
                eta_min=1e-6
            )
        else:
            self.scheduler = StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config['early_stop_patience'],
            min_delta=config['early_stop_min_delta'],
            save_path=MODELS_DIR / 'best_model.pth'
        )
        
        # 记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc='Train')
        
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': total_loss / (total // len(labels)),
                'acc': 100 * correct / total
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validate'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """完整训练流程"""
        print(f"开始训练... 设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(self.config['num_epochs']):
            print(f"\n[Epoch {epoch+1}/{self.config['num_epochs']}]")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # 早停
            if self.early_stopping(val_loss, self.model):
                print("早停触发，训练结束。")
                break
            
            # 定期保存
            if (epoch + 1) % self.config['save_interval'] == 0:
                save_path = MODELS_DIR / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save(self.model.state_dict(), save_path)
                print(f"模型已保存: {save_path}")
        
        # 绘制训练曲线
        Visualization.plot_training_curves(
            self.history['train_loss'],
            self.history['val_loss'],
            self.history['train_acc'],
            self.history['val_acc'],
            save_path=RESULTS_DIR / 'training_curves.png'
        )
        
        print("\n训练完成！")


def main():
    parser = argparse.ArgumentParser(description='训练图像伪造检测模型')
    parser.add_argument('--model', type=str, default='efficientnet',
                      choices=['resnet50', 'efficientnet', 'xception', 'multi_task'],
                      help='模型架构')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--data-dir', type=str, default='./data/raw',
                      help='数据目录')
    
    args = parser.parse_args()
    
    # 更新配置
    config = TRAINING.copy()
    config['num_epochs'] = args.epochs
    
    # 创建模型
    print(f"创建模型: {args.model}")
    model = ModelFactory.create_model(
        args.model,
        num_classes=2,
        pretrained=True,
        dropout=0.3
    )
    
    # 创建数据加载器
    print(f"加载数据: {args.data_dir}")
    dataset_builder = DatasetBuilder(
        args.data_dir,
        image_size=IMAGE_SIZE,
        augment_params=AUGMENTATION_PARAMS
    )
    
    try:
        train_loader, val_loader, test_loader = dataset_builder.create_dataset_from_directory(
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS
        )
    except ValueError as e:
        print(f"错误: {e}")
        print("请确保数据目录结构为: data_dir/real/* 和 data_dir/fake/*")
        return
    
    # 训练
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
    
    # 保存最终模型
    final_model_path = MODELS_DIR / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存: {final_model_path}")


if __name__ == '__main__':
    main()
