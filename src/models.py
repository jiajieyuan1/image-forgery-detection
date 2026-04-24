"""
深度学习检测模型
包括 ResNet50, EfficientNet, XceptionNet, Vision Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class AttentionModule(nn.Module):
    """注意力机制模块"""
    
    def __init__(self, channels, reduction=16):
        super(AttentionModule, self).__init__()
        
        # Squeeze-and-Excitation (SE) 模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Convolutional Block Attention Module (CBAM)
        self.channel_attn = self.se
        # 空间注意力：输入是 mean 和 max concatenated，所以是 2 通道
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        channel_out = x * self.channel_attn(x)
        
        # 空间注意力
        spatial_attn = self.spatial_attn(
            torch.cat([channel_out.mean(1, keepdim=True), 
                      channel_out.max(1, keepdim=True)[0]], dim=1)
        )
        out = channel_out * spatial_attn
        
        return out


class ForgeryDetectionHead(nn.Module):
    """伪造检测头"""
    
    def __init__(self, in_features, num_classes=2, hidden_dim=512, dropout=0.3):
        super(ForgeryDetectionHead, self).__init__()
        
        self.attention = AttentionModule(in_features if in_features > 1 else 1)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 置信度预测
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.attention(x)
        features = F.adaptive_avg_pool2d(x, 1).flatten(1)
        
        logits = self.classifier(x)
        
        return logits


class ResNet50Detector(nn.Module):
    """基于ResNet50的伪造检测器"""
    
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super(ResNet50Detector, self).__init__()
        
        # 加载预训练的ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # 移除原始分类器
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 添加检测头
        self.head = ForgeryDetectionHead(
            in_features, num_classes=num_classes, dropout=dropout
        )
    
    def forward(self, x):
        features = self.backbone(x)  # [B, 2048]
        
        # 重塑为 [B, 2048, 1, 1] 以使用注意力模块
        features = features.view(features.size(0), -1, 1, 1)
        
        logits = self.head(features)
        
        return logits


class EfficientNetDetector(nn.Module):
    """基于EfficientNet的伪造检测器"""
    
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super(EfficientNetDetector, self).__init__()
        
        # 加载预训练的EfficientNet
        self.backbone = models.efficientnet_b4(pretrained=pretrained)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.head = ForgeryDetectionHead(
            in_features, num_classes=num_classes, dropout=dropout
        )
    
    def forward(self, x):
        features = self.backbone.features(x)  # [B, 1280, 8, 8]
        logits = self.head(features)
        return logits


class XceptionDetector(nn.Module):
    """基于Xception的伪造检测器"""
    
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super(XceptionDetector, self).__init__()
        
        # 加载预训练的Xception
        self.backbone = models.resnext50_32x4d(pretrained=pretrained)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.head = ForgeryDetectionHead(
            in_features, num_classes=num_classes, dropout=dropout
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1, 1, 1)
        logits = self.head(features)
        return logits


class EnsembleDetector(nn.Module):
    """模型集成检测器"""
    
    def __init__(self, model_list, num_classes=2, fusion='voting'):
        """
        Args:
            model_list: 模型列表
            num_classes: 类别数
            fusion: 融合策略 ('voting', 'average', 'weighted')
        """
        super(EnsembleDetector, self).__init__()
        
        self.models = nn.ModuleList(model_list)
        self.num_classes = num_classes
        self.fusion = fusion
    
    def forward(self, x):
        outputs = []
        
        for model in self.models:
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            outputs.append(probs)
        
        # 融合预测
        if self.fusion == 'voting':
            # 投票融合
            preds = torch.stack([torch.argmax(o, dim=1) for o in outputs])
            mode_pred = torch.mode(preds, dim=0)[0]
            return F.one_hot(mode_pred, self.num_classes).float()
        
        elif self.fusion == 'average':
            # 平均融合
            avg_probs = torch.mean(torch.stack(outputs), dim=0)
            return avg_probs
        
        elif self.fusion == 'weighted':
            # 加权融合
            weights = torch.ones(len(outputs), device=x.device) / len(outputs)
            weighted_probs = sum(w * p for w, p in zip(weights, outputs))
            return weighted_probs
        
        return torch.mean(torch.stack(outputs), dim=0)


class MultiTaskDetector(nn.Module):
    """多任务学习检测器"""
    
    def __init__(self, backbone_name='efficientnet', num_classes=2, dropout=0.3):
        super(MultiTaskDetector, self).__init__()
        
        # 骨干网络
        if backbone_name == 'efficientnet':
            self.backbone = models.efficientnet_b4(weights='DEFAULT')
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights='DEFAULT')
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        # 分类任务头
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # 定位任务头（检测篡改区域）
        self.localization_head = nn.Sequential(
            nn.Conv2d(in_features, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),  # Heatmap
            nn.Sigmoid()
        )
        
        # 置信度估计头
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone.features(x) if hasattr(self.backbone, 'features') else self.backbone(x)
        
        # 分类
        classification_logits = self.classification_head(features)
        
        # 定位
        localization_map = self.localization_head(features)
        
        # 置信度
        confidence = self.confidence_head(features)
        
        return {
            'classification': classification_logits,
            'localization': localization_map,
            'confidence': confidence
        }


class ModelFactory:
    """模型工厂"""
    
    @staticmethod
    def create_model(model_name, num_classes=2, pretrained=True, dropout=0.3):
        """创建指定的模型"""
        
        if model_name == 'resnet50':
            return ResNet50Detector(num_classes, pretrained, dropout)
        
        elif model_name == 'efficientnet':
            return EfficientNetDetector(num_classes, pretrained, dropout)
        
        elif model_name == 'xception':
            return XceptionDetector(num_classes, pretrained, dropout)
        
        elif model_name == 'multi_task':
            return MultiTaskDetector('efficientnet', num_classes, dropout)
        
        else:
            raise ValueError(f"未知模型: {model_name}")
    
    @staticmethod
    def create_ensemble(model_names, num_classes=2):
        """创建集成模型"""
        models_list = [
            ModelFactory.create_model(name, num_classes) 
            for name in model_names
        ]
        return EnsembleDetector(models_list, num_classes)
