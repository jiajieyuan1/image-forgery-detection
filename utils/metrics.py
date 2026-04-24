"""
工具函数：评估指标、可视化等
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, auc, precision_recall_curve
)
import seaborn as sns
import cv2
from pathlib import Path


class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_proba=None):
        """计算分类指标"""
        metrics = {}
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # 基本指标
        tn, fp, fn, tp = cm.ravel()
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['tp'] = tp
        
        # 精度、召回率、F1
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (
            metrics['precision'] + metrics['recall'] if (metrics['precision'] + metrics['recall']) > 0 else 1
        )
        
        # AUC-ROC
        if y_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        
        # 分类报告
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=['Real', 'Fake'], output_dict=True
        )
        
        return metrics
    
    @staticmethod
    def calculate_per_class_metrics(y_true, y_pred):
        """计算每类指标"""
        unique_classes = np.unique(y_true)
        class_metrics = {}
        
        for cls in unique_classes:
            mask = y_true == cls
            class_pred = y_pred[mask]
            class_true = y_true[mask]
            
            accuracy = np.mean(class_pred == class_true)
            class_metrics[f'class_{cls}'] = {
                'accuracy': accuracy,
                'samples': np.sum(mask)
            }
        
        return class_metrics


class Visualization:
    """可视化工具"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path=None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    @staticmethod
    def plot_roc_curve(y_true, y_proba, save_path=None):
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    @staticmethod
    def plot_precision_recall_curve(y_true, y_proba, save_path=None):
        """绘制精度-召回曲线"""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='o', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    @staticmethod
    def visualize_predictions(image_path, prediction_result, save_path=None):
        """可视化预测结果"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.imshow(img)
        axes.axis('off')
        
        # 添加文本标注
        class_name = prediction_result['class_name']
        confidence = prediction_result['confidence']
        
        color = 'red' if class_name == 'Fake' else 'green'
        title = f'{class_name} (Confidence: {confidence:.3f})'
        axes.set_title(title, fontsize=16, color=color, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    @staticmethod
    def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(train_losses, label='Train Loss')
        axes[0].plot(val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 精度曲线
        axes[1].plot(train_accs, label='Train Accuracy')
        axes[1].plot(val_accs, label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()


class ExplanationGenerator:
    """解释生成器"""
    
    @staticmethod
    def generate_prediction_explanation(prediction_result):
        """生成预测解释"""
        explanation = {
            'prediction': prediction_result['class_name'],
            'confidence': f"{prediction_result['confidence']:.1%}",
            'reasoning': []
        }
        
        real_prob = prediction_result['probabilities']['real']
        fake_prob = prediction_result['probabilities']['fake']
        
        if prediction_result['is_fake']:
            explanation['reasoning'].append(
                f"模型以{fake_prob:.1%}的置信度认为这是伪造图像。"
            )
            
            if real_prob > 0.3:
                explanation['reasoning'].append(
                    "但仍存在{:.1%}的概率这是真实图像，请谨慎对待。".format(real_prob)
                )
        else:
            explanation['reasoning'].append(
                f"模型以{real_prob:.1%}的置信度认为这是真实图像。"
            )
        
        return explanation
    
    @staticmethod
    def generate_detection_report(results_list):
        """生成检测报告"""
        total = len(results_list)
        fake_count = sum(1 for r in results_list if r['is_fake'])
        real_count = total - fake_count
        
        report = f"""
        检测报告
        ========
        总图像数: {total}
        伪造图像数: {fake_count} ({fake_count/total:.1%})
        真实图像数: {real_count} ({real_count/total:.1%})
        
        平均置信度: {np.mean([r['confidence'] for r in results_list]):.3f}
        """
        
        return report.strip()


def create_demo_images(output_dir, num_samples=10):
    """创建演示图像"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建真实图像（颜色梯度）
    real_dir = output_dir / 'real'
    real_dir.mkdir(exist_ok=True)
    
    for i in range(num_samples // 2):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        # 创建颜色梯度
        for x in range(256):
            img[:, x] = [x, 128, 255-x]
        cv2.imwrite(str(real_dir / f'real_{i}.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # 创建伪造图像（噪声图案）
    fake_dir = output_dir / 'fake'
    fake_dir.mkdir(exist_ok=True)
    
    for i in range(num_samples // 2):
        img = np.random.randn(256, 256, 3) * 50 + 128
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(str(fake_dir / f'fake_{i}.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
