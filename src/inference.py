"""
推理引擎
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model, device='cuda', confidence_threshold=0.5):
        """
        Args:
            model: 已加载的模型
            device: 使用的设备 ('cuda' 或 'cpu')
            confidence_threshold: 置信度阈值
        """
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model.eval()
        self.model = self.model.to(device)
    
    def _compute_aggressive_score(self, fake_prob, real_prob):
        """
        v9核心创新：激进得分转换函数
        将raw_fake_prob通过多层非线性变换转换为激进评分
        
        算法思路：
        1. 如果fake>real，应用上升幂函数（夸大差距）
        2. 如果real>fake但fake不太低，仍给予一定分数
        3. 多层非线性变换确保低分也有变化空间
        """
        # 基础计算
        raw_fake = fake_prob
        prob_ratio = fake_prob / (real_prob + 1e-6)
        
        # 变换1：比率信号放大 - 由于fake/real比率可能极端，用log来平衡
        ratio_transform = np.log(prob_ratio + 0.001) / np.log(10)  # 对数放大
        
        # 变换2：基于fake概率的多项式变换
        # 当fake很低时（如0.12），仍给予一定权重
        if fake_prob < 0.3:
            # 极低概率情况：使用aggressivep^0.3（平方根会放大小数）
            poly_transform = np.power(fake_prob, 0.4) * 5  # 0.12^0.4 ≈ 0.45
        elif fake_prob < 0.5:
            # 低概率情况：p^0.5
            poly_transform = np.sqrt(fake_prob) * 2.5  # 0.48^0.5 ≈ 0.69
        elif fake_prob < 0.7:
            # 中等概率：p^0.6
            poly_transform = np.power(fake_prob, 0.6) * 2.0
        else:
            # 高概率：直接使用
            poly_transform = fake_prob * 1.2
        
        # 变换3：entropy信号（高熵往往表示AI）
        entropy = -fake_prob * np.log(fake_prob + 1e-6) - real_prob * np.log(real_prob + 1e-6)
        entropy_boost = entropy * 0.15  # 熵的0-0.7范围映射到0-0.1
        
        # 变换4：综合多层信号
        # 权重分配
        w_poly = 0.5    # 多项式变换权重
        w_ratio = 0.3   # 比率信号权重
        w_entropy = 0.2 # 熵信号权重
        
        # 边界处理：ratio_transform可能为负（fake<real时）
        ratio_transform = max(ratio_transform, -0.5)  # 最负-0.5
        
        # 综合得分
        adjusted_score = w_poly * poly_transform + w_ratio * (ratio_transform + 1) / 2 + w_entropy * entropy_boost
        
        # 最终映射到[0, 1]
        adjusted_score = np.clip(adjusted_score, 0, 1)
        
        return adjusted_score
    
    def preprocess_image(self, image_path, image_size=(256, 256)):
        """预处理单张图像"""
        from PIL import Image
        
        # 尝试用cv2加载，失败则用PIL
        img = cv2.imread(str(image_path))
        if img is None:
            try:
                # PIL作为备选方案
                pil_img = Image.open(str(image_path)).convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                raise ValueError(f"无法加载图像：{image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)
        
        # 标准化
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # 转换为tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img = img.unsqueeze(0).to(self.device)
        
        return img
    
    @torch.no_grad()
    def infer_single_image(self, image_path) -> Dict:
        """单张图像推理 - v9版本（激进得分转换+改应逻辑）"""
        img_tensor = self.preprocess_image(image_path)
        
        # 前向传播
        logits = self.model(img_tensor)
        probs = F.softmax(logits, dim=1)
        
        real_prob = probs[0, 0].item()
        fake_prob = probs[0, 1].item()
        prob_diff = abs(fake_prob - real_prob)
        
        # ========== v9 核心创新：激进得分转换 ==========
        # 不再直接用raw_fake_prob，而是通过多层变换获得激进分数
        adjusted_fake_score = self._compute_aggressive_score(fake_prob, real_prob)
        
        # ========== v6: 多特征激进判断 ==========
        # 目标：最大化准确率，不考虑代码复杂度
        
        # 计算多个特征指标
        fake_real_ratio = fake_prob / (real_prob + 1e-6)
        entropy = -real_prob * np.log(real_prob + 1e-6) - fake_prob * np.log(fake_prob + 1e-6)
        
        # 特征1: 异常熵检测（AI图片通常熵更高=更不确定）
        high_entropy = entropy > 0.65  
        
        # 特征2: 比例异常
        ratio_signal = fake_real_ratio > 1.0 or (fake_real_ratio < 1.0 and fake_real_ratio > 0.8)
        
        # 特征3: 极端概率（偏向某一方）
        extreme_signal = fake_prob > 0.7 or real_prob > 0.7
        
        # 特征4: 微弱但一致的信号（fake略高）
        weak_consistent = fake_prob > real_prob and prob_diff > 0.05
        
        # 特征5: 边界不确定（两者接近）
        boundary_uncertain = 0.45 <= fake_prob <= 0.55
        
        # ========== v9 多层判断逻辑（基于adjusted_fake_score） ==========
        
        # 计算基准值
        adjusted_real_score = 1.0 - adjusted_fake_score  # 对称分数
        
        # 层级1: 极强信号（adjusted_score >= 0.80）
        if adjusted_fake_score >= 0.80:
            final_class = 1
            confidence_level = "Very High"
            detection_level = "Extremely Likely AI"
            reason = f"Strong adjusted AI signal: {adjusted_fake_score:.1%}"
            
        elif adjusted_real_score >= 0.80:
            final_class = 0
            confidence_level = "Very High"
            detection_level = "Extremely Likely Real"
            reason = f"Strong adjusted Real signal: {adjusted_real_score:.1%}"
        
        # 层级2: 中强信号 (60-80%)
        elif adjusted_fake_score >= 0.60:
            final_class = 1
            confidence_level = "High"
            detection_level = "Likely AI"
            reason = f"Moderate-high adjusted AI: {adjusted_fake_score:.1%}"
            
        elif adjusted_real_score >= 0.60:
            final_class = 0
            confidence_level = "High"
            detection_level = "Likely Real"
            reason = f"Moderate-high adjusted Real: {adjusted_real_score:.1%}"
        
        # 层级3: 中等信号 (40-60%)
        elif adjusted_fake_score >= 0.40:
            final_class = 1
            confidence_level = "Medium"
            detection_level = "Moderate AI"
            reason = f"Moderate adjusted AI: {adjusted_fake_score:.1%}"
            
        elif adjusted_real_score >= 0.40:
            final_class = 0
            confidence_level = "Medium"
            detection_level = "Moderate Real"
            reason = f"Moderate adjusted Real: {adjusted_real_score:.1%}"
        
        # 层级4: 弱信号 (30-40%)
        elif adjusted_fake_score >= 0.30:
            final_class = 1
            confidence_level = "Low"
            detection_level = "Weak AI Signal"
            reason = f"Weak adjusted AI: {adjusted_fake_score:.1%}"
        
        # 层级5: 极弱信号 (> 0.15)
        elif adjusted_fake_score > 0.15:
            final_class = 1
            confidence_level = "Very Low"
            detection_level = "Possible AI"
            reason = f"Very weak adjusted AI: {adjusted_fake_score:.1%}"
        
        else:
            # 默认：真实
            final_class = 0
            confidence_level = "Very Low"
            detection_level = "Likely Real"
            reason = f"No sufficient AI signal: {adjusted_fake_score:.1%}"
        
        # ========== 返回结果 ==========
        display_score = adjusted_fake_score  # 使用调整后的得分
        
        return {
            'class': final_class,
            'class_name': 'Fake' if final_class == 1 else 'Real',
            'confidence': display_score,
            'probabilities': {
                'real': real_prob,
                'fake': fake_prob
            },
            'probability_difference': prob_diff,
            'fake_real_ratio': fake_real_ratio,
            'entropy': entropy,
            'detection_level': detection_level,
            'detection_reason': reason,
            'confidence_level': confidence_level,
            'is_fake': final_class == 1,
            'is_confident': confidence_level in ['High', 'Very High', 'Medium'],
            'ai_detection_score': display_score,  # 使用调整后的激进得分
            'adjusted_fake_score': adjusted_fake_score,  # 新增：显示调整后的得分
            'raw_fake_prob': fake_prob,
            'raw_real_prob': real_prob,
            'display_score': display_score,
            'high_entropy': high_entropy,
            'weak_consistent': weak_consistent,
            'ratio_signal': ratio_signal
        }
    
    @torch.no_grad()
    def infer_batch(self, image_paths: List) -> List[Dict]:
        """批量推理"""
        results = []
        for img_path in image_paths:
            result = self.infer_single_image(img_path)
            result['image_path'] = str(img_path)
            results.append(result)
        return results
    
    @torch.no_grad()
    def infer_with_heatmap(self, image_path, visualization=True) -> Dict:
        """包含热力图的推理（用于篡改定位）"""
        img = cv2.imread(str(image_path))
        original_img = img.copy()
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        
        # 预处理
        img_normalized = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_normalized - mean) / std
        
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # 推理
        logits = self.model(img_tensor)
        probs = F.softmax(logits, dim=1)
        
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
        
        result = {
            'class': pred_class,
            'class_name': 'Real' if pred_class == 0 else 'Fake',
            'confidence': confidence,
            'probabilities': {
                'real': probs[0, 0].item(),
                'fake': probs[0, 1].item()
            }
        }
        
        if visualization:
            result['original_image'] = original_img
            result['processed_image'] = img
        
        return result


class EnsembleInference:
    """集成推理"""
    
    def __init__(self, model_paths: List[str], device='cuda', ensemble_strategy='average'):
        """
        Args:
            model_paths: 模型路径列表
            device: 设备
            ensemble_strategy: 集成策略 ('average', 'voting', 'weighted')
        """
        self.device = device
        self.ensemble_strategy = ensemble_strategy
        self.engines = []
        
        for model_path in model_paths:
            model = torch.load(model_path, map_location=device)
            engine = InferenceEngine(model, device)
            self.engines.append(engine)
    
    @torch.no_grad()
    def infer(self, image_path) -> Dict:
        """集成推理"""
        results = []
        all_probs = []
        
        for engine in self.engines:
            result = engine.infer_single_image(image_path)
            results.append(result)
            all_probs.append([result['probabilities']['real'], 
                            result['probabilities']['fake']])
        
        all_probs = np.array(all_probs)
        
        if self.ensemble_strategy == 'average':
            avg_probs = all_probs.mean(axis=0)
            pred_class = np.argmax(avg_probs)
            confidence = avg_probs[pred_class]
        
        elif self.ensemble_strategy == 'voting':
            preds = np.argmax(all_probs, axis=1)
            pred_class = np.bincount(preds).argmax()
            confidence = np.mean([all_probs[i, pred_class] for i in range(len(self.engines))])
        
        elif self.ensemble_strategy == 'weighted':
            weights = np.array([1.0] * len(self.engines)) / len(self.engines)
            weighted_probs = (all_probs * weights[:, np.newaxis]).sum(axis=0)
            pred_class = np.argmax(weighted_probs)
            confidence = weighted_probs[pred_class]
        
        return {
            'class': int(pred_class),
            'class_name': 'Real' if pred_class == 0 else 'Fake',
            'confidence': float(confidence),
            'probabilities': {
                'real': float(avg_probs[0]),
                'fake': float(avg_probs[1])
            },
            'is_fake': pred_class == 1,
            'num_models': len(self.engines),
            'consensus_strength': float(np.std(all_probs, axis=0).min())  # 模型一致性
        }


class TilingInference:
    """分块推理（用于大图像）"""
    
    def __init__(self, model, tile_size=256, overlap=0.2, device='cuda'):
        """
        Args:
            model: 模型
            tile_size: 块大小
            overlap: 重叠比例
            device: 设备
        """
        self.engine = InferenceEngine(model, device)
        self.tile_size = tile_size
        self.overlap = overlap
    
    def infer_large_image(self, image_path) -> Dict:
        """对大图像进行分块推理"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法加载图像：{image_path}")
        
        h, w = img.shape[:2]
        
        # 计算分块参数
        step = int(self.tile_size * (1 - self.overlap))
        
        predictions = []
        confidences = []
        
        # 分块处理
        for y in range(0, h - self.tile_size, step):
            for x in range(0, w - self.tile_size, step):
                tile = img[y:y+self.tile_size, x:x+self.tile_size]
                
                # 推理
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                tile_normalized = tile_rgb.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                tile_normalized = (tile_normalized - mean) / std
                
                tile_tensor = torch.from_numpy(tile_normalized.transpose(2, 0, 1)).float()
                tile_tensor = tile_tensor.unsqueeze(0).to(self.engine.device)
                
                with torch.no_grad():
                    logits = self.engine.model(tile_tensor)
                    probs = F.softmax(logits, dim=1)
                
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0, pred].item()
                
                predictions.append(pred)
                confidences.append(conf)
        
        # 集成分块预测
        if predictions:
            final_pred = np.bincount(predictions).argmax()
            final_conf = np.mean(confidences)
        else:
            final_pred = 0
            final_conf = 0.0
        
        return {
            'class': int(final_pred),
            'class_name': 'Real' if final_pred == 0 else 'Fake',
            'confidence': float(final_conf),
            'num_tiles': len(predictions),
            'tile_predictions_stats': {
                'mean_confidence': float(np.mean(confidences)),
                'std_confidence': float(np.std(confidences)),
                'fake_ratio': float(np.mean(predictions))
            }
        }
