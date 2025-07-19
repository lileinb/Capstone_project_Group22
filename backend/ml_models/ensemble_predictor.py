"""
集成预测器
负责融合多个模型的预测结果
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """集成预测器类"""
    
    def __init__(self, strategy: str = 'weighted_voting', weights: Dict[str, float] = None):
        """
        初始化集成预测器
        
        Args:
            strategy: 集成策略 ('weighted_voting', 'simple_voting', 'confidence_based')
            weights: 模型权重字典
        """
        self.strategy = strategy
        self.weights = weights or {
            'catboost': 0.6,
            'xgboost': 0.4
        }
    
    def predict(self, model_probabilities: Dict[str, np.ndarray], threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        集成预测
        
        Args:
            model_probabilities: 各模型的预测概率字典
            threshold: 分类阈值
            
        Returns:
            集成预测结果和概率
        """
        try:
            if not model_probabilities:
                raise ValueError("没有可用的模型预测结果")
            
            # 获取样本数量
            sample_size = len(list(model_probabilities.values())[0])
            
            if self.strategy == 'weighted_voting':
                ensemble_probs = self._weighted_voting(model_probabilities)
            elif self.strategy == 'simple_voting':
                ensemble_probs = self._simple_voting(model_probabilities)
            elif self.strategy == 'confidence_based':
                ensemble_probs = self._confidence_based_voting(model_probabilities)
            else:
                # 默认使用加权投票
                ensemble_probs = self._weighted_voting(model_probabilities)
            
            # 根据阈值生成预测结果
            ensemble_predictions = (ensemble_probs >= threshold).astype(int)
            
            return ensemble_predictions, ensemble_probs
            
        except Exception as e:
            logger.error(f"集成预测失败: {e}")
            # 返回默认值
            sample_size = len(list(model_probabilities.values())[0]) if model_probabilities else 1
            return np.zeros(sample_size), np.zeros(sample_size)
    
    def _weighted_voting(self, model_probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """
        加权投票集成
        
        Args:
            model_probabilities: 各模型的预测概率
            
        Returns:
            集成概率
        """
        ensemble_probs = np.zeros(len(list(model_probabilities.values())[0]))
        total_weight = 0
        
        for model_name, probs in model_probabilities.items():
            weight = self.weights.get(model_name.lower(), 1.0)
            ensemble_probs += weight * probs
            total_weight += weight
        
        if total_weight > 0:
            ensemble_probs /= total_weight
        
        return ensemble_probs
    
    def _simple_voting(self, model_probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """
        简单平均投票
        
        Args:
            model_probabilities: 各模型的预测概率
            
        Returns:
            集成概率
        """
        probs_array = np.array(list(model_probabilities.values()))
        return np.mean(probs_array, axis=0)
    
    def _confidence_based_voting(self, model_probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """
        基于置信度的投票
        
        Args:
            model_probabilities: 各模型的预测概率
            
        Returns:
            集成概率
        """
        ensemble_probs = np.zeros(len(list(model_probabilities.values())[0]))
        
        for i in range(len(ensemble_probs)):
            # 计算每个样本的置信度权重
            confidences = []
            probs = []
            
            for model_name, model_probs in model_probabilities.items():
                prob = model_probs[i]
                # 置信度定义为距离0.5的距离
                confidence = abs(prob - 0.5) * 2
                confidences.append(confidence)
                probs.append(prob)
            
            # 基于置信度加权
            if sum(confidences) > 0:
                weights = np.array(confidences) / sum(confidences)
                ensemble_probs[i] = np.sum(np.array(probs) * weights)
            else:
                ensemble_probs[i] = np.mean(probs)
        
        return ensemble_probs
    
    def get_model_contributions(self, model_probabilities: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        获取各模型的贡献度
        
        Args:
            model_probabilities: 各模型的预测概率
            
        Returns:
            模型贡献度字典
        """
        contributions = {}
        
        if self.strategy == 'weighted_voting':
            total_weight = sum(self.weights.get(name.lower(), 1.0) for name in model_probabilities.keys())
            for model_name in model_probabilities.keys():
                weight = self.weights.get(model_name.lower(), 1.0)
                contributions[model_name] = weight / total_weight if total_weight > 0 else 0
        else:
            # 简单平均
            num_models = len(model_probabilities)
            for model_name in model_probabilities.keys():
                contributions[model_name] = 1.0 / num_models if num_models > 0 else 0
        
        return contributions
    
    def analyze_agreement(self, model_probabilities: Dict[str, np.ndarray], threshold: float = 0.5) -> Dict[str, Any]:
        """
        分析模型间的一致性
        
        Args:
            model_probabilities: 各模型的预测概率
            threshold: 分类阈值
            
        Returns:
            一致性分析结果
        """
        try:
            # 转换为预测结果
            model_predictions = {}
            for model_name, probs in model_probabilities.items():
                model_predictions[model_name] = (probs >= threshold).astype(int)
            
            # 计算一致性
            predictions_array = np.array(list(model_predictions.values()))
            
            # 完全一致的样本比例
            full_agreement = np.mean(np.all(predictions_array == predictions_array[0], axis=0))
            
            # 多数一致的样本比例
            majority_votes = np.sum(predictions_array, axis=0)
            majority_agreement = np.mean((majority_votes == 0) | (majority_votes == len(model_predictions)))
            
            # 平均概率方差（衡量不确定性）
            probs_array = np.array(list(model_probabilities.values()))
            avg_variance = np.mean(np.var(probs_array, axis=0))
            
            return {
                'full_agreement_rate': round(full_agreement, 3),
                'majority_agreement_rate': round(majority_agreement, 3),
                'average_probability_variance': round(avg_variance, 3),
                'model_count': len(model_probabilities),
                'high_confidence_samples': np.sum(np.max(probs_array, axis=0) > 0.8),
                'low_confidence_samples': np.sum(np.max(probs_array, axis=0) < 0.6)
            }
            
        except Exception as e:
            logger.error(f"一致性分析失败: {e}")
            return {
                'full_agreement_rate': 0.0,
                'majority_agreement_rate': 0.0,
                'average_probability_variance': 0.0,
                'model_count': 0,
                'high_confidence_samples': 0,
                'low_confidence_samples': 0
            }
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        更新模型权重
        
        Args:
            new_weights: 新的权重字典
        """
        self.weights.update(new_weights)
        logger.info(f"更新模型权重: {new_weights}")
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """
        获取集成器信息
        
        Returns:
            集成器信息字典
        """
        return {
            'strategy': self.strategy,
            'weights': self.weights,
            'available_strategies': ['weighted_voting', 'simple_voting', 'confidence_based']
        }
