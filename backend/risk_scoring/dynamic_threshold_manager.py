"""
动态风险阈值管理器
基于数据分布自动计算风险阈值，确保合理的风险等级分布
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DynamicThresholdManager:
    """动态风险阈值管理器"""
    
    def __init__(self):
        """初始化动态阈值管理器"""
        # 目标风险分布比例
        self.target_distribution = {
            'low': 0.60,        # 60%低风险
            'medium': 0.25,     # 25%中风险  
            'high': 0.12,       # 12%高风险
            'critical': 0.03    # 3%极高风险
        }
        
        # 默认固定阈值（备用）
        self.default_thresholds = {
            'low': 40,
            'medium': 60, 
            'high': 80,
            'critical': 100
        }
        
        # 四分类映射
        self.four_class_mapping = {
            'low': 0, 'medium': 1, 'high': 2, 'critical': 3
        }
        
        # 优化参数
        self.max_iterations = 5
        self.convergence_threshold = 0.05
        
        logger.info("动态阈值管理器初始化完成")
    
    def optimize_thresholds_iteratively(self, risk_scores: List[float], 
                                      max_iterations: int = None) -> Dict[str, float]:
        """
        迭代优化阈值
        
        Args:
            risk_scores: 风险评分列表
            max_iterations: 最大迭代次数
            
        Returns:
            优化后的阈值
        """
        try:
            if not risk_scores or len(risk_scores) < 10:
                logger.warning("风险评分数据不足，使用默认阈值")
                return self.default_thresholds.copy()
            
            if max_iterations is None:
                max_iterations = self.max_iterations
            
            scores_array = np.array(risk_scores)
            best_thresholds = self.default_thresholds.copy()
            best_deviation = float('inf')
            
            for iteration in range(max_iterations):
                # 计算当前阈值下的分布
                current_thresholds = self._calculate_percentile_thresholds(scores_array)
                
                # 分析分布质量
                analysis = self.analyze_distribution(risk_scores, current_thresholds)
                current_deviation = analysis['total_deviation']
                
                if current_deviation < best_deviation:
                    best_deviation = current_deviation
                    best_thresholds = current_thresholds
                
                # 如果分布已经足够合理，停止迭代
                if analysis['is_reasonable']:
                    logger.info(f"阈值优化完成，迭代{iteration+1}次，偏差: {current_deviation:.3f}")
                    break
            
            return best_thresholds
            
        except Exception as e:
            logger.error(f"迭代优化阈值失败: {e}")
            return self.default_thresholds.copy()
    
    def _calculate_percentile_thresholds(self, scores_array: np.ndarray) -> Dict[str, float]:
        """基于百分位数计算阈值 - 修复版本"""
        try:
            # 基于目标分布计算百分位数
            low_percentile = self.target_distribution['low'] * 100
            medium_percentile = (self.target_distribution['low'] + self.target_distribution['medium']) * 100
            high_percentile = (self.target_distribution['low'] + self.target_distribution['medium'] + self.target_distribution['high']) * 100

            # 计算原始百分位数阈值
            raw_thresholds = {
                'low': float(np.percentile(scores_array, low_percentile)),
                'medium': float(np.percentile(scores_array, medium_percentile)),
                'high': float(np.percentile(scores_array, high_percentile)),
                'critical': 100.0
            }

            # 应用最小阈值约束，确保合理的风险分布 - 优化极高风险识别
            min_thresholds = {
                'low': 20,      # 低风险最低20分
                'medium': 40,   # 中风险最低40分
                'high': 65,     # 高风险最低65分 (降低门槛)
                'critical': 100
            }

            # 确保阈值递增且满足最小值要求
            adjusted_thresholds = {
                'low': max(raw_thresholds['low'], min_thresholds['low']),
                'medium': max(raw_thresholds['medium'], min_thresholds['medium']),
                'high': max(raw_thresholds['high'], min_thresholds['high']),
                'critical': 100.0
            }

            # 确保阈值严格递增
            if adjusted_thresholds['medium'] <= adjusted_thresholds['low']:
                adjusted_thresholds['medium'] = adjusted_thresholds['low'] + 10
            if adjusted_thresholds['high'] <= adjusted_thresholds['medium']:
                adjusted_thresholds['high'] = adjusted_thresholds['medium'] + 15

            logger.info(f"动态阈值: {adjusted_thresholds}")
            return adjusted_thresholds
            
        except Exception as e:
            logger.warning(f"百分位数阈值计算失败: {e}")
            return self.default_thresholds.copy()
    
    def analyze_distribution(self, risk_scores: List[float], 
                           thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        分析风险分布质量
        
        Args:
            risk_scores: 风险评分列表
            thresholds: 阈值字典
            
        Returns:
            分布分析结果
        """
        try:
            if not risk_scores:
                return {'total_deviation': 1.0, 'is_reasonable': False}
            
            # 根据阈值分类
            distribution_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
            
            for score in risk_scores:
                if score <= thresholds['low']:
                    distribution_counts['low'] += 1
                elif score <= thresholds['medium']:
                    distribution_counts['medium'] += 1
                elif score <= thresholds['high']:
                    distribution_counts['high'] += 1
                else:
                    distribution_counts['critical'] += 1
            
            # 计算百分比
            total_samples = len(risk_scores)
            distribution_pct = {
                level: count / total_samples 
                for level, count in distribution_counts.items()
            }
            
            # 计算与目标分布的偏差
            deviation = {}
            total_deviation = 0
            for level in ['low', 'medium', 'high', 'critical']:
                dev = abs(distribution_pct[level] - self.target_distribution[level])
                deviation[level] = dev
                total_deviation += dev
            
            # 判断分布是否合理
            is_reasonable = total_deviation < self.convergence_threshold
            
            return {
                'distribution_counts': distribution_counts,
                'distribution_pct': distribution_pct,
                'target_distribution': self.target_distribution,
                'deviation': deviation,
                'total_deviation': total_deviation,
                'is_reasonable': is_reasonable,
                'thresholds_used': thresholds
            }
            
        except Exception as e:
            logger.error(f"分布分析失败: {e}")
            return {'total_deviation': 1.0, 'is_reasonable': False}
    
    def optimize_thresholds_for_four_class(self, four_class_labels: List[int]) -> Dict[str, float]:
        """
        基于四分类标签优化阈值
        
        Args:
            four_class_labels: 四分类标签列表 (0=low, 1=medium, 2=high, 3=critical)
            
        Returns:
            优化后的阈值
        """
        try:
            if not four_class_labels or len(four_class_labels) < 10:
                logger.warning("四分类标签数据不足，使用默认阈值")
                return self.default_thresholds.copy()
            
            labels_array = np.array(four_class_labels)
            
            # 计算当前分布
            current_dist = np.bincount(labels_array, minlength=4) / len(labels_array)
            
            # 计算分布偏差
            target_dist = np.array([
                self.target_distribution['low'],
                self.target_distribution['medium'],
                self.target_distribution['high'],
                self.target_distribution['critical']
            ])
            
            deviation = np.abs(current_dist - target_dist)
            total_deviation = np.sum(deviation)
            
            logger.info(f"四分类分布偏差: {total_deviation:.3f}")
            
            # 如果分布已经很好，返回默认阈值
            if total_deviation < self.convergence_threshold:
                logger.info("四分类分布已达到目标，无需调整")
                return self.default_thresholds.copy()
            
            # 否则返回调整后的阈值
            return self._adjust_thresholds_for_distribution(current_dist, target_dist)
            
        except Exception as e:
            logger.error(f"四分类阈值优化失败: {e}")
            return self.default_thresholds.copy()
    
    def _adjust_thresholds_for_distribution(self, current_dist: np.ndarray, 
                                          target_dist: np.ndarray) -> Dict[str, float]:
        """调整阈值以改善分布"""
        try:
            # 简单的调整策略
            adjusted_thresholds = self.default_thresholds.copy()
            
            # 如果低风险比例过高，提高低风险阈值
            if current_dist[0] > target_dist[0]:
                adjusted_thresholds['low'] = min(adjusted_thresholds['low'] + 5, 50)
            
            # 如果高风险比例过低，降低高风险阈值
            if current_dist[2] < target_dist[2]:
                adjusted_thresholds['high'] = max(adjusted_thresholds['high'] - 5, 70)
            
            return adjusted_thresholds
            
        except Exception as e:
            logger.warning(f"分布调整失败: {e}")
            return self.default_thresholds.copy()
    
    def analyze_four_class_distribution(self, four_class_labels: List[int]) -> Dict[str, Any]:
        """分析四分类标签分布"""
        try:
            if not four_class_labels:
                return {'success': False, 'error': '标签数据为空'}
            
            labels_array = np.array(four_class_labels)
            
            # 计算分布
            counts = np.bincount(labels_array, minlength=4)
            total = len(labels_array)
            distribution = counts / total
            
            # 计算与目标分布的偏差
            target_dist = np.array([
                self.target_distribution['low'],
                self.target_distribution['medium'],
                self.target_distribution['high'],
                self.target_distribution['critical']
            ])
            
            deviation = np.abs(distribution - target_dist)
            total_deviation = np.sum(deviation)
            
            # 分布质量评估
            is_reasonable = total_deviation < self.convergence_threshold
            
            result = {
                'success': True,
                'distribution': {
                    'low': {'count': int(counts[0]), 'percentage': float(distribution[0] * 100)},
                    'medium': {'count': int(counts[1]), 'percentage': float(distribution[1] * 100)},
                    'high': {'count': int(counts[2]), 'percentage': float(distribution[2] * 100)},
                    'critical': {'count': int(counts[3]), 'percentage': float(distribution[3] * 100)}
                },
                'target_distribution': {
                    'low': float(self.target_distribution['low'] * 100),
                    'medium': float(self.target_distribution['medium'] * 100),
                    'high': float(self.target_distribution['high'] * 100),
                    'critical': float(self.target_distribution['critical'] * 100)
                },
                'deviation': {
                    'low': float(deviation[0]),
                    'medium': float(deviation[1]),
                    'high': float(deviation[2]),
                    'critical': float(deviation[3])
                },
                'total_deviation': float(total_deviation),
                'is_reasonable': is_reasonable,
                'quality_grade': self._get_distribution_quality_grade(total_deviation)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"四分类分布分析失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_distribution_quality_grade(self, total_deviation: float) -> str:
        """获取分布质量等级"""
        if total_deviation < 0.05:
            return 'A'  # 优秀
        elif total_deviation < 0.1:
            return 'B'  # 良好
        elif total_deviation < 0.2:
            return 'C'  # 一般
        else:
            return 'D'  # 需要改进
