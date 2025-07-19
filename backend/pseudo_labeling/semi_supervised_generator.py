"""
半监督四分类标签生成器
利用原始欺诈标签作为锚点，生成四级风险分类标签
专门针对电商场景优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入配置管理
try:
    from config.optimization_config import optimization_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    optimization_config = None


class SemiSupervisedLabelGenerator:
    """半监督四分类标签生成器"""
    
    def __init__(self):
        """初始化半监督标签生成器"""
        self.target_classes = ['low', 'medium', 'high', 'extreme']
        self.class_mapping = {'low': 0, 'medium': 1, 'high': 2, 'extreme': 3}
        
        # 加载配置
        self.config = self._load_config()
        
        # 权重配置
        self.original_label_weight = self.config.get('original_label_weight', 0.6)
        self.clustering_weight = self.config.get('clustering_weight', 0.25)
        self.rule_weight = self.config.get('rule_weight', 0.15)
        
        # 目标分布
        self.target_distribution = self.config.get('target_distribution', {
            'low': 0.60, 'medium': 0.25, 'high': 0.12, 'extreme': 0.03
        })
        
        logger.info("半监督四分类标签生成器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if CONFIG_AVAILABLE and optimization_config:
            return optimization_config.get_risk_scoring_config().get('label_generation', {})
        else:
            return {
                'original_label_weight': 0.6,
                'clustering_weight': 0.25,
                'rule_weight': 0.15,
                'target_distribution': {
                    'low': 0.60, 'medium': 0.25, 'high': 0.12, 'extreme': 0.03
                }
            }
    
    def generate_four_class_labels(self, data: pd.DataFrame, 
                                 original_labels: Optional[pd.Series] = None,
                                 cluster_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        生成四分类标签
        
        Args:
            data: 输入数据
            original_labels: 原始欺诈标签（1=欺诈，0=正常）
            cluster_results: 聚类结果
            
        Returns:
            四分类标签生成结果
        """
        try:
            if data is None or data.empty:
                logger.error("输入数据为空")
                return self._empty_result()
            
            start_time = datetime.now()
            logger.info(f"开始生成四分类标签，数据量: {len(data)}")
            
            # 1. 检查原始标签
            if original_labels is None and 'is_fraudulent' in data.columns:
                original_labels = data['is_fraudulent']
                logger.info("使用数据中的is_fraudulent列作为原始标签")
            
            # 2. 提取电商特征
            ecommerce_features = self._extract_ecommerce_features(data)
            
            # 3. 生成基础四分类标签
            if original_labels is not None:
                base_labels = self._generate_labels_with_original(
                    data, original_labels, ecommerce_features
                )
            else:
                base_labels = self._generate_labels_without_original(
                    data, ecommerce_features, cluster_results
                )
            
            # 4. 聚类辅助校正
            if cluster_results is not None:
                corrected_labels = self._apply_clustering_correction(
                    base_labels, cluster_results, data
                )
            else:
                corrected_labels = base_labels
            
            # 5. 业务规则校正
            final_labels = self._apply_business_rules_correction(
                corrected_labels, data, ecommerce_features
            )
            
            # 6. 分布调整
            adjusted_labels, confidences = self._adjust_distribution(
                final_labels, data
            )
            
            # 7. 生成结果
            result = self._generate_result(
                adjusted_labels, confidences, data, original_labels
            )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            result['generation_time'] = generation_time
            
            logger.info(f"四分类标签生成完成，耗时: {generation_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"四分类标签生成失败: {e}")
            return self._empty_result()
    
    def _extract_ecommerce_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """提取电商特定特征"""
        try:
            features = {}
            
            # 基础特征
            features['transaction_amount'] = data.get('transaction_amount', pd.Series([0] * len(data))).values
            features['quantity'] = data.get('quantity', pd.Series([1] * len(data))).values
            features['customer_age'] = data.get('customer_age', pd.Series([30] * len(data))).values
            features['account_age_days'] = data.get('account_age_days', pd.Series([365] * len(data))).values
            features['transaction_hour'] = data.get('transaction_hour', pd.Series([12] * len(data))).values
            
            # 电商特定特征
            features['avg_order_value'] = features['transaction_amount'] / features['quantity']
            features['is_large_order'] = (features['quantity'] > 5).astype(int)
            features['is_high_value'] = (features['transaction_amount'] > 1000).astype(int)
            features['is_new_account'] = (features['account_age_days'] < 30).astype(int)
            features['is_night_transaction'] = np.isin(features['transaction_hour'], [0,1,2,3,4,5,22,23]).astype(int)
            features['is_young_customer'] = (features['customer_age'] < 25).astype(int)
            features['is_old_customer'] = (features['customer_age'] > 65).astype(int)
            
            return features
            
        except Exception as e:
            logger.warning(f"电商特征提取失败: {e}")
            return {}
    
    def _generate_labels_with_original(self, data: pd.DataFrame, 
                                     original_labels: pd.Series,
                                     features: Dict[str, np.ndarray]) -> np.ndarray:
        """基于原始标签生成四分类标签"""
        try:
            labels = np.zeros(len(data), dtype=int)
            
            for i in range(len(data)):
                if original_labels.iloc[i] == 1:  # 原始欺诈标签
                    # 基于特征细分高风险和极高风险
                    if self._is_extreme_fraud_pattern(features, i):
                        labels[i] = 3  # extreme
                    else:
                        labels[i] = 2  # high
                else:  # 原始正常标签
                    # 基于特征细分低风险和中风险
                    if self._has_suspicious_patterns(features, i):
                        labels[i] = 1  # medium
                    else:
                        labels[i] = 0  # low
            
            return labels
            
        except Exception as e:
            logger.warning(f"基于原始标签生成失败: {e}")
            return np.zeros(len(data), dtype=int)
    
    def _is_extreme_fraud_pattern(self, features: Dict[str, np.ndarray], idx: int) -> bool:
        """识别极高风险的欺诈模式"""
        try:
            extreme_indicators = [
                features['is_high_value'][idx] == 1,      # 大额交易
                features['is_new_account'][idx] == 1,     # 新账户
                features['is_night_transaction'][idx] == 1, # 夜间交易
                features['is_large_order'][idx] == 1,     # 大量购买
                features['is_young_customer'][idx] == 1,  # 年轻客户
            ]
            
            # 满足3个以上条件认为是极高风险
            return sum(extreme_indicators) >= 3
            
        except Exception as e:
            logger.warning(f"极高风险模式识别失败: {e}")
            return False
    
    def _has_suspicious_patterns(self, features: Dict[str, np.ndarray], idx: int) -> bool:
        """识别可疑模式（中风险）"""
        try:
            suspicious_indicators = [
                features['is_high_value'][idx] == 1,      # 大额交易
                features['is_new_account'][idx] == 1,     # 新账户
                features['is_night_transaction'][idx] == 1, # 夜间交易
                features['is_large_order'][idx] == 1,     # 大量购买
                features['is_young_customer'][idx] == 1 or features['is_old_customer'][idx] == 1,  # 异常年龄
            ]
            
            # 满足1-2个条件认为是中风险
            return 1 <= sum(suspicious_indicators) <= 2
            
        except Exception as e:
            logger.warning(f"可疑模式识别失败: {e}")
            return False
    
    def _generate_labels_without_original(self, data: pd.DataFrame,
                                        features: Dict[str, np.ndarray],
                                        cluster_results: Optional[Dict] = None) -> np.ndarray:
        """无原始标签时生成四分类标签"""
        try:
            labels = np.zeros(len(data), dtype=int)
            
            # 基于特征规则生成标签
            for i in range(len(data)):
                if self._is_extreme_fraud_pattern(features, i):
                    labels[i] = 3  # extreme
                elif self._is_high_risk_pattern(features, i):
                    labels[i] = 2  # high
                elif self._has_suspicious_patterns(features, i):
                    labels[i] = 1  # medium
                else:
                    labels[i] = 0  # low
            
            return labels
            
        except Exception as e:
            logger.warning(f"无原始标签生成失败: {e}")
            return np.zeros(len(data), dtype=int)
    
    def _is_high_risk_pattern(self, features: Dict[str, np.ndarray], idx: int) -> bool:
        """识别高风险模式"""
        try:
            high_risk_indicators = [
                features['is_high_value'][idx] == 1 and features['is_new_account'][idx] == 1,
                features['is_large_order'][idx] == 1 and features['is_night_transaction'][idx] == 1,
                features['transaction_amount'][idx] > 2000,
                features['quantity'][idx] > 10
            ]
            
            return any(high_risk_indicators)
            
        except Exception as e:
            logger.warning(f"高风险模式识别失败: {e}")
            return False
    
    def _apply_clustering_correction(self, labels: np.ndarray,
                                   cluster_results: Dict,
                                   data: pd.DataFrame) -> np.ndarray:
        """应用聚类校正"""
        try:
            corrected_labels = labels.copy()
            cluster_labels = cluster_results.get('cluster_labels', [])
            cluster_risk_mapping = cluster_results.get('cluster_risk_mapping', {})

            if not cluster_labels or not cluster_risk_mapping:
                return corrected_labels

            for i, cluster_id in enumerate(cluster_labels):
                if i >= len(corrected_labels):
                    break

                cluster_info = cluster_risk_mapping.get(cluster_id, {})
                cluster_risk_level = cluster_info.get('risk_level', 'low')

                # 聚类结果与基础标签的加权融合
                cluster_class = self.class_mapping.get(cluster_risk_level, 0)
                base_class = corrected_labels[i]

                # 加权平均
                weighted_class = (base_class * self.original_label_weight +
                                cluster_class * self.clustering_weight)
                corrected_labels[i] = int(round(weighted_class))

                # 确保在有效范围内
                corrected_labels[i] = max(0, min(3, corrected_labels[i]))

            return corrected_labels

        except Exception as e:
            logger.warning(f"聚类校正失败: {e}")
            return labels

    def _apply_business_rules_correction(self, labels: np.ndarray,
                                       data: pd.DataFrame,
                                       features: Dict[str, np.ndarray]) -> np.ndarray:
        """应用业务规则校正"""
        try:
            corrected_labels = labels.copy()

            for i in range(len(corrected_labels)):
                # 规则1：大额新账户夜间交易 -> 至少高风险
                if (features['is_high_value'][i] == 1 and
                    features['is_new_account'][i] == 1 and
                    features['is_night_transaction'][i] == 1):
                    corrected_labels[i] = max(corrected_labels[i], 2)  # 至少high

                # 规则2：极大额交易 -> 至少中风险
                if features['transaction_amount'][i] > 5000:
                    corrected_labels[i] = max(corrected_labels[i], 1)  # 至少medium

                # 规则3：异常年龄大量购买 -> 提升风险等级
                if ((features['is_young_customer'][i] == 1 or features['is_old_customer'][i] == 1) and
                    features['is_large_order'][i] == 1):
                    corrected_labels[i] = min(corrected_labels[i] + 1, 3)

            return corrected_labels

        except Exception as e:
            logger.warning(f"业务规则校正失败: {e}")
            return labels

    def _adjust_distribution(self, labels: np.ndarray,
                           data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """调整标签分布以接近目标分布"""
        try:
            adjusted_labels = labels.copy()

            # 计算当前分布
            current_dist = np.bincount(labels, minlength=4) / len(labels)
            target_dist = np.array([
                self.target_distribution['low'],
                self.target_distribution['medium'],
                self.target_distribution['high'],
                self.target_distribution['extreme']
            ])

            # 计算需要调整的样本数
            total_samples = len(labels)
            target_counts = (target_dist * total_samples).astype(int)
            current_counts = np.bincount(labels, minlength=4)

            # 生成置信度（基于分布调整的程度）
            confidences = np.ones(len(labels)) * 0.7

            # 如果分布偏差较大，进行调整
            for class_idx in range(4):
                if current_counts[class_idx] > target_counts[class_idx]:
                    # 该类别样本过多，降级一些样本
                    excess = current_counts[class_idx] - target_counts[class_idx]
                    class_indices = np.where(adjusted_labels == class_idx)[0]

                    if len(class_indices) > 0 and excess > 0:
                        # 随机选择要降级的样本
                        downgrade_indices = np.random.choice(
                            class_indices,
                            min(excess, len(class_indices)),
                            replace=False
                        )

                        for idx in downgrade_indices:
                            if class_idx > 0:  # 不能再降级low类别
                                adjusted_labels[idx] = class_idx - 1
                                confidences[idx] = 0.5  # 降低置信度

            return adjusted_labels, confidences

        except Exception as e:
            logger.warning(f"分布调整失败: {e}")
            return labels, np.ones(len(labels)) * 0.7

    def _generate_result(self, labels: np.ndarray, confidences: np.ndarray,
                        data: pd.DataFrame, original_labels: Optional[pd.Series] = None) -> Dict[str, Any]:
        """生成最终结果"""
        try:
            # 转换为类别标签
            class_labels = [self.target_classes[label] for label in labels]

            # 计算分布
            distribution = {}
            for i, class_name in enumerate(self.target_classes):
                count = np.sum(labels == i)
                distribution[class_name] = {
                    'count': int(count),
                    'percentage': float(count / len(labels) * 100)
                }

            # 计算质量指标
            quality_metrics = self._calculate_quality_metrics(
                labels, confidences, original_labels
            )

            result = {
                'labels': labels.tolist(),
                'confidences': confidences.tolist(),
                'class_labels': class_labels,
                'distribution': distribution,
                'quality_metrics': quality_metrics,
                'target_distribution': self.target_distribution,
                'success': True,
                'total_samples': len(labels)
            }

            return result

        except Exception as e:
            logger.error(f"结果生成失败: {e}")
            return self._empty_result()

    def _calculate_quality_metrics(self, labels: np.ndarray,
                                 confidences: np.ndarray,
                                 original_labels: Optional[pd.Series] = None) -> Dict[str, Any]:
        """计算质量指标"""
        try:
            metrics = {
                'avg_confidence': float(np.mean(confidences)),
                'min_confidence': float(np.min(confidences)),
                'max_confidence': float(np.max(confidences)),
                'high_confidence_ratio': float(np.sum(confidences >= 0.8) / len(confidences))
            }

            # 如果有原始标签，计算一致性
            if original_labels is not None:
                # 将四分类映射回二分类进行对比
                binary_pred = (labels >= 2).astype(int)  # high和extreme映射为1
                original_binary = original_labels.values

                if len(binary_pred) == len(original_binary):
                    accuracy = np.mean(binary_pred == original_binary)
                    metrics['binary_accuracy'] = float(accuracy)

                    # 计算欺诈检测指标
                    tp = np.sum((binary_pred == 1) & (original_binary == 1))
                    fp = np.sum((binary_pred == 1) & (original_binary == 0))
                    fn = np.sum((binary_pred == 0) & (original_binary == 1))

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    metrics.update({
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1)
                    })

            return metrics

        except Exception as e:
            logger.warning(f"质量指标计算失败: {e}")
            return {'avg_confidence': 0.5}

    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'labels': [],
            'confidences': [],
            'class_labels': [],
            'distribution': {},
            'quality_metrics': {},
            'generation_time': 0,
            'success': False
        }
