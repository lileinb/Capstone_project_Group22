"""
特征选择器
从52个特征中选择最重要的15-20个核心特征，提升计算性能
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """特征选择器"""
    
    def __init__(self, target_features: int = 18):
        """
        初始化特征选择器
        
        Args:
            target_features: 目标特征数量
        """
        self.target_features = target_features
        self.selected_features = []
        self.feature_importance_scores = {}
        self.selection_methods = {}
        
        # 预定义的核心特征（业务重要性高）
        self.core_features = [
            'transaction_amount',
            'customer_age', 
            'account_age_days',
            'transaction_hour',
            'device_used',
            'payment_method',
            'shipping_address',
            'billing_address'
        ]
        
        # 特征类别分组
        self.feature_groups = {
            'amount_features': [
                'transaction_amount', 'amount_zscore', 'amount_log', 'amount_percentile',
                'amount_category', 'high_amount_flag'
            ],
            'time_features': [
                'transaction_hour', 'hour_category', 'is_weekend', 'is_holiday',
                'time_since_last_transaction', 'transaction_frequency'
            ],
            'account_features': [
                'customer_age', 'account_age_days', 'age_category', 'account_age_category',
                'customer_risk_score', 'account_activity_score'
            ],
            'device_payment_features': [
                'device_used', 'payment_method', 'device_risk_score', 'payment_risk_score',
                'device_payment_combination'
            ],
            'address_features': [
                'shipping_address', 'billing_address', 'address_match', 'address_risk_score',
                'shipping_country', 'billing_country'
            ],
            'product_features': [
                'product_category', 'quantity', 'category_risk_score', 'quantity_category',
                'electronics_flag', 'high_risk_category_flag'
            ],
            'behavioral_features': [
                'transaction_velocity', 'unusual_pattern_flag', 'behavior_consistency_score',
                'pattern_deviation_score', 'anomaly_score'
            ],
            'statistical_features': [
                'amount_zscore', 'age_zscore', 'account_age_zscore', 'statistical_outlier_score',
                'composite_risk_score', 'risk_interaction_score'
            ]
        }
    
    def select_features(self, data: pd.DataFrame, 
                       pseudo_labels: np.ndarray = None,
                       method: str = 'hybrid') -> List[str]:
        """
        选择最重要的特征
        
        Args:
            data: 特征数据
            pseudo_labels: 伪标签（可选）
            method: 选择方法 ('statistical', 'importance', 'hybrid')
            
        Returns:
            选择的特征列表
        """
        try:
            logger.info(f"开始特征选择，原始特征数: {len(data.columns)}")
            
            # 1. 确保核心特征被包含
            available_core_features = [f for f in self.core_features if f in data.columns]
            logger.info(f"可用核心特征: {len(available_core_features)}")
            
            # 2. 根据方法选择特征
            if method == 'statistical':
                selected_features = self._statistical_selection(data, pseudo_labels)
            elif method == 'importance':
                selected_features = self._importance_based_selection(data, pseudo_labels)
            else:  # hybrid
                selected_features = self._hybrid_selection(data, pseudo_labels)
            
            # 3. 确保核心特征被包含
            final_features = list(set(available_core_features + selected_features))
            
            # 4. 如果特征数量超过目标，进行进一步筛选
            if len(final_features) > self.target_features:
                final_features = self._final_selection(data, final_features, pseudo_labels)
            
            # 5. 如果特征数量不足，补充重要特征
            if len(final_features) < self.target_features:
                final_features = self._supplement_features(data, final_features)
            
            self.selected_features = final_features[:self.target_features]
            
            logger.info(f"特征选择完成，选择了 {len(self.selected_features)} 个特征")
            logger.info(f"选择的特征: {self.selected_features}")
            
            return self.selected_features
            
        except Exception as e:
            logger.error(f"特征选择失败: {e}")
            # 返回核心特征作为备选
            return [f for f in self.core_features if f in data.columns][:self.target_features]
    
    def _statistical_selection(self, data: pd.DataFrame, 
                             pseudo_labels: np.ndarray = None) -> List[str]:
        """基于统计方法的特征选择"""
        selected_features = []
        
        try:
            # 计算特征的统计重要性
            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # 方差筛选
            feature_variances = data[numeric_features].var()
            high_variance_features = feature_variances[feature_variances > 0.01].index.tolist()
            
            # 相关性分析
            if len(high_variance_features) > 1:
                correlation_matrix = data[high_variance_features].corr().abs()
                
                # 移除高度相关的特征
                upper_triangle = correlation_matrix.where(
                    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
                )
                
                to_drop = [column for column in upper_triangle.columns 
                          if any(upper_triangle[column] > 0.9)]
                
                selected_features = [f for f in high_variance_features if f not in to_drop]
            else:
                selected_features = high_variance_features
            
            # 如果有伪标签，使用监督方法
            if pseudo_labels is not None and len(np.unique(pseudo_labels)) > 1:
                try:
                    # 使用互信息进行特征选择
                    selector = SelectKBest(score_func=mutual_info_classif, 
                                         k=min(self.target_features, len(selected_features)))
                    
                    X_selected = data[selected_features].fillna(0)
                    selector.fit(X_selected, pseudo_labels)
                    
                    selected_indices = selector.get_support(indices=True)
                    selected_features = [selected_features[i] for i in selected_indices]
                    
                except Exception as e:
                    logger.warning(f"监督特征选择失败，使用无监督方法: {e}")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"统计特征选择失败: {e}")
            return []
    
    def _importance_based_selection(self, data: pd.DataFrame,
                                  pseudo_labels: np.ndarray = None) -> List[str]:
        """基于重要性的特征选择"""
        selected_features = []
        
        try:
            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if pseudo_labels is not None and len(np.unique(pseudo_labels)) > 1:
                # 使用随机森林计算特征重要性
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                
                X = data[numeric_features].fillna(0)
                rf.fit(X, pseudo_labels)
                
                # 获取特征重要性
                feature_importance = pd.DataFrame({
                    'feature': numeric_features,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                selected_features = feature_importance.head(self.target_features)['feature'].tolist()
                
                # 保存重要性分数
                self.feature_importance_scores = dict(zip(
                    feature_importance['feature'], 
                    feature_importance['importance']
                ))
                
            else:
                # 无监督情况下，基于业务规则选择
                selected_features = self._business_rule_selection(data)
            
            return selected_features
            
        except Exception as e:
            logger.error(f"重要性特征选择失败: {e}")
            return []
    
    def _hybrid_selection(self, data: pd.DataFrame,
                         pseudo_labels: np.ndarray = None) -> List[str]:
        """混合方法特征选择"""
        try:
            # 1. 统计方法选择
            statistical_features = self._statistical_selection(data, pseudo_labels)
            
            # 2. 重要性方法选择
            importance_features = self._importance_based_selection(data, pseudo_labels)
            
            # 3. 业务规则选择
            business_features = self._business_rule_selection(data)
            
            # 4. 合并并去重
            all_features = list(set(statistical_features + importance_features + business_features))
            
            # 5. 按重要性排序
            if self.feature_importance_scores:
                all_features.sort(key=lambda x: self.feature_importance_scores.get(x, 0), reverse=True)
            
            return all_features
            
        except Exception as e:
            logger.error(f"混合特征选择失败: {e}")
            return self._business_rule_selection(data)
    
    def _business_rule_selection(self, data: pd.DataFrame) -> List[str]:
        """基于业务规则的特征选择"""
        selected_features = []
        
        # 从每个特征组中选择最重要的特征
        for group_name, features in self.feature_groups.items():
            available_features = [f for f in features if f in data.columns]
            
            if group_name == 'amount_features':
                # 金额特征：选择2-3个最重要的
                priority = ['transaction_amount', 'amount_zscore', 'amount_percentile']
                selected = [f for f in priority if f in available_features][:3]
                selected_features.extend(selected)
                
            elif group_name == 'time_features':
                # 时间特征：选择2个最重要的
                priority = ['transaction_hour', 'hour_category']
                selected = [f for f in priority if f in available_features][:2]
                selected_features.extend(selected)
                
            elif group_name == 'account_features':
                # 账户特征：选择3个最重要的
                priority = ['customer_age', 'account_age_days', 'customer_risk_score']
                selected = [f for f in priority if f in available_features][:3]
                selected_features.extend(selected)
                
            elif group_name == 'device_payment_features':
                # 设备支付特征：选择2-3个
                priority = ['device_used', 'payment_method', 'device_payment_combination']
                selected = [f for f in priority if f in available_features][:3]
                selected_features.extend(selected)
                
            elif group_name == 'address_features':
                # 地址特征：选择2个
                priority = ['address_match', 'address_risk_score']
                selected = [f for f in priority if f in available_features][:2]
                selected_features.extend(selected)
                
            else:
                # 其他特征组：选择1-2个最重要的
                selected_features.extend(available_features[:2])
        
        return list(set(selected_features))
    
    def _final_selection(self, data: pd.DataFrame, features: List[str],
                        pseudo_labels: np.ndarray = None) -> List[str]:
        """最终特征选择"""
        if len(features) <= self.target_features:
            return features
        
        # 优先保留核心特征
        core_in_features = [f for f in features if f in self.core_features]
        other_features = [f for f in features if f not in self.core_features]
        
        # 如果有重要性分数，按分数排序
        if self.feature_importance_scores:
            other_features.sort(key=lambda x: self.feature_importance_scores.get(x, 0), reverse=True)
        
        # 选择特征
        remaining_slots = self.target_features - len(core_in_features)
        final_features = core_in_features + other_features[:remaining_slots]
        
        return final_features
    
    def _supplement_features(self, data: pd.DataFrame, current_features: List[str]) -> List[str]:
        """补充特征到目标数量"""
        all_features = data.columns.tolist()
        available_features = [f for f in all_features if f not in current_features]
        
        # 优先添加核心特征
        core_to_add = [f for f in self.core_features if f in available_features]
        
        # 然后添加其他特征
        other_to_add = [f for f in available_features if f not in core_to_add]
        
        needed = self.target_features - len(current_features)
        to_add = (core_to_add + other_to_add)[:needed]
        
        return current_features + to_add
    
    def get_feature_importance_report(self) -> Dict[str, Any]:
        """获取特征重要性报告"""
        return {
            'selected_features': self.selected_features,
            'feature_count': len(self.selected_features),
            'target_count': self.target_features,
            'importance_scores': self.feature_importance_scores,
            'core_features_included': [f for f in self.selected_features if f in self.core_features],
            'selection_summary': {
                'total_selected': len(self.selected_features),
                'core_features': len([f for f in self.selected_features if f in self.core_features]),
                'other_features': len([f for f in self.selected_features if f not in self.core_features])
            }
        }
