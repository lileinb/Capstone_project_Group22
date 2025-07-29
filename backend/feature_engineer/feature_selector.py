"""
Feature Selector
Select the most important 15-20 core features from 52 features to improve computational performance
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """Feature Selector"""

    def __init__(self, target_features: int = 18):
        """
        Initialize feature selector

        Args:
            target_features: Target number of features
        """
        self.target_features = target_features
        self.selected_features = []
        self.feature_importance_scores = {}
        self.selection_methods = {}
        
        # Predefined core features (high business importance) - 优化版
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

        # 聚类友好特征（高区分度）
        self.clustering_friendly_features = [
            # 金额相关
            'amount_zscore', 'amount_percentile', 'amount_log', 'amount_rank',
            'is_large_amount', 'amount_deviation', 'amount_risk_score',
            # 时间相关
            'hour_risk_score', 'is_night_transaction', 'is_weekend',
            'time_risk_score', 'is_off_hours', 'is_deep_night',
            # 账户相关
            'account_age_risk_score', 'is_new_account', 'is_very_new_account',
            'account_maturity', 'customer_risk_score',
            # 复合特征
            'composite_risk_score', 'anomaly_score', 'risk_interaction',
            'behavior_consistency_score', 'pattern_deviation_score'
        ]

        # Feature category grouping
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

    def select_clustering_optimized_features(self, data: pd.DataFrame,
                                           max_features: int = 12) -> List[str]:
        """
        专门为聚类优化的特征选择

        Args:
            data: 特征数据
            max_features: 最大特征数量

        Returns:
            优化的特征列表
        """
        try:
            logger.info(f"🎯 开始聚类优化特征选择，原始特征数: {len(data.columns)}")

            # 第一步：创建增强特征
            enhanced_data = self._create_clustering_features(data.copy())

            # 第二步：特征质量评估
            feature_scores = self._evaluate_clustering_feature_quality(enhanced_data)

            # 第三步：智能特征组合选择
            optimal_features = self._select_optimal_clustering_combination(
                enhanced_data, feature_scores, max_features
            )

            logger.info(f"✅ 聚类优化特征选择完成，选择了 {len(optimal_features)} 个特征")
            logger.info(f"选择的特征: {optimal_features}")

            return optimal_features

        except Exception as e:
            logger.error(f"聚类优化特征选择失败: {e}")
            # 返回基础特征
            return [f for f in self.core_features if f in data.columns][:max_features]

    def _create_clustering_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建聚类友好的特征"""
        logger.info("🔧 创建聚类友好特征")

        # 金额特征增强
        if 'transaction_amount' in data.columns:
            # 基础统计特征
            data['amount_log'] = np.log1p(data['transaction_amount'])
            data['amount_sqrt'] = np.sqrt(data['transaction_amount'])
            data['amount_zscore'] = (data['transaction_amount'] - data['transaction_amount'].mean()) / data['transaction_amount'].std()
            data['amount_percentile'] = data['transaction_amount'].rank(pct=True)
            data['amount_rank'] = data['transaction_amount'].rank()

            # 异常检测特征
            Q1 = data['transaction_amount'].quantile(0.25)
            Q3 = data['transaction_amount'].quantile(0.75)
            IQR = Q3 - Q1
            data['is_amount_outlier'] = ((data['transaction_amount'] < (Q1 - 1.5 * IQR)) |
                                       (data['transaction_amount'] > (Q3 + 1.5 * IQR))).astype(int)
            data['amount_deviation'] = abs(data['transaction_amount'] - data['transaction_amount'].median())

            # 分类特征
            data['is_large_amount'] = (data['transaction_amount'] > data['transaction_amount'].quantile(0.8)).astype(int)
            data['is_small_amount'] = (data['transaction_amount'] < data['transaction_amount'].quantile(0.2)).astype(int)

        # 时间特征增强
        if 'transaction_hour' in data.columns:
            # 时间分类
            data['is_night_transaction'] = data['transaction_hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
            data['is_business_hours'] = data['transaction_hour'].isin([9, 10, 11, 12, 13, 14, 15, 16, 17]).astype(int)
            data['is_evening'] = data['transaction_hour'].isin([18, 19, 20, 21]).astype(int)
            data['is_early_morning'] = data['transaction_hour'].isin([6, 7, 8]).astype(int)

            # 风险评分
            risk_hours = {0: 3, 1: 3, 2: 3, 3: 3, 4: 2, 5: 2, 22: 2, 23: 3}
            data['hour_risk_score'] = data['transaction_hour'].map(risk_hours).fillna(1)

            # 周期性特征
            data['hour_sin'] = np.sin(2 * np.pi * data['transaction_hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['transaction_hour'] / 24)

        # 账户特征增强
        if 'account_age_days' in data.columns:
            data['account_age_log'] = np.log1p(data['account_age_days'])
            data['is_new_account'] = (data['account_age_days'] < 30).astype(int)
            data['is_very_new_account'] = (data['account_age_days'] < 7).astype(int)
            data['is_mature_account'] = (data['account_age_days'] > 365).astype(int)
            data['account_age_percentile'] = data['account_age_days'].rank(pct=True)

        # 客户特征增强
        if 'customer_age' in data.columns:
            data['customer_age_zscore'] = (data['customer_age'] - data['customer_age'].mean()) / data['customer_age'].std()
            data['is_young_customer'] = (data['customer_age'] < 25).astype(int)
            data['is_senior_customer'] = (data['customer_age'] > 65).astype(int)
            data['is_prime_age'] = ((data['customer_age'] >= 25) & (data['customer_age'] <= 45)).astype(int)

        # 复合特征
        risk_components = []
        if 'amount_zscore' in data.columns:
            risk_components.append('amount_zscore')
        if 'hour_risk_score' in data.columns:
            risk_components.append('hour_risk_score')
        if 'is_new_account' in data.columns:
            risk_components.append('is_new_account')

        if len(risk_components) >= 2:
            data['composite_risk'] = data[risk_components].sum(axis=1)
            data['risk_interaction'] = data[risk_components].prod(axis=1)

        logger.info(f"特征创建完成，当前特征数: {len(data.columns)}")
        return data

    def _evaluate_clustering_feature_quality(self, data: pd.DataFrame) -> Dict[str, float]:
        """评估特征对聚类的质量"""
        feature_scores = {}
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()

        # 排除标签和ID类特征
        exclude_patterns = ['is_fraudulent', '_id', 'customer_id', 'transaction_id']
        candidate_features = [f for f in numeric_features
                            if not any(pattern in f.lower() for pattern in exclude_patterns)]

        for feature in candidate_features:
            if feature not in data.columns:
                continue

            feature_data = data[feature].dropna()
            if len(feature_data) == 0:
                feature_scores[feature] = 0.0
                continue

            score = 0.0

            # 1. 方差评分 (25%) - 高方差更好
            variance = feature_data.var()
            if variance > 0:
                # 标准化方差评分
                variance_score = min(np.log1p(variance) / 10, 1.0)
                score += variance_score * 0.25

            # 2. 分布评分 (20%) - 偏向有区分度的分布
            try:
                # 使用四分位数范围评估分布
                q75, q25 = np.percentile(feature_data, [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    # IQR相对于标准差的比例
                    iqr_ratio = iqr / (feature_data.std() + 1e-8)
                    distribution_score = min(iqr_ratio / 2, 1.0)
                    score += distribution_score * 0.2
            except:
                pass

            # 3. 唯一值评分 (15%) - 适中的唯一值数量最好
            n_unique = feature_data.nunique()
            total_samples = len(feature_data)
            unique_ratio = n_unique / total_samples

            if 0.01 <= unique_ratio <= 0.8:  # 理想范围
                unique_score = 1.0
            elif unique_ratio > 0.8:  # 太多唯一值
                unique_score = max(0.2, 1.0 - (unique_ratio - 0.8) * 2)
            else:  # 太少唯一值
                unique_score = unique_ratio / 0.01

            score += unique_score * 0.15

            # 4. 特征类型奖励 (25%)
            type_score = 0.0
            if any(suffix in feature for suffix in ['_zscore', '_percentile', '_rank']):
                type_score = 1.0  # 标准化特征最好
            elif any(suffix in feature for suffix in ['_log', '_sqrt', '_deviation']):
                type_score = 0.8  # 变换特征很好
            elif any(prefix in feature for prefix in ['is_', 'has_']):
                type_score = 0.6  # 二元特征不错
            elif any(suffix in feature for suffix in ['_score', '_risk']):
                type_score = 0.9  # 评分特征很好
            elif feature in self.core_features:
                type_score = 0.7  # 核心特征不错
            else:
                type_score = 0.3  # 其他特征一般

            score += type_score * 0.25

            # 5. 聚类友好性奖励 (15%)
            clustering_bonus = 0.0
            if feature in self.clustering_friendly_features:
                clustering_bonus = 1.0
            elif any(pattern in feature for pattern in ['composite', 'interaction', 'combined']):
                clustering_bonus = 0.8

            score += clustering_bonus * 0.15

            feature_scores[feature] = min(score, 1.0)  # 限制在[0,1]范围内

        return feature_scores

    def _select_optimal_clustering_combination(self, data: pd.DataFrame,
                                             feature_scores: Dict[str, float],
                                             max_features: int) -> List[str]:
        """选择最优的特征组合"""
        # 按分数排序
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        selected_features = []
        correlation_threshold = 0.85  # 相关性阈值

        # 确保包含核心特征中的最佳特征
        core_available = [f for f in self.core_features if f in data.columns]
        for feature in core_available[:3]:  # 最多3个核心特征
            if len(selected_features) < max_features:
                selected_features.append(feature)

        # 添加高分特征
        for feature, score in sorted_features:
            if len(selected_features) >= max_features:
                break

            if feature in selected_features:
                continue

            if score < 0.4:  # 分数太低跳过
                continue

            # 检查相关性
            is_redundant = False
            for selected_feature in selected_features:
                if feature in data.columns and selected_feature in data.columns:
                    try:
                        corr = abs(data[feature].corr(data[selected_feature]))
                        if corr > correlation_threshold:
                            is_redundant = True
                            break
                    except:
                        continue

            if not is_redundant:
                selected_features.append(feature)

        # 确保特征数量合理
        if len(selected_features) < 4:
            # 添加备用特征
            backup_features = [f for f in data.select_dtypes(include=[np.number]).columns
                             if f not in selected_features and 'is_fraudulent' not in f]
            selected_features.extend(backup_features[:4-len(selected_features)])

        return selected_features[:max_features]

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
