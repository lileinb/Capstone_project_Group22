#!/usr/bin/env python3
"""
智能聚类优化器
自动化特征选择、参数调优、风险阈值优化
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import logging
from typing import Dict, List, Tuple, Any, Optional
import itertools

logger = logging.getLogger(__name__)

class IntelligentClusterOptimizer:
    """智能聚类优化器"""
    
    def __init__(self):
        """初始化优化器"""
        self.scaler = StandardScaler()
        self.best_config = None
        self.optimization_history = []
        
        # 特征重要性权重
        self.feature_importance = {
            'transaction_amount': 0.25,
            'customer_age': 0.15,
            'account_age_days': 0.20,
            'transaction_hour': 0.15,
            'quantity': 0.10,
            'is_fraudulent': 0.15  # 用于特征选择，不用于聚类
        }
    
    def auto_optimize_clustering(self, data: pd.DataFrame, 
                               target_risk_distribution: Dict[str, float] = None) -> Dict[str, Any]:
        """
        自动优化聚类
        
        Args:
            data: 输入数据
            target_risk_distribution: 目标风险分布 {'low': 0.4, 'medium': 0.3, 'high': 0.2, 'critical': 0.1}
        
        Returns:
            最优聚类配置和结果
        """
        logger.info("🚀 开始智能聚类优化")
        
        if target_risk_distribution is None:
            target_risk_distribution = {'low': 0.5, 'medium': 0.3, 'high': 0.15, 'critical': 0.05}
        
        try:
            # 第1步：数据预处理和特征工程
            processed_data = self._advanced_feature_engineering(data)
            
            # 第2步：智能特征选择
            optimal_features = self._intelligent_feature_selection(processed_data)
            
            # 第3步：聚类算法和参数优化
            best_clustering = self._optimize_clustering_parameters(processed_data[optimal_features])
            
            # 第4步：风险阈值自适应调整
            optimal_thresholds = self._adaptive_risk_threshold_optimization(
                best_clustering, processed_data, target_risk_distribution
            )
            
            # 第5步：生成最终结果
            final_result = self._generate_optimized_result(
                best_clustering, optimal_thresholds, optimal_features, processed_data
            )
            
            logger.info("✅ 智能聚类优化完成")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ 智能聚类优化失败: {e}")
            return self._fallback_clustering(data)
    
    def _advanced_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """高级特征工程"""
        logger.info("🔧 执行高级特征工程")
        
        processed_data = data.copy()
        
        # 基础特征标准化
        numeric_features = ['transaction_amount', 'quantity', 'customer_age', 'account_age_days']
        for feature in numeric_features:
            if feature in processed_data.columns:
                # Z-score标准化
                processed_data[f'{feature}_zscore'] = (
                    processed_data[feature] - processed_data[feature].mean()
                ) / processed_data[feature].std()
                
                # 分位数特征
                processed_data[f'{feature}_percentile'] = processed_data[feature].rank(pct=True)
        
        # 时间特征增强
        if 'transaction_hour' in processed_data.columns:
            processed_data['is_night_transaction'] = (
                (processed_data['transaction_hour'] >= 22) | 
                (processed_data['transaction_hour'] <= 6)
            ).astype(int)
            
            processed_data['is_business_hour'] = (
                (processed_data['transaction_hour'] >= 9) & 
                (processed_data['transaction_hour'] <= 17)
            ).astype(int)
        
        # 账户风险特征
        if 'account_age_days' in processed_data.columns:
            processed_data['is_new_account'] = (processed_data['account_age_days'] < 30).astype(int)
            processed_data['is_very_new_account'] = (processed_data['account_age_days'] < 7).astype(int)
            processed_data['account_age_risk_score'] = np.where(
                processed_data['account_age_days'] < 30, 
                100 - processed_data['account_age_days'] * 3, 
                10
            )
        
        # 交易金额风险特征
        if 'transaction_amount' in processed_data.columns:
            amount_q95 = processed_data['transaction_amount'].quantile(0.95)
            amount_q75 = processed_data['transaction_amount'].quantile(0.75)
            
            processed_data['is_high_amount'] = (
                processed_data['transaction_amount'] > amount_q95
            ).astype(int)
            
            processed_data['is_medium_amount'] = (
                (processed_data['transaction_amount'] > amount_q75) & 
                (processed_data['transaction_amount'] <= amount_q95)
            ).astype(int)
            
            processed_data['amount_risk_score'] = np.where(
                processed_data['transaction_amount'] > amount_q95, 80,
                np.where(processed_data['transaction_amount'] > amount_q75, 40, 10)
            )
        
        # 组合风险特征
        if all(col in processed_data.columns for col in ['is_new_account', 'is_high_amount']):
            processed_data['high_risk_combination'] = (
                processed_data['is_new_account'] & processed_data['is_high_amount']
            ).astype(int)
        
        if all(col in processed_data.columns for col in ['is_night_transaction', 'is_high_amount']):
            processed_data['suspicious_pattern'] = (
                processed_data['is_night_transaction'] & processed_data['is_high_amount']
            ).astype(int)
        
        logger.info(f"✅ 特征工程完成，生成 {len(processed_data.columns)} 个特征")
        return processed_data
    
    def _intelligent_feature_selection(self, data: pd.DataFrame) -> List[str]:
        """智能特征选择"""
        logger.info("🎯 执行智能特征选择")
        
        # 排除非数值特征和标签
        exclude_features = ['is_fraudulent', 'payment_method', 'device']
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        candidate_features = [f for f in numeric_features if f not in exclude_features]
        
        if len(candidate_features) < 3:
            logger.warning("可用特征太少，使用所有数值特征")
            return candidate_features
        
        # 方法1：基于方差的特征选择
        feature_variance = data[candidate_features].var()
        high_variance_features = feature_variance[feature_variance > 0.01].index.tolist()
        
        # 方法2：基于相关性的特征选择
        if 'is_fraudulent' in data.columns:
            correlations = data[candidate_features].corrwith(data['is_fraudulent']).abs()
            high_corr_features = correlations.nlargest(min(8, len(candidate_features))).index.tolist()
        else:
            high_corr_features = candidate_features[:8]
        
        # 方法3：基于聚类友好性的特征选择
        clustering_friendly_features = []
        for feature in candidate_features:
            if feature.endswith('_zscore') or feature.endswith('_percentile') or feature.endswith('_score'):
                clustering_friendly_features.append(feature)
        
        # 综合选择
        selected_features = list(set(high_variance_features + high_corr_features + clustering_friendly_features))
        
        # 确保至少有5个特征
        if len(selected_features) < 5:
            selected_features = candidate_features[:min(8, len(candidate_features))]
        
        # 限制最多12个特征避免维度诅咒
        selected_features = selected_features[:12]
        
        logger.info(f"✅ 选择了 {len(selected_features)} 个最优特征: {selected_features}")
        return selected_features
    
    def _optimize_clustering_parameters(self, feature_data: pd.DataFrame) -> Dict[str, Any]:
        """优化聚类算法和参数"""
        logger.info("⚙️ 优化聚类算法和参数")
        
        # 标准化数据
        scaled_data = self.scaler.fit_transform(feature_data)
        
        best_score = -1
        best_config = None
        
        # 测试不同的聚类配置
        configurations = [
            # KMeans配置
            {'algorithm': 'kmeans', 'n_clusters': 3, 'init': 'k-means++', 'n_init': 20},
            {'algorithm': 'kmeans', 'n_clusters': 4, 'init': 'k-means++', 'n_init': 20},
            {'algorithm': 'kmeans', 'n_clusters': 5, 'init': 'k-means++', 'n_init': 20},
            {'algorithm': 'kmeans', 'n_clusters': 6, 'init': 'k-means++', 'n_init': 20},
            
            # DBSCAN配置
            {'algorithm': 'dbscan', 'eps': 0.5, 'min_samples': 10},
            {'algorithm': 'dbscan', 'eps': 0.8, 'min_samples': 15},
            {'algorithm': 'dbscan', 'eps': 1.0, 'min_samples': 20},
        ]
        
        for config in configurations:
            try:
                if config['algorithm'] == 'kmeans':
                    model = KMeans(
                        n_clusters=config['n_clusters'],
                        init=config['init'],
                        n_init=config['n_init'],
                        random_state=42
                    )
                    labels = model.fit_predict(scaled_data)
                    
                elif config['algorithm'] == 'dbscan':
                    model = DBSCAN(
                        eps=config['eps'],
                        min_samples=config['min_samples']
                    )
                    labels = model.fit_predict(scaled_data)
                
                # 评估聚类质量
                if len(set(labels)) > 1 and -1 not in labels:  # 确保有多个有效聚类
                    silhouette = silhouette_score(scaled_data, labels)
                    calinski = calinski_harabasz_score(scaled_data, labels)
                    
                    # 综合评分 (轮廓系数权重更高)
                    combined_score = silhouette * 0.7 + (calinski / 1000) * 0.3
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_config = {
                            'config': config,
                            'model': model,
                            'labels': labels,
                            'silhouette_score': silhouette,
                            'calinski_score': calinski,
                            'combined_score': combined_score,
                            'n_clusters': len(set(labels))
                        }
                        
                        logger.info(f"新的最佳配置: {config}, 轮廓系数: {silhouette:.3f}")
                
            except Exception as e:
                logger.warning(f"配置 {config} 失败: {e}")
                continue
        
        if best_config is None:
            logger.warning("所有配置都失败，使用默认KMeans")
            model = KMeans(n_clusters=4, random_state=42)
            labels = model.fit_predict(scaled_data)
            best_config = {
                'config': {'algorithm': 'kmeans', 'n_clusters': 4},
                'model': model,
                'labels': labels,
                'silhouette_score': silhouette_score(scaled_data, labels),
                'calinski_score': calinski_harabasz_score(scaled_data, labels),
                'n_clusters': 4
            }
        
        logger.info(f"✅ 最优聚类配置: 轮廓系数 {best_config['silhouette_score']:.3f}")
        return best_config

    def _adaptive_risk_threshold_optimization(self, clustering_result: Dict[str, Any],
                                            data: pd.DataFrame,
                                            target_distribution: Dict[str, float]) -> Dict[str, float]:
        """自适应风险阈值优化"""
        logger.info("🎯 执行自适应风险阈值优化")

        # 计算每个聚类的风险特征
        labels = clustering_result['labels']
        cluster_risk_scores = []

        for cluster_id in range(clustering_result['n_clusters']):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]

            if len(cluster_data) == 0:
                continue

            # 计算聚类风险评分
            risk_score = self._calculate_cluster_risk_score(cluster_data)
            cluster_risk_scores.append(risk_score)

        if not cluster_risk_scores:
            logger.warning("无法计算聚类风险评分，使用默认阈值")
            return {'low': 15, 'medium': 30, 'high': 50, 'critical': 100}

        # 基于实际风险评分分布计算最优阈值
        cluster_risk_scores = sorted(cluster_risk_scores)
        n_clusters = len(cluster_risk_scores)

        # 根据目标分布计算阈值
        low_threshold = np.percentile(cluster_risk_scores, target_distribution['low'] * 100)
        medium_threshold = np.percentile(cluster_risk_scores,
                                       (target_distribution['low'] + target_distribution['medium']) * 100)
        high_threshold = np.percentile(cluster_risk_scores,
                                     (1 - target_distribution['critical']) * 100)

        # 确保阈值合理性
        optimal_thresholds = {
            'low': max(10, min(low_threshold, 25)),
            'medium': max(20, min(medium_threshold, 40)),
            'high': max(35, min(high_threshold, 60)),
            'critical': 100
        }

        logger.info(f"✅ 优化后阈值: {optimal_thresholds}")
        return optimal_thresholds

    def _calculate_cluster_risk_score(self, cluster_data: pd.DataFrame) -> float:
        """计算聚类风险评分"""
        risk_score = 0.0

        # 欺诈率风险
        if 'is_fraudulent' in cluster_data.columns:
            fraud_rate = cluster_data['is_fraudulent'].mean()
            if fraud_rate > 0.15:
                risk_score += 50
            elif fraud_rate > 0.08:
                risk_score += 35
            elif fraud_rate > 0.04:
                risk_score += 25
            elif fraud_rate > 0.02:
                risk_score += 15
            else:
                risk_score += 10

        # 交易金额风险
        if 'transaction_amount' in cluster_data.columns:
            avg_amount = cluster_data['transaction_amount'].mean()
            if avg_amount > 2000:
                risk_score += 30
            elif avg_amount > 1000:
                risk_score += 20
            elif avg_amount > 500:
                risk_score += 10

        # 账户年龄风险
        if 'account_age_days' in cluster_data.columns:
            avg_age = cluster_data['account_age_days'].mean()
            if avg_age < 30:
                risk_score += 25
            elif avg_age < 90:
                risk_score += 15
            elif avg_age < 180:
                risk_score += 5

        # 时间模式风险
        if 'is_night_transaction' in cluster_data.columns:
            night_rate = cluster_data['is_night_transaction'].mean()
            if night_rate > 0.3:
                risk_score += 20
            elif night_rate > 0.15:
                risk_score += 10

        # 组合风险
        if 'high_risk_combination' in cluster_data.columns:
            combo_rate = cluster_data['high_risk_combination'].mean()
            if combo_rate > 0.1:
                risk_score += 15

        return min(100, risk_score)

    def _generate_optimized_result(self, clustering_result: Dict[str, Any],
                                 optimal_thresholds: Dict[str, float],
                                 selected_features: List[str],
                                 data: pd.DataFrame) -> Dict[str, Any]:
        """生成优化后的最终结果"""
        logger.info("📊 生成优化后的最终结果")

        return {
            'clustering_result': clustering_result,
            'optimal_thresholds': optimal_thresholds,
            'selected_features': selected_features,
            'optimization_summary': {
                'silhouette_score': clustering_result['silhouette_score'],
                'calinski_score': clustering_result['calinski_score'],
                'n_clusters': clustering_result['n_clusters'],
                'algorithm': clustering_result['config']['algorithm'],
                'feature_count': len(selected_features),
                'thresholds': optimal_thresholds
            },
            'recommendations': self._generate_recommendations(clustering_result, optimal_thresholds)
        }

    def _generate_recommendations(self, clustering_result: Dict[str, Any],
                                optimal_thresholds: Dict[str, float]) -> List[str]:
        """生成优化建议"""
        recommendations = []

        silhouette = clustering_result['silhouette_score']

        if silhouette > 0.5:
            recommendations.append("✅ 聚类质量优秀，轮廓系数 > 0.5")
        elif silhouette > 0.3:
            recommendations.append("⚠️ 聚类质量良好，轮廓系数 > 0.3")
        else:
            recommendations.append("❌ 聚类质量较差，建议检查数据质量或调整特征")

        if clustering_result['n_clusters'] >= 4:
            recommendations.append("✅ 聚类数量适中，支持四层风险分布")
        else:
            recommendations.append("⚠️ 聚类数量较少，可能影响风险分层效果")

        if optimal_thresholds['low'] < 20:
            recommendations.append("✅ 阈值设置合理，有利于风险分层")
        else:
            recommendations.append("⚠️ 阈值可能偏高，建议进一步降低")

        return recommendations

    def _fallback_clustering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """备用聚类方案"""
        logger.warning("使用备用聚类方案")

        # 简单的特征选择
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_fraudulent' in numeric_features:
            numeric_features.remove('is_fraudulent')

        selected_features = numeric_features[:6]  # 选择前6个数值特征

        # 简单的KMeans聚类
        scaled_data = self.scaler.fit_transform(data[selected_features])
        model = KMeans(n_clusters=4, random_state=42)
        labels = model.fit_predict(scaled_data)

        clustering_result = {
            'config': {'algorithm': 'kmeans', 'n_clusters': 4},
            'model': model,
            'labels': labels,
            'silhouette_score': silhouette_score(scaled_data, labels),
            'calinski_score': calinski_harabasz_score(scaled_data, labels),
            'n_clusters': 4
        }

        return {
            'clustering_result': clustering_result,
            'optimal_thresholds': {'low': 15, 'medium': 30, 'high': 50, 'critical': 100},
            'selected_features': selected_features,
            'optimization_summary': {
                'silhouette_score': clustering_result['silhouette_score'],
                'n_clusters': 4,
                'algorithm': 'kmeans',
                'feature_count': len(selected_features),
                'note': '备用方案'
            },
            'recommendations': ['⚠️ 使用备用聚类方案，建议检查数据质量']
        }
