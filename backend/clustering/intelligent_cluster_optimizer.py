#!/usr/bin/env python3
"""
Intelligent Clustering Optimizer
Automated feature selection, parameter tuning, and risk threshold optimization
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
    """Intelligent Clustering Optimizer"""

    def __init__(self):
        """Initialize optimizer"""
        self.scaler = StandardScaler()
        self.best_config = None
        self.optimization_history = []
        
        # Feature importance weights
        self.feature_importance = {
            'transaction_amount': 0.25,
            'customer_age': 0.15,
            'account_age_days': 0.20,
            'transaction_hour': 0.15,
            'quantity': 0.10,
            'is_fraudulent': 0.15  # Used for feature selection, not for clustering
        }
    
    def auto_optimize_clustering(self, data: pd.DataFrame, 
                               target_risk_distribution: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Automatically optimize clustering

        Args:
            data: Input data
            target_risk_distribution: Target risk distribution {'low': 0.4, 'medium': 0.3, 'high': 0.2, 'critical': 0.1}

        Returns:
            Optimal clustering configuration and results
        """
        logger.info("🚀 Starting intelligent clustering optimization")
        
        if target_risk_distribution is None:
            target_risk_distribution = {'low': 0.5, 'medium': 0.3, 'high': 0.15, 'critical': 0.05}
        
        try:
            # Step 1: Data preprocessing and feature engineering
            processed_data = self._advanced_feature_engineering(data)

            # Step 2: Intelligent feature selection (with enhanced features)
            optimal_features = self._intelligent_feature_selection(processed_data)

            # 使用增强后的数据（如果可用）
            enhanced_data = getattr(self, 'enhanced_data', processed_data)

            # Step 3: Clustering algorithm and parameter optimization
            best_clustering = self._optimize_clustering_parameters(enhanced_data[optimal_features])

            # Step 4: Adaptive risk threshold adjustment
            optimal_thresholds = self._adaptive_risk_threshold_optimization(
                best_clustering, enhanced_data, target_risk_distribution
            )

            # Step 5: Generate final results
            final_result = self._generate_optimized_result(
                best_clustering, optimal_thresholds, optimal_features, enhanced_data
            )

            logger.info("✅ Intelligent clustering optimization completed")
            return final_result

        except Exception as e:
            logger.error(f"❌ Intelligent clustering optimization failed: {e}")
            return self._fallback_clustering(data)
    
    def _advanced_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering"""
        logger.info("🔧 Executing advanced feature engineering")
        
        processed_data = data.copy()
        
        # Basic feature standardization
        numeric_features = ['transaction_amount', 'quantity', 'customer_age', 'account_age_days']
        for feature in numeric_features:
            if feature in processed_data.columns:
                # Z-score standardization
                processed_data[f'{feature}_zscore'] = (
                    processed_data[feature] - processed_data[feature].mean()
                ) / processed_data[feature].std()

                # Percentile features
                processed_data[f'{feature}_percentile'] = processed_data[feature].rank(pct=True)
        
        # Time feature enhancement
        if 'transaction_hour' in processed_data.columns:
            processed_data['is_night_transaction'] = (
                (processed_data['transaction_hour'] >= 22) |
                (processed_data['transaction_hour'] <= 6)
            ).astype(int)

            processed_data['is_business_hour'] = (
                (processed_data['transaction_hour'] >= 9) &
                (processed_data['transaction_hour'] <= 17)
            ).astype(int)
        
        # Account risk features
        if 'account_age_days' in processed_data.columns:
            processed_data['is_new_account'] = (processed_data['account_age_days'] < 30).astype(int)
            processed_data['is_very_new_account'] = (processed_data['account_age_days'] < 7).astype(int)
            processed_data['account_age_risk_score'] = np.where(
                processed_data['account_age_days'] < 30,
                100 - processed_data['account_age_days'] * 3,
                10
            )
        
        # Transaction amount risk features
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
        
        # Combined risk features
        if all(col in processed_data.columns for col in ['is_new_account', 'is_high_amount']):
            processed_data['high_risk_combination'] = (
                processed_data['is_new_account'] & processed_data['is_high_amount']
            ).astype(int)

        if all(col in processed_data.columns for col in ['is_night_transaction', 'is_high_amount']):
            processed_data['suspicious_pattern'] = (
                processed_data['is_night_transaction'] & processed_data['is_high_amount']
            ).astype(int)
        
        logger.info(f"✅ Feature engineering completed, generated {len(processed_data.columns)} features")
        return processed_data
    
    def _intelligent_feature_selection(self, data: pd.DataFrame) -> List[str]:
        """Intelligent feature selection - 使用增强版特征选择器"""
        logger.info("🎯 Executing enhanced intelligent feature selection")

        try:
            # 使用增强版特征选择器
            from backend.feature_engineer.feature_selector import FeatureSelector
            feature_selector = FeatureSelector(target_features=12)

            # 创建增强特征并选择最优特征
            enhanced_data = feature_selector._create_clustering_features(data.copy())
            selected_features = feature_selector.select_clustering_optimized_features(enhanced_data, max_features=10)

            # 更新数据以包含增强特征
            self.enhanced_data = enhanced_data

            logger.info(f"✅ Enhanced feature selection completed: {len(selected_features)} features")
            logger.info(f"Selected features: {selected_features}")
            return selected_features

        except Exception as e:
            logger.error(f"Enhanced feature selection failed: {e}")
            # 回退到基础方法
            return self._fallback_feature_selection(data)

    def _fallback_feature_selection(self, data: pd.DataFrame) -> List[str]:
        """回退的特征选择方法"""
        logger.warning("Using fallback feature selection")

        # Exclude non-numeric features and labels
        exclude_features = ['is_fraudulent', 'payment_method', 'device']
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        candidate_features = [f for f in numeric_features if f not in exclude_features]

        if len(candidate_features) < 3:
            logger.warning("Too few available features, using all numeric features")
            return candidate_features

        # Method 1: Variance-based feature selection
        feature_variance = data[candidate_features].var()
        high_variance_features = feature_variance[feature_variance > 0.01].index.tolist()

        # Method 2: Correlation-based feature selection
        if 'is_fraudulent' in data.columns:
            correlations = data[candidate_features].corrwith(data['is_fraudulent']).abs()
            high_corr_features = correlations.nlargest(min(8, len(candidate_features))).index.tolist()
        else:
            high_corr_features = candidate_features[:8]

        # Method 3: Clustering-friendly feature selection
        clustering_friendly_features = []
        for feature in candidate_features:
            if feature.endswith('_zscore') or feature.endswith('_percentile') or feature.endswith('_score'):
                clustering_friendly_features.append(feature)

        # Comprehensive selection
        selected_features = list(set(high_variance_features + high_corr_features + clustering_friendly_features))

        # Ensure at least 5 features
        if len(selected_features) < 5:
            selected_features = candidate_features[:min(8, len(candidate_features))]

        # Limit to maximum 12 features to avoid curse of dimensionality
        selected_features = selected_features[:12]

        logger.info(f"✅ Fallback feature selection: {len(selected_features)} features")
        return selected_features
    
    def _optimize_clustering_parameters(self, feature_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize clustering algorithm and parameters - 使用增强版聚类分析器"""
        logger.info("⚙️ Using enhanced clustering analyzer for optimization")

        try:
            # 使用标准化的聚类流程，确保数据预处理正确
            from sklearn.preprocessing import RobustScaler

            # 数据预处理和缩放
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(feature_data)
            scaled_df = pd.DataFrame(scaled_data, columns=feature_data.columns, index=feature_data.index)

            # 使用增强版聚类分析器
            from backend.clustering.cluster_analyzer import ClusterAnalyzer
            cluster_analyzer = ClusterAnalyzer()

            # 直接使用智能算法选择，使用缩放后的数据
            result = cluster_analyzer._auto_select_best_algorithm(scaled_df, feature_data)

            if result and result.get('cluster_count', 0) > 0:
                # 提取结果（_auto_select_best_algorithm的返回格式）
                algorithm = result.get('algorithm', 'kmeans')
                n_clusters = result.get('cluster_count', 4)
                labels = result.get('cluster_labels', [])

                # 手动计算质量指标，因为_auto_select_best_algorithm不返回quality_metrics
                try:
                    from sklearn.metrics import silhouette_score, calinski_harabasz_score
                    if len(set(labels)) > 1:
                        silhouette_score_val = silhouette_score(scaled_df, labels)
                        calinski_score_val = calinski_harabasz_score(scaled_df, labels)
                    else:
                        silhouette_score_val = 0.0
                        calinski_score_val = 0.0
                except Exception as e:
                    logger.warning(f"质量指标计算失败: {e}")
                    silhouette_score_val = 0.0
                    calinski_score_val = 0.0

                logger.info(f"✅ Enhanced clustering optimization: {algorithm}, {n_clusters} clusters, silhouette: {silhouette_score_val:.3f}")

                return {
                    'config': {
                        'algorithm': algorithm,
                        'n_clusters': n_clusters
                    },
                    'labels': np.array(labels),
                    'silhouette_score': silhouette_score_val,
                    'calinski_score': calinski_score_val,
                    'n_clusters': n_clusters
                }
            else:
                logger.warning("Enhanced clustering failed, using fallback method")
                return self._fallback_clustering_optimization(feature_data)

        except Exception as e:
            logger.error(f"Enhanced clustering optimization failed: {e}")
            return self._fallback_clustering_optimization(feature_data)

    def _fallback_clustering_optimization(self, feature_data: pd.DataFrame) -> Dict[str, Any]:
        """回退的聚类优化方法"""
        logger.info("Using fallback clustering optimization")

        # Standardize data
        scaled_data = self.scaler.fit_transform(feature_data)

        best_score = -1
        best_config = None

        # Test different clustering configurations
        configurations = [
            # KMeans configurations
            {'algorithm': 'kmeans', 'n_clusters': 3, 'init': 'k-means++', 'n_init': 20},
            {'algorithm': 'kmeans', 'n_clusters': 4, 'init': 'k-means++', 'n_init': 20},
            {'algorithm': 'kmeans', 'n_clusters': 5, 'init': 'k-means++', 'n_init': 20},
            {'algorithm': 'kmeans', 'n_clusters': 6, 'init': 'k-means++', 'n_init': 20},

            # DBSCAN configurations
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
                
                # Evaluate clustering quality
                if len(set(labels)) > 1 and -1 not in labels:  # Ensure multiple valid clusters
                    silhouette = silhouette_score(scaled_data, labels)
                    calinski = calinski_harabasz_score(scaled_data, labels)

                    # Combined score (silhouette coefficient has higher weight)
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
                        
                        logger.info(f"New best configuration: {config}, silhouette score: {silhouette:.3f}")

            except Exception as e:
                logger.warning(f"Configuration {config} failed: {e}")
                continue

        if best_config is None:
            logger.warning("All configurations failed, using default KMeans")
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
        
        logger.info(f"✅ Optimal clustering configuration: silhouette score {best_config['silhouette_score']:.3f}")
        return best_config

    def _adaptive_risk_threshold_optimization(self, clustering_result: Dict[str, Any],
                                            data: pd.DataFrame,
                                            target_distribution: Dict[str, float]) -> Dict[str, float]:
        """Adaptive risk threshold optimization"""
        logger.info("🎯 Executing adaptive risk threshold optimization")

        # Calculate risk features for each cluster
        labels = clustering_result['labels']
        cluster_risk_scores = []

        for cluster_id in range(clustering_result['n_clusters']):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]

            if len(cluster_data) == 0:
                continue

            # Calculate cluster risk score
            risk_score = self._calculate_cluster_risk_score(cluster_data)
            cluster_risk_scores.append(risk_score)

        if not cluster_risk_scores:
            logger.warning("Unable to calculate cluster risk scores, using default thresholds")
            return {'low': 15, 'medium': 30, 'high': 50, 'critical': 100}

        # Calculate optimal thresholds based on actual risk score distribution
        cluster_risk_scores = sorted(cluster_risk_scores)
        n_clusters = len(cluster_risk_scores)

        # Calculate thresholds based on target distribution
        low_threshold = np.percentile(cluster_risk_scores, target_distribution['low'] * 100)
        medium_threshold = np.percentile(cluster_risk_scores,
                                       (target_distribution['low'] + target_distribution['medium']) * 100)
        high_threshold = np.percentile(cluster_risk_scores,
                                     (1 - target_distribution['critical']) * 100)

        # Ensure threshold reasonableness
        optimal_thresholds = {
            'low': max(10, min(low_threshold, 25)),
            'medium': max(20, min(medium_threshold, 40)),
            'high': max(35, min(high_threshold, 60)),
            'critical': 100
        }

        logger.info(f"✅ Optimized thresholds: {optimal_thresholds}")
        return optimal_thresholds

    def _calculate_cluster_risk_score(self, cluster_data: pd.DataFrame) -> float:
        """Calculate cluster risk score"""
        risk_score = 0.0

        # Fraud rate risk
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

        # Transaction amount risk
        if 'transaction_amount' in cluster_data.columns:
            avg_amount = cluster_data['transaction_amount'].mean()
            if avg_amount > 2000:
                risk_score += 30
            elif avg_amount > 1000:
                risk_score += 20
            elif avg_amount > 500:
                risk_score += 10

        # Account age risk
        if 'account_age_days' in cluster_data.columns:
            avg_age = cluster_data['account_age_days'].mean()
            if avg_age < 30:
                risk_score += 25
            elif avg_age < 90:
                risk_score += 15
            elif avg_age < 180:
                risk_score += 5

        # Time pattern risk
        if 'is_night_transaction' in cluster_data.columns:
            night_rate = cluster_data['is_night_transaction'].mean()
            if night_rate > 0.3:
                risk_score += 20
            elif night_rate > 0.15:
                risk_score += 10

        # Combined risk
        if 'high_risk_combination' in cluster_data.columns:
            combo_rate = cluster_data['high_risk_combination'].mean()
            if combo_rate > 0.1:
                risk_score += 15

        return min(100, risk_score)

    def _generate_optimized_result(self, clustering_result: Dict[str, Any],
                                 optimal_thresholds: Dict[str, float],
                                 selected_features: List[str],
                                 data: pd.DataFrame) -> Dict[str, Any]:
        """Generate optimized final results - 兼容标准聚类格式"""
        logger.info("📊 Generating optimized final results")

        # 生成详细的聚类信息，兼容风险映射器
        cluster_details = self._generate_cluster_details(clustering_result, data)

        # 返回标准聚类分析器兼容的格式
        return {
            # 标准聚类分析器格式
            'algorithm': clustering_result['config']['algorithm'],
            'n_clusters': clustering_result['n_clusters'],
            'cluster_count': clustering_result['n_clusters'],  # 兼容性字段
            'cluster_labels': clustering_result['labels'].tolist(),
            'silhouette_score': clustering_result['silhouette_score'],
            'cluster_details': cluster_details,
            'quality_metrics': {
                'silhouette_score': clustering_result['silhouette_score'],
                'calinski_harabasz_score': clustering_result['calinski_score']
            },

            # 智能优化器特有信息
            'selected_features': selected_features,
            'optimal_thresholds': optimal_thresholds,
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
        """Generate optimization recommendations"""
        recommendations = []

        silhouette = clustering_result['silhouette_score']

        if silhouette > 0.5:
            recommendations.append("✅ Excellent clustering quality, silhouette score > 0.5")
        elif silhouette > 0.3:
            recommendations.append("⚠️ Good clustering quality, silhouette score > 0.3")
        else:
            recommendations.append("❌ Poor clustering quality, recommend checking data quality or adjusting features")

        if clustering_result['n_clusters'] >= 4:
            recommendations.append("✅ Appropriate cluster count, supports four-tier risk distribution")
        else:
            recommendations.append("⚠️ Few clusters, may affect risk stratification effectiveness")

        if optimal_thresholds['low'] < 20:
            recommendations.append("✅ Reasonable threshold settings, beneficial for risk stratification")
        else:
            recommendations.append("⚠️ Thresholds may be too high, recommend further reduction")

        return recommendations

    def _generate_cluster_details(self, clustering_result: Dict[str, Any], data: pd.DataFrame) -> List[Dict[str, Any]]:
        """生成详细的聚类信息，兼容风险映射器"""
        cluster_details = []
        labels = clustering_result['labels']
        n_clusters = clustering_result['n_clusters']

        for cluster_id in range(n_clusters):
            # 获取该聚类的数据
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]

            if len(cluster_data) == 0:
                continue

            # 计算聚类特征
            detail = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100,
            }

            # 交易金额统计
            if 'transaction_amount' in cluster_data.columns:
                detail['avg_transaction_amount'] = cluster_data['transaction_amount'].mean()
                detail['transaction_amount_std'] = cluster_data['transaction_amount'].std()
                detail['median_transaction_amount'] = cluster_data['transaction_amount'].median()
                detail['max_transaction_amount'] = cluster_data['transaction_amount'].max()
                detail['min_transaction_amount'] = cluster_data['transaction_amount'].min()

                # 大额交易比例
                high_amount_threshold = cluster_data['transaction_amount'].quantile(0.8)
                detail['high_amount_rate'] = (cluster_data['transaction_amount'] > high_amount_threshold).mean()

            # 客户年龄统计
            if 'customer_age' in cluster_data.columns:
                detail['avg_customer_age'] = cluster_data['customer_age'].mean()
                detail['customer_age_std'] = cluster_data['customer_age'].std()

            # 账户年龄统计
            if 'account_age_days' in cluster_data.columns:
                detail['avg_account_age_days'] = cluster_data['account_age_days'].mean()
                detail['account_age_std'] = cluster_data['account_age_days'].std()

                # 新账户比例
                detail['new_account_rate'] = (cluster_data['account_age_days'] < 90).mean()

            # 时间模式
            if 'transaction_hour' in cluster_data.columns:
                detail['avg_transaction_hour'] = cluster_data['transaction_hour'].mean()
                detail['most_common_hour'] = cluster_data['transaction_hour'].mode().iloc[0] if not cluster_data['transaction_hour'].mode().empty else 12

                # 夜间交易比例
                night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
                detail['night_transaction_rate'] = cluster_data['transaction_hour'].isin(night_hours).mean()

            # 欺诈率
            if 'is_fraudulent' in cluster_data.columns:
                detail['fraud_rate'] = cluster_data['is_fraudulent'].mean()
            else:
                detail['fraud_rate'] = 0.0

            # 支付方式分布
            if 'payment_method' in cluster_data.columns:
                payment_mode = cluster_data['payment_method'].mode()
                detail['common_payment_method'] = payment_mode.iloc[0] if not payment_mode.empty else 'unknown'

            # 设备分布
            if 'device_used' in cluster_data.columns:
                device_mode = cluster_data['device_used'].mode()
                detail['common_device'] = device_mode.iloc[0] if not device_mode.empty else 'unknown'

            # 产品类别分布
            if 'product_category' in cluster_data.columns:
                category_mode = cluster_data['product_category'].mode()
                detail['common_category'] = category_mode.iloc[0] if not category_mode.empty else 'unknown'

            # 地址风险
            if 'shipping_address' in cluster_data.columns:
                detail['different_shipping_rate'] = (cluster_data['shipping_address'] == 'different').mean()

            if 'billing_address' in cluster_data.columns:
                detail['different_billing_rate'] = (cluster_data['billing_address'] == 'different').mean()

            cluster_details.append(detail)

        return cluster_details

    def _fallback_clustering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback clustering solution"""
        logger.warning("Using fallback clustering solution")

        # Simple feature selection
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_fraudulent' in numeric_features:
            numeric_features.remove('is_fraudulent')

        selected_features = numeric_features[:6]  # Select first 6 numeric features

        # Simple KMeans clustering
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
                'note': 'Fallback solution'
            },
            'recommendations': ['⚠️ Using fallback clustering solution, recommend checking data quality']
        }
