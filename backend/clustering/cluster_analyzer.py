"""
Clustering Analyzer
Clustering analysis based on real dataset features
Supports K-means, DBSCAN, Gaussian Mixture clustering
Used to identify abnormal transaction patterns and user behavior groups
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, Any, List, Tuple
import logging
from .cluster_risk_mapper import ClusterRiskMapper
from .intelligent_cluster_optimizer import IntelligentClusterOptimizer

# Import configuration management
try:
    from config.optimization_config import optimization_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    optimization_config = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterAnalyzer:
    """Clustering analyzer based on real datasets"""

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.risk_mapper = ClusterRiskMapper()  # Risk mapper
        self.intelligent_optimizer = IntelligentClusterOptimizer()  # Intelligent optimizer

        # Load configuration
        self.config = self._load_clustering_config()

        # Key features for clustering - optimized version, using more discriminative features
        self.clustering_features = [
            # Original basic features
            'transaction_amount', 'quantity', 'customer_age', 'account_age_days', 'transaction_hour',
            # Engineering features - amount related
            'amount_zscore', 'amount_percentile', 'is_large_amount', 'amount_risk_score',
            # Engineering features - time related
            'time_risk_score', 'is_night_transaction', 'is_early_morning',
            # Engineering features - account related
            'is_new_account', 'account_age_risk_score', 'is_frequent_user',
            # Engineering features - composite risk
            'composite_risk_score', 'anomaly_score'
        ]

        # Fallback basic features (if engineering features are not available)
        self.fallback_features = [
            'transaction_amount', 'quantity', 'customer_age', 'account_age_days', 'transaction_hour'
        ]

        # 分类特征编码映射
        self.categorical_mappings = {
            'payment_method': {'credit card': 1, 'debit card': 2, 'bank transfer': 3, 'PayPal': 4},
            'product_category': {'clothing': 1, 'electronics': 2, 'home & garden': 3, 'health & beauty': 4, 'toys & games': 5},
            'device_used': {'desktop': 1, 'mobile': 2, 'tablet': 3}
        }

    def analyze_clusters(self, data: pd.DataFrame, algorithm: str = 'auto') -> Dict[str, Any]:
        """
        Perform clustering analysis based on real dataset

        Args:
            data: Cleaned DataFrame
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'gaussian_mixture')

        Returns:
            Clustering analysis result dictionary
        """
        if data is None or data.empty:
            logger.error("Input data is empty")
            return self._empty_result()

        try:
            # Prepare clustering features
            cluster_data = self._prepare_clustering_data(data)
            if cluster_data is None or cluster_data.empty:
                logger.error("Unable to prepare clustering data")
                return self._empty_result()

            # Execute clustering - optimized version
            if algorithm == 'auto':
                # Intelligently select best algorithm
                results = self._auto_select_best_algorithm(cluster_data, data)
            elif algorithm == 'kmeans':
                results = self._kmeans_clustering(cluster_data, data)
            elif algorithm == 'dbscan':
                results = self._dbscan_clustering(cluster_data, data)
            elif algorithm == 'gaussian_mixture':
                results = self._gaussian_mixture_clustering(cluster_data, data)
            else:
                logger.warning(f"Unsupported clustering algorithm: {algorithm}, using intelligent selection")
                results = self._auto_select_best_algorithm(cluster_data, data)

            # Add clustering quality evaluation
            if len(set(results['cluster_labels'])) > 1:
                results['quality_metrics'] = self._evaluate_clustering_quality(
                    cluster_data, results['cluster_labels']
                )

            # Analyze anomalous groups
            results['anomaly_analysis'] = self._analyze_anomalies(data, results['cluster_labels'])

            # Add risk mapping
            risk_mapping_results = self.risk_mapper.map_clusters_to_risk_levels(results, data)
            results.update(risk_mapping_results)

            # 将风险映射结果合并到cluster_details中
            cluster_risk_mapping = risk_mapping_results.get('cluster_risk_mapping', {})
            if cluster_risk_mapping and 'cluster_details' in results:
                for detail in results['cluster_details']:
                    cluster_id = detail.get('cluster_id', -1)
                    if cluster_id in cluster_risk_mapping:
                        # 更新风险等级和评分
                        risk_info = cluster_risk_mapping[cluster_id]
                        detail['risk_level'] = risk_info.get('risk_level', detail.get('risk_level', 'low'))
                        detail['risk_score'] = risk_info.get('risk_score', 0)
                        detail['risk_indicators'] = risk_info.get('risk_indicators', {})
                        detail['risk_explanation'] = risk_info.get('risk_explanation', [])

            logger.info(f"聚类分析完成: {algorithm}, 发现{results['cluster_count']}个群体")
            return results

        except Exception as e:
            logger.error(f"聚类分析失败: {e}")
            return self._empty_result()

    def _prepare_clustering_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备聚类数据 - 优化版本"""
        # 使用增强版特征选择器
        from backend.feature_engineer.feature_selector import FeatureSelector
        feature_selector = FeatureSelector(target_features=12)

        # 创建增强特征并选择最优特征
        enhanced_data = feature_selector._create_clustering_features(data.copy())
        selected_features = feature_selector.select_clustering_optimized_features(enhanced_data, max_features=10)

        # 使用增强后的数据
        data = enhanced_data

        if not selected_features:
            logger.warning("没有可用的聚类特征")
            return None

        cluster_df = pd.DataFrame()

        # 添加选中的特征（确保使用增强版特征选择的结果）
        for feature in selected_features:
            if feature in data.columns:
                cluster_df[feature] = data[feature]
            else:
                logger.warning(f"选中的特征 {feature} 不在数据中")

        # 只有在选中特征不足时才添加编码分类特征
        if len(cluster_df.columns) < 5:
            logger.info("选中特征不足，添加编码分类特征")
            for cat_feature, mapping in self.categorical_mappings.items():
                if cat_feature in data.columns and len(cluster_df.columns) < 8:
                    # 转换为字符串以避免分类类型问题
                    feature_values = data[cat_feature].astype(str)
                    cluster_df[f'{cat_feature}_encoded'] = feature_values.map(mapping).fillna(0)

        # 检查数据有效性
        if cluster_df.empty or cluster_df.shape[1] == 0:
            logger.warning("没有可用的聚类特征")
            return None

        # 优化的数据预处理
        cluster_df = self._enhanced_data_preprocessing(cluster_df)

        if cluster_df is None or cluster_df.empty:
            logger.warning("数据预处理后为空")
            return None

        # 使用更鲁棒的标准化方法
        cluster_df_scaled = self._robust_feature_scaling(cluster_df)

        logger.info(f"聚类数据准备完成，特征数: {cluster_df_scaled.shape[1]}, 样本数: {cluster_df_scaled.shape[0]}")
        logger.info(f"使用的特征: {list(cluster_df_scaled.columns)}")

        return cluster_df_scaled

    def _select_optimal_clustering_features(self, data: pd.DataFrame) -> List[str]:
        """Intelligently select optimal clustering features - enhanced version"""
        logger.info("🎯 Starting intelligent feature selection")

        # Step 1: Create more discriminative features
        enhanced_data = self._create_discriminative_features(data.copy())

        # Step 2: Multi-level feature selection
        candidate_features = self._get_candidate_features(enhanced_data)

        # Step 3: Score features based on clustering friendliness
        scored_features = self._score_features_for_clustering(enhanced_data, candidate_features)

        # Step 4: Select optimal feature combination
        final_features = self._select_optimal_feature_combination(enhanced_data, scored_features)

        logger.info(f"✅ Selected {len(final_features)} optimized features: {final_features}")
        return final_features

    def _create_discriminative_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建更有区分度的特征"""
        logger.info("🔧 创建区分性特征")

        # 金额异常特征
        if 'transaction_amount' in data.columns:
            amount_mean = data['transaction_amount'].mean()
            amount_std = data['transaction_amount'].std()

            data['amount_deviation'] = abs(data['transaction_amount'] - amount_mean) / amount_std
            data['is_extreme_amount'] = (data['amount_deviation'] > 2).astype(int)
            data['amount_log'] = np.log1p(data['transaction_amount'])
            data['amount_rank'] = data['transaction_amount'].rank(pct=True)

        # 时间风险特征
        if 'transaction_hour' in data.columns:
            # 深夜交易风险
            data['is_deep_night'] = data['transaction_hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
            # 工作时间外交易
            data['is_off_hours'] = (~data['transaction_hour'].isin([9, 10, 11, 12, 13, 14, 15, 16, 17])).astype(int)
            # 时间风险评分
            risk_hours = {0: 3, 1: 3, 2: 3, 3: 3, 4: 2, 5: 2, 22: 2, 23: 3}
            data['hour_risk_score'] = data['transaction_hour'].map(risk_hours).fillna(1)

        # 账户风险特征
        if 'account_age_days' in data.columns:
            data['is_very_new_account'] = (data['account_age_days'] < 30).astype(int)
            data['is_new_account'] = (data['account_age_days'] < 90).astype(int)
            data['account_maturity'] = np.log1p(data['account_age_days'])

        # 用户行为特征
        if 'customer_age' in data.columns:
            data['is_young_customer'] = (data['customer_age'] < 25).astype(int)
            data['is_senior_customer'] = (data['customer_age'] > 65).astype(int)

        # 复合风险特征
        risk_features = []
        if 'amount_deviation' in data.columns:
            risk_features.append('amount_deviation')
        if 'hour_risk_score' in data.columns:
            risk_features.append('hour_risk_score')
        if 'is_very_new_account' in data.columns:
            risk_features.append('is_very_new_account')

        if len(risk_features) >= 2:
            data['composite_risk'] = data[risk_features].sum(axis=1)
            data['risk_interaction'] = data[risk_features].prod(axis=1)

        logger.info(f"特征工程完成，当前特征数: {len(data.columns)}")
        return data

    def _get_candidate_features(self, data: pd.DataFrame) -> List[str]:
        """获取候选特征"""
        # 排除非数值特征和标签
        exclude_patterns = ['is_fraudulent', '_encoded', 'payment_method', 'device_used', 'product_category']
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()

        candidate_features = []
        for feature in numeric_features:
            if not any(pattern in feature for pattern in exclude_patterns):
                candidate_features.append(feature)

        return candidate_features

    def _score_features_for_clustering(self, data: pd.DataFrame, features: List[str]) -> Dict[str, float]:
        """为聚类评分特征"""
        feature_scores = {}

        for feature in features:
            if feature not in data.columns:
                continue

            feature_data = data[feature].dropna()
            if len(feature_data) == 0:
                feature_scores[feature] = 0.0
                continue

            score = 0.0

            # 1. 方差评分 (30%)
            variance = feature_data.var()
            if variance > 0:
                score += min(variance / 10, 1.0) * 0.3

            # 2. 分布评分 (25%) - 偏向正态分布的特征
            try:
                from scipy import stats
                _, p_value = stats.normaltest(feature_data)
                if p_value > 0.05:  # 接近正态分布
                    score += 0.25
                else:
                    score += 0.1
            except:
                score += 0.1

            # 3. 区分度评分 (25%) - 基于分位数范围
            q75, q25 = np.percentile(feature_data, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                score += min(iqr / feature_data.std(), 1.0) * 0.25

            # 4. 特征类型奖励 (20%)
            if any(suffix in feature for suffix in ['_zscore', '_percentile', '_score', '_deviation', '_rank']):
                score += 0.2
            elif any(suffix in feature for suffix in ['_log', '_sqrt', 'composite_', 'risk_']):
                score += 0.15
            elif feature in ['transaction_amount', 'customer_age', 'account_age_days']:
                score += 0.1

            feature_scores[feature] = score

        return feature_scores

    def _select_optimal_feature_combination(self, data: pd.DataFrame, scored_features: Dict[str, float]) -> List[str]:
        """选择最优特征组合"""
        # 按分数排序
        sorted_features = sorted(scored_features.items(), key=lambda x: x[1], reverse=True)

        selected_features = []
        correlation_threshold = 0.8

        for feature, score in sorted_features:
            if len(selected_features) >= 10:  # 限制特征数量
                break

            if score < 0.3:  # 分数太低的特征跳过
                continue

            # 检查与已选特征的相关性
            is_redundant = False
            for selected_feature in selected_features:
                if feature in data.columns and selected_feature in data.columns:
                    try:
                        corr = data[feature].corr(data[selected_feature])
                        if abs(corr) > correlation_threshold:
                            is_redundant = True
                            break
                    except:
                        continue

            if not is_redundant:
                selected_features.append(feature)

        # 确保至少有基础特征
        essential_features = ['transaction_amount', 'customer_age', 'account_age_days', 'transaction_hour']
        for feature in essential_features:
            if feature in data.columns and feature not in selected_features and len(selected_features) < 10:
                selected_features.append(feature)

        return selected_features[:8]  # 最多8个特征

    def _is_feature_valid_for_clustering(self, feature_series: pd.Series) -> bool:
        """检查特征是否适合聚类"""
        # 检查缺失值比例
        if feature_series.isna().sum() / len(feature_series) > 0.5:
            return False

        # 检查方差
        if feature_series.var() < 1e-6:
            return False

        # 检查是否全为同一值
        if feature_series.nunique() <= 1:
            return False

        return True

    def _robust_feature_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """使用更鲁棒的特征缩放方法"""
        from sklearn.preprocessing import RobustScaler, QuantileTransformer

        # 检查数据分布，选择合适的缩放方法
        skewness_threshold = 2.0
        highly_skewed_features = []

        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                skewness = abs(data[col].skew())
                if skewness > skewness_threshold:
                    highly_skewed_features.append(col)

        scaled_data = data.copy()

        if len(highly_skewed_features) > len(data.columns) * 0.5:
            # 如果大部分特征都高度偏斜，使用分位数变换
            logger.info("使用分位数变换处理高度偏斜的数据")
            transformer = QuantileTransformer(output_distribution='normal', random_state=42)
            scaled_values = transformer.fit_transform(data)
            scaled_data = pd.DataFrame(scaled_values, columns=data.columns, index=data.index)
        else:
            # 否则使用鲁棒缩放器
            logger.info("使用鲁棒缩放器处理数据")
            scaler = RobustScaler()
            scaled_values = scaler.fit_transform(data)
            scaled_data = pd.DataFrame(scaled_values, columns=data.columns, index=data.index)

        return scaled_data

    def _auto_select_best_algorithm(self, cluster_data: pd.DataFrame, original_data: pd.DataFrame) -> Dict[str, Any]:
        """Intelligently select best clustering algorithm"""
        logger.info("🤖 Starting intelligent algorithm selection")

        algorithms_to_try = ['dbscan', 'kmeans', 'gaussian_mixture']
        best_result = None
        best_score = -1

        for algorithm in algorithms_to_try:
            try:
                logger.info(f"Trying algorithm: {algorithm}")

                if algorithm == 'dbscan':
                    result = self._dbscan_clustering(cluster_data, original_data)
                elif algorithm == 'kmeans':
                    result = self._kmeans_clustering(cluster_data, original_data)
                elif algorithm == 'gaussian_mixture':
                    result = self._gaussian_mixture_clustering(cluster_data, original_data)

                # 评估聚类质量
                labels = result['cluster_labels']
                if len(set(labels)) > 1:
                    # 计算综合评分
                    silhouette = silhouette_score(cluster_data, labels)
                    calinski = calinski_harabasz_score(cluster_data, labels)

                    # 对于欺诈检测，我们更关心能否发现异常群体
                    cluster_count = result['cluster_count']
                    size_balance = self._evaluate_cluster_size_balance(labels)

                    # 改进的综合评分机制 - 偏向4个风险等级
                    # 对于欺诈检测，我们更重视能发现异常模式的能力
                    silhouette_weight = 0.35 if silhouette > 0.2 else 0.25
                    cluster_quality = self._evaluate_cluster_quality_for_fraud_detection(labels, original_data)

                    # 聚类数量偏好：4-6个聚类最佳，对应风险等级
                    cluster_preference = 0.0
                    if cluster_count == 4:
                        cluster_preference = 0.3  # 最佳：4个聚类对应4个风险等级
                    elif cluster_count == 5:
                        cluster_preference = 0.25  # 次佳：5个聚类
                    elif cluster_count == 3:
                        cluster_preference = 0.2   # 可接受：3个聚类
                    elif cluster_count == 6:
                        cluster_preference = 0.15  # 可接受：6个聚类
                    elif cluster_count == 2:
                        cluster_preference = 0.1   # 较差：只有2个聚类
                    elif cluster_count == 1:
                        cluster_preference = -0.3  # 很差：只有1个聚类
                    else:
                        cluster_preference = max(0, 0.1 - abs(cluster_count - 4) * 0.05)

                    combined_score = (
                        silhouette * silhouette_weight +  # 轮廓系数（降低权重）
                        (calinski / 2000) * 0.1 +  # Calinski-Harabasz指数
                        size_balance * 0.1 +  # 聚类大小平衡性
                        cluster_preference +  # 聚类数量偏好（新增）
                        cluster_quality * 0.15  # 欺诈检测质量
                    )

                    logger.info(f"{algorithm} 评分: {combined_score:.3f} (轮廓系数: {silhouette:.3f})")

                    if combined_score > best_score:
                        best_score = combined_score
                        best_result = result
                        logger.info(f"🏆 新的最佳算法: {algorithm}")

            except Exception as e:
                logger.warning(f"算法 {algorithm} 失败: {e}")
                continue

        if best_result is None:
            logger.warning("所有算法都失败，使用默认KMeans")
            best_result = self._kmeans_clustering(cluster_data, original_data)

        logger.info(f"✅ 最终选择算法: {best_result['algorithm']}, 评分: {best_score:.3f}")
        return best_result

    def _evaluate_cluster_size_balance(self, labels: List[int]) -> float:
        """评估聚类大小的平衡性"""
        from collections import Counter

        label_counts = Counter(labels)
        if -1 in label_counts:  # 移除DBSCAN的噪声点
            del label_counts[-1]

        if len(label_counts) <= 1:
            return 0.0

        sizes = list(label_counts.values())
        total = sum(sizes)
        proportions = [s / total for s in sizes]

        # 计算平衡性（越接近均匀分布越好）
        ideal_proportion = 1.0 / len(proportions)
        balance_score = 1.0 - sum(abs(p - ideal_proportion) for p in proportions) / 2

        return balance_score

    def _evaluate_cluster_quality_for_fraud_detection(self, labels: List[int], data: pd.DataFrame) -> float:
        """评估聚类对欺诈检测的质量"""
        try:
            from collections import Counter

            # 计算聚类的异常特征分布
            quality_score = 0.0
            label_counts = Counter(labels)

            # 移除噪声点
            if -1 in label_counts:
                del label_counts[-1]

            if len(label_counts) <= 1:
                return 0.0

            # 检查是否有小聚类（可能是异常群体）
            total_samples = sum(label_counts.values())
            small_clusters = sum(1 for count in label_counts.values() if count / total_samples < 0.1)
            if small_clusters > 0:
                quality_score += 0.3  # 有小聚类是好事

            # 检查聚类间的特征差异
            if 'transaction_amount' in data.columns:
                cluster_amounts = []
                for label in label_counts.keys():
                    cluster_mask = [l == label for l in labels]
                    cluster_data = data[cluster_mask]
                    if len(cluster_data) > 0:
                        cluster_amounts.append(cluster_data['transaction_amount'].mean())

                if len(cluster_amounts) > 1:
                    amount_std = np.std(cluster_amounts)
                    amount_mean = np.mean(cluster_amounts)
                    if amount_mean > 0:
                        cv = amount_std / amount_mean  # 变异系数
                        quality_score += min(cv, 1.0) * 0.4

            # 检查时间分布差异
            if 'transaction_hour' in data.columns:
                hour_distributions = []
                for label in label_counts.keys():
                    cluster_mask = [l == label for l in labels]
                    cluster_data = data[cluster_mask]
                    if len(cluster_data) > 0:
                        night_ratio = sum(cluster_data['transaction_hour'].isin([0, 1, 2, 3, 4, 5])) / len(cluster_data)
                        hour_distributions.append(night_ratio)

                if len(hour_distributions) > 1:
                    hour_std = np.std(hour_distributions)
                    quality_score += hour_std * 0.3

            return min(quality_score, 1.0)

        except Exception as e:
            logger.warning(f"欺诈检测质量评估失败: {e}")
            return 0.0

    def _load_clustering_config(self) -> Dict[str, Any]:
        """Load clustering configuration"""
        if CONFIG_AVAILABLE and optimization_config:
            return optimization_config.get_clustering_config()
        else:
            # Default configuration
            return {
                "auto_k_optimization": True,
                "max_k": 10,
                "min_k": 2,
                "dbscan_auto_params": True,
                "feature_selection": {
                    "enabled": True,
                    "variance_threshold": 0.01,
                    "correlation_threshold": 0.95,
                    "min_features": 3
                },
                "data_quality": {
                    "outlier_method": "iqr",
                    "outlier_factor": 1.5,
                    "missing_value_threshold": 0.5,
                    "quality_threshold": 70.0
                }
            }

    def _enhanced_data_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """增强的数据预处理"""
        try:
            processed_data = data.copy()

            # 1. 处理缺失值
            processed_data = self._handle_missing_values_smart(processed_data)

            # 2. 处理异常值
            processed_data = self._handle_outliers_iqr(processed_data)

            # 3. 处理无穷值
            processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
            processed_data = processed_data.fillna(processed_data.median())

            # 4. 数据质量检查
            quality_score = self._calculate_data_quality_score(processed_data)
            logger.info(f"数据质量评分: {quality_score:.2f}/100")

            return processed_data

        except Exception as e:
            logger.error(f"增强数据预处理失败: {e}")
            return data.fillna(0)  # 回退到简单处理

    def _handle_missing_values_smart(self, data: pd.DataFrame) -> pd.DataFrame:
        """智能缺失值处理"""
        try:
            processed_data = data.copy()

            for column in processed_data.columns:
                missing_ratio = processed_data[column].isnull().sum() / len(processed_data)

                if missing_ratio > 0:
                    if missing_ratio > 0.5:
                        # 缺失值过多，删除该列
                        logger.warning(f"列 {column} 缺失值比例 {missing_ratio:.2f}，将被删除")
                        processed_data = processed_data.drop(columns=[column])
                    else:
                        # 根据数据类型选择填充策略
                        if processed_data[column].dtype in ['int64', 'float64']:
                            # 数值型：使用中位数
                            fill_value = processed_data[column].median()
                        else:
                            # 其他类型：使用众数
                            mode_values = processed_data[column].mode()
                            fill_value = mode_values.iloc[0] if not mode_values.empty else 0

                        processed_data[column] = processed_data[column].fillna(fill_value)

            return processed_data

        except Exception as e:
            logger.warning(f"智能缺失值处理失败: {e}")
            return data.fillna(0)

    def _handle_outliers_iqr(self, data: pd.DataFrame) -> pd.DataFrame:
        """使用IQR方法处理异常值"""
        try:
            processed_data = data.copy()

            for column in processed_data.columns:
                if processed_data[column].dtype in ['int64', 'float64']:
                    Q1 = processed_data[column].quantile(0.25)
                    Q3 = processed_data[column].quantile(0.75)
                    IQR = Q3 - Q1

                    if IQR > 0:  # 避免除零错误
                        # 定义异常值边界
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        # 统计异常值数量
                        outliers_count = ((processed_data[column] < lower_bound) |
                                        (processed_data[column] > upper_bound)).sum()

                        if outliers_count > 0:
                            outlier_ratio = outliers_count / len(processed_data)
                            logger.info(f"列 {column} 发现 {outliers_count} 个异常值 ({outlier_ratio:.2%})")

                            # 使用Winsorization方法处理异常值
                            processed_data[column] = processed_data[column].clip(lower_bound, upper_bound)

            return processed_data

        except Exception as e:
            logger.warning(f"异常值处理失败: {e}")
            return data

    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score"""
        try:
            if data.empty:
                return 0.0

            # 1. Completeness score (40%)
            completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 40

            # 2. Consistency score (30%) - based on data type consistency
            consistency = 30  # Assume data types are consistent

            # 3. Validity score (20%) - based on numerical range reasonableness
            validity = 0
            for column in data.columns:
                if data[column].dtype in ['int64', 'float64']:
                    # Check for infinite values or extreme values
                    if not np.isinf(data[column]).any() and data[column].std() > 0:
                        validity += 20 / len(data.columns)

            # 4. Uniqueness score (10%) - based on duplicate row proportion
            uniqueness = (1 - data.duplicated().sum() / len(data)) * 10

            total_score = completeness + consistency + validity + uniqueness
            return min(100.0, max(0.0, total_score))

        except Exception as e:
            logger.warning(f"数据质量评分计算失败: {e}")
            return 50.0  # 默认中等质量

    def _find_optimal_k(self, data: pd.DataFrame, max_k: int = None) -> int:
        """自动确定最优K值"""
        try:
            # 从配置获取参数 - 扩大搜索范围
            if max_k is None:
                max_k = self.config.get('max_k', 12)  # 增加到12
            min_k = self.config.get('min_k', 2)

            if len(data) < 10:
                return min(min_k, len(data))

            # 检查是否启用自动K优化
            if not self.config.get('auto_k_optimization', True):
                return self.n_clusters

            # 计算不同K值的评估指标
            k_range = range(min_k, min(max_k + 1, len(data) // 2))
            inertias = []
            silhouette_scores = []

            for k in k_range:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                    labels = kmeans.fit_predict(data)

                    # 计算惯性（肘部法则）
                    inertias.append(kmeans.inertia_)

                    # 计算轮廓系数
                    if len(set(labels)) > 1:
                        sil_score = silhouette_score(data, labels)
                        silhouette_scores.append(sil_score)
                    else:
                        silhouette_scores.append(0)

                except Exception as e:
                    logger.warning(f"K={k}时聚类失败: {e}")
                    inertias.append(float('inf'))
                    silhouette_scores.append(0)

            # 使用肘部法则找到最优K
            optimal_k_elbow = self._find_elbow_point(list(k_range), inertias)

            # 使用轮廓系数找到最优K
            if silhouette_scores:
                optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
            else:
                optimal_k_silhouette = optimal_k_elbow

            # 综合考虑两种方法，但偏向4个聚类（对应4个风险等级）
            candidate_k_values = [optimal_k_elbow, optimal_k_silhouette]

            # 如果4在候选范围内且轮廓系数合理，优先选择4
            if 4 in k_range and len(silhouette_scores) >= (4 - min_k):
                k4_silhouette = silhouette_scores[4 - min_k]
                max_silhouette = max(silhouette_scores) if silhouette_scores else 0

                # 放宽K=4的选择条件，更强调业务需求
                if k4_silhouette >= max_silhouette - 0.4 and k4_silhouette > 0.3:
                    optimal_k = 4
                    logger.info(f"优先选择K=4（风险等级对应），轮廓系数: {k4_silhouette:.3f}")
                else:
                    optimal_k = optimal_k_silhouette
            else:
                # 如果4不在范围内，按原逻辑选择
                if abs(optimal_k_elbow - optimal_k_silhouette) <= 1:
                    optimal_k = optimal_k_silhouette
                else:
                    optimal_k = optimal_k_silhouette

            # 确保K值在合理范围内
            optimal_k = max(min_k, min(optimal_k, max_k))

            logger.info(f"K值优化: 肘部法则={optimal_k_elbow}, 轮廓系数={optimal_k_silhouette}, 最终选择={optimal_k}")
            logger.info(f"轮廓系数分布: {dict(zip(k_range, silhouette_scores))}")
            return optimal_k

        except Exception as e:
            logger.warning(f"K值优化失败: {e}, 使用默认值")
            return 4  # 使用固定默认值而不是self.n_clusters

    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """使用肘部法则找到最优K值"""
        try:
            if len(k_values) < 3:
                return k_values[0] if k_values else 2

            # 计算二阶导数来找到肘部点
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)

            # 找到二阶导数最大的点（肘部）
            if len(diffs2) > 0:
                elbow_idx = np.argmax(diffs2) + 2  # +2因为二阶导数的索引偏移
                if elbow_idx < len(k_values):
                    return k_values[elbow_idx]

            # 如果找不到明显的肘部，选择中间值
            return k_values[len(k_values) // 2]

        except Exception as e:
            logger.warning(f"肘部点计算失败: {e}")
            return k_values[0] if k_values else 2

    def _find_optimal_dbscan_params(self, data: pd.DataFrame) -> Tuple[float, int]:
        """自动确定DBSCAN最优参数"""
        try:
            from sklearn.neighbors import NearestNeighbors

            # 1. 确定min_samples (通常设为特征数量的2倍)
            min_samples = max(4, min(2 * data.shape[1], len(data) // 10))

            # 2. 使用k-distance图确定eps
            # 计算每个点到第k个最近邻的距离
            neighbors = NearestNeighbors(n_neighbors=min_samples)
            neighbors_fit = neighbors.fit(data)
            distances, indices = neighbors_fit.kneighbors(data)

            # 取第k个最近邻的距离并排序
            k_distances = distances[:, min_samples-1]
            k_distances = np.sort(k_distances)

            # 寻找肘部点作为eps
            eps = self._find_eps_elbow_point(k_distances)

            # 验证参数的有效性
            eps, min_samples = self._validate_dbscan_params(data, eps, min_samples)

            return eps, min_samples

        except Exception as e:
            logger.warning(f"DBSCAN参数优化失败: {e}, 使用默认参数")
            return 0.5, 5

    def _find_eps_elbow_point(self, k_distances: np.ndarray) -> float:
        """在k-distance图中找到肘部点确定eps"""
        try:
            if len(k_distances) < 10:
                return np.median(k_distances)

            # 计算一阶和二阶导数
            x = np.arange(len(k_distances))
            diffs = np.diff(k_distances)
            diffs2 = np.diff(diffs)

            # 找到二阶导数最大的点
            if len(diffs2) > 0:
                elbow_idx = np.argmax(diffs2)
                # 确保索引在有效范围内
                elbow_idx = min(elbow_idx, len(k_distances) - 1)
                return k_distances[elbow_idx]
            else:
                # 如果找不到肘部，使用75分位数
                return np.percentile(k_distances, 75)

        except Exception as e:
            logger.warning(f"eps肘部点计算失败: {e}")
            return np.median(k_distances) if len(k_distances) > 0 else 0.5

    def _validate_dbscan_params(self, data: pd.DataFrame, eps: float, min_samples: int) -> Tuple[float, int]:
        """验证和调整DBSCAN参数"""
        try:
            # 测试参数是否会产生合理的聚类结果
            test_dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            test_labels = test_dbscan.fit_predict(data)

            n_clusters = len(set(test_labels)) - (1 if -1 in test_labels else 0)
            noise_ratio = (test_labels == -1).sum() / len(test_labels)

            # 调整参数以获得更好的结果
            if n_clusters == 0 or noise_ratio > 0.5:
                # 聚类数太少或噪声太多，减小eps
                eps = eps * 0.7
                logger.info(f"调整eps: {eps:.3f} (减小)")
            elif n_clusters > len(data) // 10:
                # 聚类数太多，增大eps
                eps = eps * 1.3
                logger.info(f"调整eps: {eps:.3f} (增大)")

            # 确保min_samples在合理范围内
            min_samples = max(3, min(min_samples, len(data) // 5))

            return eps, min_samples

        except Exception as e:
            logger.warning(f"DBSCAN参数验证失败: {e}")
            return eps, min_samples

    def _kmeans_clustering(self, cluster_data: pd.DataFrame, original_data: pd.DataFrame) -> Dict[str, Any]:
        """K-means聚类 - 优化版本"""
        # 自动确定最优K值
        optimal_k = self._find_optimal_k(cluster_data)
        logger.info(f"自动确定最优K值: {optimal_k}")

        # 使用更好的KMeans参数
        kmeans = KMeans(
            n_clusters=optimal_k,
            random_state=self.random_state,
            n_init=20,  # 增加初始化次数
            max_iter=500,  # 增加最大迭代次数
            algorithm='lloyd'  # 使用经典算法确保稳定性
        )
        labels = kmeans.fit_predict(cluster_data)

        # 分析每个聚类
        cluster_details = []
        for i in range(optimal_k):  # 使用optimal_k而不是self.n_clusters
            cluster_mask = labels == i
            cluster_size = np.sum(cluster_mask)

            if cluster_size > 0:
                # 获取该聚类的原始数据
                cluster_original = original_data[cluster_mask]

                # 计算聚类特征
                cluster_info = self._analyze_cluster_characteristics(cluster_original, i)
                cluster_info['size'] = cluster_size
                cluster_info['percentage'] = round(cluster_size / len(labels) * 100, 2)

                cluster_details.append(cluster_info)

        return {
            'algorithm': 'kmeans',
            'cluster_count': optimal_k,  # 使用optimal_k
            'cluster_labels': labels.tolist(),
            'cluster_details': cluster_details,
            'centers': kmeans.cluster_centers_.tolist()
        }

    def _dbscan_clustering(self, cluster_data: pd.DataFrame, original_data: pd.DataFrame) -> Dict[str, Any]:
        """DBSCAN聚类 - 优化版本"""
        # 自动确定最优参数
        optimal_eps, optimal_min_samples = self._find_optimal_dbscan_params(cluster_data)
        logger.info(f"自动确定DBSCAN参数: eps={optimal_eps:.3f}, min_samples={optimal_min_samples}")

        dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
        labels = dbscan.fit_predict(cluster_data)

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

        cluster_details = []
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points

            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)

            if cluster_size > 0:
                cluster_original = original_data[cluster_mask]
                cluster_info = self._analyze_cluster_characteristics(cluster_original, label)
                cluster_info['size'] = cluster_size
                cluster_info['percentage'] = round(cluster_size / len(labels) * 100, 2)
                cluster_details.append(cluster_info)

        # Analyze noise points
        noise_count = np.sum(labels == -1)

        return {
            'algorithm': 'dbscan',
            'cluster_count': n_clusters,
            'cluster_labels': labels.tolist(),
            'cluster_details': cluster_details,
            'noise_points': noise_count,
            'noise_percentage': round(noise_count / len(labels) * 100, 2)
        }

    def _gaussian_mixture_clustering(self, cluster_data: pd.DataFrame, original_data: pd.DataFrame) -> Dict[str, Any]:
        """高斯混合模型聚类"""
        gmm = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
        labels = gmm.fit_predict(cluster_data)

        cluster_details = []
        for i in range(self.n_clusters):
            cluster_mask = labels == i
            cluster_size = np.sum(cluster_mask)

            if cluster_size > 0:
                cluster_original = original_data[cluster_mask]
                cluster_info = self._analyze_cluster_characteristics(cluster_original, i)
                cluster_info['size'] = cluster_size
                cluster_info['percentage'] = round(cluster_size / len(labels) * 100, 2)
                cluster_details.append(cluster_info)

        return {
            'algorithm': 'gaussian_mixture',
            'cluster_count': self.n_clusters,
            'cluster_labels': labels.tolist(),
            'cluster_details': cluster_details,
            'bic_score': gmm.bic(cluster_data),
            'aic_score': gmm.aic(cluster_data)
        }

    def fit_kmeans(self, data: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> np.ndarray:
        self.method = 'kmeans'
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.labels_ = self.model.fit_predict(data)
        self.centers_ = self.model.cluster_centers_
        logger.info(f"K-means聚类完成，聚类数: {n_clusters}")
        return self.labels_

    def _analyze_cluster_characteristics(self, cluster_data: pd.DataFrame, cluster_id: int) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        characteristics = {
            'cluster_id': cluster_id,
            'avg_transaction_amount': 0,
            'avg_customer_age': 0,
            'common_payment_method': 'unknown',
            'common_device': 'unknown',
            'common_category': 'unknown',
            'fraud_rate': 0,
            'risk_level': 'low'
        }

        if cluster_data.empty:
            return characteristics

        # 交易金额特征
        if 'transaction_amount' in cluster_data.columns:
            amounts = cluster_data['transaction_amount']
            characteristics['avg_transaction_amount'] = round(amounts.mean(), 2)
            characteristics['median_transaction_amount'] = round(amounts.median(), 2)
            characteristics['transaction_amount_std'] = round(amounts.std(), 2)

            # 计算高金额交易比例（大于平均值+2倍标准差）
            mean_amount = amounts.mean()
            std_amount = amounts.std()
            high_threshold = mean_amount + 2 * std_amount if std_amount > 0 else mean_amount * 2
            characteristics['high_amount_rate'] = round((amounts > high_threshold).mean(), 3)

        # 客户年龄特征
        if 'customer_age' in cluster_data.columns:
            characteristics['avg_customer_age'] = round(cluster_data['customer_age'].mean(), 1)

        # 最常见的分类特征
        if 'payment_method' in cluster_data.columns:
            characteristics['common_payment_method'] = cluster_data['payment_method'].mode().iloc[0] if not cluster_data['payment_method'].mode().empty else 'unknown'

        if 'device_used' in cluster_data.columns:
            characteristics['common_device'] = cluster_data['device_used'].mode().iloc[0] if not cluster_data['device_used'].mode().empty else 'unknown'

        if 'product_category' in cluster_data.columns:
            characteristics['common_category'] = cluster_data['product_category'].mode().iloc[0] if not cluster_data['product_category'].mode().empty else 'unknown'

        # Fraud rate
        if 'is_fraudulent' in cluster_data.columns:
            characteristics['fraud_rate'] = round(cluster_data['is_fraudulent'].mean(), 3)

            # Determine risk level based on fraud rate
            fraud_rate = characteristics['fraud_rate']
            if fraud_rate > 0.1:
                characteristics['risk_level'] = 'high'
            elif fraud_rate > 0.05:
                characteristics['risk_level'] = 'medium'
            else:
                characteristics['risk_level'] = 'low'

        # Time patterns
        if 'transaction_hour' in cluster_data.columns:
            characteristics['common_hour'] = int(cluster_data['transaction_hour'].mode().iloc[0]) if not cluster_data['transaction_hour'].mode().empty else 12

            # Calculate night transaction ratio (22:00-5:00)
            night_transactions = ((cluster_data['transaction_hour'] >= 22) | (cluster_data['transaction_hour'] <= 5))
            characteristics['night_transaction_rate'] = round(night_transactions.mean(), 3)

        # Account age features
        if 'account_age_days' in cluster_data.columns:
            account_ages = cluster_data['account_age_days']
            characteristics['avg_account_age_days'] = round(account_ages.mean(), 1)
            characteristics['median_account_age_days'] = round(account_ages.median(), 1)

            # Calculate new account ratio (less than 90 days)
            new_accounts = (account_ages < 90)
            characteristics['new_account_rate'] = round(new_accounts.mean(), 3)

        # Device usage patterns
        if 'device' in cluster_data.columns:
            # Mobile device usage ratio
            mobile_usage = (cluster_data['device'] == 'mobile')
            characteristics['mobile_device_rate'] = round(mobile_usage.mean(), 3)
            characteristics['common_device'] = cluster_data['device'].mode().iloc[0] if not cluster_data['device'].mode().empty else 'unknown'

        # 支付方式模式
        if 'payment_method' in cluster_data.columns:
            # 银行转账比例
            bank_transfer_usage = (cluster_data['payment_method'] == 'bank_transfer')
            characteristics['bank_transfer_rate'] = round(bank_transfer_usage.mean(), 3)

        # 商品类别风险
        if 'product_category' in cluster_data.columns:
            # 高风险类别比例（electronics通常风险较高）
            electronics_rate = (cluster_data['product_category'] == 'electronics').mean()
            characteristics['electronics_rate'] = round(electronics_rate, 3)
        else:
            characteristics['electronics_rate'] = 0.3  # 默认值

        # 地址一致性（如果有相关字段）
        if 'shipping_address' in cluster_data.columns and 'billing_address' in cluster_data.columns:
            address_mismatch_rate = (cluster_data['shipping_address'] != cluster_data['billing_address']).mean()
            characteristics['address_mismatch_rate'] = round(address_mismatch_rate, 3)
        else:
            # 如果没有地址字段，基于其他因素估算地址风险
            characteristics['address_mismatch_rate'] = 0.1  # 默认值

        # 商品类别风险
        if 'product_category' in cluster_data.columns:
            characteristics['electronics_rate'] = round(
                len(cluster_data[cluster_data['product_category'] == 'electronics']) / len(cluster_data), 3
            )

        # Address consistency
        if 'shipping_address' in cluster_data.columns and 'billing_address' in cluster_data.columns:
            address_mismatch = cluster_data['shipping_address'] != cluster_data['billing_address']
            characteristics['address_mismatch_rate'] = round(address_mismatch.mean(), 3)

        # Transaction amount statistics
        if 'transaction_amount' in cluster_data.columns:
            characteristics['transaction_amount_std'] = round(cluster_data['transaction_amount'].std(), 2)
            characteristics['high_amount_rate'] = round(
                len(cluster_data[cluster_data['transaction_amount'] > 1000]) / len(cluster_data), 3
            )

        return characteristics

    def _evaluate_clustering_quality(self, cluster_data: pd.DataFrame, labels: List[int]) -> Dict[str, float]:
        """评估聚类质量"""
        try:
            silhouette = silhouette_score(cluster_data, labels)
            calinski_harabasz = calinski_harabasz_score(cluster_data, labels)

            return {
                'silhouette_score': round(silhouette, 3),
                'calinski_harabasz_score': round(calinski_harabasz, 3)
            }
        except Exception as e:
            logger.warning(f"聚类质量评估失败: {e}")
            return {'silhouette_score': 0, 'calinski_harabasz_score': 0}

    def _analyze_anomalies(self, data: pd.DataFrame, labels: List[int]) -> Dict[str, Any]:
        """分析异常群体"""
        anomaly_info = {
            'high_risk_clusters': [],
            'small_clusters': [],
            'unusual_patterns': []
        }

        # Count size and fraud rate for each cluster
        unique_labels = set(labels)
        total_size = len(labels)

        for label in unique_labels:
            if label == -1:  # DBSCAN noise points
                continue

            cluster_mask = [l == label for l in labels]
            cluster_data = data[cluster_mask]
            cluster_size = len(cluster_data)

            # Small clusters (proportion less than 5%)
            if cluster_size / total_size < 0.05:
                anomaly_info['small_clusters'].append({
                    'cluster_id': label,
                    'size': cluster_size,
                    'percentage': round(cluster_size / total_size * 100, 2)
                })

            # High-risk clusters (fraud rate > 10%)
            if 'is_fraudulent' in cluster_data.columns:
                fraud_rate = cluster_data['is_fraudulent'].mean()
                if fraud_rate > 0.1:
                    anomaly_info['high_risk_clusters'].append({
                        'cluster_id': label,
                        'fraud_rate': round(fraud_rate, 3),
                        'size': cluster_size
                    })

        return anomaly_info

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            'algorithm': 'none',
            'cluster_count': 0,
            'cluster_labels': [],
            'cluster_details': [],
            'quality_metrics': {},
            'anomaly_analysis': {}
        }

    def fit_dbscan(self, data: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        self.method = 'dbscan'
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = self.model.fit_predict(data)
        self.centers_ = None
        logger.info(f"DBSCAN clustering completed, number of clusters: {len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)}")
        return self.labels_

    def fit_gmm(self, data: pd.DataFrame, n_components: int = 4, random_state: int = 42) -> np.ndarray:
        self.method = 'gmm'
        self.model = GaussianMixture(n_components=n_components, random_state=random_state)
        self.labels_ = self.model.fit_predict(data)
        self.centers_ = self.model.means_
        logger.info(f"高斯混合聚类完成，组件数: {n_components}")
        return self.labels_

    def get_cluster_centers(self):
        return self.centers_

    def get_labels(self):
        return self.labels_

    def get_method(self):
        return self.method

    def intelligent_auto_clustering(self, data: pd.DataFrame,
                                   target_risk_distribution: Dict[str, float] = None) -> Dict[str, Any]:
        """
        智能自动聚类 - 一键最优聚类

        Args:
            data: 输入数据
            target_risk_distribution: 目标风险分布

        Returns:
            完整的聚类分析结果
        """
        logger.info("🚀 开始智能自动聚类")

        try:
            # 使用智能优化器进行自动优化
            optimization_result = self.intelligent_optimizer.auto_optimize_clustering(
                data, target_risk_distribution
            )

            # 智能优化器现在返回标准格式，直接使用
            # 更新风险映射器的阈值
            optimal_thresholds = optimization_result.get('optimal_thresholds', {})
            if optimal_thresholds:
                self.risk_mapper.cluster_risk_thresholds = optimal_thresholds

            # 执行风险映射（如果还没有风险信息）
            if 'cluster_details' in optimization_result and optimization_result['cluster_details']:
                cluster_details = optimization_result['cluster_details']

                # 检查是否需要风险映射
                needs_risk_mapping = True
                if cluster_details and len(cluster_details) > 0:
                    first_cluster = cluster_details[0]
                    if 'risk_level' in first_cluster and first_cluster['risk_level'] != 'unknown':
                        needs_risk_mapping = False

                if needs_risk_mapping:
                    risk_mapping_result = self.risk_mapper.map_clusters_to_risk_levels(
                        {'cluster_details': cluster_details}, data
                    )

                    # 更新聚类详情中的风险信息
                    cluster_risk_mapping = risk_mapping_result.get('cluster_risk_mapping', {})
                    for detail in cluster_details:
                        cluster_id = detail.get('cluster_id', -1)
                        if cluster_id in cluster_risk_mapping:
                            risk_info = cluster_risk_mapping[cluster_id]
                            detail['risk_level'] = risk_info.get('risk_level', 'low')
                            detail['risk_score'] = risk_info.get('risk_score', 0)

            # 返回优化结果（已经是标准格式）
            final_result = optimization_result

            logger.info("✅ 智能自动聚类完成")
            return final_result

        except Exception as e:
            logger.error(f"❌ 智能自动聚类失败: {e}")
            # 回退到传统聚类
            return self.analyze_clusters(data, algorithm='kmeans')

    def _generate_cluster_details_from_labels(self, data: pd.DataFrame,
                                            labels: np.ndarray,
                                            features: List[str]) -> List[Dict[str, Any]]:
        """从聚类标签生成聚类详情"""
        cluster_details = []

        for cluster_id in range(len(set(labels))):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]

            if len(cluster_data) == 0:
                continue

            # Basic statistics
            detail = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100
            }

            # Numerical feature statistics
            for feature in features:
                if feature in cluster_data.columns and pd.api.types.is_numeric_dtype(cluster_data[feature]):
                    detail[f'avg_{feature}'] = cluster_data[feature].mean()
                    detail[f'{feature}_std'] = cluster_data[feature].std()

            # Special feature calculations
            if 'is_fraudulent' in cluster_data.columns:
                detail['fraud_rate'] = cluster_data['is_fraudulent'].mean()

            if 'transaction_amount' in cluster_data.columns:
                detail['avg_transaction_amount'] = cluster_data['transaction_amount'].mean()
                detail['transaction_amount_std'] = cluster_data['transaction_amount'].std()
                # High amount transaction ratio
                high_amount_threshold = data['transaction_amount'].quantile(0.75)
                detail['high_amount_rate'] = (cluster_data['transaction_amount'] > high_amount_threshold).mean()

            if 'account_age_days' in cluster_data.columns:
                detail['avg_account_age_days'] = cluster_data['account_age_days'].mean()
                detail['new_account_rate'] = (cluster_data['account_age_days'] < 30).mean()

            if 'transaction_hour' in cluster_data.columns:
                detail['night_transaction_rate'] = (
                    (cluster_data['transaction_hour'] >= 22) |
                    (cluster_data['transaction_hour'] <= 6)
                ).mean()
                detail['common_hour'] = cluster_data['transaction_hour'].mode().iloc[0] if len(cluster_data['transaction_hour'].mode()) > 0 else 12

            # Categorical feature statistics
            if 'device' in cluster_data.columns:
                device_counts = cluster_data['device'].value_counts()
                detail['common_device'] = device_counts.index[0] if len(device_counts) > 0 else 'unknown'
                detail['mobile_device_rate'] = (cluster_data['device'] == 'mobile').mean()

            if 'payment_method' in cluster_data.columns:
                payment_counts = cluster_data['payment_method'].value_counts()
                detail['common_payment_method'] = payment_counts.index[0] if len(payment_counts) > 0 else 'unknown'
                detail['bank_transfer_rate'] = (cluster_data['payment_method'] == 'bank_transfer').mean()

            cluster_details.append(detail)

        return cluster_details