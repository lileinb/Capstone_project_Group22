"""
聚类分析器
基于真实数据集特征的聚类分析
支持K-means、DBSCAN、高斯混合聚类
用于识别异常交易模式和用户行为群体
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

# 导入配置管理
try:
    from config.optimization_config import optimization_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    optimization_config = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterAnalyzer:
    """基于真实数据集的聚类分析器"""

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.risk_mapper = ClusterRiskMapper()  # 风险映射器
        self.intelligent_optimizer = IntelligentClusterOptimizer()  # 智能优化器

        # 加载配置
        self.config = self._load_clustering_config()

        # 用于聚类的关键特征
        self.clustering_features = [
            'transaction_amount', 'quantity', 'customer_age', 'account_age_days',
            'transaction_hour'
        ]

        # 分类特征编码映射
        self.categorical_mappings = {
            'payment_method': {'credit card': 1, 'debit card': 2, 'bank transfer': 3, 'PayPal': 4},
            'product_category': {'clothing': 1, 'electronics': 2, 'home & garden': 3, 'health & beauty': 4, 'toys & games': 5},
            'device_used': {'desktop': 1, 'mobile': 2, 'tablet': 3}
        }

    def analyze_clusters(self, data: pd.DataFrame, algorithm: str = 'kmeans') -> Dict[str, Any]:
        """
        基于真实数据集进行聚类分析

        Args:
            data: 清理后的DataFrame
            algorithm: 聚类算法 ('kmeans', 'dbscan', 'gaussian_mixture')

        Returns:
            聚类分析结果字典
        """
        if data is None or data.empty:
            logger.error("输入数据为空")
            return self._empty_result()

        try:
            # 准备聚类特征
            cluster_data = self._prepare_clustering_data(data)
            if cluster_data is None or cluster_data.empty:
                logger.error("无法准备聚类数据")
                return self._empty_result()

            # 执行聚类
            if algorithm == 'kmeans':
                results = self._kmeans_clustering(cluster_data, data)
            elif algorithm == 'dbscan':
                results = self._dbscan_clustering(cluster_data, data)
            elif algorithm == 'gaussian_mixture':
                results = self._gaussian_mixture_clustering(cluster_data, data)
            else:
                logger.warning(f"不支持的聚类算法: {algorithm}，使用默认kmeans")
                results = self._kmeans_clustering(cluster_data, data)

            # 添加聚类质量评估
            if len(set(results['cluster_labels'])) > 1:
                results['quality_metrics'] = self._evaluate_clustering_quality(
                    cluster_data, results['cluster_labels']
                )

            # 分析异常群体
            results['anomaly_analysis'] = self._analyze_anomalies(data, results['cluster_labels'])

            # 添加风险映射
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
        """准备聚类数据"""
        cluster_df = pd.DataFrame()

        # 添加数值特征
        for feature in self.clustering_features:
            if feature in data.columns:
                cluster_df[feature] = data[feature]

        # 编码分类特征
        for cat_feature, mapping in self.categorical_mappings.items():
            if cat_feature in data.columns:
                # 转换为字符串以避免分类类型问题
                feature_values = data[cat_feature].astype(str)
                cluster_df[f'{cat_feature}_encoded'] = feature_values.map(mapping).fillna(0)

        # 添加工程特征（如果存在）
        engineering_features = [
            'time_risk_score', 'amount_risk_score', 'device_risk_score',
            'account_age_risk_score', 'is_night_transaction', 'is_large_amount'
        ]

        for feature in engineering_features:
            if feature in data.columns:
                cluster_df[feature] = data[feature]

        # 检查数据有效性
        if cluster_df.empty or cluster_df.shape[1] == 0:
            logger.warning("没有可用的聚类特征")
            return None

        # 优化的数据预处理
        cluster_df = self._enhanced_data_preprocessing(cluster_df)

        if cluster_df is None or cluster_df.empty:
            logger.warning("数据预处理后为空")
            return None

        # 标准化特征
        cluster_df_scaled = pd.DataFrame(
            self.scaler.fit_transform(cluster_df),
            columns=cluster_df.columns,
            index=cluster_df.index
        )

        return cluster_df_scaled

    def _load_clustering_config(self) -> Dict[str, Any]:
        """加载聚类配置"""
        if CONFIG_AVAILABLE and optimization_config:
            return optimization_config.get_clustering_config()
        else:
            # 默认配置
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
        """计算数据质量评分"""
        try:
            if data.empty:
                return 0.0

            # 1. 完整性评分 (40%)
            completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 40

            # 2. 一致性评分 (30%) - 基于数据类型一致性
            consistency = 30  # 假设数据类型一致

            # 3. 有效性评分 (20%) - 基于数值范围合理性
            validity = 0
            for column in data.columns:
                if data[column].dtype in ['int64', 'float64']:
                    # 检查是否有无穷值或极端值
                    if not np.isinf(data[column]).any() and data[column].std() > 0:
                        validity += 20 / len(data.columns)

            # 4. 唯一性评分 (10%) - 基于重复行比例
            uniqueness = (1 - data.duplicated().sum() / len(data)) * 10

            total_score = completeness + consistency + validity + uniqueness
            return min(100.0, max(0.0, total_score))

        except Exception as e:
            logger.warning(f"数据质量评分计算失败: {e}")
            return 50.0  # 默认中等质量

    def _find_optimal_k(self, data: pd.DataFrame, max_k: int = None) -> int:
        """自动确定最优K值"""
        try:
            # 从配置获取参数
            if max_k is None:
                max_k = self.config.get('max_k', 10)
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

            # 综合考虑两种方法，优先选择轮廓系数较高的K值
            if abs(optimal_k_elbow - optimal_k_silhouette) <= 1:
                optimal_k = optimal_k_silhouette
            else:
                # 如果差异较大，选择轮廓系数最高的
                optimal_k = optimal_k_silhouette

            # 确保K值在合理范围内
            optimal_k = max(2, min(optimal_k, self.n_clusters))

            logger.info(f"K值优化: 肘部法则={optimal_k_elbow}, 轮廓系数={optimal_k_silhouette}, 最终选择={optimal_k}")
            return optimal_k

        except Exception as e:
            logger.warning(f"K值优化失败: {e}, 使用默认值")
            return self.n_clusters

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

        kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(cluster_data)

        # 分析每个聚类
        cluster_details = []
        for i in range(self.n_clusters):
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
            'cluster_count': self.n_clusters,
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
                continue  # 跳过噪声点

            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)

            if cluster_size > 0:
                cluster_original = original_data[cluster_mask]
                cluster_info = self._analyze_cluster_characteristics(cluster_original, label)
                cluster_info['size'] = cluster_size
                cluster_info['percentage'] = round(cluster_size / len(labels) * 100, 2)
                cluster_details.append(cluster_info)

        # 分析噪声点
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
        """分析聚类特征"""
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

        # 欺诈率
        if 'is_fraudulent' in cluster_data.columns:
            characteristics['fraud_rate'] = round(cluster_data['is_fraudulent'].mean(), 3)

            # 基于欺诈率确定风险等级
            fraud_rate = characteristics['fraud_rate']
            if fraud_rate > 0.1:
                characteristics['risk_level'] = 'high'
            elif fraud_rate > 0.05:
                characteristics['risk_level'] = 'medium'
            else:
                characteristics['risk_level'] = 'low'

        # 时间模式
        if 'transaction_hour' in cluster_data.columns:
            characteristics['common_hour'] = int(cluster_data['transaction_hour'].mode().iloc[0]) if not cluster_data['transaction_hour'].mode().empty else 12

            # 计算夜间交易比例 (22点-5点)
            night_transactions = ((cluster_data['transaction_hour'] >= 22) | (cluster_data['transaction_hour'] <= 5))
            characteristics['night_transaction_rate'] = round(night_transactions.mean(), 3)

        # 账户年龄特征
        if 'account_age_days' in cluster_data.columns:
            account_ages = cluster_data['account_age_days']
            characteristics['avg_account_age_days'] = round(account_ages.mean(), 1)
            characteristics['median_account_age_days'] = round(account_ages.median(), 1)

            # 计算新账户比例 (小于90天)
            new_accounts = (account_ages < 90)
            characteristics['new_account_rate'] = round(new_accounts.mean(), 3)

        # 设备使用模式
        if 'device' in cluster_data.columns:
            # 移动设备使用比例
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

        # 地址一致性
        if 'shipping_address' in cluster_data.columns and 'billing_address' in cluster_data.columns:
            address_mismatch = cluster_data['shipping_address'] != cluster_data['billing_address']
            characteristics['address_mismatch_rate'] = round(address_mismatch.mean(), 3)

        # 交易金额统计
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

        # 统计每个聚类的大小和欺诈率
        unique_labels = set(labels)
        total_size = len(labels)

        for label in unique_labels:
            if label == -1:  # DBSCAN的噪声点
                continue

            cluster_mask = [l == label for l in labels]
            cluster_data = data[cluster_mask]
            cluster_size = len(cluster_data)

            # 小聚类（占比小于5%）
            if cluster_size / total_size < 0.05:
                anomaly_info['small_clusters'].append({
                    'cluster_id': label,
                    'size': cluster_size,
                    'percentage': round(cluster_size / total_size * 100, 2)
                })

            # 高风险聚类（欺诈率>10%）
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
        """返回空结果"""
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
        logger.info(f"DBSCAN聚类完成，簇数: {len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)}")
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

            # 提取优化结果
            clustering_result = optimization_result['clustering_result']
            optimal_thresholds = optimization_result['optimal_thresholds']
            selected_features = optimization_result['selected_features']

            # 更新风险映射器的阈值
            self.risk_mapper.cluster_risk_thresholds = optimal_thresholds

            # 生成聚类详情
            cluster_details = self._generate_cluster_details_from_labels(
                data, clustering_result['labels'], selected_features
            )

            # 执行风险映射
            risk_mapping_result = self.risk_mapper.map_clusters_to_risk_levels(
                {'cluster_details': cluster_details}, data
            )

            # 合并结果
            final_result = {
                'algorithm': clustering_result['config']['algorithm'],
                'n_clusters': clustering_result['n_clusters'],
                'cluster_details': cluster_details,
                'cluster_labels': clustering_result['labels'].tolist(),
                'silhouette_score': clustering_result['silhouette_score'],
                'calinski_harabasz_score': clustering_result['calinski_score'],
                'selected_features': selected_features,
                'optimal_thresholds': optimal_thresholds,
                'optimization_summary': optimization_result['optimization_summary'],
                'recommendations': optimization_result['recommendations'],
                'risk_mapping': risk_mapping_result
            }

            logger.info("✅ 智能自动聚类完成")
            return final_result

        except Exception as e:
            logger.error(f"❌ 智能自动聚类失败: {e}")
            # 回退到传统聚类
            return self.analyze_clusters(data, algorithm='kmeans', n_clusters=4)

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

            # 基础统计
            detail = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100
            }

            # 数值特征统计
            for feature in features:
                if feature in cluster_data.columns and pd.api.types.is_numeric_dtype(cluster_data[feature]):
                    detail[f'avg_{feature}'] = cluster_data[feature].mean()
                    detail[f'{feature}_std'] = cluster_data[feature].std()

            # 特殊特征计算
            if 'is_fraudulent' in cluster_data.columns:
                detail['fraud_rate'] = cluster_data['is_fraudulent'].mean()

            if 'transaction_amount' in cluster_data.columns:
                detail['avg_transaction_amount'] = cluster_data['transaction_amount'].mean()
                detail['transaction_amount_std'] = cluster_data['transaction_amount'].std()
                # 高额交易比例
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

            # 分类特征统计
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