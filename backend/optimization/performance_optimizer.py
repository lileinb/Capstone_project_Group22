"""
性能优化器
提供分层采样、向量化计算和缓存机制，提升风险评分计算性能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import logging
import time
from functools import wraps
import pickle
import os
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, cache_dir: str = "cache"):
        """
        初始化性能优化器
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        self.cache_enabled = True
        self.cache_ttl = 3600  # 缓存有效期1小时
        
        # 采样配置
        self.sampling_config = {
            'large_dataset_threshold': 10000,  # 大数据集阈值
            'sample_ratio': 0.3,               # 采样比例
            'min_cluster_samples': 100,        # 每个聚类最小样本数
            'max_total_samples': 5000          # 最大总样本数
        }
        
        # 向量化配置
        self.vectorization_config = {
            'batch_size': 1000,                # 批处理大小
            'use_parallel': True,              # 是否使用并行计算
            'n_jobs': -1                       # 并行作业数
        }
        
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def optimize_clustering(self, data: pd.DataFrame, 
                          n_clusters: int = 5,
                          use_sampling: bool = True) -> Dict[str, Any]:
        """
        优化聚类计算
        
        Args:
            data: 输入数据
            n_clusters: 聚类数量
            use_sampling: 是否使用采样
            
        Returns:
            聚类结果
        """
        try:
            start_time = time.time()
            
            # 检查缓存
            cache_key = f"clustering_{len(data)}_{n_clusters}_{hash(str(data.columns.tolist()))}"
            cached_result = self._get_cache(cache_key)
            if cached_result:
                logger.info("使用缓存的聚类结果")
                return cached_result
            
            # 数据预处理
            numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
            
            # 决定是否使用采样
            if use_sampling and len(data) > self.sampling_config['large_dataset_threshold']:
                logger.info(f"数据量大({len(data)})，使用采样优化")
                sample_data, sample_indices = self._stratified_sampling(numeric_data)
                clustering_data = sample_data
            else:
                clustering_data = numeric_data
                sample_indices = None
            
            # 标准化
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(clustering_data)
            
            # 使用MiniBatch K-means进行快速聚类
            if len(clustering_data) > 1000:
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    batch_size=min(1000, len(clustering_data) // 10),
                    random_state=42,
                    n_init=3
                )
            else:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # 如果使用了采样，需要为全量数据分配聚类标签
            if sample_indices is not None:
                full_labels = self._assign_full_labels(
                    data, sample_indices, cluster_labels, kmeans, scaler
                )
            else:
                full_labels = cluster_labels
            
            # 计算聚类质量指标
            quality_metrics = self._calculate_clustering_quality(scaled_data, cluster_labels)
            
            # 生成聚类详情
            cluster_details = self._generate_cluster_details(data, full_labels, n_clusters)
            
            result = {
                'cluster_labels': full_labels,
                'cluster_centers': kmeans.cluster_centers_,
                'quality_metrics': quality_metrics,
                'cluster_details': cluster_details,
                'n_clusters': n_clusters,
                'computation_time': time.time() - start_time,
                'used_sampling': sample_indices is not None,
                'sample_size': len(clustering_data) if sample_indices is not None else len(data)
            }
            
            # 缓存结果
            self._set_cache(cache_key, result)
            
            logger.info(f"聚类完成，耗时: {result['computation_time']:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"聚类优化失败: {e}")
            raise
    
    def _stratified_sampling(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """分层采样"""
        try:
            # 基于主要特征进行分层
            if 'transaction_amount' in data.columns:
                # 基于交易金额分层
                data['amount_bin'] = pd.qcut(data['transaction_amount'], 
                                           q=5, labels=False, duplicates='drop')
                strata_column = 'amount_bin'
            else:
                # 使用第一个数值列分层
                strata_column = data.columns[0]
                data['temp_bin'] = pd.qcut(data[strata_column], 
                                         q=5, labels=False, duplicates='drop')
                strata_column = 'temp_bin'
            
            # 计算每层的样本数
            total_samples = min(
                int(len(data) * self.sampling_config['sample_ratio']),
                self.sampling_config['max_total_samples']
            )
            
            # 分层采样
            sample_indices = []
            for stratum in data[strata_column].unique():
                if pd.isna(stratum):
                    continue
                    
                stratum_data = data[data[strata_column] == stratum]
                stratum_size = len(stratum_data)
                
                # 计算该层的采样数量
                stratum_samples = max(
                    int(total_samples * stratum_size / len(data)),
                    min(50, stratum_size)  # 每层至少50个样本
                )
                
                # 随机采样
                if stratum_samples < stratum_size:
                    sampled_indices = np.random.choice(
                        stratum_data.index, 
                        size=stratum_samples, 
                        replace=False
                    )
                else:
                    sampled_indices = stratum_data.index
                
                sample_indices.extend(sampled_indices)
            
            # 移除临时列
            if 'temp_bin' in data.columns:
                data = data.drop('temp_bin', axis=1)
            if 'amount_bin' in data.columns:
                data = data.drop('amount_bin', axis=1)
            
            sample_data = data.loc[sample_indices]
            
            logger.info(f"分层采样完成: {len(data)} -> {len(sample_data)}")
            return sample_data, np.array(sample_indices)
            
        except Exception as e:
            logger.error(f"分层采样失败: {e}")
            # 回退到简单随机采样
            sample_size = min(
                int(len(data) * self.sampling_config['sample_ratio']),
                self.sampling_config['max_total_samples']
            )
            sample_indices = np.random.choice(data.index, size=sample_size, replace=False)
            return data.loc[sample_indices], sample_indices
    
    def _assign_full_labels(self, full_data: pd.DataFrame, 
                           sample_indices: np.ndarray,
                           sample_labels: np.ndarray,
                           kmeans_model, scaler) -> np.ndarray:
        """为全量数据分配聚类标签"""
        try:
            # 对全量数据进行标准化和预测
            numeric_data = full_data.select_dtypes(include=[np.number]).fillna(0)
            scaled_full_data = scaler.transform(numeric_data)
            full_labels = kmeans_model.predict(scaled_full_data)
            
            return full_labels
            
        except Exception as e:
            logger.error(f"分配全量标签失败: {e}")
            # 回退方案：使用最近邻分配
            return self._nearest_neighbor_assignment(full_data, sample_indices, sample_labels)
    
    def _nearest_neighbor_assignment(self, full_data: pd.DataFrame,
                                   sample_indices: np.ndarray,
                                   sample_labels: np.ndarray) -> np.ndarray:
        """使用最近邻方法分配标签"""
        try:
            from sklearn.neighbors import KNeighborsClassifier
            
            numeric_data = full_data.select_dtypes(include=[np.number]).fillna(0)
            sample_data = numeric_data.loc[sample_indices]
            
            # 训练KNN分类器
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(sample_data, sample_labels)
            
            # 预测全量数据的标签
            full_labels = knn.predict(numeric_data)
            
            return full_labels
            
        except Exception as e:
            logger.error(f"最近邻分配失败: {e}")
            # 最后的回退方案：随机分配
            return np.random.randint(0, len(np.unique(sample_labels)), size=len(full_data))
    
    def _calculate_clustering_quality(self, data: np.ndarray, 
                                    labels: np.ndarray) -> Dict[str, float]:
        """计算聚类质量指标"""
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(data, labels)
                ch_score = calinski_harabasz_score(data, labels)
            else:
                silhouette = 0.0
                ch_score = 0.0
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': ch_score,
                'n_clusters': len(np.unique(labels)),
                'n_samples': len(data)
            }
            
        except Exception as e:
            logger.error(f"计算聚类质量失败: {e}")
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'n_clusters': len(np.unique(labels)),
                'n_samples': len(data)
            }
    
    def _generate_cluster_details(self, data: pd.DataFrame, 
                                labels: np.ndarray,
                                n_clusters: int) -> List[Dict]:
        """生成聚类详情"""
        cluster_details = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            detail = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100
            }
            
            # 计算数值特征的统计信息
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in cluster_data.columns:
                    detail[f'avg_{col}'] = cluster_data[col].mean()
                    detail[f'{col}_std'] = cluster_data[col].std()
            
            # 计算分类特征的分布
            categorical_cols = cluster_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col in cluster_data.columns:
                    value_counts = cluster_data[col].value_counts(normalize=True)
                    if len(value_counts) > 0:
                        detail[f'{col}_top_value'] = value_counts.index[0]
                        detail[f'{col}_top_ratio'] = value_counts.iloc[0]
            
            cluster_details.append(detail)
        
        return cluster_details
    
    def vectorized_risk_calculation(self, data: pd.DataFrame,
                                  risk_function,
                                  **kwargs) -> List[float]:
        """向量化风险计算"""
        try:
            start_time = time.time()
            
            # 检查是否可以向量化
            if hasattr(risk_function, 'vectorized') and risk_function.vectorized:
                # 直接向量化计算
                risk_scores = risk_function(data, **kwargs)
            else:
                # 批量处理
                risk_scores = []
                batch_size = self.vectorization_config['batch_size']
                
                for i in range(0, len(data), batch_size):
                    batch_data = data.iloc[i:i+batch_size]
                    batch_scores = [risk_function(row, **kwargs) for _, row in batch_data.iterrows()]
                    risk_scores.extend(batch_scores)
            
            computation_time = time.time() - start_time
            logger.info(f"向量化风险计算完成，耗时: {computation_time:.2f}秒")
            
            return risk_scores
            
        except Exception as e:
            logger.error(f"向量化风险计算失败: {e}")
            raise
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if not self.cache_enabled:
            return None
        
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                # 检查缓存是否过期
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < timedelta(seconds=self.cache_ttl):
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                else:
                    # 删除过期缓存
                    os.remove(cache_file)
            
            return None
            
        except Exception as e:
            logger.warning(f"获取缓存失败: {e}")
            return None
    
    def _set_cache(self, key: str, value: Any):
        """设置缓存"""
        if not self.cache_enabled:
            return
        
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
                
        except Exception as e:
            logger.warning(f"设置缓存失败: {e}")
    
    def clear_cache(self):
        """清除所有缓存"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info("缓存已清除")
            
        except Exception as e:
            logger.error(f"清除缓存失败: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        
        return {
            'cache_enabled': self.cache_enabled,
            'cache_files_count': len(cache_files),
            'cache_ttl': self.cache_ttl,
            'sampling_config': self.sampling_config,
            'vectorization_config': self.vectorization_config
        }
