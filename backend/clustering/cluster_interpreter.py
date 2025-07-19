"""
聚类解释器
输出每个聚类的特征均值、样本数、异常群体识别
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterInterpreter:
    """聚类解释器"""
    def __init__(self):
        pass

    def interpret_clusters(self, clustering_results: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """
        解释聚类结果，识别异常聚类
        Args:
            clustering_results: 聚类分析结果
            data: 原始数据
        Returns:
            聚类解释结果
        """
        try:
            cluster_details = clustering_results.get('cluster_details', [])
            anomaly_clusters = []
            normal_clusters = []

            for cluster in cluster_details:
                cluster_id = cluster.get('cluster_id', 0)
                size = cluster.get('size', 0)

                # 简单异常检测：聚类大小过小或过大
                total_samples = len(data)
                cluster_ratio = size / total_samples if total_samples > 0 else 0

                if cluster_ratio < 0.05 or cluster_ratio > 0.4:  # 异常阈值
                    anomaly_clusters.append({
                        'cluster_id': cluster_id,
                        'size': size,
                        'ratio': cluster_ratio,
                        'anomaly_type': 'size_anomaly',
                        'description': f'聚类{cluster_id}大小异常: {size}个样本({cluster_ratio:.2%})'
                    })
                else:
                    normal_clusters.append({
                        'cluster_id': cluster_id,
                        'size': size,
                        'ratio': cluster_ratio
                    })

            return {
                'anomaly_clusters': anomaly_clusters,
                'normal_clusters': normal_clusters,
                'total_clusters': len(cluster_details),
                'anomaly_count': len(anomaly_clusters)
            }
        except Exception as e:
            logger.error(f"聚类解释失败: {e}")
            return {
                'anomaly_clusters': [],
                'normal_clusters': [],
                'total_clusters': 0,
                'anomaly_count': 0
            }

    def analyze_clusters(self, data: pd.DataFrame, cluster_labels: List[int], features: List[str]) -> Dict[str, Any]:
        """
        分析聚类结果，提供详细的聚类特征分析
        Args:
            data: 原始数据
            cluster_labels: 聚类标签
            features: 用于聚类的特征
        Returns:
            聚类分析结果
        """
        try:
            if len(cluster_labels) == 0:
                return self._empty_analysis_result()

            # 添加聚类标签到数据
            analysis_data = data.copy()
            analysis_data['cluster_label'] = cluster_labels

            # 获取唯一聚类标签
            unique_clusters = sorted(set(cluster_labels))
            cluster_analysis = []

            for cluster_id in unique_clusters:
                if cluster_id == -1:  # DBSCAN噪声点
                    continue

                cluster_mask = analysis_data['cluster_label'] == cluster_id
                cluster_data = analysis_data[cluster_mask]

                if len(cluster_data) == 0:
                    continue

                # 计算聚类特征统计
                cluster_stats = {}
                for feature in features:
                    if feature in cluster_data.columns:
                        cluster_stats[feature] = {
                            'mean': float(cluster_data[feature].mean()),
                            'std': float(cluster_data[feature].std()),
                            'min': float(cluster_data[feature].min()),
                            'max': float(cluster_data[feature].max())
                        }

                # 计算聚类风险特征
                risk_features = {}
                if 'is_fraudulent' in cluster_data.columns:
                    fraud_rate = cluster_data['is_fraudulent'].mean()
                    risk_features['fraud_rate'] = float(fraud_rate)
                    risk_features['risk_level'] = 'high' if fraud_rate > 0.1 else 'medium' if fraud_rate > 0.05 else 'low'

                cluster_info = {
                    'cluster_id': cluster_id,
                    'size': len(cluster_data),
                    'percentage': round(len(cluster_data) / len(analysis_data) * 100, 2),
                    'feature_stats': cluster_stats,
                    'risk_features': risk_features
                }

                cluster_analysis.append(cluster_info)

            # 计算整体统计
            total_clusters = len(unique_clusters)
            if -1 in unique_clusters:  # 排除噪声点
                total_clusters -= 1

            return {
                'cluster_analysis': cluster_analysis,
                'total_clusters': total_clusters,
                'total_samples': len(data),
                'features_used': features,
                'anomaly_clusters': self._identify_anomaly_clusters(cluster_analysis)
            }

        except Exception as e:
            logger.error(f"聚类分析失败: {e}")
            return self._empty_analysis_result()

    def _identify_anomaly_clusters(self, cluster_analysis: List[Dict]) -> List[Dict]:
        """识别异常聚类"""
        anomaly_clusters = []

        for cluster in cluster_analysis:
            # 基于大小识别异常
            if cluster['percentage'] < 5:  # 小于5%的聚类
                anomaly_clusters.append({
                    'cluster_id': cluster['cluster_id'],
                    'anomaly_type': 'small_cluster',
                    'description': f"聚类{cluster['cluster_id']}样本数过少({cluster['size']}个)"
                })

            # 基于风险率识别异常
            risk_features = cluster.get('risk_features', {})
            if risk_features.get('fraud_rate', 0) > 0.2:  # 欺诈率超过20%
                anomaly_clusters.append({
                    'cluster_id': cluster['cluster_id'],
                    'anomaly_type': 'high_risk',
                    'description': f"聚类{cluster['cluster_id']}欺诈率过高({risk_features['fraud_rate']:.2%})"
                })

        return anomaly_clusters

    def _empty_analysis_result(self) -> Dict[str, Any]:
        """返回空的分析结果"""
        return {
            'cluster_analysis': [],
            'total_clusters': 0,
            'total_samples': 0,
            'features_used': [],
            'anomaly_clusters': []
        }