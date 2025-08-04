"""
Cluster Interpreter
Output feature means, sample counts, and anomaly group identification for each cluster
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterInterpreter:
    """Cluster Interpreter"""
    def __init__(self):
        pass

    def interpret_clusters(self, clustering_results: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Interpret clustering results and identify anomaly clusters
        Args:
            clustering_results: Clustering analysis results
            data: Original data
        Returns:
            Cluster interpretation results
        """
        try:
            cluster_details = clustering_results.get('cluster_details', [])
            anomaly_clusters = []
            normal_clusters = []

            for cluster in cluster_details:
                cluster_id = cluster.get('cluster_id', 0)
                size = cluster.get('size', 0)

                # Simple anomaly detection: cluster size too small or too large
                total_samples = len(data)
                cluster_ratio = size / total_samples if total_samples > 0 else 0

                if cluster_ratio < 0.05 or cluster_ratio > 0.4:  # Anomaly threshold
                    anomaly_clusters.append({
                        'cluster_id': cluster_id,
                        'size': size,
                        'ratio': cluster_ratio,
                        'anomaly_type': 'size_anomaly',
                        'description': f'Cluster {cluster_id} size anomaly: {size} samples ({cluster_ratio:.2%})'
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
            logger.error(f"Cluster interpretation failed: {e}")
            return {
                'anomaly_clusters': [],
                'normal_clusters': [],
                'total_clusters': 0,
                'anomaly_count': 0
            }

    def analyze_clusters(self, data: pd.DataFrame, cluster_labels: List[int], features: List[str]) -> Dict[str, Any]:
        """
        Analyze clustering results and provide detailed cluster feature analysis
        Args:
            data: Original data
            cluster_labels: Cluster labels
            features: Features used for clustering
        Returns:
            Cluster analysis results
        """
        try:
            if len(cluster_labels) == 0:
                return self._empty_analysis_result()

            # Add cluster labels to data
            analysis_data = data.copy()
            analysis_data['cluster_label'] = cluster_labels

            # Get unique cluster labels
            unique_clusters = sorted(set(cluster_labels))
            cluster_analysis = []

            for cluster_id in unique_clusters:
                if cluster_id == -1:  # DBSCAN noise points
                    continue

                cluster_mask = analysis_data['cluster_label'] == cluster_id
                cluster_data = analysis_data[cluster_mask]

                if len(cluster_data) == 0:
                    continue

                # Calculate cluster feature statistics
                cluster_stats = {}
                for feature in features:
                    if feature in cluster_data.columns:
                        cluster_stats[feature] = {
                            'mean': float(cluster_data[feature].mean()),
                            'std': float(cluster_data[feature].std()),
                            'min': float(cluster_data[feature].min()),
                            'max': float(cluster_data[feature].max())
                        }

                # Calculate cluster risk features
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

            # Calculate overall statistics
            total_clusters = len(unique_clusters)
            if -1 in unique_clusters:  # Exclude noise points
                total_clusters -= 1

            return {
                'cluster_analysis': cluster_analysis,
                'total_clusters': total_clusters,
                'total_samples': len(data),
                'features_used': features,
                'anomaly_clusters': self._identify_anomaly_clusters(cluster_analysis)
            }

        except Exception as e:
            logger.error(f"Cluster analysis failed: {e}")
            return self._empty_analysis_result()

    def _identify_anomaly_clusters(self, cluster_analysis: List[Dict]) -> List[Dict]:
        """Identify anomaly clusters"""
        anomaly_clusters = []

        for cluster in cluster_analysis:
            # Identify anomalies based on size
            if cluster['percentage'] < 5:  # Clusters with less than 5%
                anomaly_clusters.append({
                    'cluster_id': cluster['cluster_id'],
                    'anomaly_type': 'small_cluster',
                    'description': f"Cluster {cluster['cluster_id']} has too few samples ({cluster['size']} samples)"
                })

            # Identify anomalies based on risk rate
            risk_features = cluster.get('risk_features', {})
            if risk_features.get('fraud_rate', 0) > 0.2:  # Fraud rate exceeds 20%
                anomaly_clusters.append({
                    'cluster_id': cluster['cluster_id'],
                    'anomaly_type': 'high_risk',
                    'description': f"Cluster {cluster['cluster_id']} has excessively high fraud rate ({risk_features['fraud_rate']:.2%})"
                })

        return anomaly_clusters

    def _empty_analysis_result(self) -> Dict[str, Any]:
        """Return empty analysis result"""
        return {
            'cluster_analysis': [],
            'total_clusters': 0,
            'total_samples': 0,
            'features_used': [],
            'anomaly_clusters': []
        }