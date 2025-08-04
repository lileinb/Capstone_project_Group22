"""
Unified Data Integration Pipeline
Solves data transfer and duplicate computation issues between modules
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class UnifiedDataPipeline:
    """Unified data integration pipeline"""

    def __init__(self):
        """Initialize pipeline"""
        self.integrated_data = None
        self.feature_mapping = {}
        self.data_sources = {}
        self.integration_log = []
        
    def integrate_all_sources(self,
                            engineered_features: pd.DataFrame,
                            clustering_results: Optional[Dict] = None,
                            risk_scores: Optional[Dict] = None,
                            pseudo_labels: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Integrate all data sources

        Args:
            engineered_features: Feature engineering results
            clustering_results: Clustering analysis results
            risk_scores: Risk scoring results
            pseudo_labels: Pseudo label generation results

        Returns:
            Integrated dataset and metadata
        """
        try:
            logger.info("🔄 Starting unified data integration")
            start_time = datetime.now()

            # 1. Validate input data
            if not self._validate_inputs(engineered_features, clustering_results, risk_scores, pseudo_labels):
                return self._empty_result("Input data validation failed")

            # 2. Initialize base data
            integrated_df = engineered_features.copy()
            self.data_sources['engineered_features'] = len(engineered_features.columns)

            # 3. Integrate clustering information
            if clustering_results:
                integrated_df = self._integrate_clustering_data(integrated_df, clustering_results)
                self.data_sources['clustering'] = True

            # 4. Integrate risk scoring information
            if risk_scores:
                integrated_df = self._integrate_risk_scores(integrated_df, risk_scores)
                self.data_sources['risk_scores'] = True
            
            # 5. 整合伪标签信息
            if pseudo_labels:
                integrated_df = self._integrate_pseudo_labels(integrated_df, pseudo_labels)
                self.data_sources['pseudo_labels'] = True

            # 6. Create enhanced features
            integrated_df = self._create_enhanced_features(integrated_df)

            # 7. Data quality check
            quality_report = self._generate_quality_report(integrated_df)

            # 8. Save results
            self.integrated_data = integrated_df

            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"✅ Data integration completed, time taken: {processing_time:.2f} seconds")
            logger.info(f"📊 Integrated data: {len(integrated_df)} rows, {len(integrated_df.columns)} columns")

            return {
                'success': True,
                'integrated_data': integrated_df,
                'feature_mapping': self.feature_mapping,
                'data_sources': self.data_sources,
                'quality_report': quality_report,
                'processing_time': processing_time,
                'integration_log': self.integration_log
            }

        except Exception as e:
            logger.error(f"❌ Data integration failed: {e}")
            return self._empty_result(f"Integration failed: {str(e)}")
    
    def _validate_inputs(self, engineered_features, clustering_results, risk_scores, pseudo_labels) -> bool:
        """验证输入数据"""
        try:
            # 检查基础特征数据
            if engineered_features is None or engineered_features.empty:
                logger.error("特征工程数据为空")
                return False
            
            # 检查数据长度一致性
            base_length = len(engineered_features)
            
            if clustering_results and 'cluster_labels' in clustering_results:
                cluster_labels = clustering_results['cluster_labels']
                if len(cluster_labels) != base_length:
                    logger.warning(f"聚类标签长度不匹配: {len(cluster_labels)} vs {base_length}")
            
            if risk_scores and 'detailed_results' in risk_scores:
                risk_results = risk_scores['detailed_results']
                if len(risk_results) != base_length:
                    logger.warning(f"风险评分长度不匹配: {len(risk_results)} vs {base_length}")
            
            if pseudo_labels and 'labels' in pseudo_labels:
                labels = pseudo_labels['labels']
                if len(labels) != base_length:
                    logger.warning(f"伪标签长度不匹配: {len(labels)} vs {base_length}")
            
            return True
            
        except Exception as e:
            logger.error(f"输入验证失败: {e}")
            return False
    
    def _integrate_clustering_data(self, df: pd.DataFrame, clustering_results: Dict) -> pd.DataFrame:
        """整合聚类数据"""
        try:
            logger.info("🔗 整合聚类数据")
            
            # 添加聚类标签
            if 'cluster_labels' in clustering_results:
                cluster_labels = clustering_results['cluster_labels']
                df['cluster_id'] = cluster_labels[:len(df)]
                self.feature_mapping['cluster_id'] = 'clustering'
                self.integration_log.append("添加聚类标签")
            
            # 添加聚类统计信息
            if 'cluster_details' in clustering_results:
                cluster_details = clustering_results['cluster_details']
                cluster_stats = {}
                
                for detail in cluster_details:
                    cluster_id = detail.get('cluster_id', 0)
                    cluster_stats[cluster_id] = {
                        'size': detail.get('size', 0),
                        'fraud_rate': detail.get('fraud_rate', 0),
                        'avg_amount': detail.get('avg_transaction_amount', 0),
                        'risk_level': detail.get('risk_level', 'low')
                    }
                
                # 为每行添加聚类统计特征
                if 'cluster_id' in df.columns:
                    df['cluster_size'] = df['cluster_id'].map(lambda x: cluster_stats.get(x, {}).get('size', 0))
                    df['cluster_fraud_rate'] = df['cluster_id'].map(lambda x: cluster_stats.get(x, {}).get('fraud_rate', 0))
                    df['cluster_avg_amount'] = df['cluster_id'].map(lambda x: cluster_stats.get(x, {}).get('avg_amount', 0))
                    df['cluster_risk_level'] = df['cluster_id'].map(lambda x: cluster_stats.get(x, {}).get('risk_level', 'low'))
                    
                    # 编码风险等级
                    risk_level_mapping = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
                    df['cluster_risk_encoded'] = df['cluster_risk_level'].map(risk_level_mapping).fillna(0)
                    
                    self.feature_mapping.update({
                        'cluster_size': 'clustering',
                        'cluster_fraud_rate': 'clustering',
                        'cluster_avg_amount': 'clustering',
                        'cluster_risk_level': 'clustering',
                        'cluster_risk_encoded': 'clustering'
                    })
                    
                    self.integration_log.append("添加聚类统计特征")
            
            return df
            
        except Exception as e:
            logger.error(f"聚类数据整合失败: {e}")
            return df
    
    def _integrate_risk_scores(self, df: pd.DataFrame, risk_scores: Dict) -> pd.DataFrame:
        """整合风险评分数据"""
        try:
            logger.info("🎯 整合风险评分数据")
            
            if 'detailed_results' in risk_scores:
                detailed_results = risk_scores['detailed_results']
                
                # 提取风险评分信息
                risk_score_values = []
                risk_levels = []
                risk_confidences = []
                
                for result in detailed_results[:len(df)]:
                    risk_score_values.append(result.get('risk_score', 0))
                    risk_levels.append(result.get('risk_level', 'low'))
                    risk_confidences.append(result.get('confidence', 0.5))
                
                # 添加风险评分特征
                df['risk_score'] = risk_score_values
                df['risk_level'] = risk_levels
                df['risk_confidence'] = risk_confidences
                
                # 编码风险等级
                risk_level_mapping = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
                df['risk_level_encoded'] = df['risk_level'].map(risk_level_mapping).fillna(0)
                
                # 创建风险区间特征
                df['is_high_risk'] = (df['risk_score'] >= 70).astype(int)
                df['is_medium_risk'] = ((df['risk_score'] >= 40) & (df['risk_score'] < 70)).astype(int)
                df['is_low_risk'] = (df['risk_score'] < 40).astype(int)
                
                self.feature_mapping.update({
                    'risk_score': 'risk_scoring',
                    'risk_level': 'risk_scoring',
                    'risk_confidence': 'risk_scoring',
                    'risk_level_encoded': 'risk_scoring',
                    'is_high_risk': 'risk_scoring',
                    'is_medium_risk': 'risk_scoring',
                    'is_low_risk': 'risk_scoring'
                })
                
                self.integration_log.append("添加风险评分特征")
            
            return df
            
        except Exception as e:
            logger.error(f"风险评分数据整合失败: {e}")
            return df
    
    def _integrate_pseudo_labels(self, df: pd.DataFrame, pseudo_labels: Dict) -> pd.DataFrame:
        """整合伪标签数据"""
        try:
            logger.info("🏷️ 整合伪标签数据")
            
            # 添加伪标签
            if 'labels' in pseudo_labels:
                labels = pseudo_labels['labels']
                df['pseudo_label'] = labels[:len(df)]
                self.feature_mapping['pseudo_label'] = 'pseudo_labeling'
                self.integration_log.append("添加伪标签")
            
            # 添加置信度
            if 'confidences' in pseudo_labels:
                confidences = pseudo_labels['confidences']
                df['pseudo_confidence'] = confidences[:len(df)]
                
                # 创建置信度区间特征
                df['is_high_confidence'] = (df['pseudo_confidence'] >= 0.8).astype(int)
                df['is_medium_confidence'] = ((df['pseudo_confidence'] >= 0.6) & (df['pseudo_confidence'] < 0.8)).astype(int)
                df['is_low_confidence'] = (df['pseudo_confidence'] < 0.6).astype(int)
                
                self.feature_mapping.update({
                    'pseudo_confidence': 'pseudo_labeling',
                    'is_high_confidence': 'pseudo_labeling',
                    'is_medium_confidence': 'pseudo_labeling',
                    'is_low_confidence': 'pseudo_labeling'
                })
                
                self.integration_log.append("添加伪标签置信度特征")
            
            return df

        except Exception as e:
            logger.error(f"伪标签数据整合失败: {e}")
            return df

    def _create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建增强特征"""
        try:
            logger.info("⚡ 创建增强特征")

            # 1. 聚类-风险交互特征
            if 'cluster_fraud_rate' in df.columns and 'risk_score' in df.columns:
                df['cluster_risk_interaction'] = df['cluster_fraud_rate'] * df['risk_score'] / 100
                self.feature_mapping['cluster_risk_interaction'] = 'enhanced'

            # 2. 风险-置信度一致性特征
            if 'risk_score' in df.columns and 'pseudo_confidence' in df.columns:
                df['risk_confidence_alignment'] = abs(df['risk_score'] / 100 - df['pseudo_confidence'])
                df['is_consistent_prediction'] = (df['risk_confidence_alignment'] < 0.3).astype(int)
                self.feature_mapping.update({
                    'risk_confidence_alignment': 'enhanced',
                    'is_consistent_prediction': 'enhanced'
                })

            # 3. 多维度风险综合评分
            risk_components = []
            if 'risk_score' in df.columns:
                risk_components.append('risk_score')
            if 'cluster_fraud_rate' in df.columns:
                risk_components.append('cluster_fraud_rate')

            if len(risk_components) >= 2:
                # 标准化后加权平均
                for component in risk_components:
                    df[f'{component}_normalized'] = (df[component] - df[component].min()) / (df[component].max() - df[component].min() + 1e-8)

                df['composite_risk_score'] = (
                    df.get('risk_score_normalized', 0) * 0.7 +
                    df.get('cluster_fraud_rate_normalized', 0) * 0.3
                ) * 100

                self.feature_mapping['composite_risk_score'] = 'enhanced'

            # 4. 异常检测特征
            if 'transaction_amount' in df.columns and 'cluster_avg_amount' in df.columns:
                df['amount_deviation_from_cluster'] = abs(df['transaction_amount'] - df['cluster_avg_amount'])
                df['is_amount_outlier'] = (df['amount_deviation_from_cluster'] > df['cluster_avg_amount'] * 2).astype(int)
                self.feature_mapping.update({
                    'amount_deviation_from_cluster': 'enhanced',
                    'is_amount_outlier': 'enhanced'
                })

            # 5. 标签质量评估特征
            if 'pseudo_label' in df.columns and 'risk_level_encoded' in df.columns:
                # 标签与风险等级的一致性
                df['label_risk_consistency'] = (df['pseudo_label'] == (df['risk_level_encoded'] >= 2).astype(int)).astype(int)
                self.feature_mapping['label_risk_consistency'] = 'enhanced'

            self.integration_log.append("创建增强特征")
            return df

        except Exception as e:
            logger.error(f"Enhanced feature creation failed: {e}")
            return df

    def _generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality report"""
        try:
            report = {
                'total_samples': len(df),
                'total_features': len(df.columns),
                'feature_sources': {},
                'data_completeness': {},
                'feature_statistics': {}
            }

            # Count features by source
            for feature, source in self.feature_mapping.items():
                if source not in report['feature_sources']:
                    report['feature_sources'][source] = 0
                report['feature_sources'][source] += 1

            # Data completeness check
            for column in df.columns:
                missing_count = df[column].isnull().sum()
                report['data_completeness'][column] = {
                    'missing_count': int(missing_count),
                    'missing_rate': float(missing_count / len(df))
                }

            # Key feature statistics
            key_features = ['risk_score', 'pseudo_confidence', 'cluster_fraud_rate', 'composite_risk_score']
            for feature in key_features:
                if feature in df.columns:
                    report['feature_statistics'][feature] = {
                        'mean': float(df[feature].mean()),
                        'std': float(df[feature].std()),
                        'min': float(df[feature].min()),
                        'max': float(df[feature].max())
                    }

            return report

        except Exception as e:
            logger.error(f"Quality report generation failed: {e}")
            return {'error': str(e)}

    def _empty_result(self, error_message: str) -> Dict[str, Any]:
        """Return empty result"""
        return {
            'success': False,
            'error': error_message,
            'integrated_data': None,
            'feature_mapping': {},
            'data_sources': {},
            'quality_report': {},
            'processing_time': 0,
            'integration_log': []
        }

    def get_feature_importance_mapping(self) -> Dict[str, List[str]]:
        """获取特征重要性映射"""
        importance_mapping = {
            'high_importance': [],
            'medium_importance': [],
            'low_importance': []
        }

        # 高重要性特征
        high_importance_features = [
            'risk_score', 'composite_risk_score', 'pseudo_confidence',
            'cluster_fraud_rate', 'risk_confidence_alignment'
        ]

        # 中等重要性特征
        medium_importance_features = [
            'cluster_risk_interaction', 'is_consistent_prediction',
            'label_risk_consistency', 'is_amount_outlier'
        ]

        for feature in self.feature_mapping.keys():
            if feature in high_importance_features:
                importance_mapping['high_importance'].append(feature)
            elif feature in medium_importance_features:
                importance_mapping['medium_importance'].append(feature)
            else:
                importance_mapping['low_importance'].append(feature)

        return importance_mapping

    def export_integrated_data(self, output_path: str = None) -> bool:
        """导出整合后的数据"""
        try:
            if self.integrated_data is None:
                logger.error("没有可导出的整合数据")
                return False

            if output_path is None:
                output_path = f"data/integrated/unified_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            self.integrated_data.to_csv(output_path, index=False)
            logger.info(f"✅ 整合数据已导出到: {output_path}")
            return True

        except Exception as e:
            logger.error(f"数据导出失败: {e}")
            return False
