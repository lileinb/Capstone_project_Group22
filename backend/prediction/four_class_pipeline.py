"""
Four-Class Prediction Pipeline
End-to-end four-level risk classification prediction system
Integrates clustering, semi-supervised label generation, model prediction and dynamic thresholds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FourClassPredictionPipeline:
    """Four-class prediction pipeline"""

    def __init__(self):
        """Initialize four-class prediction pipeline"""
        self.target_classes = ['low', 'medium', 'high', 'critical']
        self.class_mapping = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}

        # Initialize components
        self._initialize_components()

        logger.info("Four-class prediction pipeline initialization completed")
    
    def _initialize_components(self):
        """Initialize all components"""
        try:
            # Clustering analyzer
            from backend.clustering.cluster_analyzer import ClusterAnalyzer
            self.cluster_analyzer = ClusterAnalyzer()

            # Semi-supervised label generator
            from backend.pseudo_labeling.semi_supervised_generator import SemiSupervisedLabelGenerator
            self.label_generator = SemiSupervisedLabelGenerator()

            # Four-class model manager
            from backend.ml_models.four_class_model_manager import FourClassModelManager
            self.model_manager = FourClassModelManager()

            # Four-class risk calculator
            from backend.risk_scoring.four_class_risk_calculator import FourClassRiskCalculator
            self.risk_calculator = FourClassRiskCalculator(enable_dynamic_thresholds=True)

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    def predict_risk_levels(self, data: pd.DataFrame, 
                          use_clustering: bool = True,
                          use_models: bool = True,
                          use_dynamic_thresholds: bool = True) -> Dict[str, Any]:
        """
        端到端四分类风险预测
        
        Args:
            data: 输入数据
            use_clustering: 是否使用聚类分析
            use_models: 是否使用训练好的模型
            use_dynamic_thresholds: 是否使用动态阈值
            
        Returns:
            四分类预测结果
        """
        try:
            if data is None or data.empty:
                logger.error("输入数据为空")
                return self._empty_result()
            
            start_time = datetime.now()
            logger.info(f"开始四分类风险预测，数据量: {len(data)}")
            
            # 第一步：聚类分析（可选）
            cluster_results = None
            if use_clustering:
                logger.info("执行聚类分析...")
                cluster_results = self.cluster_analyzer.analyze_clusters(
                    data, algorithm='kmeans'
                )
                if cluster_results.get('cluster_count', 0) > 0:
                    logger.info(f"聚类完成，发现 {cluster_results['cluster_count']} 个聚类")
                else:
                    logger.warning("聚类分析失败，将跳过聚类信息")
                    cluster_results = None
            
            # 第二步：生成四分类标签
            logger.info("生成四分类标签...")
            label_results = self.label_generator.generate_four_class_labels(
                data, cluster_results=cluster_results
            )
            
            if not label_results['success']:
                logger.warning("Four-class label generation failed, using fallback method")
                label_results = self._generate_fallback_labels(data)

            # Step 3: Model prediction (optional)
            model_predictions = None
            if use_models and self.model_manager.get_available_models():
                logger.info("Executing model prediction...")

                # Prepare feature data
                feature_data = self._prepare_feature_data(data)

                if feature_data is not None:
                    model_predictions = self.model_manager.predict_four_class(
                        feature_data, use_ensemble=True
                    )

                    if model_predictions.get('predictions'):
                        logger.info("Model prediction completed")
                    else:
                        logger.warning("Model prediction failed")
                        model_predictions = None
                else:
                    logger.warning("Feature data preparation failed, skipping model prediction")
            
            # 第四步：风险评分计算
            logger.info("计算风险评分...")
            risk_results = self.risk_calculator.calculate_four_class_risk_scores(
                data, 
                cluster_results=cluster_results,
                model_predictions=model_predictions
            )
            
            if not risk_results['success']:
                logger.error("风险评分计算失败")
                return self._empty_result()
            
            # 第五步：结果整合
            final_result = self._integrate_results(
                label_results, model_predictions, risk_results, data
            )
            
            prediction_time = (datetime.now() - start_time).total_seconds()
            final_result['prediction_time'] = prediction_time
            
            logger.info(f"四分类风险预测完成，耗时: {prediction_time:.2f}秒")
            return final_result
            
        except Exception as e:
            logger.error(f"四分类风险预测失败: {e}")
            return self._empty_result()
    
    def _prepare_feature_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """准备模型特征数据"""
        try:
            # 选择数值特征
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # 排除标签列
            if 'is_fraudulent' in numeric_columns:
                numeric_columns.remove('is_fraudulent')
            
            if not numeric_columns:
                logger.warning("没有找到数值特征")
                return None
            
            feature_data = data[numeric_columns].copy()
            
            # 处理缺失值和无穷值
            feature_data = feature_data.fillna(0)
            feature_data = feature_data.replace([np.inf, -np.inf], 0)
            
            return feature_data
            
        except Exception as e:
            logger.warning(f"特征数据准备失败: {e}")
            return None
    
    def _generate_fallback_labels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成备用标签（当半监督生成失败时）"""
        try:
            n_samples = len(data)
            
            # 简单的基于规则的标签生成
            labels = []
            confidences = []
            
            for i in range(n_samples):
                score = 0
                
                # 基于一些基础特征判断
                amount = data.get('transaction_amount', pd.Series([100] * n_samples)).iloc[i]
                if amount > 5000:
                    score += 3
                elif amount > 1000:
                    score += 2
                elif amount > 500:
                    score += 1
                
                account_age = data.get('account_age_days', pd.Series([365] * n_samples)).iloc[i]
                if account_age < 7:
                    score += 2
                elif account_age < 30:
                    score += 1
                
                # 转换为风险等级
                if score >= 4:
                    labels.append(3)  # critical
                    confidences.append(0.7)
                elif score >= 3:
                    labels.append(2)  # high
                    confidences.append(0.6)
                elif score >= 1:
                    labels.append(1)  # medium
                    confidences.append(0.5)
                else:
                    labels.append(0)  # low
                    confidences.append(0.8)
            
            # 计算分布
            distribution = {}
            for i, class_name in enumerate(self.target_classes):
                count = labels.count(i)
                distribution[class_name] = {
                    'count': count,
                    'percentage': float(count / len(labels) * 100)
                }
            
            return {
                'success': True,
                'labels': labels,
                'confidences': confidences,
                'class_labels': [self.target_classes[label] for label in labels],
                'distribution': distribution,
                'method': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"备用标签生成失败: {e}")
            return {'success': False}
    
    def _integrate_results(self, label_results: Dict, model_predictions: Optional[Dict],
                          risk_results: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """整合所有结果"""
        try:
            n_samples = len(data)
            
            # 基础结果来自风险评分计算器
            integrated_result = risk_results.copy()
            
            # 添加标签生成信息
            if label_results['success']:
                integrated_result['label_generation'] = {
                    'method': label_results.get('method', 'semi_supervised'),
                    'distribution': label_results['distribution'],
                    'avg_confidence': np.mean(label_results['confidences'])
                }
            
            # 添加模型预测信息
            if model_predictions and model_predictions.get('predictions'):
                integrated_result['model_predictions'] = {
                    'available_models': list(model_predictions['predictions'].keys()),
                    'model_distributions': model_predictions['distribution'],
                    'confidence_stats': model_predictions['confidence_stats']
                }
                
                # 如果有集成模型结果，添加详细信息
                if 'ensemble' in model_predictions['predictions']:
                    ensemble_preds = model_predictions['predictions']['ensemble']
                    ensemble_levels = [self.target_classes[pred] for pred in ensemble_preds]
                    
                    # 计算模型预测与风险评分的一致性
                    risk_levels = [result['risk_level'] for result in integrated_result['detailed_results']]
                    consistency = sum(1 for i in range(len(ensemble_levels)) 
                                    if ensemble_levels[i] == risk_levels[i]) / len(ensemble_levels)
                    
                    integrated_result['model_consistency'] = float(consistency)
            
            # 添加预测方法信息
            integrated_result['prediction_methods'] = {
                'clustering_used': label_results.get('method') != 'fallback',
                'models_used': model_predictions is not None,
                'dynamic_thresholds_used': integrated_result['threshold_type'] == 'dynamic'
            }
            
            # 添加质量评估
            integrated_result['quality_assessment'] = self._assess_prediction_quality(
                integrated_result, label_results, model_predictions
            )
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"结果整合失败: {e}")
            return risk_results
    
    def _assess_prediction_quality(self, integrated_result: Dict, 
                                 label_results: Dict, 
                                 model_predictions: Optional[Dict]) -> Dict[str, Any]:
        """评估预测质量"""
        try:
            quality_score = 0
            quality_factors = []
            
            # 1. 分布合理性 (30分)
            distribution = integrated_result['distribution']
            target_dist = {'low': 60, 'medium': 25, 'high': 12, 'critical': 3}
            
            dist_score = 0
            for class_name, target_pct in target_dist.items():
                actual_pct = distribution[class_name]['percentage']
                deviation = abs(actual_pct - target_pct) / target_pct
                dist_score += max(0, 30 - deviation * 30) / 4
            
            quality_score += dist_score
            quality_factors.append(f"分布合理性: {dist_score:.1f}/30")
            
            # 2. 标签生成质量 (25分)
            if label_results['success']:
                label_confidence = np.mean(label_results['confidences'])
                label_score = label_confidence * 25
                quality_score += label_score
                quality_factors.append(f"标签质量: {label_score:.1f}/25")
            
            # 3. 模型一致性 (25分)
            if model_predictions and 'model_consistency' in integrated_result:
                consistency_score = integrated_result['model_consistency'] * 25
                quality_score += consistency_score
                quality_factors.append(f"模型一致性: {consistency_score:.1f}/25")
            else:
                quality_factors.append("模型一致性: 0/25 (无模型预测)")
            
            # 4. 阈值优化 (20分)
            if integrated_result['threshold_type'] == 'dynamic':
                threshold_score = 20
                if 'distribution_analysis' in integrated_result:
                    if integrated_result['distribution_analysis'].get('is_reasonable', False):
                        threshold_score = 20
                    else:
                        threshold_score = 15
            else:
                threshold_score = 10
            
            quality_score += threshold_score
            quality_factors.append(f"阈值优化: {threshold_score}/20")
            
            # 质量等级
            if quality_score >= 80:
                quality_grade = 'A'
            elif quality_score >= 70:
                quality_grade = 'B'
            elif quality_score >= 60:
                quality_grade = 'C'
            else:
                quality_grade = 'D'
            
            return {
                'overall_score': float(quality_score),
                'quality_grade': quality_grade,
                'quality_factors': quality_factors,
                'recommendations': self._get_quality_recommendations(quality_score, integrated_result)
            }
            
        except Exception as e:
            logger.warning(f"质量评估失败: {e}")
            return {'overall_score': 50.0, 'quality_grade': 'C'}
    
    def _get_quality_recommendations(self, quality_score: float, 
                                   result: Dict) -> List[str]:
        """获取质量改进建议"""
        recommendations = []
        
        if quality_score < 70:
            recommendations.append("建议增加训练数据量以提高模型性能")
        
        if result['threshold_type'] == 'fixed':
            recommendations.append("建议启用动态阈值以优化风险分布")
        
        if 'model_predictions' not in result:
            recommendations.append("建议训练四分类模型以提高预测准确性")
        
        if result.get('model_consistency', 0) < 0.7:
            recommendations.append("建议重新训练模型或调整集成权重")
        
        return recommendations
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'success': False,
            'total_samples': 0,
            'distribution': {},
            'statistics': {},
            'detailed_results': [],
            'error': '预测失败'
        }
