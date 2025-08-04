"""
三层预测架构流水线
集成欺诈检测 → 四分类风险评级 → 攻击类型分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreeLayerPredictionPipeline:
    """三层预测架构流水线"""
    
    def __init__(self):
        """初始化三层预测流水线"""
        self.layers = ['fraud_detection', 'risk_classification', 'attack_analysis']
        
        # 初始化各层组件
        self._initialize_components()
        
        logger.info("三层预测架构流水线初始化完成")
    
    def _initialize_components(self):
        """Initialize components for each layer"""
        try:
            # First layer: Fraud detection
            from backend.clustering.cluster_analyzer import ClusterAnalyzer
            from backend.feature_engineer.risk_features import RiskFeatureEngineer
            self.cluster_analyzer = ClusterAnalyzer()
            self.feature_engineer = RiskFeatureEngineer()

            # Second layer: Four-class risk grading
            from backend.prediction.four_class_pipeline import FourClassPredictionPipeline
            self.four_class_pipeline = FourClassPredictionPipeline()

            # Third layer: Attack type analysis
            from backend.attack_classification.attack_classifier import AttackClassifier
            self.attack_classifier = AttackClassifier()

            logger.info("Three-layer prediction components initialized successfully")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    
    def predict_comprehensive(self, data: pd.DataFrame, 
                            enable_all_layers: bool = True) -> Dict[str, Any]:
        """
        执行完整的三层预测
        
        Args:
            data: 输入数据
            enable_all_layers: 是否启用所有层
            
        Returns:
            综合预测结果
        """
        try:
            if data is None or data.empty:
                logger.error("输入数据为空")
                return self._empty_result()
            
            start_time = datetime.now()
            logger.info(f"开始三层预测，数据量: {len(data)}")
            
            # 初始化结果
            comprehensive_result = {
                'success': True,
                'total_samples': len(data),
                'layers_executed': [],
                'layer_results': {},
                'integrated_results': [],
                'performance_metrics': {}
            }
            
            # 第一层：欺诈检测和特征工程
            layer1_result = self._execute_layer1_fraud_detection(data)
            comprehensive_result['layer_results']['fraud_detection'] = layer1_result
            comprehensive_result['layers_executed'].append('fraud_detection')
            
            if not layer1_result['success']:
                logger.error("第一层欺诈检测失败")
                return self._partial_result(comprehensive_result, "第一层失败")
            
            # 第二层：四分类风险评级
            layer2_result = self._execute_layer2_risk_classification(
                layer1_result['engineered_data'], 
                layer1_result['clustering_results']
            )
            comprehensive_result['layer_results']['risk_classification'] = layer2_result
            comprehensive_result['layers_executed'].append('risk_classification')
            
            if not layer2_result['success']:
                logger.warning("第二层风险分类失败，继续执行第三层")
            
            # 第三层：攻击类型分析
            if enable_all_layers:
                layer3_result = self._execute_layer3_attack_analysis(
                    layer1_result['engineered_data'],
                    layer2_result if layer2_result['success'] else None
                )
                comprehensive_result['layer_results']['attack_analysis'] = layer3_result
                comprehensive_result['layers_executed'].append('attack_analysis')
            
            # 整合所有层的结果
            integrated_results = self._integrate_layer_results(
                data, layer1_result, layer2_result, 
                layer3_result if enable_all_layers else None
            )
            comprehensive_result['integrated_results'] = integrated_results
            
            # 计算性能指标
            prediction_time = (datetime.now() - start_time).total_seconds()
            comprehensive_result['performance_metrics'] = {
                'total_time': prediction_time,
                'samples_per_second': len(data) / prediction_time,
                'layer_count': len(comprehensive_result['layers_executed'])
            }
            
            logger.info(f"三层预测完成，耗时: {prediction_time:.2f}秒")
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"三层预测失败: {e}")
            return self._empty_result()
    
    def _execute_layer1_fraud_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """执行第一层：欺诈检测"""
        try:
            logger.info("执行第一层：欺诈检测和特征工程")
            
            # 特征工程
            engineered_data = self.feature_engineer.engineer_features(data)
            
            if engineered_data is None or engineered_data.empty:
                return {'success': False, 'error': '特征工程失败'}
            
            # 聚类分析
            clustering_results = self.cluster_analyzer.analyze_clusters(
                engineered_data, algorithm='kmeans'
            )
            
            if not clustering_results or clustering_results.get('cluster_count', 0) == 0:
                return {'success': False, 'error': '聚类分析失败'}
            
            # 基础欺诈检测（基于聚类）
            fraud_indicators = self._calculate_fraud_indicators(
                engineered_data, clustering_results
            )
            
            return {
                'success': True,
                'engineered_data': engineered_data,
                'clustering_results': clustering_results,
                'fraud_indicators': fraud_indicators,
                'layer': 'fraud_detection'
            }
            
        except Exception as e:
            logger.error(f"第一层执行失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_layer2_risk_classification(self, engineered_data: pd.DataFrame,
                                          clustering_results: Dict) -> Dict[str, Any]:
        """执行第二层：四分类风险评级"""
        try:
            logger.info("执行第二层：四分类风险评级")
            
            # 使用四分类预测流水线
            risk_results = self.four_class_pipeline.predict_risk_levels(
                engineered_data,
                use_clustering=True,
                use_models=True,
                use_dynamic_thresholds=True
            )
            
            if not risk_results.get('success', False):
                return {'success': False, 'error': '四分类风险评级失败'}
            
            return {
                'success': True,
                'risk_results': risk_results,
                'layer': 'risk_classification'
            }
            
        except Exception as e:
            logger.error(f"第二层执行失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_layer3_attack_analysis(self, engineered_data: pd.DataFrame,
                                      risk_results: Optional[Dict] = None) -> Dict[str, Any]:
        """执行第三层：攻击类型分析"""
        try:
            logger.info("执行第三层：攻击类型分析")
            
            # 准备攻击分析的输入数据
            analysis_data = engineered_data.copy()
            
            # 如果有风险评级结果，添加风险信息
            if risk_results and risk_results.get('success'):
                detailed_results = risk_results['risk_results'].get('detailed_results', [])
                if detailed_results:
                    risk_scores = [r['risk_score'] for r in detailed_results]
                    risk_levels = [r['risk_level'] for r in detailed_results]
                    
                    analysis_data['risk_score'] = risk_scores[:len(analysis_data)]
                    analysis_data['risk_level'] = risk_levels[:len(analysis_data)]
            
            # 执行攻击类型分析
            attack_results = self.attack_classifier.classify_attacks(analysis_data)
            
            if not attack_results.get('success', False):
                return {'success': False, 'error': '攻击类型分析失败'}
            
            return {
                'success': True,
                'attack_results': attack_results,
                'layer': 'attack_analysis'
            }
            
        except Exception as e:
            logger.error(f"第三层执行失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_fraud_indicators(self, data: pd.DataFrame, 
                                  clustering_results: Dict) -> Dict[str, Any]:
        """计算欺诈指标"""
        try:
            # 基于聚类结果计算欺诈指标
            cluster_labels = clustering_results.get('cluster_labels', [])
            cluster_risk_mapping = clustering_results.get('cluster_risk_mapping', {})
            
            fraud_scores = []
            fraud_flags = []
            
            for i, cluster_id in enumerate(cluster_labels):
                if i >= len(data):
                    break
                
                cluster_info = cluster_risk_mapping.get(cluster_id, {})
                risk_level = cluster_info.get('risk_level', 'low')
                
                # 基于风险等级计算欺诈评分
                if risk_level == 'critical':
                    fraud_score = 0.9
                    fraud_flag = True
                elif risk_level == 'high':
                    fraud_score = 0.7
                    fraud_flag = True
                elif risk_level == 'medium':
                    fraud_score = 0.4
                    fraud_flag = False
                else:  # low
                    fraud_score = 0.1
                    fraud_flag = False
                
                fraud_scores.append(fraud_score)
                fraud_flags.append(fraud_flag)
            
            # 计算统计信息
            fraud_count = sum(fraud_flags)
            fraud_rate = fraud_count / len(fraud_flags) if fraud_flags else 0
            avg_fraud_score = np.mean(fraud_scores) if fraud_scores else 0
            
            return {
                'fraud_scores': fraud_scores,
                'fraud_flags': fraud_flags,
                'fraud_count': fraud_count,
                'fraud_rate': fraud_rate,
                'avg_fraud_score': avg_fraud_score
            }
            
        except Exception as e:
            logger.warning(f"欺诈指标计算失败: {e}")
            return {}
    
    def _integrate_layer_results(self, original_data: pd.DataFrame,
                               layer1_result: Dict, layer2_result: Dict,
                               layer3_result: Optional[Dict] = None) -> List[Dict]:
        """整合各层结果"""
        try:
            integrated_results = []
            n_samples = len(original_data)
            
            # 获取各层结果
            fraud_indicators = layer1_result.get('fraud_indicators', {})
            fraud_scores = fraud_indicators.get('fraud_scores', [0] * n_samples)
            fraud_flags = fraud_indicators.get('fraud_flags', [False] * n_samples)
            
            # 风险评级结果
            risk_results = None
            if layer2_result.get('success'):
                risk_results = layer2_result['risk_results'].get('detailed_results', [])
            
            # 攻击分析结果
            attack_results = None
            if layer3_result and layer3_result.get('success'):
                attack_results = layer3_result['attack_results'].get('detailed_results', [])
            
            # 整合每个样本的结果
            for i in range(n_samples):
                sample_result = {
                    'sample_index': i,
                    'fraud_detection': {
                        'fraud_score': fraud_scores[i] if i < len(fraud_scores) else 0,
                        'is_fraud': fraud_flags[i] if i < len(fraud_flags) else False
                    }
                }
                
                # 添加风险评级信息
                if risk_results and i < len(risk_results):
                    risk_info = risk_results[i]
                    sample_result['risk_classification'] = {
                        'risk_score': risk_info.get('risk_score', 0),
                        'risk_level': risk_info.get('risk_level', 'low'),
                        'risk_class': risk_info.get('risk_class', 0)
                    }
                else:
                    sample_result['risk_classification'] = {
                        'risk_score': 0,
                        'risk_level': 'unknown',
                        'risk_class': -1
                    }
                
                # Add attack analysis information
                if attack_results and i < len(attack_results):
                    attack_info = attack_results[i]
                    sample_result['attack_analysis'] = {
                        'attack_type': attack_info.get('attack_type', 'unknown'),
                        'attack_confidence': attack_info.get('confidence', 0),
                        'threat_level': attack_info.get('threat_level', 'low')
                    }
                else:
                    sample_result['attack_analysis'] = {
                        'attack_type': 'unknown',
                        'attack_confidence': 0,
                        'threat_level': 'unknown'
                    }
                
                # Calculate comprehensive threat score
                sample_result['comprehensive_threat'] = self._calculate_comprehensive_threat(
                    sample_result
                )
                
                integrated_results.append(sample_result)
            
            return integrated_results
            
        except Exception as e:
            logger.error(f"Result integration failed: {e}")
            return []
    
    def _calculate_comprehensive_threat(self, sample_result: Dict) -> Dict[str, Any]:
        """计算综合威胁评分"""
        try:
            # 获取各层评分
            fraud_score = sample_result['fraud_detection']['fraud_score']
            risk_score = sample_result['risk_classification']['risk_score'] / 100  # 归一化到0-1
            attack_confidence = sample_result['attack_analysis']['attack_confidence']
            
            # 加权计算综合威胁评分
            weights = {'fraud': 0.4, 'risk': 0.4, 'attack': 0.2}
            comprehensive_score = (
                fraud_score * weights['fraud'] +
                risk_score * weights['risk'] +
                attack_confidence * weights['attack']
            )
            
            # 确定威胁等级
            if comprehensive_score >= 0.8:
                threat_level = 'critical'
            elif comprehensive_score >= 0.6:
                threat_level = 'high'
            elif comprehensive_score >= 0.4:
                threat_level = 'medium'
            else:
                threat_level = 'low'
            
            return {
                'comprehensive_score': float(comprehensive_score),
                'threat_level': threat_level,
                'confidence': float(min(fraud_score, risk_score, attack_confidence)),
                'contributing_factors': {
                    'fraud_contribution': fraud_score * weights['fraud'],
                    'risk_contribution': risk_score * weights['risk'],
                    'attack_contribution': attack_confidence * weights['attack']
                }
            }
            
        except Exception as e:
            logger.warning(f"Comprehensive threat score calculation failed: {e}")
            return {
                'comprehensive_score': 0.0,
                'threat_level': 'unknown',
                'confidence': 0.0,
                'contributing_factors': {}
            }

    def _partial_result(self, result: Dict, error_msg: str) -> Dict[str, Any]:
        """Return partial result"""
        result['success'] = False
        result['error'] = error_msg
        return result
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'success': False,
            'total_samples': 0,
            'layers_executed': [],
            'layer_results': {},
            'integrated_results': [],
            'performance_metrics': {},
            'error': '预测失败'
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """获取流水线状态"""
        try:
            status = {
                'pipeline_ready': True,
                'components_status': {},
                'last_prediction': None,
                'total_predictions': 0
            }
            
            # 检查各组件状态
            components = [
                ('cluster_analyzer', self.cluster_analyzer),
                ('feature_engineer', self.feature_engineer),
                ('four_class_pipeline', self.four_class_pipeline),
                ('attack_classifier', self.attack_classifier)
            ]
            
            for name, component in components:
                try:
                    # 简单的健康检查
                    status['components_status'][name] = 'ready' if component else 'not_ready'
                except:
                    status['components_status'][name] = 'error'
            
            return status
            
        except Exception as e:
            logger.error(f"状态检查失败: {e}")
            return {'pipeline_ready': False, 'error': str(e)}
