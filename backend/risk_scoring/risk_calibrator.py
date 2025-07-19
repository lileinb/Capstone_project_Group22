"""
风险评分校准器
使用少量真实标签校准无监督风险评分系统
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskCalibrator:
    """风险评分校准器"""
    
    def __init__(self):
        """初始化校准器"""
        self.calibration_history = []
        self.best_thresholds = None
        self.calibration_metrics = None
        
    def calibrate_risk_scoring(self, data: pd.DataFrame, 
                             unsupervised_results: Dict[str, Any],
                             sample_ratio: float = 0.1,
                             validation_ratio: float = 0.2) -> Dict[str, Any]:
        """
        校准风险评分系统
        
        Args:
            data: 包含真实标签的完整数据
            unsupervised_results: 无监督风险评分结果
            sample_ratio: 用于校准的标签比例
            validation_ratio: 用于验证的数据比例
            
        Returns:
            校准结果
        """
        try:
            if 'is_fraudulent' not in data.columns:
                logger.error("数据中缺少真实标签列 'is_fraudulent'")
                return self._empty_calibration_result()
            
            logger.info(f"开始风险评分校准，使用 {sample_ratio*100:.1f}% 标签进行校准")
            
            # 1. 数据准备和分割
            calibration_data, validation_data = self._prepare_calibration_data(
                data, unsupervised_results, sample_ratio, validation_ratio
            )
            
            if not calibration_data or not validation_data:
                logger.error("数据准备失败")
                return self._empty_calibration_result()
            
            # 2. 分析当前评分与真实标签的关系
            score_analysis = self._analyze_score_distribution(calibration_data)
            
            # 3. 优化风险阈值
            threshold_optimization = self._optimize_thresholds(calibration_data)
            
            # 4. 在验证集上评估性能
            validation_results = self._validate_calibration(
                validation_data, threshold_optimization['optimized_thresholds']
            )
            
            # 5. 生成校准报告
            calibration_report = self._generate_calibration_report(
                score_analysis, threshold_optimization, validation_results
            )
            
            # 6. 记录校准历史
            self._record_calibration_history(calibration_report)
            
            logger.info("风险评分校准完成")
            return calibration_report
            
        except Exception as e:
            logger.error(f"风险评分校准失败: {e}")
            return self._empty_calibration_result()
    
    def _prepare_calibration_data(self, data: pd.DataFrame, 
                                unsupervised_results: Dict[str, Any],
                                sample_ratio: float, 
                                validation_ratio: float) -> Tuple[Dict, Dict]:
        """准备校准和验证数据"""
        try:
            # 获取无监督评分结果
            risk_results = unsupervised_results.get('results', [])
            if not risk_results:
                logger.error("无监督评分结果为空")
                return None, None
            
            # 构建评分和标签对应关系
            scores = []
            labels = []
            indices = []
            
            for i, result in enumerate(risk_results):
                if i < len(data):
                    scores.append(result.get('risk_score', 0))
                    labels.append(data.iloc[i]['is_fraudulent'])
                    indices.append(i)
            
            if not scores:
                logger.error("无法构建评分-标签对应关系")
                return None, None
            
            # 分层采样，确保欺诈和正常样本的比例
            splitter = StratifiedShuffleSplit(
                n_splits=1, 
                test_size=validation_ratio, 
                random_state=42
            )
            
            train_idx, val_idx = next(splitter.split(scores, labels))
            
            # 从训练集中采样校准数据
            calibration_size = int(len(train_idx) * sample_ratio)
            calibration_indices = np.random.choice(
                train_idx, size=calibration_size, replace=False
            )
            
            # 构建校准数据
            calibration_data = {
                'scores': [scores[i] for i in calibration_indices],
                'labels': [labels[i] for i in calibration_indices],
                'indices': [indices[i] for i in calibration_indices],
                'size': len(calibration_indices)
            }
            
            # 构建验证数据
            validation_data = {
                'scores': [scores[i] for i in val_idx],
                'labels': [labels[i] for i in val_idx],
                'indices': [indices[i] for i in val_idx],
                'size': len(val_idx)
            }
            
            logger.info(f"数据准备完成 - 校准集: {calibration_data['size']}, 验证集: {validation_data['size']}")
            return calibration_data, validation_data
            
        except Exception as e:
            logger.error(f"数据准备失败: {e}")
            return None, None
    
    def _analyze_score_distribution(self, calibration_data: Dict) -> Dict[str, Any]:
        """分析评分分布"""
        scores = np.array(calibration_data['scores'])
        labels = np.array(calibration_data['labels'])
        
        # 欺诈和正常交易的评分统计
        fraud_scores = scores[labels == 1]
        normal_scores = scores[labels == 0]
        
        analysis = {
            'total_samples': len(scores),
            'fraud_samples': len(fraud_scores),
            'normal_samples': len(normal_scores),
            'fraud_rate': len(fraud_scores) / len(scores),
            
            'fraud_score_stats': {
                'mean': float(np.mean(fraud_scores)) if len(fraud_scores) > 0 else 0,
                'std': float(np.std(fraud_scores)) if len(fraud_scores) > 0 else 0,
                'median': float(np.median(fraud_scores)) if len(fraud_scores) > 0 else 0,
                'min': float(np.min(fraud_scores)) if len(fraud_scores) > 0 else 0,
                'max': float(np.max(fraud_scores)) if len(fraud_scores) > 0 else 0
            },
            
            'normal_score_stats': {
                'mean': float(np.mean(normal_scores)) if len(normal_scores) > 0 else 0,
                'std': float(np.std(normal_scores)) if len(normal_scores) > 0 else 0,
                'median': float(np.median(normal_scores)) if len(normal_scores) > 0 else 0,
                'min': float(np.min(normal_scores)) if len(normal_scores) > 0 else 0,
                'max': float(np.max(normal_scores)) if len(normal_scores) > 0 else 0
            }
        }
        
        # 计算分离度
        if len(fraud_scores) > 0 and len(normal_scores) > 0:
            analysis['score_separation'] = float(
                abs(np.mean(fraud_scores) - np.mean(normal_scores))
            )
            
            # 计算重叠度
            fraud_range = (np.min(fraud_scores), np.max(fraud_scores))
            normal_range = (np.min(normal_scores), np.max(normal_scores))
            
            overlap_start = max(fraud_range[0], normal_range[0])
            overlap_end = min(fraud_range[1], normal_range[1])
            
            if overlap_start < overlap_end:
                overlap_ratio = (overlap_end - overlap_start) / (
                    max(fraud_range[1], normal_range[1]) - min(fraud_range[0], normal_range[0])
                )
                analysis['score_overlap_ratio'] = float(overlap_ratio)
            else:
                analysis['score_overlap_ratio'] = 0.0
        else:
            analysis['score_separation'] = 0.0
            analysis['score_overlap_ratio'] = 1.0
        
        return analysis
    
    def _optimize_thresholds(self, calibration_data: Dict) -> Dict[str, Any]:
        """优化风险阈值"""
        scores = np.array(calibration_data['scores'])
        labels = np.array(calibration_data['labels'])
        
        optimization_results = {}
        
        try:
            # 计算ROC曲线
            fpr, tpr, thresholds = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            
            # 找到最优阈值（Youden's J statistic）
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # 基于最优阈值设计风险等级阈值
            optimized_thresholds = {
                'low': max(5, optimal_threshold - 25),
                'medium': max(20, optimal_threshold - 10),
                'high': optimal_threshold,
                'critical': min(95, optimal_threshold + 15)
            }
            
            # 计算不同阈值下的性能
            threshold_performance = {}
            for level, threshold in optimized_thresholds.items():
                predictions = (scores >= threshold).astype(int)
                
                if len(np.unique(predictions)) > 1:
                    performance = {
                        'accuracy': float(accuracy_score(labels, predictions)),
                        'precision': float(precision_score(labels, predictions, zero_division=0)),
                        'recall': float(recall_score(labels, predictions, zero_division=0)),
                        'f1_score': float(f1_score(labels, predictions, zero_division=0))
                    }
                else:
                    performance = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
                
                threshold_performance[level] = performance
            
            optimization_results = {
                'roc_auc': float(roc_auc),
                'optimal_threshold': float(optimal_threshold),
                'optimized_thresholds': optimized_thresholds,
                'threshold_performance': threshold_performance,
                'roc_curve_data': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"阈值优化失败: {e}")
            # 使用默认阈值
            optimization_results = {
                'roc_auc': 0.5,
                'optimal_threshold': 50.0,
                'optimized_thresholds': {'low': 25, 'medium': 50, 'high': 75, 'critical': 90},
                'threshold_performance': {},
                'roc_curve_data': {'fpr': [], 'tpr': [], 'thresholds': []}
            }
        
        return optimization_results
    
    def _validate_calibration(self, validation_data: Dict, 
                            optimized_thresholds: Dict[str, float]) -> Dict[str, Any]:
        """在验证集上验证校准效果"""
        scores = np.array(validation_data['scores'])
        labels = np.array(validation_data['labels'])
        
        validation_results = {}
        
        # 使用高风险阈值进行二分类预测
        high_threshold = optimized_thresholds.get('high', 75)
        predictions = (scores >= high_threshold).astype(int)
        
        # 计算性能指标
        try:
            validation_results['performance_metrics'] = {
                'accuracy': float(accuracy_score(labels, predictions)),
                'precision': float(precision_score(labels, predictions, zero_division=0)),
                'recall': float(recall_score(labels, predictions, zero_division=0)),
                'f1_score': float(f1_score(labels, predictions, zero_division=0))
            }
            
            # 混淆矩阵
            cm = confusion_matrix(labels, predictions)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                validation_results['confusion_matrix'] = {
                    'true_negative': int(tn),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'true_positive': int(tp)
                }
                
                # 计算特异性
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                validation_results['performance_metrics']['specificity'] = float(specificity)
            
            # 分类报告
            validation_results['classification_report'] = classification_report(
                labels, predictions, output_dict=True, zero_division=0
            )
            
        except Exception as e:
            logger.error(f"验证指标计算失败: {e}")
            validation_results = {
                'performance_metrics': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0},
                'confusion_matrix': {'true_negative': 0, 'false_positive': 0, 'false_negative': 0, 'true_positive': 0},
                'classification_report': {}
            }
        
        return validation_results
    
    def _generate_calibration_report(self, score_analysis: Dict, 
                                   threshold_optimization: Dict,
                                   validation_results: Dict) -> Dict[str, Any]:
        """生成校准报告"""
        return {
            'calibration_timestamp': datetime.now().isoformat(),
            'score_analysis': score_analysis,
            'threshold_optimization': threshold_optimization,
            'validation_results': validation_results,
            'calibration_summary': {
                'data_quality': self._assess_data_quality(score_analysis),
                'optimization_success': threshold_optimization.get('roc_auc', 0) > 0.6,
                'validation_performance': validation_results.get('performance_metrics', {}).get('f1_score', 0),
                'recommended_action': self._get_recommendation(
                    score_analysis, threshold_optimization, validation_results
                )
            }
        }
    
    def _assess_data_quality(self, score_analysis: Dict) -> str:
        """评估数据质量"""
        separation = score_analysis.get('score_separation', 0)
        overlap = score_analysis.get('score_overlap_ratio', 1)
        fraud_rate = score_analysis.get('fraud_rate', 0)
        
        if separation > 30 and overlap < 0.3 and 0.01 < fraud_rate < 0.2:
            return 'excellent'
        elif separation > 20 and overlap < 0.5 and 0.005 < fraud_rate < 0.3:
            return 'good'
        elif separation > 10 and overlap < 0.7:
            return 'fair'
        else:
            return 'poor'
    
    def _get_recommendation(self, score_analysis: Dict, 
                          threshold_optimization: Dict,
                          validation_results: Dict) -> str:
        """获取推荐行动"""
        roc_auc = threshold_optimization.get('roc_auc', 0.5)
        f1_score = validation_results.get('performance_metrics', {}).get('f1_score', 0)
        data_quality = self._assess_data_quality(score_analysis)
        
        if roc_auc > 0.8 and f1_score > 0.7 and data_quality in ['excellent', 'good']:
            return 'apply_calibration'
        elif roc_auc > 0.6 and f1_score > 0.5:
            return 'apply_with_caution'
        else:
            return 'improve_features'
    
    def _record_calibration_history(self, calibration_report: Dict):
        """记录校准历史"""
        history_record = {
            'timestamp': calibration_report['calibration_timestamp'],
            'roc_auc': calibration_report['threshold_optimization'].get('roc_auc', 0),
            'validation_f1': calibration_report['validation_results'].get('performance_metrics', {}).get('f1_score', 0),
            'data_quality': calibration_report['calibration_summary']['data_quality'],
            'recommendation': calibration_report['calibration_summary']['recommended_action']
        }
        
        self.calibration_history.append(history_record)
        
        # 保持历史记录不超过50条
        if len(self.calibration_history) > 50:
            self.calibration_history = self.calibration_history[-50:]
    
    def _empty_calibration_result(self) -> Dict[str, Any]:
        """返回空的校准结果"""
        return {
            'calibration_timestamp': datetime.now().isoformat(),
            'score_analysis': {},
            'threshold_optimization': {
                'roc_auc': 0.5,
                'optimal_threshold': 50.0,
                'optimized_thresholds': {'low': 25, 'medium': 50, 'high': 75, 'critical': 90},
                'threshold_performance': {},
                'roc_curve_data': {'fpr': [], 'tpr': [], 'thresholds': []}
            },
            'validation_results': {
                'performance_metrics': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0},
                'confusion_matrix': {'true_negative': 0, 'false_positive': 0, 'false_negative': 0, 'true_positive': 0},
                'classification_report': {}
            },
            'calibration_summary': {
                'data_quality': 'poor',
                'optimization_success': False,
                'validation_performance': 0,
                'recommended_action': 'improve_features'
            }
        }
    
    def get_calibration_history(self) -> List[Dict]:
        """获取校准历史"""
        return self.calibration_history.copy()
    
    def apply_best_calibration(self) -> Optional[Dict[str, float]]:
        """应用最佳校准结果"""
        if not self.calibration_history:
            return None
        
        # 找到F1分数最高的校准结果
        best_calibration = max(self.calibration_history, key=lambda x: x.get('validation_f1', 0))
        
        if best_calibration.get('validation_f1', 0) > 0.5:
            return best_calibration
        else:
            return None
