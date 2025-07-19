"""
伪标签生成器
基于多种无监督和半监督方法生成高质量伪标签
支持风险评分、聚类分析、规则匹配等多种标签生成策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PseudoLabelGenerator:
    """伪标签生成器"""

    def __init__(self):
        """初始化伪标签生成器"""
        self.label_history = []
        self.confidence_threshold = 0.7
        
        # 标签生成策略配置
        self.strategies = {
            'risk_based': {
                'name': '基于风险评分',
                'description': '根据风险评分阈值生成标签',
                'weight': 0.3
            },
            'cluster_based': {
                'name': '基于聚类分析',
                'description': '根据聚类欺诈率生成标签',
                'weight': 0.25
            },
            'rule_based': {
                'name': '基于专家规则',
                'description': '根据业务规则生成标签',
                'weight': 0.25
            },
            'ensemble': {
                'name': '集成多策略',
                'description': '综合多种策略的结果',
                'weight': 0.2
            }
        }

    def generate_pseudo_labels(self, data: pd.DataFrame, strategy: str = 'ensemble') -> Dict[str, Any]:
        """
        生成伪标签
        
        Args:
            data: 输入数据
            strategy: 标签生成策略
            
        Returns:
            伪标签生成结果
        """
        if data is None or data.empty:
            logger.error("输入数据为空")
            return self._empty_result()

        try:
            logger.info(f"开始生成伪标签，策略: {strategy}")
            
            if strategy == 'risk_based':
                result = self._generate_risk_based_labels(data)
            elif strategy == 'cluster_based':
                result = self._generate_cluster_based_labels(data)
            elif strategy == 'rule_based':
                result = self._generate_rule_based_labels(data)
            elif strategy == 'ensemble':
                result = self._generate_ensemble_labels(data)
            else:
                logger.warning(f"未知策略: {strategy}，使用默认集成策略")
                result = self._generate_ensemble_labels(data)
            
            # 记录生成历史
            self._record_generation_history(result, strategy)
            
            logger.info(f"伪标签生成完成，生成了 {len(result.get('labels', []))} 个标签")
            return result
            
        except Exception as e:
            logger.error(f"伪标签生成失败: {e}")
            return self._empty_result()

    def _generate_risk_based_labels(self, data: pd.DataFrame,
                                  unsupervised_risk_results: Optional[Dict] = None) -> Dict[str, Any]:
        """基于无监督风险评分生成伪标签"""
        # 如果没有提供无监督风险评分结果，则计算
        if unsupervised_risk_results is None:
            # 延迟导入避免循环依赖
            try:
                from backend.risk_scoring.risk_calculator import UnsupervisedRiskCalculator
                from backend.clustering.cluster_analyzer import ClusterAnalyzer

                # 先进行聚类分析
                cluster_analyzer = ClusterAnalyzer(n_clusters=4, random_state=42)
                clustering_results = cluster_analyzer.analyze_clusters(data, algorithm='kmeans')

                # 计算无监督风险评分
                risk_calculator = UnsupervisedRiskCalculator()
                unsupervised_risk_results = risk_calculator.calculate_unsupervised_risk_score(
                    data, clustering_results
                )
            except ImportError as e:
                logger.warning(f"无法导入风险计算模块: {e}")
                return self._generate_fallback_risk_labels(data)

        labels = []
        confidences = []
        risk_factors_list = []

        results = unsupervised_risk_results.get('results', [])

        for result in results:
            risk_score = result.get('risk_score', 0)
            risk_level = result.get('risk_level', 'low')
            risk_factors = result.get('risk_factors', [])

            # 基于无监督风险评分生成二分类标签
            if risk_level == 'critical':
                label = 1  # 极高风险 -> 欺诈
                confidence = min(0.95, 0.85 + (risk_score - 75) / 100)
            elif risk_level == 'high':
                label = 1  # 高风险 -> 欺诈
                confidence = min(0.90, 0.70 + (risk_score - 50) / 100)
            elif risk_level == 'medium':
                # 中风险区间，基于具体评分决定
                if risk_score >= 60:
                    label = 1
                    confidence = 0.60 + (risk_score - 50) / 200
                else:
                    label = 0
                    confidence = 0.55 + (50 - risk_score) / 200
            else:  # low risk
                label = 0  # 低风险 -> 正常
                confidence = min(0.90, 0.70 + (50 - risk_score) / 100)

            # 基于风险因素数量调整置信度
            risk_factor_count = len(risk_factors)
            if risk_factor_count >= 3:
                confidence = min(0.95, confidence + 0.1)
            elif risk_factor_count >= 2:
                confidence = min(0.90, confidence + 0.05)

            labels.append(label)
            confidences.append(round(confidence, 3))
            risk_factors_list.append(risk_factors)

        # 计算质量指标
        high_confidence_labels = [l for l, c in zip(labels, confidences) if c >= self.confidence_threshold]

        return {
            'strategy': 'risk_based',
            'labels': labels,
            'confidences': confidences,
            'risk_factors': risk_factors_list,
            'label_distribution': pd.Series(labels).value_counts().to_dict(),
            'avg_confidence': round(np.mean(confidences), 3),
            'high_confidence_count': len(high_confidence_labels),
            'high_confidence_fraud_rate': sum(high_confidence_labels) / max(1, len(high_confidence_labels)),
            'metadata': {
                'avg_risk_score': unsupervised_risk_results.get('average_risk_score', 0),
                'risk_distribution': unsupervised_risk_results.get('risk_distribution', {}),
                'total_samples': len(labels),
                'confidence_threshold': self.confidence_threshold
            }
        }

    def _generate_cluster_based_labels(self, data: pd.DataFrame,
                                     clustering_results: Optional[Dict] = None) -> Dict[str, Any]:
        """基于聚类风险映射生成伪标签"""
        # 如果没有提供聚类结果，则计算
        if clustering_results is None:
            from backend.clustering.cluster_analyzer import ClusterAnalyzer

            cluster_analyzer = ClusterAnalyzer(n_clusters=4, random_state=42)
            clustering_results = cluster_analyzer.analyze_clusters(data, algorithm='kmeans')

        cluster_labels = clustering_results.get('cluster_labels', [])
        cluster_risk_mapping = clustering_results.get('cluster_risk_mapping', {})

        # 如果没有风险映射，使用传统方法
        if not cluster_risk_mapping:
            cluster_details = clustering_results.get('cluster_details', [])
            cluster_risk_mapping = {}

            for detail in cluster_details:
                cluster_id = detail.get('cluster_id', 0)
                fraud_rate = detail.get('fraud_rate', 0)

                # 基于聚类欺诈率生成标签
                if fraud_rate >= 0.15:
                    cluster_risk_mapping[cluster_id] = {'label': 1, 'risk_level': 'high'}
                elif fraud_rate >= 0.08:
                    cluster_risk_mapping[cluster_id] = {'label': 1 if fraud_rate >= 0.12 else 0, 'risk_level': 'medium'}
                else:
                    cluster_risk_mapping[cluster_id] = {'label': 0, 'risk_level': 'low'}

        # 生成伪标签和置信度
        labels = []
        confidences = []
        cluster_info_list = []

        for cluster_id in cluster_labels:
            # 获取聚类风险信息
            if isinstance(cluster_risk_mapping.get(cluster_id), dict):
                cluster_info = cluster_risk_mapping[cluster_id]
                risk_level = cluster_info.get('risk_level', 'low')
                risk_score = cluster_info.get('risk_score', 0)

                # 基于聚类风险等级生成标签
                if risk_level == 'critical':
                    label = 1
                    confidence = min(0.95, 0.80 + risk_score / 500)
                elif risk_level == 'high':
                    label = 1
                    confidence = min(0.85, 0.65 + risk_score / 400)
                elif risk_level == 'medium':
                    label = 1 if risk_score >= 50 else 0
                    confidence = 0.55 + abs(risk_score - 50) / 200
                else:  # low
                    label = 0
                    confidence = min(0.80, 0.60 + (50 - risk_score) / 200)

                cluster_info_list.append({
                    'cluster_id': cluster_id,
                    'risk_level': risk_level,
                    'risk_score': risk_score
                })
            else:
                # 兼容旧格式
                label = cluster_risk_mapping.get(cluster_id, 0)
                confidence = 0.6  # 默认置信度
                cluster_info_list.append({
                    'cluster_id': cluster_id,
                    'risk_level': 'unknown',
                    'risk_score': 0
                })

            labels.append(label)
            confidences.append(round(confidence, 3))

        # 计算质量指标
        high_confidence_labels = [l for l, c in zip(labels, confidences) if c >= self.confidence_threshold]

        return {
            'strategy': 'cluster_based',
            'labels': labels,
            'confidences': confidences,
            'cluster_info': cluster_info_list,
            'label_distribution': pd.Series(labels).value_counts().to_dict(),
            'avg_confidence': round(np.mean(confidences), 3),
            'high_confidence_count': len(high_confidence_labels),
            'high_confidence_fraud_rate': sum(high_confidence_labels) / max(1, len(high_confidence_labels)),
            'metadata': {
                'n_clusters': len(set(cluster_labels)),
                'cluster_risk_distribution': clustering_results.get('risk_distribution', {}),
                'clustering_algorithm': clustering_results.get('algorithm', 'kmeans'),
                'confidence_threshold': self.confidence_threshold
            }
        }

    def _generate_rule_based_labels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """基于专家规则生成伪标签"""
        labels = []
        confidences = []
        rule_matches = []
        
        for idx, row in data.iterrows():
            label = 0
            confidence = 0.5
            matched_rules = []
            
            # 规则1: 大额夜间交易
            if (row.get('transaction_amount', 0) > 1000 and 
                row.get('transaction_hour', 12) in [0, 1, 2, 3, 4, 5, 22, 23]):
                label = 1
                confidence += 0.3
                matched_rules.append('大额夜间交易')
            
            # 规则2: 新账户大额交易
            if (row.get('account_age_days', 365) < 30 and 
                row.get('transaction_amount', 0) > 500):
                label = 1
                confidence += 0.25
                matched_rules.append('新账户大额交易')
            
            # 规则3: 异常年龄 + 高风险商品
            if (row.get('customer_age', 30) < 18 or row.get('customer_age', 30) > 70) and \
               row.get('product_category', '') == 'electronics':
                label = 1
                confidence += 0.2
                matched_rules.append('异常年龄高风险商品')
            
            # 规则4: 移动设备 + 银行转账 + 大额
            if (row.get('device_used', '') == 'mobile' and 
                row.get('payment_method', '') == 'bank transfer' and
                row.get('transaction_amount', 0) > 300):
                label = 1
                confidence += 0.2
                matched_rules.append('移动设备银行转账')
            
            # 规则5: 地址不一致 + 大额交易
            shipping_addr = str(row.get('shipping_address', ''))
            billing_addr = str(row.get('billing_address', ''))
            if (shipping_addr != billing_addr and 
                row.get('transaction_amount', 0) > 200):
                label = 1
                confidence += 0.15
                matched_rules.append('地址不一致大额交易')
            
            # 正常交易规则
            if not matched_rules:
                # 小额 + 正常时间 + 老账户
                if (row.get('transaction_amount', 0) < 100 and
                    9 <= row.get('transaction_hour', 12) <= 21 and
                    row.get('account_age_days', 0) > 90):
                    confidence = 0.8
                    matched_rules.append('正常交易模式')
            
            confidence = min(0.95, max(0.1, confidence))
            
            labels.append(label)
            confidences.append(round(confidence, 3))
            rule_matches.append(matched_rules)
        
        return {
            'strategy': 'rule_based',
            'labels': labels,
            'confidences': confidences,
            'label_distribution': pd.Series(labels).value_counts().to_dict(),
            'avg_confidence': round(np.mean(confidences), 3),
            'high_confidence_count': len([c for c in confidences if c >= self.confidence_threshold]),
            'metadata': {
                'rule_matches': rule_matches,
                'total_rules': 5,
                'rule_coverage': len([r for r in rule_matches if r]) / len(rule_matches)
            }
        }

    def _generate_ensemble_labels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """集成多种策略生成伪标签"""
        # 首先计算无监督风险评分和聚类结果
        from backend.risk_scoring.risk_calculator import UnsupervisedRiskCalculator
        from backend.clustering.cluster_analyzer import ClusterAnalyzer

        # 进行聚类分析
        cluster_analyzer = ClusterAnalyzer(n_clusters=4, random_state=42)
        clustering_results = cluster_analyzer.analyze_clusters(data, algorithm='kmeans')

        # 计算无监督风险评分
        risk_calculator = UnsupervisedRiskCalculator()
        unsupervised_risk_results = risk_calculator.calculate_unsupervised_risk_score(
            data, clustering_results
        )

        # 获取各策略结果
        risk_result = self._generate_risk_based_labels(data, unsupervised_risk_results)
        cluster_result = self._generate_cluster_based_labels(data, clustering_results)
        rule_result = self._generate_rule_based_labels(data)

        risk_labels = np.array(risk_result['labels'])
        cluster_labels = np.array(cluster_result['labels'])
        rule_labels = np.array(rule_result['labels'])

        risk_confidences = np.array(risk_result['confidences'])
        cluster_confidences = np.array(cluster_result['confidences'])
        rule_confidences = np.array(rule_result['confidences'])

        # 动态权重调整 - 基于各策略的质量
        risk_quality = risk_result.get('high_confidence_fraud_rate', 0.5)
        cluster_quality = cluster_result.get('high_confidence_fraud_rate', 0.5)
        rule_quality = rule_result.get('metadata', {}).get('rule_coverage', 0.3)

        # 基础权重
        base_weights = [0.45, 0.35, 0.20]  # 无监督风险评分权重最高

        # 质量调整
        quality_scores = [risk_quality, cluster_quality, rule_quality]
        quality_weights = np.array(quality_scores) / sum(quality_scores)

        # 最终权重 = 基础权重 * 0.7 + 质量权重 * 0.3
        final_weights = np.array(base_weights) * 0.7 + quality_weights * 0.3
        final_weights = final_weights / sum(final_weights)  # 归一化

        # 加权投票
        ensemble_scores = (
            risk_labels * risk_confidences * final_weights[0] +
            cluster_labels * cluster_confidences * final_weights[1] +
            rule_labels * rule_confidences * final_weights[2]
        )

        # 生成最终标签 - 使用动态阈值
        avg_confidence = np.mean([np.mean(risk_confidences), np.mean(cluster_confidences), np.mean(rule_confidences)])
        threshold = max(0.4, min(0.6, avg_confidence * 0.8))

        ensemble_labels = (ensemble_scores > threshold).astype(int)

        # 计算集成置信度
        ensemble_confidences = []
        strategy_agreements = []

        for i in range(len(ensemble_labels)):
            # 基于一致性和权重计算置信度
            votes = [risk_labels[i], cluster_labels[i], rule_labels[i]]
            vote_confidences = [risk_confidences[i], cluster_confidences[i], rule_confidences[i]]

            # 计算策略一致性
            agreement_score = 0
            if votes[0] == votes[1]:  # 风险评分与聚类一致
                agreement_score += 0.4
            if votes[0] == votes[2]:  # 风险评分与规则一致
                agreement_score += 0.3
            if votes[1] == votes[2]:  # 聚类与规则一致
                agreement_score += 0.3

            strategy_agreements.append(agreement_score)

            if len(set(votes)) == 1:  # 完全一致
                confidence = np.average(vote_confidences, weights=final_weights)
                confidence = min(0.95, confidence + 0.1)  # 一致性奖励
            elif agreement_score >= 0.4:  # 部分一致
                confidence = np.average(vote_confidences, weights=final_weights) * 0.85
            else:  # 完全分歧
                confidence = max(vote_confidences) * 0.6  # 大幅降低置信度

            ensemble_confidences.append(round(confidence, 3))

        # 计算质量指标
        high_confidence_labels = [l for l, c in zip(ensemble_labels, ensemble_confidences) if c >= self.confidence_threshold]

        return {
            'strategy': 'ensemble',
            'labels': ensemble_labels.tolist(),
            'confidences': ensemble_confidences,
            'label_distribution': pd.Series(ensemble_labels).value_counts().to_dict(),
            'avg_confidence': round(np.mean(ensemble_confidences), 3),
            'high_confidence_count': len(high_confidence_labels),
            'high_confidence_fraud_rate': sum(high_confidence_labels) / max(1, len(high_confidence_labels)),
            'metadata': {
                'component_results': {
                    'risk_based': risk_result['label_distribution'],
                    'cluster_based': cluster_result['label_distribution'],
                    'rule_based': rule_result['label_distribution']
                },
                'final_weights': final_weights.tolist(),
                'base_weights': base_weights,
                'quality_scores': quality_scores,
                'dynamic_threshold': threshold,
                'avg_strategy_agreement': round(np.mean(strategy_agreements), 3),
                'full_agreement_rate': np.mean([
                    risk_labels[i] == cluster_labels[i] == rule_labels[i]
                    for i in range(len(risk_labels))
                ]),
                'confidence_threshold': self.confidence_threshold,
                'unsupervised_risk_avg': unsupervised_risk_results.get('average_risk_score', 0)
            }
        }

    def _record_generation_history(self, result: Dict, strategy: str):
        """记录生成历史"""
        history_record = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'total_labels': len(result.get('labels', [])),
            'positive_rate': result.get('label_distribution', {}).get(1, 0) / len(result.get('labels', [1])),
            'avg_confidence': result.get('avg_confidence', 0),
            'high_confidence_rate': result.get('high_confidence_count', 0) / len(result.get('labels', [1]))
        }
        
        self.label_history.append(history_record)
        
        # 保持历史记录不超过100条
        if len(self.label_history) > 100:
            self.label_history = self.label_history[-100:]

    def get_label_quality_metrics(self, pseudo_labels: List[int], true_labels: List[int] = None) -> Dict:
        """计算伪标签质量指标"""
        if true_labels is None:
            return {
                'label_distribution': pd.Series(pseudo_labels).value_counts().to_dict(),
                'positive_rate': sum(pseudo_labels) / len(pseudo_labels),
                'total_labels': len(pseudo_labels)
            }
        
        # 如果有真实标签，计算准确性指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'accuracy': round(accuracy_score(true_labels, pseudo_labels), 3),
            'precision': round(precision_score(true_labels, pseudo_labels), 3),
            'recall': round(recall_score(true_labels, pseudo_labels), 3),
            'f1_score': round(f1_score(true_labels, pseudo_labels), 3),
            'label_distribution': pd.Series(pseudo_labels).value_counts().to_dict(),
            'positive_rate': sum(pseudo_labels) / len(pseudo_labels)
        }

    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'strategy': 'none',
            'labels': [],
            'confidences': [],
            'label_distribution': {},
            'avg_confidence': 0,
            'high_confidence_count': 0,
            'metadata': {}
        }

    def _generate_fallback_risk_labels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成回退风险标签（当主要方法失败时使用）"""
        try:
            logger.info("使用回退方法生成风险标签")

            labels = []
            confidences = []

            # 简单的基于规则的标签生成
            for _, row in data.iterrows():
                label = 0
                confidence = 0.6

                # 基于交易金额的简单规则
                amount = row.get('transaction_amount', 0)
                if amount > 2000:
                    label = 1
                    confidence = 0.8
                elif amount > 1000:
                    label = 1
                    confidence = 0.7

                labels.append(label)
                confidences.append(confidence)

            return {
                'strategy': 'fallback_risk',
                'labels': labels,
                'confidences': confidences,
                'label_distribution': pd.Series(labels).value_counts().to_dict(),
                'avg_confidence': np.mean(confidences),
                'high_confidence_count': sum(1 for c in confidences if c > 0.7),
                'metadata': {'method': 'fallback_rules', 'quality_level': 'basic'}
            }

        except Exception as e:
            logger.error(f"回退标签生成失败: {e}")
            return self._empty_result()

    def get_generation_history(self) -> List[Dict]:
        """获取生成历史"""
        return self.label_history.copy()

    def update_confidence_threshold(self, threshold: float):
        """更新置信度阈值"""
        self.confidence_threshold = max(0.1, min(0.9, threshold))
        logger.info(f"置信度阈值更新为: {self.confidence_threshold}")

    def generate_high_quality_pseudo_labels(self, data: pd.DataFrame,
                                          min_confidence: float = 0.8,
                                          use_calibration: bool = True) -> Dict[str, Any]:
        """
        生成高质量伪标签

        Args:
            data: 输入数据
            min_confidence: 最小置信度阈值
            use_calibration: 是否使用校准结果

        Returns:
            高质量伪标签结果
        """
        try:
            logger.info(f"开始生成高质量伪标签，最小置信度: {min_confidence}")

            # 1. 进行聚类分析
            from backend.clustering.cluster_analyzer import ClusterAnalyzer
            cluster_analyzer = ClusterAnalyzer(n_clusters=4, random_state=42)
            clustering_results = cluster_analyzer.analyze_clusters(data, algorithm='kmeans')

            # 2. 计算无监督风险评分
            from backend.risk_scoring.risk_calculator import UnsupervisedRiskCalculator
            risk_calculator = UnsupervisedRiskCalculator()

            # 如果使用校准，尝试执行校准
            if use_calibration and 'is_fraudulent' in data.columns:
                try:
                    calibration_results = risk_calculator.perform_full_calibration(
                        data, clustering_results, sample_ratio=0.1
                    )
                    unsupervised_risk_results = calibration_results.get('calibrated_results', {})
                    calibration_applied = calibration_results.get('calibration_applied', False)
                    logger.info(f"校准{'成功' if calibration_applied else '未'}应用")
                except Exception as e:
                    logger.warning(f"校准失败，使用未校准结果: {e}")
                    unsupervised_risk_results = risk_calculator.calculate_unsupervised_risk_score(
                        data, clustering_results
                    )
                    calibration_applied = False
            else:
                unsupervised_risk_results = risk_calculator.calculate_unsupervised_risk_score(
                    data, clustering_results
                )
                calibration_applied = False

            # 3. 生成集成伪标签
            ensemble_result = self._generate_ensemble_labels(data)

            # 4. 筛选高质量标签
            labels = ensemble_result['labels']
            confidences = ensemble_result['confidences']

            high_quality_indices = [i for i, c in enumerate(confidences) if c >= min_confidence]
            high_quality_labels = [labels[i] for i in high_quality_indices]
            high_quality_confidences = [confidences[i] for i in high_quality_indices]

            # 5. 生成质量报告
            quality_report = self._generate_quality_report(
                labels, confidences, high_quality_indices,
                unsupervised_risk_results, ensemble_result
            )

            result = {
                'strategy': 'high_quality_ensemble',
                'all_labels': labels,
                'all_confidences': confidences,
                'high_quality_indices': high_quality_indices,
                'high_quality_labels': high_quality_labels,
                'high_quality_confidences': high_quality_confidences,
                'quality_report': quality_report,
                'calibration_applied': calibration_applied,
                'min_confidence_threshold': min_confidence,
                'metadata': {
                    'total_samples': len(labels),
                    'high_quality_count': len(high_quality_labels),
                    'high_quality_rate': len(high_quality_labels) / len(labels),
                    'avg_confidence_all': round(np.mean(confidences), 3),
                    'avg_confidence_hq': round(np.mean(high_quality_confidences), 3) if high_quality_confidences else 0,
                    'fraud_rate_all': sum(labels) / len(labels),
                    'fraud_rate_hq': sum(high_quality_labels) / len(high_quality_labels) if high_quality_labels else 0
                }
            }

            logger.info(f"高质量伪标签生成完成，筛选出 {len(high_quality_labels)}/{len(labels)} 个高质量标签")
            return result

        except Exception as e:
            logger.error(f"高质量伪标签生成失败: {e}")
            return self._empty_result()

    def _generate_quality_report(self, labels: List[int], confidences: List[float],
                               high_quality_indices: List[int],
                               unsupervised_risk_results: Dict,
                               ensemble_result: Dict) -> Dict[str, Any]:
        """生成质量报告"""
        try:
            # 置信度分布分析
            confidence_bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
            confidence_distribution = {}

            for i in range(len(confidence_bins) - 1):
                bin_name = f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}"
                count = sum(1 for c in confidences if confidence_bins[i] <= c < confidence_bins[i+1])
                confidence_distribution[bin_name] = count

            # 风险等级分布
            risk_results = unsupervised_risk_results.get('results', [])
            risk_level_distribution = {}
            for result in risk_results:
                level = result.get('risk_level', 'unknown')
                risk_level_distribution[level] = risk_level_distribution.get(level, 0) + 1

            # 高质量标签的风险等级分布
            hq_risk_distribution = {}
            for idx in high_quality_indices:
                if idx < len(risk_results):
                    level = risk_results[idx].get('risk_level', 'unknown')
                    hq_risk_distribution[level] = hq_risk_distribution.get(level, 0) + 1

            return {
                'confidence_distribution': confidence_distribution,
                'risk_level_distribution': risk_level_distribution,
                'high_quality_risk_distribution': hq_risk_distribution,
                'strategy_agreement': ensemble_result.get('metadata', {}).get('full_agreement_rate', 0),
                'avg_risk_score': unsupervised_risk_results.get('average_risk_score', 0),
                'quality_score': self._calculate_overall_quality_score(
                    confidences, high_quality_indices, ensemble_result
                )
            }

        except Exception as e:
            logger.error(f"质量报告生成失败: {e}")
            return {}

    def _calculate_overall_quality_score(self, confidences: List[float],
                                       high_quality_indices: List[int],
                                       ensemble_result: Dict) -> float:
        """计算整体质量评分"""
        try:
            # 基础分数：平均置信度
            base_score = np.mean(confidences) * 100

            # 高质量比例奖励
            hq_ratio = len(high_quality_indices) / len(confidences)
            hq_bonus = hq_ratio * 20

            # 策略一致性奖励
            agreement_rate = ensemble_result.get('metadata', {}).get('full_agreement_rate', 0)
            agreement_bonus = agreement_rate * 15

            # 标签平衡性评估
            labels = ensemble_result.get('labels', [])
            fraud_rate = sum(labels) / len(labels) if labels else 0
            # 理想欺诈率在5-15%之间
            if 0.05 <= fraud_rate <= 0.15:
                balance_bonus = 10
            elif 0.02 <= fraud_rate <= 0.25:
                balance_bonus = 5
            else:
                balance_bonus = 0

            total_score = base_score + hq_bonus + agreement_bonus + balance_bonus
            return round(min(100, total_score), 2)

        except Exception as e:
            logger.error(f"质量评分计算失败: {e}")
            return 0.0
