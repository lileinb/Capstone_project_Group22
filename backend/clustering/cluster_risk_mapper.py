"""
聚类风险映射器
将聚类结果映射到风险等级，支持无监督风险评估
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusterRiskMapper:
    """聚类风险映射器"""
    
    def __init__(self):
        """初始化聚类风险映射器"""
        # 重新设计权重 - 平衡多维度风险特征
        self.risk_indicator_weights = {
            'amount_risk': 0.25,           # 交易金额风险 (提升权重)
            'account_age_risk': 0.20,      # 账户年龄风险 (提升权重)
            'time_pattern_risk': 0.18,     # 时间模式风险 (提升权重)
            'fraud_rate_risk': 0.15,       # 欺诈率风险 (降低权重，避免过度依赖)
            'device_payment_risk': 0.10,   # 设备支付风险 (提升权重)
            'address_risk': 0.07,          # 地址风险
            'category_risk': 0.03,         # 商品类别风险
            'statistical_risk': 0.02       # 统计异常风险
        }

        # 大幅降低风险等级阈值 - 确保四层分布
        self.cluster_risk_thresholds = {
            'low': 15,      # 0-15: 低风险 (大幅降低)
            'medium': 30,   # 16-30: 中风险 (大幅降低)
            'high': 50,     # 31-50: 高风险 (大幅降低)
            'critical': 100 # 51-100: 极高风险 (大幅降低)
        }
    
    def map_clusters_to_risk_levels(self, cluster_results: Dict[str, Any], 
                                  data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        将聚类结果映射到风险等级
        
        Args:
            cluster_results: 聚类分析结果
            data: 原始数据
            
        Returns:
            聚类ID到风险信息的映射
        """
        try:
            cluster_risk_mapping = {}
            cluster_details = cluster_results.get('cluster_details', [])
            
            logger.info(f"开始映射 {len(cluster_details)} 个聚类到风险等级")
            
            for cluster_detail in cluster_details:
                cluster_id = cluster_detail.get('cluster_id', -1)
                
                # 计算聚类风险指标
                risk_indicators = self._calculate_cluster_risk_indicators(cluster_detail)
                
                # 计算综合风险评分
                risk_score = self._calculate_comprehensive_risk_score(risk_indicators)
                
                # 确定风险等级
                risk_level = self._determine_cluster_risk_level(risk_score)
                
                # 生成风险解释
                risk_explanation = self._generate_risk_explanation(risk_indicators, risk_score)
                
                # 计算聚类质量指标
                quality_metrics = self._calculate_cluster_quality_metrics(cluster_detail, cluster_results)
                
                cluster_risk_mapping[cluster_id] = {
                    'cluster_id': cluster_id,
                    'risk_score': round(risk_score, 2),
                    'risk_level': risk_level,
                    'risk_indicators': risk_indicators,
                    'risk_explanation': risk_explanation,
                    'quality_metrics': quality_metrics,
                    'size': cluster_detail.get('size', 0),
                    'percentage': cluster_detail.get('percentage', 0),
                    'timestamp': datetime.now().isoformat()
                }
            
            # 添加整体风险分布统计
            risk_distribution = self._calculate_risk_distribution(cluster_risk_mapping)
            
            logger.info(f"聚类风险映射完成，风险分布: {risk_distribution}")
            
            return {
                'cluster_risk_mapping': cluster_risk_mapping,
                'risk_distribution': risk_distribution,
                'mapping_summary': self._generate_mapping_summary(cluster_risk_mapping)
            }
            
        except Exception as e:
            logger.error(f"聚类风险映射失败: {e}")
            return {'cluster_risk_mapping': {}, 'risk_distribution': {}, 'mapping_summary': {}}
    
    def _calculate_cluster_risk_indicators(self, cluster_detail: Dict) -> Dict[str, float]:
        """计算聚类风险指标"""
        indicators = {}
        
        # 1. 交易金额风险指标
        avg_amount = cluster_detail.get('avg_transaction_amount', 0)
        amount_std = cluster_detail.get('transaction_amount_std', 0)
        high_amount_rate = cluster_detail.get('high_amount_rate', 0)
        
        amount_risk = 0
        # 增强大额交易检测 - 与四分类计算器保持一致
        if avg_amount > 5000:
            amount_risk += 60  # 超大额交易
        elif avg_amount > 2000:
            amount_risk += 45  # 大额交易
        elif avg_amount > 1000:
            amount_risk += 30  # 中大额交易
        elif avg_amount > 500:
            amount_risk += 20  # 中等交易
        elif avg_amount > 200:
            amount_risk += 10  # 小额交易

        # 增强高额交易比例检测
        if high_amount_rate > 0.5:
            amount_risk += 40  # 超高比例
        elif high_amount_rate > 0.3:
            amount_risk += 30  # 高比例
        elif high_amount_rate > 0.1:
            amount_risk += 15  # 中等比例

        if amount_std > avg_amount * 0.8:  # 高变异性
            amount_risk += 25  # 增强变异性权重
        
        indicators['amount_risk'] = min(100, amount_risk)
        
        # 2. 时间模式风险指标
        night_rate = cluster_detail.get('night_transaction_rate', 0)
        common_hour = cluster_detail.get('common_hour', 12)
        
        time_risk = 0
        # 增强夜间交易检测
        if night_rate > 0.6:
            time_risk += 60  # 超高夜间交易比例
        elif night_rate > 0.4:
            time_risk += 45  # 高夜间交易比例
        elif night_rate > 0.2:
            time_risk += 30  # 中等夜间交易比例
        elif night_rate > 0.1:
            time_risk += 15  # 低夜间交易比例

        # 增强深夜时间检测
        if 1 <= common_hour <= 4:
            time_risk += 40  # 深夜时间(1-4点)
        elif common_hour <= 5 or common_hour >= 22:
            time_risk += 25  # 一般夜间时间
        
        indicators['time_pattern_risk'] = min(100, time_risk)
        
        # 3. 账户年龄风险指标
        avg_account_age = cluster_detail.get('avg_account_age_days', 365)
        new_account_rate = cluster_detail.get('new_account_rate', 0)
        
        account_risk = 0
        # 增强新账户检测 - 与四分类计算器保持一致
        if avg_account_age < 7:
            account_risk += 50  # 超新账户(1周内)
        elif avg_account_age < 30:
            account_risk += 40  # 新账户(1月内)
        elif avg_account_age < 90:
            account_risk += 25  # 较新账户(3月内)
        elif avg_account_age < 180:
            account_risk += 15  # 中等账户(6月内)

        # 增强新账户比例检测
        if new_account_rate > 0.7:
            account_risk += 45  # 超高新账户比例
        elif new_account_rate > 0.5:
            account_risk += 35  # 高新账户比例
        elif new_account_rate > 0.2:
            account_risk += 20  # 中等新账户比例
        
        indicators['account_age_risk'] = min(100, account_risk)
        
        # 4. 设备支付风险指标
        mobile_rate = cluster_detail.get('mobile_device_rate', 0)
        bank_transfer_rate = cluster_detail.get('bank_transfer_rate', 0)
        
        device_payment_risk = 0
        if mobile_rate > 0.8:
            device_payment_risk += 25
        elif mobile_rate > 0.6:
            device_payment_risk += 15
        
        if bank_transfer_rate > 0.6:
            device_payment_risk += 30
        elif bank_transfer_rate > 0.3:
            device_payment_risk += 15
        
        if mobile_rate > 0.7 and bank_transfer_rate > 0.4:
            device_payment_risk += 20  # 组合风险
        
        indicators['device_payment_risk'] = min(100, device_payment_risk)
        
        # 5. 地址风险指标
        address_mismatch_rate = cluster_detail.get('address_mismatch_rate', 0)
        
        address_risk = 0
        if address_mismatch_rate > 0.6:
            address_risk += 60
        elif address_mismatch_rate > 0.3:
            address_risk += 35
        elif address_mismatch_rate > 0.1:
            address_risk += 15
        
        indicators['address_risk'] = min(100, address_risk)
        
        # 6. 商品类别风险指标
        electronics_rate = cluster_detail.get('electronics_rate', 0)
        
        category_risk = 0
        if electronics_rate > 0.7:
            category_risk += 40
        elif electronics_rate > 0.4:
            category_risk += 20
        
        indicators['category_risk'] = min(100, category_risk)

        # 7. 欺诈率风险指标 (调整 - 更细粒度，降低权重影响)
        fraud_rate = cluster_detail.get('fraud_rate', 0)

        fraud_risk = 0
        if fraud_rate > 0.3:
            fraud_risk += 70  # 超高欺诈率
        elif fraud_rate > 0.15:
            fraud_risk += 55  # 高欺诈率
        elif fraud_rate > 0.08:
            fraud_risk += 40  # 中等欺诈率
        elif fraud_rate > 0.04:
            fraud_risk += 30  # 中低欺诈率
        elif fraud_rate > 0.02:
            fraud_risk += 25  # 低欺诈率
        elif fraud_rate > 0.01:
            fraud_risk += 20  # 极低欺诈率
        elif fraud_rate > 0:
            fraud_risk += 15  # 微量欺诈率
        else:
            fraud_risk += 10  # 零欺诈率也给基础分

        indicators['fraud_rate_risk'] = min(100, fraud_risk)

        # 8. 统计异常风险（基于聚类大小和分布）
        cluster_size = cluster_detail.get('size', 0)
        cluster_percentage = cluster_detail.get('percentage', 0)
        
        statistical_risk = 0
        if cluster_percentage < 5:  # 小聚类可能是异常
            statistical_risk += 30
        elif cluster_percentage > 50:  # 过大聚类可能包含异常
            statistical_risk += 15
        
        indicators['statistical_risk'] = min(100, statistical_risk)
        
        return indicators
    
    def _calculate_comprehensive_risk_score(self, risk_indicators: Dict[str, float]) -> float:
        """计算综合风险评分"""
        total_score = 0.0

        # 基础加权评分
        for indicator, score in risk_indicators.items():
            weight = self.risk_indicator_weights.get(indicator, 0)
            total_score += score * weight

        # 添加组合风险奖励机制 - 确保四层分布 (增强影响力)
        combination_bonus = self._calculate_combination_risk_bonus(risk_indicators)
        total_score += combination_bonus * 1.5  # 增强组合奖励影响力

        return min(100, total_score)

    def _calculate_combination_risk_bonus(self, risk_indicators: Dict[str, float]) -> float:
        """计算组合风险奖励分数"""
        bonus = 0.0

        # 获取各项风险评分
        amount_risk = risk_indicators.get('amount_risk', 0)
        account_risk = risk_indicators.get('account_age_risk', 0)
        time_risk = risk_indicators.get('time_pattern_risk', 0)
        device_risk = risk_indicators.get('device_payment_risk', 0)
        fraud_risk = risk_indicators.get('fraud_rate_risk', 0)

        # 高风险组合模式识别
        high_risk_indicators = sum(1 for score in [amount_risk, account_risk, time_risk, device_risk] if score > 40)

        if high_risk_indicators >= 3:
            bonus += 15  # 多个高风险指标组合
        elif high_risk_indicators >= 2:
            bonus += 8   # 两个高风险指标组合

        # 特定高风险组合
        if amount_risk > 50 and account_risk > 40:
            bonus += 10  # 大额交易 + 新账户

        if time_risk > 40 and device_risk > 30:
            bonus += 8   # 异常时间 + 高风险设备

        if fraud_risk > 30 and (amount_risk > 40 or account_risk > 40):
            bonus += 12  # 有欺诈记录 + 其他高风险因素

        # 确保低风险聚类也有适当分数分布
        if all(score < 30 for score in [amount_risk, account_risk, time_risk, device_risk]):
            # 所有指标都较低，但仍需要一些差异化
            if fraud_risk > 15:
                bonus += 5   # 即使其他指标低，有一定欺诈率
            elif max(amount_risk, account_risk, time_risk, device_risk) > 20:
                bonus += 3   # 有一个指标稍高

        return min(25, bonus)  # 限制奖励分数上限
    
    def _determine_cluster_risk_level(self, risk_score: float) -> str:
        """确定聚类风险等级"""
        if risk_score >= self.cluster_risk_thresholds['critical']:
            return 'critical'
        elif risk_score >= self.cluster_risk_thresholds['high']:
            return 'high'
        elif risk_score >= self.cluster_risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_risk_explanation(self, risk_indicators: Dict[str, float], 
                                 risk_score: float) -> List[str]:
        """生成风险解释"""
        explanations = []
        
        # 找出主要风险因素
        sorted_indicators = sorted(risk_indicators.items(), key=lambda x: x[1], reverse=True)
        
        for indicator, score in sorted_indicators[:3]:  # 取前3个主要风险因素
            if score > 30:
                if indicator == 'amount_risk':
                    explanations.append(f"交易金额异常 (风险分数: {score:.1f})")
                elif indicator == 'time_pattern_risk':
                    explanations.append(f"时间模式异常 (风险分数: {score:.1f})")
                elif indicator == 'account_age_risk':
                    explanations.append(f"账户年龄风险 (风险分数: {score:.1f})")
                elif indicator == 'device_payment_risk':
                    explanations.append(f"设备支付风险 (风险分数: {score:.1f})")
                elif indicator == 'address_risk':
                    explanations.append(f"地址不一致风险 (风险分数: {score:.1f})")
                elif indicator == 'category_risk':
                    explanations.append(f"商品类别风险 (风险分数: {score:.1f})")
                elif indicator == 'statistical_risk':
                    explanations.append(f"统计异常风险 (风险分数: {score:.1f})")
        
        if not explanations:
            explanations.append("风险因素较低，属于正常交易模式")
        
        return explanations
    
    def _calculate_cluster_quality_metrics(self, cluster_detail: Dict, 
                                         cluster_results: Dict) -> Dict[str, float]:
        """计算聚类质量指标"""
        quality_metrics = cluster_results.get('quality_metrics', {})
        
        # 添加聚类特定的质量指标
        cluster_size = cluster_detail.get('size', 0)
        cluster_percentage = cluster_detail.get('percentage', 0)
        
        # 聚类紧密度（基于大小和比例）
        compactness = 1.0 - abs(cluster_percentage - 20) / 100  # 理想比例约20%
        
        return {
            'silhouette_score': quality_metrics.get('silhouette_score', 0),
            'calinski_harabasz_score': quality_metrics.get('calinski_harabasz_score', 0),
            'cluster_compactness': round(max(0, compactness), 3),
            'relative_size': round(cluster_percentage / 100, 3)
        }
    
    def _calculate_risk_distribution(self, cluster_risk_mapping: Dict) -> Dict[str, int]:
        """计算风险分布"""
        distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for cluster_info in cluster_risk_mapping.values():
            risk_level = cluster_info.get('risk_level', 'low')
            if risk_level in distribution:
                distribution[risk_level] += 1
        
        return distribution
    
    def _generate_mapping_summary(self, cluster_risk_mapping: Dict) -> Dict[str, Any]:
        """生成映射总结"""
        if not cluster_risk_mapping:
            return {}
        
        risk_scores = [info['risk_score'] for info in cluster_risk_mapping.values()]
        cluster_sizes = [info['size'] for info in cluster_risk_mapping.values()]
        
        return {
            'total_clusters': len(cluster_risk_mapping),
            'avg_risk_score': round(np.mean(risk_scores), 2),
            'max_risk_score': round(max(risk_scores), 2),
            'min_risk_score': round(min(risk_scores), 2),
            'risk_score_std': round(np.std(risk_scores), 2),
            'avg_cluster_size': round(np.mean(cluster_sizes), 0),
            'total_samples': sum(cluster_sizes)
        }
