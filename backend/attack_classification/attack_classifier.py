"""
攻击类型分类器
基于真实数据集特征模式识别和分类不同类型的欺诈攻击
支持的攻击类型：
- 账户接管攻击 (Account Takeover)
- 身份盗用攻击 (Identity Theft)
- 批量欺诈攻击 (Bulk Fraud)
- 测试性攻击 (Testing Attack)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttackClassifier:
    """基于真实数据集的攻击类型分类器"""

    def __init__(self, models_dir: str = "models"):
        """
        初始化攻击分类器

        Args:
            models_dir: 模型文件目录
        """
        self.models_dir = models_dir
        self.attack_patterns = self._define_attack_patterns()

        # 用于攻击分类的关键特征
        self.classification_features = [
            'transaction_amount', 'quantity', 'customer_age', 'account_age_days',
            'transaction_hour'
        ]

        # 为了兼容性，添加 risk_calculator 属性（如果有代码尝试访问它）
        self.risk_calculator = None

    def _define_attack_patterns(self) -> Dict:
        """定义基于真实数据集的攻击模式"""
        return {
            'account_takeover': {
                'description': '账户接管攻击',
                'characteristics': [
                    '新账户大额交易',
                    '异常时间段交易',
                    '设备类型突变',
                    '地址不一致'
                ],
                'risk_level': 'CRITICAL',
                'detection_rules': {
                    'account_age_days': {'max': 30},
                    'transaction_amount': {'min': 500},
                    'transaction_hour': {'values': [0, 1, 2, 3, 4, 5, 22, 23]},
                    'address_mismatch': True
                },
                'weight': 0.3
            },
            'identity_theft': {
                'description': '身份盗用攻击',
                'characteristics': [
                    '异常年龄段交易',
                    '高风险商品类别',
                    '移动设备偏好',
                    '银行转账支付'
                ],
                'risk_level': 'HIGH',
                'detection_rules': {
                    'customer_age': {'extreme': True},  # 极端年龄
                    'product_category': {'values': ['electronics']},
                    'device_used': {'values': ['mobile']},
                    'payment_method': {'values': ['bank transfer']}
                },
                'weight': 0.25
            },
            'bulk_fraud': {
                'description': '批量欺诈攻击',
                'characteristics': [
                    '相似交易金额',
                    '固定数量购买',
                    '集中时间段',
                    '相同支付方式'
                ],
                'risk_level': 'HIGH',
                'detection_rules': {
                    'amount_similarity': 0.9,  # 金额相似度阈值
                    'quantity': {'values': [1, 2]},  # 固定数量
                    'time_clustering': True,  # 时间聚集
                    'payment_concentration': 0.8  # 支付方式集中度
                },
                'weight': 0.25
            },
            'testing_attack': {
                'description': '测试性攻击',
                'characteristics': [
                    '小额多次交易',
                    '短时间内频繁',
                    '新账户活动',
                    '多种支付尝试'
                ],
                'risk_level': 'MEDIUM',
                'detection_rules': {
                    'transaction_amount': {'max': 50},
                    'frequency_threshold': 5,  # 短时间内交易次数
                    'account_age_days': {'max': 7},
                    'payment_diversity': True  # 多种支付方式
                },
                'weight': 0.2
            }
        }

    def classify_attacks(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        对交易数据进行攻击类型分类

        Args:
            data: 清理后的交易数据

        Returns:
            攻击分类结果
        """
        if data is None or data.empty:
            logger.error("输入数据为空")
            return self._empty_classification_result()

        try:
            # 只分析欺诈交易
            fraud_data = data[data.get('is_fraudulent', 0) == 1] if 'is_fraudulent' in data.columns else data

            if fraud_data.empty:
                logger.info("没有发现欺诈交易")
                return {
                    'total_transactions': len(data),
                    'fraud_transactions': 0,
                    'attack_types': {},
                    'classification_results': []
                }

            # 对每笔欺诈交易进行分类
            classification_results = []
            attack_type_counts = {attack_type: 0 for attack_type in self.attack_patterns.keys()}

            for idx, row in fraud_data.iterrows():
                attack_type = self._classify_single_transaction(row)
                attack_type_counts[attack_type] += 1

                classification_results.append({
                    'transaction_id': row.get('transaction_id', f'tx_{idx}'),
                    'customer_id': row.get('customer_id', f'customer_{idx}'),
                    'attack_type': attack_type,
                    'confidence': self._calculate_classification_confidence(row, attack_type),
                    'risk_level': self.attack_patterns[attack_type]['risk_level'],
                    'characteristics': self._get_matching_characteristics(row, attack_type)
                })

            # 生成攻击模式分析
            pattern_analysis = self._analyze_attack_patterns(fraud_data, classification_results)

            return {
                'total_transactions': len(data),
                'fraud_transactions': len(fraud_data),
                'attack_types': attack_type_counts,
                'classification_results': classification_results,
                'pattern_analysis': pattern_analysis,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"攻击分类失败: {e}")
            return self._empty_classification_result()

    def _classify_single_transaction(self, transaction: pd.Series) -> str:
        """对单笔交易进行攻击类型分类"""
        scores = {}

        for attack_type, pattern in self.attack_patterns.items():
            score = self._calculate_pattern_score(transaction, pattern)
            scores[attack_type] = score

        # 返回得分最高的攻击类型
        best_attack_type = max(scores, key=scores.get)
        return best_attack_type

    def _calculate_pattern_score(self, transaction: pd.Series, pattern: Dict) -> float:
        """计算交易与攻击模式的匹配分数"""
        score = 0.0
        rules = pattern['detection_rules']

        # 检查各种规则
        for rule_name, rule_value in rules.items():
            if rule_name in transaction.index:
                rule_score = self._evaluate_rule(transaction[rule_name], rule_value)
                score += rule_score

        # 应用权重
        return score * pattern['weight']

    def _evaluate_rule(self, value, rule) -> float:
        """评估单个规则"""
        if isinstance(rule, dict):
            if 'min' in rule and value < rule['min']:
                return 0.0
            if 'max' in rule and value > rule['max']:
                return 0.0
            if 'values' in rule and value not in rule['values']:
                return 0.0
            if 'extreme' in rule and rule['extreme']:
                # 检查极端值
                if isinstance(value, (int, float)):
                    if value < 18 or value > 70:  # 极端年龄
                        return 1.0
                return 0.0
            return 1.0
        elif isinstance(rule, bool):
            return 1.0 if rule else 0.0
        else:
            return 1.0 if value == rule else 0.0

    def _calculate_classification_confidence(self, transaction: pd.Series, attack_type: str) -> float:
        """计算分类置信度"""
        pattern = self.attack_patterns[attack_type]
        total_rules = len(pattern['detection_rules'])
        matched_rules = 0

        for rule_name, rule_value in pattern['detection_rules'].items():
            if rule_name in transaction.index:
                if self._evaluate_rule(transaction[rule_name], rule_value) > 0:
                    matched_rules += 1

        return round(matched_rules / total_rules, 2) if total_rules > 0 else 0.0

    def _get_matching_characteristics(self, transaction: pd.Series, attack_type: str) -> List[str]:
        """获取匹配的攻击特征"""
        pattern = self.attack_patterns[attack_type]
        matching_chars = []

        # 基于检测规则判断匹配的特征
        rules = pattern['detection_rules']

        if 'account_age_days' in rules and 'account_age_days' in transaction.index:
            if transaction['account_age_days'] <= rules['account_age_days'].get('max', float('inf')):
                matching_chars.append('新账户')

        if 'transaction_amount' in rules and 'transaction_amount' in transaction.index:
            if transaction['transaction_amount'] >= rules['transaction_amount'].get('min', 0):
                matching_chars.append('大额交易')
            elif transaction['transaction_amount'] <= rules['transaction_amount'].get('max', float('inf')):
                matching_chars.append('小额交易')

        if 'transaction_hour' in rules and 'transaction_hour' in transaction.index:
            if transaction['transaction_hour'] in rules['transaction_hour'].get('values', []):
                matching_chars.append('异常时间')

        return matching_chars

    def _analyze_attack_patterns(self, fraud_data: pd.DataFrame, classification_results: List[Dict]) -> Dict:
        """分析攻击模式"""
        analysis = {
            'dominant_attack_type': '',
            'time_patterns': {},
            'amount_patterns': {},
            'device_patterns': {},
            'recommendations': []
        }

        if not classification_results:
            return analysis

        # 找出主要攻击类型
        attack_counts = {}
        for result in classification_results:
            attack_type = result['attack_type']
            attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1

        analysis['dominant_attack_type'] = max(attack_counts, key=attack_counts.get)

        # 时间模式分析
        if 'transaction_hour' in fraud_data.columns:
            hour_counts = fraud_data['transaction_hour'].value_counts().to_dict()
            analysis['time_patterns'] = {
                'peak_hours': sorted(hour_counts, key=hour_counts.get, reverse=True)[:3],
                'night_transactions': len(fraud_data[
                    (fraud_data['transaction_hour'] >= 22) | (fraud_data['transaction_hour'] <= 6)
                ])
            }

        # 金额模式分析
        if 'transaction_amount' in fraud_data.columns:
            analysis['amount_patterns'] = {
                'avg_amount': round(fraud_data['transaction_amount'].mean(), 2),
                'large_amounts': len(fraud_data[fraud_data['transaction_amount'] > 500]),
                'small_amounts': len(fraud_data[fraud_data['transaction_amount'] < 50])
            }

        # 设备模式分析
        if 'device_used' in fraud_data.columns:
            device_counts = fraud_data['device_used'].value_counts().to_dict()
            analysis['device_patterns'] = device_counts

        # 生成建议
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """生成防护建议"""
        recommendations = []

        dominant_type = analysis['dominant_attack_type']

        if dominant_type == 'account_takeover':
            recommendations.extend([
                '加强新账户监控',
                '实施多因素认证',
                '监控异常时间段交易'
            ])
        elif dominant_type == 'identity_theft':
            recommendations.extend([
                '验证客户身份信息',
                '限制高风险商品类别',
                '加强移动设备安全'
            ])
        elif dominant_type == 'bulk_fraud':
            recommendations.extend([
                '检测批量交易模式',
                '实施交易频率限制',
                '监控相似交易行为'
            ])
        elif dominant_type == 'testing_attack':
            recommendations.extend([
                '限制小额交易频率',
                '监控新账户活动',
                '检测支付方式异常'
            ])

        return recommendations

    def _empty_classification_result(self) -> Dict:
        """返回空的分类结果"""
        return {
            'total_transactions': 0,
            'fraud_transactions': 0,
            'attack_types': {},
            'classification_results': [],
            'pattern_analysis': {},
            'timestamp': datetime.now().isoformat()
        }