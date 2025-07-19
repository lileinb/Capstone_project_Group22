"""
风险阈值管理器
管理风险评分阈值和动态调整规则
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThresholdManager:
    """风险阈值管理器"""
    
    def __init__(self, config_file: str = "config/risk_thresholds.json"):
        """
        初始化阈值管理器
        
        Args:
            config_file: 阈值配置文件路径
        """
        self.config_file = config_file
        self.thresholds = self._load_default_thresholds()
        self.dynamic_rules = []
        self.historical_data = []
        self._load_config()
    
    def _load_default_thresholds(self) -> Dict:
        """加载默认阈值配置"""
        return {
            'risk_levels': {
                'LOW': {'min': 0.0, 'max': 0.2, 'color': '#28a745'},
                'LOW_MEDIUM': {'min': 0.2, 'max': 0.4, 'color': '#ffc107'},
                'MEDIUM': {'min': 0.4, 'max': 0.6, 'color': '#fd7e14'},
                'MEDIUM_HIGH': {'min': 0.6, 'max': 0.8, 'color': '#dc3545'},
                'HIGH': {'min': 0.8, 'max': 1.0, 'color': '#721c24'}
            },
            'alert_thresholds': {
                'high_risk_alert': 0.8,
                'medium_risk_alert': 0.6,
                'low_risk_alert': 0.2
            },
            'confidence_thresholds': {
                'high_confidence': 0.8,
                'medium_confidence': 0.6,
                'low_confidence': 0.4
            },
            'model_consistency_thresholds': {
                'high_consistency': 0.1,
                'medium_consistency': 0.2,
                'low_consistency': 0.3
            }
        }
    
    def _load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.thresholds.update(config.get('thresholds', {}))
                    self.dynamic_rules = config.get('dynamic_rules', [])
                    logger.info("成功加载阈值配置")
            else:
                self._save_config()
                logger.info("创建默认阈值配置文件")
        except Exception as e:
            logger.error(f"加载阈值配置时出错: {e}")
    
    def _save_config(self):
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            config = {
                'thresholds': self.thresholds,
                'dynamic_rules': self.dynamic_rules,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info("阈值配置已保存")
        except Exception as e:
            logger.error(f"保存阈值配置时出错: {e}")
    
    def get_risk_level(self, risk_score: float) -> str:
        """
        根据风险评分确定风险等级
        
        Args:
            risk_score: 风险评分 (0-1)
            
        Returns:
            风险等级字符串
        """
        for level, config in self.thresholds['risk_levels'].items():
            if config['min'] <= risk_score <= config['max']:
                return level
        return 'UNKNOWN'
    
    def get_risk_color(self, risk_level: str) -> str:
        """获取风险等级对应的颜色"""
        return self.thresholds['risk_levels'].get(risk_level, {}).get('color', '#6c757d')
    
    def should_alert(self, risk_score: float, confidence: float) -> Dict:
        """
        判断是否需要发出警报
        
        Args:
            risk_score: 风险评分
            confidence: 置信度
            
        Returns:
            警报信息字典
        """
        alerts = {
            'should_alert': False,
            'alert_level': None,
            'alert_reason': [],
            'recommended_action': None
        }
        
        # 检查风险评分警报
        if risk_score >= self.thresholds['alert_thresholds']['high_risk_alert']:
            alerts['should_alert'] = True
            alerts['alert_level'] = 'HIGH'
            alerts['alert_reason'].append('高风险评分')
            alerts['recommended_action'] = '立即阻止交易并人工审核'
        
        elif risk_score >= self.thresholds['alert_thresholds']['medium_risk_alert']:
            alerts['should_alert'] = True
            alerts['alert_level'] = 'MEDIUM'
            alerts['alert_reason'].append('中等风险评分')
            alerts['recommended_action'] = '标记交易并进一步监控'
        
        # 检查置信度警报
        if confidence < self.thresholds['confidence_thresholds']['low_confidence']:
            alerts['should_alert'] = True
            alerts['alert_level'] = alerts['alert_level'] or 'LOW'
            alerts['alert_reason'].append('低置信度预测')
            alerts['recommended_action'] = alerts['recommended_action'] or '需要人工审核'
        
        return alerts
    
    def add_dynamic_rule(self, rule: Dict):
        """
        添加动态规则
        
        Args:
            rule: 规则字典，包含条件、动作等
        """
        rule['id'] = len(self.dynamic_rules) + 1
        rule['created_at'] = datetime.now().isoformat()
        rule['active'] = rule.get('active', True)
        
        self.dynamic_rules.append(rule)
        self._save_config()
        logger.info(f"添加动态规则: {rule.get('name', '未命名规则')}")
    
    def remove_dynamic_rule(self, rule_id: int):
        """移除动态规则"""
        self.dynamic_rules = [rule for rule in self.dynamic_rules if rule['id'] != rule_id]
        self._save_config()
        logger.info(f"移除动态规则: {rule_id}")
    
    def apply_dynamic_rules(self, transaction_data: Dict, risk_result: Dict) -> Dict:
        """
        应用动态规则
        
        Args:
            transaction_data: 交易数据
            risk_result: 风险评分结果
            
        Returns:
            应用规则后的结果
        """
        modified_result = risk_result.copy()
        
        for rule in self.dynamic_rules:
            if not rule.get('active', True):
                continue
            
            if self._evaluate_rule_condition(rule, transaction_data, risk_result):
                modified_result = self._apply_rule_action(rule, modified_result)
        
        return modified_result
    
    def _evaluate_rule_condition(self, rule: Dict, transaction_data: Dict, risk_result: Dict) -> bool:
        """评估规则条件"""
        condition = rule.get('condition', {})
        condition_type = condition.get('type')
        
        if condition_type == 'risk_score_threshold':
            threshold = condition.get('threshold', 0.5)
            operator = condition.get('operator', '>')
            risk_score = risk_result.get('risk_score', 0)
            
            if operator == '>':
                return risk_score > threshold
            elif operator == '>=':
                return risk_score >= threshold
            elif operator == '<':
                return risk_score < threshold
            elif operator == '<=':
                return risk_score <= threshold
            elif operator == '==':
                return risk_score == threshold
        
        elif condition_type == 'feature_value':
            feature = condition.get('feature')
            value = condition.get('value')
            operator = condition.get('operator', '==')
            
            if feature in transaction_data:
                feature_value = transaction_data[feature]
                
                if operator == '>':
                    return feature_value > value
                elif operator == '>=':
                    return feature_value >= value
                elif operator == '<':
                    return feature_value < value
                elif operator == '<=':
                    return feature_value <= value
                elif operator == '==':
                    return feature_value == value
        
        elif condition_type == 'time_based':
            # 时间相关条件（简化实现）
            current_hour = datetime.now().hour
            start_hour = condition.get('start_hour', 0)
            end_hour = condition.get('end_hour', 23)
            
            return start_hour <= current_hour <= end_hour
        
        return False
    
    def _apply_rule_action(self, rule: Dict, risk_result: Dict) -> Dict:
        """应用规则动作"""
        action = rule.get('action', {})
        action_type = action.get('type')
        
        if action_type == 'adjust_risk_score':
            adjustment = action.get('adjustment', 0)
            risk_result['risk_score'] = max(0, min(1, risk_result['risk_score'] + adjustment))
            risk_result['risk_level'] = self.get_risk_level(risk_result['risk_score'])
            risk_result['rule_applied'] = rule.get('name', f"规则{rule['id']}")
        
        elif action_type == 'set_risk_level':
            new_level = action.get('risk_level', 'MEDIUM')
            risk_result['risk_level'] = new_level
            risk_result['rule_applied'] = rule.get('name', f"规则{rule['id']}")
        
        elif action_type == 'add_flag':
            flag = action.get('flag', '')
            if 'flags' not in risk_result:
                risk_result['flags'] = []
            risk_result['flags'].append(flag)
            risk_result['rule_applied'] = rule.get('name', f"规则{rule['id']}")
        
        return risk_result
    
    def update_thresholds(self, new_thresholds: Dict):
        """更新阈值配置"""
        self.thresholds.update(new_thresholds)
        self._save_config()
        logger.info("阈值配置已更新")
    
    def get_threshold_summary(self) -> Dict:
        """获取阈值配置摘要"""
        return {
            'risk_levels': len(self.thresholds['risk_levels']),
            'alert_thresholds': self.thresholds['alert_thresholds'],
            'confidence_thresholds': self.thresholds['confidence_thresholds'],
            'dynamic_rules': len(self.dynamic_rules),
            'active_rules': len([r for r in self.dynamic_rules if r.get('active', True)])
        }
    
    def add_historical_data(self, transaction_data: Dict, risk_result: Dict):
        """添加历史数据用于动态调整"""
        historical_entry = {
            'timestamp': datetime.now().isoformat(),
            'transaction_data': transaction_data,
            'risk_result': risk_result
        }
        self.historical_data.append(historical_entry)
        
        # 保持历史数据在合理范围内
        if len(self.historical_data) > 10000:
            self.historical_data = self.historical_data[-5000:]
    
    def analyze_performance_trends(self) -> Dict:
        """分析性能趋势"""
        if len(self.historical_data) < 10:
            return {'message': '历史数据不足，无法分析趋势'}
        
        risk_scores = [entry['risk_result']['risk_score'] for entry in self.historical_data]
        
        return {
            'total_transactions': len(self.historical_data),
            'average_risk_score': np.mean(risk_scores),
            'risk_score_std': np.std(risk_scores),
            'high_risk_percentage': len([s for s in risk_scores if s >= 0.8]) / len(risk_scores) * 100,
            'low_risk_percentage': len([s for s in risk_scores if s <= 0.2]) / len(risk_scores) * 100
        }
    
    def suggest_threshold_adjustments(self) -> List[Dict]:
        """建议阈值调整"""
        if len(self.historical_data) < 50:
            return []
        
        suggestions = []
        analysis = self.analyze_performance_trends()
        
        # 基于高风险交易比例调整建议
        high_risk_pct = analysis['high_risk_percentage']
        if high_risk_pct > 15:  # 高风险交易过多
            suggestions.append({
                'type': 'lower_high_risk_threshold',
                'current_threshold': self.thresholds['alert_thresholds']['high_risk_alert'],
                'suggested_threshold': self.thresholds['alert_thresholds']['high_risk_alert'] - 0.05,
                'reason': f'高风险交易比例过高({high_risk_pct:.1f}%)，建议降低阈值'
            })
        
        elif high_risk_pct < 5:  # 高风险交易过少
            suggestions.append({
                'type': 'raise_high_risk_threshold',
                'current_threshold': self.thresholds['alert_thresholds']['high_risk_alert'],
                'suggested_threshold': self.thresholds['alert_thresholds']['high_risk_alert'] + 0.05,
                'reason': f'高风险交易比例过低({high_risk_pct:.1f}%)，建议提高阈值'
            })
        
        return suggestions 