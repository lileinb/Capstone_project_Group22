"""
Attack Pattern Analyzer
Analyze attack patterns, trends and correlations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import json
import os
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttackPatternAnalyzer:
    """Attack pattern analyzer"""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize attack pattern analyzer

        Args:
            data_dir: Data directory
        """
        self.data_dir = data_dir
        self.attack_records = []
        self.pattern_cache = {}
        self.trend_data = defaultdict(list)
        self.cluster_patterns = {}  # Store cluster-based attack patterns
        self._load_attack_records()

    def _load_attack_records(self):
        """Load attack records"""
        try:
            records_file = os.path.join(self.data_dir, 'attack_records.json')
            if os.path.exists(records_file):
                with open(records_file, 'r', encoding='utf-8') as f:
                    self.attack_records = json.load(f)
                logger.info(f"Loaded {len(self.attack_records)} attack records")
        except Exception as e:
            logger.error(f"Error loading attack records: {e}")

    def _save_attack_records(self):
        """Save attack records"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            records_file = os.path.join(self.data_dir, 'attack_records.json')
            with open(records_file, 'w', encoding='utf-8') as f:
                json.dump(self.attack_records, f, indent=2, ensure_ascii=False)
            logger.info("攻击记录已保存")
        except Exception as e:
            logger.error(f"保存攻击记录时出错: {e}")
    
    def add_attack_record(self, attack_data: Dict):
        """添加攻击记录"""
        record = {
            'id': len(self.attack_records) + 1,
            'timestamp': datetime.now().isoformat(),
            'attack_type': attack_data.get('attack_type', 'unknown'),
            'risk_score': attack_data.get('risk_score', 0.0),
            'confidence': attack_data.get('confidence', 0.0),
            'transaction_data': attack_data.get('transaction_data', {}),
            'attack_characteristics': attack_data.get('attack_characteristics', []),
            'recommended_actions': attack_data.get('recommended_actions', [])
        }
        
        self.attack_records.append(record)
        self._save_attack_records()
        
        # 更新趋势数据
        self._update_trend_data(record)
    
    def _update_trend_data(self, record: Dict):
        """更新趋势数据"""
        attack_type = record['attack_type']
        timestamp = datetime.fromisoformat(record['timestamp'])
        
        self.trend_data[attack_type].append({
            'timestamp': timestamp,
            'risk_score': record['risk_score'],
            'confidence': record['confidence']
        })
    
    def analyze_attack_patterns(self, time_window: str = '7d') -> Dict:
        """
        Analyze attack patterns

        Args:
            time_window: Time window ('1d', '7d', '30d', 'all')

        Returns:
            Attack pattern analysis results
        """
        try:
            # Filter records within time window
            filtered_records = self._filter_records_by_time(self.attack_records, time_window)

            if not filtered_records:
                return {'message': 'No attack records in specified time window'}

            # Analyze attack type distribution
            attack_type_distribution = self._analyze_attack_type_distribution(filtered_records)

            # Analyze risk score distribution
            risk_score_analysis = self._analyze_risk_score_distribution(filtered_records)

            # Analyze time patterns
            time_pattern_analysis = self._analyze_time_patterns(filtered_records)

            # Analyze feature correlation
            feature_correlation = self._analyze_feature_correlation(filtered_records)

            # Analyze attack trends
            trend_analysis = self._analyze_attack_trends(time_window)
            
            return {
                'time_window': time_window,
                'total_attacks': len(filtered_records),
                'attack_type_distribution': attack_type_distribution,
                'risk_score_analysis': risk_score_analysis,
                'time_pattern_analysis': time_pattern_analysis,
                'feature_correlation': feature_correlation,
                'trend_analysis': trend_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"分析攻击模式时出错: {e}")
            return {'error': str(e)}
    
    def _filter_records_by_time(self, records: List[Dict], time_window: str) -> List[Dict]:
        """根据时间窗口过滤记录"""
        if time_window == 'all':
            return records
        
        now = datetime.now()
        
        if time_window == '1d':
            cutoff = now - timedelta(days=1)
        elif time_window == '7d':
            cutoff = now - timedelta(days=7)
        elif time_window == '30d':
            cutoff = now - timedelta(days=30)
        else:
            return records
        
        filtered_records = []
        for record in records:
            try:
                record_time = datetime.fromisoformat(record['timestamp'])
                if record_time >= cutoff:
                    filtered_records.append(record)
            except:
                continue
        
        return filtered_records
    
    def _analyze_attack_type_distribution(self, records: List[Dict]) -> Dict:
        """分析攻击类型分布"""
        attack_type_counts = Counter()
        attack_type_risk_scores = defaultdict(list)
        
        for record in records:
            attack_type = record.get('attack_type', 'unknown')
            attack_type_counts[attack_type] += 1
            attack_type_risk_scores[attack_type].append(record.get('risk_score', 0.0))
        
        distribution = {}
        total_records = len(records)
        
        for attack_type, count in attack_type_counts.items():
            risk_scores = attack_type_risk_scores[attack_type]
            distribution[attack_type] = {
                'count': count,
                'percentage': (count / total_records) * 100,
                'average_risk_score': np.mean(risk_scores),
                'risk_score_std': np.std(risk_scores)
            }
        
        return distribution
    
    def _analyze_risk_score_distribution(self, records: List[Dict]) -> Dict:
        """分析风险评分分布"""
        risk_scores = [record.get('risk_score', 0.0) for record in records]
        
        return {
            'total_records': len(risk_scores),
            'average_risk_score': np.mean(risk_scores),
            'median_risk_score': np.median(risk_scores),
            'risk_score_std': np.std(risk_scores),
            'min_risk_score': np.min(risk_scores),
            'max_risk_score': np.max(risk_scores),
            'high_risk_percentage': len([s for s in risk_scores if s >= 0.8]) / len(risk_scores) * 100,
            'medium_risk_percentage': len([s for s in risk_scores if 0.4 <= s < 0.8]) / len(risk_scores) * 100,
            'low_risk_percentage': len([s for s in risk_scores if s < 0.4]) / len(risk_scores) * 100
        }
    
    def _analyze_time_patterns(self, records: List[Dict]) -> Dict:
        """分析时间模式"""
        hourly_distribution = defaultdict(int)
        daily_distribution = defaultdict(int)
        
        for record in records:
            try:
                timestamp = datetime.fromisoformat(record['timestamp'])
                hour = timestamp.hour
                day = timestamp.strftime('%A')  # 星期几
                
                hourly_distribution[hour] += 1
                daily_distribution[day] += 1
            except:
                continue
        
        return {
            'hourly_distribution': dict(hourly_distribution),
            'daily_distribution': dict(daily_distribution),
            'peak_hour': max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else None,
            'peak_day': max(daily_distribution.items(), key=lambda x: x[1])[0] if daily_distribution else None
        }
    
    def _analyze_feature_correlation(self, records: List[Dict]) -> Dict:
        """分析特征关联性"""
        # 分析攻击类型与风险评分的关联
        attack_type_risk_correlation = {}
        
        for record in records:
            attack_type = record.get('attack_type', 'unknown')
            risk_score = record.get('risk_score', 0.0)
            
            if attack_type not in attack_type_risk_correlation:
                attack_type_risk_correlation[attack_type] = []
            
            attack_type_risk_correlation[attack_type].append(risk_score)
        
        # 计算平均风险评分
        for attack_type, risk_scores in attack_type_risk_correlation.items():
            attack_type_risk_correlation[attack_type] = np.mean(risk_scores)
        
        return {
            'attack_type_risk_correlation': attack_type_risk_correlation,
            'high_risk_attack_types': [k for k, v in attack_type_risk_correlation.items() if v >= 0.7],
            'low_risk_attack_types': [k for k, v in attack_type_risk_correlation.items() if v < 0.4]
        }
    
    def _analyze_attack_trends(self, time_window: str) -> Dict:
        """分析攻击趋势"""
        trends = {}
        
        for attack_type, trend_data in self.trend_data.items():
            if not trend_data:
                continue
            
            # 按时间排序
            sorted_data = sorted(trend_data, key=lambda x: x['timestamp'])
            
            # 计算趋势指标
            if len(sorted_data) >= 2:
                recent_data = sorted_data[-10:]  # 最近10条记录
                older_data = sorted_data[:-10] if len(sorted_data) > 10 else sorted_data
                
                if older_data:
                    recent_avg_risk = np.mean([d['risk_score'] for d in recent_data])
                    older_avg_risk = np.mean([d['risk_score'] for d in older_data])
                    
                    risk_trend = 'increasing' if recent_avg_risk > older_avg_risk else 'decreasing'
                    risk_change = recent_avg_risk - older_avg_risk
                else:
                    risk_trend = 'stable'
                    risk_change = 0.0
                
                trends[attack_type] = {
                    'total_occurrences': len(sorted_data),
                    'recent_occurrences': len(recent_data),
                    'average_risk_score': np.mean([d['risk_score'] for d in sorted_data]),
                    'risk_trend': risk_trend,
                    'risk_change': risk_change,
                    'frequency_trend': self._calculate_frequency_trend(sorted_data)
                }
        
        return trends
    
    def _calculate_frequency_trend(self, sorted_data: List[Dict]) -> str:
        """计算频率趋势"""
        if len(sorted_data) < 2:
            return 'stable'
        
        # 计算时间间隔
        time_intervals = []
        for i in range(1, len(sorted_data)):
            interval = (sorted_data[i]['timestamp'] - sorted_data[i-1]['timestamp']).total_seconds() / 3600  # 小时
            time_intervals.append(interval)
        
        if not time_intervals:
            return 'stable'
        
        # 分析间隔趋势
        recent_intervals = time_intervals[-5:] if len(time_intervals) >= 5 else time_intervals
        older_intervals = time_intervals[:-5] if len(time_intervals) > 5 else []
        
        if older_intervals:
            recent_avg_interval = np.mean(recent_intervals)
            older_avg_interval = np.mean(older_intervals)
            
            if recent_avg_interval < older_avg_interval:
                return 'increasing'  # 间隔变小，频率增加
            elif recent_avg_interval > older_avg_interval:
                return 'decreasing'  # 间隔变大，频率减少
            else:
                return 'stable'
        
        return 'stable'
    
    def detect_anomalies(self, time_window: str = '7d') -> List[Dict]:
        """
        Detect anomaly patterns

        Args:
            time_window: Time window

        Returns:
            List of anomaly patterns
        """
        anomalies = []

        # Analyze attack records
        pattern_analysis = self.analyze_attack_patterns(time_window)

        if 'error' in pattern_analysis:
            return anomalies

        # Detect abnormal attack types
        attack_distribution = pattern_analysis.get('attack_type_distribution', {})
        for attack_type, stats in attack_distribution.items():
            if stats['percentage'] > 50:  # Single attack type proportion too high
                anomalies.append({
                    'type': 'high_attack_type_concentration',
                    'attack_type': attack_type,
                    'percentage': stats['percentage'],
                    'severity': 'high' if stats['percentage'] > 70 else 'medium',
                    'description': f'Attack type {attack_type} proportion too high ({stats["percentage"]:.1f}%)'
                })
        
        # 检测异常风险评分
        risk_analysis = pattern_analysis.get('risk_score_analysis', {})
        if risk_analysis.get('high_risk_percentage', 0) > 30:
            anomalies.append({
                'type': 'high_risk_concentration',
                'percentage': risk_analysis['high_risk_percentage'],
                'severity': 'high',
                'description': f'高风险攻击占比过高 ({risk_analysis["high_risk_percentage"]:.1f}%)'
            })
        
        # 检测时间异常
        time_analysis = pattern_analysis.get('time_pattern_analysis', {})
        hourly_dist = time_analysis.get('hourly_distribution', {})
        if hourly_dist:
            max_hour_count = max(hourly_dist.values())
            total_count = sum(hourly_dist.values())
            if max_hour_count / total_count > 0.3:  # 单一小时占比过高
                peak_hour = max(hourly_dist.items(), key=lambda x: x[1])[0]
                anomalies.append({
                    'type': 'time_concentration',
                    'peak_hour': peak_hour,
                    'percentage': (max_hour_count / total_count) * 100,
                    'severity': 'medium',
                    'description': f'攻击集中在 {peak_hour} 时 ({max_hour_count/total_count*100:.1f}%)'
                })
        
        return anomalies
    
    def generate_pattern_report(self, time_window: str = '7d') -> Dict:
        """Generate pattern analysis report"""
        pattern_analysis = self.analyze_attack_patterns(time_window)
        anomalies = self.detect_anomalies(time_window)

        # Generate recommendations
        recommendations = self._generate_recommendations(pattern_analysis, anomalies)
        
        return {
            'summary': {
                'time_window': time_window,
                'total_attacks': pattern_analysis.get('total_attacks', 0),
                'anomalies_detected': len(anomalies),
                'report_generated_at': datetime.now().isoformat()
            },
            'analysis': pattern_analysis,
            'anomalies': anomalies,
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, pattern_analysis: Dict, anomalies: List[Dict]) -> List[str]:
        """Generate recommendations"""
        recommendations = []

        # Generate recommendations based on anomalies
        for anomaly in anomalies:
            if anomaly['type'] == 'high_attack_type_concentration':
                recommendations.append(
                    f"Recommend strengthening protection measures against {anomaly['attack_type']} type attacks"
                )
            elif anomaly['type'] == 'high_risk_concentration':
                recommendations.append(
                    "Recommend improving overall security protection level and strengthening risk monitoring"
                )
            elif anomaly['type'] == 'time_concentration':
                recommendations.append(
                    f"Recommend strengthening monitoring and protection at {anomaly['peak_hour']} o'clock"
                )

        # Generate recommendations based on trends
        trend_analysis = pattern_analysis.get('trend_analysis', {})
        for attack_type, trend in trend_analysis.items():
            if trend.get('risk_trend') == 'increasing':
                recommendations.append(
                    f"Recommend paying attention to the rising risk trend of {attack_type} attacks"
                )
            if trend.get('frequency_trend') == 'increasing':
                recommendations.append(
                    f"Recommend strengthening frequency monitoring of {attack_type} attacks"
                )

        return recommendations
    
    def get_attack_statistics(self) -> Dict:
        """Get attack statistics"""
        if not self.attack_records:
            return {'message': 'No attack records'}
        
        total_records = len(self.attack_records)
        attack_types = Counter([r.get('attack_type', 'unknown') for r in self.attack_records])
        risk_scores = [r.get('risk_score', 0.0) for r in self.attack_records]
        
        return {
            'total_attacks': total_records,
            'unique_attack_types': len(attack_types),
            'most_common_attack_type': attack_types.most_common(1)[0] if attack_types else None,
            'average_risk_score': np.mean(risk_scores),
            'high_risk_attacks': len([s for s in risk_scores if s >= 0.8]),
            'attack_type_distribution': dict(attack_types)
        }

    def analyze_cluster_based_attack_patterns(self, data: pd.DataFrame,
                                            cluster_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于聚类结果分析攻击模式

        Args:
            data: 原始数据
            cluster_results: 聚类分析结果

        Returns:
            基于聚类的攻击模式分析结果
        """
        try:
            logger.info("开始基于聚类的攻击模式分析")

            cluster_labels = cluster_results.get('cluster_labels', [])
            cluster_details = cluster_results.get('cluster_details', [])

            if not cluster_labels or not cluster_details:
                logger.warning("聚类结果不完整，无法进行攻击模式分析")
                return {}

            attack_patterns = {}

            for detail in cluster_details:
                cluster_id = detail.get('cluster_id', -1)
                risk_level = detail.get('risk_level', 'low')
                fraud_rate = detail.get('fraud_rate', 0)
                size = detail.get('size', 0)

                # 获取该聚类的数据
                cluster_mask = [label == cluster_id for label in cluster_labels]
                cluster_data = data[cluster_mask] if len(cluster_mask) == len(data) else pd.DataFrame()

                if cluster_data.empty:
                    continue

                # 分析攻击模式特征
                pattern_features = self._extract_cluster_attack_features(cluster_data)

                # 确定攻击类型
                attack_type = self._classify_cluster_attack_type(pattern_features, risk_level, fraud_rate)

                # 计算威胁等级
                threat_level = self._calculate_cluster_threat_level(risk_level, fraud_rate, size, len(data))

                attack_patterns[cluster_id] = {
                    'cluster_id': cluster_id,
                    'attack_type': attack_type,
                    'threat_level': threat_level,
                    'risk_level': risk_level,
                    'fraud_rate': fraud_rate,
                    'affected_samples': size,
                    'pattern_features': pattern_features,
                    'recommendations': self._generate_cluster_attack_recommendations(attack_type, threat_level)
                }

            # 生成总体攻击态势分析
            overall_analysis = self._generate_cluster_overall_analysis(attack_patterns, data)

            result = {
                'cluster_attack_patterns': attack_patterns,
                'overall_analysis': overall_analysis,
                'analysis_timestamp': datetime.now().isoformat(),
                'total_clusters_analyzed': len(attack_patterns)
            }

            # 缓存结果
            self.cluster_patterns = attack_patterns

            logger.info(f"聚类攻击模式分析完成，发现 {len(attack_patterns)} 种模式")
            return result

        except Exception as e:
            logger.error(f"聚类攻击模式分析失败: {e}")
            return {}

    def _extract_cluster_attack_features(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """提取聚类的攻击特征"""
        features = {}

        try:
            # 交易金额特征
            if 'transaction_amount' in cluster_data.columns:
                features['avg_amount'] = float(cluster_data['transaction_amount'].mean())
                features['amount_variance'] = float(cluster_data['transaction_amount'].var())
                features['high_amount_rate'] = float((cluster_data['transaction_amount'] >
                                                    cluster_data['transaction_amount'].quantile(0.9)).mean())

            # 时间模式特征
            if 'transaction_hour' in cluster_data.columns:
                hour_dist = cluster_data['transaction_hour'].value_counts(normalize=True)
                features['peak_hours'] = hour_dist.head(3).index.tolist()
                features['night_transaction_rate'] = float(cluster_data['transaction_hour'].isin([22, 23, 0, 1, 2, 3]).mean())

            # 账户特征
            if 'account_age_days' in cluster_data.columns:
                features['avg_account_age'] = float(cluster_data['account_age_days'].mean())
                features['new_account_rate'] = float((cluster_data['account_age_days'] < 90).mean())

            return features

        except Exception as e:
            logger.warning(f"攻击特征提取失败: {e}")
            return {}

    def _classify_cluster_attack_type(self, features: Dict, risk_level: str, fraud_rate: float) -> str:
        """基于聚类特征分类攻击类型"""
        if risk_level == 'critical' or fraud_rate > 0.3:
            return 'high_risk_fraud'
        elif risk_level == 'high' or fraud_rate > 0.15:
            if features.get('night_transaction_rate', 0) > 0.5:
                return 'suspicious_timing'
            elif features.get('high_amount_rate', 0) > 0.3:
                return 'large_amount_fraud'
            else:
                return 'moderate_risk_fraud'
        elif risk_level == 'medium' or fraud_rate > 0.05:
            return 'potential_fraud'
        else:
            return 'normal_behavior'

    def _calculate_cluster_threat_level(self, risk_level: str, fraud_rate: float,
                                      cluster_size: int, total_size: int) -> str:
        """计算聚类威胁等级"""
        impact_ratio = cluster_size / total_size

        if risk_level == 'critical' or (fraud_rate > 0.3 and impact_ratio > 0.1):
            return 'critical'
        elif risk_level == 'high' or (fraud_rate > 0.15 and impact_ratio > 0.05):
            return 'high'
        elif risk_level == 'medium' or fraud_rate > 0.05:
            return 'medium'
        else:
            return 'low'

    def _generate_cluster_attack_recommendations(self, attack_type: str, threat_level: str) -> List[str]:
        """生成基于聚类的攻击应对建议"""
        recommendations = []

        if threat_level == 'critical':
            recommendations.extend([
                "立即启动紧急响应程序",
                "暂停相关账户的交易权限",
                "通知安全团队进行深度调查"
            ])
        elif threat_level == 'high':
            recommendations.extend([
                "加强对该模式的监控",
                "实施额外的身份验证",
                "考虑临时限制交易额度"
            ])
        elif threat_level == 'medium':
            recommendations.extend([
                "增加人工审核频率",
                "部署针对性的检测规则"
            ])

        if attack_type == 'suspicious_timing':
            recommendations.append("重点监控夜间交易活动")
        elif attack_type == 'large_amount_fraud':
            recommendations.append("对大额交易实施严格审核")

        return recommendations

    def _generate_cluster_overall_analysis(self, attack_patterns: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """生成聚类攻击态势总体分析"""
        threat_levels = [pattern['threat_level'] for pattern in attack_patterns.values()]
        attack_types = [pattern['attack_type'] for pattern in attack_patterns.values()]

        return {
            'total_threat_clusters': len([t for t in threat_levels if t in ['high', 'critical']]),
            'dominant_attack_type': max(set(attack_types), key=attack_types.count) if attack_types else 'none',
            'overall_threat_level': 'high' if 'critical' in threat_levels else
                                  'medium' if 'high' in threat_levels else 'low',
            'affected_population_ratio': sum(p['affected_samples'] for p in attack_patterns.values()) / len(data),
            'recommendations': [
                "基于聚类分析，建议重点关注高风险聚类",
                "实施差异化的风险控制策略",
                "定期更新聚类模型以适应新的攻击模式"
            ]
        }