"""
报告生成器
生成综合分析报告，包括风险分析、攻击模式、趋势分析等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
import logging
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, reports_dir: str = "reports"):
        """
        初始化报告生成器
        
        Args:
            reports_dir: 报告输出目录
        """
        self.reports_dir = reports_dir
        self.report_templates = self._load_report_templates()
        os.makedirs(reports_dir, exist_ok=True)
    
    def _load_report_templates(self) -> Dict:
        """加载报告模板"""
        return {
            'executive_summary': {
                'title': '执行摘要',
                'sections': ['overview', 'key_findings', 'risk_metrics', 'recommendations']
            },
            'detailed_analysis': {
                'title': '详细分析',
                'sections': ['risk_analysis', 'attack_patterns', 'trend_analysis', 'anomaly_detection']
            },
            'technical_report': {
                'title': '技术报告',
                'sections': ['model_performance', 'feature_analysis', 'clustering_results', 'threshold_analysis']
            },
            'dashboard_report': {
                'title': '仪表板报告',
                'sections': ['metrics_overview', 'visualizations', 'alerts_summary', 'action_items']
            }
        }
    
    def generate_comprehensive_report(self, 
                                   risk_data: Dict,
                                   clustering_data: Dict,
                                   attack_data: Dict,
                                   time_window: str = '7d') -> Dict:
        """
        生成综合报告
        
        Args:
            risk_data: 风险分析数据
            clustering_data: 聚类分析数据
            attack_data: 攻击分析数据
            time_window: 时间窗口
            
        Returns:
            综合报告字典
        """
        try:
            report = {
                'report_id': f"REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'generated_at': datetime.now().isoformat(),
                'time_window': time_window,
                'report_type': 'comprehensive',
                'sections': {}
            }
            
            # 生成执行摘要
            report['sections']['executive_summary'] = self._generate_executive_summary(
                risk_data, clustering_data, attack_data
            )
            
            # 生成风险分析
            report['sections']['risk_analysis'] = self._generate_risk_analysis(risk_data)
            
            # 生成聚类分析
            report['sections']['clustering_analysis'] = self._generate_clustering_analysis(clustering_data)
            
            # 生成攻击分析
            report['sections']['attack_analysis'] = self._generate_attack_analysis(attack_data)
            
            # 生成趋势分析
            report['sections']['trend_analysis'] = self._generate_trend_analysis(
                risk_data, attack_data
            )
            
            # 生成异常检测
            report['sections']['anomaly_detection'] = self._generate_anomaly_detection(
                risk_data, attack_data
            )
            
            # 生成建议和行动项
            report['sections']['recommendations'] = self._generate_recommendations(
                risk_data, clustering_data, attack_data
            )
            
            # 生成关键指标
            report['sections']['key_metrics'] = self._generate_key_metrics(
                risk_data, clustering_data, attack_data
            )
            
            return report

        except Exception as e:
            logger.error(f"综合报告生成失败: {e}")
            raise

    def generate_pdf_report(self, report_data: Dict) -> str:
        """
        生成PDF报告

        Args:
            report_data: 报告数据

        Returns:
            PDF文件路径
        """
        try:
            import os
            from datetime import datetime

            # 创建报告目录
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = os.path.join(reports_dir, f"fraud_analysis_report_{timestamp}.pdf")

            # 这里应该使用PDF生成库，暂时创建一个占位文件
            with open(pdf_path, "w", encoding="utf-8") as f:
                f.write("PDF报告生成功能正在开发中...")

            return pdf_path

        except Exception as e:
            logger.error(f"PDF报告生成失败: {e}")
            raise

    def generate_excel_report(self, report_data: Dict) -> str:
        """
        生成Excel报告

        Args:
            report_data: 报告数据

        Returns:
            Excel文件路径
        """
        try:
            import os
            from datetime import datetime

            # 创建报告目录
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_path = os.path.join(reports_dir, f"fraud_analysis_report_{timestamp}.xlsx")

            # 使用pandas创建Excel报告
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # 写入概览数据
                if 'overview' in report_data:
                    overview_df = pd.DataFrame([report_data['overview']])
                    overview_df.to_excel(writer, sheet_name='概览', index=False)

                # 写入详细数据
                if 'details' in report_data:
                    details_df = pd.DataFrame(report_data['details'])
                    details_df.to_excel(writer, sheet_name='详细数据', index=False)

            return excel_path

        except Exception as e:
            logger.error(f"Excel报告生成失败: {e}")
            raise

    def generate_html_report(self, report_data: Dict) -> str:
        """
        生成HTML报告

        Args:
            report_data: 报告数据

        Returns:
            HTML文件路径
        """
        try:
            import os
            from datetime import datetime

            # 创建报告目录
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = os.path.join(reports_dir, f"fraud_analysis_report_{timestamp}.html")

            # 生成HTML内容
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>欺诈检测分析报告</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; text-align: center; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>欺诈检测分析报告</h1>
                    <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>

                <div class="section">
                    <h2>报告概览</h2>
                    <p>本报告包含了欺诈检测系统的分析结果和关键指标。</p>
                </div>

                <div class="section">
                    <h2>数据统计</h2>
                    <div class="metric">
                        <strong>总样本数:</strong> {report_data.get('total_samples', 'N/A')}
                    </div>
                    <div class="metric">
                        <strong>欺诈样本数:</strong> {report_data.get('fraud_samples', 'N/A')}
                    </div>
                    <div class="metric">
                        <strong>欺诈率:</strong> {report_data.get('fraud_rate', 'N/A')}%
                    </div>
                </div>
            </body>
            </html>
            """

            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            return html_path

        except Exception as e:
            logger.error(f"HTML报告生成失败: {e}")
            raise
            
        except Exception as e:
            logger.error(f"生成综合报告时出错: {e}")
            return {'error': str(e)}
    
    def _generate_executive_summary(self, risk_data: Dict, clustering_data: Dict, attack_data: Dict) -> Dict:
        """生成执行摘要"""
        summary = {
            'title': '执行摘要',
            'overview': {
                'total_transactions': risk_data.get('total_transactions', 0),
                'high_risk_transactions': risk_data.get('high_risk_count', 0),
                'risk_score_average': risk_data.get('average_risk_score', 0.0),
                'attack_types_detected': len(attack_data.get('attack_types', [])),
                'clusters_identified': clustering_data.get('cluster_count', 0)
            },
            'key_findings': [],
            'risk_level': 'LOW'
        }
        
        # 确定整体风险等级
        avg_risk = risk_data.get('average_risk_score', 0.0)
        if avg_risk >= 0.7:
            summary['risk_level'] = 'HIGH'
        elif avg_risk >= 0.4:
            summary['risk_level'] = 'MEDIUM'
        
        # 生成关键发现
        if risk_data.get('high_risk_count', 0) > 0:
            summary['key_findings'].append(
                f"发现 {risk_data['high_risk_count']} 笔高风险交易"
            )
        
        if attack_data.get('attack_types'):
            summary['key_findings'].append(
                f"识别出 {len(attack_data['attack_types'])} 种攻击类型"
            )
        
        if clustering_data.get('anomaly_clusters'):
            summary['key_findings'].append(
                f"发现 {len(clustering_data['anomaly_clusters'])} 个异常聚类"
            )
        
        return summary
    
    def _generate_risk_analysis(self, risk_data: Dict) -> Dict:
        """生成风险分析"""
        analysis = {
            'title': '风险分析',
            'risk_distribution': {
                'low_risk': risk_data.get('low_risk_percentage', 0.0),
                'medium_risk': risk_data.get('medium_risk_percentage', 0.0),
                'high_risk': risk_data.get('high_risk_percentage', 0.0)
            },
            'risk_metrics': {
                'average_risk_score': risk_data.get('average_risk_score', 0.0),
                'risk_score_std': risk_data.get('risk_score_std', 0.0),
                'max_risk_score': risk_data.get('max_risk_score', 0.0),
                'min_risk_score': risk_data.get('min_risk_score', 0.0)
            },
            'model_performance': risk_data.get('model_performance', {}),
            'confidence_analysis': risk_data.get('confidence_analysis', {})
        }
        
        return analysis
    
    def _generate_clustering_analysis(self, clustering_data: Dict) -> Dict:
        """生成聚类分析"""
        analysis = {
            'title': '聚类分析',
            'cluster_summary': {
                'total_clusters': clustering_data.get('cluster_count', 0),
                'anomaly_clusters': len(clustering_data.get('anomaly_clusters', [])),
                'normal_clusters': len(clustering_data.get('normal_clusters', []))
            },
            'cluster_details': clustering_data.get('cluster_details', []),
            'anomaly_analysis': clustering_data.get('anomaly_analysis', {}),
            'feature_importance': clustering_data.get('feature_importance', {})
        }
        
        return analysis
    
    def _generate_attack_analysis(self, attack_data: Dict) -> Dict:
        """生成攻击分析"""
        analysis = {
            'title': '攻击分析',
            'attack_types': attack_data.get('attack_types', []),
            'attack_distribution': attack_data.get('attack_distribution', {}),
            'attack_trends': attack_data.get('attack_trends', {}),
            'pattern_analysis': attack_data.get('pattern_analysis', {}),
            'recommended_actions': attack_data.get('recommended_actions', [])
        }
        
        return analysis
    
    def _generate_trend_analysis(self, risk_data: Dict, attack_data: Dict) -> Dict:
        """生成趋势分析"""
        analysis = {
            'title': '趋势分析',
            'risk_trends': {
                'trend_direction': self._determine_trend_direction(risk_data.get('risk_trends', [])),
                'trend_strength': self._calculate_trend_strength(risk_data.get('risk_trends', [])),
                'periodic_patterns': self._identify_periodic_patterns(risk_data.get('risk_trends', []))
            },
            'attack_trends': attack_data.get('attack_trends', {}),
            'seasonal_patterns': self._analyze_seasonal_patterns(risk_data, attack_data)
        }
        
        return analysis
    
    def _generate_anomaly_detection(self, risk_data: Dict, attack_data: Dict) -> Dict:
        """生成异常检测"""
        anomalies = []
        
        # 检测高风险异常
        if risk_data.get('high_risk_percentage', 0) > 20:
            anomalies.append({
                'type': 'high_risk_concentration',
                'severity': 'high',
                'description': f'高风险交易占比过高 ({risk_data["high_risk_percentage"]:.1f}%)',
                'recommendation': '建议加强风险监控和防护措施'
            })
        
        # 检测攻击异常
        if attack_data.get('attack_types'):
            for attack_type, stats in attack_data.get('attack_distribution', {}).items():
                if stats.get('percentage', 0) > 50:
                    anomalies.append({
                        'type': 'attack_type_concentration',
                        'severity': 'medium',
                        'description': f'攻击类型 {attack_type} 占比过高 ({stats["percentage"]:.1f}%)',
                        'recommendation': f'建议加强对 {attack_type} 攻击的防护'
                    })
        
        return {
            'title': '异常检测',
            'anomalies_detected': len(anomalies),
            'anomaly_list': anomalies,
            'anomaly_severity_distribution': self._calculate_anomaly_severity(anomalies)
        }
    
    def _generate_recommendations(self, risk_data: Dict, clustering_data: Dict, attack_data: Dict) -> Dict:
        """Generate recommendations and action items"""
        recommendations = []
        action_items = []

        # Recommendations based on risk analysis
        if risk_data.get('high_risk_percentage', 0) > 15:
            recommendations.append({
                'category': 'risk_management',
                'priority': 'high',
                'description': 'High-risk transaction proportion is high, recommend strengthening risk monitoring',
                'action_items': [
                    'Adjust risk threshold settings',
                    'Add manual review process',
                    'Strengthen real-time monitoring'
                ]
            })

        # Recommendations based on attack analysis
        if attack_data.get('attack_types'):
            recommendations.append({
                'category': 'security_enhancement',
                'priority': 'high',
                'description': 'Multiple attack types detected, recommend strengthening security protection',
                'action_items': [
                    'Update security policies',
                    'Strengthen identity verification',
                    'Implement multi-layer protection'
                ]
            })
        
        # 基于聚类分析的建议
        if clustering_data.get('anomaly_clusters'):
            recommendations.append({
                'category': 'pattern_analysis',
                'priority': 'medium',
                'description': '发现异常交易模式，建议深入分析',
                'action_items': [
                    '分析异常聚类特征',
                    '建立模式识别规则',
                    '优化检测算法'
                ]
            })
        
        return {
            'title': '建议和行动项',
            'recommendations': recommendations,
            'action_items': action_items,
            'priority_distribution': self._calculate_priority_distribution(recommendations)
        }
    
    def _generate_key_metrics(self, risk_data: Dict, clustering_data: Dict, attack_data: Dict) -> Dict:
        """生成关键指标"""
        metrics = {
            'title': '关键指标',
            'risk_metrics': {
                'total_transactions': risk_data.get('total_transactions', 0),
                'fraud_detection_rate': risk_data.get('fraud_detection_rate', 0.0),
                'false_positive_rate': risk_data.get('false_positive_rate', 0.0),
                'average_risk_score': risk_data.get('average_risk_score', 0.0)
            },
            'security_metrics': {
                'attack_types_detected': len(attack_data.get('attack_types', [])),
                'anomalies_detected': len(clustering_data.get('anomaly_clusters', [])),
                'response_time': risk_data.get('average_response_time', 0.0),
                'system_uptime': 99.9  # 示例值
            },
            'performance_metrics': {
                'model_accuracy': risk_data.get('model_accuracy', 0.0),
                'processing_speed': risk_data.get('processing_speed', 0.0),
                'throughput': risk_data.get('throughput', 0.0)
            }
        }
        
        return metrics
    
    def _determine_trend_direction(self, trend_data: List[float]) -> str:
        """确定趋势方向"""
        if len(trend_data) < 2:
            return 'stable'
        
        # 简单线性回归
        x = np.arange(len(trend_data))
        slope = np.polyfit(x, trend_data, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_trend_strength(self, trend_data: List[float]) -> float:
        """计算趋势强度"""
        if len(trend_data) < 2:
            return 0.0
        
        # 计算R²值
        x = np.arange(len(trend_data))
        correlation_matrix = np.corrcoef(x, trend_data)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy ** 2
        
        return r_squared
    
    def _identify_periodic_patterns(self, trend_data: List[float]) -> List[Dict]:
        """识别周期性模式"""
        patterns = []
        
        # 简化实现 - 检测基本周期性
        if len(trend_data) >= 24:  # 至少24个数据点
            # 检测每日模式
            daily_pattern = self._detect_daily_pattern(trend_data)
            if daily_pattern:
                patterns.append(daily_pattern)
        
        return patterns
    
    def _detect_daily_pattern(self, data: List[float]) -> Optional[Dict]:
        """检测每日模式"""
        if len(data) < 24:
            return None
        
        # 计算每小时的均值
        hourly_means = []
        for hour in range(24):
            hour_data = [data[i] for i in range(hour, len(data), 24)]
            if hour_data:
                hourly_means.append(np.mean(hour_data))
        
        if len(hourly_means) == 24:
            peak_hour = np.argmax(hourly_means)
            return {
                'type': 'daily_pattern',
                'peak_hour': peak_hour,
                'pattern_strength': np.std(hourly_means)
            }
        
        return None
    
    def _analyze_seasonal_patterns(self, risk_data: Dict, attack_data: Dict) -> Dict:
        """分析季节性模式"""
        return {
            'weekly_patterns': {},
            'monthly_patterns': {},
            'seasonal_trends': {}
        }
    
    def _calculate_anomaly_severity(self, anomalies: List[Dict]) -> Dict:
        """计算异常严重程度分布"""
        severity_counts = defaultdict(int)
        
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'unknown')
            severity_counts[severity] += 1
        
        return dict(severity_counts)
    
    def _calculate_priority_distribution(self, recommendations: List[Dict]) -> Dict:
        """计算建议优先级分布"""
        priority_counts = defaultdict(int)
        
        for rec in recommendations:
            priority = rec.get('priority', 'medium')
            priority_counts[priority] += 1
        
        return dict(priority_counts)
    
    def save_report(self, report: Dict, filename: Optional[str] = None) -> str:
        """保存报告到文件"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"comprehensive_report_{timestamp}.json"
            
            filepath = os.path.join(self.reports_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"报告已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存报告时出错: {e}")
            return ""
    
    def generate_report_summary(self, report: Dict) -> Dict:
        """生成报告摘要"""
        summary = {
            'report_id': report.get('report_id', ''),
            'generated_at': report.get('generated_at', ''),
            'time_window': report.get('time_window', ''),
            'key_highlights': [],
            'critical_issues': [],
            'recommendations_count': 0
        }
        
        # 提取关键亮点
        sections = report.get('sections', {})
        
        if 'executive_summary' in sections:
            exec_summary = sections['executive_summary']
            summary['key_highlights'].append(
                f"风险等级: {exec_summary.get('risk_level', 'UNKNOWN')}"
            )
            summary['key_highlights'].append(
                f"高风险交易: {exec_summary.get('overview', {}).get('high_risk_transactions', 0)}"
            )
        
        if 'anomaly_detection' in sections:
            anomaly_section = sections['anomaly_detection']
            summary['critical_issues'] = [
                anomaly['description'] for anomaly in anomaly_section.get('anomaly_list', [])
                if anomaly.get('severity') == 'high'
            ]
        
        if 'recommendations' in sections:
            recommendations = sections['recommendations'].get('recommendations', [])
            summary['recommendations_count'] = len(recommendations)
        
        return summary 