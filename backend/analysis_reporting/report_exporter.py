"""
报告导出器
支持多种格式的报告导出，包括PDF、Excel、HTML等
"""

import pandas as pd
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import logging
import base64
from io import BytesIO

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportExporter:
    """报告导出器"""
    
    def __init__(self, export_dir: str = "exports"):
        """
        初始化报告导出器
        
        Args:
            export_dir: 导出文件目录
        """
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
    
    def export_to_excel(self, report: Dict, filename: Optional[str] = None) -> str:
        """
        导出报告到Excel文件
        
        Args:
            report: 报告数据
            filename: 文件名
            
        Returns:
            导出文件路径
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"fraud_analysis_report_{timestamp}.xlsx"
            
            filepath = os.path.join(self.export_dir, filename)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 导出执行摘要
                self._export_executive_summary(report, writer)
                
                # 导出风险分析
                self._export_risk_analysis(report, writer)
                
                # 导出聚类分析
                self._export_clustering_analysis(report, writer)
                
                # 导出攻击分析
                self._export_attack_analysis(report, writer)
                
                # 导出趋势分析
                self._export_trend_analysis(report, writer)
                
                # 导出异常检测
                self._export_anomaly_detection(report, writer)
                
                # 导出建议和行动项
                self._export_recommendations(report, writer)
                
                # 导出关键指标
                self._export_key_metrics(report, writer)
            
            logger.info(f"报告已导出到Excel: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"导出Excel报告时出错: {e}")
            return ""
    
    def _export_executive_summary(self, report: Dict, writer):
        """导出执行摘要"""
        sections = report.get('sections', {})
        exec_summary = sections.get('executive_summary', {})
        
        # 创建执行摘要数据
        summary_data = []
        
        # 概览信息
        overview = exec_summary.get('overview', {})
        summary_data.append(['概览信息', ''])
        summary_data.append(['总交易数', overview.get('total_transactions', 0)])
        summary_data.append(['高风险交易数', overview.get('high_risk_transactions', 0)])
        summary_data.append(['平均风险评分', f"{overview.get('risk_score_average', 0.0):.3f}"])
        summary_data.append(['检测到的攻击类型数', overview.get('attack_types_detected', 0)])
        summary_data.append(['识别的聚类数', overview.get('clusters_identified', 0)])
        summary_data.append(['', ''])
        
        # 关键发现
        key_findings = exec_summary.get('key_findings', [])
        summary_data.append(['关键发现', ''])
        for i, finding in enumerate(key_findings, 1):
            summary_data.append([f"{i}.", finding])
        summary_data.append(['', ''])
        
        # 风险等级
        summary_data.append(['整体风险等级', exec_summary.get('risk_level', 'UNKNOWN')])
        
        # 创建DataFrame并写入Excel
        df = pd.DataFrame(summary_data, columns=['项目', '数值'])
        df.to_excel(writer, sheet_name='执行摘要', index=False)
    
    def _export_risk_analysis(self, report: Dict, writer):
        """导出风险分析"""
        sections = report.get('sections', {})
        risk_analysis = sections.get('risk_analysis', {})
        
        # 风险分布数据
        risk_dist = risk_analysis.get('risk_distribution', {})
        risk_data = [
            ['风险等级', '百分比'],
            ['低风险', f"{risk_dist.get('low_risk', 0.0):.2f}%"],
            ['中等风险', f"{risk_dist.get('medium_risk', 0.0):.2f}%"],
            ['高风险', f"{risk_dist.get('high_risk', 0.0):.2f}%"]
        ]
        
        df_risk_dist = pd.DataFrame(risk_data[1:], columns=risk_data[0])
        df_risk_dist.to_excel(writer, sheet_name='风险分布', index=False)
        
        # 风险指标数据
        risk_metrics = risk_analysis.get('risk_metrics', {})
        metrics_data = [
            ['指标', '数值'],
            ['平均风险评分', f"{risk_metrics.get('average_risk_score', 0.0):.3f}"],
            ['风险评分标准差', f"{risk_metrics.get('risk_score_std', 0.0):.3f}"],
            ['最大风险评分', f"{risk_metrics.get('max_risk_score', 0.0):.3f}"],
            ['最小风险评分', f"{risk_metrics.get('min_risk_score', 0.0):.3f}"]
        ]
        
        df_metrics = pd.DataFrame(metrics_data[1:], columns=metrics_data[0])
        df_metrics.to_excel(writer, sheet_name='风险指标', index=False)
    
    def _export_clustering_analysis(self, report: Dict, writer):
        """导出聚类分析"""
        sections = report.get('sections', {})
        clustering_analysis = sections.get('clustering_analysis', {})
        
        # 聚类摘要
        cluster_summary = clustering_analysis.get('cluster_summary', {})
        summary_data = [
            ['指标', '数值'],
            ['总聚类数', cluster_summary.get('total_clusters', 0)],
            ['异常聚类数', cluster_summary.get('anomaly_clusters', 0)],
            ['正常聚类数', cluster_summary.get('normal_clusters', 0)]
        ]
        
        df_summary = pd.DataFrame(summary_data[1:], columns=summary_data[0])
        df_summary.to_excel(writer, sheet_name='聚类摘要', index=False)
        
        # 聚类详情
        cluster_details = clustering_analysis.get('cluster_details', [])
        if cluster_details:
            details_data = []
            for detail in cluster_details:
                details_data.append([
                    detail.get('cluster_id', ''),
                    detail.get('size', 0),
                    detail.get('risk_level', ''),
                    detail.get('anomaly_score', 0.0),
                    detail.get('key_features', '')
                ])
            
            df_details = pd.DataFrame(details_data, 
                                   columns=['聚类ID', '大小', '风险等级', '异常评分', '关键特征'])
            df_details.to_excel(writer, sheet_name='聚类详情', index=False)
    
    def _export_attack_analysis(self, report: Dict, writer):
        """导出攻击分析"""
        sections = report.get('sections', {})
        attack_analysis = sections.get('attack_analysis', {})
        
        # 攻击类型分布
        attack_dist = attack_analysis.get('attack_distribution', {})
        if attack_dist:
            dist_data = []
            for attack_type, stats in attack_dist.items():
                dist_data.append([
                    attack_type,
                    stats.get('count', 0),
                    f"{stats.get('percentage', 0.0):.2f}%",
                    f"{stats.get('average_risk_score', 0.0):.3f}"
                ])
            
            df_dist = pd.DataFrame(dist_data, 
                                 columns=['攻击类型', '数量', '百分比', '平均风险评分'])
            df_dist.to_excel(writer, sheet_name='攻击类型分布', index=False)
        
        # 推荐动作
        recommended_actions = attack_analysis.get('recommended_actions', [])
        if recommended_actions:
            actions_data = [[i+1, action] for i, action in enumerate(recommended_actions)]
            df_actions = pd.DataFrame(actions_data, columns=['序号', '推荐动作'])
            df_actions.to_excel(writer, sheet_name='推荐动作', index=False)
    
    def _export_trend_analysis(self, report: Dict, writer):
        """导出趋势分析"""
        sections = report.get('sections', {})
        trend_analysis = sections.get('trend_analysis', {})
        
        # 风险趋势
        risk_trends = trend_analysis.get('risk_trends', {})
        trend_data = [
            ['趋势指标', '数值'],
            ['趋势方向', risk_trends.get('trend_direction', 'unknown')],
            ['趋势强度', f"{risk_trends.get('trend_strength', 0.0):.3f}"]
        ]
        
        df_trends = pd.DataFrame(trend_data[1:], columns=trend_data[0])
        df_trends.to_excel(writer, sheet_name='趋势分析', index=False)
    
    def _export_anomaly_detection(self, report: Dict, writer):
        """导出异常检测"""
        sections = report.get('sections', {})
        anomaly_detection = sections.get('anomaly_detection', {})
        
        # 异常列表
        anomaly_list = anomaly_detection.get('anomaly_list', [])
        if anomaly_list:
            anomaly_data = []
            for anomaly in anomaly_list:
                anomaly_data.append([
                    anomaly.get('type', ''),
                    anomaly.get('severity', ''),
                    anomaly.get('description', ''),
                    anomaly.get('recommendation', '')
                ])
            
            df_anomalies = pd.DataFrame(anomaly_data, 
                                      columns=['异常类型', '严重程度', '描述', '建议'])
            df_anomalies.to_excel(writer, sheet_name='异常检测', index=False)
    
    def _export_recommendations(self, report: Dict, writer):
        """导出建议和行动项"""
        sections = report.get('sections', {})
        recommendations = sections.get('recommendations', {})
        
        # 建议列表
        rec_list = recommendations.get('recommendations', [])
        if rec_list:
            rec_data = []
            for rec in rec_list:
                rec_data.append([
                    rec.get('category', ''),
                    rec.get('priority', ''),
                    rec.get('description', ''),
                    '; '.join(rec.get('action_items', []))
                ])
            
            df_rec = pd.DataFrame(rec_data, 
                                columns=['类别', '优先级', '描述', '行动项'])
            df_rec.to_excel(writer, sheet_name='建议和行动项', index=False)
    
    def _export_key_metrics(self, report: Dict, writer):
        """导出关键指标"""
        sections = report.get('sections', {})
        key_metrics = sections.get('key_metrics', {})
        
        # 风险指标
        risk_metrics = key_metrics.get('risk_metrics', {})
        risk_data = [
            ['风险指标', '数值'],
            ['总交易数', risk_metrics.get('total_transactions', 0)],
            ['欺诈检测率', f"{risk_metrics.get('fraud_detection_rate', 0.0):.2f}%"],
            ['误报率', f"{risk_metrics.get('false_positive_rate', 0.0):.2f}%"],
            ['平均风险评分', f"{risk_metrics.get('average_risk_score', 0.0):.3f}"]
        ]
        
        df_risk = pd.DataFrame(risk_data[1:], columns=risk_data[0])
        df_risk.to_excel(writer, sheet_name='风险指标', index=False)
        
        # 安全指标
        security_metrics = key_metrics.get('security_metrics', {})
        security_data = [
            ['安全指标', '数值'],
            ['检测到的攻击类型数', security_metrics.get('attack_types_detected', 0)],
            ['检测到的异常数', security_metrics.get('anomalies_detected', 0)],
            ['响应时间(ms)', security_metrics.get('response_time', 0.0)],
            ['系统可用性(%)', security_metrics.get('system_uptime', 99.9)]
        ]
        
        df_security = pd.DataFrame(security_data[1:], columns=security_data[0])
        df_security.to_excel(writer, sheet_name='安全指标', index=False)
    
    def export_to_html(self, report: Dict, filename: Optional[str] = None) -> str:
        """
        导出报告到HTML文件
        
        Args:
            report: 报告数据
            filename: 文件名
            
        Returns:
            导出文件路径
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"fraud_analysis_report_{timestamp}.html"
            
            filepath = os.path.join(self.export_dir, filename)
            
            # 生成HTML内容
            html_content = self._generate_html_content(report)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"报告已导出到HTML: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"导出HTML报告时出错: {e}")
            return ""
    
    def _generate_html_content(self, report: Dict) -> str:
        """生成HTML内容"""
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>欺诈风险分析报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 3px; }
        .high-risk { color: #dc3545; }
        .medium-risk { color: #fd7e14; }
        .low-risk { color: #28a745; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .recommendation { background-color: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>欺诈风险分析报告</h1>
        <p>生成时间: {generated_at}</p>
        <p>时间窗口: {time_window}</p>
    </div>
    
    {content}
</body>
</html>
        """
        
        # 生成报告内容
        content = self._generate_html_sections(report)
        
        return html_template.format(
            generated_at=report.get('generated_at', ''),
            time_window=report.get('time_window', ''),
            content=content
        )
    
    def _generate_html_sections(self, report: Dict) -> str:
        """生成HTML章节内容"""
        sections = report.get('sections', {})
        content = ""
        
        # 执行摘要
        if 'executive_summary' in sections:
            content += self._generate_executive_summary_html(sections['executive_summary'])
        
        # 风险分析
        if 'risk_analysis' in sections:
            content += self._generate_risk_analysis_html(sections['risk_analysis'])
        
        # 攻击分析
        if 'attack_analysis' in sections:
            content += self._generate_attack_analysis_html(sections['attack_analysis'])
        
        # 异常检测
        if 'anomaly_detection' in sections:
            content += self._generate_anomaly_detection_html(sections['anomaly_detection'])
        
        # 建议和行动项
        if 'recommendations' in sections:
            content += self._generate_recommendations_html(sections['recommendations'])
        
        return content
    
    def _generate_executive_summary_html(self, summary: Dict) -> str:
        """生成执行摘要HTML"""
        overview = summary.get('overview', {})
        risk_level = summary.get('risk_level', 'UNKNOWN')
        risk_class = 'high-risk' if risk_level == 'HIGH' else 'medium-risk' if risk_level == 'MEDIUM' else 'low-risk'
        
        html = f"""
        <div class="section">
            <h2>执行摘要</h2>
            <div class="metric">总交易数: {overview.get('total_transactions', 0)}</div>
            <div class="metric">高风险交易数: {overview.get('high_risk_transactions', 0)}</div>
            <div class="metric">平均风险评分: {overview.get('risk_score_average', 0.0):.3f}</div>
            <div class="metric">整体风险等级: <span class="{risk_class}">{risk_level}</span></div>
            
            <h3>关键发现</h3>
            <ul>
        """
        
        for finding in summary.get('key_findings', []):
            html += f"<li>{finding}</li>"
        
        html += "</ul></div>"
        return html
    
    def _generate_risk_analysis_html(self, analysis: Dict) -> str:
        """生成风险分析HTML"""
        risk_dist = analysis.get('risk_distribution', {})
        risk_metrics = analysis.get('risk_metrics', {})
        
        html = f"""
        <div class="section">
            <h2>风险分析</h2>
            <h3>风险分布</h3>
            <table>
                <tr><th>风险等级</th><th>百分比</th></tr>
                <tr><td>低风险</td><td>{risk_dist.get('low_risk', 0.0):.2f}%</td></tr>
                <tr><td>中等风险</td><td>{risk_dist.get('medium_risk', 0.0):.2f}%</td></tr>
                <tr><td>高风险</td><td>{risk_dist.get('high_risk', 0.0):.2f}%</td></tr>
            </table>
            
            <h3>风险指标</h3>
            <div class="metric">平均风险评分: {risk_metrics.get('average_risk_score', 0.0):.3f}</div>
            <div class="metric">风险评分标准差: {risk_metrics.get('risk_score_std', 0.0):.3f}</div>
            <div class="metric">最大风险评分: {risk_metrics.get('max_risk_score', 0.0):.3f}</div>
            <div class="metric">最小风险评分: {risk_metrics.get('min_risk_score', 0.0):.3f}</div>
        </div>
        """
        
        return html
    
    def _generate_attack_analysis_html(self, analysis: Dict) -> str:
        """生成攻击分析HTML"""
        attack_dist = analysis.get('attack_distribution', {})
        recommended_actions = analysis.get('recommended_actions', [])
        
        html = """
        <div class="section">
            <h2>攻击分析</h2>
        """
        
        if attack_dist:
            html += "<h3>攻击类型分布</h3><table><tr><th>攻击类型</th><th>数量</th><th>百分比</th><th>平均风险评分</th></tr>"
            for attack_type, stats in attack_dist.items():
                html += f"<tr><td>{attack_type}</td><td>{stats.get('count', 0)}</td><td>{stats.get('percentage', 0.0):.2f}%</td><td>{stats.get('average_risk_score', 0.0):.3f}</td></tr>"
            html += "</table>"
        
        if recommended_actions:
            html += "<h3>推荐动作</h3><ul>"
            for action in recommended_actions:
                html += f"<li>{action}</li>"
            html += "</ul>"
        
        html += "</div>"
        return html
    
    def _generate_anomaly_detection_html(self, detection: Dict) -> str:
        """生成异常检测HTML"""
        anomaly_list = detection.get('anomaly_list', [])
        
        html = f"""
        <div class="section">
            <h2>异常检测</h2>
            <p>检测到的异常数量: {detection.get('anomalies_detected', 0)}</p>
        """
        
        if anomaly_list:
            html += "<table><tr><th>异常类型</th><th>严重程度</th><th>描述</th><th>建议</th></tr>"
            for anomaly in anomaly_list:
                severity_class = 'high-risk' if anomaly.get('severity') == 'high' else 'medium-risk'
                html += f"<tr><td>{anomaly.get('type', '')}</td><td class='{severity_class}'>{anomaly.get('severity', '')}</td><td>{anomaly.get('description', '')}</td><td>{anomaly.get('recommendation', '')}</td></tr>"
            html += "</table>"
        
        html += "</div>"
        return html
    
    def _generate_recommendations_html(self, recommendations: Dict) -> str:
        """生成建议和行动项HTML"""
        rec_list = recommendations.get('recommendations', [])
        
        html = """
        <div class="section">
            <h2>建议和行动项</h2>
        """
        
        for rec in rec_list:
            priority_class = 'high-risk' if rec.get('priority') == 'high' else 'medium-risk'
            html += f"""
            <div class="recommendation">
                <h4 class="{priority_class}">{rec.get('category', '')} - {rec.get('priority', '')}优先级</h4>
                <p>{rec.get('description', '')}</p>
                <ul>
            """
            for action in rec.get('action_items', []):
                html += f"<li>{action}</li>"
            html += "</ul></div>"
        
        html += "</div>"
        return html
    
    def export_to_json(self, report: Dict, filename: Optional[str] = None) -> str:
        """
        导出报告到JSON文件
        
        Args:
            report: 报告数据
            filename: 文件名
            
        Returns:
            导出文件路径
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"fraud_analysis_report_{timestamp}.json"
            
            filepath = os.path.join(self.export_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"报告已导出到JSON: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"导出JSON报告时出错: {e}")
            return ""
    
    def get_export_formats(self) -> List[str]:
        """获取支持的导出格式"""
        return ['excel', 'html', 'json']
    
    def get_export_summary(self) -> Dict:
        """获取导出摘要"""
        return {
            'export_directory': self.export_dir,
            'supported_formats': self.get_export_formats(),
            'total_exports': len(os.listdir(self.export_dir)) if os.path.exists(self.export_dir) else 0
        } 