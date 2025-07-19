#!/usr/bin/env python3
"""
风险预测结果显示组件
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any

def display_risk_prediction_results(risk_results: Dict[str, Any]):
    """显示风险预测结果"""
    if not risk_results or not risk_results.get('success', False):
        st.warning("⚠️ 没有可显示的风险预测结果")
        return
    
    st.markdown("### 📈 个体风险分析结果")
    
    # 1. 总体统计
    _display_overall_statistics(risk_results)
    
    # 2. 风险分层分析
    _display_risk_stratification(risk_results)
    
    # 3. 攻击类型分析
    _display_attack_type_analysis(risk_results)
    
    # 4. 个体详细分析
    _display_individual_analysis(risk_results)
    
    # 5. 防护建议
    _display_protection_recommendations(risk_results)

def _display_overall_statistics(risk_results: Dict[str, Any]):
    """显示总体统计"""
    st.markdown("#### 📊 总体统计")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_samples = risk_results.get('total_samples', 0)
    processing_time = risk_results.get('processing_time', 0)
    
    with col1:
        st.metric("分析样本数", f"{total_samples:,}")
    
    with col2:
        st.metric("处理时间", f"{processing_time:.2f}秒")
    
    # 计算平均风险评分
    risk_scores = risk_results.get('risk_scores', [])
    if risk_scores:
        avg_risk_score = np.mean(risk_scores)
        with col3:
            st.metric("平均风险评分", f"{avg_risk_score:.1f}")
        
        # 计算高风险比例
        high_risk_count = sum(1 for score in risk_scores if score >= 60)
        high_risk_percentage = high_risk_count / len(risk_scores) * 100
        with col4:
            st.metric("高风险比例", f"{high_risk_percentage:.1f}%")
    else:
        with col3:
            st.metric("平均风险评分", "N/A")
        with col4:
            st.metric("高风险比例", "N/A")

def _display_risk_stratification(risk_results: Dict[str, Any]):
    """显示风险分层分析"""
    st.markdown("#### 🎯 风险分层分析")
    
    stratification_stats = risk_results.get('stratification_stats', {})
    
    if not stratification_stats:
        st.warning("⚠️ 没有风险分层数据")
        return
    
    # 创建两列布局
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 风险分层饼图
        labels = []
        values = []
        colors = []
        
        color_map = {
            'low': '#28a745',      # 绿色
            'medium': '#ffc107',   # 黄色
            'high': '#fd7e14',     # 橙色
            'critical': '#dc3545'  # 红色
        }
        
        name_map = {
            'low': '低风险',
            'medium': '中风险', 
            'high': '高风险',
            'critical': '极高风险'
        }
        
        for level, stats in stratification_stats.items():
            count = stats.get('count', 0)
            if count > 0:
                labels.append(f"{name_map.get(level, level)} ({count})")
                values.append(count)
                colors.append(color_map.get(level, '#6c757d'))
        
        if values:
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig_pie.update_layout(
                title="风险分层分布",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # 风险分层详细表格
        st.markdown("**分层详细信息**")
        
        stratification_data = []
        for level, stats in stratification_stats.items():
            stratification_data.append({
                '风险等级': name_map.get(level, level),
                '用户数量': stats.get('count', 0),
                '占比': f"{stats.get('percentage', 0):.1f}%",
                '目标占比': f"{stats.get('target_percentage', 0):.1f}%",
                '平均评分': f"{stats.get('average_score', 0):.1f}",
                '描述': stats.get('description', '')
            })
        
        if stratification_data:
            df_stratification = pd.DataFrame(stratification_data)
            st.dataframe(df_stratification, use_container_width=True)

def _display_attack_type_analysis(risk_results: Dict[str, Any]):
    """显示攻击类型分析"""
    st.markdown("#### 🔍 攻击类型分析")
    
    attack_predictions = risk_results.get('attack_predictions', [])
    
    if not attack_predictions:
        st.warning("⚠️ 没有攻击类型预测数据")
        return
    
    # 统计攻击类型分布
    attack_counts = {}
    attack_names = {}
    
    for pred in attack_predictions:
        attack_type = pred.get('attack_type', 'unknown')
        attack_name = pred.get('attack_name', '未知攻击')
        
        attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1
        attack_names[attack_type] = attack_name
    
    # 创建两列布局
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 攻击类型分布图
        if attack_counts:
            # 过滤掉'none'类型
            filtered_counts = {k: v for k, v in attack_counts.items() if k != 'none' and v > 0}
            
            if filtered_counts:
                labels = [attack_names.get(k, k) for k in filtered_counts.keys()]
                values = list(filtered_counts.values())
                
                fig_attack = px.bar(
                    x=labels,
                    y=values,
                    title="攻击类型分布",
                    labels={'x': '攻击类型', 'y': '检测数量'},
                    color=values,
                    color_continuous_scale='Reds'
                )
                
                fig_attack.update_layout(height=400)
                st.plotly_chart(fig_attack, use_container_width=True)
            else:
                st.info("✅ 未检测到明显的攻击行为")
    
    with col2:
        # 攻击类型详细信息
        st.markdown("**攻击类型详情**")
        
        attack_data = []
        for attack_type, count in attack_counts.items():
            if attack_type != 'none' and count > 0:
                attack_data.append({
                    '攻击类型': attack_names.get(attack_type, attack_type),
                    '检测数量': count,
                    '占比': f"{count / len(attack_predictions) * 100:.1f}%"
                })
        
        if attack_data:
            df_attacks = pd.DataFrame(attack_data)
            st.dataframe(df_attacks, use_container_width=True)
        else:
            st.success("✅ 未检测到攻击行为")

def _display_individual_analysis(risk_results: Dict[str, Any]):
    """显示个体详细分析"""
    st.markdown("#### 👤 个体详细分析")
    
    individual_analyses = risk_results.get('individual_analyses', [])
    
    if not individual_analyses:
        st.warning("⚠️ 没有个体分析数据")
        return
    
    # 显示高风险用户
    high_risk_users = [
        analysis for analysis in individual_analyses 
        if analysis.get('risk_level') in ['high', 'critical']
    ]
    
    if high_risk_users:
        st.markdown("**🚨 高风险用户详情**")
        
        # 限制显示数量
        display_count = min(10, len(high_risk_users))
        
        for i, analysis in enumerate(high_risk_users[:display_count]):
            with st.expander(f"用户 {analysis.get('user_id', f'user_{i}')} - {analysis.get('risk_level', 'unknown')}风险"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**风险评分**: {analysis.get('risk_score', 0):.1f}")
                    st.write(f"**风险等级**: {analysis.get('risk_level', 'unknown')}")
                    st.write(f"**监控级别**: {analysis.get('monitoring_level', 'unknown')}")
                
                with col2:
                    attack_pred = analysis.get('attack_prediction', {})
                    st.write(f"**攻击类型**: {attack_pred.get('attack_name', '未知')}")
                    st.write(f"**置信度**: {attack_pred.get('confidence', 0):.2f}")
                
                # 显示建议措施
                actions = analysis.get('recommended_actions', [])
                if actions:
                    st.write("**建议措施**:")
                    for action in actions:
                        st.write(f"- {action}")
        
        if len(high_risk_users) > display_count:
            st.info(f"还有 {len(high_risk_users) - display_count} 个高风险用户未显示")
    else:
        st.success("✅ 未发现高风险用户")

def _display_protection_recommendations(risk_results: Dict[str, Any]):
    """显示防护建议"""
    st.markdown("#### 🛡️ 防护建议")
    
    protection_recommendations = risk_results.get('protection_recommendations', {})
    
    if not protection_recommendations:
        st.warning("⚠️ 没有防护建议数据")
        return
    
    # 立即行动建议
    immediate_actions = protection_recommendations.get('immediate_actions', [])
    if immediate_actions:
        st.markdown("**🚨 立即行动**")
        for action in immediate_actions:
            st.error(f"🔥 {action}")
    
    # 监控增强建议
    monitoring_enhancements = protection_recommendations.get('monitoring_enhancements', [])
    if monitoring_enhancements:
        st.markdown("**👁️ 监控增强**")
        for enhancement in monitoring_enhancements:
            st.warning(f"⚠️ {enhancement}")
    
    # 系统改进建议
    system_improvements = protection_recommendations.get('system_improvements', [])
    if system_improvements:
        st.markdown("**🔧 系统改进**")
        for improvement in system_improvements:
            st.info(f"💡 {improvement}")
    
    # 如果没有特殊建议
    if not any([immediate_actions, monitoring_enhancements, system_improvements]):
        st.success("✅ 当前风险水平可控，继续保持现有安全措施")

def display_risk_score_distribution(risk_results: Dict[str, Any]):
    """显示风险评分分布"""
    st.markdown("#### 📊 风险评分分布")
    
    risk_scores = risk_results.get('risk_scores', [])
    
    if not risk_scores:
        st.warning("⚠️ 没有风险评分数据")
        return
    
    # 创建风险评分分布直方图
    fig_hist = px.histogram(
        x=risk_scores,
        nbins=20,
        title="风险评分分布",
        labels={'x': '风险评分', 'y': '用户数量'},
        color_discrete_sequence=['#1f77b4']
    )
    
    # 添加风险阈值线
    fig_hist.add_vline(x=40, line_dash="dash", line_color="green", annotation_text="低风险阈值")
    fig_hist.add_vline(x=60, line_dash="dash", line_color="orange", annotation_text="中风险阈值")
    fig_hist.add_vline(x=80, line_dash="dash", line_color="red", annotation_text="高风险阈值")
    
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # 显示统计信息
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("最低评分", f"{min(risk_scores):.1f}")
    with col2:
        st.metric("最高评分", f"{max(risk_scores):.1f}")
    with col3:
        st.metric("平均评分", f"{np.mean(risk_scores):.1f}")
    with col4:
        st.metric("标准差", f"{np.std(risk_scores):.1f}")
