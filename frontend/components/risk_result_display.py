#!/usr/bin/env python3
"""
Risk Prediction Results Display Component
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any

def display_risk_prediction_results(risk_results: Dict[str, Any]):
    """Display risk prediction results"""
    if not risk_results or not risk_results.get('success', False):
        st.warning("⚠️ No risk prediction results to display")
        return

    st.markdown("### 📈 Individual Risk Analysis Results")

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
    """Display overall statistics"""
    st.markdown("#### 📊 Overall Statistics")

    col1, col2, col3, col4 = st.columns(4)

    total_samples = risk_results.get('total_samples', 0)
    processing_time = risk_results.get('processing_time', 0)

    with col1:
        st.metric("Analyzed Samples", f"{total_samples:,}")

    with col2:
        st.metric("Processing Time", f"{processing_time:.2f}s")

    # 计算平均风险评分
    risk_scores = risk_results.get('risk_scores', [])
    if risk_scores:
        avg_risk_score = np.mean(risk_scores)
        with col3:
            st.metric("Average Risk Score", f"{avg_risk_score:.1f}")

        # 计算高风险比例
        high_risk_count = sum(1 for score in risk_scores if score >= 60)
        high_risk_percentage = high_risk_count / len(risk_scores) * 100
        with col4:
            st.metric("High Risk Ratio", f"{high_risk_percentage:.1f}%")
    else:
        with col3:
            st.metric("Average Risk Score", "N/A")
        with col4:
            st.metric("High Risk Ratio", "N/A")

def _display_risk_stratification(risk_results: Dict[str, Any]):
    """Display risk stratification analysis"""
    st.markdown("#### 🎯 Risk Stratification Analysis")

    stratification_stats = risk_results.get('stratification_stats', {})

    if not stratification_stats:
        st.warning("⚠️ No risk stratification data")
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
            'low': 'Low Risk',
            'medium': 'Medium Risk',
            'high': 'High Risk',
            'critical': 'Critical Risk'
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
                title="Risk Stratification Distribution",
                height=400,
                showlegend=True
            )

            st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # 风险分层详细表格
        st.markdown("**Stratification Details**")

        stratification_data = []
        for level, stats in stratification_stats.items():
            stratification_data.append({
                'Risk Level': name_map.get(level, level),
                'User Count': stats.get('count', 0),
                'Percentage': f"{stats.get('percentage', 0):.1f}%",
                'Target Percentage': f"{stats.get('target_percentage', 0):.1f}%",
                'Average Score': f"{stats.get('average_score', 0):.1f}",
                'Description': stats.get('description', '')
            })

        if stratification_data:
            df_stratification = pd.DataFrame(stratification_data)
            st.dataframe(df_stratification, use_container_width=True)

def _display_attack_type_analysis(risk_results: Dict[str, Any]):
    """Display attack type analysis"""
    st.markdown("#### 🔍 Attack Type Analysis")

    attack_predictions = risk_results.get('attack_predictions', [])

    if not attack_predictions:
        st.warning("⚠️ No attack type prediction data")
        return

    # 统计攻击类型分布
    attack_counts = {}
    attack_names = {}

    for pred in attack_predictions:
        attack_type = pred.get('attack_type', 'unknown')
        attack_name = pred.get('attack_name', 'Unknown Attack')

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
                    title="Attack Type Distribution",
                    labels={'x': 'Attack Type', 'y': 'Detection Count'},
                    color=values,
                    color_continuous_scale='Reds'
                )

                fig_attack.update_layout(height=400)
                st.plotly_chart(fig_attack, use_container_width=True)
            else:
                st.info("✅ No obvious attack behavior detected")

    with col2:
        # 攻击类型详细信息
        st.markdown("**Attack Type Details**")
        
        attack_data = []
        for attack_type, count in attack_counts.items():
            if attack_type != 'none' and count > 0:
                attack_data.append({
                    'Attack Type': attack_names.get(attack_type, attack_type),
                    'Detection Count': count,
                    'Percentage': f"{count / len(attack_predictions) * 100:.1f}%"
                })

        if attack_data:
            df_attacks = pd.DataFrame(attack_data)
            st.dataframe(df_attacks, use_container_width=True)
        else:
            st.success("✅ No attack behavior detected")

def _display_individual_analysis(risk_results: Dict[str, Any]):
    """Display individual detailed analysis"""
    st.markdown("#### 👤 Individual Detailed Analysis")

    individual_analyses = risk_results.get('individual_analyses', [])

    if not individual_analyses:
        st.warning("⚠️ No individual analysis data")
        return

    # 显示高风险用户
    high_risk_users = [
        analysis for analysis in individual_analyses
        if analysis.get('risk_level') in ['high', 'critical']
    ]

    if high_risk_users:
        st.markdown("**🚨 High Risk User Details**")

        # 限制显示数量
        display_count = min(10, len(high_risk_users))

        for i, analysis in enumerate(high_risk_users[:display_count]):
            with st.expander(f"User {analysis.get('user_id', f'user_{i}')} - {analysis.get('risk_level', 'unknown')} risk"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Risk Score**: {analysis.get('risk_score', 0):.1f}")
                    st.write(f"**Risk Level**: {analysis.get('risk_level', 'unknown')}")
                    st.write(f"**Monitoring Level**: {analysis.get('monitoring_level', 'unknown')}")

                with col2:
                    attack_pred = analysis.get('attack_prediction', {})
                    st.write(f"**Attack Type**: {attack_pred.get('attack_name', 'Unknown')}")
                    st.write(f"**Confidence**: {attack_pred.get('confidence', 0):.2f}")

                # 显示建议措施
                actions = analysis.get('recommended_actions', [])
                if actions:
                    st.write("**Recommended Actions**:")
                    for action in actions:
                        st.write(f"- {action}")

        if len(high_risk_users) > display_count:
            st.info(f"{len(high_risk_users) - display_count} more high-risk users not displayed")
    else:
        st.success("✅ No high-risk users found")

def _display_protection_recommendations(risk_results: Dict[str, Any]):
    """Display protection recommendations"""
    st.markdown("#### 🛡️ Protection Recommendations")

    protection_recommendations = risk_results.get('protection_recommendations', {})

    if not protection_recommendations:
        st.warning("⚠️ No protection recommendation data")
        return

    # 立即行动建议
    immediate_actions = protection_recommendations.get('immediate_actions', [])
    if immediate_actions:
        st.markdown("**🚨 Immediate Actions**")
        for action in immediate_actions:
            st.error(f"🔥 {action}")

    # 监控增强建议
    monitoring_enhancements = protection_recommendations.get('monitoring_enhancements', [])
    if monitoring_enhancements:
        st.markdown("**👁️ Monitoring Enhancements**")
        for enhancement in monitoring_enhancements:
            st.warning(f"⚠️ {enhancement}")

    # 系统改进建议
    system_improvements = protection_recommendations.get('system_improvements', [])
    if system_improvements:
        st.markdown("**🔧 System Improvements**")
        for improvement in system_improvements:
            st.info(f"💡 {improvement}")

    # 如果没有特殊建议
    if not any([immediate_actions, monitoring_enhancements, system_improvements]):
        st.success("✅ Current risk level is manageable, continue maintaining existing security measures")

def display_risk_score_distribution(risk_results: Dict[str, Any]):
    """Display risk score distribution"""
    st.markdown("#### 📊 Risk Score Distribution")
    
    risk_scores = risk_results.get('risk_scores', [])

    if not risk_scores:
        st.warning("⚠️ No risk score data available")
        return

    # 创建风险评分分布直方图
    fig_hist = px.histogram(
        x=risk_scores,
        nbins=20,
        title="Risk Score Distribution",
        labels={'x': 'Risk Score', 'y': 'User Count'},
        color_discrete_sequence=['#1f77b4']
    )

    # 添加风险阈值线
    fig_hist.add_vline(x=40, line_dash="dash", line_color="green", annotation_text="Low Risk Threshold")
    fig_hist.add_vline(x=60, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
    fig_hist.add_vline(x=80, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")

    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

    # 显示统计信息
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Minimum Score", f"{min(risk_scores):.1f}")
    with col2:
        st.metric("Maximum Score", f"{max(risk_scores):.1f}")
    with col3:
        st.metric("Average Score", f"{np.mean(risk_scores):.1f}")
    with col4:
        st.metric("Standard Deviation", f"{np.std(risk_scores):.1f}")
