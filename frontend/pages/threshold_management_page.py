"""
动态阈值管理页面
实时监控和调整四分类风险阈值
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入后端模块
from backend.risk_scoring.dynamic_threshold_manager import DynamicThresholdManager

def show():
    """显示动态阈值管理页面"""
    st.markdown('<div class="sub-header">🎛️ 动态阈值管理中心</div>', unsafe_allow_html=True)
    
    # 初始化session state
    _initialize_session_state()
    
    # 检查前置条件
    if not _check_prerequisites():
        return
    
    # 显示系统说明
    _show_system_description()
    
    # 显示当前阈值状态
    _show_current_threshold_status()
    
    # 阈值优化控制面板
    _show_threshold_optimization_panel()
    
    # 实时分布监控
    _show_real_time_distribution_monitoring()
    
    # 阈值调整历史
    _show_threshold_adjustment_history()

def _initialize_session_state():
    """初始化session state"""
    if 'threshold_manager' not in st.session_state:
        st.session_state.threshold_manager = DynamicThresholdManager()
    if 'threshold_history' not in st.session_state:
        st.session_state.threshold_history = []
    if 'current_thresholds' not in st.session_state:
        st.session_state.current_thresholds = None

def _check_prerequisites():
    """检查前置条件"""
    if 'four_class_risk_results' not in st.session_state or st.session_state.four_class_risk_results is None:
        st.warning("⚠️ 请先完成四分类风险评分！")
        st.info("💡 请在'🎯 风险评分'页面完成四分类风险评分")
        return False
    return True

def _show_system_description():
    """显示系统说明"""
    with st.expander("📖 动态阈值管理系统说明", expanded=False):
        st.markdown("""
        ### 🎯 系统功能
        - **实时监控**: 监控当前风险分布和阈值效果
        - **智能优化**: 基于目标分布自动优化阈值
        - **手动调整**: 支持手动微调阈值参数
        - **历史追踪**: 记录阈值调整历史和效果

        ### 📊 目标分布
        - 🟢 **低风险**: 60% (正常交易)
        - 🟡 **中风险**: 25% (需要监控)
        - 🟠 **高风险**: 12% (需要关注)
        - 🔴 **极高风险**: 3% (需要处理)

        ### 🔧 优化策略
        1. 基于当前分布计算偏差
        2. 使用迭代算法优化阈值
        3. 验证新阈值的分布效果
        4. 应用最优阈值配置
        """)

def _show_current_threshold_status():
    """显示当前阈值状态"""
    st.markdown("### 📊 当前阈值状态")
    
    risk_results = st.session_state.four_class_risk_results
    current_thresholds = risk_results.get('thresholds', {})
    distribution = risk_results.get('distribution', {})
    
    # 阈值信息
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 当前阈值设置")
        threshold_type = risk_results.get('threshold_type', 'unknown')
        st.info(f"**阈值类型**: {threshold_type}")
        
        if current_thresholds:
            st.markdown(f"- 🟢 **低风险**: 0 - {current_thresholds.get('low', 40):.1f}")
            st.markdown(f"- 🟡 **中风险**: {current_thresholds.get('low', 40):.1f} - {current_thresholds.get('medium', 60):.1f}")
            st.markdown(f"- 🟠 **高风险**: {current_thresholds.get('medium', 60):.1f} - {current_thresholds.get('high', 80):.1f}")
            st.markdown(f"- 🔴 **极高风险**: {current_thresholds.get('high', 80):.1f} - 100")
    
    with col2:
        st.markdown("#### 📈 实际分布情况")
        if distribution:
            target_dist = {'low': 60, 'medium': 25, 'high': 12, 'critical': 3}
            
            for level, data in distribution.items():
                actual_pct = data['percentage']
                target_pct = target_dist.get(level, 0)
                deviation = actual_pct - target_pct
                
                icon = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}.get(level, '⚪')
                
                if abs(deviation) <= 5:
                    status = "✅"
                elif abs(deviation) <= 10:
                    status = "⚠️"
                else:
                    status = "❌"
                
                st.markdown(f"- {icon} **{level.title()}**: {actual_pct:.1f}% (目标: {target_pct}%) {status}")

def _show_threshold_optimization_panel():
    """显示阈值优化控制面板"""
    st.markdown("### 🎛️ 阈值优化控制面板")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 自动优化阈值", type="primary", use_container_width=True):
            _execute_threshold_optimization()
    
    with col2:
        if st.button("📊 分析当前分布", use_container_width=True):
            _analyze_current_distribution()
    
    with col3:
        if st.button("🎯 重置为默认阈值", use_container_width=True):
            _reset_to_default_thresholds()

def _execute_threshold_optimization():
    """执行阈值优化"""
    try:
        with st.spinner("正在优化阈值..."):
            risk_results = st.session_state.four_class_risk_results
            detailed_results = risk_results.get('detailed_results', [])
            
            if not detailed_results:
                st.error("❌ 没有可用的风险评分数据")
                return
            
            # 提取风险评分
            risk_scores = [r['risk_score'] for r in detailed_results]
            
            # 使用动态阈值管理器优化
            threshold_manager = st.session_state.threshold_manager
            optimized_thresholds = threshold_manager.optimize_thresholds_iteratively(risk_scores)
            
            # 分析优化效果
            analysis = threshold_manager.analyze_distribution(risk_scores, optimized_thresholds)
            
            # 更新session state
            st.session_state.current_thresholds = optimized_thresholds
            
            # 记录历史
            import datetime
            history_entry = {
                'timestamp': datetime.datetime.now(),
                'action': 'auto_optimization',
                'thresholds': optimized_thresholds.copy(),
                'total_deviation': analysis['total_deviation'],
                'is_reasonable': analysis['is_reasonable']
            }
            st.session_state.threshold_history.append(history_entry)
            
            # 显示结果
            if analysis['is_reasonable']:
                st.success(f"✅ 阈值优化成功！分布偏差: {analysis['total_deviation']:.3f}")
            else:
                st.warning(f"⚠️ 阈值已优化，但分布仍需调整。偏差: {analysis['total_deviation']:.3f}")
            
            # 显示新阈值
            st.info("**优化后的阈值**:")
            for level, threshold in optimized_thresholds.items():
                if level != 'critical':
                    st.write(f"- {level.title()}: {threshold:.1f}")
            
    except Exception as e:
        st.error(f"❌ 阈值优化失败: {str(e)}")

def _analyze_current_distribution():
    """分析当前分布"""
    try:
        risk_results = st.session_state.four_class_risk_results
        distribution = risk_results.get('distribution', {})
        
        if not distribution:
            st.error("❌ 没有可用的分布数据")
            return
        
        # 计算分布偏差
        target_dist = {'low': 60, 'medium': 25, 'high': 12, 'critical': 3}
        total_deviation = 0
        
        st.markdown("#### 📊 分布偏差分析")
        
        for level, data in distribution.items():
            actual_pct = data['percentage']
            target_pct = target_dist.get(level, 0)
            deviation = actual_pct - target_pct
            total_deviation += abs(deviation)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"{level.title()}", f"{actual_pct:.1f}%")
            with col2:
                st.metric("目标", f"{target_pct}%")
            with col3:
                st.metric("偏差", f"{deviation:+.1f}%")
            with col4:
                if abs(deviation) <= 3:
                    st.success("✅ 良好")
                elif abs(deviation) <= 8:
                    st.warning("⚠️ 一般")
                else:
                    st.error("❌ 需要调整")
        
        # 总体评估
        st.markdown("---")
        if total_deviation <= 10:
            st.success(f"✅ **总体评估**: 分布良好 (总偏差: {total_deviation:.1f}%)")
        elif total_deviation <= 20:
            st.warning(f"⚠️ **总体评估**: 分布一般 (总偏差: {total_deviation:.1f}%)")
        else:
            st.error(f"❌ **总体评估**: 分布需要优化 (总偏差: {total_deviation:.1f}%)")
            
    except Exception as e:
        st.error(f"❌ 分布分析失败: {str(e)}")

def _reset_to_default_thresholds():
    """重置为默认阈值"""
    default_thresholds = {
        'low': 40,
        'medium': 60,
        'high': 80,
        'critical': 100
    }
    
    st.session_state.current_thresholds = default_thresholds
    
    # 记录历史
    import datetime
    history_entry = {
        'timestamp': datetime.datetime.now(),
        'action': 'reset_to_default',
        'thresholds': default_thresholds.copy(),
        'total_deviation': None,
        'is_reasonable': None
    }
    st.session_state.threshold_history.append(history_entry)
    
    st.success("✅ 已重置为默认阈值")
    st.info("**默认阈值**: 低风险: 40, 中风险: 60, 高风险: 80")

def _show_real_time_distribution_monitoring():
    """显示实时分布监控"""
    st.markdown("### 📈 实时分布监控")
    
    risk_results = st.session_state.four_class_risk_results
    distribution = risk_results.get('distribution', {})
    
    if not distribution:
        st.warning("⚠️ 没有可用的分布数据")
        return
    
    # 创建分布对比图
    col1, col2 = st.columns(2)
    
    with col1:
        # 当前分布 vs 目标分布
        levels = ['low', 'medium', 'high', 'critical']
        actual_values = [distribution.get(level, {}).get('percentage', 0) for level in levels]
        target_values = [60, 25, 12, 3]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='实际分布',
            x=levels,
            y=actual_values,
            marker_color=['#22c55e', '#f59e0b', '#f97316', '#ef4444']
        ))
        fig.add_trace(go.Bar(
            name='目标分布',
            x=levels,
            y=target_values,
            marker_color=['#22c55e', '#f59e0b', '#f97316', '#ef4444'],
            opacity=0.5
        ))
        
        fig.update_layout(
            title="分布对比",
            xaxis_title="风险等级",
            yaxis_title="百分比 (%)",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 偏差雷达图
        levels_cn = ['低风险', '中风险', '高风险', '极高风险']
        deviations = [abs(actual_values[i] - target_values[i]) for i in range(4)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=deviations,
            theta=levels_cn,
            fill='toself',
            name='分布偏差',
            marker_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(deviations) + 5]
                )),
            title="分布偏差雷达图"
        )
        st.plotly_chart(fig, use_container_width=True)

def _show_threshold_adjustment_history():
    """显示阈值调整历史"""
    st.markdown("### 📋 阈值调整历史")
    
    if not st.session_state.threshold_history:
        st.info("💡 暂无阈值调整历史")
        return
    
    # 显示历史记录
    history_df = pd.DataFrame(st.session_state.threshold_history)
    
    # 格式化显示
    for i, record in enumerate(reversed(st.session_state.threshold_history[-10:])):  # 显示最近10条
        with st.expander(f"调整记录 {len(st.session_state.threshold_history) - i}: {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**操作类型**: {record['action']}")
                if record['total_deviation'] is not None:
                    st.markdown(f"**分布偏差**: {record['total_deviation']:.3f}")
                if record['is_reasonable'] is not None:
                    status = "✅ 合理" if record['is_reasonable'] else "⚠️ 需要调整"
                    st.markdown(f"**分布状态**: {status}")
            
            with col2:
                st.markdown("**阈值设置**:")
                thresholds = record['thresholds']
                for level, value in thresholds.items():
                    if level != 'critical':
                        st.markdown(f"- {level.title()}: {value:.1f}")
    
    # 清理历史按钮
    if st.button("🗑️ 清理历史记录"):
        st.session_state.threshold_history = []
        st.success("✅ 历史记录已清理")
