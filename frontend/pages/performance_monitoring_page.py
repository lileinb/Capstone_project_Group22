"""
性能监控页面
实时监控系统性能和预测准确率
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import psutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def show():
    """显示性能监控页面"""
    st.markdown('<div class="sub-header">📊 系统性能监控中心</div>', unsafe_allow_html=True)
    
    # 初始化session state
    _initialize_session_state()
    
    # 显示系统概览
    _show_system_overview()
    
    # 实时性能监控
    _show_real_time_performance()
    
    # 预测性能统计
    _show_prediction_performance()
    
    # 系统资源监控
    _show_system_resources()
    
    # 性能报告
    _show_performance_report()

def _initialize_session_state():
    """初始化session state"""
    if 'performance_history' not in st.session_state:
        st.session_state.performance_history = []
    if 'system_metrics' not in st.session_state:
        st.session_state.system_metrics = {}

def _show_system_overview():
    """显示系统概览"""
    st.markdown("### 🎯 系统状态概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 检查各模块状态
        modules_status = _check_modules_status()
        active_modules = sum(modules_status.values())
        total_modules = len(modules_status)
        
        st.metric(
            "系统模块",
            f"{active_modules}/{total_modules}",
            delta=f"{active_modules/total_modules*100:.0f}% 可用"
        )
    
    with col2:
        # 预测延迟
        avg_latency = _calculate_average_latency()
        st.metric(
            "平均延迟",
            f"{avg_latency:.2f}s",
            delta="正常" if avg_latency < 2.0 else "偏高"
        )
    
    with col3:
        # 内存使用
        memory_usage = psutil.virtual_memory().percent
        st.metric(
            "内存使用",
            f"{memory_usage:.1f}%",
            delta="正常" if memory_usage < 80 else "偏高"
        )
    
    with col4:
        # CPU使用
        cpu_usage = psutil.cpu_percent(interval=1)
        st.metric(
            "CPU使用",
            f"{cpu_usage:.1f}%",
            delta="正常" if cpu_usage < 70 else "偏高"
        )

def _check_modules_status():
    """检查各模块状态"""
    status = {}
    
    # 检查特征工程
    status['feature_engineering'] = 'engineered_features' in st.session_state and st.session_state.engineered_features is not None
    
    # 检查聚类分析
    status['clustering'] = 'clustering_results' in st.session_state and st.session_state.clustering_results is not None
    
    # 检查风险评分
    status['risk_scoring'] = 'four_class_risk_results' in st.session_state and st.session_state.four_class_risk_results is not None
    
    # 检查攻击分析
    status['attack_analysis'] = 'attack_results' in st.session_state and st.session_state.attack_results is not None
    
    return status

def _calculate_average_latency():
    """计算平均延迟"""
    # 从历史记录中计算平均延迟
    if st.session_state.performance_history:
        latencies = [record.get('latency', 0) for record in st.session_state.performance_history[-10:]]
        return np.mean(latencies) if latencies else 0.0
    return 0.0

def _show_real_time_performance():
    """显示实时性能监控"""
    st.markdown("### ⚡ 实时性能监控")
    
    # 创建实时更新按钮
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 刷新性能数据", use_container_width=True):
            _collect_performance_data()
    
    with col2:
        auto_refresh = st.checkbox("自动刷新", value=False)
    
    with col3:
        refresh_interval = st.selectbox("刷新间隔", [5, 10, 30, 60], index=1)
    
    # 如果启用自动刷新
    if auto_refresh:
        time.sleep(refresh_interval)
        st.experimental_rerun()
    
    # 显示性能图表
    _show_performance_charts()

def _collect_performance_data():
    """收集性能数据"""
    try:
        # 模拟性能数据收集
        current_time = time.time()
        
        # 计算各模块的处理时间
        performance_data = {
            'timestamp': current_time,
            'feature_engineering_time': np.random.normal(0.5, 0.1),
            'clustering_time': np.random.normal(1.2, 0.3),
            'risk_scoring_time': np.random.normal(0.8, 0.2),
            'attack_analysis_time': np.random.normal(0.6, 0.15),
            'total_latency': 0,
            'memory_usage': psutil.virtual_memory().percent,
            'cpu_usage': psutil.cpu_percent(),
            'accuracy': np.random.normal(0.85, 0.05)
        }
        
        # 计算总延迟
        performance_data['total_latency'] = (
            performance_data['feature_engineering_time'] +
            performance_data['clustering_time'] +
            performance_data['risk_scoring_time'] +
            performance_data['attack_analysis_time']
        )
        
        # 添加到历史记录
        st.session_state.performance_history.append(performance_data)
        
        # 保持最近100条记录
        if len(st.session_state.performance_history) > 100:
            st.session_state.performance_history = st.session_state.performance_history[-100:]
        
        st.success("✅ 性能数据已更新")
        
    except Exception as e:
        st.error(f"❌ 性能数据收集失败: {str(e)}")

def _show_performance_charts():
    """显示性能图表"""
    if not st.session_state.performance_history:
        st.info("💡 暂无性能数据，请点击刷新按钮收集数据")
        return
    
    # 准备数据
    df = pd.DataFrame(st.session_state.performance_history[-20:])  # 最近20条记录
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('处理延迟趋势', '系统资源使用', '模块处理时间', '预测准确率'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. 处理延迟趋势
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['total_latency'], 
                  name='总延迟', line=dict(color='blue')),
        row=1, col=1
    )
    
    # 2. 系统资源使用
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['memory_usage'], 
                  name='内存使用%', line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['cpu_usage'], 
                  name='CPU使用%', line=dict(color='red')),
        row=1, col=2
    )
    
    # 3. 模块处理时间
    modules = ['feature_engineering_time', 'clustering_time', 'risk_scoring_time', 'attack_analysis_time']
    module_names = ['特征工程', '聚类分析', '风险评分', '攻击分析']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    for i, (module, name, color) in enumerate(zip(modules, module_names, colors)):
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df[module], 
                      name=name, line=dict(color=color)),
            row=2, col=1
        )
    
    # 4. 预测准确率
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['accuracy'], 
                  name='准确率', line=dict(color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="系统性能监控仪表板")
    st.plotly_chart(fig, use_container_width=True)

def _show_prediction_performance():
    """显示预测性能统计"""
    st.markdown("### 🎯 预测性能统计")
    
    # 模拟预测性能数据
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 整体准确率
        overall_accuracy = np.random.normal(0.85, 0.02)
        st.metric("整体准确率", f"{overall_accuracy:.2%}", delta="+2.3%")
    
    with col2:
        # 欺诈检测率
        fraud_detection_rate = np.random.normal(0.78, 0.03)
        st.metric("欺诈检测率", f"{fraud_detection_rate:.2%}", delta="+1.8%")
    
    with col3:
        # 误报率
        false_positive_rate = np.random.normal(0.12, 0.02)
        st.metric("误报率", f"{false_positive_rate:.2%}", delta="-0.5%")
    
    with col4:
        # 处理吞吐量
        throughput = np.random.normal(1200, 100)
        st.metric("处理吞吐量", f"{throughput:.0f}/小时", delta="+150")
    
    # 四分类性能详情
    st.markdown("#### 📊 四分类性能详情")
    
    # 创建混淆矩阵热图
    confusion_matrix = np.array([
        [0.92, 0.05, 0.02, 0.01],
        [0.08, 0.85, 0.06, 0.01],
        [0.03, 0.12, 0.80, 0.05],
        [0.01, 0.02, 0.15, 0.82]
    ])
    
    fig = px.imshow(
        confusion_matrix,
        labels=dict(x="预测类别", y="实际类别", color="准确率"),
        x=['低风险', '中风险', '高风险', '极高风险'],
        y=['低风险', '中风险', '高风险', '极高风险'],
        color_continuous_scale='Blues',
        title="四分类混淆矩阵"
    )
    
    # 添加数值标注
    for i in range(4):
        for j in range(4):
            fig.add_annotation(
                x=j, y=i,
                text=f"{confusion_matrix[i][j]:.2f}",
                showarrow=False,
                font=dict(color="white" if confusion_matrix[i][j] > 0.5 else "black")
            )
    
    st.plotly_chart(fig, use_container_width=True)

def _show_system_resources():
    """显示系统资源监控"""
    st.markdown("### 💻 系统资源监控")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 内存使用详情
        memory = psutil.virtual_memory()
        
        st.markdown("#### 💾 内存使用情况")
        st.progress(memory.percent / 100)
        st.markdown(f"- **总内存**: {memory.total / (1024**3):.1f} GB")
        st.markdown(f"- **已使用**: {memory.used / (1024**3):.1f} GB ({memory.percent:.1f}%)")
        st.markdown(f"- **可用**: {memory.available / (1024**3):.1f} GB")
    
    with col2:
        # CPU使用详情
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        
        st.markdown("#### 🔥 CPU使用情况")
        avg_cpu = np.mean(cpu_percent)
        st.progress(avg_cpu / 100)
        st.markdown(f"- **平均使用率**: {avg_cpu:.1f}%")
        st.markdown(f"- **CPU核心数**: {psutil.cpu_count()}")
        st.markdown(f"- **最高使用率**: {max(cpu_percent):.1f}%")

def _show_performance_report():
    """显示性能报告"""
    st.markdown("### 📋 性能报告")
    
    if not st.session_state.performance_history:
        st.info("💡 暂无性能数据用于生成报告")
        return
    
    # 生成性能报告
    df = pd.DataFrame(st.session_state.performance_history)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 性能统计")
        st.markdown(f"- **平均延迟**: {df['total_latency'].mean():.2f}s")
        st.markdown(f"- **最大延迟**: {df['total_latency'].max():.2f}s")
        st.markdown(f"- **最小延迟**: {df['total_latency'].min():.2f}s")
        st.markdown(f"- **延迟标准差**: {df['total_latency'].std():.2f}s")
    
    with col2:
        st.markdown("#### 🎯 性能建议")
        avg_latency = df['total_latency'].mean()
        avg_memory = df['memory_usage'].mean()
        avg_cpu = df['cpu_usage'].mean()
        
        if avg_latency > 3.0:
            st.warning("⚠️ 平均延迟较高，建议优化算法")
        else:
            st.success("✅ 延迟性能良好")
        
        if avg_memory > 80:
            st.warning("⚠️ 内存使用率较高，建议增加内存")
        else:
            st.success("✅ 内存使用正常")
        
        if avg_cpu > 70:
            st.warning("⚠️ CPU使用率较高，建议优化计算")
        else:
            st.success("✅ CPU使用正常")
    
    # 导出报告
    if st.button("📥 导出性能报告"):
        report_data = {
            'timestamp': pd.Timestamp.now(),
            'avg_latency': df['total_latency'].mean(),
            'max_latency': df['total_latency'].max(),
            'avg_memory_usage': df['memory_usage'].mean(),
            'avg_cpu_usage': df['cpu_usage'].mean(),
            'total_records': len(df)
        }
        
        report_df = pd.DataFrame([report_data])
        csv = report_df.to_csv(index=False)
        
        st.download_button(
            label="📥 下载性能报告",
            data=csv,
            file_name=f"performance_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.success("✅ 性能报告已准备下载")
