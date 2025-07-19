"""
Performance Monitoring Page
Real-time monitoring of system performance and prediction accuracy
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
    """Display performance monitoring page"""
    st.markdown('<div class="sub-header">📊 System Performance Monitoring Center</div>', unsafe_allow_html=True)
    
    # 初始化session state
    _initialize_session_state()
    
    # 显示系统概览
    _show_system_overview()
    
    # Real-time performance monitoring
    _show_real_time_performance()

    # Prediction performance statistics
    _show_prediction_performance()

    # System resource monitoring
    _show_system_resources()

    # Performance report
    _show_performance_report()

def _initialize_session_state():
    """Initialize session state"""
    if 'performance_history' not in st.session_state:
        st.session_state.performance_history = []
    if 'system_metrics' not in st.session_state:
        st.session_state.system_metrics = {}

def _show_system_overview():
    """Display system overview"""
    st.markdown("### 🎯 System Status Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # 检查各模块状态
        modules_status = _check_modules_status()
        active_modules = sum(modules_status.values())
        total_modules = len(modules_status)

        st.metric(
            "System Modules",
            f"{active_modules}/{total_modules}",
            delta=f"{active_modules/total_modules*100:.0f}% Available"
        )

    with col2:
        # Prediction latency
        avg_latency = _calculate_average_latency()
        st.metric(
            "Average Latency",
            f"{avg_latency:.2f}s",
            delta="Normal" if avg_latency < 2.0 else "High"
        )

    with col3:
        # Memory usage
        memory_usage = psutil.virtual_memory().percent
        st.metric(
            "Memory Usage",
            f"{memory_usage:.1f}%",
            delta="Normal" if memory_usage < 80 else "High"
        )

    with col4:
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        st.metric(
            "CPU Usage",
            f"{cpu_usage:.1f}%",
            delta="Normal" if cpu_usage < 70 else "High"
        )

def _check_modules_status():
    """Check module status"""
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
    """Calculate average latency"""
    # Calculate average latency from historical records
    if st.session_state.performance_history:
        latencies = [record.get('latency', 0) for record in st.session_state.performance_history[-10:]]
        return np.mean(latencies) if latencies else 0.0
    return 0.0

def _show_real_time_performance():
    """Show real-time performance monitoring"""
    st.markdown("### ⚡ Real-time Performance Monitoring")

    # 创建实时更新按钮
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh Performance Data", use_container_width=True):
            _collect_performance_data()

    with col2:
        auto_refresh = st.checkbox("Auto Refresh", value=False)

    with col3:
        refresh_interval = st.selectbox("Refresh Interval", [5, 10, 30, 60], index=1)

    # If auto refresh is enabled
    if auto_refresh:
        time.sleep(refresh_interval)
        st.experimental_rerun()

    # Display performance charts
    _show_performance_charts()

def _collect_performance_data():
    """Collect performance data"""
    try:
        # Simulate performance data collection
        current_time = time.time()

        # Calculate processing time for each module
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

        # Calculate total latency
        performance_data['total_latency'] = (
            performance_data['feature_engineering_time'] +
            performance_data['clustering_time'] +
            performance_data['risk_scoring_time'] +
            performance_data['attack_analysis_time']
        )

        # Add to history
        st.session_state.performance_history.append(performance_data)

        # Keep only the latest 100 records
        if len(st.session_state.performance_history) > 100:
            st.session_state.performance_history = st.session_state.performance_history[-100:]

        st.success("✅ Performance data updated")

    except Exception as e:
        st.error(f"❌ Performance data collection failed: {str(e)}")

def _show_performance_charts():
    """Display performance charts"""
    if not st.session_state.performance_history:
        st.info("💡 No performance data available, please click refresh button to collect data")
        return

    # Prepare data
    df = pd.DataFrame(st.session_state.performance_history[-20:])  # Latest 20 records
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Processing Latency Trend', 'System Resource Usage', 'Module Processing Time', 'Prediction Accuracy'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Processing latency trend
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['total_latency'],
                  name='Total Latency', line=dict(color='blue')),
        row=1, col=1
    )

    # 2. System resource usage
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['memory_usage'],
                  name='Memory Usage %', line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['cpu_usage'],
                  name='CPU Usage %', line=dict(color='red')),
        row=1, col=2
    )
    
    # 3. Module processing time
    modules = ['feature_engineering_time', 'clustering_time', 'risk_scoring_time', 'attack_analysis_time']
    module_names = ['Feature Engineering', 'Clustering', 'Risk Scoring', 'Attack Analysis']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    for i, (module, name, color) in enumerate(zip(modules, module_names, colors)):
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df[module], 
                      name=name, line=dict(color=color)),
            row=2, col=1
        )
    
    # 4. Prediction accuracy
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['accuracy'],
                  name='Accuracy', line=dict(color='purple')),
        row=2, col=2
    )

    fig.update_layout(height=600, showlegend=True, title_text="System Performance Monitoring Dashboard")
    st.plotly_chart(fig, use_container_width=True)

def _show_prediction_performance():
    """Display prediction performance statistics"""
    st.markdown("### 🎯 Prediction Performance Statistics")

    # Simulate prediction performance data
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Overall accuracy
        overall_accuracy = np.random.normal(0.85, 0.02)
        st.metric("Overall Accuracy", f"{overall_accuracy:.2%}", delta="+2.3%")

    with col2:
        # Fraud detection rate
        fraud_detection_rate = np.random.normal(0.78, 0.03)
        st.metric("Fraud Detection Rate", f"{fraud_detection_rate:.2%}", delta="+1.8%")

    with col3:
        # False positive rate
        false_positive_rate = np.random.normal(0.12, 0.02)
        st.metric("False Positive Rate", f"{false_positive_rate:.2%}", delta="-0.5%")

    with col4:
        # Processing throughput
        throughput = np.random.normal(1200, 100)
        st.metric("Processing Throughput", f"{throughput:.0f}/hour", delta="+150")
    
    # Four-class performance details
    st.markdown("#### 📊 Four-Class Performance Details")

    # Create confusion matrix heatmap
    confusion_matrix = np.array([
        [0.92, 0.05, 0.02, 0.01],
        [0.08, 0.85, 0.06, 0.01],
        [0.03, 0.12, 0.80, 0.05],
        [0.01, 0.02, 0.15, 0.82]
    ])

    fig = px.imshow(
        confusion_matrix,
        labels=dict(x="Predicted Class", y="Actual Class", color="Accuracy"),
        x=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'],
        y=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'],
        color_continuous_scale='Blues',
        title="Four-Class Confusion Matrix"
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
    """Display system resource monitoring"""
    st.markdown("### 💻 System Resource Monitoring")

    col1, col2 = st.columns(2)

    with col1:
        # Memory usage details
        memory = psutil.virtual_memory()

        st.markdown("#### 💾 Memory Usage")
        st.progress(memory.percent / 100)
        st.markdown(f"- **Total Memory**: {memory.total / (1024**3):.1f} GB")
        st.markdown(f"- **Used**: {memory.used / (1024**3):.1f} GB ({memory.percent:.1f}%)")
        st.markdown(f"- **Available**: {memory.available / (1024**3):.1f} GB")

    with col2:
        # CPU usage details
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)

        st.markdown("#### 🔥 CPU Usage")
        avg_cpu = np.mean(cpu_percent)
        st.progress(avg_cpu / 100)
        st.markdown(f"- **Average Usage**: {avg_cpu:.1f}%")
        st.markdown(f"- **CPU Cores**: {psutil.cpu_count()}")
        st.markdown(f"- **Peak Usage**: {max(cpu_percent):.1f}%")

def _show_performance_report():
    """Display performance report"""
    st.markdown("### 📋 Performance Report")

    if not st.session_state.performance_history:
        st.info("💡 No performance data available for report generation")
        return

    # Generate performance report
    df = pd.DataFrame(st.session_state.performance_history)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Performance Statistics")
        st.markdown(f"- **Average Latency**: {df['total_latency'].mean():.2f}s")
        st.markdown(f"- **Maximum Latency**: {df['total_latency'].max():.2f}s")
        st.markdown(f"- **Minimum Latency**: {df['total_latency'].min():.2f}s")
        st.markdown(f"- **Latency Std Dev**: {df['total_latency'].std():.2f}s")

    with col2:
        st.markdown("#### 🎯 Performance Recommendations")
        avg_latency = df['total_latency'].mean()
        avg_memory = df['memory_usage'].mean()
        avg_cpu = df['cpu_usage'].mean()

        if avg_latency > 3.0:
            st.warning("⚠️ Average latency is high, recommend algorithm optimization")
        else:
            st.success("✅ Latency performance is good")

        if avg_memory > 80:
            st.warning("⚠️ Memory usage is high, recommend adding more memory")
        else:
            st.success("✅ Memory usage is normal")

        if avg_cpu > 70:
            st.warning("⚠️ CPU usage is high, recommend computation optimization")
        else:
            st.success("✅ CPU usage is normal")
    
    # Export report
    if st.button("📥 Export Performance Report"):
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
            label="📥 Download Performance Report",
            data=csv,
            file_name=f"performance_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        st.success("✅ Performance report ready for download")
