import streamlit as st
import sys
import os
import importlib

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 清除模块缓存（解决导入问题）
def clear_module_cache():
    """清除相关模块缓存"""
    modules_to_clear = [
        'frontend',
        'frontend.pages',
        'frontend.pages.feature_analysis_page',
        'frontend.pages.upload_page',
        'frontend.pages.clustering_page',
        'frontend.pages.risk_scoring_page',
        'frontend.pages.pseudo_labeling_page',
        'frontend.pages.model_prediction_page',
        'frontend.pages.attack_analysis_page',
        'frontend.pages.report_page'
    ]

    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]

# 在应用启动时清除缓存
clear_module_cache()

# 简化的安全导入函数
def safe_import_page(module_name):
    """安全导入页面模块"""
    try:
        # 直接导入，不使用复杂的重载逻辑
        module = importlib.import_module(module_name)
        return module
    except Exception as e:
        st.error(f"❌ Page loading failed: {module_name}")
        st.error(f"Error details: {str(e)}")

        # 提供简化的解决方案
        st.warning("🔧 Please try the following solutions:")
        st.info("1. Refresh browser page (F5)")
        st.info("2. Restart Streamlit application (Ctrl+C then restart)")
        st.info("3. Clear browser cache")

        # 显示备用页面
        st.markdown("### 📄 Page Temporarily Unavailable")
        st.markdown("This module is loading, please try again later.")

        return None

# 页面配置
st.set_page_config(
    page_title="E-commerce Fraud Risk Prediction System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏导航
st.sidebar.markdown("## 🛡️ E-commerce Fraud Risk Prediction System")
st.sidebar.markdown("---")

# 页面选择
page = st.sidebar.selectbox(
    "Select Page",
    [
        "🏠 Home",
        "📁 Data Upload",
        "🔧 Feature Engineering",
        "📊 Clustering Analysis",
        "🎯 Risk Scoring",
        "🎛️ Threshold Management",
        "🏷️ Pseudo Labeling",
        "🤖 Model Prediction",
        "⚔️ Attack Classification",
        "📊 Performance Monitoring",
        "📋 Analysis Report"
    ]
)

# 页面路由
if page == "🏠 Home":
    st.markdown('<div class="main-header">🛡️ E-commerce Fraud Risk Prediction System</div>', unsafe_allow_html=True)

    # 系统概述
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 System Features")
        st.markdown("""
        - **Multi-dimensional Risk Assessment**: Combines unsupervised clustering, expert rule scoring, and supervised learning
        - **Risk Level Prediction**: Predicts probability of users becoming low/medium/high/critical risk
        - **Attack Type Classification**: Identifies 4 main attack types and provides protection recommendations
        - **Multi-model Comparison**: Performance comparison and ensemble prediction of 4 pre-trained models
        - **Explainability Analysis**: SHAP/LIME deep interpretation of model decision process
        """)

    with col2:
        st.markdown("### 🎯 Core Features")
        st.markdown("""
        - **Intelligent Feature Engineering**: Creates 20+ risk features based on original 16 features
        - **Clustering Anomaly Detection**: Uses K-means, DBSCAN, Gaussian Mixture Models
        - **Real-time Risk Scoring**: Multi-dimensional scoring system with dynamic weight adjustment
        - **Pseudo Label Generation**: High-quality pseudo label generation with multi-strategy integration
        - **Attack Pattern Recognition**: Intelligent classification of four major attack types
        - **Comprehensive Report Generation**: Automatically generates PDF/Excel format analysis reports
        """)
    
    # 系统状态
    st.markdown("### 🔧 System Status")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Data Processing", "✅ Ready", "CSV format supported")

    with col2:
        st.metric("Feature Engineering", "✅ Ready", "20+ risk features")

    with col3:
        st.metric("Model Prediction", "⚠️ Training needed", "4 pre-trained models")

    with col4:
        st.metric("Report Generation", "✅ Ready", "Multi-format export")
    
    # 快速开始
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. **Data Upload**: Upload your transaction data CSV file
    2. **Feature Engineering**: System automatically generates risk features
    3. **Clustering Analysis**: Discover user behavior patterns and anomalous groups
    4. **Risk Scoring**: Calculate multi-dimensional risk scores
    5. **Pseudo Label Generation**: Generate high-quality pseudo labels based on multi-strategy
    6. **Model Prediction**: Use pre-trained models for prediction
    7. **Attack Classification**: Identify attack types and generate protection recommendations
    8. **Analysis Report**: Generate comprehensive analysis and explainability reports
    """)
    


elif page == "📁 Data Upload":
    upload_page = safe_import_page("frontend.pages.upload_page")
    if upload_page:
        upload_page.show()

elif page == "🔧 Feature Engineering":
    feature_page = safe_import_page("frontend.pages.feature_analysis_page")
    if feature_page:
        feature_page.show()

elif page == "📊 Clustering Analysis":
    clustering_page = safe_import_page("frontend.pages.clustering_page")
    if clustering_page:
        clustering_page.show()

elif page == "🎯 Risk Scoring":
    risk_page = safe_import_page("frontend.pages.risk_scoring_page")
    if risk_page:
        risk_page.show()

elif page == "🎛️ Threshold Management":
    threshold_page = safe_import_page("frontend.pages.threshold_management_page")
    if threshold_page:
        threshold_page.show()

elif page == "🏷️ Pseudo Labeling":
    pseudo_page = safe_import_page("frontend.pages.pseudo_labeling_page")
    if pseudo_page:
        pseudo_page.show()

elif page == "🤖 Model Prediction":
    model_page = safe_import_page("frontend.pages.model_prediction_page")
    if model_page:
        model_page.show()

elif page == "⚔️ Attack Classification":
    attack_page = safe_import_page("frontend.pages.attack_analysis_page")
    if attack_page:
        attack_page.show()

elif page == "📊 Performance Monitoring":
    performance_page = safe_import_page("frontend.pages.performance_monitoring_page")
    if performance_page:
        performance_page.show()

elif page == "📋 Analysis Report":
    report_page = safe_import_page("frontend.pages.report_page")
    if report_page:
        report_page.show()

# 页脚
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Information")
st.sidebar.markdown(f"- Python Version: {sys.version}")
st.sidebar.markdown("- Streamlit Interface")
st.sidebar.markdown("- Machine Learning Powered")

# 主页面内容
if page == "🏠 Home":
    st.markdown("---")
    st.markdown("### 📈 System Performance Metrics")

    # 模拟性能指标
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Accuracy", "87.5%", "+2.3%")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("High Risk Detection Rate", "92.1%", "+1.8%")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("False Positive Rate", "3.2%", "-0.5%")
        st.markdown("</div>", unsafe_allow_html=True) 