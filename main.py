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
        st.error(f"❌ 页面加载失败: {module_name}")
        st.error(f"错误详情: {str(e)}")

        # 提供简化的解决方案
        st.warning("🔧 请尝试以下解决方案:")
        st.info("1. 刷新浏览器页面 (F5)")
        st.info("2. 重启Streamlit应用 (Ctrl+C 然后重新运行)")
        st.info("3. 清除浏览器缓存")

        # 显示备用页面
        st.markdown("### 📄 页面暂时不可用")
        st.markdown("该功能模块正在加载中，请稍后再试。")

        return None

# 页面配置
st.set_page_config(
    page_title="电商欺诈风险预测系统",
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
st.sidebar.markdown("## 🛡️ 电商欺诈风险预测系统")
st.sidebar.markdown("---")

# 页面选择
page = st.sidebar.selectbox(
    "选择页面",
    [
        "🏠 首页",
        "📁 数据上传",
        "🔧 特征工程",
        "📊 聚类分析",
        "🎯 风险评分",
        "🎛️ 阈值管理",
        "🏷️ 伪标签生成",
        "🤖 模型预测",
        "⚔️ 攻击分类",
        "📊 性能监控",
        "📋 分析报告"
    ]
)

# 页面路由
if page == "🏠 首页":
    st.markdown('<div class="main-header">🛡️ 电商欺诈风险预测系统</div>', unsafe_allow_html=True)
    
    # 系统概述
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 系统功能")
        st.markdown("""
        - **多维度风险评估**: 结合无监督聚类、专家规则评分和监督学习
        - **风险等级预测**: 预测用户未来成为低/中/高/极高风险的概率
        - **攻击类型分类**: 识别4种主要攻击类型并提供防护建议
        - **多模型对比**: 4个预训练模型性能对比和集成预测
        - **可解释性分析**: SHAP/LIME深度解释模型决策过程
        """)
    
    with col2:
        st.markdown("### 🎯 核心特性")
        st.markdown("""
        - **智能特征工程**: 基于原始16个特征创建20+个风险特征
        - **聚类异常检测**: 使用K-means、DBSCAN、高斯混合模型
        - **实时风险评分**: 动态权重调整的多维度评分系统
        - **伪标签生成**: 多策略集成的高质量伪标签生成
        - **攻击模式识别**: 四大攻击类型的智能分类
        - **综合报告生成**: 自动生成PDF/Excel格式分析报告
        """)
    
    # 系统状态
    st.markdown("### 🔧 系统状态")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("数据处理", "✅ 就绪", "支持CSV格式")
    
    with col2:
        st.metric("特征工程", "✅ 就绪", "20+风险特征")
    
    with col3:
        st.metric("模型预测", "⚠️ 需训练", "4个预训练模型")
    
    with col4:
        st.metric("报告生成", "✅ 就绪", "多格式导出")
    
    # 快速开始
    st.markdown("### 🚀 快速开始")
    st.markdown("""
    1. **数据上传**: 上传您的交易数据CSV文件
    2. **特征工程**: 系统自动生成风险特征
    3. **聚类分析**: 发现用户行为模式和异常群体
    4. **风险评分**: 计算多维度风险评分
    5. **伪标签生成**: 基于多策略生成高质量伪标签
    6. **模型预测**: 使用预训练模型进行预测
    6. **攻击分类**: 识别攻击类型并生成防护建议
    7. **分析报告**: 生成综合分析和可解释性报告
    """)
    
    # 数据集信息
    st.markdown("### 📁 数据集信息")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**数据集1**: Fraudulent_E-Commerce_Transaction_Data_sampled_50000.csv")
        st.markdown("- 记录数: 50,000条")
        st.markdown("- 特征数: 16个原始特征")
        st.markdown("- 欺诈率: 约5%")
    
    with col2:
        st.markdown("**数据集2**: Fraudulent_E-Commerce_Transaction_Data_2_sampled_50000.csv")
        st.markdown("- 记录数: 23,634条")
        st.markdown("- 特征数: 16个原始特征")
        st.markdown("- 欺诈率: 约5%")

elif page == "📁 数据上传":
    upload_page = safe_import_page("frontend.pages.upload_page")
    if upload_page:
        upload_page.show()

elif page == "🔧 特征工程":
    feature_page = safe_import_page("frontend.pages.feature_analysis_page")
    if feature_page:
        feature_page.show()

elif page == "📊 聚类分析":
    clustering_page = safe_import_page("frontend.pages.clustering_page")
    if clustering_page:
        clustering_page.show()

elif page == "🎯 风险评分":
    risk_page = safe_import_page("frontend.pages.risk_scoring_page")
    if risk_page:
        risk_page.show()

elif page == "🎛️ 阈值管理":
    threshold_page = safe_import_page("frontend.pages.threshold_management_page")
    if threshold_page:
        threshold_page.show()

elif page == "🏷️ 伪标签生成":
    pseudo_page = safe_import_page("frontend.pages.pseudo_labeling_page")
    if pseudo_page:
        pseudo_page.show()

elif page == "🤖 模型预测":
    model_page = safe_import_page("frontend.pages.model_prediction_page")
    if model_page:
        model_page.show()

elif page == "⚔️ 攻击分类":
    attack_page = safe_import_page("frontend.pages.attack_analysis_page")
    if attack_page:
        attack_page.show()

elif page == "📊 性能监控":
    performance_page = safe_import_page("frontend.pages.performance_monitoring_page")
    if performance_page:
        performance_page.show()

elif page == "📋 分析报告":
    report_page = safe_import_page("frontend.pages.report_page")
    if report_page:
        report_page.show()

# 页脚
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 系统信息")
st.sidebar.markdown(f"- Python版本: {sys.version}")
st.sidebar.markdown("- Streamlit界面")
st.sidebar.markdown("- 机器学习驱动")

# 主页面内容
if page == "🏠 首页":
    st.markdown("---")
    st.markdown("### 📈 系统性能指标")
    
    # 模拟性能指标
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("平均准确率", "87.5%", "+2.3%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("高风险识别率", "92.1%", "+1.8%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("误报率", "3.2%", "-0.5%")
        st.markdown("</div>", unsafe_allow_html=True) 