"""
Intelligent Risk Prediction Page
Individual analysis and attack type inference based on risk scoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入后端模块（使用安全导入）
try:
    from backend.prediction.individual_risk_predictor import IndividualRiskPredictor
    from backend.clustering.cluster_analyzer import ClusterAnalyzer
    PREDICTION_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ 风险预测模块导入失败: {e}")
    st.info("💡 请检查风险预测模块和依赖包是否正确安装")
    PREDICTION_AVAILABLE = False
    IndividualRiskPredictor = None
    ClusterAnalyzer = None

def _check_prerequisites():
    """检查前置条件"""
    if 'engineered_features' not in st.session_state or st.session_state.engineered_features is None:
        st.warning("⚠️ 请先完成特征工程！")
        st.info("💡 请在'🔧 特征工程'页面完成特征生成")
        return False

    if 'clustering_results' not in st.session_state or st.session_state.clustering_results is None:
        st.warning("⚠️ 建议先完成聚类分析以获得更准确的风险评估！")
        st.info("💡 请在'📊 聚类分析'页面完成聚类分析")
        # 不强制要求聚类结果，但会给出提示

    return True


# 删除了所有旧的显示函数，使用新的风险预测显示组件


def _execute_individual_risk_prediction(engineered_data, clustering_results, use_clustering, risk_thresholds):
    """执行个体风险预测"""
    if not PREDICTION_AVAILABLE:
        st.error("❌ 风险预测模块不可用，无法进行预测")
        st.info("💡 请检查以下项目:")
        st.info("1. 确保 backend/prediction 目录存在")
        st.info("2. 确保风险评分模块完整")
        st.info("3. 安装必要的依赖包: pip install scikit-learn pandas numpy")
        return

    try:
        with st.spinner("正在进行智能风险预测..."):
            # 准备数据
            X = engineered_data.copy()

            # 只保留数值特征
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_cols]

            # 处理缺失值和无穷值
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            X = X.astype(float)

            # 创建个体风险预测器
            risk_predictor = IndividualRiskPredictor()

            # 更新风险阈值
            if risk_thresholds:
                risk_predictor.risk_thresholds = risk_thresholds

            st.info(f"✅ 开始分析 {len(X)} 个样本的个体风险")

            # 执行个体风险预测
            clustering_data = clustering_results if use_clustering else None
            risk_results = risk_predictor.predict_individual_risks(
                X,
                clustering_data,
                use_four_class_labels=True
            )

            # 检查预测结果
            if risk_results.get('success', False):
                # 保存结果到session state
                st.session_state.individual_risk_results = risk_results
                st.session_state.risk_stratification = risk_results.get('stratification_stats', {})

                # 显示预测统计
                total_samples = risk_results.get('total_samples', 0)
                processing_time = risk_results.get('processing_time', 0)

                st.success(f"✅ 个体风险预测完成！")
                st.info(f"📊 成功分析 {total_samples} 个样本，耗时 {processing_time:.2f} 秒")

                # 显示动态阈值信息
                if 'dynamic_thresholds' in risk_results:
                    thresholds = risk_results['dynamic_thresholds']
                    st.info(f"🎚️ 动态阈值: 低风险(<{thresholds.get('low', 40):.1f}) | "
                           f"中风险({thresholds.get('low', 40):.1f}-{thresholds.get('medium', 60):.1f}) | "
                           f"高风险({thresholds.get('medium', 60):.1f}-{thresholds.get('high', 80):.1f}) | "
                           f"极高风险(>{thresholds.get('high', 80):.1f})")

                # 显示风险分层统计
                stratification_stats = risk_results.get('stratification_stats', {})
                if stratification_stats:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        low_count = stratification_stats.get('low', {}).get('count', 0)
                        low_pct = stratification_stats.get('low', {}).get('percentage', 0)
                        st.metric("低风险用户", f"{low_count} ({low_pct:.1f}%)")

                    with col2:
                        medium_count = stratification_stats.get('medium', {}).get('count', 0)
                        medium_pct = stratification_stats.get('medium', {}).get('percentage', 0)
                        st.metric("中风险用户", f"{medium_count} ({medium_pct:.1f}%)")

                    with col3:
                        high_count = stratification_stats.get('high', {}).get('count', 0)
                        high_pct = stratification_stats.get('high', {}).get('percentage', 0)
                        st.metric("高风险用户", f"{high_count} ({high_pct:.1f}%)")

                    with col4:
                        critical_count = stratification_stats.get('critical', {}).get('count', 0)
                        critical_pct = stratification_stats.get('critical', {}).get('percentage', 0)
                        st.metric("极高风险用户", f"{critical_count} ({critical_pct:.1f}%)")

                # 显示主要攻击类型
                protection_recommendations = risk_results.get('protection_recommendations', {})
                attack_distribution = protection_recommendations.get('attack_type_distribution', {})

                if attack_distribution:
                    st.markdown("#### 🎯 检测到的主要攻击类型")
                    for attack_type, count in sorted(attack_distribution.items(), key=lambda x: x[1], reverse=True)[:3]:
                        if attack_type != 'none' and count > 0:
                            st.info(f"🔍 {attack_type}: {count} 个案例")

            else:
                error_msg = risk_results.get('error', '未知错误')
                st.error(f"❌ 个体风险预测失败: {error_msg}")

    except Exception as e:
        st.error(f"❌ 个体风险预测执行失败: {str(e)}")
        st.exception(e)


def show():
    """显示智能风险预测页面"""
    st.markdown('<div class="sub-header">🎯 智能风险预测与个体分析</div>', unsafe_allow_html=True)

    # 检查前置条件
    if not _check_prerequisites():
        return

    # 检查风险预测可用性
    if not PREDICTION_AVAILABLE:
        st.error("❌ 风险预测功能不可用")
        st.info("💡 风险预测模块导入失败，请检查:")
        st.info("1. backend/prediction 目录是否存在")
        st.info("2. 风险评分模块是否完整")
        st.info("3. 必要的Python包是否已安装")

        with st.expander("📋 安装指南"):
            st.code("""
# 安装基础依赖
pip install scikit-learn pandas numpy

# 检查模块结构
ls backend/prediction/
ls backend/risk_scoring/
            """)
        return


    # 初始化session state
    if 'individual_risk_results' not in st.session_state:
        st.session_state.individual_risk_results = None
    if 'risk_stratification' not in st.session_state:
        st.session_state.risk_stratification = None

    # 获取特征工程数据和聚类结果
    engineered_data = st.session_state.engineered_features
    clustering_results = st.session_state.get('clustering_results', None)

    st.markdown("### 📊 数据概览")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("记录数", f"{len(engineered_data):,}")

    with col2:
        st.metric("特征数", f"{len(engineered_data.columns)}")

    with col3:
        if clustering_results:
            cluster_count = clustering_results.get('cluster_count', 0)
            st.metric("聚类数量", f"{cluster_count}")
        else:
            st.metric("聚类状态", "未聚类")

    with col4:
        numeric_features = len(engineered_data.select_dtypes(include=['number']).columns)
        st.metric("数值特征", f"{numeric_features}")

    # 风险预测配置区域
    st.markdown("### ⚙️ 智能风险预测配置")

    st.markdown("""
    **风险预测特点：**
    - **个体分析**: 为每个用户计算详细的风险评分和攻击类型推断
    - **风险分层**: 将用户分为低、中、高、极高四个风险等级
    - **攻击类型推断**: 识别账户接管、身份盗用、批量欺诈、测试性攻击等类型
    - **防护建议**: 为不同风险等级提供针对性的防护措施
    """)

    # 预测配置
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📋 预测设置")

        # 是否使用聚类结果
        use_clustering = st.checkbox(
            "使用聚类结果增强预测",
            value=clustering_results is not None,
            help="基于聚类结果可以提供更准确的风险评估"
        )

        # 风险分层模式
        stratification_mode = st.selectbox(
            "风险分层模式",
            ["标准四分层", "自定义分层"],
            help="选择风险分层方式"
        )

    with col2:
        st.markdown("#### ⚙️ 风险参数")

        # 风险阈值配置
        if stratification_mode == "自定义分层":
            st.markdown("**自定义风险阈值**")
            low_threshold = st.slider("低风险阈值", 0, 50, 40, help="0-此值为低风险")
            medium_threshold = st.slider("中风险阈值", low_threshold, 80, 60, help="低风险阈值-此值为中风险")
            high_threshold = st.slider("高风险阈值", medium_threshold, 100, 80, help="中风险阈值-此值为高风险")

            risk_thresholds = {
                'low': low_threshold,
                'medium': medium_threshold,
                'high': high_threshold,
                'critical': 100
            }
        else:
            # 使用标准阈值
            risk_thresholds = {
                'low': 40,
                'medium': 60,
                'high': 80,
                'critical': 100
            }
            st.info("使用标准风险阈值：低(0-40)、中(41-60)、高(61-80)、极高(81-100)")

        # 显示预期分布
        st.markdown("**预期风险分布**")
        st.text("低风险: ~60%")
        st.text("中风险: ~25%")
        st.text("高风险: ~12%")
        st.text("极高风险: ~3%")

    # 执行风险预测
    st.markdown("---")

    # 预测按钮
    if st.button("🎯 执行智能风险预测", type="primary", help="基于风险评分进行个体分析和攻击类型推断"):
        _execute_individual_risk_prediction(engineered_data, clustering_results, use_clustering, risk_thresholds)

    # 显示风险预测结果
    if st.session_state.individual_risk_results is not None:
        # 导入结果显示组件
        try:
            from frontend.components.risk_result_display import (
                display_risk_prediction_results,
                display_risk_score_distribution
            )

            risk_results = st.session_state.individual_risk_results

            # 显示主要结果
            display_risk_prediction_results(risk_results)

            # 显示风险评分分布
            with st.expander("📊 风险评分分布分析", expanded=False):
                display_risk_score_distribution(risk_results)

        except ImportError as e:
            st.error(f"❌ 结果显示组件导入失败: {e}")
            # 降级显示基础结果
            _display_basic_risk_results(st.session_state.individual_risk_results)

    else:
        # 显示智能风险预测说明
        st.markdown("### 📝 智能风险预测说明")

        st.markdown("""
        **智能风险预测特点：**

        🎯 **个体风险分析**
        - 为每个用户计算详细的风险评分（0-100分）
        - 基于多维度特征进行综合评估
        - 提供个性化的风险分析报告

        🏷️ **四层风险分层**
        - **低风险** (0-40分): 正常用户，基础监控
        - **中风险** (41-60分): 需要关注，增强监控
        - **高风险** (61-80分): 重点关注，严密监控
        - **极高风险** (81-100分): 立即处理，实时监控

        🔍 **攻击类型推断**
        - **账户接管攻击**: 攻击者获取用户账户控制权
        - **身份盗用攻击**: 使用他人身份信息进行欺诈
        - **批量欺诈攻击**: 大规模自动化欺诈行为
        - **测试性攻击**: 小额测试以验证支付方式

        🛡️ **防护建议**
        - 针对不同风险等级提供具体的防护措施
        - 基于攻击类型推荐相应的安全策略
        - 提供系统改进和监控增强建议

        📊 **数据驱动**
        - 基于聚类分析增强预测准确性
        - 使用无监督学习识别异常模式
        - 结合业务规则和统计分析
        """)

        # 下一步指引
        st.markdown("---")
        st.markdown("### 🚀 开始使用")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("""
            💡 **使用步骤**:
            1. 确保已完成特征工程
            2. 建议先完成聚类分析（可选）
            3. 配置风险预测参数
            4. 点击"执行智能风险预测"按钮
            5. 查看详细的个体风险分析结果
            """)


def _display_basic_risk_results(risk_results: Dict[str, Any]):
    """基础风险结果显示（备用函数）"""
    st.markdown("### 📈 风险预测结果（基础显示）")

    if not risk_results or not risk_results.get('success', False):
        st.error("❌ 风险预测失败")
        return

    # 显示基础统计
    total_samples = risk_results.get('total_samples', 0)
    processing_time = risk_results.get('processing_time', 0)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("分析样本数", f"{total_samples:,}")
    with col2:
        st.metric("处理时间", f"{processing_time:.2f}秒")

    # 显示风险分层统计
    stratification_stats = risk_results.get('stratification_stats', {})
    if stratification_stats:
        st.markdown("#### 风险分层统计")
        for level, stats in stratification_stats.items():
            count = stats.get('count', 0)
            percentage = stats.get('percentage', 0)
            st.write(f"**{level}风险**: {count} 用户 ({percentage:.1f}%)")

    # 显示攻击类型分布
    protection_recommendations = risk_results.get('protection_recommendations', {})
    attack_distribution = protection_recommendations.get('attack_type_distribution', {})

    if attack_distribution:
        st.markdown("#### 攻击类型分布")
        for attack_type, count in attack_distribution.items():
            if attack_type != 'none' and count > 0:
                st.write(f"**{attack_type}**: {count} 个案例")

    st.success("✅ 基础风险预测结果显示完成")