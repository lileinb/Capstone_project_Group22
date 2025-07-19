"""
特征工程页面
负责特征生成、分析和可视化
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
from backend.feature_engineer.risk_features import RiskFeatureEngineer

def show():
    """显示特征工程页面"""
    st.markdown('<div class="sub-header">🔧 特征工程与风险特征生成</div>', unsafe_allow_html=True)
    
    # 检查是否有清理后的数据
    if 'cleaned_data' not in st.session_state or st.session_state.cleaned_data is None:
        st.warning("⚠️ 请先上传并清理数据！")
        st.info("💡 请在'📁 数据上传'页面完成数据准备")
        return
    
    # 初始化session state
    if 'engineered_features' not in st.session_state:
        st.session_state.engineered_features = None
    if 'feature_info' not in st.session_state:
        st.session_state.feature_info = None
    
    # 获取清理后的数据
    cleaned_data = st.session_state.cleaned_data

    # 检查关键字段（使用更新后的列名）
    required_columns = ['customer_id', 'transaction_amount', 'payment_method', 'transaction_hour']
    missing_columns = [col for col in required_columns if col not in cleaned_data.columns]
    if missing_columns:
        st.error(f"❌ 数据缺少以下关键字段，无法执行特征工程: {', '.join(missing_columns)}")
        st.info(f"💡 当前数据列名: {list(cleaned_data.columns)}")
        return
    
    st.markdown("### 📊 原始数据概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("记录数", f"{len(cleaned_data):,}")
    
    with col2:
        st.metric("原始特征数", f"{len(cleaned_data.columns)}")
    
    with col3:
        numeric_cols = len(cleaned_data.select_dtypes(include=['number']).columns)
        st.metric("数值特征", f"{numeric_cols}")
    
    with col4:
        categorical_cols = len(cleaned_data.select_dtypes(include=['object']).columns)
        st.metric("分类特征", f"{categorical_cols}")
    
    # 特征工程区域
    st.markdown("### 🔧 风险特征工程")
    
    st.markdown("""
    **风险特征生成说明：**
    - **时间风险特征**: 交易时间风险评分、工作日vs周末模式、节假日异常检测
    - **金额风险特征**: 交易金额标准化分数、用户历史平均金额对比、金额异常程度评估
    - **设备和地理特征**: 设备类型风险评分、IP地址地理位置分析、地址一致性检查
    - **账户行为特征**: 账户年龄风险评估、交易频率分析、支付方式多样性
    """)
    
    # 特征工程参数设置
    st.markdown("#### ⚙️ 特征工程参数")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 时间特征参数
        st.markdown("**时间特征参数**")
        time_weight = st.slider("时间权重", 0.1, 2.0, 1.0, 0.1, help="时间相关特征的权重")
        night_risk_threshold = st.slider("夜间风险阈值", 22, 6, 23, help="夜间交易时间阈值")
        
        # 金额特征参数
        st.markdown("**金额特征参数**")
        amount_weight = st.slider("金额权重", 0.1, 2.0, 1.0, 0.1, help="金额相关特征的权重")
        amount_std_threshold = st.slider("金额标准差阈值", 1.0, 5.0, 2.0, 0.1, help="金额异常检测标准差倍数")
    
    with col2:
        # 设备特征参数
        st.markdown("**设备特征参数**")
        device_weight = st.slider("设备权重", 0.1, 2.0, 1.0, 0.1, help="设备相关特征的权重")
        
        # 账户特征参数
        st.markdown("**账户特征参数**")
        account_weight = st.slider("账户权重", 0.1, 2.0, 1.0, 0.1, help="账户相关特征的权重")
        account_age_threshold = st.slider("账户年龄阈值", 30, 365, 90, help="新账户年龄阈值(天)")
    
    # 执行特征工程
    if st.button("🚀 执行特征工程", type="primary", help="基于当前参数生成风险特征"):
        try:
            with st.spinner("正在生成风险特征..."):
                # 创建特征工程器
                feature_engineer = RiskFeatureEngineer()
                
                # 生成特征
                engineered_data = feature_engineer.engineer_all_features(cleaned_data)
                
                # 计算特征重要性
                if 'is_fraudulent' in engineered_data.columns:
                    feature_importance = feature_engineer.calculate_feature_importance(engineered_data)

                # 保存结果
                st.session_state.engineered_features = engineered_data
                st.session_state.feature_info = feature_engineer.get_feature_info()
                
                st.success("✅ 风险特征生成完成！")
                
        except Exception as e:
            st.error(f"❌ 特征工程失败: {e}")
            st.exception(e)
    
    # 显示特征工程结果
    if st.session_state.engineered_features is not None:
        st.markdown("### 📈 特征工程结果")
        
        engineered_data = st.session_state.engineered_features
        feature_info = st.session_state.feature_info
        
        # 特征统计
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总特征数", f"{len(engineered_data.columns)}")
        
        with col2:
            original_features = len(cleaned_data.columns)
            risk_features = feature_info.get('risk_features', [])
            new_features = len(risk_features)
            st.metric("新增特征", f"{new_features}")
        
        with col3:
            numeric_features = len(engineered_data.select_dtypes(include=['number']).columns)
            st.metric("数值特征", f"{numeric_features}")
        
        with col4:
            categorical_features = len(engineered_data.select_dtypes(include=['object']).columns)
            st.metric("分类特征", f"{categorical_features}")
        
        # 特征分类展示
        st.markdown("#### 📋 特征分类")
        
        if feature_info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**原始特征**")
                original_features = feature_info.get('original_features', [])
                for feature in original_features:
                    st.markdown(f"- {feature}")
            
            with col2:
                st.markdown("**新增风险特征**")
                risk_features = feature_info.get('risk_features', [])
                for feature in risk_features:
                    st.markdown(f"- {feature}")
        
        # 特征重要性分析
        st.markdown("#### 🎯 特征重要性分析")
        
        # 计算特征相关性
        numeric_cols = engineered_data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            correlation_matrix = engineered_data[numeric_cols].corr()
            if not correlation_matrix.empty:
                # 相关性热力图
                fig = px.imshow(
                    correlation_matrix,
                    title="特征相关性热力图",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 高风险特征分析
            if 'is_fraudulent' in engineered_data.columns:
                st.markdown("**高风险特征分析**")
                
                # 计算与欺诈标签的相关性
                fraud_corr = engineered_data[numeric_cols].corrwith(engineered_data['is_fraudulent']).abs().sort_values(ascending=False)
                
                # 显示前10个最相关的特征
                top_features = fraud_corr.head(10)
                
                fig = px.bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation='h',
                    title="与欺诈标签最相关的特征",
                    labels={'x': '相关系数', 'y': '特征'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 特征分布分析
        st.markdown("#### 📊 特征分布分析")
        
        # 选择要分析的特征
        if len(numeric_cols) > 0:
            selected_feature = st.selectbox("选择特征查看分布", numeric_cols)
            
            # 创建分布图
            fig = px.histogram(
                engineered_data,
                x=selected_feature,
                title=f"{selected_feature} 分布图",
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 特征统计信息
            st.markdown(f"**{selected_feature} 统计信息**")
            stats = engineered_data[selected_feature].describe()
            st.dataframe(stats.to_frame(), use_container_width=True)
        
        # 风险特征可视化
        st.markdown("#### 🎨 风险特征可视化")
        
        # 时间风险特征
        time_features = [col for col in engineered_data.columns if 'time' in col.lower() or 'hour' in col.lower()]
        if time_features:
            st.markdown("**时间风险特征**")
            
            # 选择时间特征
            selected_time_feature = st.selectbox("选择时间特征", time_features)
            if selected_time_feature in engineered_data.columns:
                time_distribution = engineered_data.groupby(selected_time_feature).size().reset_index()
                time_distribution.columns = [selected_time_feature, 'count']
                fig = px.line(
                    time_distribution,
                    x=selected_time_feature,
                    y='count',
                    title=f"{selected_time_feature} 分布"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 金额风险特征
        amount_features = [col for col in engineered_data.columns if 'amount' in col.lower() or 'price' in col.lower()]
        if amount_features:
            st.markdown("**金额风险特征**")
            
            # 选择金额特征
            selected_amount_feature = st.selectbox("选择金额特征", amount_features)
            if selected_amount_feature in engineered_data.columns:
                fig = px.histogram(
                    engineered_data,
                    x=selected_amount_feature,
                    title=f"{selected_amount_feature} 分布",
                    nbins=50
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 数据预览
        st.markdown("#### 📋 特征工程后数据预览")
        st.dataframe(engineered_data.head(10), use_container_width=True)
        
        # 下一步按钮
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🚀 进入聚类分析", type="primary", use_container_width=True):
                st.success("✅ 特征工程完成，可以进入聚类分析页面！")
                st.info("💡 请在侧边栏选择'📊 聚类分析'页面继续")
    
    else:
        # 显示特征工程说明
        st.markdown("### 📝 特征工程说明")
        
        st.markdown("""
        **风险特征类型：**
        
        1. **时间风险特征**
           - 交易时间风险评分（深夜、凌晨高风险）
           - 工作日vs周末交易模式
           - 节假日交易异常检测
        
        2. **金额风险特征**
           - 交易金额标准化分数
           - 用户历史平均金额对比
           - 金额异常程度评估
        
        3. **设备和地理特征**
           - 设备类型风险评分
           - IP地址地理位置分析
           - 地址一致性检查
        
        4. **账户行为特征**
           - 账户年龄风险评估
           - 交易频率分析
           - 支付方式多样性
        
        **特征工程流程：**
        1. 保留所有原始特征
        2. 基于业务逻辑创建新特征
        3. 特征重要性分析和排序
        4. 特征质量检查和验证
        """) 