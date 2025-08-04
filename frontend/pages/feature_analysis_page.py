"""
Feature Engineering Page
Responsible for feature generation, analysis and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import backend modules
from backend.feature_engineer.risk_features import RiskFeatureEngineer

def show():
    """Display feature engineering page"""
    st.markdown('<div class="sub-header">🔧 Feature Engineering & Risk Feature Generation</div>', unsafe_allow_html=True)

    # Check if cleaned data exists
    if 'cleaned_data' not in st.session_state or st.session_state.cleaned_data is None:
        st.warning("⚠️ Please upload and clean data first!")
        st.info("💡 Please complete data preparation on the '📁 Data Upload' page")
        return

    # Initialize session state
    if 'engineered_features' not in st.session_state:
        st.session_state.engineered_features = None
    if 'feature_info' not in st.session_state:
        st.session_state.feature_info = None

    # Get cleaned data
    cleaned_data = st.session_state.cleaned_data

    # Check key fields (using updated column names)
    required_columns = ['customer_id', 'transaction_amount', 'payment_method', 'transaction_hour']
    missing_columns = [col for col in required_columns if col not in cleaned_data.columns]
    if missing_columns:
        st.error(f"❌ Data missing the following key fields, cannot perform feature engineering: {', '.join(missing_columns)}")
        st.info(f"💡 Current data column names: {list(cleaned_data.columns)}")
        return

    st.markdown("### 📊 Original Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Records", f"{len(cleaned_data):,}")

    with col2:
        st.metric("Original Features", f"{len(cleaned_data.columns)}")

    with col3:
        numeric_cols = len(cleaned_data.select_dtypes(include=['number']).columns)
        st.metric("Numerical Features", f"{numeric_cols}")

    with col4:
        categorical_cols = len(cleaned_data.select_dtypes(include=['object']).columns)
        st.metric("Categorical Features", f"{categorical_cols}")

    # Feature engineering area
    st.markdown("### 🔧 Risk Feature Engineering")

    st.markdown("""
    **Risk Feature Generation Description:**
    - **Time Risk Features**: Transaction time risk scoring, weekday vs weekend patterns, holiday anomaly detection
    - **Amount Risk Features**: Transaction amount standardized scores, user historical average comparison, amount anomaly assessment
    - **Device and Geographic Features**: Device type risk scoring, IP address geolocation analysis, address consistency check
    - **Account Behavior Features**: Account age risk assessment, transaction frequency analysis, payment method diversity
    """)
    
    # Feature engineering parameter settings
    st.markdown("#### ⚙️ Feature Engineering Parameters")

    col1, col2 = st.columns(2)

    with col1:
        # 时间特征参数
        st.markdown("**Time Feature Parameters**")
        time_weight = st.slider("Time Weight", 0.1, 2.0, 1.0, 0.1, help="Weight for time-related features")
        night_risk_threshold = st.slider("Night Risk Threshold", 22, 6, 23, help="Night transaction time threshold")

        # 金额特征参数
        st.markdown("**Amount Feature Parameters**")
        amount_weight = st.slider("Amount Weight", 0.1, 2.0, 1.0, 0.1, help="Weight for amount-related features")
        amount_std_threshold = st.slider("Amount Std Threshold", 1.0, 5.0, 2.0, 0.1, help="Standard deviation multiplier for amount anomaly detection")

    with col2:
        # 设备特征参数
        st.markdown("**Device Feature Parameters**")
        device_weight = st.slider("Device Weight", 0.1, 2.0, 1.0, 0.1, help="Weight for device-related features")

        # 账户特征参数
        st.markdown("**Account Feature Parameters**")
        account_weight = st.slider("Account Weight", 0.1, 2.0, 1.0, 0.1, help="Weight for account-related features")
        account_age_threshold = st.slider("Account Age Threshold", 30, 365, 90, help="New account age threshold (days)")
    
    # Execute feature engineering
    if st.button("🚀 Execute Feature Engineering", type="primary", help="Generate risk features based on current parameters"):
        try:
            with st.spinner("Generating risk features..."):
                # Create feature engineer
                feature_engineer = RiskFeatureEngineer()

                # Generate features
                engineered_data = feature_engineer.engineer_all_features(cleaned_data)

                # Calculate feature importance
                if 'is_fraudulent' in engineered_data.columns:
                    feature_importance = feature_engineer.calculate_feature_importance(engineered_data)

                # Save results
                st.session_state.engineered_features = engineered_data
                st.session_state.feature_info = feature_engineer.get_feature_info()

                st.success("✅ Risk feature generation completed!")

        except Exception as e:
            st.error(f"❌ Feature engineering failed: {e}")
            st.exception(e)

    # 显示特征工程结果
    if st.session_state.engineered_features is not None:
        st.markdown("### 📈 Feature Engineering Results")

        engineered_data = st.session_state.engineered_features
        feature_info = st.session_state.feature_info

        # 特征统计
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Features", f"{len(engineered_data.columns)}")

        with col2:
            original_features = len(cleaned_data.columns)
            risk_features = feature_info.get('risk_features', [])
            new_features = len(risk_features)
            st.metric("New Features", f"{new_features}")

        with col3:
            numeric_features = len(engineered_data.select_dtypes(include=['number']).columns)
            st.metric("Numerical Features", f"{numeric_features}")

        with col4:
            categorical_features = len(engineered_data.select_dtypes(include=['object']).columns)
            st.metric("Categorical Features", f"{categorical_features}")

        # 特征分类展示
        st.markdown("#### 📋 Feature Classification")

        if feature_info:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original Features**")
                original_features = feature_info.get('original_features', [])
                for feature in original_features:
                    st.markdown(f"- {feature}")

            with col2:
                st.markdown("**New Risk Features**")
                risk_features = feature_info.get('risk_features', [])
                for feature in risk_features:
                    st.markdown(f"- {feature}")

        # 特征重要性分析
        st.markdown("#### 🎯 Feature Importance Analysis")

        # 计算特征相关性
        numeric_cols = engineered_data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            correlation_matrix = engineered_data[numeric_cols].corr()
            if not correlation_matrix.empty:
                # 相关性热力图
                fig = px.imshow(
                    correlation_matrix,
                    title="Feature Correlation Heatmap",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")

            # High risk feature analysis
            if 'is_fraudulent' in engineered_data.columns:
                st.markdown("**High Risk Feature Analysis**")

                # Calculate correlation with fraud labels
                fraud_corr = engineered_data[numeric_cols].corrwith(engineered_data['is_fraudulent']).abs().sort_values(ascending=False)

                # Display top 10 most correlated features
                top_features = fraud_corr.head(10)

                fig = px.bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation='h',
                    title="Features Most Correlated with Fraud Labels",
                    labels={'x': 'Correlation Coefficient', 'y': 'Features'}
                )
                st.plotly_chart(fig, use_container_width=True, key="fraud_correlation_bar")
        
        # 特征分布分析
        st.markdown("#### 📊 Feature Distribution Analysis")

        # 选择要分析的特征
        if len(numeric_cols) > 0:
            selected_feature = st.selectbox("Select feature to view distribution", numeric_cols, key="feature_distribution_select")

            # 创建分布图
            fig = px.histogram(
                engineered_data,
                x=selected_feature,
                title=f"{selected_feature} Distribution",
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True, key="feature_distribution_hist")

            # 特征统计信息
            st.markdown(f"**{selected_feature} Statistics**")
            stats = engineered_data[selected_feature].describe()
            st.dataframe(stats.to_frame(), use_container_width=True)

        # 风险特征可视化
        st.markdown("#### 🎨 Risk Feature Visualization")

        # 时间风险特征
        time_features = [col for col in engineered_data.columns if 'time' in col.lower() or 'hour' in col.lower()]
        if time_features:
            st.markdown("**Time Risk Features**")

            # 选择时间特征
            selected_time_feature = st.selectbox("Select time feature", time_features, key="time_feature_select")
            if selected_time_feature in engineered_data.columns:
                time_distribution = engineered_data.groupby(selected_time_feature).size().reset_index()
                time_distribution.columns = [selected_time_feature, 'count']
                fig = px.line(
                    time_distribution,
                    x=selected_time_feature,
                    y='count',
                    title=f"{selected_time_feature} Distribution"
                )
                st.plotly_chart(fig, use_container_width=True, key="time_feature_line")
        
        # 金额风险特征
        amount_features = [col for col in engineered_data.columns if 'amount' in col.lower() or 'price' in col.lower()]
        if amount_features:
            st.markdown("**Amount Risk Features**")

            # 选择金额特征
            selected_amount_feature = st.selectbox("Select amount feature", amount_features, key="amount_feature_select")
            if selected_amount_feature in engineered_data.columns:
                fig = px.histogram(
                    engineered_data,
                    x=selected_amount_feature,
                    title=f"{selected_amount_feature} Distribution",
                    nbins=50
                )
                st.plotly_chart(fig, use_container_width=True, key="amount_feature_hist")

        # 数据预览
        st.markdown("#### 📋 Post-Feature Engineering Data Preview")
        st.dataframe(engineered_data.head(10), use_container_width=True)

        # 下一步按钮
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if st.button("🚀 Proceed to Clustering Analysis", type="primary", use_container_width=True):
                st.success("✅ Feature engineering completed, ready for clustering analysis!")
                st.info("💡 Please select '📊 Clustering Analysis' page from the sidebar to continue")

    else:
        # Display feature engineering description
        st.markdown("### 📝 Feature Engineering Description")

        st.markdown("""
        **Risk Feature Types:**

        1. **Time Risk Features**
           - Transaction time risk scoring (late night, early morning high risk)
           - Weekday vs weekend transaction patterns
           - Holiday transaction anomaly detection

        2. **Amount Risk Features**
           - Transaction amount standardized scores
           - User historical average amount comparison
           - Amount anomaly degree assessment

        3. **Device and Geographic Features**
           - Device type risk scoring
           - IP address geolocation analysis
           - Address consistency check

        4. **Account Behavior Features**
           - Account age risk assessment
           - Transaction frequency analysis
           - Payment method diversity

        **Feature Engineering Process:**
        1. Retain all original features
        2. Create new features based on business logic
        3. Feature importance analysis and ranking
        4. Feature quality check and validation
        """)