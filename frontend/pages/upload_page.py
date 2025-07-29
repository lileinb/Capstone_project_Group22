"""
Data Upload Page
Responsible for data upload, quality check and preprocessing
"""

import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import backend modules
from backend.data_processor.data_loader import DataLoader
from backend.data_processor.data_cleaner import DataCleaner

def show():
    """Display data upload page"""
    st.markdown('<div class="sub-header">📁 Data Upload & Preprocessing</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'data_info' not in st.session_state:
        st.session_state.data_info = None

    # Data upload area
    st.markdown("### 📁 Data Upload")

    uploaded_file = st.file_uploader(
        "Select CSV File",
        type=['csv'],
        help="Supports CSV format transaction data files"
    )

    # Process uploaded file
    if uploaded_file is not None:
        try:
            data_loader = DataLoader()
            data = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = data
            st.session_state.data_info = data_loader.get_dataset_info(data)
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"❌ File upload failed: {e}")

    # Data quality check
    if st.session_state.uploaded_data is not None:
        st.markdown("### 📊 Data Quality Check")

        # 数据基本信息
        data = st.session_state.uploaded_data
        info = st.session_state.data_info

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{info['shape'][0]:,}")

        with col2:
            st.metric("Feature Count", f"{info['shape'][1]}")

        with col3:
            missing_count = sum(info['missing_values'].values())
            st.metric("Missing Values", f"{missing_count:,}")

        with col4:
            st.metric("Duplicate Rows", f"{info['duplicate_rows']:,}")

        # 数据质量报告
        st.markdown("#### 📋 Data Quality Report")

        # 缺失值分析
        if missing_count > 0:
            st.markdown("**Missing Value Analysis**")
            missing_df = pd.DataFrame(list(info['missing_values'].items()),
                                    columns=['Feature', 'Missing Count'])
            missing_df['Missing Rate (%)'] = (missing_df['Missing Count'] / len(data) * 100).round(2)

            fig = px.bar(missing_df, x='Feature', y='Missing Rate (%)',
                        title="Feature Missing Value Distribution",
                        color='Missing Rate (%)',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True, key="missing_values_bar")

        # 数据类型分布
        st.markdown("**Data Type Distribution**")
        type_counts = pd.Series(info['data_types']).value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index,
                    title="Data Type Distribution")
        st.plotly_chart(fig, use_container_width=True, key="data_types_pie")

        # 数据预览
        st.markdown("#### 📋 Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # 数据清理
        st.markdown("### 🔧 Data Cleaning")

        if st.button("🧹 Execute Data Cleaning", help="Automatically clean missing values, duplicates, etc. in data"):
            try:
                data_cleaner = DataCleaner()
                cleaned_data = data_cleaner.clean_data(data)
                st.session_state.cleaned_data = cleaned_data

                # 清理结果对比
                original_shape = data.shape
                cleaned_shape = cleaned_data.shape

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Original Records", f"{original_shape[0]:,}")

                with col2:
                    st.metric("Cleaned Records", f"{cleaned_shape[0]:,}")

                with col3:
                    removed = original_shape[0] - cleaned_shape[0]
                    st.metric("Removed Records", f"{removed:,}")

                st.success("✅ Data cleaning completed!")

            except Exception as e:
                st.error(f"❌ Data cleaning failed: {e}")
        
        # 清理后的数据统计
        if st.session_state.cleaned_data is not None:
            st.markdown("#### 📊 Cleaned Data Statistics")

            cleaned_data = st.session_state.cleaned_data

            # 数值特征统计
            numeric_cols = cleaned_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.markdown("**Numerical Feature Statistics**")
                st.dataframe(cleaned_data[numeric_cols].describe(), use_container_width=True)

                # 数值特征分布图
                if len(numeric_cols) > 0:
                    st.markdown("**Numerical Feature Distribution**")
                    selected_numeric = st.selectbox("Select feature to view distribution", numeric_cols, key="upload_numeric_select")

                    fig = px.histogram(cleaned_data, x=selected_numeric,
                                     title=f"{selected_numeric} Distribution",
                                     nbins=50)
                    st.plotly_chart(fig, use_container_width=True, key="numeric_distribution_hist")

            # 分类特征统计
            categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.markdown("**Categorical Feature Statistics**")
                selected_categorical = st.selectbox("Select categorical feature", categorical_cols, key="upload_categorical_select")

                value_counts = cleaned_data[selected_categorical].value_counts().head(10)
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"{selected_categorical} Value Distribution",
                           labels={'x': selected_categorical, 'y': 'Frequency'})
                st.plotly_chart(fig, use_container_width=True, key="categorical_distribution_bar")
        
        # 下一步按钮
        if st.session_state.cleaned_data is not None:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])

            with col2:
                if st.button("🚀 Proceed to Feature Engineering", type="primary", use_container_width=True):
                    st.success("✅ Data preparation completed, ready for feature engineering!")
                    st.info("💡 Please select '🔧 Feature Engineering' page from the sidebar to continue")

    else:
        # 显示上传说明
        st.markdown("### 📝 Upload Instructions")

        st.markdown("""
        **Supported Data Formats:**
        - CSV file format
        - Contains transaction-related fields
        - Recommended file size < 100MB

        **System Supported Standard Fields:**
        - Transaction ID
        - Customer ID
        - Transaction Amount
        - Transaction Date
        - Payment Method (credit card, debit card, bank transfer, PayPal)
        - Product Category (electronics, clothing, home & garden, health & beauty, toys & games)
        - Quantity (1-5)
        - Customer Age (18-74)
        - Customer Location
        - Device Used (mobile, tablet, desktop)
        - IP Address
        - Shipping Address
        - Billing Address
        - Is Fraudulent (0/1)
        - Account Age Days (1-365)
        - Transaction Hour (0-23)

        **Data Quality Requirements:**
        - Missing value ratio < 20%
        - Correct data format
        - Matching field types
        """)
        
