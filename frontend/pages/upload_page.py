"""
数据上传页面
负责数据上传、质量检查和预处理
"""

import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入后端模块
from backend.data_processor.data_loader import DataLoader
from backend.data_processor.data_cleaner import DataCleaner

def show():
    """显示数据上传页面"""
    st.markdown('<div class="sub-header">📁 数据上传与预处理</div>', unsafe_allow_html=True)
    
    # 初始化session state
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'data_info' not in st.session_state:
        st.session_state.data_info = None
    
    # 数据上传区域
    st.markdown("### 📁 数据上传")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "选择CSV文件",
            type=['csv'],
            help="支持CSV格式的交易数据文件"
        )
    
    with col2:
        if st.button("📊 加载示例数据1", help="加载第一个示例数据集"):
            try:
                data_loader = DataLoader()
                data = data_loader.load_dataset("Dataset/Fraudulent_E-Commerce_Transaction_Data_sampled_50000.csv")
                st.session_state.uploaded_data = data
                st.session_state.data_info = data_loader.get_dataset_info(data)
                st.success("✅ 示例数据1加载成功！")
            except Exception as e:
                st.error(f"❌ 加载示例数据失败: {e}")
    
    with col3:
        if st.button("📊 加载示例数据2", help="加载第二个示例数据集"):
            try:
                data_loader = DataLoader()
                data = data_loader.load_dataset("Dataset/Fraudulent_E-Commerce_Transaction_Data_2_sampled_50000.csv")
                st.session_state.uploaded_data = data
                st.session_state.data_info = data_loader.get_dataset_info(data)
                st.success("✅ 示例数据2加载成功！")
            except Exception as e:
                st.error(f"❌ 加载示例数据失败: {e}")
    
    # 处理上传的文件
    if uploaded_file is not None:
        try:
            data_loader = DataLoader()
            data = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = data
            st.session_state.data_info = data_loader.get_dataset_info(data)
            st.success("✅ 文件上传成功！")
        except Exception as e:
            st.error(f"❌ 文件上传失败: {e}")
    
    # 数据质量检查
    if st.session_state.uploaded_data is not None:
        st.markdown("### 📊 数据质量检查")
        
        # 数据基本信息
        data = st.session_state.uploaded_data
        info = st.session_state.data_info
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总记录数", f"{info['shape'][0]:,}")
        
        with col2:
            st.metric("特征数量", f"{info['shape'][1]}")
        
        with col3:
            missing_count = sum(info['missing_values'].values())
            st.metric("缺失值", f"{missing_count:,}")
        
        with col4:
            st.metric("重复行", f"{info['duplicate_rows']:,}")
        
        # 数据质量报告
        st.markdown("#### 📋 数据质量报告")
        
        # 缺失值分析
        if missing_count > 0:
            st.markdown("**缺失值分析**")
            missing_df = pd.DataFrame(list(info['missing_values'].items()), 
                                    columns=['特征', '缺失值数量'])
            missing_df['缺失率(%)'] = (missing_df['缺失值数量'] / len(data) * 100).round(2)
            
            fig = px.bar(missing_df, x='特征', y='缺失率(%)', 
                        title="特征缺失值分布",
                        color='缺失率(%)',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        # 数据类型分布
        st.markdown("**数据类型分布**")
        type_counts = pd.Series(info['data_types']).value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index,
                    title="数据类型分布")
        st.plotly_chart(fig, use_container_width=True)
        
        # 数据预览
        st.markdown("#### 📋 数据预览")
        st.dataframe(data.head(10), use_container_width=True)
        
        # 数据清理
        st.markdown("### 🔧 数据清理")
        
        if st.button("🧹 执行数据清理", help="自动清理数据中的缺失值、重复值等"):
            try:
                data_cleaner = DataCleaner()
                cleaned_data = data_cleaner.clean_data(data)
                st.session_state.cleaned_data = cleaned_data
                
                # 清理结果对比
                original_shape = data.shape
                cleaned_shape = cleaned_data.shape
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("原始记录数", f"{original_shape[0]:,}")
                
                with col2:
                    st.metric("清理后记录数", f"{cleaned_shape[0]:,}")
                
                with col3:
                    removed = original_shape[0] - cleaned_shape[0]
                    st.metric("移除记录数", f"{removed:,}")
                
                st.success("✅ 数据清理完成！")
                
            except Exception as e:
                st.error(f"❌ 数据清理失败: {e}")
        
        # 清理后的数据统计
        if st.session_state.cleaned_data is not None:
            st.markdown("#### 📊 清理后数据统计")
            
            cleaned_data = st.session_state.cleaned_data
            
            # 数值特征统计
            numeric_cols = cleaned_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.markdown("**数值特征统计**")
                st.dataframe(cleaned_data[numeric_cols].describe(), use_container_width=True)
                
                # 数值特征分布图
                if len(numeric_cols) > 0:
                    st.markdown("**数值特征分布**")
                    selected_numeric = st.selectbox("选择特征查看分布", numeric_cols)
                    
                    fig = px.histogram(cleaned_data, x=selected_numeric, 
                                     title=f"{selected_numeric} 分布图",
                                     nbins=50)
                    st.plotly_chart(fig, use_container_width=True)
            
            # 分类特征统计
            categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.markdown("**分类特征统计**")
                selected_categorical = st.selectbox("选择分类特征", categorical_cols)
                
                value_counts = cleaned_data[selected_categorical].value_counts().head(10)
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"{selected_categorical} 值分布",
                           labels={'x': selected_categorical, 'y': '频次'})
                st.plotly_chart(fig, use_container_width=True)
        
        # 下一步按钮
        if st.session_state.cleaned_data is not None:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("🚀 进入特征工程", type="primary", use_container_width=True):
                    st.success("✅ 数据准备完成，可以进入特征工程页面！")
                    st.info("💡 请在侧边栏选择'🔧 特征工程'页面继续")
    
    else:
        # 显示上传说明
        st.markdown("### 📝 上传说明")
        
        st.markdown("""
        **支持的数据格式：**
        - CSV文件格式
        - 包含交易相关字段
        - 建议文件大小 < 100MB
        
        **系统支持的标准字段：**
        - Transaction ID (交易ID)
        - Customer ID (客户ID)
        - Transaction Amount (交易金额)
        - Transaction Date (交易日期)
        - Payment Method (支付方式: credit card, debit card, bank transfer, PayPal)
        - Product Category (产品类别: electronics, clothing, home & garden, health & beauty, toys & games)
        - Quantity (数量: 1-5)
        - Customer Age (客户年龄: 18-74)
        - Customer Location (客户位置)
        - Device Used (使用设备: mobile, tablet, desktop)
        - IP Address (IP地址)
        - Shipping Address (收货地址)
        - Billing Address (账单地址)
        - Is Fraudulent (是否欺诈: 0/1)
        - Account Age Days (账户年龄天数: 1-365)
        - Transaction Hour (交易小时: 0-23)
        
        **数据质量要求：**
        - 缺失值比例 < 20%
        - 数据格式正确
        - 字段类型匹配
        """)
        
        # 显示示例数据信息
        st.markdown("### 📊 示例数据集信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**数据集1**")
            st.markdown("- 文件名: Fraudulent_E-Commerce_Transaction_Data_sampled_50000.csv")
            st.markdown("- 记录数: 50,000条")
            st.markdown("- 特征数: 16个")
            st.markdown("- 欺诈率: 约5%")
            st.markdown("- 数据质量: 良好")
        
        with col2:
            st.markdown("**数据集2**")
            st.markdown("- 文件名: Fraudulent_E-Commerce_Transaction_Data_2_sampled_50000.csv")
            st.markdown("- 记录数: 23,634条")
            st.markdown("- 特征数: 16个")
            st.markdown("- 欺诈率: 约5%")
            st.markdown("- 数据质量: 良好") 