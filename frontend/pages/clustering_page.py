"""
聚类分析页面
负责用户行为模式聚类和异常群体识别
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
from backend.clustering.cluster_analyzer import ClusterAnalyzer
from backend.clustering.cluster_interpreter import ClusterInterpreter

def show():
    """显示聚类分析页面"""
    st.markdown('<div class="sub-header">📊 聚类分析与异常群体识别</div>', unsafe_allow_html=True)
    
    # 检查是否有特征工程数据
    if 'engineered_features' not in st.session_state or st.session_state.engineered_features is None:
        st.warning("⚠️ 请先完成特征工程！")
        st.info("💡 请在'🔧 特征工程'页面完成特征生成")
        return
    
    # 初始化session state
    if 'clustering_results' not in st.session_state:
        st.session_state.clustering_results = None
    if 'cluster_labels' not in st.session_state:
        st.session_state.cluster_labels = None
    if 'cluster_analysis' not in st.session_state:
        st.session_state.cluster_analysis = None
    
    # 获取特征工程数据
    engineered_data = st.session_state.engineered_features
    
    st.markdown("### 📊 数据概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("记录数", f"{len(engineered_data):,}")
    
    with col2:
        st.metric("特征数", f"{len(engineered_data.columns)}")
    
    with col3:
        numeric_features = len(engineered_data.select_dtypes(include=['number']).columns)
        st.metric("数值特征", f"{numeric_features}")
    
    with col4:
        if 'is_fraudulent' in engineered_data.columns:
            fraud_rate = (engineered_data['is_fraudulent'].sum() / len(engineered_data) * 100).round(2)
            st.metric("欺诈率", f"{fraud_rate}%")
        else:
            st.metric("欺诈率", "N/A")
    
    # 聚类分析区域
    st.markdown("### 🔍 聚类分析配置")
    
    st.markdown("""
    **聚类算法说明：**
    - **K-means聚类**: 基于距离的硬聚类，适合发现球形聚类
    - **DBSCAN聚类**: 基于密度的聚类，适合发现不规则形状的聚类
    - **高斯混合模型**: 概率聚类方法，提供软聚类结果
    """)
    
    # 智能聚类选项
    st.markdown("### 🤖 智能聚类模式")

    # 添加智能聚类选择
    clustering_mode = st.radio(
        "选择聚类模式",
        ["🤖 智能自动聚类 (推荐)", "⚙️ 手动参数调整"],
        help="智能模式会自动选择最优特征、算法和参数，手动模式允许自定义配置"
    )

    if clustering_mode == "🤖 智能自动聚类 (推荐)":
        st.info("""
        🎯 **智能聚类模式特点**:
        - 🔍 自动特征选择和工程
        - ⚙️ 自动算法和参数优化
        - 📊 自动风险阈值调整
        - 🎯 确保四层风险分布
        - 📈 最大化聚类质量指标
        """)

        # 目标风险分布设置
        st.markdown("#### 🎯 目标风险分布")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            target_low = st.slider("低风险比例", 0.2, 0.8, 0.5, 0.05, help="目标低风险用户比例")
        with col2:
            target_medium = st.slider("中风险比例", 0.1, 0.5, 0.3, 0.05, help="目标中风险用户比例")
        with col3:
            target_high = st.slider("高风险比例", 0.05, 0.4, 0.15, 0.05, help="目标高风险用户比例")
        with col4:
            target_critical = st.slider("极高风险比例", 0.01, 0.2, 0.05, 0.01, help="目标极高风险用户比例")

        # 确保比例总和为1
        total = target_low + target_medium + target_high + target_critical
        if abs(total - 1.0) > 0.01:
            st.warning(f"⚠️ 风险比例总和应为100%，当前为{total*100:.1f}%")

        target_risk_distribution = {
            'low': target_low,
            'medium': target_medium,
            'high': target_high,
            'critical': target_critical
        }

        # 智能聚类按钮
        if st.button("🚀 开始智能聚类", type="primary", help="一键执行最优聚类分析"):
            with st.spinner("🔄 正在执行智能聚类优化..."):
                try:
                    analyzer = ClusterAnalyzer()
                    result = analyzer.intelligent_auto_clustering(
                        engineered_data, target_risk_distribution
                    )

                    # 存储智能聚类结果
                    st.session_state.clustering_result = result

                    # 同时以标准格式存储，供其他模块使用
                    standard_clustering_results = {
                        'algorithm': result.get('algorithm', 'kmeans'),
                        'n_clusters': result.get('n_clusters', 3),
                        'cluster_count': result.get('n_clusters', 3),
                        'cluster_details': result.get('cluster_details', []),
                        'cluster_labels': result.get('cluster_labels', []),
                        'silhouette_score': result.get('silhouette_score', 0),
                        'calinski_harabasz_score': result.get('calinski_harabasz_score', 0),
                        'selected_features': result.get('selected_features', []),
                        'optimal_thresholds': result.get('optimal_thresholds', {}),
                        'risk_mapping': result.get('risk_mapping', {}),
                        'optimization_summary': result.get('optimization_summary', {}),
                        'source': 'intelligent_clustering'
                    }

                    st.session_state.clustering_results = standard_clustering_results
                    st.session_state.cluster_labels = result.get('cluster_labels', [])
                    st.session_state.cluster_analysis = {
                        'algorithm': result.get('algorithm', 'kmeans'),
                        'features': result.get('selected_features', []),
                        'n_clusters': result.get('n_clusters', 3),
                        'cluster_sizes': {},
                        'source': 'intelligent_clustering'
                    }

                    st.success("✅ 智能聚类完成！")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 智能聚类失败: {e}")

        # 如果有智能聚类结果，显示优化摘要
        if ('clustering_result' in st.session_state and
            st.session_state.clustering_result is not None and
            'optimization_summary' in st.session_state.clustering_result):
            result = st.session_state.clustering_result
            summary = result['optimization_summary']

            st.markdown("### 📊 智能优化摘要")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("轮廓系数", f"{summary['silhouette_score']:.3f}")
            with col2:
                st.metric("聚类数量", summary['n_clusters'])
            with col3:
                st.metric("选择特征数", summary['feature_count'])
            with col4:
                st.metric("算法", summary['algorithm'].upper())

            # 显示优化建议
            if 'recommendations' in result:
                st.markdown("#### 💡 优化建议")
                for rec in result['recommendations']:
                    st.write(f"- {rec}")

            # 显示选择的特征
            if 'selected_features' in result:
                st.markdown("#### 🎯 自动选择的特征")
                st.write(", ".join(result['selected_features']))

            # 显示优化后的阈值
            if 'optimal_thresholds' in result:
                st.markdown("#### ⚙️ 优化后的风险阈值")
                thresholds = result['optimal_thresholds']
                st.write(f"低风险: <{thresholds['low']}, 中风险: {thresholds['low']}-{thresholds['medium']}, "
                        f"高风险: {thresholds['medium']}-{thresholds['high']}, 极高风险: >{thresholds['high']}")

        return  # 智能模式下不需要手动参数设置

    # 手动参数调整模式
    st.markdown("### ⚙️ 手动参数调整")

    # 聚类参数设置
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⚙️ 聚类算法选择")

        # 算法选择
        algorithm = st.selectbox(
            "选择聚类算法",
            ["K-means", "DBSCAN", "Gaussian Mixture"],
            help="选择适合数据特点的聚类算法"
        )
        
        # 特征选择
        numeric_cols = engineered_data.select_dtypes(include=['number']).columns.tolist()
        if 'is_fraudulent' in numeric_cols:
            numeric_cols.remove('is_fraudulent')  # 排除标签列
        
        selected_features = st.multiselect(
            "选择聚类特征",
            numeric_cols,
            default=numeric_cols[:min(10, len(numeric_cols))],
            help="选择用于聚类的数值特征"
        )
    
    with col2:
        st.markdown("#### 📊 算法参数")
        
        # 默认参数
        n_clusters = 5
        random_state = 42

        if algorithm == "K-means":
            n_clusters = st.slider("聚类数量", 2, 10, 5, help="K-means聚类数量")
            max_iter = st.slider("最大迭代次数", 100, 1000, 300, help="最大迭代次数")
            random_state = st.slider("随机种子", 0, 100, 42, help="随机种子")

        elif algorithm == "DBSCAN":
            eps = st.slider("邻域半径", 0.1, 2.0, 0.5, 0.1, help="DBSCAN邻域半径")
            min_samples = st.slider("最小样本数", 2, 20, 5, help="形成核心点所需的最小样本数")
            n_clusters = 5  # DBSCAN不需要预设聚类数，但ClusterAnalyzer需要这个参数

        elif algorithm == "Gaussian Mixture":
            n_clusters = st.slider("混合成分数", 2, 10, 5, help="高斯混合成分数量")
            max_iter = st.slider("最大迭代次数", 100, 1000, 300, help="最大迭代次数")
            random_state = st.slider("随机种子", 0, 100, 42, help="随机种子")
    
    # 执行聚类分析
    col1, col2 = st.columns([3, 1])

    with col1:
        run_clustering = st.button("🚀 执行聚类分析", type="primary", help="基于当前参数进行聚类分析")

    with col2:
        clear_cache = st.button("🗑️ 清除缓存", help="清除之前的聚类结果")

    if clear_cache:
        # 清除session state中的聚类相关数据
        keys_to_clear = ['clustering_results', 'cluster_labels', 'clustering_analysis', 'cluster_analysis']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ 缓存已清除！")
        st.rerun()

    if run_clustering:
        try:
            with st.spinner("正在进行聚类分析..."):
                # 准备数据
                clustering_data = engineered_data[selected_features].copy()
                
                # 数据标准化
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                clustering_data_scaled = scaler.fit_transform(clustering_data)
                
                # 创建聚类分析器
                cluster_analyzer = ClusterAnalyzer(n_clusters=n_clusters, random_state=random_state)

                # 执行聚类（使用新的API）
                algorithm_map = {
                    "K-means": "kmeans",
                    "DBSCAN": "dbscan",
                    "Gaussian Mixture": "gaussian_mixture"
                }

                clustering_results = cluster_analyzer.analyze_clusters(
                    engineered_data,
                    algorithm=algorithm_map.get(algorithm, "kmeans")
                )

                # 提取聚类标签
                cluster_labels = clustering_results.get('cluster_labels', [])
                
                # 保存聚类结果
                st.session_state.cluster_labels = cluster_labels
                st.session_state.clustering_results = clustering_results
                st.session_state.clustering_analysis = {
                    'algorithm': algorithm,
                    'features': selected_features,
                    'n_clusters': len(np.unique(cluster_labels)),
                    'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict()
                }
                
                # 聚类解释
                cluster_interpreter = ClusterInterpreter()
                cluster_analysis = cluster_interpreter.analyze_clusters(
                    engineered_data, cluster_labels, selected_features
                )
                st.session_state.cluster_analysis = cluster_analysis
                
                st.success("✅ 聚类分析完成！")

                # 调试信息
                with st.expander("🔍 调试信息", expanded=False):
                    st.write("聚类结果键:", list(clustering_results.keys()))
                    st.write("聚类数量:", clustering_results.get('cluster_count', 0))

                    cluster_details = clustering_results.get('cluster_details', [])
                    st.write("聚类详情数量:", len(cluster_details))

                    for i, detail in enumerate(cluster_details):
                        st.write(f"聚类 {i}:")
                        st.write(f"  - 大小: {detail.get('size', 0)}")
                        st.write(f"  - 欺诈率: {detail.get('fraud_rate', 0):.4f}")
                        st.write(f"  - 风险等级: {detail.get('risk_level', 'unknown')}")
                
        except Exception as e:
            st.error(f"❌ 聚类分析失败: {e}")
            st.exception(e)
    
    # 显示聚类结果
    if st.session_state.clustering_results is not None:
        st.markdown("### 📈 聚类分析结果")
        
        clustering_results = st.session_state.clustering_results
        cluster_labels = st.session_state.cluster_labels
        cluster_analysis = st.session_state.cluster_analysis
        
        # 聚类统计
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("算法", clustering_results['algorithm'])
        
        with col2:
            st.metric("聚类数量", clustering_results.get('cluster_count', 0))
        
        with col3:
            st.metric("特征数量", len(selected_features))
        
        with col4:
            total_records = len(cluster_labels)
            st.metric("总记录数", f"{total_records:,}")
        
        # 聚类分布
        st.markdown("#### 📊 聚类分布")
        
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        
        # 聚类大小分布图
        fig = px.bar(
            x=cluster_sizes.index,
            y=cluster_sizes.values,
            title="聚类大小分布",
            labels={'x': '聚类标签', 'y': '记录数'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 聚类大小表格
        cluster_df = pd.DataFrame({
            '聚类标签': cluster_sizes.index,
            '记录数': cluster_sizes.values,
            '占比(%)': (cluster_sizes.values / total_records * 100).round(2)
        })
        st.dataframe(cluster_df, use_container_width=True)
        
        # 聚类特征分析
        if clustering_results and 'cluster_details' in clustering_results:
            st.markdown("#### 🔍 聚类特征分析")

            cluster_details = clustering_results['cluster_details']

            # 聚类详情表格
            if cluster_details:
                st.markdown("**聚类详情**")

                details_data = []
                for detail in cluster_details:
                    details_data.append({
                        '聚类ID': detail.get('cluster_id', 'N/A'),
                        '大小': detail.get('size', 0),
                        '占比(%)': detail.get('percentage', 0),
                        '平均交易金额': detail.get('avg_transaction_amount', 0),
                        '平均客户年龄': detail.get('avg_customer_age', 0),
                        '常见支付方式': detail.get('common_payment_method', 'unknown'),
                        '常见设备': detail.get('common_device', 'unknown'),
                        '欺诈率': detail.get('fraud_rate', 0),
                        '风险等级': detail.get('risk_level', 'low')
                    })

                details_df = pd.DataFrame(details_data)
                st.dataframe(details_df, use_container_width=True)

                # 聚类风险等级分布 - 修复：使用风险映射器的结果
                # 优先使用风险映射器的结果，如果没有则使用cluster_details中的风险等级
                cluster_risk_mapping = clustering_results.get('cluster_risk_mapping', {})

                if cluster_risk_mapping:
                    # 使用风险映射器的结果
                    risk_levels = []
                    for detail in cluster_details:
                        cluster_id = detail.get('cluster_id', -1)
                        if cluster_id in cluster_risk_mapping:
                            risk_level = cluster_risk_mapping[cluster_id].get('risk_level', 'low')
                        else:
                            risk_level = detail.get('risk_level', 'low')
                        risk_levels.append(risk_level)
                else:
                    # 回退到cluster_details中的风险等级
                    risk_levels = [detail.get('risk_level', 'low') for detail in cluster_details]

                risk_counts = pd.Series(risk_levels).value_counts()

                if len(risk_counts) > 0:
                    # 定义风险等级颜色
                    risk_colors = {
                        'low': '#22c55e',      # 绿色
                        'medium': '#f59e0b',   # 黄色
                        'high': '#f97316',     # 橙色
                        'critical': '#ef4444'  # 红色
                    }

                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="聚类风险等级分布",
                        color=risk_counts.index,
                        color_discrete_map=risk_colors
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 显示风险分布统计
                    st.markdown("**风险分布统计:**")
                    for level, count in risk_counts.items():
                        percentage = count / len(risk_levels) * 100
                        st.markdown(f"- {level.title()}: {count} 个聚类 ({percentage:.1f}%)")
            
        # 聚类质量评估
        if clustering_results and 'quality_metrics' in clustering_results:
            st.markdown("#### 📊 聚类质量评估")

            quality_metrics = clustering_results['quality_metrics']

            col1, col2 = st.columns(2)

            with col1:
                silhouette_score = quality_metrics.get('silhouette_score', 0)
                st.metric(
                    "轮廓系数 (Silhouette Score)",
                    f"{silhouette_score:.3f}",
                    help="范围[-1,1]，越接近1表示聚类效果越好"
                )

            with col2:
                calinski_score = quality_metrics.get('calinski_harabasz_score', 0)
                st.metric(
                    "Calinski-Harabasz指数",
                    f"{calinski_score:.1f}",
                    help="值越大表示聚类效果越好"
                )
        
        # 聚类可视化
        st.markdown("#### 🎨 聚类可视化")
        
        # 选择可视化特征
        if len(selected_features) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_feature = st.selectbox("X轴特征", selected_features, index=0)
            
            with col2:
                y_feature = st.selectbox("Y轴特征", selected_features, index=1)
            
            # 创建散点图
            plot_data = pd.DataFrame({
                'x': engineered_data[x_feature],
                'y': engineered_data[y_feature],
                'cluster': cluster_labels
            })
            
            fig = px.scatter(
                plot_data,
                x='x',
                y='y',
                color='cluster',
                title=f"聚类结果散点图 ({x_feature} vs {y_feature})",
                labels={'x': x_feature, 'y': y_feature}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 异常聚类识别
        st.markdown("#### ⚠️ 异常聚类识别")

        if clustering_results and 'anomaly_analysis' in clustering_results:
            anomaly_analysis = clustering_results['anomaly_analysis']

            # 高风险聚类
            high_risk_clusters = anomaly_analysis.get('high_risk_clusters', [])
            if high_risk_clusters:
                st.warning(f"发现 {len(high_risk_clusters)} 个高风险聚类")

                for cluster_info in high_risk_clusters:
                    with st.expander(f"高风险聚类 {cluster_info['cluster_id']}"):
                        st.markdown(f"**欺诈率**: {cluster_info['fraud_rate']:.3f}")
                        st.markdown(f"**聚类大小**: {cluster_info['size']} 条记录")

            # 小聚类
            small_clusters = anomaly_analysis.get('small_clusters', [])
            if small_clusters:
                st.info(f"发现 {len(small_clusters)} 个小聚类（可能的异常模式）")

                for cluster_info in small_clusters:
                    with st.expander(f"小聚类 {cluster_info['cluster_id']}"):
                        st.markdown(f"**聚类大小**: {cluster_info['size']} 条记录")
                        st.markdown(f"**占比**: {cluster_info['percentage']:.2f}%")

            if not high_risk_clusters and not small_clusters:
                st.success("✅ 未发现明显异常聚类")
        
        # 聚类与欺诈关系分析
        if 'is_fraudulent' in engineered_data.columns:
            st.markdown("#### 🎯 聚类与欺诈关系分析")
            
            # 计算每个聚类的欺诈率
            fraud_by_cluster = engineered_data.groupby(cluster_labels)['is_fraudulent'].agg(['count', 'sum', 'mean'])
            fraud_by_cluster.columns = ['总记录数', '欺诈记录数', '欺诈率']
            fraud_by_cluster['欺诈率(%)'] = (fraud_by_cluster['欺诈率'] * 100).round(2)
            
            # 欺诈率分布图
            fig = px.bar(
                x=fraud_by_cluster.index,
                y=fraud_by_cluster['欺诈率(%)'],
                title="各聚类欺诈率分布",
                labels={'x': '聚类标签', 'y': '欺诈率(%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 欺诈率表格
            st.dataframe(fraud_by_cluster, use_container_width=True)
        
        # 数据预览
        st.markdown("#### 📋 聚类结果预览")
        
        # 添加聚类标签到数据
        result_data = engineered_data.copy()
        result_data['cluster_label'] = cluster_labels
        
        st.dataframe(result_data.head(10), use_container_width=True)
        
        # 下一步按钮
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🚀 进入风险评分", type="primary", use_container_width=True):
                st.success("✅ 聚类分析完成，可以进入风险评分页面！")
                st.info("💡 请在侧边栏选择'🎯 风险评分'页面继续")
    
    else:
        # 显示聚类分析说明
        st.markdown("### 📝 聚类分析说明")
        
        st.markdown("""
        **聚类算法特点：**
        
        1. **K-means聚类**
           - 基于欧几里得距离的硬聚类
           - 适合发现球形或凸形聚类
           - 计算速度快，结果稳定
           - 需要预先指定聚类数量
        
        2. **DBSCAN聚类**
           - 基于密度的聚类算法
           - 能发现不规则形状的聚类
           - 自动识别异常点
           - 不需要预先指定聚类数量
        
        3. **高斯混合模型**
           - 概率聚类方法
           - 提供软聚类结果
           - 能处理重叠聚类
           - 适合复杂的数据分布
        
        **聚类分析目标：**
        - 发现用户行为模式
        - 识别异常群体
        - 为风险评分提供依据
        - 支持个性化风险策略
        """) 