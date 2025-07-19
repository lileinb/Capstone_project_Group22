"""
Clustering Analysis Page
Responsible for user behavior pattern clustering and anomalous group identification
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
    """Display clustering analysis page"""
    st.markdown('<div class="sub-header">📊 Clustering Analysis & Anomalous Group Identification</div>', unsafe_allow_html=True)

    # 检查是否有特征工程数据
    if 'engineered_features' not in st.session_state or st.session_state.engineered_features is None:
        st.warning("⚠️ Please complete feature engineering first!")
        st.info("💡 Please complete feature generation on the '🔧 Feature Engineering' page")
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
    
    st.markdown("### 📊 Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Records", f"{len(engineered_data):,}")

    with col2:
        st.metric("Features", f"{len(engineered_data.columns)}")

    with col3:
        numeric_features = len(engineered_data.select_dtypes(include=['number']).columns)
        st.metric("Numerical Features", f"{numeric_features}")

    with col4:
        if 'is_fraudulent' in engineered_data.columns:
            fraud_rate = (engineered_data['is_fraudulent'].sum() / len(engineered_data) * 100).round(2)
            st.metric("Fraud Rate", f"{fraud_rate}%")
        else:
            st.metric("Fraud Rate", "N/A")

    # 聚类分析区域
    st.markdown("### 🔍 Clustering Analysis Configuration")

    st.markdown("""
    **Clustering Algorithm Description:**
    - **K-means Clustering**: Distance-based hard clustering, suitable for discovering spherical clusters
    - **DBSCAN Clustering**: Density-based clustering, suitable for discovering irregularly shaped clusters
    - **Gaussian Mixture Model**: Probabilistic clustering method, provides soft clustering results
    """)
    
    # 智能聚类选项
    st.markdown("### 🤖 Intelligent Clustering Mode")

    # 添加智能聚类选择
    clustering_mode = st.radio(
        "Select Clustering Mode",
        ["🤖 Intelligent Auto Clustering (Recommended)", "⚙️ Manual Parameter Adjustment"],
        help="Intelligent mode automatically selects optimal features, algorithms and parameters, manual mode allows custom configuration"
    )

    if clustering_mode == "🤖 Intelligent Auto Clustering (Recommended)":
        st.info("""
        🎯 **Intelligent Clustering Mode Features**:
        - 🔍 Automatic feature selection and engineering
        - ⚙️ Automatic algorithm and parameter optimization
        - 📊 Automatic risk threshold adjustment
        - 🎯 Ensure four-tier risk distribution
        - 📈 Maximize clustering quality metrics
        """)

        # 目标风险分布设置
        st.markdown("#### 🎯 Target Risk Distribution")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            target_low = st.slider("Low Risk Ratio", 0.2, 0.8, 0.5, 0.05, help="Target low risk user ratio")
        with col2:
            target_medium = st.slider("Medium Risk Ratio", 0.1, 0.5, 0.3, 0.05, help="Target medium risk user ratio")
        with col3:
            target_high = st.slider("High Risk Ratio", 0.05, 0.4, 0.15, 0.05, help="Target high risk user ratio")
        with col4:
            target_critical = st.slider("Critical Risk Ratio", 0.01, 0.2, 0.05, 0.01, help="Target critical risk user ratio")

        # 确保比例总和为1
        total = target_low + target_medium + target_high + target_critical
        if abs(total - 1.0) > 0.01:
            st.warning(f"⚠️ Risk ratio sum should be 100%, currently {total*100:.1f}%")

        target_risk_distribution = {
            'low': target_low,
            'medium': target_medium,
            'high': target_high,
            'critical': target_critical
        }

        # 智能聚类按钮
        if st.button("🚀 Start Intelligent Clustering", type="primary", help="One-click execution of optimal clustering analysis"):
            with st.spinner("🔄 Executing intelligent clustering optimization..."):
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

                    st.success("✅ Intelligent clustering completed!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Intelligent clustering failed: {e}")

        # 如果有智能聚类结果，显示优化摘要
        if ('clustering_result' in st.session_state and
            st.session_state.clustering_result is not None and
            'optimization_summary' in st.session_state.clustering_result):
            result = st.session_state.clustering_result
            summary = result['optimization_summary']

            st.markdown("### 📊 Intelligent Optimization Summary")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Silhouette Score", f"{summary['silhouette_score']:.3f}")
            with col2:
                st.metric("Cluster Count", summary['n_clusters'])
            with col3:
                st.metric("Selected Features", summary['feature_count'])
            with col4:
                st.metric("Algorithm", summary['algorithm'].upper())

            # 显示优化建议
            if 'recommendations' in result:
                st.markdown("#### 💡 Optimization Recommendations")
                for rec in result['recommendations']:
                    st.write(f"- {rec}")

            # 显示选择的特征
            if 'selected_features' in result:
                st.markdown("#### 🎯 Auto-Selected Features")
                st.write(", ".join(result['selected_features']))

            # 显示优化后的阈值
            if 'optimal_thresholds' in result:
                st.markdown("#### ⚙️ Optimized Risk Thresholds")
                thresholds = result['optimal_thresholds']
                st.write(f"Low Risk: <{thresholds['low']}, Medium Risk: {thresholds['low']}-{thresholds['medium']}, "
                        f"High Risk: {thresholds['medium']}-{thresholds['high']}, Critical Risk: >{thresholds['high']}")

        return  # 智能模式下不需要手动参数设置

    # 手动参数调整模式
    st.markdown("### ⚙️ Manual Parameter Adjustment")

    # 聚类参数设置
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⚙️ Clustering Algorithm Selection")

        # 算法选择
        algorithm = st.selectbox(
            "Select Clustering Algorithm",
            ["K-means", "DBSCAN", "Gaussian Mixture"],
            help="Choose clustering algorithm suitable for data characteristics"
        )

        # 特征选择
        numeric_cols = engineered_data.select_dtypes(include=['number']).columns.tolist()
        if 'is_fraudulent' in numeric_cols:
            numeric_cols.remove('is_fraudulent')  # 排除标签列

        selected_features = st.multiselect(
            "Select Clustering Features",
            numeric_cols,
            default=numeric_cols[:min(10, len(numeric_cols))],
            help="Select numerical features for clustering"
        )
    
    with col2:
        st.markdown("#### 📊 Algorithm Parameters")

        # 默认参数
        n_clusters = 5
        random_state = 42

        if algorithm == "K-means":
            n_clusters = st.slider("Number of Clusters", 2, 10, 5, help="K-means cluster count")
            max_iter = st.slider("Max Iterations", 100, 1000, 300, help="Maximum number of iterations")
            random_state = st.slider("Random Seed", 0, 100, 42, help="Random seed")

        elif algorithm == "DBSCAN":
            eps = st.slider("Neighborhood Radius", 0.1, 2.0, 0.5, 0.1, help="DBSCAN neighborhood radius")
            min_samples = st.slider("Min Samples", 2, 20, 5, help="Minimum samples required to form a core point")
            n_clusters = 5  # DBSCAN不需要预设聚类数，但ClusterAnalyzer需要这个参数

        elif algorithm == "Gaussian Mixture":
            n_clusters = st.slider("Number of Components", 2, 10, 5, help="Number of Gaussian mixture components")
            max_iter = st.slider("Max Iterations", 100, 1000, 300, help="Maximum number of iterations")
            random_state = st.slider("Random Seed", 0, 100, 42, help="Random seed")
    
    # 执行聚类分析
    col1, col2 = st.columns([3, 1])

    with col1:
        run_clustering = st.button("🚀 Execute Clustering Analysis", type="primary", help="Perform clustering analysis based on current parameters")

    with col2:
        clear_cache = st.button("🗑️ Clear Cache", help="Clear previous clustering results")

    if clear_cache:
        # 清除session state中的聚类相关数据
        keys_to_clear = ['clustering_results', 'cluster_labels', 'clustering_analysis', 'cluster_analysis']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ Cache cleared!")
        st.rerun()

    if run_clustering:
        try:
            with st.spinner("Performing clustering analysis..."):
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
                
                st.success("✅ Clustering analysis completed!")

                # 调试信息
                with st.expander("🔍 Debug Information", expanded=False):
                    st.write("Clustering result keys:", list(clustering_results.keys()))
                    st.write("Cluster count:", clustering_results.get('cluster_count', 0))

                    cluster_details = clustering_results.get('cluster_details', [])
                    st.write("Cluster details count:", len(cluster_details))

                    for i, detail in enumerate(cluster_details):
                        st.write(f"Cluster {i}:")
                        st.write(f"  - Size: {detail.get('size', 0)}")
                        st.write(f"  - Fraud rate: {detail.get('fraud_rate', 0):.4f}")
                        st.write(f"  - Risk level: {detail.get('risk_level', 'unknown')}")

        except Exception as e:
            st.error(f"❌ Clustering analysis failed: {e}")
            st.exception(e)
    
    # 显示聚类结果
    if st.session_state.clustering_results is not None:
        st.markdown("### 📈 Clustering Analysis Results")

        clustering_results = st.session_state.clustering_results
        cluster_labels = st.session_state.cluster_labels
        cluster_analysis = st.session_state.cluster_analysis

        # 聚类统计
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Algorithm", clustering_results['algorithm'])

        with col2:
            st.metric("Cluster Count", clustering_results.get('cluster_count', 0))

        with col3:
            st.metric("Feature Count", len(selected_features))

        with col4:
            total_records = len(cluster_labels)
            st.metric("Total Records", f"{total_records:,}")

        # 聚类分布
        st.markdown("#### 📊 Cluster Distribution")
        
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        
        # 聚类大小分布图
        fig = px.bar(
            x=cluster_sizes.index,
            y=cluster_sizes.values,
            title="Cluster Size Distribution",
            labels={'x': 'Cluster Label', 'y': 'Record Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # 聚类大小表格
        cluster_df = pd.DataFrame({
            'Cluster Label': cluster_sizes.index,
            'Record Count': cluster_sizes.values,
            'Percentage (%)': (cluster_sizes.values / total_records * 100).round(2)
        })
        st.dataframe(cluster_df, use_container_width=True)
        
        # 聚类特征分析
        if clustering_results and 'cluster_details' in clustering_results:
            st.markdown("#### 🔍 Cluster Feature Analysis")

            cluster_details = clustering_results['cluster_details']

            # 聚类详情表格
            if cluster_details:
                st.markdown("**Cluster Details**")

                details_data = []
                for detail in cluster_details:
                    details_data.append({
                        'Cluster ID': detail.get('cluster_id', 'N/A'),
                        'Size': detail.get('size', 0),
                        'Percentage (%)': detail.get('percentage', 0),
                        'Avg Transaction Amount': detail.get('avg_transaction_amount', 0),
                        'Avg Customer Age': detail.get('avg_customer_age', 0),
                        'Common Payment Method': detail.get('common_payment_method', 'unknown'),
                        'Common Device': detail.get('common_device', 'unknown'),
                        'Fraud Rate': detail.get('fraud_rate', 0),
                        'Risk Level': detail.get('risk_level', 'low')
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
                        title="Cluster Risk Level Distribution",
                        color=risk_counts.index,
                        color_discrete_map=risk_colors
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 显示风险分布统计
                    st.markdown("**Risk Distribution Statistics:**")
                    for level, count in risk_counts.items():
                        percentage = count / len(risk_levels) * 100
                        st.markdown(f"- {level.title()}: {count} clusters ({percentage:.1f}%)")
            
        # 聚类质量评估
        if clustering_results and 'quality_metrics' in clustering_results:
            st.markdown("#### 📊 Clustering Quality Assessment")

            quality_metrics = clustering_results['quality_metrics']

            col1, col2 = st.columns(2)

            with col1:
                silhouette_score = quality_metrics.get('silhouette_score', 0)
                st.metric(
                    "Silhouette Score",
                    f"{silhouette_score:.3f}",
                    help="Range [-1,1], closer to 1 indicates better clustering"
                )

            with col2:
                calinski_score = quality_metrics.get('calinski_harabasz_score', 0)
                st.metric(
                    "Calinski-Harabasz Index",
                    f"{calinski_score:.1f}",
                    help="Higher values indicate better clustering"
                )
        
        # 聚类可视化
        st.markdown("#### 🎨 Clustering Visualization")

        # 选择可视化特征
        if len(selected_features) >= 2:
            col1, col2 = st.columns(2)

            with col1:
                x_feature = st.selectbox("X-axis Feature", selected_features, index=0)

            with col2:
                y_feature = st.selectbox("Y-axis Feature", selected_features, index=1)

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
                title=f"Clustering Results Scatter Plot ({x_feature} vs {y_feature})",
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