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

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import backend modules
from backend.clustering.cluster_analyzer import ClusterAnalyzer
from backend.clustering.cluster_interpreter import ClusterInterpreter

def show():
    """Display clustering analysis page"""
    st.markdown('<div class="sub-header">📊 Clustering Analysis & Anomalous Group Identification</div>', unsafe_allow_html=True)

    # Check if feature engineering data exists
    if 'engineered_features' not in st.session_state or st.session_state.engineered_features is None:
        st.warning("⚠️ Please complete feature engineering first!")
        st.info("💡 Please complete feature generation on the '🔧 Feature Engineering' page")
        return

    # Initialize session state
    if 'clustering_results' not in st.session_state:
        st.session_state.clustering_results = None
    if 'cluster_labels' not in st.session_state:
        st.session_state.cluster_labels = None
    if 'cluster_analysis' not in st.session_state:
        st.session_state.cluster_analysis = None
    
    # Get feature engineering data
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

    # Clustering analysis area
    st.markdown("### 🔍 Clustering Analysis Configuration")

    st.markdown("""
    **Clustering Algorithm Description:**
    - **K-means Clustering**: Distance-based hard clustering, suitable for discovering spherical clusters
    - **DBSCAN Clustering**: Density-based clustering, suitable for discovering irregularly shaped clusters
    - **Gaussian Mixture Model**: Probabilistic clustering method, provides soft clustering results
    """)
    
    # Intelligent clustering options
    st.markdown("### 🤖 Intelligent Clustering Mode")

    # Add intelligent clustering selection
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

        # Target risk distribution settings
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

        # Ensure total ratio equals 1
        total = target_low + target_medium + target_high + target_critical
        if abs(total - 1.0) > 0.01:
            st.warning(f"⚠️ Risk ratio sum should be 100%, currently {total*100:.1f}%")

        target_risk_distribution = {
            'low': target_low,
            'medium': target_medium,
            'high': target_high,
            'critical': target_critical
        }

        # Intelligent clustering button
        if st.button("🚀 Start Intelligent Clustering", type="primary", help="One-click execution of optimal clustering analysis"):
            with st.spinner("🔄 Executing intelligent clustering optimization..."):
                try:
                    analyzer = ClusterAnalyzer()
                    result = analyzer.intelligent_auto_clustering(
                        engineered_data, target_risk_distribution
                    )

                    # Store intelligent clustering results
                    st.session_state.clustering_result = result

                    # Also store in standard format for use by other modules
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

        # If intelligent clustering results exist, display optimization summary
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

            # Display optimization recommendations
            if 'recommendations' in result:
                st.markdown("#### 💡 Optimization Recommendations")
                for rec in result['recommendations']:
                    st.write(f"- {rec}")

            # Display selected features
            if 'selected_features' in result:
                st.markdown("#### 🎯 Auto-Selected Features")
                st.write(", ".join(result['selected_features']))

            # Display optimized thresholds
            if 'optimal_thresholds' in result:
                st.markdown("#### ⚙️ Optimized Risk Thresholds")
                thresholds = result['optimal_thresholds']
                st.write(f"Low Risk: <{thresholds['low']}, Medium Risk: {thresholds['low']}-{thresholds['medium']}, "
                        f"High Risk: {thresholds['medium']}-{thresholds['high']}, Critical Risk: >{thresholds['high']}")

        return  # No manual parameter setting needed in intelligent mode

    # Manual parameter adjustment mode
    st.markdown("### ⚙️ Manual Parameter Adjustment")

    # Clustering parameter settings
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⚙️ Clustering Algorithm Selection")

        # Algorithm selection
        algorithm = st.selectbox(
            "Select Clustering Algorithm",
            ["K-means", "DBSCAN", "Gaussian Mixture"],
            help="Choose clustering algorithm suitable for data characteristics"
        )

        # Feature selection
        numeric_cols = engineered_data.select_dtypes(include=['number']).columns.tolist()
        if 'is_fraudulent' in numeric_cols:
            numeric_cols.remove('is_fraudulent')  # Exclude label column

        selected_features = st.multiselect(
            "Select Clustering Features",
            numeric_cols,
            default=numeric_cols[:min(10, len(numeric_cols))],
            help="Select numerical features for clustering"
        )
    
    with col2:
        st.markdown("#### 📊 Algorithm Parameters")

        # Default parameters
        n_clusters = 5
        random_state = 42

        if algorithm == "K-means":
            n_clusters = st.slider("Number of Clusters", 2, 10, 5, help="K-means cluster count")
            max_iter = st.slider("Max Iterations", 100, 1000, 300, help="Maximum number of iterations")
            random_state = st.slider("Random Seed", 0, 100, 42, help="Random seed")

        elif algorithm == "DBSCAN":
            eps = st.slider("Neighborhood Radius", 0.1, 2.0, 0.5, 0.1, help="DBSCAN neighborhood radius")
            min_samples = st.slider("Min Samples", 2, 20, 5, help="Minimum samples required to form a core point")
            n_clusters = 5  # DBSCAN doesn't need preset cluster count, but ClusterAnalyzer needs this parameter

        elif algorithm == "Gaussian Mixture":
            n_clusters = st.slider("Number of Components", 2, 10, 5, help="Number of Gaussian mixture components")
            max_iter = st.slider("Max Iterations", 100, 1000, 300, help="Maximum number of iterations")
            random_state = st.slider("Random Seed", 0, 100, 42, help="Random seed")
    
    # Execute clustering analysis
    col1, col2 = st.columns([3, 1])

    with col1:
        run_clustering = st.button("🚀 Execute Clustering Analysis", type="primary", help="Perform clustering analysis based on current parameters")

    with col2:
        clear_cache = st.button("🗑️ Clear Cache", help="Clear previous clustering results")

    if clear_cache:
        # Clear clustering-related data in session state
        keys_to_clear = ['clustering_results', 'cluster_labels', 'clustering_analysis', 'cluster_analysis']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ Cache cleared!")
        st.rerun()

    if run_clustering:
        try:
            with st.spinner("Performing clustering analysis..."):
                # Prepare data
                clustering_data = engineered_data[selected_features].copy()

                # Data standardization
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                clustering_data_scaled = scaler.fit_transform(clustering_data)

                # Create clustering analyzer
                cluster_analyzer = ClusterAnalyzer(n_clusters=n_clusters, random_state=random_state)

                # Execute clustering (using new API)
                algorithm_map = {
                    "K-means": "kmeans",
                    "DBSCAN": "dbscan",
                    "Gaussian Mixture": "gaussian_mixture"
                }

                clustering_results = cluster_analyzer.analyze_clusters(
                    engineered_data,
                    algorithm=algorithm_map.get(algorithm, "kmeans")
                )

                # Extract cluster labels
                cluster_labels = clustering_results.get('cluster_labels', [])

                # Save clustering results
                st.session_state.cluster_labels = cluster_labels
                st.session_state.clustering_results = clustering_results
                st.session_state.clustering_analysis = {
                    'algorithm': algorithm,
                    'features': selected_features,
                    'n_clusters': len(np.unique(cluster_labels)),
                    'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict()
                }
                
                # Cluster interpretation
                cluster_interpreter = ClusterInterpreter()
                cluster_analysis = cluster_interpreter.analyze_clusters(
                    engineered_data, cluster_labels, selected_features
                )
                st.session_state.cluster_analysis = cluster_analysis

                st.success("✅ Clustering analysis completed!")

                # Debug information
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
    
    # Display clustering results
    if st.session_state.clustering_results is not None:
        st.markdown("### 📈 Clustering Analysis Results")

        clustering_results = st.session_state.clustering_results
        cluster_labels = st.session_state.cluster_labels
        cluster_analysis = st.session_state.cluster_analysis

        # Clustering statistics
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

        # Cluster distribution
        st.markdown("#### 📊 Cluster Distribution")

        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()

        # Cluster size distribution chart
        fig = px.bar(
            x=cluster_sizes.index,
            y=cluster_sizes.values,
            title="Cluster Size Distribution",
            labels={'x': 'Cluster Label', 'y': 'Record Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cluster size table
        cluster_df = pd.DataFrame({
            'Cluster Label': cluster_sizes.index,
            'Record Count': cluster_sizes.values,
            'Percentage (%)': (cluster_sizes.values / total_records * 100).round(2)
        })
        st.dataframe(cluster_df, use_container_width=True)

        # Cluster feature analysis
        if clustering_results and 'cluster_details' in clustering_results:
            st.markdown("#### 🔍 Cluster Feature Analysis")

            cluster_details = clustering_results['cluster_details']

            # Cluster details table
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

                # Cluster risk level distribution - Fix: Use risk mapper results
                # Prioritize risk mapper results, fallback to risk levels in cluster_details
                cluster_risk_mapping = clustering_results.get('cluster_risk_mapping', {})

                if cluster_risk_mapping:
                    # Use risk mapper results
                    risk_levels = []
                    for detail in cluster_details:
                        cluster_id = detail.get('cluster_id', -1)
                        if cluster_id in cluster_risk_mapping:
                            risk_level = cluster_risk_mapping[cluster_id].get('risk_level', 'low')
                        else:
                            risk_level = detail.get('risk_level', 'low')
                        risk_levels.append(risk_level)
                else:
                    # Fallback to risk levels in cluster_details
                    risk_levels = [detail.get('risk_level', 'low') for detail in cluster_details]

                risk_counts = pd.Series(risk_levels).value_counts()

                if len(risk_counts) > 0:
                    # Define risk level colors
                    risk_colors = {
                        'low': '#22c55e',      # Green
                        'medium': '#f59e0b',   # Yellow
                        'high': '#f97316',     # Orange
                        'critical': '#ef4444'  # Red
                    }

                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Cluster Risk Level Distribution",
                        color=risk_counts.index,
                        color_discrete_map=risk_colors
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display risk distribution statistics
                    st.markdown("**Risk Distribution Statistics:**")
                    for level, count in risk_counts.items():
                        percentage = count / len(risk_levels) * 100
                        st.markdown(f"- {level.title()}: {count} clusters ({percentage:.1f}%)")

        # Clustering quality assessment
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
        
        # Clustering visualization
        st.markdown("#### 🎨 Clustering Visualization")

        # Select visualization features
        if len(selected_features) >= 2:
            col1, col2 = st.columns(2)

            with col1:
                x_feature = st.selectbox("X-axis Feature", selected_features, index=0)

            with col2:
                y_feature = st.selectbox("Y-axis Feature", selected_features, index=1)

            # Create scatter plot
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

        # Anomalous cluster identification
        st.markdown("#### ⚠️ Anomalous Cluster Identification")

        if clustering_results and 'anomaly_analysis' in clustering_results:
            anomaly_analysis = clustering_results['anomaly_analysis']

            # High-risk clusters
            high_risk_clusters = anomaly_analysis.get('high_risk_clusters', [])
            if high_risk_clusters:
                st.warning(f"Found {len(high_risk_clusters)} high-risk clusters")

                for cluster_info in high_risk_clusters:
                    with st.expander(f"High-risk Cluster {cluster_info['cluster_id']}"):
                        st.markdown(f"**Fraud Rate**: {cluster_info['fraud_rate']:.3f}")
                        st.markdown(f"**Cluster Size**: {cluster_info['size']} records")

            # Small clusters
            small_clusters = anomaly_analysis.get('small_clusters', [])
            if small_clusters:
                st.info(f"Found {len(small_clusters)} small clusters (potential anomalous patterns)")

                for cluster_info in small_clusters:
                    with st.expander(f"Small Cluster {cluster_info['cluster_id']}"):
                        st.markdown(f"**Cluster Size**: {cluster_info['size']} records")
                        st.markdown(f"**Percentage**: {cluster_info['percentage']:.2f}%")

            if not high_risk_clusters and not small_clusters:
                st.success("✅ No obvious anomalous clusters found")
        
        # Cluster-fraud relationship analysis
        if 'is_fraudulent' in engineered_data.columns:
            st.markdown("#### 🎯 Cluster-Fraud Relationship Analysis")

            # Calculate fraud rate for each cluster
            fraud_by_cluster = engineered_data.groupby(cluster_labels)['is_fraudulent'].agg(['count', 'sum', 'mean'])
            fraud_by_cluster.columns = ['Total Records', 'Fraud Records', 'Fraud Rate']
            fraud_by_cluster['Fraud Rate (%)'] = (fraud_by_cluster['Fraud Rate'] * 100).round(2)

            # Fraud rate distribution chart
            fig = px.bar(
                x=fraud_by_cluster.index,
                y=fraud_by_cluster['Fraud Rate (%)'],
                title="Fraud Rate Distribution by Cluster",
                labels={'x': 'Cluster Label', 'y': 'Fraud Rate (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Fraud rate table
            st.dataframe(fraud_by_cluster, use_container_width=True)
        
        # Data preview
        st.markdown("#### 📋 Clustering Results Preview")

        # Add cluster labels to data
        result_data = engineered_data.copy()
        result_data['cluster_label'] = cluster_labels

        st.dataframe(result_data.head(10), use_container_width=True)

        # Next step button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if st.button("🚀 Proceed to Risk Scoring", type="primary", use_container_width=True):
                st.success("✅ Clustering analysis completed, ready to proceed to risk scoring!")
                st.info("💡 Please select '🎯 Risk Scoring' page from the sidebar to continue")
    
    else:
        # Display clustering analysis description
        st.markdown("### 📝 Clustering Analysis Description")

        st.markdown("""
        **Clustering Algorithm Features:**

        1. **K-means Clustering**
           - Hard clustering based on Euclidean distance
           - Suitable for discovering spherical or convex clusters
           - Fast computation and stable results
           - Requires pre-specifying the number of clusters

        2. **DBSCAN Clustering**
           - Density-based clustering algorithm
           - Can discover irregularly shaped clusters
           - Automatically identifies outliers
           - Does not require pre-specifying the number of clusters

        3. **Gaussian Mixture Model**
           - Probabilistic clustering method
           - Provides soft clustering results
           - Can handle overlapping clusters
           - Suitable for complex data distributions

        **Clustering Analysis Objectives:**
        - Discover user behavior patterns
        - Identify anomalous groups
        - Provide basis for risk scoring
        - Support personalized risk strategies
        """)