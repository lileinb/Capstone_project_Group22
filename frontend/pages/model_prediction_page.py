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

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import backend modules (using safe import)
try:
    from backend.prediction.individual_risk_predictor import IndividualRiskPredictor
    from backend.clustering.cluster_analyzer import ClusterAnalyzer
    PREDICTION_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Risk prediction module import failed: {e}")
    st.info("💡 Please check if risk prediction modules and dependencies are correctly installed")
    PREDICTION_AVAILABLE = False
    IndividualRiskPredictor = None
    ClusterAnalyzer = None

def _check_prerequisites():
    """Check prerequisites with enhanced dependency validation"""
    # 必需的依赖
    if 'engineered_features' not in st.session_state or st.session_state.engineered_features is None:
        st.warning("⚠️ Please complete feature engineering first!")
        st.info("💡 Please complete feature generation on the '🔧 Feature Engineering' page")
        return False

    # 检查可选但推荐的依赖
    missing_recommended = []
    if 'clustering_results' not in st.session_state or st.session_state.clustering_results is None:
        missing_recommended.append("📊 Clustering Analysis")
    if 'pseudo_labels' not in st.session_state or st.session_state.pseudo_labels is None:
        if 'high_quality_labels' not in st.session_state or st.session_state.high_quality_labels is None:
            missing_recommended.append("🏷️ Pseudo Labeling")
    if 'four_class_risk_results' not in st.session_state or st.session_state.four_class_risk_results is None:
        missing_recommended.append("🎯 Risk Scoring")

    if missing_recommended:
        st.info("💡 **Enhanced Prediction Available**: For better model performance, consider completing:")
        for item in missing_recommended:
            st.info(f"   • {item}")
        st.info("🚀 **Current Mode**: Basic prediction (using engineered features only)")

        # Display available enhancements
        with st.expander("🔧 Available Enhancements", expanded=False):
            if 'pseudo_labels' in st.session_state or 'high_quality_labels' in st.session_state:
                st.success("✅ Pseudo labels available - Enhanced training enabled")
            else:
                st.info("🏷️ Pseudo labels: Improve model training with generated labels")

            if 'four_class_risk_results' in st.session_state:
                st.success("✅ Risk scoring available - Risk-aware prediction enabled")
            else:
                st.info("🎯 Risk scoring: Enable risk-stratified predictions")
    else:
        st.success("✅ All dependencies available - Full enhanced mode enabled!")

    return True


# Removed all old display functions, using new risk prediction display components


def _execute_individual_risk_prediction(engineered_data, clustering_results, use_clustering, risk_thresholds):
    """Execute individual risk prediction"""
    if not PREDICTION_AVAILABLE:
        st.error("❌ Risk prediction module unavailable, cannot perform prediction")
        st.info("💡 Please check the following items:")
        st.info("1. Ensure backend/prediction directory exists")
        st.info("2. Ensure risk scoring module is complete")
        st.info("3. Install necessary dependencies: pip install scikit-learn pandas numpy")
        return

    try:
        with st.spinner("Performing intelligent risk prediction..."):
            # Prepare data
            X = engineered_data.copy()

            # Keep only numeric features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_cols]

            # Handle missing values and infinite values
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            X = X.astype(float)

            # Create individual risk predictor
            risk_predictor = IndividualRiskPredictor()

            # Update risk thresholds
            if risk_thresholds:
                risk_predictor.risk_thresholds = risk_thresholds

            st.info(f"✅ Starting analysis of {len(X)} individual risk samples")

            # Execute individual risk prediction
            clustering_data = clustering_results if use_clustering else None
            risk_results = risk_predictor.predict_individual_risks(
                X,
                clustering_data,
                use_four_class_labels=True
            )

            # Check prediction results
            if risk_results.get('success', False):
                # Save results to session state
                st.session_state.individual_risk_results = risk_results
                st.session_state.risk_stratification = risk_results.get('stratification_stats', {})

                # Display prediction statistics
                total_samples = risk_results.get('total_samples', 0)
                processing_time = risk_results.get('processing_time', 0)

                st.success(f"✅ Individual risk prediction completed!")
                st.info(f"📊 Successfully analyzed {total_samples} samples, time taken: {processing_time:.2f} seconds")

                # Display dynamic threshold information
                if 'dynamic_thresholds' in risk_results:
                    thresholds = risk_results['dynamic_thresholds']
                    st.info(f"🎚️ Dynamic Thresholds: Low Risk(<{thresholds.get('low', 40):.1f}) | "
                           f"Medium Risk({thresholds.get('low', 40):.1f}-{thresholds.get('medium', 60):.1f}) | "
                           f"High Risk({thresholds.get('medium', 60):.1f}-{thresholds.get('high', 80):.1f}) | "
                           f"Critical Risk(>{thresholds.get('high', 80):.1f})")

                # Display risk stratification statistics
                stratification_stats = risk_results.get('stratification_stats', {})
                if stratification_stats:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        low_count = stratification_stats.get('low', {}).get('count', 0)
                        low_pct = stratification_stats.get('low', {}).get('percentage', 0)
                        st.metric("Low Risk Users", f"{low_count} ({low_pct:.1f}%)")

                    with col2:
                        medium_count = stratification_stats.get('medium', {}).get('count', 0)
                        medium_pct = stratification_stats.get('medium', {}).get('percentage', 0)
                        st.metric("Medium Risk Users", f"{medium_count} ({medium_pct:.1f}%)")

                    with col3:
                        high_count = stratification_stats.get('high', {}).get('count', 0)
                        high_pct = stratification_stats.get('high', {}).get('percentage', 0)
                        st.metric("High Risk Users", f"{high_count} ({high_pct:.1f}%)")

                    with col4:
                        critical_count = stratification_stats.get('critical', {}).get('count', 0)
                        critical_pct = stratification_stats.get('critical', {}).get('percentage', 0)
                        st.metric("Critical Risk Users", f"{critical_count} ({critical_pct:.1f}%)")

                # Display main attack types
                protection_recommendations = risk_results.get('protection_recommendations', {})
                attack_distribution = protection_recommendations.get('attack_type_distribution', {})

                if attack_distribution:
                    st.markdown("#### 🎯 Detected Main Attack Types")
                    for attack_type, count in sorted(attack_distribution.items(), key=lambda x: x[1], reverse=True)[:3]:
                        if attack_type != 'none' and count > 0:
                            st.info(f"🔍 {attack_type}: {count} cases")

            else:
                error_msg = risk_results.get('error', 'Unknown error')
                st.error(f"❌ Individual risk prediction failed: {error_msg}")

    except Exception as e:
        st.error(f"❌ Individual risk prediction execution failed: {str(e)}")
        st.exception(e)


def show():
    """Show intelligent risk prediction page"""
    st.markdown('<div class="sub-header">🎯 Intelligent Risk Prediction & Individual Analysis</div>', unsafe_allow_html=True)

    # Check prerequisites
    if not _check_prerequisites():
        return

    # Check risk prediction availability
    if not PREDICTION_AVAILABLE:
        st.error("❌ Risk prediction functionality unavailable")
        st.info("💡 Risk prediction module import failed, please check:")
        st.info("1. Whether backend/prediction directory exists")
        st.info("2. Whether risk scoring module is complete")
        st.info("3. Whether necessary Python packages are installed")

        with st.expander("📋 Installation Guide"):
            st.code("""
# Install basic dependencies
pip install scikit-learn pandas numpy

# Check module structure
ls backend/prediction/
ls backend/risk_scoring/
            """)
        return


    # Initialize session state
    if 'individual_risk_results' not in st.session_state:
        st.session_state.individual_risk_results = None
    if 'risk_stratification' not in st.session_state:
        st.session_state.risk_stratification = None

    # Get feature engineering data and clustering results
    engineered_data = st.session_state.engineered_features
    clustering_results = st.session_state.get('clustering_results', None)

    st.markdown("### 📊 Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Record Count", f"{len(engineered_data):,}")

    with col2:
        st.metric("Feature Count", f"{len(engineered_data.columns)}")

    with col3:
        if clustering_results:
            cluster_count = clustering_results.get('cluster_count', 0)
            st.metric("Cluster Count", f"{cluster_count}")
        else:
            st.metric("Clustering Status", "Not Clustered")

    with col4:
        numeric_features = len(engineered_data.select_dtypes(include=['number']).columns)
        st.metric("Numeric Features", f"{numeric_features}")

    # Risk prediction configuration area
    st.markdown("### ⚙️ Intelligent Risk Prediction Configuration")

    st.markdown("""
    **Risk Prediction Features:**
    - **Individual Analysis**: Calculate detailed risk scores and attack type inference for each user
    - **Risk Stratification**: Classify users into low, medium, high, and critical risk levels
    - **Attack Type Inference**: Identify account takeover, identity theft, bulk fraud, testing attacks, etc.
    - **Protection Recommendations**: Provide targeted protection measures for different risk levels
    """)

    # Prediction configuration
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📋 Prediction Settings")

        # Whether to use clustering results
        use_clustering = st.checkbox(
            "Use clustering results to enhance prediction",
            value=clustering_results is not None,
            help="Based on clustering results can provide more accurate risk assessment"
        )

        # Risk stratification mode
        stratification_mode = st.selectbox(
            "Risk Stratification Mode",
            ["Standard Four-tier", "Custom Stratification"],
            help="Select risk stratification method"
        )

    with col2:
        st.markdown("#### ⚙️ Risk Parameters")

        # 风险阈值配置
        if stratification_mode == "Custom Stratification":
            st.markdown("**Custom Risk Thresholds**")
            low_threshold = st.slider("Low Risk Threshold", 0, 50, 40, help="0 to this value is low risk")
            medium_threshold = st.slider("Medium Risk Threshold", low_threshold, 80, 60, help="Low risk threshold to this value is medium risk")
            high_threshold = st.slider("High Risk Threshold", medium_threshold, 100, 80, help="Medium risk threshold to this value is high risk")

            risk_thresholds = {
                'low': low_threshold,
                'medium': medium_threshold,
                'high': high_threshold,
                'critical': 100
            }
        else:
            # Use standard thresholds
            risk_thresholds = {
                'low': 40,
                'medium': 60,
                'high': 80,
                'critical': 100
            }
            st.info("Using standard risk thresholds: Low(0-40), Medium(41-60), High(61-80), Critical(81-100)")

        # Display expected distribution
        st.markdown("**Expected Risk Distribution**")
        st.text("Low Risk: ~60%")
        st.text("Medium Risk: ~25%")
        st.text("High Risk: ~12%")
        st.text("Critical Risk: ~3%")

    # Execute risk prediction
    st.markdown("---")

    # Prediction button
    if st.button("🎯 Execute Intelligent Risk Prediction", type="primary", help="Perform individual analysis and attack type inference based on risk scoring"):
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
            with st.expander("📊 Risk Score Distribution Analysis", expanded=False):
                display_risk_score_distribution(risk_results)

        except ImportError as e:
            st.error(f"❌ 结果显示组件导入失败: {e}")
            # 降级显示基础结果
            _display_basic_risk_results(st.session_state.individual_risk_results)

    else:
        # Display intelligent risk prediction description
        st.markdown("### 📝 Intelligent Risk Prediction Description")

        st.markdown("""
        **Intelligent Risk Prediction Features:**

        🎯 **Individual Risk Analysis**
        - Calculate detailed risk scores for each user (0-100 points)
        - Comprehensive assessment based on multi-dimensional features
        - Provide personalized risk analysis reports

        🏷️ **Four-tier Risk Stratification**
        - **Low Risk** (0-40 points): Normal users, basic monitoring
        - **Medium Risk** (41-60 points): Requires attention, enhanced monitoring
        - **High Risk** (61-80 points): Focus attention, close monitoring
        - **Critical Risk** (81-100 points): Immediate action, real-time monitoring

        🔍 **Attack Type Inference**
        - **Account Takeover Attack**: Attackers gain control of user accounts
        - **Identity Theft Attack**: Using others' identity information for fraud
        - **Bulk Fraud Attack**: Large-scale automated fraud behavior
        - **Testing Attack**: Small amount testing to verify payment methods

        🛡️ **Protection Recommendations**
        - Provide specific protection measures for different risk levels
        - Recommend corresponding security strategies based on attack types
        - Provide system improvement and monitoring enhancement suggestions

        📊 **Data-Driven**
        - Enhance prediction accuracy based on clustering analysis
        - Use unsupervised learning to identify anomalous patterns
        - Combine business rules and statistical analysis
        """)

        # 下一步指引
        st.markdown("---")
        st.markdown("### 🚀 Getting Started")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("""
            💡 **Usage Steps**:
            1. Ensure feature engineering is completed
            2. Recommend completing clustering analysis first (optional)
            3. Configure risk prediction parameters
            4. Click "Execute Intelligent Risk Prediction" button
            5. View detailed individual risk analysis results
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
        st.metric("Analysis Samples", f"{total_samples:,}")
    with col2:
        st.metric("Processing Time", f"{processing_time:.2f}s")

    # Display risk stratification statistics
    stratification_stats = risk_results.get('stratification_stats', {})
    if stratification_stats:
        st.markdown("#### Risk Stratification Statistics")
        for level, stats in stratification_stats.items():
            count = stats.get('count', 0)
            percentage = stats.get('percentage', 0)
            st.write(f"**{level} Risk**: {count} users ({percentage:.1f}%)")

    # Display attack type distribution
    protection_recommendations = risk_results.get('protection_recommendations', {})
    attack_distribution = protection_recommendations.get('attack_type_distribution', {})

    if attack_distribution:
        st.markdown("#### Attack Type Distribution")
        for attack_type, count in attack_distribution.items():
            if attack_type != 'none' and count > 0:
                st.write(f"**{attack_type}**: {count} cases")

    st.success("✅ Basic risk prediction results display completed")