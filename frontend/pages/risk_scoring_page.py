"""
Risk Scoring Page
Based on four-class intelligent risk scoring system
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
from backend.risk_scoring.four_class_risk_calculator import FourClassRiskCalculator
from backend.risk_scoring.dynamic_threshold_manager import DynamicThresholdManager
from backend.clustering.cluster_analyzer import ClusterAnalyzer

def show():
    """Display four-class risk scoring page"""
    st.markdown('<div class="sub-header">🎯 Four-Class Intelligent Risk Scoring System</div>', unsafe_allow_html=True)

    # Check prerequisites
    if not _check_prerequisites():
        return

    # Initialize session state
    _initialize_session_state()

    # Get data
    engineered_data = st.session_state.engineered_features
    clustering_results = st.session_state.clustering_results

    # Display system description
    _show_system_description()

    # Data overview
    _show_data_overview(engineered_data, clustering_results)

    # Execute four-class risk scoring
    data_size = len(engineered_data) if engineered_data is not None else 0
    if data_size > 0:
        estimated_time = max(1, data_size * 0.008)  # Four-class mode: approximately 8ms/record
        st.caption(f"📊 Data Volume: {data_size:,} records | Estimated Time: {estimated_time:.1f}s (Four-class algorithm)")

    if st.button("🎯 Execute Four-Class Risk Scoring", type="primary", help="Use four-class algorithm for precise risk grading"):
        _execute_four_class_risk_scoring(engineered_data, clustering_results)

    # Display four-class risk scoring results
    if st.session_state.four_class_risk_results:
        _show_four_class_results()
        _show_next_steps()


def _check_prerequisites():
    """Check prerequisites"""
    if 'engineered_features' not in st.session_state or st.session_state.engineered_features is None:
        st.warning("⚠️ Please complete feature engineering first!")
        st.info("💡 Please complete feature generation on the '🔧 Feature Engineering' page")
        return False

    if 'clustering_results' not in st.session_state or st.session_state.clustering_results is None:
        st.warning("⚠️ Please complete clustering analysis first!")
        st.info("💡 Please complete clustering analysis on the '📊 Clustering Analysis' page")
        return False

    return True


def _initialize_session_state():
    """Initialize session state"""
    if 'four_class_risk_results' not in st.session_state:
        st.session_state.four_class_risk_results = None
    if 'four_class_risk_calculator' not in st.session_state:
        st.session_state.four_class_risk_calculator = FourClassRiskCalculator(enable_dynamic_thresholds=True)
    if 'dynamic_threshold_manager' not in st.session_state:
        st.session_state.dynamic_threshold_manager = DynamicThresholdManager()


def _show_system_description():
    """Display system description"""
    with st.expander("📖 Four-Class Intelligent Risk Scoring System Description", expanded=False):
        st.markdown("""
        ### 🎯 System Features
        - **Four-Class Scoring**: Precisely categorize into low, medium, high, and critical risk levels
        - **Intelligent Thresholds**: Dynamically adjust risk thresholds to ensure reasonable risk distribution
        - **Multi-dimensional Assessment**: Comprehensive consideration of cluster anomaly, feature deviation, business rules, etc.
        - **Semi-supervised Learning**: Utilize original labels to improve scoring accuracy
        - **Real-time Optimization**: Automatically optimize scoring algorithms based on data distribution

        ### 📊 Four-Class Risk Levels
        - 🟢 **Low Risk** (0-40 points): Normal transactions, approximately 60%
        - 🟡 **Medium Risk** (40-60 points): Requires monitoring, approximately 25%
        - 🟠 **High Risk** (60-80 points): Requires focused attention, approximately 12%
        - 🔴 **Critical Risk** (80-100 points): Requires immediate action, approximately 3%

        ### 📊 Scoring Dimensions
        1. **Cluster Anomaly** (25%): Based on cluster risk level
        2. **Feature Deviation** (30%): Individual feature deviation from cluster center
        3. **Business Rules** (25%): Expert rules based on e-commerce scenarios
        4. **Statistical Outliers** (15%): Anomaly detection based on statistical distribution
        5. **Pattern Consistency** (5%): Consistency assessment based on cluster quality

        ### 🔧 Workflow
        1. Generate four-class labels based on clustering results and original labels
        2. Calculate precise risk scores using multi-dimensional algorithms
        3. Dynamically optimize risk thresholds to ensure reasonable distribution
        4. Generate final four-class risk levels and detailed reports
        """)


def _show_data_overview(engineered_data, clustering_results):
    """Display data overview"""
    st.markdown("### 📊 Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Transaction Records", f"{len(engineered_data):,}")

    with col2:
        st.metric("Feature Dimensions", f"{len(engineered_data.columns)}")

    with col3:
        cluster_count = clustering_results.get('cluster_count', 0)
        st.metric("Cluster Count", f"{cluster_count}")

    with col4:
        fraud_rate = engineered_data.get('is_fraudulent', pd.Series([0])).mean()
        st.metric("Fraud Rate", f"{fraud_rate:.2%}")





def _execute_four_class_risk_scoring(engineered_data, clustering_results):
    """Execute four-class risk scoring"""
    try:
        with st.spinner("Calculating risk scores using four-class algorithm..."):
            # Record start time
            import time
            start_time = time.time()

            # Use four-class risk calculator
            four_class_calculator = st.session_state.four_class_risk_calculator

            risk_results = four_class_calculator.calculate_four_class_risk_scores(
                engineered_data, cluster_results=clustering_results
            )

            # Record end time
            end_time = time.time()
            calculation_time = end_time - start_time

            st.session_state.four_class_risk_results = risk_results

            if risk_results and risk_results.get('success'):
                # Display success message
                success_msg = f"✅ 🎯 Four-class risk scoring completed!"
                success_msg += f" Processed {risk_results['total_samples']} transactions in {calculation_time:.2f} seconds"
                st.success(success_msg)

                # Display four-class advantages
                st.info("🚀 **Four-class Advantages**: Uses advanced technologies such as dynamic thresholds, multi-dimensional scoring, and semi-supervised learning")

                # Display basic statistics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    avg_score = risk_results['statistics']['avg_risk_score']
                    st.metric("Average Risk Score", f"{avg_score:.2f}")

                with col2:
                    high_risk_pct = risk_results['high_risk_percentage']
                    st.metric("High Risk Ratio", f"{high_risk_pct:.1f}%")

                with col3:
                    threshold_type = risk_results['threshold_type']
                    st.metric("Threshold Type", threshold_type)

                with col4:
                    total_samples = risk_results['total_samples']
                    st.metric("Processed Samples", f"{total_samples:,}")

                # Display distribution validation
                distribution = risk_results.get('distribution', {})
                if distribution:
                    low_pct = distribution.get('low', {}).get('percentage', 0)
                    medium_pct = distribution.get('medium', {}).get('percentage', 0)
                    high_pct = distribution.get('high', {}).get('percentage', 0)
                    critical_pct = distribution.get('critical', {}).get('percentage', 0)

                    # Check if distribution is reasonable
                    if 50 <= low_pct <= 70 and 20 <= medium_pct <= 35 and 8 <= high_pct <= 18 and 1 <= critical_pct <= 8:
                        st.success(f"📊 **Distribution Validation**: ✅ Four-class distribution is reasonable - Low Risk {low_pct:.1f}%, Medium Risk {medium_pct:.1f}%, High Risk {high_pct:.1f}%, Critical Risk {critical_pct:.1f}%")
                    else:
                        st.warning(f"📊 **Distribution Validation**: ⚠️ Distribution needs adjustment - Low Risk {low_pct:.1f}%, Medium Risk {medium_pct:.1f}%, High Risk {high_pct:.1f}%, Critical Risk {critical_pct:.1f}%")
            else:
                st.error("❌ Four-class risk scoring calculation failed")

    except Exception as e:
        st.error(f"❌ Four-class risk scoring calculation error: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")





def _show_four_class_results():
    """Display four-class risk scoring results"""
    st.markdown("### 📈 Four-Class Risk Scoring Results")

    risk_results = st.session_state.four_class_risk_results

    # Four-class risk distribution chart
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Four-Class Risk Level Distribution**")
        distribution = risk_results.get('distribution', {})

        if distribution:
            # Prepare data
            labels = []
            values = []
            colors = []

            risk_colors = {
                'low': '#22c55e',      # Green
                'medium': '#f59e0b',   # Yellow
                'high': '#f97316',     # Orange
                'critical': '#ef4444'  # Red
            }

            for level, data in distribution.items():
                labels.append(f"{level.title()} ({data['percentage']:.1f}%)")
                values.append(data['count'])
                colors.append(risk_colors.get(level, '#6b7280'))

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                textinfo='label+percent',
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])

            fig.update_layout(
                title="Four-Class Risk Distribution",
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Risk Score Distribution**")
        detailed_results = risk_results.get('detailed_results', [])
        if detailed_results:
            scores = [r['risk_score'] for r in detailed_results]
            levels = [r['risk_level'] for r in detailed_results]

            # Create grouped histogram
            fig = go.Figure()

            for level, color in [('low', '#22c55e'), ('medium', '#f59e0b'),
                               ('high', '#f97316'), ('critical', '#ef4444')]:
                level_scores = [s for s, l in zip(scores, levels) if l == level]
                if level_scores:
                    fig.add_trace(go.Histogram(
                        x=level_scores,
                        name=level.title(),
                        marker_color=color,
                        opacity=0.7,
                        nbinsx=20
                    ))

            # Add threshold lines
            thresholds = risk_results.get('thresholds', {})
            if thresholds:
                for threshold_name, threshold_value in thresholds.items():
                    if threshold_name != 'critical':  # critical is 100, no need to display
                        fig.add_vline(
                            x=threshold_value,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text=f"{threshold_name.title()}: {threshold_value:.1f}"
                        )

            fig.update_layout(
                title="Risk Score Distribution",
                xaxis_title="Risk Score",
                yaxis_title="Frequency",
                barmode='overlay',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    # Risk threshold information
    st.markdown("### 🎯 Dynamic Risk Thresholds")

    thresholds = risk_results.get('thresholds', {})
    threshold_type = risk_results.get('threshold_type', 'unknown')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📊 Current Threshold Settings")
        if thresholds:
            st.markdown(f"- 🟢 **Low Risk**: 0 - {thresholds.get('low', 40):.1f}")
            st.markdown(f"- 🟡 **Medium Risk**: {thresholds.get('low', 40):.1f} - {thresholds.get('medium', 60):.1f}")
            st.markdown(f"- 🟠 **High Risk**: {thresholds.get('medium', 60):.1f} - {thresholds.get('high', 80):.1f}")
            st.markdown(f"- 🔴 **Critical Risk**: {thresholds.get('high', 80):.1f} - 100")

    with col2:
        st.markdown("#### 🎯 Actual Distribution")
        if distribution:
            for level, data in distribution.items():
                icon = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}.get(level, '⚪')
                st.markdown(f"- {icon} **{level.title()}**: {data['percentage']:.1f}% ({data['count']})")

    with col3:
        st.markdown("#### ⚙️ System Information")
        st.markdown(f"- **Threshold Type**: {threshold_type}")
        if 'distribution_analysis' in risk_results:
            analysis = risk_results['distribution_analysis']
            if analysis.get('is_reasonable', False):
                st.markdown("- **Distribution Quality**: ✅ Reasonable")
            else:
                st.markdown("- **Distribution Quality**: ⚠️ Needs Adjustment")

        # Display weight information
        weights = risk_results.get('risk_weights', {})
        if weights:
            st.markdown("- **Scoring Weights**:")
            for component, weight in weights.items():
                st.markdown(f"  - {component}: {weight:.0%}")




# Display next steps
def _show_next_steps():
    """Display next steps"""
    st.markdown("### 🚀 Next Steps")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🤖 Next: Model Prediction", type="primary", use_container_width=True):
            st.success("✅ Risk scoring completed, ready to proceed to model prediction!")
            st.info("💡 Please select '🤖 Model Prediction' page from the sidebar to continue")

    with col2:
        if st.button("🏷️ Next: Pseudo Labeling", type="primary", use_container_width=True):
            st.success("✅ Risk scoring completed, ready to proceed to pseudo labeling!")
            st.info("💡 Please select '🏷️ Pseudo Labeling' page from the sidebar to continue")


