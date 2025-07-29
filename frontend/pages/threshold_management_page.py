"""
Dynamic Threshold Management Page
Real-time monitoring and adjustment of four-class risk thresholds
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
from backend.risk_scoring.dynamic_threshold_manager import DynamicThresholdManager

def show():
    """Display dynamic threshold management page"""
    st.markdown('<div class="sub-header">🎛️ Dynamic Threshold Management Center</div>', unsafe_allow_html=True)
    
    # Initialize session state
    _initialize_session_state()

    # Check prerequisites
    if not _check_prerequisites():
        return

    # Display system description
    _show_system_description()

    # Display current threshold status
    _show_current_threshold_status()

    # Threshold optimization control panel
    _show_threshold_optimization_panel()

    # Real-time distribution monitoring
    _show_real_time_distribution_monitoring()

    # Threshold adjustment history
    _show_threshold_adjustment_history()

def _initialize_session_state():
    """Initialize session state"""
    if 'threshold_manager' not in st.session_state:
        st.session_state.threshold_manager = DynamicThresholdManager()
    if 'threshold_history' not in st.session_state:
        st.session_state.threshold_history = []
    if 'current_thresholds' not in st.session_state:
        st.session_state.current_thresholds = None

def _check_prerequisites():
    """Check prerequisites"""
    if 'four_class_risk_results' not in st.session_state or st.session_state.four_class_risk_results is None:
        st.warning("⚠️ Please complete four-class risk scoring first!")
        st.info("💡 Please complete four-class risk scoring in the '🎯 Risk Scoring' page")
        return False
    return True

def _show_system_description():
    """Show system description"""
    with st.expander("📖 Dynamic Threshold Management System Description", expanded=False):
        st.markdown("""
        ### 🎯 System Functions
        - **Real-time Monitoring**: Monitor current risk distribution and threshold effectiveness
        - **Intelligent Optimization**: Automatically optimize thresholds based on target distribution
        - **Manual Adjustment**: Support manual fine-tuning of threshold parameters
        - **History Tracking**: Record threshold adjustment history and effects

        ### 📊 Target Distribution
        - 🟢 **Low Risk**: 60% (Normal transactions)
        - 🟡 **Medium Risk**: 25% (Need monitoring)
        - 🟠 **High Risk**: 12% (Need attention)
        - 🔴 **Critical Risk**: 3% (Need handling)

        ### 🔧 Optimization Strategy
        1. Calculate deviation based on current distribution
        2. Use iterative algorithm to optimize thresholds
        3. Validate distribution effect of new thresholds
        4. Apply optimal threshold configuration
        """)

def _show_current_threshold_status():
    """Show current threshold status"""
    st.markdown("### 📊 Current Threshold Status")

    risk_results = st.session_state.four_class_risk_results
    current_thresholds = risk_results.get('thresholds', {})
    distribution = risk_results.get('distribution', {})

    # Threshold information
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Current Threshold Settings")
        threshold_type = risk_results.get('threshold_type', 'unknown')
        st.info(f"**Threshold Type**: {threshold_type}")

        if current_thresholds:
            st.markdown(f"- 🟢 **Low Risk**: 0 - {current_thresholds.get('low', 40):.1f}")
            st.markdown(f"- 🟡 **Medium Risk**: {current_thresholds.get('low', 40):.1f} - {current_thresholds.get('medium', 60):.1f}")
            st.markdown(f"- 🟠 **High Risk**: {current_thresholds.get('medium', 60):.1f} - {current_thresholds.get('high', 80):.1f}")
            st.markdown(f"- 🔴 **Critical Risk**: {current_thresholds.get('high', 80):.1f} - 100")

    with col2:
        st.markdown("#### 📈 Actual Distribution Status")
        if distribution:
            target_dist = {'low': 60, 'medium': 25, 'high': 12, 'critical': 3}

            for level, data in distribution.items():
                actual_pct = data['percentage']
                target_pct = target_dist.get(level, 0)
                deviation = actual_pct - target_pct

                icon = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}.get(level, '⚪')

                if abs(deviation) <= 5:
                    status = "✅"
                elif abs(deviation) <= 10:
                    status = "⚠️"
                else:
                    status = "❌"

                st.markdown(f"- {icon} **{level.title()}**: {actual_pct:.1f}% (Target: {target_pct}%) {status}")

def _show_threshold_optimization_panel():
    """Show threshold optimization control panel"""
    st.markdown("### 🎛️ Threshold Optimization Control Panel")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Auto Optimize Thresholds", type="primary", use_container_width=True):
            _execute_threshold_optimization()

    with col2:
        if st.button("📊 Analyze Current Distribution", use_container_width=True):
            _analyze_current_distribution()

    with col3:
        if st.button("🎯 Reset to Default Thresholds", use_container_width=True):
            _reset_to_default_thresholds()

def _execute_threshold_optimization():
    """Execute threshold optimization"""
    try:
        with st.spinner("Optimizing thresholds..."):
            risk_results = st.session_state.four_class_risk_results
            detailed_results = risk_results.get('detailed_results', [])

            if not detailed_results:
                st.error("❌ No available risk scoring data")
                return

            # Extract risk scores
            risk_scores = [r['risk_score'] for r in detailed_results]

            # Use dynamic threshold manager for optimization
            threshold_manager = st.session_state.threshold_manager
            optimized_thresholds = threshold_manager.optimize_thresholds_iteratively(risk_scores)

            # Analyze optimization effects
            analysis = threshold_manager.analyze_distribution(risk_scores, optimized_thresholds)

            # Update session state
            st.session_state.current_thresholds = optimized_thresholds

            # Record history
            import datetime
            history_entry = {
                'timestamp': datetime.datetime.now(),
                'action': 'auto_optimization',
                'thresholds': optimized_thresholds.copy(),
                'total_deviation': analysis['total_deviation'],
                'is_reasonable': analysis['is_reasonable']
            }
            st.session_state.threshold_history.append(history_entry)

            # Display results
            if analysis['is_reasonable']:
                st.success(f"✅ Threshold optimization successful! Distribution deviation: {analysis['total_deviation']:.3f}")
            else:
                st.warning(f"⚠️ Thresholds optimized, but distribution still needs adjustment. Deviation: {analysis['total_deviation']:.3f}")

            # Display new thresholds
            st.info("**Optimized Thresholds**:")
            for level, threshold in optimized_thresholds.items():
                if level != 'critical':
                    st.write(f"- {level.title()}: {threshold:.1f}")

    except Exception as e:
        st.error(f"❌ Threshold optimization failed: {str(e)}")

def _analyze_current_distribution():
    """Analyze current distribution"""
    try:
        risk_results = st.session_state.four_class_risk_results
        distribution = risk_results.get('distribution', {})

        if not distribution:
            st.error("❌ No distribution data available")
            return

        # Calculate distribution deviation
        target_dist = {'low': 60, 'medium': 25, 'high': 12, 'critical': 3}
        total_deviation = 0

        st.markdown("#### 📊 Distribution Deviation Analysis")
        
        for level, data in distribution.items():
            actual_pct = data['percentage']
            target_pct = target_dist.get(level, 0)
            deviation = actual_pct - target_pct
            total_deviation += abs(deviation)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"{level.title()}", f"{actual_pct:.1f}%")
            with col2:
                st.metric("Target", f"{target_pct}%")
            with col3:
                st.metric("Deviation", f"{deviation:+.1f}%")
            with col4:
                if abs(deviation) <= 3:
                    st.success("✅ Good")
                elif abs(deviation) <= 8:
                    st.warning("⚠️ Fair")
                else:
                    st.error("❌ Needs Adjustment")

        # Overall assessment
        st.markdown("---")
        if total_deviation <= 10:
            st.success(f"✅ **Overall Assessment**: Good distribution (Total deviation: {total_deviation:.1f}%)")
        elif total_deviation <= 20:
            st.warning(f"⚠️ **Overall Assessment**: Fair distribution (Total deviation: {total_deviation:.1f}%)")
        else:
            st.error(f"❌ **Overall Assessment**: Distribution needs optimization (Total deviation: {total_deviation:.1f}%)")

    except Exception as e:
        st.error(f"❌ Distribution analysis failed: {str(e)}")

def _reset_to_default_thresholds():
    """Reset to default thresholds"""
    default_thresholds = {
        'low': 40,
        'medium': 60,
        'high': 80,
        'critical': 100
    }

    st.session_state.current_thresholds = default_thresholds

    # Record history
    import datetime
    history_entry = {
        'timestamp': datetime.datetime.now(),
        'action': 'reset_to_default',
        'thresholds': default_thresholds.copy(),
        'total_deviation': None,
        'is_reasonable': None
    }
    st.session_state.threshold_history.append(history_entry)

    st.success("✅ Reset to default thresholds completed")
    st.info("**Default Thresholds**: Low Risk: 40, Medium Risk: 60, High Risk: 80")

def _show_real_time_distribution_monitoring():
    """Display real-time distribution monitoring"""
    st.markdown("### 📈 Real-time Distribution Monitoring")

    risk_results = st.session_state.four_class_risk_results
    distribution = risk_results.get('distribution', {})

    if not distribution:
        st.warning("⚠️ No distribution data available")
        return

    # Create distribution comparison chart
    col1, col2 = st.columns(2)

    with col1:
        # Current distribution vs target distribution
        levels = ['low', 'medium', 'high', 'critical']
        actual_values = [distribution.get(level, {}).get('percentage', 0) for level in levels]
        target_values = [60, 25, 12, 3]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Actual Distribution',
            x=levels,
            y=actual_values,
            marker_color=['#22c55e', '#f59e0b', '#f97316', '#ef4444']
        ))
        fig.add_trace(go.Bar(
            name='Target Distribution',
            x=levels,
            y=target_values,
            marker_color=['#22c55e', '#f59e0b', '#f97316', '#ef4444'],
            opacity=0.5
        ))

        fig.update_layout(
            title="Distribution Comparison",
            xaxis_title="Risk Level",
            yaxis_title="Percentage (%)",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Deviation radar chart
        levels_en = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        deviations = [abs(actual_values[i] - target_values[i]) for i in range(4)]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=deviations,
            theta=levels_en,
            fill='toself',
            name='Distribution Deviation',
            marker_color='red'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(deviations) + 5]
                )),
            title="Distribution Deviation Radar Chart"
        )
        st.plotly_chart(fig, use_container_width=True)

def _show_threshold_adjustment_history():
    """Display threshold adjustment history"""
    st.markdown("### 📋 Threshold Adjustment History")

    if not st.session_state.threshold_history:
        st.info("💡 No threshold adjustment history available")
        return

    # Display history records
    history_df = pd.DataFrame(st.session_state.threshold_history)

    # Formatted display
    for i, record in enumerate(reversed(st.session_state.threshold_history[-10:])):  # Display last 10 records
        with st.expander(f"Adjustment Record {len(st.session_state.threshold_history) - i}: {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Action Type**: {record['action']}")
                if record['total_deviation'] is not None:
                    st.markdown(f"**Distribution Deviation**: {record['total_deviation']:.3f}")
                if record['is_reasonable'] is not None:
                    status = "✅ Reasonable" if record['is_reasonable'] else "⚠️ Needs Adjustment"
                    st.markdown(f"**Distribution Status**: {status}")

            with col2:
                st.markdown("**Threshold Settings**:")
                thresholds = record['thresholds']
                for level, value in thresholds.items():
                    if level != 'critical':
                        st.markdown(f"- {level.title()}: {value:.1f}")

    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.threshold_history = []
        st.success("✅ History cleared")
