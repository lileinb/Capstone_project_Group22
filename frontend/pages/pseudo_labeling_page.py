"""
Pseudo Label Generation Page
Provides multi-strategy pseudo label generation and quality assessment functions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import backend modules
from backend.pseudo_labeling.pseudo_label_generator import PseudoLabelGenerator
from backend.pseudo_labeling.fast_pseudo_label_generator import FastPseudoLabelGenerator

def show():
    """Display pseudo label generation page"""
    st.markdown('<div class="sub-header">🏷️ Intelligent Pseudo Label Generation System</div>', unsafe_allow_html=True)

    # Check prerequisites
    if not _check_prerequisites():
        return

    # Initialize session state
    _initialize_session_state()

    # Check prerequisites with enhanced validation
    if not _check_prerequisites():
        return

    engineered_data = st.session_state.engineered_features

    # Show system description
    _show_system_description()

    # Data overview
    _show_data_overview(engineered_data)

    # Pseudo label generation configuration
    _show_generation_config()

    # Execute pseudo label generation
    mode = st.session_state.label_generation_mode
    button_text = "🔍 Generate High-Quality Pseudo Labels (Standard Mode)" if mode == "standard" else "⚡ Quick Generate Pseudo Labels (Fast Mode)"
    button_help = "Multi-strategy integration, high-quality labels, completed in 2-3 minutes" if mode == "standard" else "Simplified algorithm, quick generation, completed within 30 seconds"

    if st.button(button_text, type="primary", help=button_help, key="main_generate_button"):
        _execute_pseudo_label_generation(engineered_data)

    # Show pseudo label results
    if st.session_state.pseudo_labels:
        _show_pseudo_label_results()

        # Label export (质量评估部分已删除)
        _show_label_export()


def _check_prerequisites():
    """Check prerequisites with enhanced dependency validation"""
    # 必需的依赖
    if 'engineered_features' not in st.session_state or st.session_state.engineered_features is None:
        st.warning("⚠️ Please complete feature engineering first!")
        st.info("💡 Please complete feature generation on the '🔧 Feature Engineering' page")
        return False

    # 可选但推荐的依赖
    missing_optional = []
    if 'clustering_results' not in st.session_state or st.session_state.clustering_results is None:
        missing_optional.append("📊 Clustering Analysis")
    if 'four_class_risk_results' not in st.session_state or st.session_state.four_class_risk_results is None:
        missing_optional.append("🎯 Risk Scoring")

    if missing_optional:
        st.info("💡 **Enhanced Mode Available**: For higher quality pseudo labels, consider completing:")
        for item in missing_optional:
            st.info(f"   • {item}")
        st.info("🚀 **Current Mode**: Fast generation (using feature-based scoring)")

        # 显示模式选择
        with st.expander("🔧 Generation Mode Selection", expanded=False):
            st.markdown("**🟢 Fast Mode (Current)**")
            st.markdown("- Uses feature-based risk scoring")
            st.markdown("- Quick generation (30 seconds)")
            st.markdown("- Good quality labels")

            st.markdown("**🟡 Enhanced Mode (Recommended)**")
            st.markdown("- Uses clustering + risk scoring results")
            st.markdown("- Higher quality labels")
            st.markdown("- Better fraud detection accuracy")
    else:
        st.success("✅ All dependencies available - Enhanced mode enabled!")

    return True


def _initialize_session_state():
    """Initialize session state"""
    if 'pseudo_labels' not in st.session_state:
        st.session_state.pseudo_labels = None
    if 'label_generator' not in st.session_state:
        st.session_state.label_generator = PseudoLabelGenerator()
    if 'fast_label_generator' not in st.session_state:
        st.session_state.fast_label_generator = FastPseudoLabelGenerator()
    if 'high_quality_labels' not in st.session_state:
        st.session_state.high_quality_labels = None
    if 'label_generation_mode' not in st.session_state:
        st.session_state.label_generation_mode = "standard"


def _show_system_description():
    """Display system description"""
    with st.expander("📖 Intelligent Pseudo Label Generation System Description", expanded=False):
        st.markdown("""
        ### 🎯 System Features
        - **Unsupervised Driven**: Generate pseudo labels based on clustering analysis and unsupervised risk scoring
        - **Multi-strategy Integration**: Integrate risk scoring, clustering analysis, and expert rules
        - **Quality First**: Automatically filter high-confidence labels to ensure label quality
        - **Intelligent Calibration**: Optionally use a small amount of real labels for calibration optimization

        ### 📊 Generation Strategies
        1. **Unsupervised Risk Scoring** (45%): Based on cluster anomaly and feature deviation
        2. **Cluster Risk Mapping** (35%): Based on cluster quality and risk level
        3. **Expert Business Rules** (20%): Rule matching based on domain knowledge

        ### 🔧 Quality Control
        - **Dynamic Weights**: Automatically adjust weights based on strategy quality
        - **Confidence Filtering**: Only retain high-confidence pseudo labels
        - **Consistency Verification**: Higher multi-strategy consistency leads to higher confidence
        - **Balance Optimization**: Automatically adjust label distribution to avoid extreme imbalance
        """)


def _show_data_overview(engineered_data):
    """Display data overview"""
    st.markdown("### 📊 Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Samples", f"{len(engineered_data):,}")

    with col2:
        if 'is_fraudulent' in engineered_data.columns:
            true_fraud_rate = round(engineered_data['is_fraudulent'].sum() / len(engineered_data) * 100, 2)
            st.metric("True Fraud Rate", f"{true_fraud_rate}%")
        else:
            st.metric("True Fraud Rate", "N/A")

    with col3:
        st.metric("Feature Count", f"{len(engineered_data.columns)}")

    with col4:
        # Check if unsupervised risk scoring results exist
        if st.session_state.get('unsupervised_risk_results'):
            avg_risk = st.session_state.unsupervised_risk_results.get('average_risk_score', 0)
            st.metric("Average Risk Score", f"{avg_risk:.1f}")
        else:
            st.metric("Average Risk Score", "To be calculated")


def _show_generation_config():
    """Show generation configuration"""
    st.markdown("### ⚙️ Pseudo Label Generation Configuration")

    # Generation mode selection
    st.markdown("#### 🎯 Generation Mode Selection")

    col_mode1, col_mode2 = st.columns(2)

    with col_mode1:
        if st.button("🔍 Standard Mode", use_container_width=True,
                    help="Complete strategy integration, high-quality labels, completed in 2-3 minutes", key="mode_standard"):
            st.session_state.label_generation_mode = "standard"

    with col_mode2:
        if st.button("⚡ Fast Mode", use_container_width=True,
                    help="Simplified algorithm, quick generation, completed within 30 seconds", key="mode_fast"):
            st.session_state.label_generation_mode = "fast"

    # Show current mode
    mode = st.session_state.label_generation_mode
    if mode == "standard":
        st.success("🔍 **Current Mode: Standard Mode** - Multi-strategy integration, high-quality labels")
    else:
        st.info("⚡ **Current Mode: Fast Mode** - Simplified algorithm, quick generation")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Quality Control Parameters**")
        min_confidence = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.55,  # 进一步降低默认值到0.55
            step=0.05,
            help="Only retain pseudo labels with confidence above this threshold"
        )

        use_calibration = st.checkbox(
            "Enable Calibration Optimization",
            value=True,
            help="Use a small amount of real labels to calibrate risk scoring thresholds"
        )

        balance_labels = st.checkbox(
            "Label Balance Optimization",
            value=True,
            help="Automatically adjust label distribution to avoid extreme imbalance"
        )

    with col2:
        st.markdown("**Strategy Weight Configuration**")

        # Show current weight configuration
        current_weights = {
            "Unsupervised Risk Scoring": 45,
            "Cluster Risk Mapping": 35,
            "Expert Business Rules": 20
        }

        for strategy, weight in current_weights.items():
            st.write(f"- {strategy}: {weight}%")

        st.info("💡 Weights will be dynamically adjusted based on actual quality of each strategy")

    # Save configuration to session state
    st.session_state.label_config = {
        'min_confidence': min_confidence,
        'use_calibration': use_calibration,
        'balance_labels': balance_labels
    }
    
    # 高级配置（仅标准模式显示）
    if mode == "standard":
        st.markdown("#### 🔧 Advanced Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        strategy = st.selectbox(
            "Select Label Generation Strategy",
            options=['ensemble', 'risk_based', 'cluster_based', 'rule_based'],
            format_func=lambda x: {
                'ensemble': '🎯 Ensemble Strategy (Recommended)',
                'risk_based': '📊 Risk Score Based',
                'cluster_based': '🔍 Cluster Analysis Based',
                'rule_based': '📋 Expert Rules Based'
            }[x],
            help="Select pseudo label generation strategy"
        )

    with col2:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Only retain labels with confidence above this threshold"
        )
    
    # 策略说明
    strategy_descriptions = {
        'ensemble': """
        **🎯 Ensemble Strategy**
        - Combines risk scoring, clustering analysis, and expert rules
        - Uses weighted voting mechanism
        - Provides highest label quality and stability
        """,
        'risk_based': """
        **📊 Risk Score Based**
        - Generate labels based on multi-dimensional risk scores
        - High risk score (>70) → Fraud label
        - Suitable for scenarios with clear risk thresholds
        """,
        'cluster_based': """
        **🔍 Clustering Based**
        - Generate labels based on cluster fraud rates
        - High fraud rate clusters → Fraud labels
        - Suitable for discovering hidden fraud patterns
        """,
        'rule_based': """
        **📋 Expert Rules Based**
        - Based on business expert experience rules
        - Includes time, amount, device and other rules
        - Suitable for scenarios with clear business logic
        """
    }
    
    st.markdown(strategy_descriptions[strategy])
    
    # 执行伪标签生成
    col1, col2 = st.columns([3, 1])

    with col1:
        generate_labels = st.button("🚀 Generate Pseudo Labels", type="primary", help="Generate pseudo labels based on selected strategy", key="generate_labels_standard")

    with col2:
        if st.button("🗑️ Clear Results", help="Clear previous generation results", key="clear_results"):
            st.session_state.pseudo_labels = None
            st.success("✅ Results cleared!")
            st.rerun()
    
    if generate_labels:
        try:
            # 获取工程化特征数据
            engineered_data = st.session_state.engineered_features
            if engineered_data is None or engineered_data.empty:
                st.error("❌ Please complete feature engineering first!")
                return

            with st.spinner("Generating pseudo labels..."):
                # Update confidence threshold
                st.session_state.label_generator.update_confidence_threshold(confidence_threshold)

                # Generate pseudo labels
                pseudo_results = st.session_state.label_generator.generate_pseudo_labels(
                    engineered_data, strategy=strategy
                )

                # Save results
                st.session_state.pseudo_labels = pseudo_results

                st.success("✅ Pseudo label generation completed!")

        except Exception as e:
            st.error(f"❌ Pseudo label generation failed: {e}")
            st.exception(e)
    
    # Display pseudo label results
    if st.session_state.pseudo_labels is not None:
        st.markdown("### 📈 Pseudo Label Generation Results")
        
        pseudo_results = st.session_state.pseudo_labels
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Generation Strategy", pseudo_results['strategy'].upper())

        with col2:
            # Compatible with different mode data structures
            all_labels = pseudo_results.get('all_labels', pseudo_results.get('labels', []))
            total_labels = len(all_labels)
            st.metric("Total Labels", f"{total_labels:,}")

        with col3:
            # Compatible with different confidence fields
            if 'metadata' in pseudo_results:
                avg_confidence = pseudo_results['metadata'].get('avg_confidence_all', 0)
            else:
                avg_confidence = pseudo_results.get('avg_confidence', 0)
            st.metric("Average Confidence", f"{avg_confidence:.3f}")

        with col4:
            # Compatible with different high confidence count fields
            if 'metadata' in pseudo_results:
                high_conf_count = pseudo_results['metadata'].get('high_quality_count', 0)
            else:
                high_conf_count = pseudo_results.get('high_confidence_count', 0)
            high_conf_rate = high_conf_count / total_labels * 100 if total_labels > 0 else 0
            st.metric("High Confidence Ratio", f"{high_conf_rate:.1f}%")
        
        # Label distribution
        st.markdown("#### 📊 Label Distribution Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Label distribution pie chart
            # Compatible with different mode data structures
            all_labels = pseudo_results.get('all_labels', pseudo_results.get('labels', []))
            if all_labels:
                label_counts = pd.Series(all_labels).value_counts()

                fig = px.pie(
                    values=label_counts.values,
                    names=['Normal Transaction', 'Fraud Transaction'],
                    title="Pseudo Label Distribution",
                    color_discrete_map={
                        'Normal Transaction': '#2E8B57',
                        'Fraud Transaction': '#DC143C'
                    }
                )
                st.plotly_chart(fig, use_container_width=True, key="label_distribution_chart")
            else:
                st.warning("⚠️ No label data to display")

        with col2:
            # Confidence distribution histogram
            # Compatible with different confidence fields
            confidences = pseudo_results.get('all_confidences', pseudo_results.get('confidences', []))

            if confidences:
                fig = px.histogram(
                    x=confidences,
                    title="Confidence Distribution",
                    nbins=20,
                    labels={'x': 'Confidence', 'y': 'Frequency'}
                )
                fig.add_vline(x=confidence_threshold, line_dash="dash", line_color="red",
                             annotation_text=f"Threshold: {confidence_threshold}")
                st.plotly_chart(fig, use_container_width=True, key="confidence_distribution_chart")
            else:
                st.warning("⚠️ No confidence data to display")
        
        # 质量评估部分已删除 - 专注于标签生成和导出

        # Next step button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if st.button("🤖 Next: Model Training", type="primary", use_container_width=True, key="next_model_training_1"):
                st.success("✅ Pseudo label generation completed, ready for model training!")
                st.info("💡 Please select '🤖 Model Training' page in the sidebar to continue")


def _execute_pseudo_label_generation(engineered_data):
    """Execute pseudo label generation"""
    try:
        config = st.session_state.label_config
        mode = st.session_state.label_generation_mode

        mode_text = "Standard Mode" if mode == "standard" else "Fast Mode"
        mode_icon = "🔍" if mode == "standard" else "⚡"

        with st.spinner(f"Generating pseudo labels using {mode_text}..."):
            # Record start time
            import time
            start_time = time.time()

            # Select generator based on mode
            if mode == "standard":
                # Use standard mode generator
                label_results = st.session_state.label_generator.generate_high_quality_pseudo_labels(
                    engineered_data,
                    min_confidence=config['min_confidence'],
                    use_calibration=config['use_calibration']
                )
            else:
                # Use fast mode generator
                risk_results = st.session_state.get('unsupervised_risk_results', None)
                label_results = st.session_state.fast_label_generator.generate_fast_pseudo_labels(
                    engineered_data,
                    risk_results=risk_results,
                    min_confidence=config['min_confidence']
                )

            # Record end time
            end_time = time.time()
            generation_time = end_time - start_time

            st.session_state.pseudo_labels = label_results
            st.session_state.high_quality_labels = label_results

            # 检查是否有标签生成（即使高质量标签很少也显示结果）
            if label_results and (label_results.get('high_quality_labels') or label_results.get('all_labels')):
                total_labels = len(label_results.get('all_labels', []))
                hq_labels = len(label_results.get('high_quality_labels', []))

                if hq_labels > 0:
                    success_msg = f"✅ {mode_icon} {mode_text} pseudo label generation completed!"
                    success_msg += f" Filtered {hq_labels} high-quality labels from {total_labels} samples, time taken: {generation_time:.2f} seconds"
                    st.success(success_msg)
                else:
                    warning_msg = f"⚠️ {mode_icon} {mode_text} pseudo label generation completed with low confidence!"
                    warning_msg += f" Generated {total_labels} labels but none met the high-quality threshold. Consider lowering the confidence threshold."
                    st.warning(warning_msg)

                # Show basic statistics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    hq_rate = label_results['metadata'].get('high_quality_rate', 0)
                    st.metric("High Quality Ratio", f"{hq_rate:.1%}")

                with col2:
                    # 如果没有高质量标签，显示所有标签的平均置信度
                    if hq_labels > 0:
                        avg_conf = label_results['metadata'].get('avg_confidence_hq', 0)
                        st.metric("Avg Confidence (HQ)", f"{avg_conf:.3f}")
                    else:
                        avg_conf = label_results['metadata'].get('avg_confidence_all', 0)
                        st.metric("Avg Confidence (All)", f"{avg_conf:.3f}")

                with col3:
                    # 如果没有高质量标签，显示所有标签的欺诈率
                    if hq_labels > 0:
                        fraud_rate = label_results['metadata'].get('fraud_rate_hq', 0)
                        st.metric("Fraud Rate (HQ)", f"{fraud_rate:.1%}")
                    else:
                        fraud_rate = label_results['metadata'].get('fraud_rate_all', 0)
                        st.metric("Fraud Rate (All)", f"{fraud_rate:.1%}")

                with col4:
                    quality_score = label_results['quality_report'].get('quality_score', 0)
                    st.metric("Quality Score", f"{quality_score:.1f}")

                # Show calibration status
                if label_results.get('calibration_applied'):
                    st.info("✅ Calibration optimization applied, risk score thresholds optimized")
                elif config['use_calibration']:
                    st.warning("⚠️ Calibration not successfully applied, using default thresholds")

            else:
                # 提供更详细的错误信息
                if label_results:
                    total_labels = len(label_results.get('all_labels', []))
                    if total_labels > 0:
                        st.error(f"❌ Generated {total_labels} labels but failed quality checks. Please lower confidence threshold or check data quality.")
                    else:
                        st.error("❌ No labels were generated. Please check your data and configuration.")
                else:
                    st.error("❌ Pseudo label generation failed completely. Please check logs for details.")

    except Exception as e:
        st.error(f"❌ Pseudo label generation failed: {str(e)}")


def _show_pseudo_label_results():
    """Show pseudo label results"""
    st.markdown("### 📈 High-Quality Pseudo Label Results")

    label_results = st.session_state.pseudo_labels

    # Results overview
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Label Distribution Comparison**")

        all_labels = label_results.get('all_labels', [])
        hq_labels = label_results.get('high_quality_labels', [])

        # Create comparison data
        all_dist = pd.Series(all_labels).value_counts()
        hq_dist = pd.Series(hq_labels).value_counts() if hq_labels else pd.Series()

        # Ensure all possible label categories are included
        all_dist = all_dist.reindex([0, 1], fill_value=0)
        hq_dist = hq_dist.reindex([0, 1], fill_value=0)

        comparison_data = pd.DataFrame({
            'All Labels': all_dist,
            'High Quality Labels': hq_dist
        }).fillna(0)

        comparison_data.index = ['Normal', 'Fraud']

        fig = px.bar(
            comparison_data,
            title="Label Distribution Comparison",
            labels={'index': 'Label Type', 'value': 'Count'},
            color_discrete_map={'All Labels': '#17a2b8', 'High Quality Labels': '#28a745'}
        )
        st.plotly_chart(fig, use_container_width=True, key="label_comparison_chart")

    with col2:
        st.markdown("**Confidence Distribution**")

        all_confidences = label_results.get('all_confidences', [])
        hq_confidences = label_results.get('high_quality_confidences', [])

        if all_confidences:
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=all_confidences,
                name='All Labels',
                opacity=0.7,
                nbinsx=20
            ))

            if hq_confidences:
                fig.add_trace(go.Histogram(
                    x=hq_confidences,
                    name='High Quality Labels',
                    opacity=0.7,
                    nbinsx=20
                ))

            fig.add_vline(
                x=label_results['min_confidence_threshold'],
                line_dash="dash",
                line_color="red",
                annotation_text="Confidence Threshold"
            )

            fig.update_layout(
                title="Confidence Distribution Comparison",
                xaxis_title="Confidence",
                yaxis_title="Frequency",
                barmode='overlay'
            )

            st.plotly_chart(fig, use_container_width=True, key="confidence_comparison_chart")


# 质量评估部分已删除 - 专注于伪标签生成和导出功能

def _show_label_export():
    """Display label export"""
    st.markdown("### 📥 Label Export & Application")

    label_results = st.session_state.pseudo_labels
    engineered_data = st.session_state.engineered_features

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Export Options**")

        export_option = st.radio(
            "Select export content",
            ["High-quality labels only", "All labels", "Label comparison report"]
        )

        include_features = st.checkbox("Include feature data", value=True)
        include_confidence = st.checkbox("Include confidence", value=True)

    with col2:
        st.markdown("**Export Statistics**")

        if export_option == "High Quality Labels Only":
            export_count = len(label_results.get('high_quality_labels', []))
            st.write(f"Export sample count: {export_count:,}")
        elif export_option == "All Labels":
            export_count = len(label_results.get('all_labels', []))
            st.write(f"Export sample count: {export_count:,}")
        else:
            export_count = len(label_results.get('all_labels', []))
            st.write(f"Report sample count: {export_count:,}")

    # Generate export data
    if st.button("📥 Generate Export File", type="secondary", key="generate_export_file"):
        try:
            if export_option == "High Quality Labels Only":
                export_data = _prepare_high_quality_export(label_results, engineered_data, include_features, include_confidence)
                filename = f"high_quality_pseudo_labels_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            elif export_option == "All Labels":
                export_data = _prepare_all_labels_export(label_results, engineered_data, include_features, include_confidence)
                filename = f"all_pseudo_labels_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            else:
                export_data = _prepare_comparison_report(label_results, engineered_data)
                filename = f"pseudo_labels_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

            csv_data = export_data.to_csv(index=False)

            st.download_button(
                label=f"📥 Download {filename}",
                data=csv_data,
                file_name=filename,
                mime="text/csv"
            )

            st.success(f"✅ Export file prepared successfully, contains {len(export_data)} records")

        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")

    # Next step button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("🤖 Next: Model Training", type="primary", use_container_width=True, key="next_model_training_2"):
            st.success("✅ High-quality pseudo label generation completed, ready for model training!")
            st.info("💡 Please select '🤖 Model Training' page in the sidebar to continue")


def _prepare_high_quality_export(label_results, engineered_data, include_features, include_confidence):
    """Prepare high-quality label export data"""
    hq_indices = label_results.get('high_quality_indices', [])
    hq_labels = label_results.get('high_quality_labels', [])
    hq_confidences = label_results.get('high_quality_confidences', [])

    # Basic data
    export_data = pd.DataFrame({
        'sample_index': hq_indices,
        'pseudo_label': hq_labels
    })

    if include_confidence:
        export_data['confidence'] = hq_confidences

    if include_features:
        # Add feature data
        feature_data = engineered_data.iloc[hq_indices].reset_index(drop=True)
        export_data = pd.concat([export_data, feature_data], axis=1)

    return export_data


def _prepare_all_labels_export(label_results, engineered_data, include_features, include_confidence):
    """Prepare all labels export data"""
    all_labels = label_results.get('all_labels', [])
    all_confidences = label_results.get('all_confidences', [])

    # Basic data
    export_data = pd.DataFrame({
        'sample_index': range(len(all_labels)),
        'pseudo_label': all_labels
    })

    if include_confidence:
        export_data['confidence'] = all_confidences

    # Mark high-quality labels
    hq_indices = set(label_results.get('high_quality_indices', []))
    export_data['is_high_quality'] = export_data['sample_index'].isin(hq_indices)

    if include_features:
        # Add feature data
        export_data = pd.concat([export_data, engineered_data.reset_index(drop=True)], axis=1)

    return export_data


def _prepare_comparison_report(label_results, engineered_data):
    """Prepare comparison report"""
    all_labels = label_results.get('all_labels', [])
    all_confidences = label_results.get('all_confidences', [])
    hq_indices = set(label_results.get('high_quality_indices', []))

    # Basic report data
    report_data = pd.DataFrame({
        'sample_index': range(len(all_labels)),
        'pseudo_label': all_labels,
        'confidence': all_confidences,
        'is_high_quality': [i in hq_indices for i in range(len(all_labels))]
    })

    # Add key features
    key_features = ['transaction_id', 'customer_id', 'transaction_amount', 'customer_age', 'account_age_days']
    available_features = [f for f in key_features if f in engineered_data.columns]

    if available_features:
        report_data = pd.concat([
            report_data,
            engineered_data[available_features].reset_index(drop=True)
        ], axis=1)

    # Add true label comparison (if available)
    if 'is_fraudulent' in engineered_data.columns:
        report_data['true_label'] = engineered_data['is_fraudulent'].reset_index(drop=True)
        report_data['label_match'] = report_data['pseudo_label'] == report_data['true_label']

    return report_data
