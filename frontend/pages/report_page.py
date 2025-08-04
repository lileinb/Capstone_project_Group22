"""
Analysis Report Page
Responsible for generating comprehensive analysis reports and explainability analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import backend modules
from backend.explainer.shap_explainer import SHAPExplainer
from backend.explainer.lime_explainer import LIMEExplainer
from backend.analysis_reporting.report_generator import ReportGenerator

def show():
    """Display analysis report page"""
    st.markdown('<div class="sub-header">📋 Comprehensive Analysis Report & Explainability Analysis</div>', unsafe_allow_html=True)

    # Check if feature engineering data exists
    if 'engineered_features' not in st.session_state or st.session_state.engineered_features is None:
        st.warning("⚠️ Please complete feature engineering first!")
        st.info("💡 Please complete feature generation on the '🔧 Feature Engineering' page")
        return

    # Initialize session state
    if 'shap_analysis' not in st.session_state:
        st.session_state.shap_analysis = None
    if 'lime_analysis' not in st.session_state:
        st.session_state.lime_analysis = None
    if 'report_data' not in st.session_state:
        st.session_state.report_data = None

    # Get feature engineering data
    engineered_data = st.session_state.engineered_features
    
    st.markdown("### 📊 Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Records", f"{len(engineered_data):,}")
    
    with col2:
        st.metric("Feature Count", f"{len(engineered_data.columns)}")

    with col3:
        numeric_features = len(engineered_data.select_dtypes(include=['number']).columns)
        st.metric("Numeric Features", f"{numeric_features}")

    with col4:
        if 'is_fraudulent' in engineered_data.columns:
            fraud_rate = (engineered_data['is_fraudulent'].sum() / len(engineered_data) * 100).round(2)
            st.metric("Fraud Rate", f"{fraud_rate}%")
        else:
            st.metric("Fraud Rate", "N/A")

    # Report configuration
    st.markdown("### ⚙️ Report Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📋 Report Content Selection")

        # Report content selection
        include_data_analysis = st.checkbox("Data Analysis Report", value=True)
        include_model_performance = st.checkbox("Model Performance Report", value=True)
        include_risk_assessment = st.checkbox("Risk Assessment Report", value=True)
        include_attack_analysis = st.checkbox("Attack Analysis Report", value=True)
        include_explainability = st.checkbox("Explainability Analysis", value=True)
        include_recommendations = st.checkbox("Protection Recommendations", value=True)

    with col2:
        st.markdown("#### 🎯 Explainability Analysis")

        # Explainability analysis selection
        include_shap_analysis = st.checkbox("SHAP Analysis", value=True)
        include_lime_analysis = st.checkbox("LIME Analysis", value=True)

        # Analysis sample size
        analysis_sample_size = st.slider(
            "Analysis Sample Size", 100, 1000, 500,
            help="Number of samples for explainability analysis"
        )

        # Feature importance threshold
        feature_importance_threshold = st.slider(
            "Feature Importance Threshold", 0.01, 0.1, 0.05, 0.01,
            help="Minimum threshold for displaying feature importance"
        )
    
    # Report format selection
    st.markdown("#### 📄 Report Format")

    col1, col2, col3 = st.columns(3)

    with col1:
        export_pdf = st.checkbox("PDF Format", value=True)

    with col2:
        export_excel = st.checkbox("Excel Format", value=True)

    with col3:
        export_html = st.checkbox("HTML Format", value=True)

    # Execute analysis
    if st.button("🚀 Generate Analysis Report", type="primary", help="Generate comprehensive analysis report based on current configuration"):
        try:
            with st.spinner("Generating analysis report..."):
                # Prepare data
                X = engineered_data.select_dtypes(include=['number'])
                if 'is_fraudulent' in X.columns:
                    y = X['is_fraudulent']
                    X = X.drop('is_fraudulent', axis=1)
                    has_labels = True
                else:
                    y = None
                    has_labels = False

                # Handle missing values
                X = X.fillna(0)

                # SHAP analysis
                if include_shap_analysis and has_labels:
                    try:
                        shap_explainer = SHAPExplainer()
                        shap_analysis = shap_explainer.analyze(X, y, sample_size=analysis_sample_size)
                        st.session_state.shap_analysis = shap_analysis
                        st.success("✅ SHAP analysis completed")
                    except Exception as e:
                        st.warning(f"⚠️ SHAP analysis failed: {e}")

                # LIME analysis
                if include_lime_analysis and has_labels:
                    try:
                        lime_explainer = LIMEExplainer()
                        lime_analysis = lime_explainer.analyze(X, y, sample_size=analysis_sample_size)
                        st.session_state.lime_analysis = lime_analysis
                        st.success("✅ LIME analysis completed")
                    except Exception as e:
                        st.warning(f"⚠️ LIME analysis failed: {e}")

                # Generate report data
                report_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'total_samples': len(engineered_data),
                    'fraud_samples': int(engineered_data['is_fraudulent'].sum()) if 'is_fraudulent' in engineered_data.columns else 0,
                    'fraud_rate': fraud_rate if 'is_fraudulent' in engineered_data.columns else 0,
                    'data_info': {
                        'total_records': len(engineered_data),
                        'total_features': len(engineered_data.columns),
                        'numeric_features': len(engineered_data.select_dtypes(include=['number']).columns),
                        'fraud_rate': fraud_rate if 'is_fraudulent' in engineered_data.columns else None
                    },
                    'analysis_config': {
                        'include_data_analysis': include_data_analysis,
                        'include_model_performance': include_model_performance,
                        'include_risk_assessment': include_risk_assessment,
                        'include_attack_analysis': include_attack_analysis,
                        'include_explainability': include_explainability,
                        'include_recommendations': include_recommendations,
                        'include_shap_analysis': include_shap_analysis,
                        'include_lime_analysis': include_lime_analysis,
                        'analysis_sample_size': analysis_sample_size,
                        'feature_importance_threshold': feature_importance_threshold
                    },
                    'export_formats': {
                        'pdf': export_pdf,
                        'excel': export_excel,
                        'html': export_html
                    }
                }
                
                # Add other analysis results
                if 'clustering_results' in st.session_state:
                    report_data['clustering_results'] = st.session_state.clustering_results

                if 'risk_scores' in st.session_state:
                    report_data['risk_analysis'] = st.session_state.risk_analysis

                if 'model_predictions' in st.session_state:
                    report_data['model_performance'] = st.session_state.model_performance

                if 'attack_results' in st.session_state:
                    report_data['attack_analysis'] = st.session_state.attack_analysis

                st.session_state.report_data = report_data

                st.success("✅ Analysis report generation completed!")

        except Exception as e:
            st.error(f"❌ Analysis report generation failed: {e}")
            st.exception(e)
    
    # Display analysis results
    if st.session_state.report_data is not None:
        st.markdown("### 📈 Analysis Results")

        report_data = st.session_state.report_data

        # Report overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Analysis Time", report_data['timestamp'])

        with col2:
            st.metric("Total Records", f"{report_data['total_samples']:,}")

        with col3:
            st.metric("Fraud Records", f"{report_data['fraud_samples']:,}")

        with col4:
            fraud_rate = (report_data['fraud_samples'] / report_data['total_samples'] * 100) if report_data['total_samples'] > 0 else 0
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        
        # SHAP analysis results
        if st.session_state.shap_analysis is not None:
            st.markdown("#### 🎯 SHAP Interpretability Analysis")

            shap_analysis = st.session_state.shap_analysis

            # Feature importance
            if 'feature_importance' in shap_analysis:
                importance_df = pd.DataFrame(shap_analysis['feature_importance'])
                importance_df = importance_df.sort_values('importance', ascending=False)

                # Show top 10 important features
                top_features = importance_df.head(10)

                fig = px.bar(
                    top_features,
                    x='feature',
                    y='importance',
                    title="SHAP Feature Importance Ranking",
                    labels={'feature': 'Feature', 'importance': 'Importance'}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Feature importance table
                st.dataframe(top_features, use_container_width=True)
            
            # Feature contribution analysis
            if 'feature_contributions' in shap_analysis:
                st.markdown("**Feature Contribution Analysis**")

                # Select sample to analyze
                if len(shap_analysis['feature_contributions']) > 0:
                    sample_index = st.selectbox(
                        "Select sample to view SHAP contributions",
                        range(len(shap_analysis['feature_contributions'])),
                        format_func=lambda x: f"Sample {x+1}"
                    )

                    if 0 <= sample_index < len(shap_analysis['feature_contributions']):
                        sample_contributions = shap_analysis['feature_contributions'][sample_index]

                        # Create waterfall chart
                        contrib_df = pd.DataFrame(sample_contributions)
                        contrib_df = contrib_df.sort_values('contribution', ascending=False)

                        fig = px.bar(
                            contrib_df,
                            x='feature',
                            y='contribution',
                            title=f"Sample {sample_index+1} SHAP Feature Contributions",
                            labels={'feature': 'Feature', 'contribution': 'Contribution Value'},
                            color='contribution',
                            color_continuous_scale='RdBu'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # LIME analysis results
        if st.session_state.lime_analysis is not None:
            st.markdown("#### 🔍 LIME Local Explanation Analysis")

            lime_analysis = st.session_state.lime_analysis

            # Local explanations
            if 'local_explanations' in lime_analysis:
                st.markdown("**Local Explanation Analysis**")

                # Select sample to analyze
                if len(lime_analysis['local_explanations']) > 0:
                    lime_sample_index = st.selectbox(
                        "Select sample to view LIME explanation",
                        range(len(lime_analysis['local_explanations'])),
                        format_func=lambda x: f"Sample {x+1}"
                    )

                    if 0 <= lime_sample_index < len(lime_analysis['local_explanations']):
                        local_explanation = lime_analysis['local_explanations'][lime_sample_index]

                        # Show local explanation
                        st.markdown(f"**LIME Explanation for Sample {lime_sample_index+1}**")

                        if 'feature_weights' in local_explanation:
                            weights_df = pd.DataFrame(local_explanation['feature_weights'])
                            weights_df = weights_df.sort_values('weight', ascending=False)

                            fig = px.bar(
                                weights_df,
                                x='feature',
                                y='weight',
                                title=f"Sample {lime_sample_index+1} LIME Feature Weights",
                                labels={'feature': 'Feature', 'weight': 'Weight'},
                                color='weight',
                                color_continuous_scale='RdBu'
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        # Comprehensive analysis
        st.markdown("#### 📊 Comprehensive Analysis")

        # Create comprehensive analysis charts
        analysis_summary = []

        # Data quality analysis
        if 'data_info' in report_data:
            data_info = report_data['data_info']
            analysis_summary.append({
                'Analysis Item': 'Data Quality',
                'Record Count': data_info['total_records'],
                'Feature Count': data_info['total_features'],
                'Numeric Features': data_info['numeric_features'],
                'Fraud Rate': data_info['fraud_rate']
            })
        
        # Clustering analysis
        if 'clustering_results' in report_data:
            clustering_results = report_data['clustering_results']
            analysis_summary.append({
                'Analysis Item': 'Clustering Analysis',
                'Algorithm': clustering_results.get('algorithm', 'N/A'),
                'Cluster Count': clustering_results.get('n_clusters', 'N/A'),
                'Feature Count': len(clustering_results.get('features', [])),
                'Record Count': len(st.session_state.cluster_labels) if 'cluster_labels' in st.session_state else 'N/A'
            })

        # Risk scoring analysis
        if 'risk_analysis' in report_data:
            risk_analysis = report_data['risk_analysis']
            analysis_summary.append({
                'Analysis Item': 'Risk Scoring',
                'Average Score': f"{risk_analysis['score_stats']['mean']:.2f}",
                'Standard Deviation': f"{risk_analysis['score_stats']['std']:.2f}",
                'Highest Score': f"{risk_analysis['score_stats']['max']:.2f}",
                'Lowest Score': f"{risk_analysis['score_stats']['min']:.2f}"
            })

        # Model performance analysis
        if 'model_performance' in report_data:
            model_performance = report_data['model_performance']
            if model_performance:
                best_model = max(model_performance.items(), key=lambda x: x[1].get('accuracy', 0))[0]
                best_accuracy = model_performance[best_model].get('accuracy', 0)
                analysis_summary.append({
                    'Analysis Item': 'Model Performance',
                    'Best Model': best_model,
                    'Best Accuracy': f"{best_accuracy:.3f}",
                    'Model Count': len(model_performance),
                    'Average Accuracy': f"{np.mean([p.get('accuracy', 0) for p in model_performance.values()]):.3f}"
                })
        
        # Attack analysis
        if 'attack_analysis' in report_data:
            attack_analysis = report_data['attack_analysis']

            # Safely get confidence data
            try:
                if 'attack_results' in st.session_state and st.session_state.attack_results:
                    attack_results = st.session_state.attack_results
                    if isinstance(attack_results, dict):
                        attack_data = attack_results.get('attack_predictions', [])
                    elif isinstance(attack_results, list):
                        attack_data = attack_results
                    else:
                        attack_data = []

                    if attack_data and isinstance(attack_data, list):
                        confidences = [r.get('confidence', 0) for r in attack_data if isinstance(r, dict) and 'confidence' in r]
                        avg_confidence = np.mean(confidences) if confidences else 0
                    else:
                        avg_confidence = 0
                else:
                    avg_confidence = 0
            except Exception:
                avg_confidence = 0

            # Safely access attack_analysis data
            if attack_analysis and isinstance(attack_analysis, dict):
                analysis_summary.append({
                    'Analysis Item': 'Attack Detection',
                    'Detected Attacks': attack_analysis.get('total_attacks', 0),
                    'Attack Types': len(attack_analysis.get('attack_types', {})),
                    'High Risk Rate': f"{attack_analysis.get('severity_distribution', {}).get('high', 0) / max(attack_analysis.get('total_attacks', 1), 1) * 100:.2f}%",
                    'Average Confidence': f"{avg_confidence:.3f}"
                })
            else:
                # Add placeholder when attack analysis is not available or invalid
                analysis_summary.append({
                    'Analysis Item': 'Attack Detection',
                    'Detected Attacks': 'Not Available',
                    'Attack Types': 'Not Available',
                    'High Risk Rate': 'Not Available',
                    'Average Confidence': 'Not Available'
                })
        
        # Show comprehensive analysis table
        if analysis_summary:
            summary_df = pd.DataFrame(analysis_summary)
            st.dataframe(summary_df, use_container_width=True)
        
        # Report export
        st.markdown("#### 📄 Report Export")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Export PDF Report", type="primary", use_container_width=True):
                if report_data is None:
                    st.error("❌ Please generate analysis report first")
                else:
                    try:
                        report_generator = ReportGenerator()
                        pdf_path = report_generator.generate_pdf_report(report_data)
                        st.success(f"✅ PDF report generated: {pdf_path}")

                        # Provide download link
                        if os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as file:
                                st.download_button(
                                    label="📥 Download PDF Report",
                                    data=file.read(),
                                    file_name="fraud_detection_report.pdf",
                                    mime="application/pdf"
                                )
                        else:
                            st.warning("⚠️ PDF file generation failed")
                    except Exception as e:
                        st.error(f"❌ PDF report generation failed: {str(e)}")
                        st.info("💡 PDF report feature is under development, please try other formats")

        with col2:
            if st.button("📊 Export Excel Report", type="primary", use_container_width=True):
                if report_data is None:
                    st.error("❌ Please generate analysis report first")
                else:
                    try:
                        report_generator = ReportGenerator()
                        excel_path = report_generator.generate_excel_report(report_data)
                        st.success(f"✅ Excel report generated: {excel_path}")

                        # Provide download link
                        if os.path.exists(excel_path):
                            with open(excel_path, "rb") as file:
                                st.download_button(
                                    label="📥 Download Excel Report",
                                    data=file.read(),
                                    file_name="fraud_detection_report.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        else:
                            st.warning("⚠️ Excel file generation failed")
                    except Exception as e:
                        st.error(f"❌ Excel report generation failed: {str(e)}")

        with col3:
            if st.button("📊 Export HTML Report", type="primary", use_container_width=True):
                if report_data is None:
                    st.error("❌ Please generate analysis report first")
                else:
                    try:
                        report_generator = ReportGenerator()
                        html_path = report_generator.generate_html_report(report_data)
                        st.success(f"✅ HTML report generated: {html_path}")

                        # Provide download link
                        if os.path.exists(html_path):
                            with open(html_path, "r", encoding="utf-8") as file:
                                st.download_button(
                                    label="📥 Download HTML Report",
                                    data=file.read(),
                                    file_name="fraud_detection_report.html",
                                    mime="text/html"
                                )
                        else:
                            st.warning("⚠️ HTML file generation failed")
                    except Exception as e:
                        st.error(f"❌ HTML report generation failed: {str(e)}")
        
        # Report completion
        st.markdown("---")
        st.success("🎉 Congratulations! You have completed the entire e-commerce fraud risk prediction system analysis process!")
        st.markdown("""
        **Analysis Process Summary:**
        1. ✅ Data upload and preprocessing
        2. ✅ Feature engineering and risk feature generation
        3. ✅ Clustering analysis and anomalous group identification
        4. ✅ Risk scoring and level classification
        5. ✅ Multi-model prediction and performance comparison
        6. ✅ Attack type classification and protection recommendations
        7. ✅ Comprehensive analysis report and explainability analysis

        **System Features:**
        - Multi-dimensional risk assessment
        - Intelligent feature engineering
        - Multi-model ensemble prediction
        - Attack type identification
        - Explainability analysis
        - Comprehensive report generation
        """)
    
    else:
        # Show analysis report description
        st.markdown("### 📝 Analysis Report Description")
        
        st.markdown("""
        **Report Content:**

        1. **Data Analysis Report**
           - Data quality assessment
           - Feature distribution analysis
           - Data statistical information

        2. **Model Performance Report**
           - Multi-model comparison analysis
           - Performance metrics explanation
           - Prediction result statistics

        3. **Risk Assessment Report**
           - Risk level distribution
           - Risk score analysis
           - Risk trend analysis

        4. **Attack Analysis Report**
           - Attack type statistics
           - Attack severity analysis
           - Attack feature analysis

        5. **Explainability Report**
           - SHAP global feature importance
           - LIME local explanation analysis
           - Feature contribution analysis

        6. **Protection Recommendation Report**
           - Overall protection recommendations
           - Targeted protection measures
           - Emergency handling recommendations

        **Explainability Analysis:**
        - **SHAP Analysis**: Global feature importance and local feature contribution
        - **LIME Analysis**: Local linear explanation for individual predictions
        - **Feature Interaction**: Analysis of interactions between features
        - **Decision Path**: Visualization of model decision process
        """)