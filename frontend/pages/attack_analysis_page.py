"""
Attack Analysis Page
Third layer of three-tier prediction architecture: attack type analysis and threat assessment
Integrated fraud detection → four-class risk grading → attack type analysis
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
from backend.attack_classification.attack_classifier import AttackClassifier

def _show_three_layer_architecture():
    """Display three-tier prediction architecture"""
    st.markdown("### 🏗️ Three-Tier Prediction Architecture")

    # Create flow chart
    col1, col2, col3 = st.columns(3)

    with col1:
        # Check first layer status
        has_features = 'engineered_features' in st.session_state and st.session_state.engineered_features is not None
        if has_features:
            st.success("✅ **Layer 1: Fraud Detection**")
            st.markdown("- Feature engineering completed")
            st.markdown("- Clustering analysis completed")
        else:
            st.error("❌ **Layer 1: Fraud Detection**")
            st.markdown("- Need to complete feature engineering")

    with col2:
        # Check second layer status
        has_risk_scoring = 'four_class_risk_results' in st.session_state and st.session_state.four_class_risk_results is not None
        if has_risk_scoring:
            st.success("✅ **Layer 2: Risk Grading**")
            st.markdown("- Four-class risk scoring completed")
            risk_results = st.session_state.four_class_risk_results
            high_risk_pct = risk_results.get('high_risk_percentage', 0)
            st.markdown(f"- High risk ratio: {high_risk_pct:.1f}%")
        else:
            st.warning("⚠️ **Layer 2: Risk Grading**")
            st.markdown("- Need to complete four-class risk scoring")

    with col3:
        # Third layer status
        has_attack_analysis = 'attack_results' in st.session_state and st.session_state.attack_results is not None
        if has_attack_analysis:
            st.success("✅ **Layer 3: Attack Analysis**")
            st.markdown("- Attack type analysis completed")
        else:
            st.info("🎯 **Layer 3: Attack Analysis**")
            st.markdown("- Current page functionality")

    # Display data flow
    st.markdown("---")
    st.markdown("**🔄 Data Flow**: Raw Data → Feature Engineering → Clustering Analysis → Four-Class Risk Scoring → Attack Type Analysis → Comprehensive Threat Assessment")

    return has_features, has_risk_scoring

def show():
    """Show attack analysis page"""
    st.markdown('<div class="sub-header">⚔️ Three-Tier Prediction Architecture: Attack Type Analysis</div>', unsafe_allow_html=True)

    # Display three-layer architecture flow
    _show_three_layer_architecture()

    # Check prerequisites
    has_features, has_risk_scoring = _show_three_layer_architecture()

    if not has_features:
        st.warning("⚠️ Please complete feature engineering and clustering analysis first!")
        st.info("💡 Please complete the first two steps in order")
        return

    if not has_risk_scoring:
        st.warning("⚠️ Please complete four-class risk scoring first!")
        st.info("💡 Please complete four-class risk scoring in the '🎯 Risk Scoring' page")
        return

    # Initialize session state
    if 'attack_results' not in st.session_state:
        st.session_state.attack_results = None
    if 'attack_analysis' not in st.session_state:
        st.session_state.attack_analysis = None
    if 'protection_advice' not in st.session_state:
        st.session_state.protection_advice = None

    # Get feature engineering data
    engineered_data = st.session_state.engineered_features
    
    st.markdown("### 📊 Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Record Count", f"{len(engineered_data):,}")

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
    
    # Attack type description
    st.markdown("### 🎯 Attack Type Description")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔐 Account Takeover Attack")
        st.markdown("- **Detection Features**: New Device + Old Account + Large Transaction + Abnormal Time")
        st.markdown("- **Severity**: High Risk (3-4 features) / Medium Risk (2 features) / Low Risk (1 feature)")
        st.markdown("- **Protection Measures**: Two-factor authentication, device restrictions, transaction monitoring")

        st.markdown("#### 🆔 Identity Theft Attack")
        st.markdown("- **Detection Features**: Address mismatch + Abnormal payment + Age mismatch + IP anomaly")
        st.markdown("- **Severity**: Based on number of feature matches")
        st.markdown("- **Protection Measures**: Identity verification, address verification, payment restrictions")

    with col2:
        st.markdown("#### 📦 Bulk Fraud Attack")
        st.markdown("- **Detection Features**: Similar IP + Multiple transactions in short time + Similar patterns + Bulk registration")
        st.markdown("- **Severity**: Based on bulk scale and time density")
        st.markdown("- **Protection Measures**: IP restrictions, frequency control, bulk detection")

        st.markdown("#### 🧪 Testing Attack")
        st.markdown("- **Detection Features**: Small multiple transactions + Multiple payment methods + Rapid succession + New account")
        st.markdown("- **Severity**: Based on testing frequency and scope")
        st.markdown("- **Protection Measures**: Payment restrictions, verification codes, account review")
    
    # Attack detection configuration
    st.markdown("### ⚙️ Attack Detection Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Detection Parameters")

        # Detection sensitivity
        detection_sensitivity = st.slider(
            "Detection Sensitivity", 0.1, 2.0, 1.0, 0.1,
            help="Attack detection sensitivity, higher values are more strict"
        )

        # Feature weights
        device_weight = st.slider("Device Feature Weight", 0.1, 2.0, 1.0, 0.1)
        time_weight = st.slider("Time Feature Weight", 0.1, 2.0, 1.0, 0.1)
        amount_weight = st.slider("Amount Feature Weight", 0.1, 2.0, 1.0, 0.1)
        location_weight = st.slider("Location Feature Weight", 0.1, 2.0, 1.0, 0.1)

    with col2:
        st.markdown("#### 📊 Severity Thresholds")

        # Severity thresholds
        low_severity_threshold = st.slider("Low Risk Threshold", 1, 3, 1, help="Number of features matched for low risk attacks")
        medium_severity_threshold = st.slider("Medium Risk Threshold", 2, 4, 2, help="Number of features matched for medium risk attacks")
        high_severity_threshold = st.slider("High Risk Threshold", 3, 5, 3, help="Number of features matched for high risk attacks")

        # Batch detection parameters
        batch_size_threshold = st.slider("Batch Size Threshold", 5, 50, 10, help="Minimum number of records for batch attacks")
        time_window = st.slider("Time Window (minutes)", 1, 60, 10, help="Time window for batch attacks")
    
    # Execute attack detection
    if st.button("🚀 Execute Attack Detection", type="primary", help="Perform attack type detection based on current configuration"):
        try:
            with st.spinner("Performing attack detection..."):
                # Create attack classifier
                attack_classifier = AttackClassifier()

                # 获取聚类和风险评分结果（如果存在）
                cluster_results = st.session_state.get('clustering_results', None)
                risk_results = st.session_state.get('risk_results', None)

                # Execute attack classification with enhanced context
                attack_results = attack_classifier.classify_attacks(
                    engineered_data,
                    cluster_results=cluster_results,
                    risk_results=risk_results
                )

                # Save results
                st.session_state.attack_results = attack_results

                # Process classification results
                classification_results = attack_results.get('classification_results', [])
                severity_distribution = {}

                # Count severity distribution
                for result in classification_results:
                    risk_level = result.get('risk_level', 'UNKNOWN')
                    severity_distribution[risk_level] = severity_distribution.get(risk_level, 0) + 1

                st.session_state.attack_analysis = {
                    'total_attacks': attack_results.get('fraud_transactions', 0),
                    'attack_types': attack_results.get('attack_types', {}),
                    'pattern_analysis': attack_results.get('pattern_analysis', {}),
                    'severity_distribution': severity_distribution,
                    'classification_results': classification_results,
                    'detection_params': {
                        'sensitivity': detection_sensitivity,
                        'weights': {
                            'device': device_weight,
                            'time': time_weight,
                            'amount': amount_weight,
                            'location': location_weight
                        }
                    }
                }

                st.success("✅ Attack detection completed!")

        except Exception as e:
            st.error(f"❌ Attack detection failed: {e}")
            st.exception(e)
    
    # Display attack detection results
    if st.session_state.attack_results is not None:
        st.markdown("### 📈 Attack Detection Results")

        attack_results = st.session_state.attack_results
        attack_analysis = st.session_state.attack_analysis
        protection_advice = st.session_state.protection_advice

        # Attack statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Detected Attacks", f"{attack_analysis['total_attacks']:,}")

        with col2:
            attack_types_count = len(attack_analysis['attack_types'])
            st.metric("Attack Types", f"{attack_types_count}")

        with col3:
            if attack_analysis['total_attacks'] > 0:
                # Fix risk level key names
                high_severity = (attack_analysis['severity_distribution'].get('CRITICAL', 0) +
                               attack_analysis['severity_distribution'].get('HIGH', 0))
                high_severity_rate = (high_severity / attack_analysis['total_attacks'] * 100)
                st.metric("High Risk Attack Rate", f"{high_severity_rate:.1f}%")
            else:
                st.metric("High Risk Attack Rate", "0%")

        with col4:
            if attack_analysis['total_attacks'] > 0:
                classification_results = attack_analysis.get('classification_results', [])
                if classification_results:
                    avg_confidence = np.mean([result.get('confidence', 0) for result in classification_results])
                    st.metric("Average Confidence", f"{avg_confidence:.3f}")
                else:
                    st.metric("Average Confidence", "N/A")
            else:
                st.metric("Average Confidence", "N/A")
        
        # Attack type distribution
        st.markdown("#### 📊 Attack Type Distribution")

        try:
            if attack_analysis['attack_types']:
                # Create attack type mapping - 更新为8种攻击类型
                attack_type_names = {
                    'account_takeover': 'Account Takeover',
                    'identity_theft': 'Identity Theft',
                    'card_testing': 'Credit Card Testing',
                    'bulk_fraud': 'Bulk Fraud',
                    'velocity_attack': 'High Velocity Attack',
                    'synthetic_identity': 'Synthetic Identity',
                    'friendly_fraud': 'Friendly Fraud',
                    'normal_behavior': 'Normal Behavior'
                }

                # Convert attack type names and filter out zero counts
                attack_types_data = []
                for attack_type, count in attack_analysis['attack_types'].items():
                    if count > 0:  # 只包含有实际数据的攻击类型
                        attack_types_data.append({
                            'Attack Type': attack_type_names.get(attack_type, attack_type),
                            'Count': count
                        })

                attack_types_df = pd.DataFrame(attack_types_data)

                if not attack_types_df.empty:
                    # 按数量排序，便于显示
                    attack_types_df = attack_types_df.sort_values('Count', ascending=False)

                    # Attack type pie chart with enhanced styling
                    fig = px.pie(
                        attack_types_df,
                        values='Count',
                        names='Attack Type',
                        title="Attack Type Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        hover_data=['Count']
                    )

                    # 优化图表显示
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                    )

                    fig.update_layout(
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="middle",
                            y=0.5,
                            xanchor="left",
                            x=1.05
                        ),
                        margin=dict(l=20, r=150, t=50, b=20)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Attack type table with percentage
                    attack_types_df['Percentage'] = (attack_types_df['Count'] / attack_types_df['Count'].sum() * 100).round(1)
                    attack_types_df['Percentage'] = attack_types_df['Percentage'].astype(str) + '%'

                    st.dataframe(attack_types_df, use_container_width=True)
                else:
                    st.info("📊 No attack type data")
            else:
                st.info("📊 No attack types detected")

        except Exception as e:
            st.error(f"❌ Attack type distribution display failed: {str(e)}")
            st.info("📊 Please try re-executing attack detection")
        
        # Risk level distribution
        st.markdown("#### ⚠️ Risk Level Distribution")

        try:
            if attack_analysis.get('severity_distribution'):
                # Risk level name mapping
                risk_level_names = {
                    'CRITICAL': 'Critical Risk',
                    'HIGH': 'High Risk',
                    'MEDIUM': 'Medium Risk',
                    'LOW': 'Low Risk'
                }

                # Convert risk level names
                risk_data = []
                for risk_level, count in attack_analysis['severity_distribution'].items():
                    risk_data.append({
                        'Risk Level': risk_level_names.get(risk_level, risk_level),
                        'Count': count,
                        'Original Level': risk_level
                    })

                risk_df = pd.DataFrame(risk_data)

                if not risk_df.empty:
                    # Risk level bar chart
                    colors = {
                        'Critical Risk': '#dc3545',
                        'High Risk': '#fd7e14',
                        'Medium Risk': '#ffc107',
                        'Low Risk': '#28a745'
                    }

                    fig = px.bar(
                        risk_df,
                        x='Risk Level',
                        y='Count',
                        title="Attack Risk Level Distribution",
                        color='Risk Level',
                        color_discrete_map=colors
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # Risk level table
                    display_df = risk_df[['Risk Level', 'Count']]
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("📊 No risk level data available")
            else:
                st.info("📊 No risk level distribution detected")

        except Exception as e:
            st.error(f"❌ Risk level distribution display failed: {str(e)}")
            st.info("📊 Please try re-executing attack detection")
        
        # Attack detail analysis
        st.markdown("#### 🔍 Attack Detail Analysis")

        # Select attack type to view details
        if attack_analysis['attack_types']:
            selected_attack_type = st.selectbox(
                "Select attack type to view details",
                list(attack_analysis['attack_types'].keys())
            )

            # Filter attacks of this type
            classification_results = attack_results.get('classification_results', [])
            type_attacks = [result for result in classification_results if result.get('attack_type') == selected_attack_type]

            if type_attacks:
                # Attack feature analysis
                st.markdown(f"**{selected_attack_type} Attack Feature Analysis**")

                # Count feature frequency
                feature_counts = {}
                for attack in type_attacks:
                    features = attack.get('detected_features', [])
                    for feature in features:
                        if feature not in feature_counts:
                            feature_counts[feature] = 0
                        feature_counts[feature] += 1

                if feature_counts:
                    feature_df = pd.DataFrame(list(feature_counts.items()),
                                           columns=['Detection Feature', 'Frequency'])
                    feature_df = feature_df.sort_values('Frequency', ascending=False)

                    fig = px.bar(
                        feature_df,
                        x='Detection Feature',
                        y='Frequency',
                        title=f"{selected_attack_type} Detection Feature Frequency"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Severity distribution
                severity_counts = {}
                for attack in type_attacks:
                    severity = attack.get('severity', 'Unknown')
                    if severity not in severity_counts:
                        severity_counts[severity] = 0
                    severity_counts[severity] += 1

                if severity_counts:
                    severity_df = pd.DataFrame(list(severity_counts.items()),
                                           columns=['Severity Level', 'Count'])

                    fig = px.pie(
                        severity_df,
                        values='Count',
                        names='Severity Level',
                        title=f"{selected_attack_type} Severity Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Protection recommendations
        st.markdown("#### 🛡️ Protection Recommendations")

        pattern_analysis = attack_analysis.get('pattern_analysis', {})
        if pattern_analysis and 'recommendations' in pattern_analysis:
            st.markdown("**Protection Recommendations Based on Attack Patterns**")

            for advice in pattern_analysis['recommendations']:
                st.markdown(f"- {advice}")

        # Main attack type information
        if pattern_analysis and 'dominant_attack_type' in pattern_analysis:
            dominant_type = pattern_analysis['dominant_attack_type']
            st.markdown(f"**Main Attack Type**: {dominant_type}")

            # Show attack pattern features
            if 'time_patterns' in pattern_analysis:
                time_patterns = pattern_analysis['time_patterns']
                if time_patterns:
                    st.markdown("**Time Pattern Features**:")
                    if 'peak_hours' in time_patterns:
                        st.markdown(f"- Peak hours: {time_patterns['peak_hours']}")
                    if 'night_transactions' in time_patterns:
                        st.markdown(f"- Night transaction count: {time_patterns['night_transactions']}")

            if 'amount_patterns' in pattern_analysis:
                amount_patterns = pattern_analysis['amount_patterns']
                if amount_patterns:
                    st.markdown("**Amount Pattern Features**:")
                    if 'avg_amount' in amount_patterns:
                        st.markdown(f"- Average amount: {amount_patterns['avg_amount']}")
                    if 'large_amounts' in amount_patterns:
                        st.markdown(f"- Large transaction count: {amount_patterns['large_amounts']}")
                    if 'small_amounts' in amount_patterns:
                        st.markdown(f"- Small transaction count: {amount_patterns['small_amounts']}")
        
        # Attack record details
        st.markdown("#### 📋 Attack Record Details")

        # Select record to view
        classification_results = st.session_state.attack_results.get('classification_results', [])
        if classification_results:
            selected_index = st.selectbox(
                "Select attack record to view details",
                range(len(classification_results)),
                format_func=lambda x: f"Record {x+1}: {classification_results[x].get('attack_type', 'Unknown')} - {classification_results[x].get('risk_level', 'Unknown')}"
            )

            if 0 <= selected_index < len(classification_results):
                attack_record = classification_results[selected_index]

                # Show attack record details
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Attack Information**")
                    st.markdown(f"- Transaction ID: {attack_record.get('transaction_id', 'Unknown')}")
                    st.markdown(f"- Customer ID: {attack_record.get('customer_id', 'Unknown')}")
                    st.markdown(f"- Attack Type: {attack_record.get('attack_type', 'Unknown')}")
                    st.markdown(f"- Risk Level: {attack_record.get('risk_level', 'Unknown')}")
                    st.markdown(f"- Confidence: {attack_record.get('confidence', 0):.3f}")

                with col2:
                    st.markdown("**Attack Features**")
                    characteristics = attack_record.get('characteristics', [])
                    if characteristics:
                        for feature in characteristics:
                            st.markdown(f"- {feature}")
                    else:
                        st.markdown("- No special features")
        
        # Next step button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if st.button("🚀 Enter Analysis Report", type="primary", use_container_width=True):
                st.success("✅ Attack classification completed, ready to enter analysis report page!")
                st.info("💡 Please select '📋 Analysis Report' page in the sidebar to continue")

    else:
        # Show attack classification description
        st.markdown("### 📝 Attack Classification Description")

        st.markdown("""
        **Four Major Attack Types:**

        1. **Account Takeover Attack**
           - Attackers gain access to legitimate user accounts
           - Use compromised accounts for fraudulent transactions
           - Usually involves device anomalies, time anomalies and other features

        2. **Identity Theft Attack**
           - Attackers forge or steal others' identity information
           - Create fake accounts or modify existing account information
           - Involves address mismatches, age inconsistencies and other features

        3. **Bulk Fraud Attack**
           - Large-scale creation of fake accounts or transactions in short time
           - Use similar IP addresses and similar transaction patterns
           - Usually have clear bulk features and time patterns

        4. **Testing Attack**
           - Attackers test system security mechanisms
           - Use small transactions to test payment processes
           - Involves multiple payment methods and rapid consecutive transactions

        **Detection Methods:**
        - Rule-based feature matching
        - Machine learning pattern recognition
        - Behavioral anomaly detection
        - Time series analysis
        """)