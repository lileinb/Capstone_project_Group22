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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入后端模块
from backend.attack_classification.attack_classifier import AttackClassifier

def _show_three_layer_architecture():
    """Display three-tier prediction architecture"""
    st.markdown("### 🏗️ Three-Tier Prediction Architecture")

    # 创建流程图
    col1, col2, col3 = st.columns(3)

    with col1:
        # 检查第一层状态
        has_features = 'engineered_features' in st.session_state and st.session_state.engineered_features is not None
        if has_features:
            st.success("✅ **Layer 1: Fraud Detection**")
            st.markdown("- Feature engineering completed")
            st.markdown("- Clustering analysis completed")
        else:
            st.error("❌ **Layer 1: Fraud Detection**")
            st.markdown("- Need to complete feature engineering")

    with col2:
        # 检查第二层状态
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
        # 第三层状态
        has_attack_analysis = 'attack_results' in st.session_state and st.session_state.attack_results is not None
        if has_attack_analysis:
            st.success("✅ **Layer 3: Attack Analysis**")
            st.markdown("- Attack type analysis completed")
        else:
            st.info("🎯 **Layer 3: Attack Analysis**")
            st.markdown("- Current page functionality")

    # 显示数据流向
    st.markdown("---")
    st.markdown("**🔄 Data Flow**: Raw Data → Feature Engineering → Clustering Analysis → Four-Class Risk Scoring → Attack Type Analysis → Comprehensive Threat Assessment")

    return has_features, has_risk_scoring

def show():
    """Show attack analysis page"""
    st.markdown('<div class="sub-header">⚔️ Three-Tier Prediction Architecture: Attack Type Analysis</div>', unsafe_allow_html=True)

    # 显示三层架构流程
    _show_three_layer_architecture()

    # 检查前置条件
    has_features, has_risk_scoring = _show_three_layer_architecture()

    if not has_features:
        st.warning("⚠️ Please complete feature engineering and clustering analysis first!")
        st.info("💡 Please complete the first two steps in order")
        return

    if not has_risk_scoring:
        st.warning("⚠️ Please complete four-class risk scoring first!")
        st.info("💡 Please complete four-class risk scoring in the '🎯 Risk Scoring' page")
        return
    
    # 初始化session state
    if 'attack_results' not in st.session_state:
        st.session_state.attack_results = None
    if 'attack_analysis' not in st.session_state:
        st.session_state.attack_analysis = None
    if 'protection_advice' not in st.session_state:
        st.session_state.protection_advice = None
    
    # 获取特征工程数据
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
    
    # 攻击类型说明
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
    
    # 攻击检测配置
    st.markdown("### ⚙️ 攻击检测配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 检测参数")
        
        # 检测敏感度
        detection_sensitivity = st.slider(
            "检测敏感度", 0.1, 2.0, 1.0, 0.1,
            help="攻击检测的敏感度，越高越严格"
        )
        
        # 特征权重
        device_weight = st.slider("设备特征权重", 0.1, 2.0, 1.0, 0.1)
        time_weight = st.slider("时间特征权重", 0.1, 2.0, 1.0, 0.1)
        amount_weight = st.slider("金额特征权重", 0.1, 2.0, 1.0, 0.1)
        location_weight = st.slider("位置特征权重", 0.1, 2.0, 1.0, 0.1)
    
    with col2:
        st.markdown("#### 📊 严重程度阈值")
        
        # 严重程度阈值
        low_severity_threshold = st.slider("低危阈值", 1, 3, 1, help="低危攻击特征匹配数")
        medium_severity_threshold = st.slider("中危阈值", 2, 4, 2, help="中危攻击特征匹配数")
        high_severity_threshold = st.slider("高危阈值", 3, 5, 3, help="高危攻击特征匹配数")
        
        # 批量检测参数
        batch_size_threshold = st.slider("批量大小阈值", 5, 50, 10, help="批量攻击的最小记录数")
        time_window = st.slider("时间窗口(分钟)", 1, 60, 10, help="批量攻击的时间窗口")
    
    # 执行攻击检测
    if st.button("🚀 执行攻击检测", type="primary", help="基于当前配置进行攻击类型检测"):
        try:
            with st.spinner("正在进行攻击检测..."):
                # 创建攻击分类器
                attack_classifier = AttackClassifier()

                # 执行攻击分类
                attack_results = attack_classifier.classify_attacks(engineered_data)

                # 保存结果
                st.session_state.attack_results = attack_results

                # 处理分类结果
                classification_results = attack_results.get('classification_results', [])
                severity_distribution = {}

                # 统计严重程度分布
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
                
                st.success("✅ 攻击检测完成！")
                
        except Exception as e:
            st.error(f"❌ 攻击检测失败: {e}")
            st.exception(e)
    
    # 显示攻击检测结果
    if st.session_state.attack_results is not None:
        st.markdown("### 📈 攻击检测结果")
        
        attack_results = st.session_state.attack_results
        attack_analysis = st.session_state.attack_analysis
        protection_advice = st.session_state.protection_advice
        
        # 攻击统计
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("检测到攻击", f"{attack_analysis['total_attacks']:,}")
        
        with col2:
            attack_types_count = len(attack_analysis['attack_types'])
            st.metric("攻击类型数", f"{attack_types_count}")
        
        with col3:
            if attack_analysis['total_attacks'] > 0:
                # 修正风险等级的键名
                high_severity = (attack_analysis['severity_distribution'].get('CRITICAL', 0) +
                               attack_analysis['severity_distribution'].get('HIGH', 0))
                high_severity_rate = (high_severity / attack_analysis['total_attacks'] * 100)
                st.metric("高危攻击率", f"{high_severity_rate:.1f}%")
            else:
                st.metric("高危攻击率", "0%")

        with col4:
            if attack_analysis['total_attacks'] > 0:
                classification_results = attack_analysis.get('classification_results', [])
                if classification_results:
                    avg_confidence = np.mean([result.get('confidence', 0) for result in classification_results])
                    st.metric("平均置信度", f"{avg_confidence:.3f}")
                else:
                    st.metric("平均置信度", "N/A")
            else:
                st.metric("平均置信度", "N/A")
        
        # 攻击类型分布
        st.markdown("#### 📊 攻击类型分布")

        try:
            if attack_analysis['attack_types']:
                # 创建攻击类型映射
                attack_type_names = {
                    'account_takeover': '账户接管攻击',
                    'identity_theft': '身份盗用攻击',
                    'bulk_fraud': '批量欺诈攻击',
                    'testing_attack': '测试性攻击'
                }

                # 转换攻击类型名称
                attack_types_data = []
                for attack_type, count in attack_analysis['attack_types'].items():
                    attack_types_data.append({
                        '攻击类型': attack_type_names.get(attack_type, attack_type),
                        '数量': count
                    })

                attack_types_df = pd.DataFrame(attack_types_data)

                if not attack_types_df.empty:
                    # 攻击类型饼图
                    fig = px.pie(
                        attack_types_df,
                        values='数量',
                        names='攻击类型',
                        title="攻击类型分布",
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

                    # 攻击类型表格
                    st.dataframe(attack_types_df, use_container_width=True)
                else:
                    st.info("📊 无攻击类型数据")
            else:
                st.info("📊 未检测到攻击类型")

        except Exception as e:
            st.error(f"❌ 攻击类型分布显示失败: {str(e)}")
            st.info("📊 请尝试重新执行攻击检测")
        
        # 风险等级分布
        st.markdown("#### ⚠️ 风险等级分布")

        try:
            if attack_analysis.get('severity_distribution'):
                # 风险等级名称映射
                risk_level_names = {
                    'CRITICAL': '极高风险',
                    'HIGH': '高风险',
                    'MEDIUM': '中等风险',
                    'LOW': '低风险'
                }

                # 转换风险等级名称
                risk_data = []
                for risk_level, count in attack_analysis['severity_distribution'].items():
                    risk_data.append({
                        '风险等级': risk_level_names.get(risk_level, risk_level),
                        '数量': count,
                        '原始等级': risk_level
                    })

                risk_df = pd.DataFrame(risk_data)

                if not risk_df.empty:
                    # 风险等级柱状图
                    colors = {
                        '极高风险': '#dc3545',
                        '高风险': '#fd7e14',
                        '中等风险': '#ffc107',
                        '低风险': '#28a745'
                    }

                    fig = px.bar(
                        risk_df,
                        x='风险等级',
                        y='数量',
                        title="攻击风险等级分布",
                        color='风险等级',
                        color_discrete_map=colors
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # 风险等级表格
                    display_df = risk_df[['风险等级', '数量']]
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("📊 无风险等级数据")
            else:
                st.info("📊 未检测到风险等级分布")

        except Exception as e:
            st.error(f"❌ 风险等级分布显示失败: {str(e)}")
            st.info("📊 请尝试重新执行攻击检测")
        
        # 攻击详情分析
        st.markdown("#### 🔍 攻击详情分析")
        
        # 选择要查看的攻击类型
        if attack_analysis['attack_types']:
            selected_attack_type = st.selectbox(
                "选择攻击类型查看详情", 
                list(attack_analysis['attack_types'].keys())
            )
            
            # 筛选该类型的攻击
            classification_results = attack_results.get('classification_results', [])
            type_attacks = [result for result in classification_results if result.get('attack_type') == selected_attack_type]
            
            if type_attacks:
                # 攻击特征分析
                st.markdown(f"**{selected_attack_type} 攻击特征分析**")
                
                # 统计特征频率
                feature_counts = {}
                for attack in type_attacks:
                    features = attack.get('detected_features', [])
                    for feature in features:
                        if feature not in feature_counts:
                            feature_counts[feature] = 0
                        feature_counts[feature] += 1
                
                if feature_counts:
                    feature_df = pd.DataFrame(list(feature_counts.items()), 
                                           columns=['检测特征', '出现次数'])
                    feature_df = feature_df.sort_values('出现次数', ascending=False)
                    
                    fig = px.bar(
                        feature_df,
                        x='检测特征',
                        y='出现次数',
                        title=f"{selected_attack_type} 检测特征频率"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 严重程度分布
                severity_counts = {}
                for attack in type_attacks:
                    severity = attack.get('severity', 'Unknown')
                    if severity not in severity_counts:
                        severity_counts[severity] = 0
                    severity_counts[severity] += 1
                
                if severity_counts:
                    severity_df = pd.DataFrame(list(severity_counts.items()), 
                                           columns=['严重程度', '数量'])
                    
                    fig = px.pie(
                        severity_df,
                        values='数量',
                        names='严重程度',
                        title=f"{selected_attack_type} 严重程度分布"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # 防护建议
        st.markdown("#### 🛡️ 防护建议")

        pattern_analysis = attack_analysis.get('pattern_analysis', {})
        if pattern_analysis and 'recommendations' in pattern_analysis:
            st.markdown("**基于攻击模式的防护建议**")

            for advice in pattern_analysis['recommendations']:
                st.markdown(f"- {advice}")

        # 主要攻击类型信息
        if pattern_analysis and 'dominant_attack_type' in pattern_analysis:
            dominant_type = pattern_analysis['dominant_attack_type']
            st.markdown(f"**主要攻击类型**: {dominant_type}")

            # 显示攻击模式特征
            if 'time_patterns' in pattern_analysis:
                time_patterns = pattern_analysis['time_patterns']
                if time_patterns:
                    st.markdown("**时间模式特征**:")
                    if 'peak_hours' in time_patterns:
                        st.markdown(f"- 高峰时段: {time_patterns['peak_hours']}")
                    if 'night_transactions' in time_patterns:
                        st.markdown(f"- 夜间交易数量: {time_patterns['night_transactions']}")

            if 'amount_patterns' in pattern_analysis:
                amount_patterns = pattern_analysis['amount_patterns']
                if amount_patterns:
                    st.markdown("**金额模式特征**:")
                    if 'avg_amount' in amount_patterns:
                        st.markdown(f"- 平均金额: {amount_patterns['avg_amount']}")
                    if 'large_amounts' in amount_patterns:
                        st.markdown(f"- 大额交易数量: {amount_patterns['large_amounts']}")
                    if 'small_amounts' in amount_patterns:
                        st.markdown(f"- 小额交易数量: {amount_patterns['small_amounts']}")
        
        # 攻击记录详情
        st.markdown("#### 📋 攻击记录详情")

        # 选择要查看的记录
        classification_results = st.session_state.attack_results.get('classification_results', [])
        if classification_results:
            selected_index = st.selectbox(
                "选择攻击记录查看详情",
                range(len(classification_results)),
                format_func=lambda x: f"记录 {x+1}: {classification_results[x].get('attack_type', 'Unknown')} - {classification_results[x].get('risk_level', 'Unknown')}"
            )
            
            if 0 <= selected_index < len(classification_results):
                attack_record = classification_results[selected_index]

                # 显示攻击记录详情
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**攻击信息**")
                    st.markdown(f"- 交易ID: {attack_record.get('transaction_id', 'Unknown')}")
                    st.markdown(f"- 客户ID: {attack_record.get('customer_id', 'Unknown')}")
                    st.markdown(f"- 攻击类型: {attack_record.get('attack_type', 'Unknown')}")
                    st.markdown(f"- 风险等级: {attack_record.get('risk_level', 'Unknown')}")
                    st.markdown(f"- 置信度: {attack_record.get('confidence', 0):.3f}")

                with col2:
                    st.markdown("**攻击特征**")
                    characteristics = attack_record.get('characteristics', [])
                    if characteristics:
                        for feature in characteristics:
                            st.markdown(f"- {feature}")
                    else:
                        st.markdown("- 无特殊特征")
        
        # 下一步按钮
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🚀 进入分析报告", type="primary", use_container_width=True):
                st.success("✅ 攻击分类完成，可以进入分析报告页面！")
                st.info("💡 请在侧边栏选择'📋 分析报告'页面继续")
    
    else:
        # 显示攻击分类说明
        st.markdown("### 📝 攻击分类说明")
        
        st.markdown("""
        **四大攻击类型：**
        
        1. **账户接管攻击 (Account Takeover)**
           - 攻击者获取合法用户的账户访问权限
           - 使用被盗账户进行欺诈交易
           - 通常涉及设备异常、时间异常等特征
        
        2. **身份盗用攻击 (Identity Theft)**
           - 攻击者伪造或盗用他人身份信息
           - 创建虚假账户或修改现有账户信息
           - 涉及地址不匹配、年龄不符等特征
        
        3. **批量欺诈攻击 (Bulk Fraud)**
           - 短时间内大量创建虚假账户或交易
           - 使用相似IP地址、相似交易模式
           - 通常有明确的批量特征和时间模式
        
        4. **测试性攻击 (Testing Attack)**
           - 攻击者测试系统安全机制
           - 使用小额交易测试支付流程
           - 涉及多种支付方式、快速连续交易
        
        **检测方法：**
        - 基于规则的特征匹配
        - 机器学习模式识别
        - 行为异常检测
        - 时间序列分析
        """) 