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

# 导入后端模块
from backend.pseudo_labeling.pseudo_label_generator import PseudoLabelGenerator
from backend.pseudo_labeling.fast_pseudo_label_generator import FastPseudoLabelGenerator

def show():
    """Display pseudo label generation page"""
    st.markdown('<div class="sub-header">🏷️ Intelligent Pseudo Label Generation System</div>', unsafe_allow_html=True)

    # 检查前置条件
    if not _check_prerequisites():
        return

    # 初始化session state
    _initialize_session_state()

    engineered_data = st.session_state.engineered_features

    # 显示系统说明
    _show_system_description()

    # 数据概览
    _show_data_overview(engineered_data)

    # 伪标签生成配置
    _show_generation_config()

    # 执行伪标签生成
    mode = st.session_state.label_generation_mode
    button_text = "🔍 Generate High-Quality Pseudo Labels (Standard Mode)" if mode == "standard" else "⚡ Quick Generate Pseudo Labels (Fast Mode)"
    button_help = "Multi-strategy integration, high-quality labels, completed in 2-3 minutes" if mode == "standard" else "Simplified algorithm, quick generation, completed within 30 seconds"

    if st.button(button_text, type="primary", help=button_help):
        _execute_pseudo_label_generation(engineered_data)

    # 显示伪标签结果
    if st.session_state.pseudo_labels:
        _show_pseudo_label_results()

        # 质量评估
        _show_quality_assessment()

        # 标签导出
        _show_label_export()


def _check_prerequisites():
    """Check prerequisites"""
    if 'engineered_features' not in st.session_state or st.session_state.engineered_features is None:
        st.warning("⚠️ Please complete feature engineering first!")
        st.info("💡 Please complete feature generation on the '🔧 Feature Engineering' page")
        return False
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
        # 检查是否有无监督风险评分结果
        if st.session_state.get('unsupervised_risk_results'):
            avg_risk = st.session_state.unsupervised_risk_results.get('average_risk_score', 0)
            st.metric("Average Risk Score", f"{avg_risk:.1f}")
        else:
            st.metric("Average Risk Score", "To be calculated")


def _show_generation_config():
    """Show generation configuration"""
    st.markdown("### ⚙️ Pseudo Label Generation Configuration")

    # 生成模式选择
    st.markdown("#### 🎯 Generation Mode Selection")

    col_mode1, col_mode2 = st.columns(2)

    with col_mode1:
        if st.button("🔍 Standard Mode", use_container_width=True,
                    help="Complete strategy integration, high-quality labels, completed in 2-3 minutes"):
            st.session_state.label_generation_mode = "standard"

    with col_mode2:
        if st.button("⚡ Fast Mode", use_container_width=True,
                    help="Simplified algorithm, quick generation, completed within 30 seconds"):
            st.session_state.label_generation_mode = "fast"

    # 显示当前模式
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
            value=0.8,
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

        # 显示当前权重配置
        current_weights = {
            "Unsupervised Risk Scoring": 45,
            "Cluster Risk Mapping": 35,
            "Expert Business Rules": 20
        }

        for strategy, weight in current_weights.items():
            st.write(f"- {strategy}: {weight}%")

        st.info("💡 Weights will be dynamically adjusted based on actual quality of each strategy")

    # 保存配置到session state
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
        generate_labels = st.button("🚀 Generate Pseudo Labels", type="primary", help="Generate pseudo labels based on selected strategy")

    with col2:
        if st.button("🗑️ Clear Results", help="Clear previous generation results"):
            st.session_state.pseudo_labels = None
            st.success("✅ Results cleared!")
            st.rerun()
    
    if generate_labels:
        try:
            # 获取工程化特征数据
            engineered_data = st.session_state.engineered_features
            if engineered_data is None or engineered_data.empty:
                st.error("❌ 请先完成特征工程步骤！")
                return

            with st.spinner("正在生成伪标签..."):
                # 更新置信度阈值
                st.session_state.label_generator.update_confidence_threshold(confidence_threshold)

                # 生成伪标签
                pseudo_results = st.session_state.label_generator.generate_pseudo_labels(
                    engineered_data, strategy=strategy
                )

                # 保存结果
                st.session_state.pseudo_labels = pseudo_results

                st.success("✅ 伪标签生成完成！")

        except Exception as e:
            st.error(f"❌ 伪标签生成失败: {e}")
            st.exception(e)
    
    # 显示伪标签结果
    if st.session_state.pseudo_labels is not None:
        st.markdown("### 📈 伪标签生成结果")
        
        pseudo_results = st.session_state.pseudo_labels
        
        # 基本统计
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("生成策略", pseudo_results['strategy'].upper())
        
        with col2:
            # 兼容不同模式的数据结构
            all_labels = pseudo_results.get('all_labels', pseudo_results.get('labels', []))
            total_labels = len(all_labels)
            st.metric("标签总数", f"{total_labels:,}")

        with col3:
            # 兼容不同的置信度字段
            if 'metadata' in pseudo_results:
                avg_confidence = pseudo_results['metadata'].get('avg_confidence_all', 0)
            else:
                avg_confidence = pseudo_results.get('avg_confidence', 0)
            st.metric("平均置信度", f"{avg_confidence:.3f}")

        with col4:
            # 兼容不同的高置信度计数字段
            if 'metadata' in pseudo_results:
                high_conf_count = pseudo_results['metadata'].get('high_quality_count', 0)
            else:
                high_conf_count = pseudo_results.get('high_confidence_count', 0)
            high_conf_rate = high_conf_count / total_labels * 100 if total_labels > 0 else 0
            st.metric("高置信度比例", f"{high_conf_rate:.1f}%")
        
        # 标签分布
        st.markdown("#### 📊 标签分布分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 标签分布饼图
            # 兼容不同模式的数据结构
            all_labels = pseudo_results.get('all_labels', pseudo_results.get('labels', []))
            if all_labels:
                label_counts = pd.Series(all_labels).value_counts()

                fig = px.pie(
                    values=label_counts.values,
                    names=['正常交易', '欺诈交易'],
                    title="伪标签分布",
                    color_discrete_map={
                        '正常交易': '#2E8B57',
                        '欺诈交易': '#DC143C'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 无标签数据可显示")

        with col2:
            # 置信度分布直方图
            # 兼容不同的置信度字段
            confidences = pseudo_results.get('all_confidences', pseudo_results.get('confidences', []))

            if confidences:
                fig = px.histogram(
                    x=confidences,
                    title="置信度分布",
                    nbins=20,
                    labels={'x': '置信度', 'y': '频次'}
                )
                fig.add_vline(x=confidence_threshold, line_dash="dash", line_color="red",
                             annotation_text=f"阈值: {confidence_threshold}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 无置信度数据可显示")
        
        # 质量评估
        st.markdown("#### 🎯 标签质量评估")
        
        # 如果有真实标签，计算准确性指标
        if 'is_fraudulent' in engineered_data.columns:
            true_labels = engineered_data['is_fraudulent'].tolist()
            # 兼容不同模式的标签字段
            all_labels = pseudo_results.get('all_labels', pseudo_results.get('labels', []))

            if all_labels and len(all_labels) == len(true_labels):
                try:
                    quality_metrics = st.session_state.label_generator.get_label_quality_metrics(
                        all_labels, true_labels
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("准确率", f"{quality_metrics['accuracy']:.3f}")

                    with col2:
                        st.metric("精确率", f"{quality_metrics['precision']:.3f}")

                    with col3:
                        st.metric("召回率", f"{quality_metrics['recall']:.3f}")

                    with col4:
                        st.metric("F1分数", f"{quality_metrics['f1_score']:.3f}")

                    # 混淆矩阵
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(true_labels, all_labels)

                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title="混淆矩阵",
                        labels=dict(x="预测标签", y="真实标签"),
                        x=['正常', '欺诈'],
                        y=['正常', '欺诈']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ 质量评估计算失败: {str(e)}")
            else:
                st.info("💡 标签数量不匹配，跳过质量评估")

        # 下一步按钮
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if st.button("🤖 下一步：模型训练", type="primary", use_container_width=True):
                st.success("✅ 伪标签生成完成，可以进入模型训练页面！")
                st.info("💡 请在侧边栏选择'🤖 模型训练'页面继续")


def _execute_pseudo_label_generation(engineered_data):
    """执行伪标签生成"""
    try:
        config = st.session_state.label_config
        mode = st.session_state.label_generation_mode

        mode_text = "标准模式" if mode == "standard" else "快速模式"
        mode_icon = "🔍" if mode == "standard" else "⚡"

        with st.spinner(f"正在使用{mode_text}生成伪标签..."):
            # 记录开始时间
            import time
            start_time = time.time()

            # 根据模式选择生成器
            if mode == "standard":
                # 使用标准模式生成器
                label_results = st.session_state.label_generator.generate_high_quality_pseudo_labels(
                    engineered_data,
                    min_confidence=config['min_confidence'],
                    use_calibration=config['use_calibration']
                )
            else:
                # 使用快速模式生成器
                risk_results = st.session_state.get('unsupervised_risk_results', None)
                label_results = st.session_state.fast_label_generator.generate_fast_pseudo_labels(
                    engineered_data,
                    risk_results=risk_results,
                    min_confidence=config['min_confidence']
                )

            # 记录结束时间
            end_time = time.time()
            generation_time = end_time - start_time

            st.session_state.pseudo_labels = label_results
            st.session_state.high_quality_labels = label_results

            if label_results and label_results.get('high_quality_labels'):
                total_labels = len(label_results.get('all_labels', []))
                hq_labels = len(label_results.get('high_quality_labels', []))

                success_msg = f"✅ {mode_icon} {mode_text}伪标签生成完成！"
                success_msg += f" 从 {total_labels} 个样本中筛选出 {hq_labels} 个高质量标签，耗时 {generation_time:.2f} 秒"
                st.success(success_msg)

                # 显示基本统计
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    hq_rate = label_results['metadata']['high_quality_rate']
                    st.metric("高质量比例", f"{hq_rate:.1%}")

                with col2:
                    avg_conf_hq = label_results['metadata']['avg_confidence_hq']
                    st.metric("平均置信度", f"{avg_conf_hq:.3f}")

                with col3:
                    fraud_rate_hq = label_results['metadata']['fraud_rate_hq']
                    st.metric("伪标签欺诈率", f"{fraud_rate_hq:.1%}")

                with col4:
                    quality_score = label_results['quality_report'].get('quality_score', 0)
                    st.metric("质量评分", f"{quality_score:.1f}")

                # 显示校准状态
                if label_results.get('calibration_applied'):
                    st.info("✅ 已应用校准优化，风险评分阈值已优化")
                elif config['use_calibration']:
                    st.warning("⚠️ 校准未成功应用，使用默认阈值")

            else:
                st.error("❌ 未能生成足够的高质量伪标签，请降低置信度阈值")

    except Exception as e:
        st.error(f"❌ 伪标签生成失败: {str(e)}")


def _show_pseudo_label_results():
    """显示伪标签结果"""
    st.markdown("### 📈 高质量伪标签结果")

    label_results = st.session_state.pseudo_labels

    # 结果概览
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**标签分布对比**")

        all_labels = label_results.get('all_labels', [])
        hq_labels = label_results.get('high_quality_labels', [])

        # 创建对比数据
        all_dist = pd.Series(all_labels).value_counts()
        hq_dist = pd.Series(hq_labels).value_counts() if hq_labels else pd.Series()

        # 确保包含所有可能的标签类别
        all_dist = all_dist.reindex([0, 1], fill_value=0)
        hq_dist = hq_dist.reindex([0, 1], fill_value=0)

        comparison_data = pd.DataFrame({
            '全部标签': all_dist,
            '高质量标签': hq_dist
        }).fillna(0)

        comparison_data.index = ['正常', '欺诈']

        fig = px.bar(
            comparison_data,
            title="标签分布对比",
            labels={'index': '标签类型', 'value': '数量'},
            color_discrete_map={'全部标签': '#17a2b8', '高质量标签': '#28a745'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**置信度分布**")

        all_confidences = label_results.get('all_confidences', [])
        hq_confidences = label_results.get('high_quality_confidences', [])

        if all_confidences:
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=all_confidences,
                name='全部标签',
                opacity=0.7,
                nbinsx=20
            ))

            if hq_confidences:
                fig.add_trace(go.Histogram(
                    x=hq_confidences,
                    name='高质量标签',
                    opacity=0.7,
                    nbinsx=20
                ))

            fig.add_vline(
                x=label_results['min_confidence_threshold'],
                line_dash="dash",
                line_color="red",
                annotation_text="置信度阈值"
            )

            fig.update_layout(
                title="置信度分布对比",
                xaxis_title="置信度",
                yaxis_title="频次",
                barmode='overlay'
            )

            st.plotly_chart(fig, use_container_width=True)


def _show_quality_assessment():
    """显示质量评估"""
    st.markdown("### 🎯 质量评估与验证")

    label_results = st.session_state.pseudo_labels
    engineered_data = st.session_state.engineered_features

    # 如果有真实标签，进行对比验证
    if 'is_fraudulent' in engineered_data.columns:
        st.markdown("**与真实标签对比验证**")

        hq_indices = label_results.get('high_quality_indices', [])
        hq_labels = label_results.get('high_quality_labels', [])

        if hq_indices and hq_labels:
            # 获取对应的真实标签
            true_labels_hq = [engineered_data.iloc[i]['is_fraudulent'] for i in hq_indices]

            # 计算性能指标
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

            accuracy = accuracy_score(true_labels_hq, hq_labels)
            precision = precision_score(true_labels_hq, hq_labels, zero_division=0)
            recall = recall_score(true_labels_hq, hq_labels, zero_division=0)
            f1 = f1_score(true_labels_hq, hq_labels, zero_division=0)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("准确率", f"{accuracy:.3f}")

            with col2:
                st.metric("精确率", f"{precision:.3f}")

            with col3:
                st.metric("召回率", f"{recall:.3f}")

            with col4:
                st.metric("F1 Score", f"{f1:.3f}")

            # 混淆矩阵
            cm = confusion_matrix(true_labels_hq, hq_labels)

            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted Label", y="True Label"),
                x=['Normal', 'Fraud'],
                y=['Normal', 'Fraud']
            )

            st.plotly_chart(fig, use_container_width=True)


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

        if export_option == "仅高质量标签":
            export_count = len(label_results.get('high_quality_labels', []))
            st.write(f"导出样本数: {export_count:,}")
        elif export_option == "全部标签":
            export_count = len(label_results.get('all_labels', []))
            st.write(f"导出样本数: {export_count:,}")
        else:
            export_count = len(label_results.get('all_labels', []))
            st.write(f"报告样本数: {export_count:,}")

    # 生成导出数据
    if st.button("📥 生成导出文件", type="secondary"):
        try:
            if export_option == "仅高质量标签":
                export_data = _prepare_high_quality_export(label_results, engineered_data, include_features, include_confidence)
                filename = f"high_quality_pseudo_labels_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            elif export_option == "全部标签":
                export_data = _prepare_all_labels_export(label_results, engineered_data, include_features, include_confidence)
                filename = f"all_pseudo_labels_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            else:
                export_data = _prepare_comparison_report(label_results, engineered_data)
                filename = f"pseudo_labels_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

            csv_data = export_data.to_csv(index=False)

            st.download_button(
                label=f"📥 下载 {filename}",
                data=csv_data,
                file_name=filename,
                mime="text/csv"
            )

            st.success(f"✅ 导出文件已准备完成，包含 {len(export_data)} 条记录")

        except Exception as e:
            st.error(f"❌ 导出失败: {str(e)}")

    # 下一步按钮
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("🤖 下一步：模型训练", type="primary", use_container_width=True):
            st.success("✅ 高质量伪标签生成完成，可以进入模型训练页面！")
            st.info("💡 请在侧边栏选择'🤖 模型训练'页面继续")


def _prepare_high_quality_export(label_results, engineered_data, include_features, include_confidence):
    """准备高质量标签导出数据"""
    hq_indices = label_results.get('high_quality_indices', [])
    hq_labels = label_results.get('high_quality_labels', [])
    hq_confidences = label_results.get('high_quality_confidences', [])

    # 基础数据
    export_data = pd.DataFrame({
        'sample_index': hq_indices,
        'pseudo_label': hq_labels
    })

    if include_confidence:
        export_data['confidence'] = hq_confidences

    if include_features:
        # 添加特征数据
        feature_data = engineered_data.iloc[hq_indices].reset_index(drop=True)
        export_data = pd.concat([export_data, feature_data], axis=1)

    return export_data


def _prepare_all_labels_export(label_results, engineered_data, include_features, include_confidence):
    """准备全部标签导出数据"""
    all_labels = label_results.get('all_labels', [])
    all_confidences = label_results.get('all_confidences', [])

    # 基础数据
    export_data = pd.DataFrame({
        'sample_index': range(len(all_labels)),
        'pseudo_label': all_labels
    })

    if include_confidence:
        export_data['confidence'] = all_confidences

    # 标记高质量标签
    hq_indices = set(label_results.get('high_quality_indices', []))
    export_data['is_high_quality'] = export_data['sample_index'].isin(hq_indices)

    if include_features:
        # 添加特征数据
        export_data = pd.concat([export_data, engineered_data.reset_index(drop=True)], axis=1)

    return export_data


def _prepare_comparison_report(label_results, engineered_data):
    """准备对比报告"""
    all_labels = label_results.get('all_labels', [])
    all_confidences = label_results.get('all_confidences', [])
    hq_indices = set(label_results.get('high_quality_indices', []))

    # 基础报告数据
    report_data = pd.DataFrame({
        'sample_index': range(len(all_labels)),
        'pseudo_label': all_labels,
        'confidence': all_confidences,
        'is_high_quality': [i in hq_indices for i in range(len(all_labels))]
    })

    # 添加关键特征
    key_features = ['transaction_id', 'customer_id', 'transaction_amount', 'customer_age', 'account_age_days']
    available_features = [f for f in key_features if f in engineered_data.columns]

    if available_features:
        report_data = pd.concat([
            report_data,
            engineered_data[available_features].reset_index(drop=True)
        ], axis=1)

    # 添加真实标签对比（如果有）
    if 'is_fraudulent' in engineered_data.columns:
        report_data['true_label'] = engineered_data['is_fraudulent'].reset_index(drop=True)
        report_data['label_match'] = report_data['pseudo_label'] == report_data['true_label']

    return report_data
