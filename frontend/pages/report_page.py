"""
分析报告页面
负责生成综合分析报告和可解释性分析
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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入后端模块
from backend.explainer.shap_explainer import SHAPExplainer
from backend.explainer.lime_explainer import LIMEExplainer
from backend.analysis_reporting.report_generator import ReportGenerator

def show():
    """显示分析报告页面"""
    st.markdown('<div class="sub-header">📋 综合分析报告与可解释性分析</div>', unsafe_allow_html=True)
    
    # 检查是否有特征工程数据
    if 'engineered_features' not in st.session_state or st.session_state.engineered_features is None:
        st.warning("⚠️ 请先完成特征工程！")
        st.info("💡 请在'🔧 特征工程'页面完成特征生成")
        return
    
    # 初始化session state
    if 'shap_analysis' not in st.session_state:
        st.session_state.shap_analysis = None
    if 'lime_analysis' not in st.session_state:
        st.session_state.lime_analysis = None
    if 'report_data' not in st.session_state:
        st.session_state.report_data = None
    
    # 获取特征工程数据
    engineered_data = st.session_state.engineered_features
    
    st.markdown("### 📊 数据概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("记录数", f"{len(engineered_data):,}")
    
    with col2:
        st.metric("特征数", f"{len(engineered_data.columns)}")
    
    with col3:
        numeric_features = len(engineered_data.select_dtypes(include=['number']).columns)
        st.metric("数值特征", f"{numeric_features}")
    
    with col4:
        if 'is_fraudulent' in engineered_data.columns:
            fraud_rate = (engineered_data['is_fraudulent'].sum() / len(engineered_data) * 100).round(2)
            st.metric("欺诈率", f"{fraud_rate}%")
        else:
            st.metric("欺诈率", "N/A")
    
    # 报告配置
    st.markdown("### ⚙️ 报告配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 报告内容选择")
        
        # 报告内容选择
        include_data_analysis = st.checkbox("数据分析报告", value=True)
        include_model_performance = st.checkbox("模型性能报告", value=True)
        include_risk_assessment = st.checkbox("风险评估报告", value=True)
        include_attack_analysis = st.checkbox("攻击分析报告", value=True)
        include_explainability = st.checkbox("可解释性分析", value=True)
        include_recommendations = st.checkbox("防护建议", value=True)
    
    with col2:
        st.markdown("#### 🎯 可解释性分析")
        
        # 可解释性分析选择
        include_shap_analysis = st.checkbox("SHAP分析", value=True)
        include_lime_analysis = st.checkbox("LIME分析", value=True)
        
        # 分析样本数
        analysis_sample_size = st.slider(
            "分析样本数", 100, 1000, 500,
            help="用于可解释性分析的样本数量"
        )
        
        # 特征重要性阈值
        feature_importance_threshold = st.slider(
            "特征重要性阈值", 0.01, 0.1, 0.05, 0.01,
            help="显示特征重要性的最小阈值"
        )
    
    # 报告格式选择
    st.markdown("#### 📄 报告格式")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        export_pdf = st.checkbox("PDF格式", value=True)
    
    with col2:
        export_excel = st.checkbox("Excel格式", value=True)
    
    with col3:
        export_html = st.checkbox("HTML格式", value=True)
    
    # 执行分析
    if st.button("🚀 生成分析报告", type="primary", help="基于当前配置生成综合分析报告"):
        try:
            with st.spinner("正在生成分析报告..."):
                # 准备数据
                X = engineered_data.select_dtypes(include=['number'])
                if 'is_fraudulent' in X.columns:
                    y = X['is_fraudulent']
                    X = X.drop('is_fraudulent', axis=1)
                    has_labels = True
                else:
                    y = None
                    has_labels = False
                
                # 处理缺失值
                X = X.fillna(0)
                
                # SHAP分析
                if include_shap_analysis and has_labels:
                    try:
                        shap_explainer = SHAPExplainer()
                        shap_analysis = shap_explainer.analyze(X, y, sample_size=analysis_sample_size)
                        st.session_state.shap_analysis = shap_analysis
                        st.success("✅ SHAP分析完成")
                    except Exception as e:
                        st.warning(f"⚠️ SHAP分析失败: {e}")
                
                # LIME分析
                if include_lime_analysis and has_labels:
                    try:
                        lime_explainer = LIMEExplainer()
                        lime_analysis = lime_explainer.analyze(X, y, sample_size=analysis_sample_size)
                        st.session_state.lime_analysis = lime_analysis
                        st.success("✅ LIME分析完成")
                    except Exception as e:
                        st.warning(f"⚠️ LIME分析失败: {e}")
                
                # 生成报告数据
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
                
                # 添加其他分析结果
                if 'clustering_results' in st.session_state:
                    report_data['clustering_results'] = st.session_state.clustering_results
                
                if 'risk_scores' in st.session_state:
                    report_data['risk_analysis'] = st.session_state.risk_analysis
                
                if 'model_predictions' in st.session_state:
                    report_data['model_performance'] = st.session_state.model_performance
                
                if 'attack_results' in st.session_state:
                    report_data['attack_analysis'] = st.session_state.attack_analysis
                
                st.session_state.report_data = report_data
                
                st.success("✅ 分析报告生成完成！")
                
        except Exception as e:
            st.error(f"❌ 分析报告生成失败: {e}")
            st.exception(e)
    
    # 显示分析结果
    if st.session_state.report_data is not None:
        st.markdown("### 📈 分析结果")
        
        report_data = st.session_state.report_data
        
        # 报告概览
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("分析时间", report_data['timestamp'])
        
        with col2:
            st.metric("记录数", f"{report_data['data_info']['total_records']:,}")
        
        with col3:
            st.metric("特征数", report_data['data_info']['total_features'])
        
        with col4:
            if report_data['data_info']['fraud_rate'] is not None:
                st.metric("欺诈率", f"{report_data['data_info']['fraud_rate']}%")
            else:
                st.metric("欺诈率", "N/A")
        
        # SHAP分析结果
        if st.session_state.shap_analysis is not None:
            st.markdown("#### 🎯 SHAP可解释性分析")
            
            shap_analysis = st.session_state.shap_analysis
            
            # 特征重要性
            if 'feature_importance' in shap_analysis:
                importance_df = pd.DataFrame(shap_analysis['feature_importance'])
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                # 显示前10个重要特征
                top_features = importance_df.head(10)
                
                fig = px.bar(
                    top_features,
                    x='feature',
                    y='importance',
                    title="SHAP特征重要性排序",
                    labels={'feature': '特征', 'importance': '重要性'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 特征重要性表格
                st.dataframe(top_features, use_container_width=True)
            
            # 特征贡献分析
            if 'feature_contributions' in shap_analysis:
                st.markdown("**特征贡献分析**")
                
                # 选择要分析的样本
                if len(shap_analysis['feature_contributions']) > 0:
                    sample_index = st.selectbox(
                        "选择样本查看SHAP贡献",
                        range(len(shap_analysis['feature_contributions'])),
                        format_func=lambda x: f"样本 {x+1}"
                    )
                    
                    if 0 <= sample_index < len(shap_analysis['feature_contributions']):
                        sample_contributions = shap_analysis['feature_contributions'][sample_index]
                        
                        # 创建瀑布图
                        contrib_df = pd.DataFrame(sample_contributions)
                        contrib_df = contrib_df.sort_values('contribution', ascending=False)
                        
                        fig = px.bar(
                            contrib_df,
                            x='feature',
                            y='contribution',
                            title=f"样本 {sample_index+1} SHAP特征贡献",
                            labels={'feature': '特征', 'contribution': '贡献值'},
                            color='contribution',
                            color_continuous_scale='RdBu'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # LIME分析结果
        if st.session_state.lime_analysis is not None:
            st.markdown("#### 🔍 LIME局部解释分析")
            
            lime_analysis = st.session_state.lime_analysis
            
            # 局部解释
            if 'local_explanations' in lime_analysis:
                st.markdown("**局部解释分析**")
                
                # 选择要分析的样本
                if len(lime_analysis['local_explanations']) > 0:
                    lime_sample_index = st.selectbox(
                        "选择样本查看LIME解释",
                        range(len(lime_analysis['local_explanations'])),
                        format_func=lambda x: f"样本 {x+1}"
                    )
                    
                    if 0 <= lime_sample_index < len(lime_analysis['local_explanations']):
                        local_explanation = lime_analysis['local_explanations'][lime_sample_index]
                        
                        # 显示局部解释
                        st.markdown(f"**样本 {lime_sample_index+1} 的LIME解释**")
                        
                        if 'feature_weights' in local_explanation:
                            weights_df = pd.DataFrame(local_explanation['feature_weights'])
                            weights_df = weights_df.sort_values('weight', ascending=False)
                            
                            fig = px.bar(
                                weights_df,
                                x='feature',
                                y='weight',
                                title=f"样本 {lime_sample_index+1} LIME特征权重",
                                labels={'feature': '特征', 'weight': '权重'},
                                color='weight',
                                color_continuous_scale='RdBu'
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        # 综合分析
        st.markdown("#### 📊 综合分析")
        
        # 创建综合分析图表
        analysis_summary = []
        
        # 数据质量分析
        if 'data_info' in report_data:
            data_info = report_data['data_info']
            analysis_summary.append({
                '分析项目': '数据质量',
                '记录数': data_info['total_records'],
                '特征数': data_info['total_features'],
                '数值特征': data_info['numeric_features'],
                '欺诈率': data_info['fraud_rate']
            })
        
        # 聚类分析
        if 'clustering_results' in report_data:
            clustering_results = report_data['clustering_results']
            analysis_summary.append({
                '分析项目': '聚类分析',
                '算法': clustering_results.get('algorithm', 'N/A'),
                '聚类数': clustering_results.get('n_clusters', 'N/A'),
                '特征数': len(clustering_results.get('features', [])),
                '记录数': len(st.session_state.cluster_labels) if 'cluster_labels' in st.session_state else 'N/A'
            })
        
        # 风险评分分析
        if 'risk_analysis' in report_data:
            risk_analysis = report_data['risk_analysis']
            analysis_summary.append({
                '分析项目': '风险评分',
                '平均评分': f"{risk_analysis['score_stats']['mean']:.2f}",
                '标准差': f"{risk_analysis['score_stats']['std']:.2f}",
                '最高评分': f"{risk_analysis['score_stats']['max']:.2f}",
                '最低评分': f"{risk_analysis['score_stats']['min']:.2f}"
            })
        
        # 模型性能分析
        if 'model_performance' in report_data:
            model_performance = report_data['model_performance']
            if model_performance:
                best_model = max(model_performance.items(), key=lambda x: x[1].get('accuracy', 0))[0]
                best_accuracy = model_performance[best_model].get('accuracy', 0)
                analysis_summary.append({
                    '分析项目': '模型性能',
                    '最佳模型': best_model,
                    '最佳准确率': f"{best_accuracy:.3f}",
                    '模型数量': len(model_performance),
                    '平均准确率': f"{np.mean([p.get('accuracy', 0) for p in model_performance.values()]):.3f}"
                })
        
        # 攻击分析
        if 'attack_analysis' in report_data:
            attack_analysis = report_data['attack_analysis']
            analysis_summary.append({
                '分析项目': '攻击检测',
                '检测到攻击': attack_analysis.get('total_attacks', 0),
                '攻击类型数': len(attack_analysis.get('attack_types', {})),
                '高危攻击率': f"{attack_analysis.get('severity_distribution', {}).get('高危', 0) / max(attack_analysis.get('total_attacks', 1), 1) * 100:.2f}%",
                '平均置信度': f"{np.mean([r.get('confidence', 0) for r in st.session_state.attack_results]) if 'attack_results' in st.session_state and st.session_state.attack_results else 0:.3f}"
            })
        
        # 显示综合分析表格
        if analysis_summary:
            summary_df = pd.DataFrame(analysis_summary)
            st.dataframe(summary_df, use_container_width=True)
        
        # 报告导出
        st.markdown("#### 📄 报告导出")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 导出PDF报告", type="primary", use_container_width=True):
                if report_data is None:
                    st.error("❌ 请先生成分析报告")
                else:
                    try:
                        report_generator = ReportGenerator()
                        pdf_path = report_generator.generate_pdf_report(report_data)
                        st.success(f"✅ PDF报告已生成: {pdf_path}")

                        # 提供下载链接
                        if os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as file:
                                st.download_button(
                                    label="📥 下载PDF报告",
                                    data=file.read(),
                                    file_name="fraud_detection_report.pdf",
                                    mime="application/pdf"
                                )
                        else:
                            st.warning("⚠️ PDF文件生成失败")
                    except Exception as e:
                        st.error(f"❌ PDF报告生成失败: {str(e)}")
                        st.info("💡 PDF报告功能正在开发中，请尝试其他格式")
        
        with col2:
            if st.button("📊 导出Excel报告", type="primary", use_container_width=True):
                if report_data is None:
                    st.error("❌ 请先生成分析报告")
                else:
                    try:
                        report_generator = ReportGenerator()
                        excel_path = report_generator.generate_excel_report(report_data)
                        st.success(f"✅ Excel报告已生成: {excel_path}")

                        # 提供下载链接
                        if os.path.exists(excel_path):
                            with open(excel_path, "rb") as file:
                                st.download_button(
                                    label="📥 下载Excel报告",
                                    data=file.read(),
                                    file_name="fraud_detection_report.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        else:
                            st.warning("⚠️ Excel文件生成失败")
                    except Exception as e:
                        st.error(f"❌ Excel报告生成失败: {str(e)}")

        with col3:
            if st.button("📊 导出HTML报告", type="primary", use_container_width=True):
                if report_data is None:
                    st.error("❌ 请先生成分析报告")
                else:
                    try:
                        report_generator = ReportGenerator()
                        html_path = report_generator.generate_html_report(report_data)
                        st.success(f"✅ HTML报告已生成: {html_path}")

                        # 提供下载链接
                        if os.path.exists(html_path):
                            with open(html_path, "r", encoding="utf-8") as file:
                                st.download_button(
                                    label="📥 下载HTML报告",
                                    data=file.read(),
                                    file_name="fraud_detection_report.html",
                                    mime="text/html"
                                )
                        else:
                            st.warning("⚠️ HTML文件生成失败")
                    except Exception as e:
                        st.error(f"❌ HTML报告生成失败: {str(e)}")
        
        # 报告完成
        st.markdown("---")
        st.success("🎉 恭喜！您已完成整个电商欺诈风险预测系统的分析流程！")
        st.markdown("""
        **分析流程总结：**
        1. ✅ 数据上传与预处理
        2. ✅ 特征工程与风险特征生成
        3. ✅ 聚类分析与异常群体识别
        4. ✅ 风险评分与等级分类
        5. ✅ 多模型预测与性能对比
        6. ✅ 攻击类型分类与防护建议
        7. ✅ 综合分析报告与可解释性分析
        
        **系统特点：**
        - 多维度风险评估
        - 智能特征工程
        - 多模型集成预测
        - 攻击类型识别
        - 可解释性分析
        - 综合报告生成
        """)
    
    else:
        # 显示分析报告说明
        st.markdown("### 📝 分析报告说明")
        
        st.markdown("""
        **报告内容：**
        
        1. **数据分析报告**
           - 数据质量评估
           - 特征分布分析
           - 数据统计信息
        
        2. **模型性能报告**
           - 多模型对比分析
           - 性能指标详解
           - 预测结果统计
        
        3. **风险评估报告**
           - 风险等级分布
           - 风险评分分析
           - 风险趋势分析
        
        4. **攻击分析报告**
           - 攻击类型统计
           - 攻击严重程度分析
           - 攻击特征分析
        
        5. **可解释性报告**
           - SHAP全局特征重要性
           - LIME局部解释分析
           - 特征贡献分析
        
        6. **防护建议报告**
           - 总体防护建议
           - 针对性防护措施
           - 紧急处理建议
        
        **可解释性分析：**
        - **SHAP分析**: 全局特征重要性和局部特征贡献
        - **LIME分析**: 单个预测的局部线性解释
        - **特征交互**: 特征之间的相互作用分析
        - **决策路径**: 模型决策过程的可视化
        """) 