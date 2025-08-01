"""
工作流程状态指示器组件
提供系统流程完成状态的可视化展示
"""

import streamlit as st
from typing import Dict, List, Tuple

def get_workflow_status() -> Dict[str, Dict]:
    """获取工作流程状态"""
    status = {}
    
    # 1. 数据上传和清理
    status['data_upload'] = {
        'name': '📁 Data Upload & Cleaning',
        'completed': ('uploaded_data' in st.session_state and st.session_state.uploaded_data is not None) and
                    ('cleaned_data' in st.session_state and st.session_state.cleaned_data is not None),
        'required': True,
        'description': 'Upload and clean raw transaction data'
    }
    
    # 2. 特征工程
    status['feature_engineering'] = {
        'name': '🔧 Feature Engineering',
        'completed': 'engineered_features' in st.session_state and st.session_state.engineered_features is not None,
        'required': True,
        'description': 'Generate risk-related features from raw data'
    }
    
    # 3. 聚类分析
    status['clustering'] = {
        'name': '📊 Clustering Analysis',
        'completed': 'clustering_results' in st.session_state and st.session_state.clustering_results is not None,
        'required': False,
        'description': 'Identify user behavior patterns and anomalous groups'
    }
    
    # 4. 风险评分
    status['risk_scoring'] = {
        'name': '🎯 Risk Scoring',
        'completed': 'four_class_risk_results' in st.session_state and st.session_state.four_class_risk_results is not None,
        'required': False,
        'description': 'Calculate comprehensive risk scores for transactions'
    }
    
    # 5. 伪标签生成
    status['pseudo_labeling'] = {
        'name': '🏷️ Pseudo Labeling',
        'completed': ('pseudo_labels' in st.session_state and st.session_state.pseudo_labels is not None) or
                    ('high_quality_labels' in st.session_state and st.session_state.high_quality_labels is not None),
        'required': False,
        'description': 'Generate high-quality pseudo labels for model training'
    }
    
    # 6. 模型训练/预测
    status['model_prediction'] = {
        'name': '🤖 Model Prediction',
        'completed': 'individual_risk_results' in st.session_state and st.session_state.individual_risk_results is not None,
        'required': False,
        'description': 'Train models and make risk predictions'
    }
    
    # 7. 攻击分析
    status['attack_analysis'] = {
        'name': '⚔️ Attack Analysis',
        'completed': 'attack_results' in st.session_state and st.session_state.attack_results is not None,
        'required': False,
        'description': 'Analyze attack patterns and threat types'
    }
    
    return status

def show_workflow_progress():
    """显示工作流程进度"""
    st.markdown("### 🔄 Workflow Progress")
    
    status = get_workflow_status()
    
    # 计算完成率
    total_steps = len(status)
    completed_steps = sum(1 for step in status.values() if step['completed'])
    required_steps = sum(1 for step in status.values() if step['required'])
    completed_required = sum(1 for step in status.values() if step['required'] and step['completed'])
    
    # 显示总体进度
    col1, col2, col3 = st.columns(3)
    
    with col1:
        progress = completed_steps / total_steps
        st.metric("Overall Progress", f"{progress:.1%}", f"{completed_steps}/{total_steps} steps")
    
    with col2:
        required_progress = completed_required / required_steps if required_steps > 0 else 1.0
        st.metric("Required Steps", f"{required_progress:.1%}", f"{completed_required}/{required_steps} steps")
    
    with col3:
        optional_completed = completed_steps - completed_required
        optional_total = total_steps - required_steps
        optional_progress = optional_completed / optional_total if optional_total > 0 else 1.0
        st.metric("Optional Steps", f"{optional_progress:.1%}", f"{optional_completed}/{optional_total} steps")
    
    # 显示详细状态
    st.markdown("#### 📋 Detailed Status")
    
    for step_key, step_info in status.items():
        col1, col2, col3 = st.columns([3, 1, 6])
        
        with col1:
            if step_info['completed']:
                st.success(f"✅ {step_info['name']}")
            elif step_info['required']:
                st.error(f"❌ {step_info['name']}")
            else:
                st.info(f"⏳ {step_info['name']}")
        
        with col2:
            if step_info['required']:
                st.markdown("**Required**")
            else:
                st.markdown("*Optional*")
        
        with col3:
            st.markdown(f"*{step_info['description']}*")

def show_next_steps():
    """显示建议的下一步操作"""
    status = get_workflow_status()
    
    # 找到下一个应该完成的步骤
    next_required = None
    next_optional = []
    
    for step_key, step_info in status.items():
        if not step_info['completed']:
            if step_info['required'] and next_required is None:
                next_required = step_info
            elif not step_info['required']:
                next_optional.append(step_info)
    
    if next_required:
        st.warning(f"⚠️ **Next Required Step**: {next_required['name']}")
        st.info(f"💡 {next_required['description']}")
    elif next_optional:
        st.info("🚀 **Suggested Next Steps** (Optional but recommended):")
        for step in next_optional[:2]:  # 只显示前两个建议
            st.info(f"   • {step['name']}: {step['description']}")
    else:
        st.success("🎉 **All workflow steps completed!** You can now generate comprehensive analysis reports.")

def show_workflow_dependencies():
    """显示工作流程依赖关系"""
    with st.expander("🔗 Workflow Dependencies", expanded=False):
        st.markdown("""
        **📊 Data Flow Dependencies:**
        
        1. **📁 Data Upload** → **🔧 Feature Engineering** (Required)
        2. **🔧 Feature Engineering** → **📊 Clustering Analysis** (Optional)
        3. **🔧 Feature Engineering** + **📊 Clustering** → **🎯 Risk Scoring** (Optional)
        4. **🔧 Feature Engineering** + **🎯 Risk Scoring** → **🏷️ Pseudo Labeling** (Enhanced)
        5. **🔧 Feature Engineering** + **🏷️ Pseudo Labeling** → **🤖 Model Prediction** (Enhanced)
        6. **🤖 Model Prediction** + **🎯 Risk Scoring** → **⚔️ Attack Analysis** (Enhanced)
        
        **🎯 Recommendation:**
        - Complete required steps first (Data Upload → Feature Engineering)
        - For best results, follow the complete workflow sequence
        - Optional steps significantly improve analysis quality
        """)

def show_compact_workflow_status():
    """显示紧凑版工作流程状态（用于侧边栏）"""
    status = get_workflow_status()
    
    st.markdown("**🔄 Workflow Status**")
    
    for step_key, step_info in status.items():
        if step_info['completed']:
            st.markdown(f"✅ {step_info['name'].split(' ')[0]}")
        elif step_info['required']:
            st.markdown(f"❌ {step_info['name'].split(' ')[0]}")
        else:
            st.markdown(f"⏳ {step_info['name'].split(' ')[0]}")
