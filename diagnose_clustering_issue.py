#!/usr/bin/env python3
"""
诊断聚类问题 - 模拟前端真实数据
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.clustering.cluster_analyzer import ClusterAnalyzer
from backend.feature_engineer.risk_features import RiskFeatureEngineer

def load_real_data():
    """加载真实的CSV数据（模拟前端加载的数据）"""
    try:
        # 尝试加载真实数据
        data_path = "data/fraud_detection_dataset.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"✅ 加载真实数据: {df.shape}")
            return df
        else:
            print("⚠️ 真实数据文件不存在，使用模拟数据")
            return create_realistic_data()
    except Exception as e:
        print(f"⚠️ 加载真实数据失败: {e}，使用模拟数据")
        return create_realistic_data()

def create_realistic_data():
    """创建更真实的数据（模拟前端可能遇到的数据）"""
    np.random.seed(42)
    n_samples = 1000
    
    # 创建基础数据
    data = {
        'transaction_amount': np.random.lognormal(4, 1.2, n_samples),
        'customer_age': np.random.normal(40, 15, n_samples),
        'account_age_days': np.random.exponential(365, n_samples),
        'transaction_hour': np.random.randint(0, 24, n_samples),
        'quantity': np.random.poisson(2, n_samples) + 1,
        'payment_method': np.random.choice(['credit card', 'debit card', 'bank transfer'], n_samples),
        'product_category': np.random.choice(['electronics', 'clothing', 'home'], n_samples),
        'device_used': np.random.choice(['desktop', 'mobile', 'tablet'], n_samples),
        'shipping_address': np.random.choice(['same', 'different'], n_samples, p=[0.8, 0.2]),
        'billing_address': np.random.choice(['same', 'different'], n_samples, p=[0.9, 0.1]),
        'is_fraudulent': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    return pd.DataFrame(data)

def diagnose_clustering_pipeline():
    """诊断完整的聚类流程"""
    print("🔍 开始诊断聚类流程")
    
    # 1. 加载数据
    print("\n📊 步骤1: 加载数据")
    raw_data = load_real_data()
    print(f"原始数据形状: {raw_data.shape}")
    print(f"原始特征: {list(raw_data.columns)}")
    
    # 2. 特征工程（模拟前端的特征工程）
    print("\n🔧 步骤2: 特征工程")
    try:
        risk_engineer = RiskFeatureEngineer()
        engineered_data = risk_engineer.engineer_risk_features(raw_data)
        print(f"工程化数据形状: {engineered_data.shape}")
        print(f"新增特征数: {len(engineered_data.columns) - len(raw_data.columns)}")
        print(f"工程化特征示例: {list(engineered_data.columns)[:10]}...")
    except Exception as e:
        print(f"❌ 特征工程失败: {e}")
        print(f"错误详情: {e}")
        engineered_data = raw_data
    
    # 3. 智能聚类
    print("\n🤖 步骤3: 智能聚类")
    cluster_analyzer = ClusterAnalyzer()
    
    try:
        intelligent_result = cluster_analyzer.intelligent_auto_clustering(engineered_data)
        
        if intelligent_result:
            print("✅ 智能聚类成功")
            print(f"   算法: {intelligent_result.get('algorithm', 'unknown')}")
            print(f"   聚类数: {intelligent_result.get('n_clusters', 0)}")
            print(f"   轮廓系数: {intelligent_result.get('silhouette_score', 0):.3f}")
            print(f"   选择特征数: {len(intelligent_result.get('selected_features', []))}")
            
            # 检查轮廓系数是否异常
            silhouette = intelligent_result.get('silhouette_score', 0)
            if silhouette > 0.8:
                print("⚠️ 警告: 轮廓系数异常高，可能存在计算错误")
            elif silhouette < 0.2:
                print("⚠️ 警告: 轮廓系数较低，聚类效果不佳")
            else:
                print("✅ 轮廓系数在合理范围内")
                
        else:
            print("❌ 智能聚类失败")
            intelligent_result = {}
            
    except Exception as e:
        print(f"❌ 智能聚类异常: {e}")
        import traceback
        traceback.print_exc()
        intelligent_result = {}
    
    # 4. 手动聚类对比
    print("\n🔧 步骤4: 手动聚类对比")
    try:
        manual_result = cluster_analyzer.analyze_clusters(engineered_data, algorithm='auto')
        
        if manual_result:
            manual_silhouette = manual_result.get('quality_metrics', {}).get('silhouette_score', 0)
            print("✅ 手动聚类成功")
            print(f"   算法: {manual_result.get('algorithm', 'unknown')}")
            print(f"   聚类数: {manual_result.get('cluster_count', 0)}")
            print(f"   轮廓系数: {manual_silhouette:.3f}")
        else:
            print("❌ 手动聚类失败")
            manual_result = {}
            
    except Exception as e:
        print(f"❌ 手动聚类异常: {e}")
        manual_result = {}
    
    # 5. 对比分析
    print("\n📊 步骤5: 对比分析")
    intelligent_silhouette = intelligent_result.get('silhouette_score', 0)
    manual_silhouette = manual_result.get('quality_metrics', {}).get('silhouette_score', 0)
    
    print(f"智能聚类轮廓系数: {intelligent_silhouette:.3f}")
    print(f"手动聚类轮廓系数: {manual_silhouette:.3f}")
    
    if intelligent_silhouette > manual_silhouette:
        improvement = ((intelligent_silhouette - manual_silhouette) / manual_silhouette) * 100 if manual_silhouette > 0 else 0
        print(f"✅ 智能聚类更好，提升 {improvement:.1f}%")
    elif manual_silhouette > intelligent_silhouette:
        degradation = ((manual_silhouette - intelligent_silhouette) / manual_silhouette) * 100 if manual_silhouette > 0 else 0
        print(f"❌ 智能聚类较差，下降 {degradation:.1f}%")
    else:
        print("⚖️ 两种方法效果相当")
    
    # 6. 问题诊断
    print("\n🔍 步骤6: 问题诊断")
    issues = []
    
    if intelligent_silhouette > 0.9:
        issues.append("智能聚类轮廓系数异常高，可能存在过拟合或计算错误")
    
    if intelligent_silhouette < 0.2:
        issues.append("智能聚类轮廓系数过低，特征选择或算法选择可能有问题")
    
    if abs(intelligent_silhouette - manual_silhouette) > 0.3:
        issues.append("智能聚类与手动聚类差异过大，可能存在实现不一致")
    
    if len(intelligent_result.get('selected_features', [])) > 15:
        issues.append("选择的特征数量过多，可能导致维度灾难")
    
    if len(intelligent_result.get('selected_features', [])) < 3:
        issues.append("选择的特征数量过少，可能信息不足")
    
    if issues:
        print("⚠️ 发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("✅ 未发现明显问题")
    
    # 7. 建议
    print("\n💡 步骤7: 优化建议")
    if intelligent_silhouette < 0.3:
        print("建议:")
        print("   1. 检查特征工程质量")
        print("   2. 调整特征选择策略")
        print("   3. 优化聚类算法参数")
        print("   4. 增加数据预处理步骤")
    elif intelligent_silhouette > 0.8:
        print("建议:")
        print("   1. 检查轮廓系数计算是否正确")
        print("   2. 验证聚类结果的业务意义")
        print("   3. 检查是否存在数据泄露")
    else:
        print("✅ 聚类效果良好，无需特别优化")
    
    return {
        'intelligent_result': intelligent_result,
        'manual_result': manual_result,
        'data_shape': engineered_data.shape,
        'issues': issues
    }

if __name__ == "__main__":
    results = diagnose_clustering_pipeline()
    
    print(f"\n🎯 诊断完成")
    print(f"数据规模: {results['data_shape']}")
    print(f"发现问题数: {len(results['issues'])}")
    
    if results['issues']:
        print("需要重点关注的问题:")
        for issue in results['issues']:
            print(f"  - {issue}")
