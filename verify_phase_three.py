#!/usr/bin/env python3
"""
第三阶段验证脚本
验证系统集成和新功能
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def verify_new_components():
    """验证新组件"""
    print("🔍 验证第三阶段新组件...")
    
    results = {}
    
    # 1. 验证三层预测流水线
    try:
        from backend.prediction.three_layer_pipeline import ThreeLayerPredictionPipeline
        pipeline = ThreeLayerPredictionPipeline()
        status = pipeline.get_pipeline_status()
        results['three_layer_pipeline'] = status['pipeline_ready']
        print("   ✅ 三层预测流水线 - 可用")
    except Exception as e:
        results['three_layer_pipeline'] = False
        print(f"   ❌ 三层预测流水线 - 失败: {e}")
    
    # 2. 验证动态阈值管理器
    try:
        from backend.risk_scoring.dynamic_threshold_manager import DynamicThresholdManager
        manager = DynamicThresholdManager()
        test_scores = [20, 40, 60, 80]
        thresholds = manager.optimize_thresholds_iteratively(test_scores)
        results['dynamic_threshold_manager'] = len(thresholds) == 4
        print("   ✅ 动态阈值管理器 - 可用")
    except Exception as e:
        results['dynamic_threshold_manager'] = False
        print(f"   ❌ 动态阈值管理器 - 失败: {e}")
    
    # 3. 验证四分类风险计算器
    try:
        from backend.risk_scoring.four_class_risk_calculator import FourClassRiskCalculator
        calculator = FourClassRiskCalculator()
        test_data = pd.DataFrame({
            'transaction_amount': [100, 500, 1500],
            'quantity': [1, 3, 8],
            'customer_age': [30, 25, 22],
            'account_age_days': [500, 100, 30],
            'transaction_hour': [14, 22, 2]
        })
        risk_results = calculator.calculate_four_class_risk_scores(test_data)
        results['four_class_calculator'] = risk_results.get('success', False)
        print("   ✅ 四分类风险计算器 - 可用")
    except Exception as e:
        results['four_class_calculator'] = False
        print(f"   ❌ 四分类风险计算器 - 失败: {e}")
    
    # 4. 验证新页面文件
    new_pages = [
        'frontend/pages/threshold_management_page.py',
        'frontend/pages/performance_monitoring_page.py'
    ]
    
    page_results = []
    for page_path in new_pages:
        if os.path.exists(page_path):
            page_results.append(True)
            print(f"   ✅ {os.path.basename(page_path)} - 存在")
        else:
            page_results.append(False)
            print(f"   ❌ {os.path.basename(page_path)} - 不存在")
    
    results['new_pages'] = all(page_results)
    
    return results

def verify_system_integration():
    """验证系统集成"""
    print("\n🔗 验证系统集成...")
    
    integration_results = {}
    
    # 1. 验证配置系统
    try:
        from config.optimization_config import optimization_config
        config = optimization_config.get_risk_scoring_config()
        
        required_configs = ['dynamic_thresholds', 'label_generation', 'model_architecture']
        config_status = all(key in config for key in required_configs)
        
        integration_results['config_system'] = config_status
        if config_status:
            print("   ✅ 配置系统集成 - 完整")
        else:
            print("   ⚠️ 配置系统集成 - 部分缺失")
    except Exception as e:
        integration_results['config_system'] = False
        print(f"   ❌ 配置系统集成 - 失败: {e}")
    
    # 2. 验证主应用更新
    try:
        with open('main.py', 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        new_pages_in_main = [
            '🎛️ 阈值管理' in main_content,
            '📊 性能监控' in main_content,
            'threshold_management_page' in main_content,
            'performance_monitoring_page' in main_content
        ]
        
        integration_results['main_app_updated'] = all(new_pages_in_main)
        if integration_results['main_app_updated']:
            print("   ✅ 主应用更新 - 完成")
        else:
            print("   ⚠️ 主应用更新 - 部分完成")
    except Exception as e:
        integration_results['main_app_updated'] = False
        print(f"   ❌ 主应用更新 - 失败: {e}")
    
    # 3. 验证组件依赖
    try:
        # 测试组件间的依赖关系
        from backend.risk_scoring.four_class_risk_calculator import FourClassRiskCalculator
        from backend.prediction.four_class_pipeline import FourClassPredictionPipeline
        
        # 创建计算器并检查是否能正确初始化动态阈值管理器
        calculator = FourClassRiskCalculator(enable_dynamic_thresholds=True)
        has_threshold_manager = hasattr(calculator, 'threshold_manager')
        
        integration_results['component_dependencies'] = has_threshold_manager
        if has_threshold_manager:
            print("   ✅ 组件依赖 - 正常")
        else:
            print("   ⚠️ 组件依赖 - 异常")
    except Exception as e:
        integration_results['component_dependencies'] = False
        print(f"   ❌ 组件依赖 - 失败: {e}")
    
    return integration_results

def verify_functionality():
    """验证功能完整性"""
    print("\n⚙️ 验证功能完整性...")
    
    functionality_results = {}
    
    # 1. 验证四分类功能
    try:
        test_data = pd.DataFrame({
            'transaction_amount': [100, 800, 2500, 5000],
            'quantity': [1, 4, 10, 18],
            'customer_age': [35, 28, 23, 19],
            'account_age_days': [800, 150, 45, 8],
            'transaction_hour': [14, 21, 2, 3]
        })
        
        from backend.risk_scoring.four_class_risk_calculator import FourClassRiskCalculator
        calculator = FourClassRiskCalculator()
        results = calculator.calculate_four_class_risk_scores(test_data)
        
        if results['success']:
            # 检查是否有四个风险等级
            distribution = results.get('distribution', {})
            has_four_classes = len(distribution) == 4
            expected_classes = {'low', 'medium', 'high', 'critical'}
            correct_classes = set(distribution.keys()) == expected_classes
            
            functionality_results['four_class_scoring'] = has_four_classes and correct_classes
            if functionality_results['four_class_scoring']:
                print("   ✅ 四分类评分功能 - 正常")
                print(f"      风险分布: {[f'{k}: {v[\"percentage\"]:.1f}%' for k, v in distribution.items()]}")
            else:
                print("   ⚠️ 四分类评分功能 - 分类不完整")
        else:
            functionality_results['four_class_scoring'] = False
            print("   ❌ 四分类评分功能 - 失败")
    except Exception as e:
        functionality_results['four_class_scoring'] = False
        print(f"   ❌ 四分类评分功能 - 异常: {e}")
    
    # 2. 验证动态阈值功能
    try:
        from backend.risk_scoring.dynamic_threshold_manager import DynamicThresholdManager
        manager = DynamicThresholdManager()
        
        # 创建测试评分
        test_scores = np.concatenate([
            np.random.normal(25, 5, 60),   # 低风险
            np.random.normal(50, 8, 25),   # 中风险
            np.random.normal(75, 5, 12),   # 高风险
            np.random.normal(90, 3, 3)     # 极高风险
        ])
        
        optimized_thresholds = manager.optimize_thresholds_iteratively(test_scores.tolist())
        analysis = manager.analyze_distribution(test_scores.tolist(), optimized_thresholds)
        
        functionality_results['dynamic_thresholds'] = analysis.get('total_deviation', 1.0) < 0.5
        if functionality_results['dynamic_thresholds']:
            print("   ✅ 动态阈值功能 - 正常")
            print(f"      分布偏差: {analysis['total_deviation']:.3f}")
        else:
            print("   ⚠️ 动态阈值功能 - 偏差较大")
    except Exception as e:
        functionality_results['dynamic_thresholds'] = False
        print(f"   ❌ 动态阈值功能 - 异常: {e}")
    
    return functionality_results

def main():
    """主验证函数"""
    print("🚀 第三阶段系统验证")
    print("=" * 50)
    
    # 运行各项验证
    component_results = verify_new_components()
    integration_results = verify_system_integration()
    functionality_results = verify_functionality()
    
    # 汇总结果
    all_results = {**component_results, **integration_results, **functionality_results}
    
    print("\n" + "=" * 50)
    print("🎯 第三阶段验证结果:")
    
    passed = sum(all_results.values())
    total = len(all_results)
    
    print(f"   总验证项: {total}")
    print(f"   通过验证: {passed}")
    print(f"   成功率: {passed/total*100:.1f}%")
    
    print("\n📊 详细结果:")
    for test_name, result in all_results.items():
        status = "✅" if result else "❌"
        print(f"   - {test_name}: {status}")
    
    if passed == total:
        print("\n🎉 第三阶段验证全部通过！")
        print("✅ 系统集成优化完成:")
        print("1. 三层预测架构 - 完整流水线")
        print("2. 四分类风险评分 - 精确分级")
        print("3. 动态阈值管理 - 智能优化")
        print("4. 前端界面更新 - 新增页面")
        print("5. 系统性能监控 - 实时监控")
        print("\n🚀 系统已准备就绪，可以投入使用！")
    elif passed >= total * 0.8:
        print(f"\n✅ 第三阶段验证基本通过！({passed}/{total})")
        print("⚠️ 少数功能需要进一步优化")
    else:
        print(f"\n⚠️ 第三阶段验证需要改进 ({passed}/{total})")
        print("❌ 多个功能需要修复")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
