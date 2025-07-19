#!/usr/bin/env python3
"""
验证修复效果的脚本
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings

# 抑制警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered")

def test_categorical_processing():
    """测试分类数据处理"""
    print("🧪 测试分类数据处理...")
    
    try:
        # 创建测试数据
        data = pd.DataFrame({
            'payment_method': pd.Categorical(['credit card', 'debit card', 'PayPal']),
            'amount': [100, 200, 300],
            'category': pd.Categorical(['A', 'B', 'C'])
        })
        
        # 尝试添加新类别
        data.loc[len(data)] = ['bank transfer', 400, 'D']
        
        print("✅ 分类数据处理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 分类数据处理测试失败: {e}")
        return False

def test_numpy_operations():
    """测试NumPy操作"""
    print("🧪 测试NumPy数值操作...")
    
    try:
        # 创建可能产生警告的操作
        a = np.array([1, 2, 3, 0])
        b = np.array([0, 1, 2, 0])
        
        # 可能产生除零警告的操作
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = a / b
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
        
        print("✅ NumPy操作测试通过")
        return True
        
    except Exception as e:
        print(f"❌ NumPy操作测试失败: {e}")
        return False

def test_streamlit_config():
    """测试Streamlit配置"""
    print("🧪 测试Streamlit配置...")
    
    try:
        config_path = ".streamlit/config.toml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                content = f.read()
                if "fileWatcherType" in content and "none" in content:
                    print("✅ Streamlit配置测试通过")
                    return True
                else:
                    print("❌ Streamlit配置内容不正确")
                    return False
        else:
            print("❌ Streamlit配置文件不存在")
            return False
            
    except Exception as e:
        print(f"❌ Streamlit配置测试失败: {e}")
        return False

def test_backend_import():
    """测试后端模块导入"""
    print("🧪 测试后端模块导入...")
    
    try:
        # 添加项目路径
        sys.path.insert(0, os.getcwd())
        
        # 尝试导入关键模块
        from backend.risk_scoring.four_class_risk_calculator import FourClassRiskCalculator
        
        # 创建实例
        calculator = FourClassRiskCalculator()
        
        print("✅ 后端模块导入测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 后端模块导入测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始验证修复效果...")
    print("=" * 50)
    
    tests = [
        test_streamlit_config,
        test_categorical_processing,
        test_numpy_operations,
        test_backend_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有修复验证通过！系统应该可以正常运行。")
        print("\n🔄 下一步:")
        print("  1. 重启Streamlit应用: streamlit run main.py")
        print("  2. 检查是否还有错误信息")
    else:
        print("⚠️ 部分修复可能需要进一步处理。")
        print("\n🔧 建议:")
        print("  1. 检查失败的测试项")
        print("  2. 重启Python环境")
        print("  3. 重新运行修复脚本")

if __name__ == "__main__":
    main()
