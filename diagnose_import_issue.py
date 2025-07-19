#!/usr/bin/env python3
"""
诊断导入问题的脚本
"""

import sys
import os
import importlib.util

def check_python_path():
    """检查Python路径设置"""
    print("🔍 检查Python路径设置...")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Python路径:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    print()

def check_file_structure():
    """检查文件结构"""
    print("📁 检查文件结构...")
    
    # 检查关键文件是否存在
    files_to_check = [
        "main.py",
        "frontend/__init__.py",
        "frontend/pages/__init__.py",
        "frontend/pages/upload_page.py",
        "frontend/pages/feature_analysis_page.py",
        "frontend/pages/clustering_page.py",
        "frontend/pages/risk_scoring_page.py",
        "frontend/pages/pseudo_labeling_page.py",
        "frontend/pages/model_prediction_page.py",
        "frontend/pages/attack_analysis_page.py",
        "frontend/pages/report_page.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 文件不存在")
    print()

def check_module_imports():
    """检查模块导入"""
    print("🔧 检查模块导入...")
    
    # 添加项目根目录到路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    modules_to_test = [
        "frontend",
        "frontend.pages",
        "frontend.pages.upload_page",
        "frontend.pages.feature_analysis_page",
        "frontend.pages.clustering_page",
        "frontend.pages.risk_scoring_page",
        "frontend.pages.pseudo_labeling_page",
        "frontend.pages.model_prediction_page",
        "frontend.pages.attack_analysis_page",
        "frontend.pages.report_page"
    ]
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name} - 导入成功")
            
            # 检查是否有show函数
            if hasattr(module, 'show'):
                print(f"   └─ 包含 show() 函数")
            else:
                print(f"   └─ ⚠️ 缺少 show() 函数")
                
        except ImportError as e:
            print(f"❌ {module_name} - 导入失败: {e}")
        except Exception as e:
            print(f"⚠️ {module_name} - 其他错误: {e}")
    print()

def check_init_files():
    """检查__init__.py文件内容"""
    print("📄 检查__init__.py文件...")
    
    init_files = [
        "frontend/__init__.py",
        "frontend/pages/__init__.py"
    ]
    
    for init_file in init_files:
        if os.path.exists(init_file):
            print(f"📄 {init_file}:")
            try:
                with open(init_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        print(f"   内容: {content[:100]}...")
                    else:
                        print("   内容: (空文件)")
            except Exception as e:
                print(f"   读取失败: {e}")
        else:
            print(f"❌ {init_file} - 文件不存在")
    print()

def check_permissions():
    """检查文件权限"""
    print("🔐 检查文件权限...")
    
    files_to_check = [
        "frontend",
        "frontend/pages",
        "frontend/__init__.py",
        "frontend/pages/__init__.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                # 检查读权限
                if os.access(file_path, os.R_OK):
                    print(f"✅ {file_path} - 可读")
                else:
                    print(f"❌ {file_path} - 不可读")
                    
                # 如果是目录，检查执行权限
                if os.path.isdir(file_path):
                    if os.access(file_path, os.X_OK):
                        print(f"   └─ 可执行")
                    else:
                        print(f"   └─ ❌ 不可执行")
                        
            except Exception as e:
                print(f"⚠️ {file_path} - 权限检查失败: {e}")
        else:
            print(f"❌ {file_path} - 文件不存在")
    print()

def suggest_fixes():
    """建议修复方案"""
    print("💡 建议修复方案:")
    print("1. 清除Python缓存:")
    print("   - 删除所有 __pycache__ 文件夹")
    print("   - 删除所有 .pyc 文件")
    print()
    print("2. 重新启动Python环境:")
    print("   - 关闭所有Python进程")
    print("   - 重新启动命令行/IDE")
    print()
    print("3. 检查文件编码:")
    print("   - 确保所有Python文件使用UTF-8编码")
    print("   - 检查文件中是否有特殊字符")
    print()
    print("4. 重新安装依赖:")
    print("   - pip install --upgrade streamlit")
    print("   - pip install --upgrade pandas numpy")
    print()
    print("5. 使用绝对导入:")
    print("   - 在main.py中使用完整的模块路径")
    print("   - 确保sys.path设置正确")

def clean_cache():
    """清除Python缓存"""
    print("🧹 清除Python缓存...")
    
    cache_dirs = []
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                cache_dirs.append(os.path.join(root, dir_name))
    
    if cache_dirs:
        print(f"发现 {len(cache_dirs)} 个缓存目录:")
        for cache_dir in cache_dirs:
            print(f"  - {cache_dir}")
        
        response = input("是否删除这些缓存目录? (y/n): ")
        if response.lower() == 'y':
            import shutil
            for cache_dir in cache_dirs:
                try:
                    shutil.rmtree(cache_dir)
                    print(f"✅ 已删除: {cache_dir}")
                except Exception as e:
                    print(f"❌ 删除失败: {cache_dir} - {e}")
        else:
            print("跳过缓存清理")
    else:
        print("未发现缓存目录")

def main():
    """主函数"""
    print("🔍 Python导入问题诊断工具")
    print("=" * 50)
    
    check_python_path()
    check_file_structure()
    check_init_files()
    check_permissions()
    check_module_imports()
    
    print("=" * 50)
    suggest_fixes()
    
    print("\n" + "=" * 50)
    clean_cache()

if __name__ == "__main__":
    main()
