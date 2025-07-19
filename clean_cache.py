#!/usr/bin/env python3
"""
清理缓存脚本
"""

import os
import shutil
import sys

def clean_pycache():
    """清理Python缓存文件"""
    print("🧹 清理Python缓存文件...")
    
    cache_dirs = []
    
    # 遍历项目目录查找__pycache__
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_path = os.path.join(root, '__pycache__')
            cache_dirs.append(cache_path)
    
    print(f"   发现 {len(cache_dirs)} 个缓存目录")
    
    # 删除缓存目录
    removed_count = 0
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
            removed_count += 1
            print(f"   ✅ 删除: {cache_dir}")
        except Exception as e:
            print(f"   ❌ 删除失败: {cache_dir} - {e}")
    
    print(f"   🎯 成功删除 {removed_count}/{len(cache_dirs)} 个缓存目录")

def clean_pyc_files():
    """清理.pyc文件"""
    print("\n🧹 清理.pyc文件...")
    
    pyc_files = []
    
    # 遍历项目目录查找.pyc文件
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                pyc_path = os.path.join(root, file)
                pyc_files.append(pyc_path)
    
    print(f"   发现 {len(pyc_files)} 个.pyc文件")
    
    # 删除.pyc文件
    removed_count = 0
    for pyc_file in pyc_files:
        try:
            os.remove(pyc_file)
            removed_count += 1
        except Exception as e:
            print(f"   ❌ 删除失败: {pyc_file} - {e}")
    
    print(f"   🎯 成功删除 {removed_count}/{len(pyc_files)} 个.pyc文件")

def clean_temp_files():
    """清理临时文件"""
    print("\n🧹 清理临时文件...")
    
    temp_patterns = ['.tmp', '.temp', '~']
    temp_files = []
    
    # 遍历项目目录查找临时文件
    for root, dirs, files in os.walk('.'):
        for file in files:
            if any(file.endswith(pattern) for pattern in temp_patterns):
                temp_path = os.path.join(root, file)
                temp_files.append(temp_path)
    
    print(f"   发现 {len(temp_files)} 个临时文件")
    
    # 删除临时文件
    removed_count = 0
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            removed_count += 1
        except Exception as e:
            print(f"   ❌ 删除失败: {temp_file} - {e}")
    
    print(f"   🎯 成功删除 {removed_count}/{len(temp_files)} 个临时文件")

def main():
    """主函数"""
    print("🚀 开始清理项目缓存")
    print("=" * 40)
    
    # 清理Python缓存
    clean_pycache()
    
    # 清理.pyc文件
    clean_pyc_files()
    
    # 清理临时文件
    clean_temp_files()
    
    print("\n" + "=" * 40)
    print("🎉 缓存清理完成！")
    print("\n✅ 已清理的内容:")
    print("1. Python __pycache__ 目录")
    print("2. .pyc 编译文件")
    print("3. 临时文件")
    print("\n💡 建议重启Python环境以确保清理生效")

if __name__ == "__main__":
    main()
