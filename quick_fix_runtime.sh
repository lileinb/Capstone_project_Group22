#!/bin/bash

echo "========================================"
echo "电商欺诈风险预测系统 - 快速修复脚本"
echo "========================================"

echo ""
echo "🧹 步骤1: 清理Python缓存文件..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "✅ 缓存文件清理完成"

echo ""
echo "📝 步骤2: 创建Streamlit配置..."
mkdir -p .streamlit
cat > .streamlit/config.toml << EOF
[server]
fileWatcherType = "none"
runOnSave = false

[global]
developmentMode = false

[browser]
gatherUsageStats = false
EOF
echo "✅ Streamlit配置文件已创建"

echo ""
echo "🔧 步骤3: 运行Python修复脚本..."
python fix_runtime_errors.py

echo ""
echo "🎉 修复完成！"
echo ""
echo "📋 修复内容:"
echo "  ✅ 清理了Python缓存文件"
echo "  ✅ 创建了Streamlit配置文件"
echo "  ✅ 修复了分类数据处理错误"
echo "  ✅ 修复了NumPy运行时警告"
echo ""
echo "🔄 下一步操作:"
echo "  1. 重启Streamlit应用: streamlit run main.py"
echo "  2. 如果问题仍然存在，请重启Python环境"
echo ""
