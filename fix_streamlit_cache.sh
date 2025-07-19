#!/bin/bash

echo "🛠️ 修复Streamlit缓存问题"
echo "================================"

echo "🧹 清除Python缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

echo "🧹 清除Streamlit缓存..."
rm -rf ~/.streamlit 2>/dev/null
rm -rf .streamlit 2>/dev/null

echo "✅ 缓存清除完成!"
echo ""
echo "💡 请按以下步骤操作:"
echo "1. 在运行Streamlit的终端中按 Ctrl+C 停止应用"
echo "2. 运行: streamlit run main.py"
echo "3. 刷新浏览器页面"
echo ""
