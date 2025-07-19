#!/bin/bash
# 🎯 智能风险预测与欺诈检测系统 - Git配置和上传脚本
# 适用于Linux/Mac系统

echo "========================================"
echo "🎯 智能风险预测系统 Git 配置脚本"
echo "========================================"
echo

# 检查是否已经是Git仓库
if [ -d ".git" ]; then
    echo "✅ 检测到现有Git仓库"
else
    echo "📁 初始化新的Git仓库..."
    git init
    echo "✅ Git仓库初始化完成"
fi

echo
echo "📋 添加文件到Git..."

# 添加重要文件
git add README.md
git add NEW_README.md
git add requirements.txt
git add main.py
git add .gitignore

# 添加后端核心模块
git add backend/prediction/
git add backend/risk_scoring/
git add backend/ml_models/

# 添加前端模块
git add frontend/pages/
git add frontend/components/

# 添加配置文件
git add config/

# 添加数据集（如果存在且不太大）
if [ -d "Dataset" ]; then
    echo "📊 添加数据集文件..."
    git add Dataset/*.csv
fi

echo "✅ 文件添加完成"

echo
echo "📝 创建初始提交..."
git commit -m "🎯 Initial commit: Intelligent Risk Prediction & Fraud Detection System

✨ Features:
- Individual risk analysis with 0-100 scoring
- Four-tier risk stratification (Low/Medium/High/Critical)
- Attack type inference (Account Takeover, Identity Theft, Bulk Fraud, Testing Attack)
- Dynamic threshold adjustment
- Interactive visualization
- Protection recommendations

🏗️ Architecture:
- Backend: Python + Machine Learning models
- Frontend: Streamlit interface
- Risk Scoring: Enhanced unsupervised algorithms
- Prediction: Individual risk predictor with dynamic thresholds

📊 Current Status:
- ✅ Unsupervised risk analysis completed
- ✅ Dynamic threshold optimization implemented
- 🔄 Supervised learning integration in progress
- 🔄 Frontend internationalization in progress

🚀 Ready for deployment and further development"

echo "✅ 初始提交完成"

echo
echo "🌐 Git仓库配置完成！"
echo
echo "📋 下一步操作："
echo "1. 在GitHub/GitLab上创建新仓库"
echo "2. 复制仓库URL"
echo "3. 运行以下命令连接远程仓库："
echo "   git remote add origin [YOUR_REPOSITORY_URL]"
echo "   git branch -M main"
echo "   git push -u origin main"
echo
echo "💡 或者运行 ./git_push.sh 脚本进行推送（需要先设置远程仓库）"
echo

read -p "按Enter键继续..."
