@echo off
chcp 65001 >nul
title 欺诈检测系统启动器

echo.
echo 🛡️ 欺诈检测系统启动器
echo ================================

echo.
echo 📍 当前目录: %CD%

echo.
echo 🐍 检查Python环境...
python --version
if errorlevel 1 (
    echo ❌ Python未安装或未添加到PATH
    echo 💡 请安装Python 3.8或更高版本
    pause
    exit /b 1
)

echo.
echo 🚀 启动应用...
python start_app.py

echo.
echo 👋 应用已退出
pause
