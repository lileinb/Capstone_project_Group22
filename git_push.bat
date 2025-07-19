@echo off
REM 🎯 智能风险预测与欺诈检测系统 - Git推送脚本

echo ========================================
echo 🚀 智能风险预测系统 Git 推送脚本
echo ========================================
echo.

REM 检查是否有远程仓库
git remote -v > nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误：未配置远程仓库
    echo 💡 请先运行以下命令设置远程仓库：
    echo    git remote add origin [YOUR_REPOSITORY_URL]
    echo.
    pause
    exit /b 1
)

echo 📋 当前远程仓库：
git remote -v
echo.

REM 检查当前状态
echo 📊 检查当前Git状态...
git status --porcelain > nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Git状态正常
) else (
    echo ❌ Git状态异常，请检查
    git status
    pause
    exit /b 1
)

echo.
echo 📝 添加所有更改...
git add .

echo.
echo 💬 请输入提交消息（或按Enter使用默认消息）：
set /p commit_msg="提交消息: "

if "%commit_msg%"=="" (
    set commit_msg=🔄 Update: System improvements and bug fixes
)

echo.
echo 📝 创建提交...
git commit -m "%commit_msg%"

if %errorlevel% neq 0 (
    echo ⚠️ 没有新的更改需要提交
) else (
    echo ✅ 提交创建成功
)

echo.
echo 🚀 推送到远程仓库...
git push origin main

if %errorlevel% equ 0 (
    echo ✅ 推送成功！
    echo 🌐 您的代码已成功上传到远程仓库
) else (
    echo ❌ 推送失败
    echo 💡 可能的解决方案：
    echo 1. 检查网络连接
    echo 2. 检查远程仓库权限
    echo 3. 尝试先拉取远程更改：git pull origin main
    echo 4. 如果有冲突，解决后重新推送
)

echo.
echo 📊 当前分支状态：
git branch -v

echo.
pause
