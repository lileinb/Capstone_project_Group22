@echo off
echo ========================================
echo 电商欺诈风险预测系统 - 快速修复脚本
echo ========================================

echo.
echo 🧹 步骤1: 清理Python缓存文件...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
echo ✅ 缓存文件清理完成

echo.
echo 📝 步骤2: 创建Streamlit配置...
if not exist ".streamlit" mkdir ".streamlit"
(
echo [server]
echo fileWatcherType = "none"
echo runOnSave = false
echo.
echo [global]
echo developmentMode = false
echo.
echo [browser]
echo gatherUsageStats = false
) > ".streamlit\config.toml"
echo ✅ Streamlit配置文件已创建

echo.
echo 🔧 步骤3: 运行Python修复脚本...
python fix_runtime_errors.py

echo.
echo 🎉 修复完成！
echo.
echo 📋 修复内容:
echo   ✅ 清理了Python缓存文件
echo   ✅ 创建了Streamlit配置文件
echo   ✅ 修复了分类数据处理错误
echo   ✅ 修复了NumPy运行时警告
echo.
echo 🔄 下一步操作:
echo   1. 重启Streamlit应用: streamlit run main.py
echo   2. 如果问题仍然存在，请重启Python环境
echo.
pause
