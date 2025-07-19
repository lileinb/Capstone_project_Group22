@echo off
echo 🛠️ 修复Streamlit缓存问题
echo ================================

echo 🧹 清除Python缓存...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul

echo 🧹 清除Streamlit缓存...
if exist "%USERPROFILE%\.streamlit" rd /s /q "%USERPROFILE%\.streamlit"
if exist ".streamlit" rd /s /q ".streamlit"

echo ✅ 缓存清除完成!
echo.
echo 💡 请按以下步骤操作:
echo 1. 在运行Streamlit的终端中按 Ctrl+C 停止应用
echo 2. 运行: streamlit run main.py
echo 3. 刷新浏览器页面
echo.
pause
