@echo off
echo  Starting Fraud Detection System...
echo  Starting Fraud Detection System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo  Python is not installed or not in PATH
    echo  Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo  Python is available
echo.

REM Check if pip is available
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo  pip is not available
    echo  Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo  pip is available
echo.

REM Install/check dependencies
echo  Checking and installing dependencies...
python -m pip install -r requirements.txt --quiet

if errorlevel 1 (
    echo  Some packages may have failed to install
    echo  Trying to install critical packages individually...
    python -m pip install streamlit pandas numpy scikit-learn lime plotly --quiet
)

echo.
echo  Running dependency check...
python check_dependencies.py

echo.
echo  Starting Streamlit application...
echo  Press Ctrl+C to stop the application
echo.

REM Start the application
python -m streamlit run main.py

echo.
echo  Application stopped
pause
