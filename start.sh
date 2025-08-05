#!/bin/bash

echo "Starting Fraud Detection System..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "[ERROR] Python is not installed"
        echo "Please install Python 3.8+ from https://python.org"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "[OK] Python is available ($PYTHON_CMD)"
echo

# Check if pip is available
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "[ERROR] pip is not available"
    echo "Please ensure pip is installed with Python"
    exit 1
fi

echo "[OK] pip is available"
echo

# Install/check dependencies
echo "Checking and installing dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo "[WARNING] Some packages may have failed to install"
    echo "Trying to install critical packages individually..."
    $PYTHON_CMD -m pip install streamlit pandas numpy scikit-learn lime plotly --quiet
fi

echo
echo "Running dependency check..."
$PYTHON_CMD check_dependencies.py

echo
echo "Starting Streamlit application..."
echo "Press Ctrl+C to stop the application"
echo

# Start the application
$PYTHON_CMD -m streamlit run main.py

echo
echo "Application stopped"
