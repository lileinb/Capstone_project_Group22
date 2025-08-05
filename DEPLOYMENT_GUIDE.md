# Deployment Guide for GitHub Users

## Problem: "No module named 'lime'" Error

When users download the project from GitHub and run it, they may encounter dependency-related errors. This guide provides comprehensive solutions.

## Root Cause Analysis

The error occurs because:
1. The `lime` package (and potentially other packages) are not installed in the user's environment
2. The project has dependencies that need to be installed separately
3. Different Python environments may have different package availability

## Solutions (In Order of Recommendation)

### Solution 1: Automatic Dependency Installation (Recommended)

**Step 1:** Check what's missing
```bash
python check_dependencies.py
```

**Step 2:** Install all dependencies
```bash
pip install -r requirements.txt
```

**Step 3:** Verify installation
```bash
python check_dependencies.py
```

### Solution 2: Manual Package Installation

If automatic installation fails, install packages individually:

```bash
# Core ML packages
pip install lime
pip install shap
pip install scikit-learn
pip install catboost
pip install xgboost

# Data processing
pip install pandas
pip install numpy
pip install imbalanced-learn

# Visualization
pip install plotly
pip install matplotlib
pip install seaborn

# Web framework
pip install streamlit

# Utilities
pip install reportlab
pip install openpyxl
pip install joblib
pip install psutil
```

### Solution 3: Virtual Environment (Best Practice)

Create an isolated environment:

```bash
# Create virtual environment
python -m venv fraud_detection_env

# Activate it
# Windows:
fraud_detection_env\Scripts\activate
# macOS/Linux:
source fraud_detection_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run main.py
```

### Solution 4: Using Conda (Alternative)

If you prefer conda:

```bash
# Create conda environment
conda create -n fraud_detection python=3.9

# Activate environment
conda activate fraud_detection

# Install packages
conda install -c conda-forge streamlit pandas numpy scikit-learn plotly matplotlib seaborn
pip install lime shap catboost xgboost imbalanced-learn reportlab openpyxl joblib psutil

# Run application
streamlit run main.py
```

## Error-Specific Solutions

### "No module named 'lime'"
```bash
pip install lime
```

### "No module named 'shap'"
```bash
pip install shap
```

### "No module named 'catboost'"
```bash
pip install catboost
```

### "No module named 'imblearn'"
```bash
pip install imbalanced-learn
```

## System-Specific Instructions

### Windows Users
```bash
# Use Command Prompt or PowerShell
pip install -r requirements.txt
streamlit run main.py
```

### macOS Users
```bash
# May need to use pip3
pip3 install -r requirements.txt
streamlit run main.py
```

### Linux Users
```bash
# May need sudo for system-wide installation
sudo pip install -r requirements.txt
# Or use user installation
pip install --user -r requirements.txt
streamlit run main.py
```

## Verification Steps

1. **Run dependency check:**
   ```bash
   python check_dependencies.py
   ```

2. **Test import in Python:**
   ```python
   import lime
   import shap
   import streamlit
   print("All packages imported successfully!")
   ```

3. **Run the application:**
   ```bash
   streamlit run main.py
   ```

## Common Issues and Solutions

### Issue: Permission Denied
**Solution:** Use `--user` flag
```bash
pip install --user -r requirements.txt
```

### Issue: Package Version Conflicts
**Solution:** Use virtual environment or upgrade packages
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Network/Proxy Issues
**Solution:** Use trusted hosts
```bash
pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
```

## For Project Maintainers

To prevent these issues for future users:

1. **Keep requirements.txt updated**
2. **Test in clean environments**
3. **Provide clear installation instructions**
4. **Use the dependency check script**

## Support

If you still encounter issues:
1. Check the error message carefully
2. Run `python check_dependencies.py`
3. Try the solutions in order
4. Create a GitHub issue with:
   - Your operating system
   - Python version (`python --version`)
   - Full error message
   - Output of dependency check script
