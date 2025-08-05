# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Quick Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run main.py
```

## Troubleshooting

### Common Issues

#### 1. "No module named 'lime'" Error

**Solution:**
```bash
pip install lime
```

If the above doesn't work, try:
```bash
pip install --upgrade lime
```

#### 2. "No module named 'shap'" Error

**Solution:**
```bash
pip install shap
```

#### 3. Other Missing Dependencies

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Dependency Check

Run the dependency check script to verify all packages are installed:

```bash
python check_dependencies.py
```

## Manual Installation of Key Packages

If automatic installation fails, install key packages manually:

```bash
# Core packages
pip install streamlit pandas numpy scikit-learn

# Machine learning packages
pip install catboost xgboost

# Explainability packages
pip install shap lime

# Visualization packages
pip install plotly matplotlib seaborn

# Other utilities
pip install imbalanced-learn reportlab openpyxl joblib psutil
```

## Virtual Environment (Recommended)

It's recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv fraud_detection_env

# Activate virtual environment
# On Windows:
fraud_detection_env\Scripts\activate
# On macOS/Linux:
source fraud_detection_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run main.py
```

## System Requirements

- **RAM:** Minimum 4GB, Recommended 8GB+
- **Storage:** At least 2GB free space
- **OS:** Windows 10+, macOS 10.14+, or Linux

## Support

If you encounter any issues:

1. Check this installation guide
2. Run the dependency check script
3. Check the GitHub issues page
4. Create a new issue with error details

## Version Information

- Python: 3.8+
- Streamlit: 1.28.0+
- Key ML packages: See requirements.txt for specific versions
