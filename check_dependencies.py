#!/usr/bin/env python3
"""
Dependency Check Script
Verifies that all required packages are installed and working
"""

import sys
import importlib
from typing import Dict, List, Tuple

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed and importable
    
    Args:
        package_name: Name of the package (for display)
        import_name: Name to use for import (if different from package_name)
    
    Returns:
        Tuple of (success, message)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        return True, f"[OK] {package_name} ({version})"
    except ImportError as e:
        return False, f"[MISSING] {package_name} - {str(e)}"
    except Exception as e:
        return False, f"[ERROR] {package_name} - {str(e)}"

def main():
    """Main dependency check function"""
    print("Checking Dependencies for Fraud Detection System")
    print("=" * 60)
    
    # Define required packages
    required_packages = [
        ("Streamlit", "streamlit"),
        ("Pandas", "pandas"),
        ("NumPy", "numpy"),
        ("Scikit-learn", "sklearn"),
        ("CatBoost", "catboost"),
        ("XGBoost", "xgboost"),
        ("SHAP", "shap"),
        ("LIME", "lime"),
        ("Plotly", "plotly"),
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        ("Imbalanced-learn", "imblearn"),
        ("ReportLab", "reportlab"),
        ("OpenPyXL", "openpyxl"),
        ("Joblib", "joblib"),
        ("PSUtil", "psutil")
    ]
    
    # Check each package
    success_count = 0
    failed_packages = []
    
    for package_name, import_name in required_packages:
        success, message = check_package(package_name, import_name)
        print(message)
        
        if success:
            success_count += 1
        else:
            failed_packages.append(package_name)
    
    print("\n" + "=" * 60)
    print(f"Summary: {success_count}/{len(required_packages)} packages installed successfully")

    if failed_packages:
        print(f"\nFailed packages: {', '.join(failed_packages)}")
        print("\nTo install missing packages, run:")
        print("pip install -r requirements.txt")
        print("\nOr install individually:")
        for package in failed_packages:
            if package == "LIME":
                print(f"pip install lime")
            elif package == "SHAP":
                print(f"pip install shap")
            elif package == "Imbalanced-learn":
                print(f"pip install imbalanced-learn")
            elif package == "Scikit-learn":
                print(f"pip install scikit-learn")
            else:
                print(f"pip install {package.lower()}")

        return 1
    else:
        print("\nAll dependencies are installed and working correctly!")
        print("You can now run the application with: streamlit run main.py")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
