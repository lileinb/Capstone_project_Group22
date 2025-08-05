#!/usr/bin/env python3
"""
Smart Startup Script for Fraud Detection System
Automatically checks and installs dependencies before starting the application
"""

import sys
import subprocess
import importlib
import os

def check_and_install_package(package_name, import_name=None):
    """Check if package is installed, install if missing"""
    if import_name is None:
        import_name = package_name

    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        print(f"Installing missing package: {package_name}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return True
        except subprocess.CalledProcessError:
            print(f"Failed to install {package_name}")
            return False

def main():
    """Main startup function"""
    print("Starting Fraud Detection System...")

    # Define packages
    critical_packages = [
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("plotly", "plotly")
    ]

    optional_packages = [
        ("shap", "shap"),
        ("catboost", "catboost"),
        ("xgboost", "xgboost"),
        ("seaborn", "seaborn"),
        ("imbalanced-learn", "imblearn"),
        ("reportlab", "reportlab"),
        ("openpyxl", "openpyxl"),
        ("matplotlib", "matplotlib"),
        ("joblib", "joblib"),
        ("psutil", "psutil")
    ]

    # Run comprehensive pre-launch check
    print("Running pre-launch check...")
    try:
        result = subprocess.run([sys.executable, "pre_launch_check.py"],
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            print("Pre-launch check failed. Please address the issues above.")
            return 1
    except Exception as e:
        print(f"Pre-launch check failed to run: {e}")
        print("Falling back to basic dependency check...")

        # Check and install critical packages
        print("\nChecking critical packages...")
        critical_failed = []
        for package_name, import_name in critical_packages:
            if not check_and_install_package(package_name, import_name):
                critical_failed.append(package_name)

        if critical_failed:
            print(f"\nCritical packages failed to install: {', '.join(critical_failed)}")
            print("Please install them manually:")
            for package in critical_failed:
                print(f"  pip install {package}")
            return 1

    # If pre-launch check passed, start the application directly
    print("\nDependencies check completed!")
    print("Starting Streamlit application...")

    # Start the Streamlit application
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"])
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"\nFailed to start application: {e}")
        print("Try running manually: streamlit run main.py")
        return 1

    return 0



if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
