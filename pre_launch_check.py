#!/usr/bin/env python3
"""
Pre-launch Check Script
Comprehensive dependency and environment check before starting the application
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"[ERROR] Python {version.major}.{version.minor} is not supported")
        print("Please upgrade to Python 3.8 or higher")
        return False

    print(f"[OK] Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                      capture_output=True, check=True)
        print("[OK] pip is available")
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] pip is not available")
        print("Please install pip or ensure it's in your PATH")
        return False

def check_requirements_file():
    """Check if requirements.txt exists"""
    req_file = Path("requirements.txt")
    if req_file.exists():
        print("[OK] requirements.txt found")
        return True
    else:
        print("[ERROR] requirements.txt not found")
        print("Please ensure requirements.txt is in the project root")
        return False

def check_critical_packages():
    """Check critical packages"""
    critical_packages = [
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("plotly", "plotly")
    ]
    
    missing_packages = []
    
    for package_name, import_name in critical_packages:
        try:
            importlib.import_module(import_name)
            print(f"[OK] {package_name}")
        except ImportError:
            print(f"[MISSING] {package_name}")
            missing_packages.append(package_name)
    
    return missing_packages

def check_optional_packages():
    """Check optional packages"""
    optional_packages = [
        ("lime", "lime"),
        ("shap", "shap"),
        ("catboost", "catboost"),
        ("xgboost", "xgboost"),
        ("scikit-learn", "sklearn")
    ]
    
    missing_packages = []
    
    for package_name, import_name in optional_packages:
        try:
            importlib.import_module(import_name)
            print(f"[OK] {package_name}")
        except ImportError:
            print(f"[OPTIONAL] {package_name} - Missing (optional)")
            missing_packages.append(package_name)
    
    return missing_packages

def install_packages(packages):
    """Install missing packages"""
    if not packages:
        return True

    print(f"\nInstalling missing packages: {', '.join(packages)}")

    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install"
        ] + packages + ["--quiet"], check=True)
        print("Installation completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        return False

def check_project_structure():
    """Check if project structure is correct"""
    required_dirs = [
        "frontend",
        "backend", 
        "utils"
    ]
    
    required_files = [
        "main.py",
        "requirements.txt"
    ]
    
    missing_items = []
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_items.append(f"Directory: {dir_name}")
    
    for file_name in required_files:
        if not Path(file_name).exists():
            missing_items.append(f"File: {file_name}")
    
    if missing_items:
        print("[ERROR] Project structure issues:")
        for item in missing_items:
            print(f"   - Missing {item}")
        return False

    print("[OK] Project structure is correct")
    return True

def main():
    """Main pre-launch check function"""
    print("Pre-launch Check for Fraud Detection System")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: Python version
    print("\n1. Checking Python version...")
    if check_python_version():
        checks_passed += 1

    # Check 2: pip availability
    print("\n2. Checking pip...")
    if check_pip():
        checks_passed += 1

    # Check 3: Project structure
    print("\n3. Checking project structure...")
    if check_project_structure():
        checks_passed += 1

    # Check 4: Requirements file
    print("\n4. Checking requirements file...")
    if check_requirements_file():
        checks_passed += 1

    # Check 5: Package dependencies
    print("\n5. Checking package dependencies...")
    print("Critical packages:")
    missing_critical = check_critical_packages()
    
    print("\nOptional packages:")
    missing_optional = check_optional_packages()
    
    if not missing_critical:
        checks_passed += 1
        print("[OK] All critical packages are available")
    else:
        print(f"Missing critical packages: {', '.join(missing_critical)}")

        # Attempt automatic installation
        print("\nAttempting to install missing critical packages...")
        if install_packages(missing_critical):
            # Re-check after installation
            print("Re-checking critical packages...")
            missing_after_install = check_critical_packages()
            if not missing_after_install:
                checks_passed += 1
                print("All critical packages are now available")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Pre-launch Check Summary: {checks_passed}/{total_checks} checks passed")

    if checks_passed == total_checks:
        print("All checks passed! You can now run the application.")
        print("Run: streamlit run main.py")
        return 0
    else:
        print("Some checks failed. Please address the issues above.")

        if missing_critical:
            print("\nTo fix missing packages, run:")
            print(f"pip install {' '.join(missing_critical)}")

        if missing_optional:
            print(f"\nOptional packages can be installed with:")
            print(f"pip install {' '.join(missing_optional)}")

        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
