"""
Dependency Error Page
Displays helpful information when dependencies are missing
"""

import streamlit as st
import sys
import subprocess
import os

def show_dependency_error(missing_packages=None, error_details=None):
    """Show dependency error page with helpful solutions"""
    
    st.markdown("# ❌ Dependency Error")
    st.markdown("---")
    
    if missing_packages:
        st.error(f"Missing packages: {', '.join(missing_packages)}")
    
    if error_details:
        st.error(f"Error details: {error_details}")
    
    st.markdown("## 🔧 Quick Solutions")
    
    # Solution tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Quick Fix", "📋 Manual Install", "🔍 Diagnosis", "📚 Help"])
    
    with tab1:
        st.markdown("### Automatic Installation")
        st.markdown("Click the button below to automatically install missing dependencies:")
        
        if st.button("🔄 Install Missing Dependencies", type="primary"):
            with st.spinner("Installing dependencies..."):
                try:
                    # Install from requirements.txt
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        st.success("✅ Dependencies installed successfully!")
                        st.info("🔄 Please refresh the page (F5) to continue")
                        st.balloons()
                    else:
                        st.error("❌ Installation failed")
                        st.code(result.stderr)
                        
                except subprocess.TimeoutExpired:
                    st.error("❌ Installation timed out")
                except Exception as e:
                    st.error(f"❌ Installation error: {e}")
    
    with tab2:
        st.markdown("### Manual Installation Commands")
        
        st.markdown("**Option 1: Install all dependencies**")
        st.code("pip install -r requirements.txt")
        
        if missing_packages:
            st.markdown("**Option 2: Install specific missing packages**")
            install_cmd = f"pip install {' '.join(missing_packages)}"
            st.code(install_cmd)
        
        st.markdown("**Option 3: Install critical packages only**")
        st.code("pip install streamlit pandas numpy plotly scikit-learn")
        
        st.markdown("**Option 4: Install with user permissions**")
        st.code("pip install --user -r requirements.txt")
        
        st.markdown("**Option 5: Upgrade existing packages**")
        st.code("pip install --upgrade -r requirements.txt")
    
    with tab3:
        st.markdown("### System Diagnosis")
        
        # Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        st.info(f"🐍 Python Version: {python_version}")
        
        # Check if pip is available
        try:
            pip_result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                      capture_output=True, text=True)
            if pip_result.returncode == 0:
                st.success("✅ pip is available")
                st.code(pip_result.stdout.strip())
            else:
                st.error("❌ pip is not available")
        except Exception as e:
            st.error(f"❌ pip check failed: {e}")
        
        # Check requirements.txt
        if os.path.exists("requirements.txt"):
            st.success("✅ requirements.txt found")
            with open("requirements.txt", "r") as f:
                requirements = f.read()
            st.code(requirements)
        else:
            st.error("❌ requirements.txt not found")
        
        # Run dependency check
        if st.button("🔍 Run Full Dependency Check"):
            with st.spinner("Checking dependencies..."):
                try:
                    result = subprocess.run([sys.executable, "check_dependencies.py"], 
                                          capture_output=True, text=True, timeout=60)
                    st.code(result.stdout)
                    if result.stderr:
                        st.error("Errors:")
                        st.code(result.stderr)
                except Exception as e:
                    st.error(f"Dependency check failed: {e}")
    
    with tab4:
        st.markdown("### 📚 Help & Documentation")
        
        st.markdown("#### Common Issues and Solutions")
        
        with st.expander("🔍 'No module named lime' Error"):
            st.markdown("""
            **Cause:** The LIME package is not installed
            
            **Solutions:**
            1. `pip install lime`
            2. `pip install --user lime`
            3. `conda install -c conda-forge lime` (if using conda)
            """)
        
        with st.expander("🔍 'No module named shap' Error"):
            st.markdown("""
            **Cause:** The SHAP package is not installed
            
            **Solutions:**
            1. `pip install shap`
            2. `pip install --user shap`
            3. `conda install -c conda-forge shap` (if using conda)
            """)
        
        with st.expander("🔍 Permission Denied Error"):
            st.markdown("""
            **Cause:** Insufficient permissions to install packages
            
            **Solutions:**
            1. Use `--user` flag: `pip install --user -r requirements.txt`
            2. Use virtual environment
            3. Run as administrator (Windows) or with sudo (Linux/Mac)
            """)
        
        with st.expander("🔍 Network/Proxy Issues"):
            st.markdown("""
            **Cause:** Network connectivity or proxy issues
            
            **Solutions:**
            1. `pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org`
            2. Configure proxy settings
            3. Use alternative package index
            """)
        
        st.markdown("#### 🆘 Still Need Help?")
        st.markdown("""
        1. **Check the documentation:** README.md, INSTALLATION.md
        2. **Run the diagnostic tools:** `python check_dependencies.py`
        3. **Use the smart startup:** `python start.py`
        4. **Create an issue** on GitHub with:
           - Your operating system
           - Python version
           - Full error message
           - Output of dependency check
        """)
        
        st.markdown("#### 🔗 Useful Links")
        st.markdown("""
        - [Python Installation Guide](https://www.python.org/downloads/)
        - [pip Documentation](https://pip.pypa.io/en/stable/)
        - [Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
        """)

def show():
    """Main function to display the dependency error page"""
    show_dependency_error()
