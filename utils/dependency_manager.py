"""
Dependency Manager
Handles automatic dependency checking and installation
"""

import sys
import subprocess
import importlib
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DependencyManager:
    """Manages project dependencies with automatic installation"""
    
    def __init__(self):
        self.critical_packages = {
            'streamlit': 'streamlit',
            'pandas': 'pandas', 
            'numpy': 'numpy',
            'scikit-learn': 'sklearn',
            'plotly': 'plotly'
        }
        
        self.optional_packages = {
            'lime': 'lime',
            'shap': 'shap',
            'catboost': 'catboost',
            'xgboost': 'xgboost',
            'seaborn': 'seaborn',
            'imbalanced-learn': 'imblearn',
            'reportlab': 'reportlab',
            'openpyxl': 'openpyxl',
            'matplotlib': 'matplotlib',
            'joblib': 'joblib',
            'psutil': 'psutil'
        }
        
        self.package_status = {}
    
    def check_package(self, package_name: str, import_name: str) -> Tuple[bool, str, Optional[str]]:
        """
        Check if a package is available
        
        Returns:
            (is_available, status_message, version)
        """
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'Unknown')
            return True, f"✅ {package_name}", version
        except ImportError as e:
            return False, f"❌ {package_name} - Not installed", None
        except Exception as e:
            return False, f"⚠️ {package_name} - {str(e)}", None
    
    def install_package(self, package_name: str) -> bool:
        """Install a package using pip"""
        try:
            logger.info(f"Installing {package_name}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name, "--quiet"
            ])
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package_name}: {e}")
            return False
    
    def check_all_dependencies(self) -> Dict[str, Dict]:
        """Check all dependencies and return status"""
        results = {
            'critical': {},
            'optional': {},
            'summary': {
                'critical_missing': [],
                'optional_missing': [],
                'all_critical_available': True
            }
        }
        
        # Check critical packages
        for package_name, import_name in self.critical_packages.items():
            available, message, version = self.check_package(package_name, import_name)
            results['critical'][package_name] = {
                'available': available,
                'message': message,
                'version': version
            }
            if not available:
                results['summary']['critical_missing'].append(package_name)
                results['summary']['all_critical_available'] = False
        
        # Check optional packages
        for package_name, import_name in self.optional_packages.items():
            available, message, version = self.check_package(package_name, import_name)
            results['optional'][package_name] = {
                'available': available,
                'message': message,
                'version': version
            }
            if not available:
                results['summary']['optional_missing'].append(package_name)
        
        return results
    
    def auto_install_missing(self, install_optional: bool = False) -> bool:
        """Automatically install missing packages"""
        results = self.check_all_dependencies()
        
        success = True
        
        # Install critical packages
        for package_name in results['summary']['critical_missing']:
            if not self.install_package(package_name):
                success = False
        
        # Install optional packages if requested
        if install_optional:
            for package_name in results['summary']['optional_missing']:
                self.install_package(package_name)  # Don't fail on optional packages
        
        return success
    
    def get_installation_commands(self) -> Dict[str, List[str]]:
        """Get installation commands for missing packages"""
        results = self.check_all_dependencies()
        
        commands = {
            'critical': [],
            'optional': [],
            'all_at_once': []
        }
        
        # Critical packages
        if results['summary']['critical_missing']:
            commands['critical'] = [
                f"pip install {' '.join(results['summary']['critical_missing'])}"
            ]
        
        # Optional packages
        if results['summary']['optional_missing']:
            commands['optional'] = [
                f"pip install {' '.join(results['summary']['optional_missing'])}"
            ]
        
        # All missing packages
        all_missing = results['summary']['critical_missing'] + results['summary']['optional_missing']
        if all_missing:
            commands['all_at_once'] = [
                f"pip install {' '.join(all_missing)}"
            ]
        
        return commands
    
    def safe_import(self, module_path: str, package_name: str = None):
        """Safely import a module with automatic installation attempt"""
        if package_name is None:
            package_name = module_path.split('.')[-1]
        
        try:
            return importlib.import_module(module_path)
        except ImportError:
            logger.warning(f"Module {module_path} not available, attempting to install {package_name}")
            if self.install_package(package_name):
                try:
                    return importlib.import_module(module_path)
                except ImportError:
                    logger.error(f"Failed to import {module_path} even after installation")
                    return None
            return None

# Global instance
dependency_manager = DependencyManager()

def check_and_install_dependencies(auto_install: bool = True) -> bool:
    """
    Check and optionally install dependencies
    
    Args:
        auto_install: Whether to automatically install missing packages
    
    Returns:
        True if all critical dependencies are available
    """
    if auto_install:
        return dependency_manager.auto_install_missing(install_optional=False)
    else:
        results = dependency_manager.check_all_dependencies()
        return results['summary']['all_critical_available']
