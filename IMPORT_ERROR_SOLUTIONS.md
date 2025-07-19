# 导入错误解决方案

## 问题描述

系统经常出现以下导入错误：
```
ImportError: cannot import name 'pages' from 'frontend'
```

## 常见原因分析

### 1. Python缓存问题 🔄
- **原因**: Python缓存了旧的模块信息
- **表现**: 修改代码后错误仍然存在
- **解决**: 清除 `__pycache__` 目录

### 2. 路径配置问题 📁
- **原因**: Python无法找到项目模块
- **表现**: 间歇性导入失败
- **解决**: 正确设置 `sys.path`

### 3. 文件权限问题 🔐
- **原因**: 文件或目录权限不足
- **表现**: 某些模块无法访问
- **解决**: 检查文件权限

### 4. 环境变量问题 ⚙️
- **原因**: `PYTHONPATH` 设置不正确
- **表现**: 模块路径解析失败
- **解决**: 设置正确的环境变量

## 解决方案

### 🚀 快速解决（推荐）

1. **使用新的启动脚本**:
   ```bash
   python start_app.py
   ```
   或双击 `start.bat` (Windows)

2. **手动清理缓存**:
   ```bash
   python diagnose_import_issue.py
   ```
   选择清理缓存选项

### 🔧 手动解决步骤

#### 步骤1: 清除Python缓存
```bash
# Windows
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"

# Linux/Mac
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

#### 步骤2: 重启Python环境
- 关闭所有Python进程
- 关闭IDE/编辑器
- 重新打开命令行

#### 步骤3: 检查项目结构
确保以下文件存在：
```
project/
├── main.py
├── frontend/
│   ├── __init__.py
│   └── pages/
│       ├── __init__.py
│       ├── upload_page.py
│       ├── feature_analysis_page.py
│       ├── clustering_page.py
│       ├── risk_scoring_page.py
│       ├── pseudo_labeling_page.py
│       ├── model_prediction_page.py
│       ├── attack_analysis_page.py
│       └── report_page.py
└── backend/
    └── ...
```

#### 步骤4: 验证导入
```python
python -c "import frontend.pages.upload_page; print('导入成功')"
```

### 🛠️ 高级解决方案

#### 方案1: 使用绝对导入
修改 `main.py` 中的导入方式：
```python
# 原来的方式
import frontend.pages.upload_page as upload_page

# 改为绝对导入
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontend.pages import upload_page
```

#### 方案2: 使用importlib动态导入
```python
import importlib

def safe_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        st.error(f"模块导入失败: {e}")
        return None

# 使用方式
upload_page = safe_import("frontend.pages.upload_page")
if upload_page:
    upload_page.show()
```

#### 方案3: 设置PYTHONPATH
```bash
# Windows
set PYTHONPATH=%CD%
streamlit run main.py

# Linux/Mac
export PYTHONPATH=$(pwd)
streamlit run main.py
```

## 预防措施

### 1. 使用虚拟环境
```bash
python -m venv fraud_detection_env
# Windows
fraud_detection_env\Scripts\activate
# Linux/Mac
source fraud_detection_env/bin/activate

pip install -r requirements.txt
```

### 2. 定期清理缓存
在项目根目录创建清理脚本：
```python
# clean_cache.py
import os
import shutil

for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        shutil.rmtree(os.path.join(root, '__pycache__'))
        print(f"已清理: {os.path.join(root, '__pycache__')}")
```

### 3. 使用requirements.txt
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
seaborn>=0.12.0
matplotlib>=3.7.0
```

### 4. 配置IDE
在PyCharm/VSCode中设置项目根目录为源码根目录。

## 故障排除检查清单

- [ ] Python版本 >= 3.8
- [ ] 所有必需包已安装
- [ ] 项目结构完整
- [ ] `__init__.py` 文件存在
- [ ] 文件权限正确
- [ ] 缓存已清理
- [ ] 环境变量设置正确
- [ ] IDE配置正确

## 常见错误信息及解决

### 错误1: `ModuleNotFoundError: No module named 'frontend'`
**解决**: 检查当前工作目录和Python路径

### 错误2: `ImportError: cannot import name 'pages'`
**解决**: 检查 `frontend/__init__.py` 和 `frontend/pages/__init__.py`

### 错误3: `AttributeError: module has no attribute 'show'`
**解决**: 检查页面模块是否包含 `show()` 函数

### 错误4: `PermissionError: [WinError 5] Access is denied`
**解决**: 以管理员权限运行或检查文件权限

## 联系支持

如果以上方案都无法解决问题，请提供：
1. 完整的错误信息
2. Python版本
3. 操作系统信息
4. 项目目录结构截图
5. 运行 `python diagnose_import_issue.py` 的输出

---

**最后更新**: 2025-01-17  
**适用版本**: Python 3.8+, Streamlit 1.28+
