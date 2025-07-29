"""
Behavioral Feature-Based E-commerce User Big Data Driven Risk Scoring Model System Configuration File
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# 模型目录
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"
USER_TRAINED_MODELS_DIR = MODELS_DIR / "user_trained"

# 报告目录
REPORTS_DIR = PROJECT_ROOT / "reports"
TEMPLATES_DIR = REPORTS_DIR / "templates"
GENERATED_REPORTS_DIR = REPORTS_DIR / "generated"

# 创建必要的目录
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR,
                 MODELS_DIR, PRETRAINED_MODELS_DIR, USER_TRAINED_MODELS_DIR,
                 REPORTS_DIR, TEMPLATES_DIR, GENERATED_REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据集配置
DATASET_CONFIG = {
    "dataset1": {
        "name": "Fraudulent_E-Commerce_Transaction_Data_sampled_50000.csv",
        "path": "Dataset/Fraudulent_E-Commerce_Transaction_Data_sampled_50000.csv",
        "records": 50000,
        "purpose": "预训练模型"
    },
    "dataset2": {
        "name": "Fraudulent_E-Commerce_Transaction_Data_2_sampled_50000.csv", 
        "path": "Dataset/Fraudulent_E-Commerce_Transaction_Data_2_sampled_50000.csv",
        "records": 23634,
        "purpose": "验证和测试"
    }
}

# 模型配置
MODEL_CONFIG = {
    "catboost": {
        "name": "CatBoost",
        "filename": "catboost_model.cbm",
        "params": {
            "iterations": 1000,
            "learning_rate": 0.1,
            "depth": 6,
            "l2_leaf_reg": 3,
            "random_seed": 42
        },
        "expected_accuracy": "85-90%"
    },
    "xgboost": {
        "name": "XGBoost", 
        "filename": "xgboost_model.pkl",
        "params": {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        "expected_accuracy": "84-89%"
    },

    "ensemble": {
        "name": "Ensemble Model",
        "filename": "ensemble_model.pkl",
        "strategy": "weighted_voting",
        "expected_accuracy": "87-92%"
    }
}

# 风险等级配置
RISK_LEVELS = {
    "low": {"min": 0, "max": 30, "label": "低风险", "color": "green"},
    "medium": {"min": 31, "max": 60, "label": "中风险", "color": "yellow"},
    "high": {"min": 61, "max": 80, "label": "高风险", "color": "orange"},
    "very_high": {"min": 81, "max": 100, "label": "极高风险", "color": "red"}
}

# 攻击类型配置
ATTACK_TYPES = {
    "account_takeover": {
        "name": "账户接管攻击",
        "english_name": "Account Takeover",
        "severity_levels": ["高危", "中危", "低危"],
        "detection_features": ["新设备", "老账户", "大额交易", "异常时间"]
    },
    "identity_theft": {
        "name": "身份盗用攻击", 
        "english_name": "Identity Theft",
        "severity_levels": ["高危", "中危", "低危"],
        "detection_features": ["地址不匹配", "异常支付", "年龄不符", "IP异常"]
    },
    "bulk_fraud": {
        "name": "批量欺诈攻击",
        "english_name": "Bulk Fraud", 
        "severity_levels": ["高危", "中危", "低危"],
        "detection_features": ["相似IP", "短时间多笔", "相似模式", "批量注册"]
    },
    "testing_attack": {
        "name": "测试性攻击",
        "english_name": "Testing Attack",
        "severity_levels": ["高危", "中危", "低危"], 
        "detection_features": ["小额多笔", "多种支付", "快速连续", "新账户"]
    }
}

# 聚类配置
CLUSTERING_CONFIG = {
    "kmeans": {
        "name": "K-means",
        "n_clusters_range": (2, 10),
        "random_state": 42
    },
    "dbscan": {
        "name": "DBSCAN", 
        "eps": 0.5,
        "min_samples": 5
    },
    "gaussian_mixture": {
        "name": "Gaussian Mixture",
        "n_components_range": (2, 8),
        "random_state": 42
    }
}

# 特征工程配置
FEATURE_ENGINEERING_CONFIG = {
    "original_features": 16,
    "target_risk_features": 20,
    "feature_categories": [
        "时间风险特征",
        "金额风险特征", 
        "设备地理特征",
        "账户行为特征"
    ]
}

# 评分权重配置
RISK_WEIGHTS = {
    "amount_weight": 0.3,
    "time_weight": 0.2,
    "device_weight": 0.2,
    "account_weight": 0.15,
    "geographic_weight": 0.15
}

# 页面配置
PAGE_CONFIG = {
    "upload_page": "数据上传与预处理",
    "feature_analysis_page": "特征工程与聚类分析", 
    "risk_scoring_page": "风险评分与等级划分",
    "model_prediction_page": "模型预测与结果展示",
    "attack_analysis_page": "攻击类型分析与防护建议",
    "report_page": "分析报告与可解释性"
} 