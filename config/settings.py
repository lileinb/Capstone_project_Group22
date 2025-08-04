"""
Behavioral Feature-Based E-commerce User Big Data Driven Risk Scoring Model System Configuration File
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"
USER_TRAINED_MODELS_DIR = MODELS_DIR / "user_trained"

# Report directories
REPORTS_DIR = PROJECT_ROOT / "reports"
TEMPLATES_DIR = REPORTS_DIR / "templates"
GENERATED_REPORTS_DIR = REPORTS_DIR / "generated"

# Create necessary directories
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR,
                 MODELS_DIR, PRETRAINED_MODELS_DIR, USER_TRAINED_MODELS_DIR,
                 REPORTS_DIR, TEMPLATES_DIR, GENERATED_REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "dataset1": {
        "name": "Fraudulent_E-Commerce_Transaction_Data_sampled_50000.csv",
        "path": "Dataset/Fraudulent_E-Commerce_Transaction_Data_sampled_50000.csv",
        "records": 50000,
        "purpose": "Pre-training models"
    },
    "dataset2": {
        "name": "Fraudulent_E-Commerce_Transaction_Data_2_sampled_50000.csv",
        "path": "Dataset/Fraudulent_E-Commerce_Transaction_Data_2_sampled_50000.csv",
        "records": 23634,
        "purpose": "Validation and testing"
    }
}

# Model configuration
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

# Risk level configuration
RISK_LEVELS = {
    "low": {"min": 0, "max": 30, "label": "Low Risk", "color": "green"},
    "medium": {"min": 31, "max": 60, "label": "Medium Risk", "color": "yellow"},
    "high": {"min": 61, "max": 80, "label": "High Risk", "color": "orange"},
    "very_high": {"min": 81, "max": 100, "label": "Very High Risk", "color": "red"}
}

# Attack type configuration
ATTACK_TYPES = {
    "account_takeover": {
        "name": "Account Takeover Attack",
        "english_name": "Account Takeover",
        "severity_levels": ["High", "Medium", "Low"],
        "detection_features": ["New Device", "Old Account", "Large Transaction", "Abnormal Time"]
    },
    "identity_theft": {
        "name": "Identity Theft Attack",
        "english_name": "Identity Theft",
        "severity_levels": ["High", "Medium", "Low"],
        "detection_features": ["Address Mismatch", "Abnormal Payment", "Age Mismatch", "IP Anomaly"]
    },
    "bulk_fraud": {
        "name": "Bulk Fraud Attack",
        "english_name": "Bulk Fraud",
        "severity_levels": ["High", "Medium", "Low"],
        "detection_features": ["Similar IP", "Multiple Transactions in Short Time", "Similar Pattern", "Bulk Registration"]
    },
    "testing_attack": {
        "name": "Testing Attack",
        "english_name": "Testing Attack",
        "severity_levels": ["High", "Medium", "Low"],
        "detection_features": ["Small Multiple Transactions", "Multiple Payment Methods", "Rapid Succession", "New Account"]
    }
}

# Clustering configuration
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

# Feature engineering configuration
FEATURE_ENGINEERING_CONFIG = {
    "original_features": 16,
    "target_risk_features": 20,
    "feature_categories": [
        "Time Risk Features",
        "Amount Risk Features",
        "Device Geographic Features",
        "Account Behavior Features"
    ]
}

# Risk scoring weight configuration
RISK_WEIGHTS = {
    "amount_weight": 0.3,
    "time_weight": 0.2,
    "device_weight": 0.2,
    "account_weight": 0.15,
    "geographic_weight": 0.15
}

# Page configuration
PAGE_CONFIG = {
    "upload_page": "Data Upload and Preprocessing",
    "feature_analysis_page": "Feature Engineering and Clustering Analysis",
    "risk_scoring_page": "Risk Scoring and Level Classification",
    "model_prediction_page": "Model Prediction and Result Display",
    "attack_analysis_page": "Attack Type Analysis and Protection Recommendations",
    "report_page": "Analysis Report and Explainability"
}