"""
SHAP解释器模块
用于全局和局部特征重要性分析
"""
import shap
import numpy as np
import pandas as pd
from typing import Any, Dict, List

class SHAPExplainer:
    def __init__(self, model=None):
        """
        可选传入模型，否则后续分析时需传入模型
        """
        self.model = model

    def analyze(self, X: pd.DataFrame, y: pd.Series = None, sample_size: int = 500) -> Dict:
        """
        对输入数据进行SHAP全局和局部解释
        Args:
            X: 特征数据
            y: 标签（可选）
            sample_size: 用于分析的样本数
        Returns:
            分析结果字典
        """
        # 采样
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
            if y is not None:
                y_sample = y.loc[X_sample.index]
            else:
                y_sample = None
        else:
            X_sample = X
            y_sample = y

        # 自动选择模型（本系统优先CatBoost/XGBoost/RandomForest）
        if self.model is None:
            try:
                from backend.ml_models.model_manager import ModelManager
                model_manager = ModelManager()
                self.model = model_manager.load_model('catboost') or \
                             model_manager.load_model('xgboost') or \
                             model_manager.load_model('randomforest')
            except Exception:
                self.model = None
        if self.model is None:
            raise RuntimeError("未能自动加载模型，请确保有可用的预训练模型。")

        # 选择解释器
        try:
            explainer = shap.TreeExplainer(self.model)
        except Exception:
            explainer = shap.Explainer(self.model)

        # 计算SHAP值
        shap_values = explainer.shap_values(X_sample)
        # 兼容二分类和多分类
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # 全局特征重要性
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        # 局部特征贡献
        feature_contributions = []
        for i in range(min(10, len(X_sample))):
            contrib = [
                {'feature': X_sample.columns[j], 'contribution': shap_values[i, j]}
                for j in range(X_sample.shape[1])
            ]
            feature_contributions.append(contrib)

        return {
            'feature_importance': importance_df.to_dict(orient='records'),
            'feature_contributions': feature_contributions
        } 