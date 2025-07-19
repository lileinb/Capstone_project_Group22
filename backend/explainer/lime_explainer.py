"""
LIME解释器模块
用于局部线性可解释性分析
"""
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from typing import Any, Dict, List

class LIMEExplainer:
    def __init__(self, model=None):
        """
        可选传入模型，否则后续分析时需传入模型
        """
        self.model = model

    def analyze(self, X: pd.DataFrame, y: pd.Series = None, sample_size: int = 500) -> Dict:
        """
        对输入数据进行LIME局部解释
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

        # LIME解释器
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_sample.values,
            feature_names=X_sample.columns.tolist(),
            class_names=['非欺诈', '欺诈'],
            mode='classification',
            discretize_continuous=True
        )

        # 局部解释（前10个样本）
        local_explanations = []
        for i in range(min(10, len(X_sample))):
            exp = explainer.explain_instance(
                X_sample.values[i],
                self.model.predict_proba,
                num_features=min(10, X_sample.shape[1])
            )
            feature_weights = [
                {'feature': f, 'weight': w} for f, w in exp.as_list()
            ]
            local_explanations.append({'feature_weights': feature_weights})

        return {
            'local_explanations': local_explanations
        } 