"""
LIME Explainer Module
For local linear interpretability analysis
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Safe import for LIME
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME package not available. Please install it using: pip install lime")

class LIMEExplainer:
    def __init__(self, model=None):
        """
        Optional model input, otherwise model needs to be passed during analysis
        """
        self.model = model

    def analyze(self, X: pd.DataFrame, y: pd.Series = None, sample_size: int = 500) -> Dict:
        """
        Perform LIME local explanation on input data
        Args:
            X: Feature data
            y: Labels (optional)
            sample_size: Sample size for analysis
        Returns:
            Analysis result dictionary
        """
        # Check if LIME is available
        if not LIME_AVAILABLE:
            logger.error("LIME package is not installed. Please install it using: pip install lime")
            return {
                'error': 'LIME package not available',
                'message': 'Please install LIME using: pip install lime',
                'local_explanations': []
            }
        # Sampling
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
            if y is not None:
                y_sample = y.loc[X_sample.index]
            else:
                y_sample = None
        else:
            X_sample = X
            y_sample = y

        # Auto-select model (prioritize CatBoost/XGBoost/RandomForest)
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
            logger.error("Failed to auto-load model, please ensure available pre-trained models exist")
            return {
                'error': 'No model available',
                'message': 'Please ensure pre-trained models are available',
                'local_explanations': []
            }

        # LIME explainer
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_sample.values,
                feature_names=X_sample.columns.tolist(),
                class_names=['Non-Fraud', 'Fraud'],
                mode='classification',
                discretize_continuous=True
            )

            # Local explanations (first 10 samples)
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

        except Exception as e:
            logger.error(f"LIME analysis failed: {e}")
            return {
                'error': str(e),
                'message': 'LIME analysis failed',
                'local_explanations': []
            }