"""
SHAP Explainer Module
For global and local feature importance analysis
"""
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List

# Configure logging
logger = logging.getLogger(__name__)

# Safe import for SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP package not available. Please install it using: pip install shap")

class SHAPExplainer:
    def __init__(self, model=None):
        """
        Optional model input, otherwise model needs to be passed during analysis
        """
        self.model = model

    def analyze(self, X: pd.DataFrame, y: pd.Series = None, sample_size: int = 500) -> Dict:
        """
        Perform SHAP global and local explanation on input data
        Args:
            X: Feature data
            y: Labels (optional)
            sample_size: Number of samples for analysis
        Returns:
            Analysis result dictionary
        """
        # Check if SHAP is available
        if not SHAP_AVAILABLE:
            logger.error("SHAP package is not installed. Please install it using: pip install shap")
            return {
                'error': 'SHAP package not available',
                'message': 'Please install SHAP using: pip install shap',
                'feature_importance': [],
                'feature_contributions': []
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
            except Exception as e:
                logger.error(f"Failed to load model from ModelManager: {e}")
                self.model = None
        if self.model is None:
            raise RuntimeError("Failed to auto-load model, please ensure there are available pre-trained models.")

        # Align features with model expectations
        try:
            X_sample = self._align_features_with_model(X_sample, self.model)
            logger.info(f"Feature alignment completed. Final features: {list(X_sample.columns)}")
        except Exception as e:
            logger.warning(f"Feature alignment failed, using original data: {e}")

        # Select explainer
        try:
            explainer = shap.TreeExplainer(self.model)
            logger.info("Using TreeExplainer for SHAP analysis")
        except Exception as e:
            logger.warning(f"TreeExplainer failed: {e}, falling back to general Explainer")
            try:
                explainer = shap.Explainer(self.model)
                logger.info("Using general Explainer for SHAP analysis")
            except Exception as e2:
                raise RuntimeError(f"Failed to create SHAP explainer: {e2}")

        # Calculate SHAP values
        try:
            logger.info(f"Computing SHAP values for {len(X_sample)} samples with {len(X_sample.columns)} features")
            shap_values = explainer.shap_values(X_sample)
            logger.info("SHAP values computed successfully")
        except Exception as e:
            logger.error(f"SHAP value calculation failed: {e}")
            logger.error(f"Model type: {type(self.model)}")
            logger.error(f"Input features: {list(X_sample.columns)}")
            logger.error(f"Input shape: {X_sample.shape}")
            raise RuntimeError(f"SHAP analysis failed: Feature mismatch or model incompatibility - {str(e)}")

        # Handle binary and multi-class compatibility
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            logger.info("Using positive class SHAP values for binary classification")

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

    def _align_features_with_model(self, X: pd.DataFrame, model) -> pd.DataFrame:
        """Align feature data with model expectations"""
        try:
            # Get expected feature names from model
            expected_features = None
            if hasattr(model, 'feature_names_'):
                expected_features = list(model.feature_names_)
            elif hasattr(model, 'get_feature_names'):
                expected_features = list(model.get_feature_names())
            elif hasattr(model, 'feature_name_'):
                expected_features = list(model.feature_name_)
            elif hasattr(model, 'get_booster'):
                # For XGBoost models
                booster = model.get_booster()
                if hasattr(booster, 'feature_names'):
                    expected_features = list(booster.feature_names)

            if expected_features is None:
                logger.warning("Cannot get expected features from model, using original data")
                return X

            logger.info(f"Model expects {len(expected_features)} features: {expected_features}")
            logger.info(f"Input data has {len(X.columns)} features: {list(X.columns)}")

            # Create feature mapping
            feature_mapping = self._create_feature_mapping()

            # Align features
            X_aligned = pd.DataFrame(index=X.index)
            missing_features = []

            for expected_feature in expected_features:
                if expected_feature in X.columns:
                    X_aligned[expected_feature] = X[expected_feature]
                else:
                    # Try mapping
                    mapped_feature = self._find_mapped_feature(expected_feature, X.columns, feature_mapping)
                    if mapped_feature:
                        X_aligned[expected_feature] = X[mapped_feature]
                        logger.info(f"Mapped '{mapped_feature}' -> '{expected_feature}'")
                    else:
                        # Fill with default value
                        X_aligned[expected_feature] = 0
                        missing_features.append(expected_feature)
                        logger.warning(f"Feature '{expected_feature}' not found, using default value 0")

            if missing_features:
                logger.warning(f"Missing features filled with defaults: {missing_features}")

            return X_aligned

        except Exception as e:
            logger.error(f"Feature alignment failed: {e}")
            return X

    def _create_feature_mapping(self) -> Dict[str, List[str]]:
        """Create feature name mapping table"""
        return {
            # Original format -> Possible alternatives
            'Transaction Amount': ['transaction_amount', 'amount', 'trans_amount', 'txn_amount'],
            'Quantity': ['quantity', 'qty', 'item_count', 'num_items'],
            'Customer Age': ['customer_age', 'age', 'user_age', 'client_age'],
            'Account Age Days': ['account_age_days', 'account_age', 'days_since_registration', 'account_days'],
            'Transaction Hour': ['transaction_hour', 'hour', 'txn_hour', 'time_hour'],
            'Is Fraudulent': ['is_fraudulent', 'fraud', 'fraudulent', 'is_fraud'],
            'Device Used': ['device_used', 'device', 'device_type'],
            'Payment Method': ['payment_method', 'payment', 'payment_type'],
            'Product Category': ['product_category', 'category', 'product_type'],
            'Customer Location': ['customer_location', 'location', 'customer_city'],
            'IP Address': ['ip_address', 'ip', 'client_ip'],
            'Shipping Address': ['shipping_address', 'ship_address', 'delivery_address'],
            'Billing Address': ['billing_address', 'bill_address', 'payment_address'],
            # Reverse mapping for engineered features
            'transaction_amount': ['Transaction Amount', 'amount', 'trans_amount'],
            'quantity': ['Quantity', 'qty', 'item_count'],
            'customer_age': ['Customer Age', 'age', 'user_age'],
            'account_age_days': ['Account Age Days', 'account_age', 'days_since_registration'],
            'transaction_hour': ['Transaction Hour', 'hour', 'txn_hour'],
            'is_fraudulent': ['Is Fraudulent', 'fraud', 'fraudulent'],
            'device_used': ['Device Used', 'device', 'device_type'],
            'payment_method': ['Payment Method', 'payment', 'payment_type'],
            'product_category': ['Product Category', 'category', 'product_type']
        }

    def _find_mapped_feature(self, expected_feature: str, available_features: List[str],
                           mapping: Dict[str, List[str]]) -> str:
        """Find mapped feature name"""
        # Direct match first
        if expected_feature in available_features:
            return expected_feature

        # Try mapping
        if expected_feature in mapping:
            for candidate in mapping[expected_feature]:
                if candidate in available_features:
                    return candidate

        # Try reverse mapping (case-insensitive)
        expected_lower = expected_feature.lower()
        for available_feature in available_features:
            available_lower = available_feature.lower()
            if expected_lower == available_lower:
                return available_feature
            # Try partial matching
            if expected_lower.replace(' ', '_') == available_lower:
                return available_feature
            if expected_lower.replace('_', ' ') == available_lower:
                return available_feature

        return None