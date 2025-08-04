"""
模型管理器
负责加载、管理和使用预训练的机器学习模型
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 尝试导入CatBoost，如果失败则使用替代方案
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    logger.warning("CatBoost未安装，将跳过CatBoost模型")
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Model manager class"""

    def __init__(self, models_dir: str = None):
        """
        Initialize model manager

        Args:
            models_dir: Model file directory, auto-detect if None
        """
        if models_dir is None:
            # Auto-detect models/pretrained directory under project root
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            self.models_dir = os.path.join(project_root, "models", "pretrained")
        else:
            self.models_dir = models_dir

        self.models = {}
        self.feature_info = {}

        # Check if model directory exists
        if not os.path.exists(self.models_dir):
            logger.warning(f"Model directory does not exist: {self.models_dir}")
            logger.info("Please ensure models/pretrained directory exists and contains pre-trained model files")
        else:
            logger.info(f"Using model directory: {self.models_dir}")

        self._load_all_models()
    
    def _load_all_models(self):
        """加载所有可用的预训练模型"""
        try:
            # 模型文件映射
            model_files = {
                'catboost': 'catboost_model.cbm',
                'xgboost': 'xgboost_model.pkl',
                'randomforest': 'randomforest_model.pkl',
                'ensemble': 'ensemble_model.pkl'
            }
            
            # 特征信息文件映射
            feature_files = {
                'catboost': 'catboost_feature_info.pkl',
                'xgboost': 'xgboost_feature_info.pkl',
                'randomforest': 'randomforest_feature_info.pkl',
                'ensemble': 'ensemble_feature_info.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = os.path.join(self.models_dir, filename)
                if os.path.exists(model_path):
                    try:
                        if model_name == 'catboost':
                            if not CATBOOST_AVAILABLE:
                                logger.warning("Skipping CatBoost model: CatBoost not installed")
                                continue
                            # CatBoost model uses special loading method
                            model = CatBoostClassifier()
                            model.load_model(model_path)
                        else:
                            # Other models use joblib loading
                            model = joblib.load(model_path)

                        self.models[model_name] = model
                        logger.info(f"Successfully loaded model: {model_name}")

                        # Load corresponding feature information
                        feature_file = feature_files.get(model_name)
                        if feature_file:
                            feature_path = os.path.join(self.models_dir, feature_file)
                            if os.path.exists(feature_path):
                                with open(feature_path, 'rb') as f:
                                    self.feature_info[model_name] = pickle.load(f)
                                logger.info(f"Successfully loaded feature information: {model_name}")

                    except Exception as e:
                        logger.warning(f"Failed to load model {model_name}: {e}")
                else:
                    logger.warning(f"Model file does not exist: {model_path}")

            if not self.models:
                logger.warning("No models loaded")
            else:
                logger.info(f"Successfully loaded {len(self.models)} models")

        except Exception as e:
            logger.error(f"Error occurred while loading models: {e}")
    
    def load_model(self, model_name: str):
        """
        Load specified model

        Args:
            model_name: Model name

        Returns:
            Model object or None
        """
        return self.models.get(model_name.lower())

    def get_available_models(self) -> list:
        """Get available model list"""
        return list(self.models.keys())
    
    def predict_with_model(self, model, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用指定模型进行预测

        Args:
            model: 模型对象
            X: 特征数据
            y: 标签数据（可选）

        Returns:
            预测结果和概率
        """
        try:
            # 特征对齐处理
            X_aligned = self._align_features_for_prediction(model, X)

            # 进行预测
            predictions = model.predict(X_aligned)

            # 获取预测概率
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_aligned)
                if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
                    probabilities = probabilities[:, 1]  # 取正类概率
                else:
                    probabilities = probabilities.flatten()
            else:
                # 如果模型不支持概率预测，使用预测结果作为概率
                probabilities = predictions.astype(float)

            return predictions, probabilities

        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            # 返回默认值
            return np.zeros(len(X)), np.zeros(len(X))

    def predict_attack_types(self, model, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        使用模型进行攻击类型多分类预测

        Args:
            model: 模型对象
            X: 特征数据
            y: 标签数据（可选）

        Returns:
            攻击类型预测结果、概率和详细信息
        """
        try:
            # 首先进行二分类预测（欺诈/正常）
            binary_predictions, binary_probabilities = self.predict_with_model(model, X, y)

            # 定义攻击类型映射
            attack_types = {
                0: 'normal',           # 正常交易
                1: 'account_takeover', # 账户接管
                2: 'identity_theft',   # 身份盗用
                3: 'bulk_fraud',       # 批量欺诈
                4: 'testing_attack'    # 测试性攻击
            }

            # 对欺诈交易进行进一步的攻击类型分类
            attack_predictions = np.zeros(len(X), dtype=int)
            attack_probabilities = np.zeros((len(X), len(attack_types)))

            # 正常交易标记为类型0
            normal_mask = binary_predictions == 0
            attack_predictions[normal_mask] = 0
            attack_probabilities[normal_mask, 0] = 1.0

            # 欺诈交易进行攻击类型分类
            fraud_mask = binary_predictions == 1
            if np.any(fraud_mask):
                fraud_data = X[fraud_mask]
                fraud_attack_types = self._classify_fraud_attack_types(fraud_data)

                # 安全地分配攻击类型
                fraud_indices = np.where(fraud_mask)[0]
                for i, attack_type in enumerate(fraud_attack_types):
                    if i < len(fraud_indices):
                        fraud_idx = fraud_indices[i]
                        attack_predictions[fraud_idx] = attack_type

                        # 为攻击类型分配概率，确保索引安全
                        if fraud_idx < len(binary_probabilities):
                            attack_probabilities[fraud_idx, attack_type] = binary_probabilities[fraud_idx]

            # 生成详细信息
            attack_details = {
                'attack_type_mapping': attack_types,
                'binary_fraud_rate': np.mean(binary_predictions),
                'attack_type_distribution': {
                    attack_types[i]: np.sum(attack_predictions == i)
                    for i in range(len(attack_types))
                },
                'total_predictions': len(X),
                'fraud_predictions': np.sum(fraud_mask),
                'normal_predictions': np.sum(normal_mask)
            }

            return attack_predictions, attack_probabilities, attack_details

        except Exception as e:
            logger.error(f"攻击类型预测失败: {e}")
            # 返回默认值
            return (np.zeros(len(X)),
                   np.zeros((len(X), 5)),
                   {'error': str(e)})

    def _classify_fraud_attack_types(self, fraud_data: pd.DataFrame) -> np.ndarray:
        """
        对欺诈交易进行攻击类型分类

        Args:
            fraud_data: 欺诈交易数据

        Returns:
            攻击类型预测数组
        """
        try:
            attack_types = np.ones(len(fraud_data), dtype=int)  # 默认为账户接管(1)

            for i, (idx, row) in enumerate(fraud_data.iterrows()):
                # 基于特征规则进行攻击类型分类
                attack_type = self._rule_based_attack_classification(row)
                attack_types[i] = attack_type

            return attack_types

        except Exception as e:
            logger.error(f"欺诈攻击类型分类失败: {e}")
            return np.ones(len(fraud_data), dtype=int)

    def _rule_based_attack_classification(self, transaction: pd.Series) -> int:
        """
        基于规则的攻击类型分类

        Args:
            transaction: 单笔交易数据

        Returns:
            攻击类型ID (1-4)
        """
        try:
            # 获取关键特征
            account_age = transaction.get('account_age_days', 365)
            transaction_amount = transaction.get('transaction_amount', 0)
            transaction_hour = transaction.get('transaction_hour', 12)
            customer_age = transaction.get('customer_age', 35)
            quantity = transaction.get('quantity', 1)

            # 规则1: 账户接管攻击 (account_takeover = 1)
            if (account_age <= 30 and
                transaction_amount >= 500 and
                transaction_hour in [0, 1, 2, 3, 4, 5, 22, 23]):
                return 1

            # 规则2: 身份盗用攻击 (identity_theft = 2)
            if (customer_age <= 18 or customer_age >= 70) and transaction_amount >= 200:
                return 2

            # 规则3: 批量欺诈攻击 (bulk_fraud = 3)
            if (quantity in [1, 2] and
                50 <= transaction_amount <= 300 and
                6 <= transaction_hour <= 18):
                return 3

            # 规则4: 测试性攻击 (testing_attack = 4)
            if (transaction_amount <= 50 and
                account_age <= 7):
                return 4

            # 默认: 账户接管攻击
            return 1

        except Exception as e:
            logger.error(f"规则分类失败: {e}")
            return 1

    def _align_features_for_prediction(self, model, X: pd.DataFrame) -> pd.DataFrame:
        """对齐预测特征与训练特征"""
        try:
            # 获取模型期望的特征
            if hasattr(model, 'feature_names_'):
                expected_features = model.feature_names_
            elif hasattr(model, 'get_feature_names'):
                expected_features = model.get_feature_names()
            elif hasattr(model, 'feature_name_'):
                expected_features = model.feature_name_
            else:
                # 如果无法获取特征名称，尝试使用基础特征
                expected_features = self._get_basic_features()

            logger.info(f"模型期望特征: {expected_features}")
            logger.info(f"输入数据特征: {list(X.columns)}")

            # 创建对齐的数据框
            X_aligned = pd.DataFrame()

            for feature in expected_features:
                if feature in X.columns:
                    X_aligned[feature] = X[feature]
                else:
                    # 尝试特征名称映射
                    mapped_feature = self._map_feature_name(feature, X.columns)
                    if mapped_feature:
                        X_aligned[feature] = X[mapped_feature]
                    else:
                        # 如果找不到对应特征，填充默认值
                        logger.warning(f"特征 {feature} 不存在，使用默认值")
                        X_aligned[feature] = 0

            return X_aligned

        except Exception as e:
            logger.error(f"特征对齐失败: {e}")
            return X

    def _get_basic_features(self) -> list:
        """获取基础特征列表"""
        return [
            'Transaction Amount',
            'Quantity',
            'Customer Age',
            'Account Age Days',
            'Transaction Hour'
        ]

    def _map_feature_name(self, expected_feature: str, available_features: list) -> str:
        """映射特征名称"""
        # 扩展的特征名称映射表
        feature_mapping = {
            'Transaction Amount': ['transaction_amount', 'amount', 'trans_amount', 'txn_amount'],
            'Quantity': ['quantity', 'qty', 'item_count', 'num_items'],
            'Customer Age': ['customer_age', 'age', 'user_age', 'client_age'],
            'Account Age Days': ['account_age_days', 'account_age', 'days_since_registration', 'account_days'],
            'Transaction Hour': ['transaction_hour', 'hour', 'txn_hour', 'time_hour'],
            'Is Fraudulent': ['is_fraudulent', 'fraud', 'fraudulent', 'is_fraud']
        }

        # 反向映射
        reverse_mapping = {}
        for key, values in feature_mapping.items():
            for value in values:
                reverse_mapping[value] = key

        # 尝试直接映射
        if expected_feature in feature_mapping:
            for candidate in feature_mapping[expected_feature]:
                if candidate in available_features:
                    return candidate

        # 尝试反向映射
        if expected_feature in reverse_mapping:
            target_key = reverse_mapping[expected_feature]
            if target_key in feature_mapping:
                for candidate in feature_mapping[target_key]:
                    if candidate in available_features:
                        return candidate

        # 尝试模糊匹配
        expected_lower = expected_feature.lower().replace(' ', '_').replace('-', '_')
        for feature in available_features:
            feature_lower = feature.lower().replace(' ', '_').replace('-', '_')
            if feature_lower == expected_lower:
                return feature

            # 部分匹配
            if expected_lower in feature_lower or feature_lower in expected_lower:
                return feature

        return None
    
    def evaluate_model(self, predictions: np.ndarray, probabilities: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            predictions: 预测结果
            probabilities: 预测概率
            y_true: 真实标签
            
        Returns:
            性能指标字典
        """
        try:
            metrics = {}
            
            # 基础分类指标
            metrics['accuracy'] = accuracy_score(y_true, predictions)
            metrics['precision'] = precision_score(y_true, predictions, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, predictions, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, predictions, average='weighted', zero_division=0)
            
            # AUC指标（如果有概率预测）
            try:
                if len(np.unique(y_true)) > 1:  # 确保有正负样本
                    metrics['auc'] = roc_auc_score(y_true, probabilities)
                else:
                    metrics['auc'] = 0.0
            except Exception:
                metrics['auc'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc': 0.0
            }
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息字典
        """
        model = self.models.get(model_name.lower())
        if model is None:
            return {}
        
        info = {
            'name': model_name,
            'type': type(model).__name__,
            'available': True
        }
        
        # 添加特征信息
        if model_name.lower() in self.feature_info:
            info['feature_info'] = self.feature_info[model_name.lower()]
        
        return info
