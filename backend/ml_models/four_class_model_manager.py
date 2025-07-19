"""
四分类模型管理器
专门用于四级风险分类的模型管理
支持CatBoost和XGBoost的四分类训练和预测
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入CatBoost
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost未安装，将跳过CatBoost相关功能")

# 尝试导入XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost未安装，将跳过XGBoost相关功能")

# 导入配置
try:
    from config.optimization_config import optimization_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    optimization_config = None


class FourClassModelManager:
    """四分类模型管理器"""
    
    def __init__(self, models_dir: str = None):
        """
        初始化四分类模型管理器
        
        Args:
            models_dir: 模型文件目录
        """
        if models_dir is None:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            self.models_dir = os.path.join(project_root, "models", "four_class")
        else:
            self.models_dir = models_dir
        
        # 确保模型目录存在
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.models = {}
        self.feature_info = {}
        
        # 四分类配置
        self.target_classes = ['low', 'medium', 'high', 'extreme']
        self.class_mapping = {'low': 0, 'medium': 1, 'high': 2, 'extreme': 3}
        
        # 加载配置
        self.config = self._load_config()
        
        # 模型权重
        self.model_weights = {
            'catboost': self.config.get('catboost_weight', 0.6),
            'xgboost': self.config.get('xgboost_weight', 0.4)
        }
        
        logger.info(f"四分类模型管理器初始化完成，模型目录: {self.models_dir}")
        self._load_existing_models()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if CONFIG_AVAILABLE and optimization_config:
            return optimization_config.get_risk_scoring_config().get('model_architecture', {})
        else:
            return {
                'catboost_weight': 0.6,
                'xgboost_weight': 0.4,
                'dynamic_weights': True,
                'class_weights': [0.6, 0.25, 0.12, 0.03]
            }
    
    def _load_existing_models(self):
        """加载已存在的模型"""
        try:
            model_files = {
                'catboost': 'catboost_four_class.cbm',
                'xgboost': 'xgboost_four_class.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = os.path.join(self.models_dir, filename)
                if os.path.exists(model_path):
                    try:
                        if model_name == 'catboost' and CATBOOST_AVAILABLE:
                            model = CatBoostClassifier()
                            model.load_model(model_path)
                            self.models[model_name] = model
                            logger.info(f"成功加载CatBoost四分类模型")
                        elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                            model = joblib.load(model_path)
                            self.models[model_name] = model
                            logger.info(f"成功加载XGBoost四分类模型")
                    except Exception as e:
                        logger.warning(f"加载{model_name}模型失败: {e}")
            
            if not self.models:
                logger.info("未找到预训练的四分类模型，需要先训练模型")
            else:
                logger.info(f"成功加载 {len(self.models)} 个四分类模型")
                
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
    
    def train_catboost_model(self, X: pd.DataFrame, y: np.ndarray, 
                           validation_data: Optional[Tuple] = None) -> bool:
        """
        训练CatBoost四分类模型
        
        Args:
            X: 特征数据
            y: 四分类标签 (0=low, 1=medium, 2=high, 3=extreme)
            validation_data: 验证数据 (X_val, y_val)
            
        Returns:
            训练是否成功
        """
        try:
            if not CATBOOST_AVAILABLE:
                logger.error("CatBoost未安装，无法训练模型")
                return False
            
            logger.info("开始训练CatBoost四分类模型...")
            
            # 配置CatBoost参数
            class_weights = self.config.get('class_weights', [0.6, 0.25, 0.12, 0.03])
            
            model = CatBoostClassifier(
                objective='MultiClass',
                classes_count=4,
                class_weights=class_weights,
                iterations=1000,
                learning_rate=0.1,
                depth=6,
                l2_leaf_reg=3,
                random_seed=42,
                eval_metric='MultiClass',
                early_stopping_rounds=100,
                verbose=100
            )
            
            # 训练模型
            if validation_data is not None:
                X_val, y_val = validation_data
                model.fit(
                    X, y,
                    eval_set=(X_val, y_val),
                    use_best_model=True
                )
            else:
                model.fit(X, y)
            
            # 保存模型
            model_path = os.path.join(self.models_dir, 'catboost_four_class.cbm')
            model.save_model(model_path)
            
            # 保存特征信息
            feature_info = {
                'feature_names': list(X.columns),
                'feature_count': len(X.columns),
                'training_samples': len(X),
                'classes': self.target_classes,
                'class_distribution': np.bincount(y).tolist()
            }
            
            feature_path = os.path.join(self.models_dir, 'catboost_features.pkl')
            with open(feature_path, 'wb') as f:
                pickle.dump(feature_info, f)
            
            self.models['catboost'] = model
            self.feature_info['catboost'] = feature_info
            
            logger.info("CatBoost四分类模型训练完成")
            return True
            
        except Exception as e:
            logger.error(f"CatBoost模型训练失败: {e}")
            return False
    
    def train_xgboost_model(self, X: pd.DataFrame, y: np.ndarray,
                          validation_data: Optional[Tuple] = None) -> bool:
        """
        训练XGBoost四分类模型
        
        Args:
            X: 特征数据
            y: 四分类标签
            validation_data: 验证数据
            
        Returns:
            训练是否成功
        """
        try:
            if not XGBOOST_AVAILABLE:
                logger.error("XGBoost未安装，无法训练模型")
                return False
            
            logger.info("开始训练XGBoost四分类模型...")
            
            # 计算类别权重
            class_counts = np.bincount(y)
            total_samples = len(y)
            class_weights = total_samples / (4 * class_counts)
            sample_weights = np.array([class_weights[label] for label in y])
            
            model = XGBClassifier(
                objective='multi:softprob',
                num_class=4,
                max_depth=6,
                learning_rate=0.1,
                n_estimators=800,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                early_stopping_rounds=100
            )
            
            # 训练模型
            if validation_data is not None:
                X_val, y_val = validation_data
                val_weights = np.array([class_weights[label] for label in y_val])
                model.fit(
                    X, y,
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val)],
                    sample_weight_eval_set=[val_weights],
                    verbose=100
                )
            else:
                model.fit(X, y, sample_weight=sample_weights)
            
            # 保存模型
            model_path = os.path.join(self.models_dir, 'xgboost_four_class.pkl')
            joblib.dump(model, model_path)
            
            # 保存特征信息
            feature_info = {
                'feature_names': list(X.columns),
                'feature_count': len(X.columns),
                'training_samples': len(X),
                'classes': self.target_classes,
                'class_distribution': np.bincount(y).tolist(),
                'feature_importance': model.feature_importances_.tolist()
            }
            
            feature_path = os.path.join(self.models_dir, 'xgboost_features.pkl')
            with open(feature_path, 'wb') as f:
                pickle.dump(feature_info, f)
            
            self.models['xgboost'] = model
            self.feature_info['xgboost'] = feature_info
            
            logger.info("XGBoost四分类模型训练完成")
            return True
            
        except Exception as e:
            logger.error(f"XGBoost模型训练失败: {e}")
            return False

    def predict_four_class(self, X: pd.DataFrame,
                          use_ensemble: bool = True) -> Dict[str, Any]:
        """
        四分类预测

        Args:
            X: 特征数据
            use_ensemble: 是否使用集成预测

        Returns:
            预测结果
        """
        try:
            if not self.models:
                logger.error("没有可用的四分类模型")
                return self._empty_prediction_result()

            predictions = {}
            probabilities = {}

            # 单模型预测
            for model_name, model in self.models.items():
                try:
                    # 预测概率
                    probs = model.predict_proba(X)
                    preds = np.argmax(probs, axis=1)

                    predictions[model_name] = preds
                    probabilities[model_name] = probs

                    logger.info(f"{model_name}四分类预测完成")

                except Exception as e:
                    logger.warning(f"{model_name}预测失败: {e}")

            if not predictions:
                logger.error("所有模型预测都失败了")
                return self._empty_prediction_result()

            # 集成预测
            if use_ensemble and len(predictions) > 1:
                ensemble_probs, ensemble_preds = self._ensemble_predict(probabilities)
                predictions['ensemble'] = ensemble_preds
                probabilities['ensemble'] = ensemble_probs

            # 生成结果
            result = self._generate_prediction_result(predictions, probabilities, X)
            return result

        except Exception as e:
            logger.error(f"四分类预测失败: {e}")
            return self._empty_prediction_result()

    def _ensemble_predict(self, model_probabilities: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """集成预测"""
        try:
            if 'catboost' in model_probabilities and 'xgboost' in model_probabilities:
                catboost_probs = model_probabilities['catboost']
                xgboost_probs = model_probabilities['xgboost']

                # 动态权重调整
                if self.config.get('dynamic_weights', True):
                    weights = self._calculate_dynamic_weights(catboost_probs, xgboost_probs)
                else:
                    weights = (self.model_weights['catboost'], self.model_weights['xgboost'])

                # 加权平均
                ensemble_probs = (catboost_probs * weights[0] +
                                xgboost_probs * weights[1])

                # 预测类别
                ensemble_preds = np.argmax(ensemble_probs, axis=1)

                logger.info(f"集成预测完成，权重: CatBoost={weights[0]:.2f}, XGBoost={weights[1]:.2f}")
                return ensemble_probs, ensemble_preds

            else:
                # 只有一个模型，直接返回
                model_name = list(model_probabilities.keys())[0]
                probs = model_probabilities[model_name]
                preds = np.argmax(probs, axis=1)
                return probs, preds

        except Exception as e:
            logger.warning(f"集成预测失败: {e}")
            # 回退到第一个可用模型
            model_name = list(model_probabilities.keys())[0]
            probs = model_probabilities[model_name]
            preds = np.argmax(probs, axis=1)
            return probs, preds

    def _calculate_dynamic_weights(self, catboost_probs: np.ndarray,
                                 xgboost_probs: np.ndarray) -> Tuple[float, float]:
        """计算动态权重"""
        try:
            # 计算每个模型的平均置信度
            catboost_confidence = np.mean(np.max(catboost_probs, axis=1))
            xgboost_confidence = np.mean(np.max(xgboost_probs, axis=1))

            # 基础权重
            base_catboost = self.model_weights['catboost']
            base_xgboost = self.model_weights['xgboost']

            # 置信度差异调整
            confidence_diff = catboost_confidence - xgboost_confidence

            if confidence_diff > 0.1:  # CatBoost更有信心
                catboost_weight = min(base_catboost + 0.1, 0.8)
                xgboost_weight = 1.0 - catboost_weight
            elif confidence_diff < -0.1:  # XGBoost更有信心
                xgboost_weight = min(base_xgboost + 0.1, 0.6)
                catboost_weight = 1.0 - xgboost_weight
            else:
                catboost_weight = base_catboost
                xgboost_weight = base_xgboost

            return catboost_weight, xgboost_weight

        except Exception as e:
            logger.warning(f"动态权重计算失败: {e}")
            return self.model_weights['catboost'], self.model_weights['xgboost']

    def _generate_prediction_result(self, predictions: Dict[str, np.ndarray],
                                  probabilities: Dict[str, np.ndarray],
                                  X: pd.DataFrame) -> Dict[str, Any]:
        """生成预测结果"""
        try:
            result = {
                'predictions': {},
                'probabilities': {},
                'risk_levels': {},
                'distribution': {},
                'confidence_stats': {},
                'total_samples': len(X)
            }

            for model_name in predictions:
                preds = predictions[model_name]
                probs = probabilities[model_name]

                # 转换为风险等级标签
                risk_levels = [self.target_classes[pred] for pred in preds]

                # 计算分布
                distribution = {}
                for i, class_name in enumerate(self.target_classes):
                    count = np.sum(preds == i)
                    distribution[class_name] = {
                        'count': int(count),
                        'percentage': float(count / len(preds) * 100)
                    }

                # 计算置信度统计
                max_probs = np.max(probs, axis=1)
                confidence_stats = {
                    'mean_confidence': float(np.mean(max_probs)),
                    'min_confidence': float(np.min(max_probs)),
                    'max_confidence': float(np.max(max_probs)),
                    'high_confidence_ratio': float(np.sum(max_probs >= 0.8) / len(max_probs))
                }

                result['predictions'][model_name] = preds.tolist()
                result['probabilities'][model_name] = probs.tolist()
                result['risk_levels'][model_name] = risk_levels
                result['distribution'][model_name] = distribution
                result['confidence_stats'][model_name] = confidence_stats

            return result

        except Exception as e:
            logger.error(f"预测结果生成失败: {e}")
            return self._empty_prediction_result()

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return list(self.models.keys())

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型信息"""
        if model_name not in self.models:
            return {}

        info = {
            'name': model_name,
            'type': type(self.models[model_name]).__name__,
            'available': True,
            'classes': self.target_classes,
            'class_count': 4
        }

        if model_name in self.feature_info:
            info.update(self.feature_info[model_name])

        return info

    def _empty_prediction_result(self) -> Dict[str, Any]:
        """返回空预测结果"""
        return {
            'predictions': {},
            'probabilities': {},
            'risk_levels': {},
            'distribution': {},
            'confidence_stats': {},
            'total_samples': 0,
            'success': False
        }
