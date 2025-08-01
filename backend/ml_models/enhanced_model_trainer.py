"""
增强模型训练器
整合所有前置步骤的结果进行模型训练
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    """增强模型训练器"""
    
    def __init__(self):
        """初始化训练器"""
        self.models = {}
        self.training_history = []
        self.feature_importance = {}
        
    def train_with_integrated_data(self, 
                                 integrated_data: pd.DataFrame,
                                 feature_mapping: Dict[str, str],
                                 target_column: str = 'pseudo_label',
                                 test_size: float = 0.2) -> Dict[str, Any]:
        """
        使用整合数据训练模型
        
        Args:
            integrated_data: 整合后的数据
            feature_mapping: 特征来源映射
            target_column: 目标列名
            test_size: 测试集比例
            
        Returns:
            训练结果
        """
        try:
            logger.info("🚀 开始增强模型训练")
            start_time = datetime.now()
            
            # 1. 数据预处理
            X, y = self._prepare_training_data(integrated_data, target_column)
            if X is None or y is None:
                return self._empty_result("数据预处理失败")
            
            # 2. 特征选择
            selected_features = self._select_training_features(X, feature_mapping)
            X_selected = X[selected_features]
            
            # 3. 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # 4. 模型训练
            model_results = self._train_multiple_models(X_train, y_train, X_test, y_test)
            
            # 5. 特征重要性分析
            feature_importance = self._analyze_feature_importance(
                model_results, selected_features, feature_mapping
            )
            
            # 6. 模型评估
            evaluation_results = self._evaluate_models(model_results, X_test, y_test)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 7. 保存训练历史
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'data_size': len(integrated_data),
                'features_used': len(selected_features),
                'models_trained': list(model_results.keys()),
                'best_model': evaluation_results.get('best_model', 'unknown'),
                'processing_time': processing_time
            }
            self.training_history.append(training_record)
            
            logger.info(f"✅ 模型训练完成，耗时: {processing_time:.2f}秒")
            
            return {
                'success': True,
                'models': model_results,
                'feature_importance': feature_importance,
                'evaluation': evaluation_results,
                'training_data_info': {
                    'total_samples': len(integrated_data),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features_used': len(selected_features),
                    'selected_features': selected_features
                },
                'processing_time': processing_time,
                'training_record': training_record
            }
            
        except Exception as e:
            logger.error(f"❌ 模型训练失败: {e}")
            return self._empty_result(f"训练失败: {str(e)}")
    
    def _prepare_training_data(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据"""
        try:
            # 检查目标列
            if target_column not in data.columns:
                logger.error(f"目标列 {target_column} 不存在")
                return None, None
            
            # 分离特征和目标
            y = data[target_column]
            X = data.drop(columns=[target_column])
            
            # 移除非数值列（除了已编码的分类特征）
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            categorical_encoded = [col for col in X.columns if col.endswith('_encoded')]
            
            feature_columns = list(set(numeric_columns) | set(categorical_encoded))
            X = X[feature_columns]
            
            # 处理缺失值
            X = X.fillna(X.mean())
            
            logger.info(f"训练数据准备完成: {len(X)} 样本, {len(X.columns)} 特征")
            return X, y
            
        except Exception as e:
            logger.error(f"训练数据准备失败: {e}")
            return None, None
    
    def _select_training_features(self, X: pd.DataFrame, feature_mapping: Dict[str, str]) -> List[str]:
        """选择训练特征"""
        try:
            # 按重要性分组特征
            high_importance = []
            medium_importance = []
            low_importance = []
            
            # 高重要性特征（来自风险评分和伪标签）
            high_priority_sources = ['risk_scoring', 'enhanced', 'pseudo_labeling']
            high_priority_features = [
                'risk_score', 'composite_risk_score', 'pseudo_confidence',
                'cluster_fraud_rate', 'risk_confidence_alignment', 'cluster_risk_interaction'
            ]
            
            # 中等重要性特征（来自聚类和特征工程）
            medium_priority_sources = ['clustering', 'engineered_features']
            medium_priority_features = [
                'is_consistent_prediction', 'label_risk_consistency', 
                'is_amount_outlier', 'cluster_size'
            ]
            
            for feature in X.columns:
                source = feature_mapping.get(feature, 'unknown')
                
                if feature in high_priority_features or source in high_priority_sources:
                    high_importance.append(feature)
                elif feature in medium_priority_features or source in medium_priority_sources:
                    medium_importance.append(feature)
                else:
                    low_importance.append(feature)
            
            # 选择特征（优先高重要性）
            selected_features = high_importance + medium_importance
            
            # 如果特征太少，添加一些低重要性特征
            if len(selected_features) < 10:
                selected_features.extend(low_importance[:10-len(selected_features)])
            
            # 限制特征数量（避免过拟合）
            if len(selected_features) > 20:
                selected_features = selected_features[:20]
            
            logger.info(f"特征选择完成: {len(selected_features)} 个特征")
            logger.info(f"高重要性: {len(high_importance)}, 中等: {len(medium_importance)}, 低重要性: {len(low_importance)}")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"特征选择失败: {e}")
            return list(X.columns)[:15]  # 返回前15个特征作为备选
    
    def _train_multiple_models(self, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """训练多个模型"""
        models_config = {
            'random_forest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                'name': 'Random Forest'
            }
        }
        
        trained_models = {}
        
        for model_key, config in models_config.items():
            try:
                logger.info(f"训练 {config['name']} 模型...")
                
                model = config['model']
                model.fit(X_train, y_train)
                
                # 预测
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # 预测概率
                if hasattr(model, 'predict_proba'):
                    train_proba = model.predict_proba(X_train)[:, 1]
                    test_proba = model.predict_proba(X_test)[:, 1]
                else:
                    train_proba = train_pred.astype(float)
                    test_proba = test_pred.astype(float)
                
                trained_models[model_key] = {
                    'model': model,
                    'name': config['name'],
                    'train_predictions': train_pred,
                    'test_predictions': test_pred,
                    'train_probabilities': train_proba,
                    'test_probabilities': test_proba
                }
                
                logger.info(f"✅ {config['name']} 训练完成")
                
            except Exception as e:
                logger.error(f"❌ {config['name']} 训练失败: {e}")
        
        return trained_models
    
    def _analyze_feature_importance(self, model_results: Dict, features: List[str], 
                                  feature_mapping: Dict[str, str]) -> Dict[str, Any]:
        """分析特征重要性"""
        try:
            importance_analysis = {}
            
            for model_key, model_info in model_results.items():
                model = model_info['model']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # 创建特征重要性字典
                    feature_importance = {}
                    for i, feature in enumerate(features):
                        feature_importance[feature] = {
                            'importance': float(importances[i]),
                            'source': feature_mapping.get(feature, 'unknown')
                        }
                    
                    # 按重要性排序
                    sorted_features = sorted(
                        feature_importance.items(), 
                        key=lambda x: x[1]['importance'], 
                        reverse=True
                    )
                    
                    importance_analysis[model_key] = {
                        'feature_importance': feature_importance,
                        'top_features': sorted_features[:10],
                        'importance_by_source': self._group_importance_by_source(
                            feature_importance, feature_mapping
                        )
                    }
            
            return importance_analysis
            
        except Exception as e:
            logger.error(f"特征重要性分析失败: {e}")
            return {}
    
    def _group_importance_by_source(self, feature_importance: Dict, 
                                  feature_mapping: Dict[str, str]) -> Dict[str, float]:
        """按来源分组特征重要性"""
        source_importance = {}
        
        for feature, info in feature_importance.items():
            source = feature_mapping.get(feature, 'unknown')
            if source not in source_importance:
                source_importance[source] = 0
            source_importance[source] += info['importance']
        
        return source_importance
    
    def _evaluate_models(self, model_results: Dict, X_test, y_test) -> Dict[str, Any]:
        """评估模型性能"""
        try:
            evaluation = {}
            best_score = 0
            best_model = None
            
            for model_key, model_info in model_results.items():
                test_pred = model_info['test_predictions']
                
                # 计算准确率
                accuracy = np.mean(test_pred == y_test)
                
                # 分类报告
                report = classification_report(y_test, test_pred, output_dict=True)
                
                # 混淆矩阵
                cm = confusion_matrix(y_test, test_pred)
                
                evaluation[model_key] = {
                    'accuracy': float(accuracy),
                    'classification_report': report,
                    'confusion_matrix': cm.tolist(),
                    'model_name': model_info['name']
                }
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model_key
            
            evaluation['best_model'] = best_model
            evaluation['best_accuracy'] = best_score
            
            return evaluation
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            return {}
    
    def _empty_result(self, error_message: str) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'success': False,
            'error': error_message,
            'models': {},
            'feature_importance': {},
            'evaluation': {},
            'training_data_info': {},
            'processing_time': 0
        }
