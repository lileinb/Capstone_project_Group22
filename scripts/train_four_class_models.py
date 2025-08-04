#!/usr/bin/env python3
"""
四分类模型训练脚本
使用半监督生成的四分类标签训练CatBoost和XGBoost模型
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data():
    """Load and prepare training data"""
    try:
        # Load raw data
        data_path = os.path.join(project_root, "data", "processed", "engineered_features.csv")
        if not os.path.exists(data_path):
            logger.error(f"Data file does not exist: {data_path}")
            return None, None

        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully, shape: {data.shape}")

        # Generate four-class labels
        from backend.pseudo_labeling.semi_supervised_generator import SemiSupervisedLabelGenerator
        from backend.clustering.cluster_analyzer import ClusterAnalyzer

        # First perform clustering analysis
        cluster_analyzer = ClusterAnalyzer()
        cluster_results = cluster_analyzer.analyze_clusters(data, algorithm='kmeans')

        # Generate four-class labels
        label_generator = SemiSupervisedLabelGenerator()
        label_results = label_generator.generate_four_class_labels(
            data,
            cluster_results=cluster_results
        )
        
        if not label_results['success']:
            logger.error("Four-class label generation failed")
            return None, None
        
        # 准备特征和标签
        # 移除非数值特征和标签列
        feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_fraudulent' in feature_columns:
            feature_columns.remove('is_fraudulent')
        
        X = data[feature_columns]
        y = np.array(label_results['labels'])
        
        # 处理缺失值和无穷值
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        logger.info(f"特征数量: {len(feature_columns)}")
        logger.info(f"标签分布: {np.bincount(y)}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"数据准备失败: {e}")
        return None, None


def train_models():
    """训练四分类模型"""
    try:
        # 加载数据
        X, y = load_and_prepare_data()
        if X is None or y is None:
            logger.error("数据加载失败")
            return False
        
        # 分割训练和验证数据
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"训练集大小: {len(X_train)}")
        logger.info(f"验证集大小: {len(X_val)}")
        
        # 创建模型管理器
        from backend.ml_models.four_class_model_manager import FourClassModelManager
        
        model_manager = FourClassModelManager()
        
        # 训练CatBoost模型
        logger.info("开始训练CatBoost四分类模型...")
        catboost_success = model_manager.train_catboost_model(
            X_train, y_train, 
            validation_data=(X_val, y_val)
        )
        
        if catboost_success:
            logger.info("CatBoost模型训练成功")
        else:
            logger.warning("CatBoost模型训练失败")
        
        # 训练XGBoost模型
        logger.info("开始训练XGBoost四分类模型...")
        xgboost_success = model_manager.train_xgboost_model(
            X_train, y_train,
            validation_data=(X_val, y_val)
        )
        
        if xgboost_success:
            logger.info("XGBoost模型训练成功")
        else:
            logger.warning("XGBoost模型训练失败")
        
        # 验证模型性能
        if catboost_success or xgboost_success:
            logger.info("开始验证模型性能...")
            evaluate_models(model_manager, X_val, y_val)
        
        return catboost_success or xgboost_success
        
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        return False


def evaluate_models(model_manager, X_val, y_val):
    """评估模型性能"""
    try:
        # 进行预测
        prediction_results = model_manager.predict_four_class(X_val, use_ensemble=True)
        
        if not prediction_results.get('predictions'):
            logger.warning("没有预测结果")
            return
        
        target_classes = ['low', 'medium', 'high', 'extreme']
        
        # 评估每个模型
        for model_name, predictions in prediction_results['predictions'].items():
            logger.info(f"\n=== {model_name.upper()} 模型性能 ===")
            
            # 分类报告
            report = classification_report(
                y_val, predictions, 
                target_names=target_classes,
                zero_division=0
            )
            logger.info(f"分类报告:\n{report}")
            
            # 混淆矩阵
            cm = confusion_matrix(y_val, predictions)
            logger.info(f"混淆矩阵:\n{cm}")
            
            # 分布统计
            distribution = prediction_results['distribution'][model_name]
            logger.info("预测分布:")
            for class_name, stats in distribution.items():
                logger.info(f"  {class_name}: {stats['count']} ({stats['percentage']:.1f}%)")
            
            # Confidence statistics
            confidence_stats = prediction_results['confidence_stats'][model_name]
            logger.info(f"Confidence statistics:")
            logger.info(f"  Average confidence: {confidence_stats['mean_confidence']:.3f}")
            logger.info(f"  High confidence ratio: {confidence_stats['high_confidence_ratio']:.3f}")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")


def main():
    """Main function"""
    logger.info("🚀 Starting four-class model training")
    logger.info("=" * 50)

    success = train_models()

    logger.info("=" * 50)
    if success:
        logger.info("🎉 Four-class model training completed!")
        logger.info("Model files saved in: models/four_class/")
        logger.info("You can use these models for four-class prediction on the model prediction page")
    else:
        logger.error("❌ Four-class model training failed")
        logger.info("Please check if data and dependencies are correctly installed")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
