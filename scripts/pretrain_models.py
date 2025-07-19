"""
模型预训练脚本
训练CatBoost、XGBoost、Random Forest和集成模型
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import DATASET_CONFIG, MODEL_CONFIG, PRETRAINED_MODELS_DIR
from backend.feature_engineer.risk_features import RiskFeatureEngineer
from backend.data_processor.data_cleaner import DataCleaner

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    """加载和预处理数据"""
    logger.info("开始加载和预处理数据...")
    
    # 加载数据集1用于训练
    dataset1_path = DATASET_CONFIG["dataset1"]["path"]
    logger.info(f"加载数据集1: {dataset1_path}")
    
    try:
        data = pd.read_csv(dataset1_path)
        logger.info(f"数据集1加载成功: {data.shape}")
    except Exception as e:
        logger.error(f"加载数据集1失败: {e}")
        return None
    
    # 数据清理
    cleaner = DataCleaner()
    cleaned_data = cleaner.comprehensive_cleaning(data)
    logger.info(f"数据清理完成: {cleaned_data.shape}")
    
    # 特征工程
    feature_engineer = RiskFeatureEngineer()
    engineered_data = feature_engineer.engineer_all_features(cleaned_data)
    logger.info(f"特征工程完成: {engineered_data.shape}")
    
    return engineered_data, feature_engineer

def prepare_training_data(data):
    """准备训练数据"""
    logger.info("准备训练数据...")
    
    # 选择数值型特征作为训练特征
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # 排除一些不需要的特征
    exclude_features = ['transaction_id', 'user_id', 'composite_risk_score', 'normalized_risk_score']
    feature_columns = [col for col in numeric_columns if col not in exclude_features]
    
    # 如果有目标列，使用它；否则创建一个模拟的目标列
    if 'fraud' in data.columns:
        target_column = 'fraud'
    elif 'is_fraud' in data.columns:
        target_column = 'is_fraud'
    else:
        # 创建模拟目标列（基于风险评分）
        if 'composite_risk_score' in data.columns:
            data['target'] = np.where(data['composite_risk_score'] > 20, 1, 0)
        else:
            # 随机生成目标列用于演示
            np.random.seed(42)
            data['target'] = np.random.choice([0, 1], size=len(data), p=[0.95, 0.05])
        target_column = 'target'
    
    X = data[feature_columns].fillna(0)
    y = data[target_column]
    
    logger.info(f"训练特征数量: {len(feature_columns)}")
    logger.info(f"目标列: {target_column}")
    logger.info(f"正样本比例: {y.mean():.3f}")
    
    return X, y, feature_columns

def train_catboost_model(X, y, feature_columns):
    """训练CatBoost模型"""
    logger.info("开始训练CatBoost模型...")
    
    try:
        from catboost import CatBoostClassifier
        
        # 获取模型参数
        params = MODEL_CONFIG["catboost"]["params"]
        
        # 创建模型
        model = CatBoostClassifier(
            iterations=params["iterations"],
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            random_seed=params["random_seed"],
            verbose=100
        )
        
        # 训练模型
        model.fit(X, y)
        
        # 保存模型
        model_path = PRETRAINED_MODELS_DIR / MODEL_CONFIG["catboost"]["filename"]
        model.save_model(str(model_path))
        
        # 保存特征列信息
        feature_info = {
            "feature_columns": feature_columns,
            "model_type": "CatBoost",
            "training_params": params
        }
        joblib.dump(feature_info, PRETRAINED_MODELS_DIR / "catboost_feature_info.pkl")
        
        logger.info(f"CatBoost模型训练完成，保存到: {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"CatBoost模型训练失败: {e}")
        return None

def train_xgboost_model(X, y, feature_columns):
    """训练XGBoost模型"""
    logger.info("开始训练XGBoost模型...")
    
    try:
        import xgboost as xgb
        
        # 获取模型参数
        params = MODEL_CONFIG["xgboost"]["params"]
        
        # 创建模型
        model = xgb.XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=params["random_state"],
            verbosity=1
        )
        
        # 训练模型
        model.fit(X, y)
        
        # 保存模型
        model_path = PRETRAINED_MODELS_DIR / MODEL_CONFIG["xgboost"]["filename"]
        joblib.dump(model, model_path)
        
        # 保存特征列信息
        feature_info = {
            "feature_columns": feature_columns,
            "model_type": "XGBoost",
            "training_params": params
        }
        joblib.dump(feature_info, PRETRAINED_MODELS_DIR / "xgboost_feature_info.pkl")
        
        logger.info(f"XGBoost模型训练完成，保存到: {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"XGBoost模型训练失败: {e}")
        return None

def train_randomforest_model(X, y, feature_columns):
    """训练Random Forest模型"""
    logger.info("开始训练Random Forest模型...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # 获取模型参数
        params = MODEL_CONFIG["randomforest"]["params"]
        
        # 创建模型
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=params["random_state"],
            n_jobs=-1
        )
        
        # 训练模型
        model.fit(X, y)
        
        # 保存模型
        model_path = PRETRAINED_MODELS_DIR / MODEL_CONFIG["randomforest"]["filename"]
        joblib.dump(model, model_path)
        
        # 保存特征列信息
        feature_info = {
            "feature_columns": feature_columns,
            "model_type": "Random Forest",
            "training_params": params
        }
        joblib.dump(feature_info, PRETRAINED_MODELS_DIR / "randomforest_feature_info.pkl")
        
        logger.info(f"Random Forest模型训练完成，保存到: {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Random Forest模型训练失败: {e}")
        return None

def create_ensemble_model(catboost_model, xgboost_model, randomforest_model, X, y, feature_columns):
    """创建集成模型"""
    logger.info("开始创建集成模型...")
    
    try:
        from sklearn.ensemble import VotingClassifier
        
        # 创建投票分类器
        ensemble = VotingClassifier(
            estimators=[
                ('catboost', catboost_model),
                ('xgboost', xgboost_model),
                ('randomforest', randomforest_model)
            ],
            voting='soft'  # 使用概率投票
        )
        
        # 训练集成模型
        ensemble.fit(X, y)
        
        # 保存集成模型
        model_path = PRETRAINED_MODELS_DIR / MODEL_CONFIG["ensemble"]["filename"]
        joblib.dump(ensemble, model_path)
        
        # 保存特征列信息
        feature_info = {
            "feature_columns": feature_columns,
            "model_type": "Ensemble",
            "strategy": MODEL_CONFIG["ensemble"]["strategy"]
        }
        joblib.dump(feature_info, PRETRAINED_MODELS_DIR / "ensemble_feature_info.pkl")
        
        logger.info(f"集成模型创建完成，保存到: {model_path}")
        return ensemble
        
    except Exception as e:
        logger.error(f"集成模型创建失败: {e}")
        return None

def evaluate_models(models, X, y):
    """评估模型性能"""
    logger.info("开始评估模型性能...")
    
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    results = {}
    
    for name, model in models.items():
        try:
            # 预测
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 计算指标
            accuracy = accuracy_score(y, y_pred)
            
            results[name] = {
                "accuracy": accuracy,
                "predictions": y_pred,
                "probabilities": y_pred_proba
            }
            
            logger.info(f"{name} 准确率: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"{name} 评估失败: {e}")
            results[name] = {"error": str(e)}
    
    return results

def main():
    """主函数"""
    logger.info("开始模型预训练流程...")
    
    # 1. 加载和预处理数据
    data_result = load_and_preprocess_data()
    if data_result is None:
        logger.error("数据加载失败，退出训练")
        return
    
    engineered_data, feature_engineer = data_result
    
    # 2. 准备训练数据
    X, y, feature_columns = prepare_training_data(engineered_data)
    
    # 3. 训练各个模型
    models = {}
    
    # CatBoost
    catboost_model = train_catboost_model(X, y, feature_columns)
    if catboost_model:
        models['CatBoost'] = catboost_model
    
    # XGBoost
    xgboost_model = train_xgboost_model(X, y, feature_columns)
    if xgboost_model:
        models['XGBoost'] = xgboost_model
    
    # Random Forest
    randomforest_model = train_randomforest_model(X, y, feature_columns)
    if randomforest_model:
        models['Random Forest'] = randomforest_model
    
    # 4. 创建集成模型
    if len(models) >= 2:
        ensemble_model = create_ensemble_model(
            catboost_model, xgboost_model, randomforest_model, X, y, feature_columns
        )
        if ensemble_model:
            models['Ensemble'] = ensemble_model
    
    # 5. 评估模型性能
    if models:
        evaluation_results = evaluate_models(models, X, y)
        logger.info("模型评估完成")
        
        # 保存评估结果
        joblib.dump(evaluation_results, PRETRAINED_MODELS_DIR / "model_evaluation_results.pkl")
    
    # 6. 保存特征工程器
    joblib.dump(feature_engineer, PRETRAINED_MODELS_DIR / "feature_engineer.pkl")
    
    logger.info("模型预训练流程完成！")
    logger.info(f"训练了 {len(models)} 个模型")
    logger.info(f"模型文件保存在: {PRETRAINED_MODELS_DIR}")

if __name__ == "__main__":
    main() 