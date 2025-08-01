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
    cleaned_data = cleaner.clean_data(data)
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
    
    # 排除目标变量、ID列和其他不需要的特征
    exclude_features = [
        'transaction_id', 'user_id', 'composite_risk_score', 'normalized_risk_score',
        'is_fraudulent', 'Is Fraudulent', 'fraud', 'is_fraud',  # 所有可能的目标变量
        'customer_id', 'Customer ID', 'transaction_id', 'Transaction ID'  # ID列
    ]

    feature_columns = [col for col in numeric_columns if col not in exclude_features]

    logger.info(f"排除的列: {[col for col in exclude_features if col in data.columns]}")
    logger.info(f"选择的特征列: {feature_columns}")
    
    # 查找目标列（支持多种可能的列名）
    possible_target_columns = [
        'is_fraudulent',  # 数据清理后的标准名称
        'Is Fraudulent',  # 原始数据集名称
        'fraud',          # 其他可能的名称
        'is_fraud'        # 其他可能的名称
    ]

    target_column = None
    for col in possible_target_columns:
        if col in data.columns:
            target_column = col
            logger.info(f"找到目标列: {col}")
            break

    if target_column is None:
        logger.error("未找到有效的目标列！")
        logger.error(f"可用列: {list(data.columns)}")
        raise ValueError("数据集中没有找到欺诈标签列")
    
    # 验证目标变量不在特征中
    if target_column in feature_columns:
        logger.error(f"目标变量 {target_column} 仍在特征列表中！这会导致数据泄露！")
        feature_columns.remove(target_column)
        logger.info(f"已从特征中移除目标变量: {target_column}")

    # 准备特征数据，处理缺失值和无穷值
    X = data[feature_columns].copy()

    # 处理缺失值
    X = X.fillna(0)

    # 处理无穷值
    X = X.replace([np.inf, -np.inf], 0)

    # 确保所有值都是有限的
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            if not np.isfinite(X[col]).all():
                logger.warning(f"列 {col} 包含非有限值，已替换为0")
                X[col] = X[col].replace([np.inf, -np.inf, np.nan], 0)

    y = data[target_column]

    # 详细的数据验证和统计
    logger.info(f"训练特征数量: {len(feature_columns)}")
    logger.info(f"目标列: {target_column}")
    logger.info(f"数据形状: X={X.shape}, y={y.shape}")
    logger.info(f"正样本数量: {y.sum()}")
    logger.info(f"负样本数量: {len(y) - y.sum()}")
    logger.info(f"正样本比例: {y.mean():.3f}")
    logger.info(f"数据不平衡比例: 1:{(len(y) - y.sum()) / y.sum():.1f}")

    # 检查是否有足够的正样本
    if y.sum() < 10:
        logger.warning(f"正样本数量过少 ({y.sum()})，可能影响模型训练效果")

    return X, y, feature_columns

def train_catboost_model(X, y, feature_columns):
    """训练CatBoost模型"""
    logger.info("开始训练CatBoost模型...")
    
    try:
        from catboost import CatBoostClassifier
        
        # 获取模型参数
        params = MODEL_CONFIG["catboost"]["params"]
        
        # 计算类别权重以处理数据不平衡
        pos_weight = (len(y) - y.sum()) / y.sum()  # 负样本数 / 正样本数

        # 创建模型，添加处理不平衡数据的参数
        model = CatBoostClassifier(
            iterations=params["iterations"],
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            random_seed=params["random_seed"],
            scale_pos_weight=pos_weight,  # 处理数据不平衡
            eval_metric='AUC',  # 使用AUC作为评估指标
            verbose=100
        )

        logger.info(f"CatBoost模型参数: scale_pos_weight={pos_weight:.2f}")
        
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
        
        # 计算类别权重以处理数据不平衡
        pos_weight = (len(y) - y.sum()) / y.sum()  # 负样本数 / 正样本数

        # 创建模型，添加处理不平衡数据的参数
        model = xgb.XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=params["random_state"],
            scale_pos_weight=pos_weight,  # 处理数据不平衡
            eval_metric='auc',  # 使用AUC作为评估指标
            verbosity=1
        )

        logger.info(f"XGBoost模型参数: scale_pos_weight={pos_weight:.2f}")
        
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



def create_ensemble_model(catboost_model, xgboost_model, X, y, feature_columns):
    """创建集成模型（CatBoost + XGBoost）"""
    logger.info("开始创建集成模型（CatBoost + XGBoost）...")

    try:
        from sklearn.ensemble import VotingClassifier

        # 验证模型是否有效
        if catboost_model is None or xgboost_model is None:
            logger.error("无法创建集成模型：基础模型为空")
            return None

        # 创建投票分类器（只使用CatBoost和XGBoost）
        ensemble = VotingClassifier(
            estimators=[
                ('catboost', catboost_model),
                ('xgboost', xgboost_model)
            ],
            voting='soft'  # 使用概率投票
        )

        logger.info("集成模型组成: CatBoost + XGBoost")
        
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

    # 4. 创建集成模型（只要有CatBoost和XGBoost就可以创建）
    if catboost_model and xgboost_model:
        ensemble_model = create_ensemble_model(
            catboost_model, xgboost_model, X, y, feature_columns
        )
        if ensemble_model:
            models['Ensemble'] = ensemble_model
    else:
        logger.warning("无法创建集成模型：需要CatBoost和XGBoost都训练成功")
    
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