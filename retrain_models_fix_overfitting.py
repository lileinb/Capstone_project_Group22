#!/usr/bin/env python3
"""
模型重训练脚本 - 解决过拟合问题
使用正则化和早停等技术防止过拟合
"""

import pandas as pd
import numpy as np
import sys
import os
import joblib
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append('.')

def generate_training_data(n_samples: int = 5000) -> pd.DataFrame:
    """生成训练数据"""
    np.random.seed(42)
    
    data = []
    
    # 生成平衡的数据集
    fraud_samples = int(n_samples * 0.3)  # 30% 欺诈样本
    normal_samples = n_samples - fraud_samples
    
    # 正常交易
    for _ in range(normal_samples):
        data.append({
            'Transaction Amount': np.random.lognormal(4, 0.8),  # 小到中等金额
            'Quantity': np.random.poisson(2) + 1,              # 少量购买
            'Customer Age': np.random.normal(35, 12),          # 正常年龄分布
            'Account Age Days': np.random.exponential(200) + 30, # 较老账户
            'Transaction Hour': np.random.choice(range(6, 23)), # 正常时间
            'Is Fraudulent': 0
        })
    
    # 欺诈交易
    for _ in range(fraud_samples):
        data.append({
            'Transaction Amount': np.random.lognormal(6, 1.2),  # 大额交易
            'Quantity': np.random.poisson(5) + 1,              # 大量购买
            'Customer Age': np.random.choice([18, 19, 20, 70, 75, 80]), # 异常年龄
            'Account Age Days': np.random.exponential(30) + 1,  # 新账户
            'Transaction Hour': np.random.choice([0, 1, 2, 3, 4, 22, 23]), # 异常时间
            'Is Fraudulent': 1
        })
    
    df = pd.DataFrame(data)
    
    # 数据清理
    df['Customer Age'] = np.clip(df['Customer Age'], 18, 80).astype(int)
    df['Account Age Days'] = np.clip(df['Account Age Days'], 1, 2000).astype(int)
    df['Transaction Amount'] = np.clip(df['Transaction Amount'], 10, 10000)
    df['Quantity'] = np.clip(df['Quantity'], 1, 20).astype(int)
    
    return df

def train_regularized_models(X_train, X_test, y_train, y_test):
    """训练正则化模型防止过拟合"""
    models = {}
    
    print("🔧 训练防过拟合模型...")
    
    # 1. 正则化随机森林
    print("   训练随机森林模型...")
    rf_model = RandomForestClassifier(
        n_estimators=100,        # 减少树的数量
        max_depth=10,           # 限制树的深度
        min_samples_split=20,   # 增加分裂所需的最小样本数
        min_samples_leaf=10,    # 增加叶节点的最小样本数
        max_features='sqrt',    # 限制特征数量
        random_state=42,
        class_weight='balanced'  # 处理类别不平衡
    )
    
    rf_model.fit(X_train, y_train)
    
    # 交叉验证评估
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1')
    print(f"   随机森林交叉验证F1分数: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # 测试集评估
    train_score = rf_model.score(X_train, y_train)
    test_score = rf_model.score(X_test, y_test)
    print(f"   训练集准确率: {train_score:.3f}")
    print(f"   测试集准确率: {test_score:.3f}")
    print(f"   过拟合程度: {train_score - test_score:.3f}")
    
    models['randomforest'] = rf_model
    
    # 2. 尝试导入和训练CatBoost（如果可用）
    try:
        import catboost as cb
        
        print("   训练CatBoost模型...")
        cat_model = cb.CatBoostClassifier(
            iterations=200,         # 减少迭代次数
            depth=6,               # 限制深度
            learning_rate=0.1,     # 降低学习率
            l2_leaf_reg=3,         # L2正则化
            random_seed=42,
            verbose=False,
            class_weights=[1, 2],  # 处理类别不平衡
            early_stopping_rounds=20  # 早停
        )
        
        cat_model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            use_best_model=True,
            verbose=False
        )
        
        train_score = cat_model.score(X_train, y_train)
        test_score = cat_model.score(X_test, y_test)
        print(f"   CatBoost训练集准确率: {train_score:.3f}")
        print(f"   CatBoost测试集准确率: {test_score:.3f}")
        print(f"   CatBoost过拟合程度: {train_score - test_score:.3f}")
        
        models['catboost'] = cat_model
        
    except ImportError:
        print("   CatBoost不可用，跳过")
    
    # 3. 尝试导入和训练XGBoost（如果可用）
    try:
        import xgboost as xgb
        
        print("   训练XGBoost模型...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,      # 减少估计器数量
            max_depth=6,           # 限制深度
            learning_rate=0.1,     # 降低学习率
            reg_alpha=1,           # L1正则化
            reg_lambda=1,          # L2正则化
            random_state=42,
            scale_pos_weight=2,    # 处理类别不平衡
            early_stopping_rounds=20
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        train_score = xgb_model.score(X_train, y_train)
        test_score = xgb_model.score(X_test, y_test)
        print(f"   XGBoost训练集准确率: {train_score:.3f}")
        print(f"   XGBoost测试集准确率: {test_score:.3f}")
        print(f"   XGBoost过拟合程度: {train_score - test_score:.3f}")
        
        models['xgboost'] = xgb_model
        
    except ImportError:
        print("   XGBoost不可用，跳过")
    
    return models

def save_models(models, feature_names):
    """保存模型"""
    print("\n💾 保存模型...")
    
    # 确保目录存在
    os.makedirs('models/pretrained', exist_ok=True)
    
    for model_name, model in models.items():
        try:
            if model_name == 'catboost':
                # CatBoost使用专用保存方法
                model.save_model(f'models/pretrained/{model_name}_model.cbm')
            else:
                # 其他模型使用joblib保存
                joblib.dump(model, f'models/pretrained/{model_name}_model.pkl')
            
            # 保存特征信息
            feature_info = {
                'feature_names': feature_names,
                'n_features': len(feature_names),
                'model_type': type(model).__name__
            }
            joblib.dump(feature_info, f'models/pretrained/{model_name}_feature_info.pkl')
            
            print(f"   ✅ {model_name} 模型保存成功")
            
        except Exception as e:
            print(f"   ❌ {model_name} 模型保存失败: {e}")

def main():
    """主函数"""
    print("🚀 模型重训练 - 解决过拟合问题")
    print("=" * 50)
    
    try:
        # 1. 生成训练数据
        print("📊 生成训练数据...")
        data = generate_training_data(5000)
        print(f"✅ 数据生成完成: {len(data)} 条")
        print(f"   欺诈率: {data['Is Fraudulent'].mean():.3f}")
        
        # 2. 准备特征和标签
        feature_columns = ['Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days', 'Transaction Hour']
        X = data[feature_columns]
        y = data['Is Fraudulent']
        
        print(f"   特征数量: {len(feature_columns)}")
        print(f"   特征名称: {feature_columns}")
        
        # 3. 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   训练集大小: {len(X_train)}")
        print(f"   测试集大小: {len(X_test)}")
        
        # 4. 训练模型
        models = train_regularized_models(X_train, X_test, y_train, y_test)
        
        # 5. 保存模型
        save_models(models, feature_columns)
        
        # 6. 模型评估报告
        print("\n📈 模型评估报告:")
        for model_name, model in models.items():
            print(f"\n{model_name.upper()} 模型:")
            y_pred = model.predict(X_test)
            print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
        
        print("\n✅ 模型重训练完成！")
        print("💡 建议:")
        print("   1. 重启Streamlit应用以加载新模型")
        print("   2. 测试模型预测功能")
        print("   3. 检查过拟合是否得到改善")
        
    except Exception as e:
        print(f"❌ 模型重训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
