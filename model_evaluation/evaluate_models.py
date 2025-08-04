#!/usr/bin/env python3
"""
模型性能评估和可视化
生成图表对比不同模型的性能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
warnings.filterwarnings('ignore')

# 设置中文字体和后端
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.switch_backend('Agg')  # 使用非交互式后端

def load_data():
    """加载并预处理数据（与训练时保持一致）"""
    print("📊 加载并预处理数据...")

    # 加载原始数据
    data_file = Path("Dataset/Fraudulent_E-Commerce_Transaction_Data_sampled_50000.csv")
    if not data_file.exists():
        data_file = Path("Dataset/Fraudulent_E-Commerce_Transaction_Data_2_sampled_50000.csv")

    df = pd.read_csv(data_file)
    print(f"原始数据形状: {df.shape}")

    # 数据清理（与训练时保持一致）
    import sys
    sys.path.append('.')
    from backend.data_processor.data_cleaner import DataCleaner
    from backend.feature_engineer.risk_features import RiskFeatureEngineer

    # 数据清理
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_data(df)
    print(f"清理后数据形状: {cleaned_data.shape}")

    # 特征工程
    feature_engineer = RiskFeatureEngineer()
    engineered_data = feature_engineer.engineer_all_features(cleaned_data)
    print(f"特征工程后数据形状: {engineered_data.shape}")

    print(f"欺诈样本: {engineered_data['is_fraudulent'].sum()}")
    print(f"正常样本: {len(engineered_data) - engineered_data['is_fraudulent'].sum()}")

    return engineered_data

def evaluate_model(model_path, X, y, model_name):
    """评估单个模型"""
    try:
        if model_path.suffix == '.cbm':
            # CatBoost模型
            import catboost as cb
            model = cb.CatBoostClassifier()
            model.load_model(str(model_path))
        else:
            # 其他模型
            model = joblib.load(model_path)
        
        # 预测
        y_pred = model.predict(X)
        
        # 获取概率预测
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)
            if y_pred_proba.shape[1] == 2:
                y_pred_proba = y_pred_proba[:, 1]
            else:
                y_pred_proba = np.max(y_pred_proba, axis=1)
        else:
            y_pred_proba = y_pred
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_pred_proba)
        }
        
        print(f"✅ {model_name} 评估完成")
        return metrics, y_pred, y_pred_proba
        
    except Exception as e:
        print(f"❌ {model_name} 评估失败: {e}")
        return None, None, None

def plot_metrics_comparison(results):
    """Plot metrics comparison chart"""
    if not results:
        print("❌ No results to plot")
        return

    print("📊 Plotting performance comparison chart...")

    models = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    # 准备数据
    data = []
    for metric in metrics_names:
        metric_values = [results[model][metric] for model in models]
        data.append(metric_values)
    
    # 绘制图表
    x = np.arange(len(models))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (metric_data, label, color) in enumerate(zip(data, metrics_labels, colors)):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, metric_data, width, label=label, color=color, alpha=0.8)
        
        # 添加数值标签
        for bar, value in zip(bars, metric_data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('model_evaluation/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Metrics comparison chart saved: model_evaluation/metrics_comparison.png")

def plot_confusion_matrices(results, y_true, predictions):
    """Plot confusion matrices comparison"""
    print("📊 Plotting confusion matrices...")
    
    n_models = len(results)
    cols = 2
    rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6*rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        if y_pred is None:
            continue
            
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制热力图
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # 添加数值标注
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                ax.text(k, j, str(cm[j, k]), ha='center', va='center', 
                       color='white' if cm[j, k] > cm.max()/2 else 'black', fontsize=14)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Fraud'])
        ax.set_yticklabels(['Normal', 'Fraud'])
        ax.set_title(f'{model_name} Confusion Matrix', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
    
    # 隐藏多余的子图
    for i in range(n_models, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('model_evaluation/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrices chart saved: model_evaluation/confusion_matrices.png")

def plot_performance_radar(results):
    """Plot performance radar chart"""
    print("📊 Plotting performance radar chart...")

    import numpy as np

    # Prepare data
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the shape
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (model_name, model_results) in enumerate(results.items()):
        values = [model_results[key] for key in metric_keys]
        values += values[:1]  # Close the shape

        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('model_evaluation/performance_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Performance radar chart saved: model_evaluation/performance_radar.png")

def main():
    """Main function"""
    print("🚀 Starting model performance evaluation...")
    
    # Load data
    df = load_data()
    
    # Model configuration (using training feature list)
    training_features = [
        'transaction_amount', 'quantity', 'customer_age', 'account_age_days', 'transaction_hour',
        'time_risk_score', 'is_night_transaction', 'is_early_morning', 'is_weekend', 'weekday',
        'time_period', 'amount_z_score', 'amount_risk_score', 'is_large_amount', 'is_small_amount',
        'amount_percentile', 'amount_anomaly_score', 'unit_price', 'is_high_quantity', 'is_single_item',
        'amount_range', 'amount_vs_user_avg', 'amount_deviation_from_user', 'device_risk_score',
        'is_mobile', 'is_desktop', 'is_tablet', 'ip_first_octet', 'ip_risk_score', 'location_name_length',
        'location_risk_score', 'account_age_risk_score', 'is_new_account', 'is_very_new_account',
        'is_mature_account', 'customer_age_risk_score', 'age_group', 'account_age_ratio',
        'user_transaction_frequency', 'is_frequent_user'
    ]

    models_config = [
        {
            'name': 'CatBoost',
            'file': 'models/pretrained/catboost_model.cbm',
            'features': training_features
        },
        {
            'name': 'XGBoost',
            'file': 'models/pretrained/xgboost_model.pkl',
            'features': training_features
        },
        {
            'name': 'Ensemble',
            'file': 'models/pretrained/ensemble_model.pkl',
            'features': training_features
        }
    ]
    
    results = {}
    predictions = {}
    y_true = df['is_fraudulent']  # Use cleaned column name

    # Evaluate all models
    for config in models_config:
        print(f"\n📈 Evaluating {config['name']} model...")

        model_path = Path(config['file'])
        if not model_path.exists():
            print(f"❌ Model file does not exist: {config['file']}")
            continue

        # Prepare features
        X = df[config['features']]

        # Evaluate model
        metrics, y_pred, y_pred_proba = evaluate_model(model_path, X, y_true, config['name'])
        
        if metrics:
            results[config['name']] = metrics
            predictions[config['name']] = y_pred
    
    if results:
        # Generate charts
        plot_metrics_comparison(results)
        plot_confusion_matrices(results, y_true, predictions)
        plot_performance_radar(results)

        # Print summary
        print("\n📋 Model Performance Summary:")
        print("=" * 70)
        print(f"{'Model':<12} {'Accuracy':<8} {'Precision':<8} {'Recall':<8} {'F1-Score':<8} {'AUC':<8}")
        print("=" * 70)

        for model_name, metrics in results.items():
            print(f"{model_name:<12} {metrics['accuracy']:<8.4f} {metrics['precision']:<8.4f} "
                  f"{metrics['recall']:<8.4f} {metrics['f1']:<8.4f} {metrics['auc']:<8.4f}")

        print("=" * 70)
        print("\n✅ Evaluation completed! Charts saved to model_evaluation/ directory")
    else:
        print("\n❌ No models were successfully evaluated")

if __name__ == "__main__":
    main()
