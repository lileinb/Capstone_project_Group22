# Model Performance Evaluation

This folder contains model performance evaluation scripts and generated charts.

## File Description

- `evaluate_models.py` - Main model evaluation script
- `metrics_comparison.png` - Model performance metrics comparison chart
- `confusion_matrices.png` - Confusion matrices comparison chart for all models
- `performance_radar.png` - Model performance radar chart

## Usage

Run the evaluation script:
```bash
python model_evaluation/evaluate_models.py
```

## Evaluation Results

The script evaluates the following 4 pre-trained models:
- CatBoost
- Ensemble (Ensemble Model)
- RandomForest (Random Forest)
- XGBoost

## Generated Charts

1. **Metrics Comparison Chart** (`metrics_comparison.png`)
   - Shows comparison of all models on accuracy, precision, recall, F1-score, AUC metrics
   - Bar chart format for intuitive comparison

2. **Confusion Matrices Chart** (`confusion_matrices.png`)
   - Shows confusion matrix for each model
   - Contains specific values for true positives, false positives, true negatives, false negatives

3. **Performance Radar Chart** (`performance_radar.png`)
   - Radar chart showing comprehensive performance of all models
   - Easy to observe model balance across different metrics

## 评估指标说明

- **准确率 (Accuracy)**: 正确预测的样本占总样本的比例
- **精确率 (Precision)**: 预测为欺诈的样本中真正是欺诈的比例
- **召回率 (Recall)**: 真正的欺诈样本中被正确识别的比例
- **F1分数 (F1-Score)**: 精确率和召回率的调和平均数
- **AUC**: ROC曲线下的面积，衡量模型的整体判别能力

## 注意事项

- 数据存在严重不平衡问题（欺诈样本仅占5%）
- 所有模型的F1分数都较低，表明需要进一步优化
- 建议使用过采样、成本敏感学习等技术改进模型性能
