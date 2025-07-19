# 电商欺诈风险预测系统 - 技术文档

## 📋 目录
- [系统概述](#系统概述)
- [架构设计](#架构设计)
- [核心功能模块](#核心功能模块)
- [前端界面功能](#前端界面功能)
- [后端核心算法](#后端核心算法)
- [数据流程](#数据流程)
- [技术实现细节](#技术实现细节)

## 🎯 系统概述

电商欺诈风险预测系统是一个基于机器学习的智能风控平台，采用多策略融合的方法进行欺诈检测。系统通过无监督学习、半监督学习和监督学习相结合的方式，实现对电商交易的全方位风险评估。

### 核心特性
- **多维度风险评估**: 结合聚类分析、专家规则和机器学习模型
- **伪标签生成**: 基于多策略的高质量伪标签生成
- **实时风险评分**: 动态阈值管理和风险等级划分
- **攻击类型识别**: 智能识别4种主要攻击模式
- **可解释性分析**: SHAP/LIME深度解释模型决策

### 技术栈
- **前端**: Streamlit (Python Web框架)
- **后端**: Python + Scikit-learn + CatBoost + XGBoost
- **数据处理**: Pandas + NumPy
- **可视化**: Plotly + Seaborn
- **机器学习**: 集成学习 + 无监督学习 + 半监督学习

## 🏗️ 架构设计

```
电商欺诈风险预测系统
├── frontend/                    # 前端界面层
│   ├── pages/                   # 页面模块
│   │   ├── upload_page.py       # 数据上传
│   │   ├── feature_analysis_page.py  # 特征工程
│   │   ├── clustering_page.py   # 聚类分析
│   │   ├── risk_scoring_page.py # 风险评分
│   │   ├── pseudo_labeling_page.py   # 伪标签生成
│   │   ├── model_prediction_page.py  # 模型预测
│   │   ├── attack_analysis_page.py   # 攻击分类
│   │   └── report_page.py       # 分析报告
│   └── __init__.py
├── backend/                     # 后端业务层
│   ├── data_processor/          # 数据处理模块
│   ├── feature_engineering/     # 特征工程模块
│   ├── clustering/              # 聚类分析模块
│   ├── risk_scoring/            # 风险评分模块
│   ├── pseudo_labeling/         # 伪标签生成模块
│   ├── ml_models/               # 机器学习模型模块
│   ├── attack_classification/   # 攻击分类模块
│   ├── explainer/               # 可解释性模块
│   └── analysis_reporting/      # 报告生成模块
├── models/                      # 模型存储
│   ├── pretrained/              # 预训练模型
│   └── user_trained/            # 用户训练模型
├── data/                        # 数据存储
└── scripts/                     # 工具脚本
```

## 🔧 核心功能模块

### 1. 数据上传与预处理模块

**前端界面**: `frontend/pages/upload_page.py`
**后端实现**: `backend/data_processor/`

#### 主要功能
- 支持CSV文件上传和在线数据加载
- 自动数据质量检测和清理
- 数据标准化和格式转换

#### 核心类和方法
```python
# backend/data_processor/data_loader.py
class DataLoader:
    def load_csv_file(self, file_path: str) -> pd.DataFrame
    def validate_data_format(self, data: pd.DataFrame) -> bool
    def get_data_summary(self, data: pd.DataFrame) -> Dict

# backend/data_processor/data_cleaner.py
class DataCleaner:
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame
    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame
```

#### 技术实现
- **数据验证**: 检查必需字段、数据类型、取值范围
- **缺失值处理**: 基于字段类型的智能填充策略
- **异常值检测**: IQR方法检测和处理异常值
- **重复数据**: 基于关键字段的去重逻辑

### 2. 特征工程模块

**前端界面**: `frontend/pages/feature_analysis_page.py`
**后端实现**: `backend/feature_engineering/` + `backend/feature_engineer/`

#### 主要功能
- 自动生成风险特征
- 特征选择和重要性分析
- 特征分布可视化

#### 核心类和方法
```python
# backend/feature_engineer/risk_features.py
class RiskFeatureEngineer:
    def engineer_all_features(self, data: pd.DataFrame) -> pd.DataFrame
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame
    def create_amount_features(self, data: pd.DataFrame) -> pd.DataFrame
    def create_customer_features(self, data: pd.DataFrame) -> pd.DataFrame
    def create_behavioral_features(self, data: pd.DataFrame) -> pd.DataFrame

# backend/feature_engineering/feature_selector.py
class FeatureSelector:
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]
    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict
```

#### 算法思路
1. **时间特征**: 交易时间、工作日/周末、节假日标识
2. **金额特征**: 金额分位数、异常金额标识、金额变化率
3. **客户特征**: 账户年龄、历史交易频率、风险评分
4. **行为特征**: 设备指纹、地理位置、交易模式

### 3. 聚类分析模块 ⭐

**前端界面**: `frontend/pages/clustering_page.py`
**后端实现**: `backend/clustering/`

#### 主要功能
- 无监督用户行为模式发现
- 聚类风险等级映射
- 聚类结果可视化和解释

#### 核心类和方法
```python
# backend/clustering/cluster_analyzer.py
class ClusterAnalyzer:
    def analyze_clusters(self, data: pd.DataFrame) -> Dict[str, Any]
    def _prepare_clustering_data(self, data: pd.DataFrame) -> pd.DataFrame
    def _perform_clustering(self, cluster_data: pd.DataFrame) -> np.ndarray
    def _analyze_cluster_characteristics(self, data: pd.DataFrame, labels: np.ndarray) -> Dict

# backend/clustering/cluster_risk_mapper.py
class ClusterRiskMapper:
    def map_clusters_to_risk_levels(self, cluster_results: Dict) -> Dict[int, str]
    def _calculate_cluster_risk_score(self, cluster_data: pd.DataFrame) -> float
    def _assign_risk_level(self, risk_score: float) -> str

# backend/clustering/cluster_interpreter.py
class ClusterInterpreter:
    def interpret_clusters(self, cluster_results: Dict) -> Dict[str, Any]
    def _generate_cluster_profiles(self, cluster_results: Dict) -> List[Dict]
```

#### 算法实现
1. **聚类算法**: K-Means聚类，自动确定最优聚类数
2. **特征选择**: 选择关键风险特征进行聚类
3. **风险映射**: 基于聚类内欺诈率和风险特征分布确定风险等级
4. **结果解释**: 生成每个聚类的特征画像和风险描述

#### 技术细节
```python
# 聚类特征选择
clustering_features = [
    'transaction_amount', 'quantity', 'customer_age', 
    'account_age_days', 'transaction_hour'
]

# 风险等级映射逻辑
def _assign_risk_level(self, risk_score: float) -> str:
    if risk_score >= 80:
        return 'critical'  # 极高风险
    elif risk_score >= 60:
        return 'high'      # 高风险
    elif risk_score >= 40:
        return 'medium'    # 中风险
    else:
        return 'low'       # 低风险
```

### 4. 风险评分模块 ⭐

**前端界面**: `frontend/pages/risk_scoring_page.py`
**后端实现**: `backend/risk_scoring/`

#### 主要功能
- 多维度无监督风险评分
- 动态阈值管理
- 风险等级分布分析

#### 核心类和方法
```python
# backend/risk_scoring/risk_calculator.py
class RiskCalculator:
    def calculate_unsupervised_risk_score(self, data: pd.DataFrame, cluster_results: Dict) -> Dict
    def _calculate_cluster_risk(self, data: pd.DataFrame, cluster_results: Dict) -> np.ndarray
    def _calculate_rule_risk(self, data: pd.DataFrame) -> np.ndarray
    def _calculate_model_risk(self, data: pd.DataFrame) -> np.ndarray
    def _combine_risk_scores(self, cluster_risk: np.ndarray, rule_risk: np.ndarray, model_risk: np.ndarray) -> np.ndarray

# backend/risk_scoring/standard_risk_calculator.py
class StandardRiskCalculator(RiskCalculator):
    def calculate_unsupervised_risk_score(self, data: pd.DataFrame, cluster_results: Dict) -> Dict

# backend/risk_scoring/fast_risk_calculator.py
class FastRiskCalculator(RiskCalculator):
    def calculate_unsupervised_risk_score(self, data: pd.DataFrame, cluster_results: Dict) -> Dict

# backend/risk_scoring/dynamic_threshold_manager.py
class DynamicThresholdManager:
    def calculate_dynamic_thresholds(self, risk_scores: np.ndarray) -> Dict[str, float]
    def _calculate_percentile_thresholds(self, risk_scores: np.ndarray) -> Dict[str, float]
```

#### 算法思路
1. **多策略融合**: 聚类风险 + 专家规则 + 模型预测
2. **权重分配**: 聚类风险(40%) + 规则风险(35%) + 模型风险(25%)
3. **动态阈值**: 基于数据分布自动调整风险等级阈值
4. **标准/快速模式**: 提供不同精度和速度的计算选项

#### 技术实现
```python
# 风险评分融合算法
def _combine_risk_scores(self, cluster_risk, rule_risk, model_risk):
    weights = {
        'cluster': 0.40,  # 聚类风险权重
        'rule': 0.35,     # 规则风险权重
        'model': 0.25     # 模型风险权重
    }

    combined_risk = (
        weights['cluster'] * cluster_risk +
        weights['rule'] * rule_risk +
        weights['model'] * model_risk
    )

    return np.clip(combined_risk, 0, 100)

# 动态阈值计算
def calculate_dynamic_thresholds(self, risk_scores):
    return {
        'low': np.percentile(risk_scores, 25),      # 25分位数
        'medium': np.percentile(risk_scores, 50),   # 50分位数
        'high': np.percentile(risk_scores, 75),     # 75分位数
        'critical': np.percentile(risk_scores, 90)  # 90分位数
    }
```

### 5. 伪标签生成模块 ⭐⭐

**前端界面**: `frontend/pages/pseudo_labeling_page.py`
**后端实现**: `backend/pseudo_labeling/`

#### 主要功能
- 多策略伪标签生成
- 标签质量评估和校准
- 高质量标签筛选

#### 核心类和方法
```python
# backend/pseudo_labeling/pseudo_label_generator.py
class PseudoLabelGenerator:
    def generate_pseudo_labels(self, data: pd.DataFrame, strategy: str = 'ensemble') -> Dict
    def generate_high_quality_pseudo_labels(self, data: pd.DataFrame, min_confidence: float = 0.8) -> Dict
    def _ensemble_strategy(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]
    def _risk_based_strategy(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]
    def _cluster_based_strategy(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]
    def _rule_based_strategy(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]

# backend/pseudo_labeling/fast_pseudo_label_generator.py
class FastPseudoLabelGenerator:
    def generate_fast_pseudo_labels(self, data: pd.DataFrame, risk_results: Dict = None) -> Dict
    def _simplified_ensemble(self, data: pd.DataFrame, risk_results: Dict) -> Tuple[np.ndarray, np.ndarray]
```

#### 算法策略
1. **集成策略 (Ensemble)**: 融合多种策略的投票结果
2. **风险策略 (Risk-based)**: 基于风险评分阈值生成标签
3. **聚类策略 (Cluster-based)**: 基于聚类风险等级生成标签
4. **规则策略 (Rule-based)**: 基于专家业务规则生成标签

#### 技术实现
```python
# 集成策略实现
def _ensemble_strategy(self, data: pd.DataFrame):
    # 获取各策略结果
    risk_labels, risk_conf = self._risk_based_strategy(data)
    cluster_labels, cluster_conf = self._cluster_based_strategy(data)
    rule_labels, rule_conf = self._rule_based_strategy(data)

    # 加权投票
    weights = {'risk': 0.45, 'cluster': 0.35, 'rule': 0.20}

    ensemble_confidence = (
        weights['risk'] * risk_conf +
        weights['cluster'] * cluster_conf +
        weights['rule'] * rule_conf
    )

    # 基于置信度阈值生成最终标签
    ensemble_labels = (ensemble_confidence >= self.confidence_threshold).astype(int)

    return ensemble_labels, ensemble_confidence

# 高质量标签筛选
def generate_high_quality_pseudo_labels(self, data: pd.DataFrame, min_confidence: float = 0.8):
    labels, confidences = self._ensemble_strategy(data)

    # 筛选高置信度标签
    high_quality_mask = confidences >= min_confidence
    high_quality_indices = np.where(high_quality_mask)[0]
    high_quality_labels = labels[high_quality_mask]
    high_quality_confidences = confidences[high_quality_mask]

    return {
        'all_labels': labels,
        'all_confidences': confidences,
        'high_quality_indices': high_quality_indices,
        'high_quality_labels': high_quality_labels,
        'high_quality_confidences': high_quality_confidences,
        'quality_rate': len(high_quality_indices) / len(labels)
    }
```

### 6. 模型预测模块

**前端界面**: `frontend/pages/model_prediction_page.py`
**后端实现**: `backend/ml_models/`

#### 主要功能
- 多模型预测和性能对比
- 集成学习预测
- 预测结果分析和可视化

#### 核心类和方法
```python
# backend/ml_models/model_manager.py
class ModelManager:
    def load_model(self, model_name: str) -> Any
    def predict_with_model(self, model, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]
    def evaluate_model(self, predictions: np.ndarray, probabilities: np.ndarray, y_true: pd.Series) -> Dict
    def get_available_models(self) -> List[str]

# backend/ml_models/ensemble_predictor.py
class EnsemblePredictor:
    def predict(self, model_probabilities: Dict[str, np.ndarray], threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]
    def _weighted_voting(self, model_probabilities: Dict[str, np.ndarray]) -> np.ndarray
    def _simple_voting(self, model_probabilities: Dict[str, np.ndarray]) -> np.ndarray
```

#### 支持的模型
1. **CatBoost**: 处理类别特征强，抗过拟合
2. **XGBoost**: 速度快，内存效率高
3. **Random Forest**: 稳定性好，不易过拟合
4. **Ensemble**: 融合上述模型的集成预测

#### 技术实现
```python
# 集成预测算法
def _weighted_voting(self, model_probabilities):
    ensemble_probs = np.zeros(len(list(model_probabilities.values())[0]))
    total_weight = 0

    # 模型权重配置
    weights = {
        'catboost': 0.4,
        'xgboost': 0.3,
        'randomforest': 0.3
    }

    for model_name, probs in model_probabilities.items():
        weight = weights.get(model_name.lower(), 1.0)
        ensemble_probs += weight * probs
        total_weight += weight

    return ensemble_probs / total_weight if total_weight > 0 else ensemble_probs
```

### 7. 攻击分类模块

**前端界面**: `frontend/pages/attack_analysis_page.py`
**后端实现**: `backend/attack_classification/`

#### 主要功能
- 识别4种主要攻击类型
- 攻击模式分析
- 防护建议生成

#### 核心类和方法
```python
# backend/attack_classification/attack_classifier.py
class AttackClassifier:
    def classify_attacks(self, data: pd.DataFrame) -> Dict[str, Any]
    def _classify_single_transaction(self, transaction: pd.Series) -> str
    def _calculate_pattern_score(self, transaction: pd.Series, pattern: Dict) -> float
    def _analyze_attack_patterns(self, data: pd.DataFrame, results: List[Dict]) -> Dict

# backend/attack_classification/attack_pattern_analyzer.py
class AttackPatternAnalyzer:
    def analyze_patterns(self, attack_results: Dict) -> Dict[str, Any]
    def _analyze_temporal_patterns(self, attack_data: pd.DataFrame) -> Dict
    def _analyze_amount_patterns(self, attack_data: pd.DataFrame) -> Dict
```

#### 攻击类型定义
1. **账户接管攻击 (Account Takeover)**
   - 特征: 新账户大额交易、异常登录时间
   - 风险等级: 极高

2. **身份盗用攻击 (Identity Theft)**
   - 特征: 个人信息不匹配、多账户关联
   - 风险等级: 高

3. **批量欺诈攻击 (Bulk Fraud)**
   - 特征: 短时间大量交易、相似交易模式
   - 风险等级: 高

4. **测试性攻击 (Testing Attack)**
   - 特征: 小额多次交易、探测性行为
   - 风险等级: 中

#### 算法实现
```python
# 攻击模式定义
attack_patterns = {
    'account_takeover': {
        'characteristics': [
            '新账户大额交易',
            '异常时间登录',
            '设备指纹变化'
        ],
        'risk_level': 'critical',
        'weight': 0.4
    },
    'identity_theft': {
        'characteristics': [
            '个人信息不匹配',
            '多账户关联',
            '地理位置异常'
        ],
        'risk_level': 'high',
        'weight': 0.3
    }
    # ... 其他攻击类型
}

# 攻击分类算法
def _classify_single_transaction(self, transaction):
    scores = {}

    for attack_type, pattern in self.attack_patterns.items():
        score = self._calculate_pattern_score(transaction, pattern)
        scores[attack_type] = score

    # 返回得分最高的攻击类型
    return max(scores, key=scores.get)
```

### 8. 可解释性分析模块

**前端界面**: `frontend/pages/report_page.py` (SHAP分析部分)
**后端实现**: `backend/explainer/`

#### 主要功能
- SHAP值计算和可视化
- 特征重要性分析
- 模型决策解释

#### 核心类和方法
```python
# backend/explainer/shap_explainer.py
class SHAPExplainer:
    def explain_predictions(self, model, X: pd.DataFrame, sample_size: int = 100) -> Dict
    def _calculate_shap_values(self, model, X: pd.DataFrame) -> np.ndarray
    def _create_shap_plots(self, shap_values: np.ndarray, X: pd.DataFrame) -> Dict
```

## 📊 数据流程

### 完整业务流程
```
1. 数据上传 → 2. 特征工程 → 3. 聚类分析 → 4. 风险评分 → 5. 伪标签生成 → 6. 模型预测 → 7. 攻击分类 → 8. 分析报告
```

#### 详细流程说明

1. **数据上传阶段**
   - 用户上传CSV文件或使用示例数据
   - 系统自动验证数据格式和完整性
   - 执行数据清理和预处理

2. **特征工程阶段**
   - 基于原始数据生成风险特征
   - 包括时间、金额、客户、行为四大类特征
   - 特征选择和重要性分析

3. **聚类分析阶段**
   - 使用K-Means对用户行为进行聚类
   - 自动确定最优聚类数量
   - 为每个聚类分配风险等级

4. **风险评分阶段**
   - 融合聚类、规则、模型三种风险评分
   - 计算综合风险分数(0-100)
   - 动态调整风险等级阈值

5. **伪标签生成阶段**
   - 基于多策略生成高质量伪标签
   - 支持集成、风险、聚类、规则四种策略
   - 筛选高置信度标签用于模型训练

6. **模型预测阶段**
   - 使用预训练模型进行欺诈预测
   - 支持多模型对比和集成预测
   - 提供概率分布和阈值分析

7. **攻击分类阶段**
   - 识别具体的攻击类型和模式
   - 分析攻击特征和时间分布
   - 生成针对性防护建议

8. **分析报告阶段**
   - 生成综合分析报告
   - SHAP可解释性分析
   - 导出结果和建议

## 🔬 技术实现细节

### 核心算法详解

#### 1. 聚类风险映射算法
```python
def _calculate_cluster_risk_score(self, cluster_data: pd.DataFrame) -> float:
    # 基于多个风险因子计算聚类风险分数
    risk_factors = {
        'high_amount_rate': self._calculate_high_amount_rate(cluster_data),
        'new_account_rate': self._calculate_new_account_rate(cluster_data),
        'unusual_time_rate': self._calculate_unusual_time_rate(cluster_data),
        'high_quantity_rate': self._calculate_high_quantity_rate(cluster_data)
    }

    # 加权计算风险分数
    weights = {'high_amount_rate': 0.3, 'new_account_rate': 0.3,
               'unusual_time_rate': 0.2, 'high_quantity_rate': 0.2}

    risk_score = sum(weights[factor] * score for factor, score in risk_factors.items())
    return min(risk_score * 100, 100)  # 归一化到0-100
```

#### 2. 伪标签质量评估算法
```python
def _evaluate_label_quality(self, labels: np.ndarray, confidences: np.ndarray) -> Dict:
    # 计算标签质量指标
    high_conf_mask = confidences >= 0.8
    medium_conf_mask = (confidences >= 0.6) & (confidences < 0.8)
    low_conf_mask = confidences < 0.6

    quality_metrics = {
        'high_quality_rate': np.sum(high_conf_mask) / len(labels),
        'medium_quality_rate': np.sum(medium_conf_mask) / len(labels),
        'low_quality_rate': np.sum(low_conf_mask) / len(labels),
        'avg_confidence': np.mean(confidences),
        'fraud_rate': np.mean(labels)
    }

    return quality_metrics
```

#### 3. 动态阈值优化算法
```python
def optimize_thresholds(self, risk_scores: np.ndarray, target_fraud_rate: float = 0.1) -> Dict:
    # 基于目标欺诈率优化阈值
    sorted_scores = np.sort(risk_scores)[::-1]  # 降序排列
    target_count = int(len(sorted_scores) * target_fraud_rate)

    if target_count > 0:
        optimal_threshold = sorted_scores[target_count - 1]
    else:
        optimal_threshold = np.percentile(risk_scores, 90)

    return {
        'optimal_threshold': optimal_threshold,
        'expected_fraud_rate': target_fraud_rate,
        'threshold_percentile': (1 - target_fraud_rate) * 100
    }
```

### 性能优化策略

#### 1. 数据处理优化
- **分块处理**: 大数据集分块处理，避免内存溢出
- **并行计算**: 使用多进程处理独立的计算任务
- **缓存机制**: 缓存中间结果，避免重复计算

#### 2. 模型加载优化
- **延迟加载**: 按需加载模型，减少启动时间
- **模型压缩**: 使用模型压缩技术减少存储空间
- **预热机制**: 预先加载常用模型到内存

#### 3. 前端性能优化
- **异步处理**: 长时间计算使用异步处理
- **进度显示**: 实时显示处理进度
- **结果缓存**: 缓存计算结果，支持快速切换

## 🎯 系统特色与创新

### 1. 多策略融合架构
- 无监督学习发现未知模式
- 半监督学习生成高质量标签
- 监督学习提供精确预测
- 专家规则补充业务逻辑

### 2. 自适应风险评估
- 动态阈值管理
- 实时风险等级调整
- 个性化风险画像
- 智能异常检测

### 3. 可解释性设计
- SHAP深度解释
- 特征重要性分析
- 决策路径可视化
- 业务友好的解释

### 4. 工程化实现
- 模块化架构设计
- 可扩展的插件机制
- 完善的错误处理
- 用户友好的界面

## 📈 系统效果评估

### 预期性能指标
- **准确率**: > 95%
- **召回率**: > 90%
- **精确率**: > 85%
- **F1分数**: > 87%
- **处理速度**: < 2秒/千条记录

### 业务价值
- **风险识别**: 提前发现潜在欺诈风险
- **损失减少**: 降低欺诈造成的经济损失
- **效率提升**: 自动化风险评估流程
- **决策支持**: 提供数据驱动的决策依据

---

**文档版本**: v1.0
**最后更新**: 2025-01-17
**维护团队**: 电商风控技术团队
