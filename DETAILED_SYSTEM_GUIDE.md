# 电商欺诈风险预测系统 - 详细使用指南

## 📋 目录
- [系统概述](#系统概述)
- [功能模块详解](#功能模块详解)
- [核心算法实现](#核心算法实现)
- [前端界面功能](#前端界面功能)
- [后端技术架构](#后端技术架构)
- [使用流程指南](#使用流程指南)

## 🎯 系统概述

电商欺诈风险预测系统是一个基于机器学习的智能风控平台，采用多策略融合的方法进行欺诈检测。系统通过无监督学习、半监督学习和监督学习相结合的方式，实现对电商交易的全方位风险评估。

### 核心特性
- **多维度风险评估**: 结合聚类分析、专家规则和机器学习模型
- **伪标签生成**: 基于多策略的高质量伪标签生成
- **实时风险评分**: 动态阈值管理和风险等级划分
- **攻击类型识别**: 智能识别4种主要攻击模式
- **可解释性分析**: SHAP/LIME深度解释模型决策

## 🔧 功能模块详解

### 1. 数据上传与预处理模块

#### 前端界面功能
**文件位置**: `frontend/pages/upload_page.py`

**主要功能**:
- 📁 CSV文件上传和验证
- 📊 数据质量检测和报告
- 🔧 自动数据清理和预处理
- 📈 数据分布可视化

**界面操作流程**:
1. 选择数据源（上传文件或使用示例数据）
2. 系统自动验证数据格式和完整性
3. 显示数据质量报告和清理建议
4. 执行数据清理和标准化处理

#### 后端技术实现
**文件位置**: `backend/data_processor/`

**核心类和方法**:
```python
# backend/data_processor/data_loader.py
class DataLoader:
    def load_csv_file(self, file_path: str) -> pd.DataFrame
        """加载CSV文件并进行基础验证"""
    
    def validate_data_format(self, data: pd.DataFrame) -> bool
        """验证数据格式是否符合要求"""
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict
        """生成数据摘要统计信息"""

# backend/data_processor/data_cleaner.py  
class DataCleaner:
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame
        """完整的数据清理流程"""
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame
        """处理缺失值：智能填充策略"""
    
    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame
        """处理重复数据：基于关键字段去重"""
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame
        """处理异常值：IQR方法检测和处理"""
```

**技术实现细节**:
- **数据验证**: 检查必需字段、数据类型、取值范围
- **缺失值处理**: 基于字段类型的智能填充策略
- **异常值检测**: IQR方法检测和处理异常值
- **重复数据**: 基于关键字段的去重逻辑

### 2. 特征工程模块

#### 前端界面功能
**文件位置**: `frontend/pages/feature_analysis_page.py`

**主要功能**:
- 🔧 自动风险特征生成
- 📊 特征重要性分析
- 📈 特征分布可视化
- 🎯 特征选择和优化

**界面操作流程**:
1. 选择特征工程策略（标准/快速模式）
2. 系统自动生成多维度风险特征
3. 显示特征重要性排序和分析
4. 可视化特征分布和相关性

#### 后端技术实现
**文件位置**: `backend/feature_engineer/risk_features.py`

**核心类和方法**:
```python
class RiskFeatureEngineer:
    def engineer_all_features(self, data: pd.DataFrame) -> pd.DataFrame
        """生成所有风险特征的主入口"""
    
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame
        """创建时间相关特征"""
    
    def create_amount_features(self, data: pd.DataFrame) -> pd.DataFrame
        """创建金额相关特征"""
    
    def create_customer_features(self, data: pd.DataFrame) -> pd.DataFrame
        """创建客户相关特征"""
    
    def create_behavioral_features(self, data: pd.DataFrame) -> pd.DataFrame
        """创建行为相关特征"""
```

**特征生成策略**:
1. **时间特征**: 交易时间、工作日/周末、节假日标识
2. **金额特征**: 金额分位数、异常金额标识、金额变化率
3. **客户特征**: 账户年龄、历史交易频率、风险评分
4. **行为特征**: 设备指纹、地理位置、交易模式

### 3. 聚类分析模块 ⭐

#### 前端界面功能
**文件位置**: `frontend/pages/clustering_page.py`

**主要功能**:
- 📊 K-Means无监督聚类分析
- 🎯 聚类数量自动优化
- 📈 聚类结果可视化
- 🔍 聚类特征解释

**界面操作流程**:
1. 选择聚类特征和参数
2. 执行K-Means聚类算法
3. 显示聚类结果和特征分布
4. 分析每个聚类的风险特征

#### 后端技术实现
**文件位置**: `backend/clustering/`

**核心类和方法**:
```python
# backend/clustering/cluster_analyzer.py
class ClusterAnalyzer:
    def analyze_clusters(self, data: pd.DataFrame) -> Dict[str, Any]
        """执行聚类分析的主方法"""
    
    def _prepare_clustering_data(self, data: pd.DataFrame) -> pd.DataFrame
        """准备聚类数据：特征选择和标准化"""
    
    def _perform_clustering(self, cluster_data: pd.DataFrame) -> np.ndarray
        """执行K-Means聚类"""
    
    def _analyze_cluster_characteristics(self, data: pd.DataFrame, labels: np.ndarray) -> Dict
        """分析聚类特征和风险特征"""

# backend/clustering/cluster_risk_mapper.py
class ClusterRiskMapper:
    def map_clusters_to_risk_levels(self, cluster_results: Dict) -> Dict[int, str]
        """将聚类映射到风险等级"""
    
    def _calculate_cluster_risk_score(self, cluster_data: pd.DataFrame) -> float
        """计算聚类风险分数"""
    
    def _assign_risk_level(self, risk_score: float) -> str
        """分配风险等级"""
```

**算法实现思路**:
1. **特征选择**: 选择关键风险特征进行聚类
2. **聚类算法**: K-Means聚类，自动确定最优聚类数
3. **风险映射**: 基于聚类内欺诈率和风险特征分布确定风险等级
4. **结果解释**: 生成每个聚类的特征画像和风险描述

**技术细节**:
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

#### 前端界面功能
**文件位置**: `frontend/pages/risk_scoring_page.py`

**主要功能**:
- 🎯 多维度无监督风险评分
- 📊 动态阈值管理
- 📈 风险等级分布分析
- ⚙️ 评分策略配置

**界面操作流程**:
1. 选择风险评分模式（标准/快速）
2. 配置评分权重和参数
3. 执行多策略风险评分
4. 分析风险分布和阈值优化

#### 后端技术实现
**文件位置**: `backend/risk_scoring/`

**核心类和方法**:
```python
# backend/risk_scoring/risk_calculator.py
class RiskCalculator:
    def calculate_unsupervised_risk_score(self, data: pd.DataFrame, cluster_results: Dict) -> Dict
        """计算无监督风险评分"""
    
    def _calculate_cluster_risk(self, data: pd.DataFrame, cluster_results: Dict) -> np.ndarray
        """计算聚类风险分数"""
    
    def _calculate_rule_risk(self, data: pd.DataFrame) -> np.ndarray
        """计算规则风险分数"""
    
    def _calculate_model_risk(self, data: pd.DataFrame) -> np.ndarray
        """计算模型风险分数"""
    
    def _combine_risk_scores(self, cluster_risk: np.ndarray, rule_risk: np.ndarray, model_risk: np.ndarray) -> np.ndarray
        """融合多种风险分数"""

# backend/risk_scoring/dynamic_threshold_manager.py
class DynamicThresholdManager:
    def calculate_dynamic_thresholds(self, risk_scores: np.ndarray) -> Dict[str, float]
        """计算动态风险阈值"""
    
    def _calculate_percentile_thresholds(self, risk_scores: np.ndarray) -> Dict[str, float]
        """基于百分位数计算阈值"""
```

**算法思路**:
1. **多策略融合**: 聚类风险 + 专家规则 + 模型预测
2. **权重分配**: 聚类风险(40%) + 规则风险(35%) + 模型风险(25%)
3. **动态阈值**: 基于数据分布自动调整风险等级阈值
4. **标准/快速模式**: 提供不同精度和速度的计算选项

**技术实现**:
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

#### 前端界面功能
**文件位置**: `frontend/pages/pseudo_labeling_page.py`

**主要功能**:
- 🏷️ 多策略伪标签生成
- 📊 标签质量评估和校准
- 🎯 高质量标签筛选
- 📈 标签分布分析

**界面操作流程**:
1. 选择伪标签生成策略
2. 配置置信度阈值和参数
3. 执行伪标签生成算法
4. 评估标签质量和分布

#### 后端技术实现
**文件位置**: `backend/pseudo_labeling/`

**核心类和方法**:
```python
# backend/pseudo_labeling/pseudo_label_generator.py
class PseudoLabelGenerator:
    def generate_pseudo_labels(self, data: pd.DataFrame, strategy: str = 'ensemble') -> Dict
        """生成伪标签的主方法"""

    def generate_high_quality_pseudo_labels(self, data: pd.DataFrame, min_confidence: float = 0.8) -> Dict
        """生成高质量伪标签"""

    def _ensemble_strategy(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]
        """集成策略：融合多种方法"""

    def _risk_based_strategy(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]
        """风险策略：基于风险评分"""

    def _cluster_based_strategy(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]
        """聚类策略：基于聚类风险等级"""

    def _rule_based_strategy(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]
        """规则策略：基于专家业务规则"""
```

**算法策略**:
1. **集成策略 (Ensemble)**: 融合多种策略的投票结果
2. **风险策略 (Risk-based)**: 基于风险评分阈值生成标签
3. **聚类策略 (Cluster-based)**: 基于聚类风险等级生成标签
4. **规则策略 (Rule-based)**: 基于专家业务规则生成标签

**技术实现**:
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

## 📊 数据流程与系统集成

### 完整业务流程
```
数据上传 → 特征工程 → 聚类分析 → 风险评分 → 伪标签生成 → 模型预测 → 攻击分类 → 分析报告
```

### 模块间数据传递
1. **数据上传** → **特征工程**: 清理后的原始数据
2. **特征工程** → **聚类分析**: 工程化特征数据
3. **聚类分析** → **风险评分**: 聚类结果和风险映射
4. **风险评分** → **伪标签生成**: 风险分数和等级
5. **伪标签生成** → **模型预测**: 高质量伪标签
6. **模型预测** → **攻击分类**: 预测结果和概率
7. **攻击分类** → **分析报告**: 攻击类型和防护建议

### 核心算法总结

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

## 🎯 系统特色与创新

### 1. 多策略融合架构
- **无监督学习**: 发现未知模式和异常行为
- **半监督学习**: 生成高质量伪标签
- **监督学习**: 提供精确预测
- **专家规则**: 补充业务逻辑

### 2. 自适应风险评估
- **动态阈值管理**: 基于数据分布自动调整
- **实时风险等级**: 根据最新数据更新
- **个性化风险画像**: 针对不同用户群体
- **智能异常检测**: 发现新型欺诈模式

### 3. 可解释性设计
- **SHAP深度解释**: 特征贡献度分析
- **决策路径可视化**: 模型决策过程透明化
- **业务友好解释**: 非技术人员易理解
- **风险因子分解**: 详细风险来源分析

---

**文档版本**: v1.0
**最后更新**: 2025-01-17
**维护团队**: 电商风控技术团队
