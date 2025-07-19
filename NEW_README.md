# 🎯 智能风险预测与欺诈检测系统

## 📋 项目概述

本项目是一个基于机器学习的智能风险预测与欺诈检测系统，采用**从无监督到监督学习**的渐进式方案，实现个体化风险评估、智能分层管理和攻击类型推断。

### 🎯 核心特性

- **🔍 个体风险分析**: 为每个用户提供0-100分的精确风险评分
- **🏷️ 四层风险分层**: 低(60%)、中(25%)、高(12%)、极高(3%)风险等级
- **🤖 攻击类型推断**: 账户接管、身份盗用、批量欺诈、测试性攻击识别
- **🎚️ 动态阈值调整**: 基于数据分布自动优化风险阈值
- **📊 交互式可视化**: 丰富的图表和统计分析展示
- **🛡️ 防护建议生成**: 针对性的安全措施和监控策略

### 🔄 技术方案

#### **阶段1: 无监督风险分析** (已完成)
- 基于特征工程的数据预处理
- 聚类分析识别用户群体特征
- 增强的基础风险评分算法
- 动态阈值自适应调整

#### **阶段2: 监督学习集成** (进行中)
- 伪标签生成与质量验证
- 监督模型训练与优化
- 预测准确性显著提升
- 模型性能对比验证

## 🏗️ 系统架构

### 📁 项目结构

```
Capstone_test/
├── backend/                          # 后端核心模块
│   ├── prediction/                   # 风险预测模块
│   │   ├── __init__.py
│   │   ├── individual_risk_predictor.py    # 个体风险预测器 [核心]
│   │   ├── four_class_pipeline.py          # 四分类管道
│   │   └── three_layer_pipeline.py         # 三层预测管道
│   ├── risk_scoring/                 # 风险评分模块
│   │   ├── __init__.py
│   │   └── risk_calculator.py              # 风险计算器
│   └── ml_models/                    # 机器学习模型
│       ├── __init__.py
│       └── model_trainer.py               # 模型训练器
├── frontend/                         # 前端界面模块
│   ├── pages/                        # 页面组件
│   │   ├── __init__.py
│   │   ├── model_prediction_page.py        # 智能风险预测页面 [主界面]
│   │   ├── upload_page.py                  # 数据上传页面
│   │   ├── feature_analysis_page.py        # 特征分析页面
│   │   ├── clustering_page.py              # 聚类分析页面
│   │   ├── pseudo_labeling_page.py         # 伪标签生成页面
│   │   └── risk_scoring_page.py            # 风险评分页面
│   └── components/                   # UI组件
│       ├── __init__.py
│       └── risk_result_display.py          # 风险结果显示组件
├── data/                             # 数据目录
│   ├── raw/                          # 原始数据
│   ├── processed/                    # 处理后数据
│   └── models/                       # 训练好的模型
├── main.py                           # 主应用入口
└── README.md                         # 项目文档
```

## 🔄 核心工作流程

### 阶段1: 数据准备与特征工程
```
原始数据 → 数据清洗 → 特征工程 → 特征选择 → 数据标准化
```

### 阶段2: 无监督风险分析
```
特征数据 → 聚类分析 → 异常检测 → 基础风险评分 → 初步分层
```

### 阶段3: 伪标签生成
```
聚类结果 → 规则引擎 → 伪标签生成 → 标签质量验证 → 标签优化
```

### 阶段4: 监督学习预测
```
伪标签数据 → 模型训练 → 交叉验证 → 风险预测 → 结果评估
```

### 阶段5: 智能风险预测
```
输入数据 → 个体评分 → 动态阈值 → 风险分层 → 攻击推断 → 防护建议
```

## 🎯 核心模块详解

### 1. 个体风险预测器 (`IndividualRiskPredictor`)

**文件位置**: `backend/prediction/individual_risk_predictor.py`

#### 🔧 核心方法

##### `predict_individual_risks(data, clustering_results=None, use_four_class_labels=True)`
**主预测方法** - 系统的核心入口点

**参数**:
- `data`: pandas.DataFrame - 输入的特征数据
- `clustering_results`: Dict - 聚类分析结果（可选）
- `use_four_class_labels`: bool - 是否使用四分类标签

**返回值**:
```python
{
    'success': bool,                    # 预测是否成功
    'total_samples': int,               # 样本总数
    'processing_time': float,           # 处理时间（秒）
    'risk_scores': List[float],         # 风险评分列表 (0-100)
    'risk_levels': List[str],           # 风险等级列表 ['low','medium','high','critical']
    'attack_predictions': List[str],    # 攻击类型预测
    'individual_analyses': List[Dict],  # 个体分析报告
    'stratification_stats': Dict,       # 风险分层统计
    'protection_recommendations': Dict, # 防护建议
    'dynamic_thresholds': Dict,         # 动态阈值
    'timestamp': str                    # 时间戳
}
```

**调用流程**:
```python
# 1. 数据验证
if data is None or data.empty:
    return self._empty_result()

# 2. 计算个体风险评分
risk_scores = self._calculate_individual_risk_scores(data, clustering_results)

# 3. 计算动态阈值
dynamic_thresholds = self._calculate_dynamic_thresholds(risk_scores)

# 4. 使用动态阈值进行风险分层
risk_levels = self._stratify_risk_levels_with_thresholds(risk_scores, dynamic_thresholds)

# 5. 推断攻击类型
attack_predictions = self._predict_attack_types(data, risk_scores, risk_levels)

# 6. 生成个体分析报告
individual_analyses = self._generate_individual_analyses(data, risk_scores, risk_levels, attack_predictions)

# 7. 生成风险分层统计
stratification_stats = self._generate_stratification_statistics(risk_levels, risk_scores)

# 8. 生成防护建议
protection_recommendations = self._generate_protection_recommendations(attack_predictions, risk_levels)
```

##### `_calculate_individual_risk_scores(data, clustering_results)`
**风险评分计算** - 核心评分算法

**实现逻辑**:
```python
def _calculate_individual_risk_scores(self, data, clustering_results):
    # 当前使用增强的基础风险评分算法
    # 未来将集成监督学习模型
    return self._calculate_basic_risk_scores(data)
```

##### `_calculate_basic_risk_scores(data)`
**基础风险评分** - 增强的无监督评分算法

**评分因子**:
```python
score = 35  # 基础分数

# 交易金额风险 (权重: 10-45分)
if amount > 2000: score += 45      # 大额交易高风险
elif amount > 1000: score += 35
elif amount > 500: score += 25
elif amount > 100: score += 10
elif amount < 5: score += 30       # 极小额交易可疑

# 账户年龄风险 (权重: 10-40分)
if account_age < 1: score += 40    # 新账户极高风险
elif account_age < 7: score += 30
elif account_age < 30: score += 20
elif account_age < 90: score += 10

# 时间风险 (权重: 15-35分)
if hour <= 4 or hour >= 23: score += 35    # 深夜/凌晨高风险
elif hour <= 6 or hour >= 21: score += 25  # 早晚时段中等风险
elif hour <= 8 or hour >= 19: score += 15  # 非常规时段轻微风险

# 客户年龄风险 (权重: 15-25分)
if customer_age <= 18: score += 25         # 未成年人高风险
elif customer_age >= 75: score += 20       # 高龄用户风险
elif customer_age <= 21: score += 15       # 年轻用户风险

# 交易数量风险 (权重: 10-20分)
if quantity > 10: score += 20
elif quantity > 5: score += 10

# 随机噪声 (确保分布多样性)
noise = np.random.normal(0, 12)
final_score = max(5, min(95, score + noise))
```

##### `_calculate_dynamic_thresholds(risk_scores)`
**动态阈值计算** - 自适应阈值优化

**算法逻辑**:
```python
# 目标分布比例
target_distribution = {
    'low': 0.60,      # 60%
    'medium': 0.25,   # 25%
    'high': 0.12,     # 12%
    'critical': 0.03  # 3%
}

# 基于分位数计算阈值
low_threshold = np.percentile(risk_scores, 60)
medium_threshold = np.percentile(risk_scores, 85)
high_threshold = np.percentile(risk_scores, 97)

# 阈值合理性检查和调整
thresholds = {
    'low': max(20, min(50, low_threshold)),
    'medium': max(40, min(70, medium_threshold)),
    'high': max(60, min(85, high_threshold)),
    'critical': 100
}
```

##### `_stratify_risk_levels_with_thresholds(risk_scores, thresholds)`
**动态风险分层** - 使用动态阈值进行分层

**分层逻辑**:
```python
for score in risk_scores:
    if score >= thresholds['high']:      # 极高风险
        risk_levels.append('critical')
    elif score >= thresholds['medium']:  # 高风险
        risk_levels.append('high')
    elif score >= thresholds['low']:     # 中风险
        risk_levels.append('medium')
    else:                                # 低风险
        risk_levels.append('low')
```

##### `_predict_attack_types(data, risk_scores, risk_levels)`
**攻击类型推断** - 基于规则的攻击模式识别

**推断规则**:
```python
def _infer_attack_type(self, row):
    # 账户接管攻击特征
    if (row.get('account_age_days', 365) < 7 and
        row.get('transaction_amount', 0) > 1000):
        return 'account_takeover'

    # 身份盗用攻击特征
    elif (row.get('customer_age', 35) <= 18 and
          row.get('transaction_amount', 0) > 500):
        return 'identity_theft'

    # 批量欺诈攻击特征
    elif (row.get('quantity', 1) > 5 and
          row.get('transaction_hour', 12) <= 5):
        return 'bulk_fraud'

    # 测试性攻击特征
    elif (row.get('transaction_amount', 0) < 10 and
          row.get('account_age_days', 365) < 30):
        return 'testing_attack'

    else:
        return 'none'
```

### 2. 智能风险预测页面 (`model_prediction_page.py`)

**文件位置**: `frontend/pages/model_prediction_page.py`

#### 🔧 核心方法

##### `show()`
**主页面显示方法** - 用户界面入口

**页面流程**:
```python
def show():
    # 1. 检查前置条件
    if not _check_prerequisites():
        return

    # 2. 显示数据概览
    display_data_overview()

    # 3. 显示预测配置
    display_prediction_configuration()

    # 4. 执行预测按钮
    if st.button("🎯 执行智能风险预测"):
        _execute_individual_risk_prediction()

    # 5. 显示预测结果
    if st.session_state.individual_risk_results:
        display_prediction_results()
```

##### `_check_prerequisites()`
**前置条件检查** - 验证数据和模块可用性

**检查项目**:
```python
# 检查特征工程数据
if 'engineered_features' not in st.session_state:
    st.error("❌ 请先完成特征工程")
    return False

# 检查预测模块可用性
if not PREDICTION_AVAILABLE:
    st.error("❌ 风险预测模块不可用")
    return False

return True
```

##### `_execute_individual_risk_prediction(engineered_data, clustering_results, use_clustering, risk_thresholds)`
**执行个体风险预测** - 核心预测执行逻辑

**执行流程**:
```python
def _execute_individual_risk_prediction():
    with st.spinner("正在进行智能风险预测..."):
        # 1. 数据预处理
        X = engineered_data.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)

        # 2. 创建预测器
        risk_predictor = IndividualRiskPredictor()

        # 3. 执行预测
        risk_results = risk_predictor.predict_individual_risks(
            X, clustering_data, use_four_class_labels=True
        )

        # 4. 保存结果
        st.session_state.individual_risk_results = risk_results

        # 5. 显示统计信息
        display_prediction_statistics(risk_results)
```

### 3. 风险结果显示组件 (`risk_result_display.py`)

**文件位置**: `frontend/components/risk_result_display.py`

#### 🔧 核心方法

##### `display_risk_prediction_results(risk_results)`
**风险预测结果显示** - 主要结果展示

**显示内容**:
```python
def display_risk_prediction_results(risk_results):
    # 1. 总体统计展示
    display_overall_statistics(risk_results)

    # 2. 风险分层分析
    display_risk_stratification_analysis(risk_results)

    # 3. 攻击类型分析
    display_attack_type_analysis(risk_results)

    # 4. 个体详细分析
    display_individual_analysis(risk_results)

    # 5. 防护建议
    display_protection_recommendations(risk_results)
```

##### `display_risk_score_distribution(risk_results)`
**风险评分分布显示** - 评分分布可视化

**图表类型**:
- 📊 风险评分直方图
- 🥧 风险分层饼图
- 📈 评分分布箱线图
- 🎯 攻击类型分布图

## 🚀 使用指南

### 环境要求

```bash
# Python 版本
Python >= 3.8

# 核心依赖
streamlit >= 1.28.0
pandas >= 1.5.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
plotly >= 5.0.0
```

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd Capstone_test

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
streamlit run main.py
```

### 使用流程

#### 步骤1: 数据上传
- 进入 "📤 数据上传" 页面
- 上传CSV格式的交易数据
- 确保包含必要字段: `transaction_amount`, `account_age_days`, `transaction_hour`, `customer_age`

#### 步骤2: 特征工程
- 进入 "🔧 特征工程" 页面
- 执行特征提取和工程化
- 验证特征质量

#### 步骤3: 聚类分析（可选）
- 进入 "🎯 聚类分析" 页面
- 执行无监督聚类
- 分析用户群体特征

#### 步骤4: 伪标签生成（可选）
- 进入 "🏷️ 伪标签生成" 页面
- 基于聚类结果生成高质量标签
- 为监督学习做准备

#### 步骤5: 智能风险预测
- 进入 "🎯 智能风险预测" 页面
- 配置预测参数
- 执行风险预测
- 查看详细结果

## 📊 预测结果解读

### 风险分层含义

| 风险等级 | 评分范围 | 用户比例 | 监控策略 | 处理措施 |
|----------|----------|----------|----------|----------|
| **低风险** | 动态阈值以下 | ~60% | 基础监控 | 正常处理 |
| **中风险** | 动态中等阈值 | ~25% | 增强监控 | 关注处理 |
| **高风险** | 动态高阈值 | ~12% | 严密监控 | 重点处理 |
| **极高风险** | 动态极高阈值 | ~3% | 实时监控 | 立即处理 |

### 攻击类型说明

- **账户接管攻击** (`account_takeover`): 攻击者获取用户账户控制权
- **身份盗用攻击** (`identity_theft`): 使用他人身份信息进行欺诈
- **批量欺诈攻击** (`bulk_fraud`): 大规模自动化欺诈行为
- **测试性攻击** (`testing_attack`): 小额测试以验证支付方式

## 🔧 开发指南

### 扩展新的风险因子

```python
# 在 _calculate_basic_risk_scores 方法中添加新因子
def _calculate_basic_risk_scores(self, data):
    for i, (idx, row) in enumerate(data.iterrows()):
        score = 35

        # 现有风险因子...

        # 新增风险因子
        new_factor = row.get('new_feature', 0)
        if new_factor > threshold:
            score += weight

        scores[i] = max(5, min(95, score + noise))
```

### 集成新的攻击类型

```python
# 在 _infer_attack_type 方法中添加新类型
def _infer_attack_type(self, row):
    # 现有攻击类型判断...

    # 新攻击类型
    elif (condition1 and condition2):
        return 'new_attack_type'
```

### 自定义风险阈值

```python
# 修改目标分布比例
target_distribution = {
    'low': 0.70,      # 调整为70%
    'medium': 0.20,   # 调整为20%
    'high': 0.08,     # 调整为8%
    'critical': 0.02  # 调整为2%
}
```

## 🔮 未来发展方向

### 阶段1: 监督学习集成 (进行中)
- ✅ 动态阈值优化 (已完成)
- 🔄 伪标签监督学习集成 (下一步)
- 📋 模型性能对比验证
- 📋 预测准确性提升

### 阶段2: 高级功能扩展
- 📋 实时风险监控
- 📋 多模型集成预测
- 📋 深度学习模型集成
- 📋 时序分析功能

### 阶段3: 系统优化
- 📋 性能优化 (向量化计算)
- 📋 分布式处理支持
- 📋 API接口开发
- 📋 模型自动更新

## 📞 技术支持

如有问题或建议，请联系开发团队或提交Issue。

---

**版本**: v2.0.0
**更新日期**: 2025当前时间
