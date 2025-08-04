"""
风险评分计算器
基于真实数据集特征的风险评分系统
支持基于规则的评分、机器学习模型评分和无监督聚类评分
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os
import json
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .risk_calibrator import RiskCalibrator

# 配置日志 - 先配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_json_safe(obj):
    """
    将对象转换为JSON安全的格式
    处理numpy类型、pandas类型等不可序列化的对象
    """
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        # 对于其他类型，尝试转换为字符串
        try:
            return str(obj)
        except:
            return None

# 导入优化模块 - 安全导入
OPTIMIZATION_AVAILABLE = False
FeatureSelector = None
PerformanceOptimizer = None

try:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    sys.path.insert(0, backend_dir)

    from feature_engineering.feature_selector import FeatureSelector
    from optimization.performance_optimizer import PerformanceOptimizer
    OPTIMIZATION_AVAILABLE = True
    logger.info("优化模块导入成功")
except ImportError as e:
    logger.info(f"优化模块不可用，使用标准模式: {e}")
except Exception as e:
    logger.warning(f"优化模块导入异常: {e}")


class RiskCalculator:
    """基于真实数据集的风险评分计算器"""

    def __init__(self, models_dir: str = "models"):
        """
        初始化风险计算器

        Args:
            models_dir: 模型文件目录
        """
        self.models_dir = models_dir
        self.models = {}
        self.feature_info = None

        # 基于真实数据集的风险评分权重
        self.risk_weights = {
            'amount_risk': 0.25,      # 交易金额风险
            'time_risk': 0.20,        # 时间风险
            'device_risk': 0.15,      # 设备风险
            'account_risk': 0.15,     # 账户风险
            'payment_risk': 0.10,     # 支付方式风险
            'address_risk': 0.10,     # 地址一致性风险
            'behavior_risk': 0.05     # 行为模式风险
        }

        # 风险等级阈值
        self.risk_thresholds = {
            'low': 30,        # 0-30: 低风险
            'medium': 60,     # 31-60: 中风险
            'high': 80,       # 61-80: 高风险
            'critical': 100   # 81-100: 极高风险
        }

        self._load_models()

    def _load_models(self):
        """加载预训练模型（可选）"""
        try:
            model_files = {
                'catboost': 'catboost_model.pkl',
                'xgboost': 'xgboost_model.pkl',
                'random_forest': 'random_forest_model.pkl',
                'ensemble': 'ensemble_model.pkl'
            }

            for model_name, filename in model_files.items():
                model_path = os.path.join(self.models_dir, filename)
                if os.path.exists(model_path):
                    try:
                        self.models[model_name] = joblib.load(model_path)
                        logger.info(f"成功加载模型: {model_name}")
                    except Exception as e:
                        logger.warning(f"加载模型{model_name}失败: {e}")
                else:
                    logger.warning(f"模型文件不存在: {model_path}")

            if not self.models:
                logger.info("未加载任何模型，将仅使用基于规则的评分")

        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            self.models = {}
    
    def _load_models(self):
        """加载预训练模型"""
        try:
            # 加载各个模型
            model_files = {
                'catboost': 'catboost_model.pkl',
                'xgboost': 'xgboost_model.pkl',
                'random_forest': 'random_forest_model.pkl',
                'ensemble': 'ensemble_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = os.path.join(self.models_dir, filename)
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"成功加载模型: {model_name}")
                else:
                    logger.warning(f"模型文件不存在: {model_path}")
            
            # 加载特征信息
            feature_info_path = os.path.join(self.models_dir, 'feature_info.pkl')
            if os.path.exists(feature_info_path):
                self.feature_info = joblib.load(feature_info_path)
                logger.info("成功加载特征信息")
            
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
    
    def calculate_risk_score(self, data: pd.DataFrame, use_models: bool = False) -> Dict:
        """
        计算综合风险评分

        Args:
            data: 交易数据 (清理后格式)
            use_models: 是否使用机器学习模型

        Returns:
            包含风险评分和详细信息的字典
        """
        try:
            if data is None or data.empty:
                raise ValueError("输入数据为空")

            # 计算基于规则的风险评分
            rule_based_scores = self._calculate_rule_based_scores(data)

            # 如果使用模型，计算模型评分
            model_scores = {}
            if use_models and self.models:
                model_scores = self._calculate_model_scores(data)

            # 综合评分
            final_scores = self._combine_scores(rule_based_scores, model_scores)

            # 生成详细结果
            results = []
            for idx, row in data.iterrows():
                score = final_scores.get(idx, 0)
                risk_level = self._determine_risk_level(score)

                result = {
                    'transaction_id': row.get('transaction_id', f'tx_{idx}'),
                    'customer_id': row.get('customer_id', f'customer_{idx}'),
                    'risk_score': round(score, 2),
                    'risk_level': risk_level,
                    'risk_factors': self._analyze_risk_factors(row),
                    'score_breakdown': rule_based_scores.get(idx, {}),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)

            # 汇总统计
            scores = [r['risk_score'] for r in results]
            summary = {
                'total_transactions': len(results),
                'average_risk_score': round(np.mean(scores), 2),
                'risk_distribution': self._get_risk_distribution(results),
                'high_risk_count': len([r for r in results if r['risk_level'] in ['high', 'critical']]),
                'results': results
            }

            return summary

        except Exception as e:
            logger.error(f"计算风险评分时出错: {e}")
            return {
                'error': str(e),
                'total_transactions': 0,
                'average_risk_score': 0,
                'results': []
            }

    def _calculate_rule_based_scores(self, data: pd.DataFrame) -> Dict[int, Dict]:
        """基于规则计算风险评分"""
        scores = {}

        for idx, row in data.iterrows():
            score_breakdown = {}
            total_score = 0

            # 1. 交易金额风险 (0-25分)
            amount_score = self._calculate_amount_risk(row)
            score_breakdown['amount_risk'] = amount_score
            total_score += amount_score * self.risk_weights['amount_risk']

            # 2. 时间风险 (0-20分)
            time_score = self._calculate_time_risk(row)
            score_breakdown['time_risk'] = time_score
            total_score += time_score * self.risk_weights['time_risk']

            # 3. 设备风险 (0-15分)
            device_score = self._calculate_device_risk(row)
            score_breakdown['device_risk'] = device_score
            total_score += device_score * self.risk_weights['device_risk']

            # 4. 账户风险 (0-15分)
            account_score = self._calculate_account_risk(row)
            score_breakdown['account_risk'] = account_score
            total_score += account_score * self.risk_weights['account_risk']

            # 5. 支付方式风险 (0-10分)
            payment_score = self._calculate_payment_risk(row)
            score_breakdown['payment_risk'] = payment_score
            total_score += payment_score * self.risk_weights['payment_risk']

            # 6. 地址一致性风险 (0-10分)
            address_score = self._calculate_address_risk(row)
            score_breakdown['address_risk'] = address_score
            total_score += address_score * self.risk_weights['address_risk']

            # 7. 行为模式风险 (0-5分)
            behavior_score = self._calculate_behavior_risk(row)
            score_breakdown['behavior_risk'] = behavior_score
            total_score += behavior_score * self.risk_weights['behavior_risk']

            # 标准化到0-100分
            final_score = min(100, max(0, total_score * 100))
            score_breakdown['total_score'] = final_score

            scores[idx] = score_breakdown

        return scores

    def _calculate_amount_risk(self, row: pd.Series) -> float:
        """计算交易金额风险"""
        amount = row.get('transaction_amount', 0)
        quantity = row.get('quantity', 1)

        # 基础金额风险
        if amount <= 50:
            amount_risk = 0.1  # 小额交易低风险
        elif amount <= 200:
            amount_risk = 0.3  # 中等金额
        elif amount <= 500:
            amount_risk = 0.6  # 较大金额
        elif amount <= 1000:
            amount_risk = 0.8  # 大额交易
        else:
            amount_risk = 1.0  # 超大额交易高风险

        # 数量风险调整
        if quantity >= 4:
            amount_risk += 0.2  # 大量购买增加风险

        return min(1.0, amount_risk)

    def _calculate_time_risk(self, row: pd.Series) -> float:
        """计算时间风险"""
        hour = row.get('transaction_hour', 12)

        # 深夜和凌晨时段风险较高
        if 0 <= hour <= 5:
            return 1.0  # 凌晨高风险
        elif 22 <= hour <= 23:
            return 0.8  # 深夜较高风险
        elif 6 <= hour <= 8:
            return 0.4  # 早晨中等风险
        elif 9 <= hour <= 21:
            return 0.2  # 正常时段低风险
        else:
            return 0.5  # 其他时段中等风险

    def _calculate_device_risk(self, row: pd.Series) -> float:
        """计算设备风险"""
        device = row.get('device_used', 'unknown')

        device_risk_map = {
            'desktop': 0.2,   # 桌面设备风险较低
            'mobile': 0.5,    # 移动设备中等风险
            'tablet': 0.4,    # 平板设备中等风险
            'unknown': 1.0    # 未知设备高风险
        }

        return device_risk_map.get(device, 0.8)

    def _calculate_account_risk(self, row: pd.Series) -> float:
        """计算账户风险"""
        age_days = row.get('account_age_days', 365)
        customer_age = row.get('customer_age', 30)

        # 账户年龄风险
        if age_days < 7:
            account_risk = 1.0  # 新账户高风险
        elif age_days < 30:
            account_risk = 0.8  # 较新账户
        elif age_days < 90:
            account_risk = 0.4  # 中等账户
        else:
            account_risk = 0.2  # 老账户低风险

        # 客户年龄风险调整
        if customer_age < 18 or customer_age > 70:
            account_risk += 0.3  # 异常年龄增加风险

        return min(1.0, account_risk)

    def _calculate_payment_risk(self, row: pd.Series) -> float:
        """计算支付方式风险"""
        payment_method = row.get('payment_method', 'unknown')

        payment_risk_map = {
            'credit card': 0.2,    # 信用卡风险较低
            'debit card': 0.3,     # 借记卡风险较低
            'bank transfer': 0.6,  # 银行转账中等风险
            'PayPal': 0.4,         # PayPal中等风险
            'unknown': 1.0         # 未知支付方式高风险
        }

        return payment_risk_map.get(payment_method, 0.8)

    def _calculate_address_risk(self, row: pd.Series) -> float:
        """计算地址一致性风险"""
        shipping_addr = str(row.get('shipping_address', ''))
        billing_addr = str(row.get('billing_address', ''))

        # 地址完全一致
        if shipping_addr == billing_addr:
            return 0.1  # 地址一致风险很低

        # 简单的地址相似性检查
        shipping_words = set(shipping_addr.lower().split())
        billing_words = set(billing_addr.lower().split())

        if len(shipping_words) > 0 and len(billing_words) > 0:
            overlap = len(shipping_words & billing_words)
            total = len(shipping_words | billing_words)
            similarity = overlap / total if total > 0 else 0

            if similarity > 0.7:
                return 0.3  # 高相似性低风险
            elif similarity > 0.4:
                return 0.6  # 中等相似性中等风险
            else:
                return 1.0  # 低相似性高风险

        return 0.8  # 默认较高风险

    def _calculate_behavior_risk(self, row: pd.Series) -> float:
        """计算行为模式风险"""
        # 基于产品类别的风险
        category = row.get('product_category', 'unknown')

        category_risk_map = {
            'electronics': 0.8,      # 电子产品风险较高
            'clothing': 0.2,         # 服装风险较低
            'home & garden': 0.3,    # 家居园艺风险较低
            'health & beauty': 0.4,  # 健康美容中等风险
            'toys & games': 0.3,     # 玩具游戏风险较低
            'unknown': 0.6           # 未知类别中等风险
        }

        return category_risk_map.get(category, 0.5)

    def _determine_risk_level(self, score: float) -> str:
        """确定风险等级"""
        if score >= self.risk_thresholds['critical']:
            return 'critical'
        elif score >= self.risk_thresholds['high']:
            return 'high'
        elif score >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'

    def _analyze_risk_factors(self, row: pd.Series) -> List[str]:
        """分析风险因素"""
        factors = []

        # 检查各种风险因素
        if row.get('transaction_amount', 0) > 1000:
            factors.append('大额交易')

        hour = row.get('transaction_hour', 12)
        if 0 <= hour <= 5:
            factors.append('凌晨交易')
        elif 22 <= hour <= 23:
            factors.append('深夜交易')

        if row.get('account_age_days', 365) < 30:
            factors.append('新账户')

        if row.get('device_used') == 'mobile':
            factors.append('移动设备')

        if row.get('payment_method') in ['bank transfer']:
            factors.append('高风险支付方式')

        if row.get('product_category') == 'electronics':
            factors.append('高风险商品类别')

        return factors

    def _combine_scores(self, rule_scores: Dict, model_scores: Dict) -> Dict[int, float]:
        """合并规则评分和模型评分"""
        combined_scores = {}

        for idx in rule_scores.keys():
            rule_score = rule_scores[idx]['total_score']

            if model_scores and idx in model_scores:
                model_score = model_scores[idx] * 100  # 模型分数转换为0-100
                # 加权平均：规则评分70%，模型评分30%
                final_score = rule_score * 0.7 + model_score * 0.3
            else:
                final_score = rule_score

            combined_scores[idx] = final_score

        return combined_scores

    def _get_risk_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """获取风险等级分布"""
        distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}

        for result in results:
            risk_level = result['risk_level']
            if risk_level in distribution:
                distribution[risk_level] += 1

        return distribution

    def _calculate_model_scores(self, data: pd.DataFrame) -> Dict[int, float]:
        """使用机器学习模型计算评分（占位符）"""
        # 这里可以集成预训练的机器学习模型
        # 目前返回空字典，表示不使用模型评分
        return {}

    def update_risk_weights(self, new_weights: Dict[str, float]):
        """更新风险权重"""
        # 验证权重总和
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"权重总和应为1.0，当前为{total_weight}")

        self.risk_weights.update(new_weights)
        logger.info(f"风险权重已更新: {self.risk_weights}")

    def get_risk_statistics(self, results: List[Dict]) -> Dict:
        """获取风险统计信息"""
        if not results:
            return {}

        scores = [r['risk_score'] for r in results]

        return {
            'total_count': len(results),
            'average_score': round(np.mean(scores), 2),
            'median_score': round(np.median(scores), 2),
            'std_score': round(np.std(scores), 2),
            'min_score': round(min(scores), 2),
            'max_score': round(max(scores), 2),
            'risk_distribution': self._get_risk_distribution(results),
            'high_risk_percentage': round(
                len([r for r in results if r['risk_level'] in ['high', 'critical']]) / len(results) * 100, 2
            )
        }
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        try:
            # 确保数据包含必要的特征
            if self.feature_info is None:
                return data
            
            required_features = self.feature_info.get('feature_names', [])
            missing_features = [col for col in required_features if col not in data.columns]
            
            if missing_features:
                # 为缺失特征添加默认值
                for feature in missing_features:
                    data[feature] = 0
            
            # 选择必要的特征
            processed_data = data[required_features].copy()
            
            # 处理缺失值
            processed_data = processed_data.fillna(0)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"数据预处理时出错: {e}")
            return data
    
    def _get_model_predictions(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get prediction results from each model"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    # Get fraud probability
                    proba = model.predict_proba(data)
                    if proba.shape[1] > 1:
                        fraud_prob = proba[:, 1]  # Assume second column is fraud probability
                    else:
                        fraud_prob = proba[:, 0]
                else:
                    # Direct prediction
                    fraud_prob = model.predict(data)

                # Ensure scalar value
                if isinstance(fraud_prob, np.ndarray):
                    fraud_prob = fraud_prob[0] if len(fraud_prob) == 1 else np.mean(fraud_prob)

                predictions[model_name] = float(fraud_prob)

            except Exception as e:
                logger.error(f"Model {model_name} prediction error: {e}")
                predictions[model_name] = 0.5  # Default value
        
        return predictions
    
    def _calculate_weighted_score(self, predictions: Dict[str, float]) -> float:
        """计算加权风险评分"""
        weighted_score = 0.0
        total_weight = 0.0
        
        for model_name, prediction in predictions.items():
            weight = self.model_weights.get(model_name, 0.1)
            weighted_score += prediction * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return 0.5
    

    
    def _calculate_confidence(self, predictions: Dict[str, float]) -> float:
        """计算预测置信度"""
        if not predictions:
            return 0.0
        
        # 基于预测一致性计算置信度
        values = list(predictions.values())
        std = np.std(values)
        mean = np.mean(values)
        
        # 标准差越小，置信度越高
        confidence = max(0.0, 1.0 - std / (mean + 1e-8))
        return min(1.0, confidence)
    

    
    def batch_calculate_risk(self, transactions: pd.DataFrame) -> List[Dict]:
        """
        批量计算风险评分
        
        Args:
            transactions: 交易数据DataFrame
            
        Returns:
            风险评分列表
        """
        results = []
        
        for idx, row in transactions.iterrows():
            single_transaction = pd.DataFrame([row])
            risk_result = self.calculate_risk_score(single_transaction)
            risk_result['transaction_id'] = idx
            results.append(risk_result)
        
        return results
    
    def update_model_weights(self, new_weights: Dict[str, float]):
        """更新模型权重"""
        self.model_weights.update(new_weights)
        logger.info(f"更新模型权重: {new_weights}")
    
    def get_model_performance(self) -> Dict:
        """获取模型性能信息"""
        return {
            'available_models': list(self.models.keys()),
            'model_weights': self.model_weights,
            'feature_info': self.feature_info is not None
        }


class UnsupervisedRiskCalculator:
    """
    无监督风险评分计算器
    基于聚类结果和特征分布进行风险评分，不依赖真实标签
    """

    def __init__(self, enable_optimization: bool = True, enable_dynamic_thresholds: bool = True):
        """
        初始化无监督风险计算器

        Args:
            enable_optimization: 是否启用性能优化（默认开启）
            enable_dynamic_thresholds: 是否启用动态阈值（默认开启）
        """
        self.cluster_risk_weights = {
            'cluster_anomaly_score': 0.25,      # 聚类异常度
            'feature_deviation_score': 0.30,    # 特征偏离度
            'business_rule_score': 0.25,        # 业务规则评分
            'statistical_outlier_score': 0.15,  # 统计异常值
            'pattern_consistency_score': 0.05   # 模式一致性
        }

        # 默认风险等级阈值
        self.default_risk_thresholds = {
            'low': 40,        # 0-40: 低风险 (约60-70%)
            'medium': 60,     # 41-60: 中风险 (约20-25%)
            'high': 80,       # 61-80: 高风险 (约10-15%)
            'critical': 100   # 81-100: 极高风险 (约5-10%)
        }

        # 当前使用的阈值（可能是动态调整的）
        self.risk_thresholds = self.default_risk_thresholds.copy()

        # 动态阈值管理
        self.enable_dynamic_thresholds = enable_dynamic_thresholds
        if enable_dynamic_thresholds:
            try:
                from .dynamic_threshold_manager import DynamicThresholdManager
                self.threshold_manager = DynamicThresholdManager()
                logger.info("动态阈值管理已启用")
            except ImportError:
                logger.warning("动态阈值管理器不可用，使用固定阈值")
                self.threshold_manager = None
                self.enable_dynamic_thresholds = False
        else:
            self.threshold_manager = None

        # 聚类风险映射缓存
        self.cluster_risk_mapping = {}
        self.calibration_results = None
        self.calibrator = RiskCalibrator()  # 校准器

        # 性能优化组件 - 安全初始化
        self.enable_optimization = False
        self.feature_selector = None
        self.performance_optimizer = None

        if enable_optimization and OPTIMIZATION_AVAILABLE:
            try:
                self.feature_selector = FeatureSelector(target_features=18)
                self.performance_optimizer = PerformanceOptimizer()
                self.enable_optimization = True
                logger.info("性能优化已启用")
            except Exception as e:
                logger.warning(f"性能优化初始化失败，使用标准模式: {e}")
                self.enable_optimization = False

        # 优化配置
        self.optimization_config = {
            'use_feature_selection': self.enable_optimization,
            'use_sampling': self.enable_optimization,
            'use_vectorization': self.enable_optimization,
            'large_dataset_threshold': 10000
        }

    def calculate_unsupervised_risk_score(self, data: pd.DataFrame,
                                        cluster_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算无监督风险评分 - 优化版本

        Args:
            data: 输入数据
            cluster_results: 聚类分析结果

        Returns:
            无监督风险评分结果
        """
        try:
            if data is None or data.empty:
                logger.error("输入数据为空")
                return self._empty_result()

            if not cluster_results or 'cluster_labels' not in cluster_results:
                logger.error("聚类结果无效")
                return self._empty_result()

            start_time = datetime.now()
            logger.info(f"开始计算无监督风险评分，数据量: {len(data)}")

            # 性能优化：特征选择
            optimized_data = self._optimize_features(data)

            # 1. 计算聚类到风险等级的映射
            cluster_risk_mapping = self._map_clusters_to_risk_levels(optimized_data, cluster_results)

            # 2. 计算风险评分 - 安全模式
            try:
                if (self.enable_optimization and
                    len(data) > self.optimization_config.get('large_dataset_threshold', 10000) and
                    self.performance_optimizer is not None):
                    logger.info("尝试使用优化模式计算风险评分")
                    risk_scores, risk_details = self._optimized_risk_calculation(
                        optimized_data, cluster_results, cluster_risk_mapping
                    )
                else:
                    logger.info("使用标准模式计算风险评分")
                    risk_scores, risk_details = self._standard_risk_calculation(
                        optimized_data, cluster_results, cluster_risk_mapping
                    )
            except Exception as e:
                logger.warning(f"优化计算失败，回退到标准模式: {e}")
                risk_scores, risk_details = self._standard_risk_calculation(
                    optimized_data, cluster_results, cluster_risk_mapping
                )

            # 3. 应用动态阈值优化
            if self.enable_dynamic_thresholds and self.threshold_manager:
                try:
                    # 计算动态阈值
                    dynamic_thresholds = self.threshold_manager.optimize_thresholds_iteratively(risk_scores)

                    # 更新风险等级
                    self._update_risk_levels_with_dynamic_thresholds(risk_details, dynamic_thresholds)

                    # 分析分布质量
                    distribution_analysis = self.threshold_manager.analyze_distribution(risk_scores, dynamic_thresholds)
                    logger.info(f"动态阈值应用完成，分布偏差: {distribution_analysis['total_deviation']:.3f}")

                except Exception as e:
                    logger.warning(f"动态阈值应用失败: {e}")
                    dynamic_thresholds = self.default_risk_thresholds
                    distribution_analysis = {}
            else:
                dynamic_thresholds = self.risk_thresholds
                distribution_analysis = {}

            # 4. 生成汇总统计
            summary = self._generate_risk_summary(risk_scores, risk_details, cluster_risk_mapping,
                                                dynamic_thresholds, distribution_analysis)

            # 添加性能统计
            computation_time = (datetime.now() - start_time).total_seconds()
            summary['performance_stats'] = {
                'computation_time': computation_time,
                'data_size': len(data),
                'optimized_features': len(optimized_data.columns) if optimized_data is not data else len(data.columns),
                'optimization_enabled': self.enable_optimization,
                'processing_mode': 'optimized' if self.enable_optimization and len(data) > self.optimization_config['large_dataset_threshold'] else 'standard'
            }

            logger.info(f"无监督风险评分完成，处理了 {len(risk_details)} 个样本，耗时: {computation_time:.2f}秒")
            return summary

        except Exception as e:
            logger.error(f"计算无监督风险评分时出错: {e}")
            return self._empty_result()

    def _update_risk_levels_with_dynamic_thresholds(self, risk_details: List[Dict],
                                                   dynamic_thresholds: Dict[str, float]) -> None:
        """使用动态阈值更新风险等级"""
        try:
            for detail in risk_details:
                risk_score = detail.get('risk_score', 0)

                # 根据动态阈值重新确定风险等级
                if risk_score <= dynamic_thresholds['low']:
                    new_level = 'low'
                elif risk_score <= dynamic_thresholds['medium']:
                    new_level = 'medium'
                elif risk_score <= dynamic_thresholds['high']:
                    new_level = 'high'
                else:
                    new_level = 'critical'

                # 更新风险等级
                old_level = detail.get('risk_level', 'unknown')
                detail['risk_level'] = new_level
                detail['dynamic_threshold_applied'] = True
                detail['threshold_change'] = old_level != new_level

        except Exception as e:
            logger.warning(f"动态阈值更新失败: {e}")

    def _optimize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """优化特征选择 - 安全版本"""
        try:
            # 如果优化未启用，直接返回原数据
            if not self.enable_optimization:
                return data

            if not self.optimization_config.get('use_feature_selection', False):
                return data

            if self.feature_selector is None:
                return data

            # 检查数据有效性
            if data is None or data.empty:
                return data

            # 选择重要特征
            selected_features = self.feature_selector.select_features(data, method='hybrid')

            if selected_features and len(selected_features) < len(data.columns):
                # 确保选择的特征都存在于数据中
                valid_features = [f for f in selected_features if f in data.columns]
                if valid_features:
                    optimized_data = data[valid_features]
                    logger.info(f"特征优化: {len(data.columns)} -> {len(valid_features)}")
                    return optimized_data

            return data

        except Exception as e:
            logger.warning(f"特征优化失败，使用原始数据: {e}")
            return data

    def _optimized_risk_calculation(self, data: pd.DataFrame,
                                   cluster_results: Dict[str, Any],
                                   cluster_risk_mapping: Dict) -> Tuple[List[float], List[Dict]]:
        """优化的风险评分计算 - 安全版本"""
        try:
            # 如果优化组件不可用，直接使用标准计算
            if not self.enable_optimization or self.performance_optimizer is None:
                return self._standard_risk_calculation(data, cluster_results, cluster_risk_mapping)

            risk_scores = []
            risk_details = []
            cluster_labels = cluster_results.get('cluster_labels', [])

            if len(cluster_labels) == 0:
                logger.warning("聚类标签为空，使用标准计算")
                return self._standard_risk_calculation(data, cluster_results, cluster_risk_mapping)

            # 批量处理
            batch_size = 1000
            for i in range(0, len(data), batch_size):
                try:
                    batch_data = data.iloc[i:i+batch_size]
                    end_idx = min(i+batch_size, len(cluster_labels))
                    batch_labels = cluster_labels[i:end_idx]

                    # 批量计算风险评分
                    batch_scores, batch_details = self._calculate_batch_risk_scores(
                        batch_data, batch_labels, cluster_risk_mapping, data, cluster_results, i
                    )

                    risk_scores.extend(batch_scores)
                    risk_details.extend(batch_details)

                except Exception as batch_e:
                    logger.warning(f"批次 {i} 计算失败，使用标准方法: {batch_e}")
                    # 对这个批次使用标准计算
                    batch_data = data.iloc[i:i+batch_size]
                    for j, (idx, row) in enumerate(batch_data.iterrows()):
                        cluster_id = cluster_labels[i+j] if i+j < len(cluster_labels) else -1
                        score_breakdown = self._calculate_individual_risk_scores(
                            row, cluster_id, cluster_risk_mapping, data, cluster_results
                        )
                        total_score = sum(
                            score_breakdown[key] * weight
                            for key, weight in self.cluster_risk_weights.items()
                            if key in score_breakdown
                        )

                        # 风险放大机制：暂时禁用以避免无限循环
                        # total_score = self._apply_risk_amplification(total_score, score_breakdown)

                        risk_level = self._determine_risk_level(total_score)

                        risk_detail = {
                            'transaction_id': str(row.get('transaction_id', f'tx_{i+j}')),
                            'customer_id': str(row.get('customer_id', f'customer_{i+j}')),
                            'cluster_id': int(cluster_id),
                            'risk_score': float(round(total_score, 2)),
                            'risk_level': str(risk_level),
                            'score_breakdown': make_json_safe(score_breakdown),
                            'risk_factors': self._identify_risk_factors(row, cluster_id, cluster_risk_mapping),
                            'timestamp': datetime.now().isoformat()
                        }

                        risk_scores.append(float(total_score))
                        risk_details.append(make_json_safe(risk_detail))

            return risk_scores, risk_details

        except Exception as e:
            logger.error(f"优化风险计算失败，回退到标准计算: {e}")
            # 回退到标准计算
            return self._standard_risk_calculation(data, cluster_results, cluster_risk_mapping)

    def _standard_risk_calculation(self, data: pd.DataFrame,
                                 cluster_results: Dict[str, Any],
                                 cluster_risk_mapping: Dict) -> Tuple[List[float], List[Dict]]:
        """标准的风险评分计算"""
        risk_scores = []
        risk_details = []
        cluster_labels = cluster_results['cluster_labels']

        for idx, row in data.iterrows():
            cluster_id = cluster_labels[idx] if idx < len(cluster_labels) else -1

            # 计算各维度风险分数
            score_breakdown = self._calculate_individual_risk_scores(
                row, cluster_id, cluster_risk_mapping, data, cluster_results
            )

            # 综合风险评分
            total_score = sum(
                score_breakdown[key] * weight
                for key, weight in self.cluster_risk_weights.items()
                if key in score_breakdown
            )

            # 风险放大机制：暂时禁用
            # total_score = self._apply_risk_amplification(total_score, score_breakdown)

            risk_level = self._determine_risk_level(total_score)

            risk_detail = {
                'transaction_id': str(row.get('transaction_id', f'tx_{idx}')),
                'customer_id': str(row.get('customer_id', f'customer_{idx}')),
                'cluster_id': int(cluster_id),
                'risk_score': float(round(total_score, 2)),
                'risk_level': str(risk_level),
                'score_breakdown': make_json_safe(score_breakdown),
                'risk_factors': self._identify_risk_factors(row, cluster_id, cluster_risk_mapping),
                'timestamp': datetime.now().isoformat()
            }

            risk_scores.append(float(total_score))
            risk_details.append(make_json_safe(risk_detail))

        return risk_scores, risk_details

    def _calculate_batch_risk_scores(self, batch_data: pd.DataFrame,
                                   batch_labels: np.ndarray,
                                   cluster_risk_mapping: Dict,
                                   full_data: pd.DataFrame,
                                   cluster_results: Dict,
                                   start_idx: int) -> Tuple[List[float], List[Dict]]:
        """批量计算风险评分"""
        batch_scores = []
        batch_details = []

        for i, (idx, row) in enumerate(batch_data.iterrows()):
            cluster_id = batch_labels[i] if i < len(batch_labels) else -1

            # 计算各维度风险分数
            score_breakdown = self._calculate_individual_risk_scores(
                row, cluster_id, cluster_risk_mapping, full_data, cluster_results
            )

            # 综合风险评分
            total_score = sum(
                score_breakdown[key] * weight
                for key, weight in self.cluster_risk_weights.items()
                if key in score_breakdown
            )

            # 风险放大机制：暂时禁用
            # total_score = self._apply_risk_amplification(total_score, score_breakdown)

            risk_level = self._determine_risk_level(total_score)

            risk_detail = {
                'transaction_id': str(row.get('transaction_id', f'tx_{start_idx + i}')),
                'customer_id': str(row.get('customer_id', f'customer_{start_idx + i}')),
                'cluster_id': int(cluster_id),
                'risk_score': float(round(total_score, 2)),
                'risk_level': str(risk_level),
                'score_breakdown': make_json_safe(score_breakdown),
                'risk_factors': self._identify_risk_factors(row, cluster_id, cluster_risk_mapping),
                'timestamp': datetime.now().isoformat()
            }

            batch_scores.append(float(total_score))
            batch_details.append(make_json_safe(risk_detail))

        return batch_scores, batch_details

    def _map_clusters_to_risk_levels(self, data: pd.DataFrame,
                                   cluster_results: Dict[str, Any]) -> Dict[int, Dict]:
        """将聚类结果映射到风险等级"""
        cluster_risk_mapping = {}
        cluster_details = cluster_results.get('cluster_details', [])

        for cluster_detail in cluster_details:
            cluster_id = cluster_detail.get('cluster_id', -1)

            # 计算聚类风险指标
            risk_indicators = self._calculate_cluster_risk_indicators(cluster_detail, data)

            # 基于风险指标计算聚类风险等级
            cluster_risk_score = self._calculate_cluster_risk_score(risk_indicators)

            cluster_risk_mapping[cluster_id] = {
                'risk_score': cluster_risk_score,
                'risk_level': self._determine_risk_level(cluster_risk_score),
                'risk_indicators': risk_indicators,
                'size': cluster_detail.get('size', 0),
                'percentage': cluster_detail.get('percentage', 0)
            }

        self.cluster_risk_mapping = cluster_risk_mapping
        return cluster_risk_mapping

    def _calculate_cluster_risk_indicators(self, cluster_detail: Dict, data: pd.DataFrame) -> Dict:
        """计算聚类风险指标"""
        indicators = {}

        # 基础统计指标
        indicators['avg_transaction_amount'] = cluster_detail.get('avg_transaction_amount', 0)
        indicators['transaction_amount_std'] = cluster_detail.get('transaction_amount_std', 0)

        # 时间模式风险
        indicators['night_transaction_rate'] = cluster_detail.get('night_transaction_rate', 0)
        indicators['common_hour'] = cluster_detail.get('common_hour', 12)

        # 账户特征
        indicators['avg_customer_age'] = cluster_detail.get('avg_customer_age', 35)
        indicators['avg_account_age_days'] = cluster_detail.get('avg_account_age_days', 365)

        # 设备和支付方式
        indicators['mobile_device_rate'] = cluster_detail.get('mobile_device_rate', 0)
        indicators['bank_transfer_rate'] = cluster_detail.get('bank_transfer_rate', 0)

        # 地理和地址
        indicators['address_mismatch_rate'] = cluster_detail.get('address_mismatch_rate', 0)

        # 商品类别风险
        indicators['electronics_rate'] = cluster_detail.get('electronics_rate', 0)

        return indicators

    def _calculate_cluster_risk_score(self, risk_indicators: Dict) -> float:
        """基于风险指标计算聚类风险分数"""
        risk_score = 0.0

        # 1. 交易金额风险 (0-25分)
        avg_amount = risk_indicators.get('avg_transaction_amount', 0)
        if avg_amount > 1000:
            risk_score += 25
        elif avg_amount > 500:
            risk_score += 15
        elif avg_amount > 200:
            risk_score += 8

        # 2. 时间模式风险 (0-20分)
        night_rate = risk_indicators.get('night_transaction_rate', 0)
        if night_rate > 0.3:
            risk_score += 20
        elif night_rate > 0.15:
            risk_score += 12
        elif night_rate > 0.05:
            risk_score += 6

        # 3. 账户风险 (0-15分)
        avg_account_age = risk_indicators.get('avg_account_age_days', 365)
        if avg_account_age < 30:
            risk_score += 15
        elif avg_account_age < 90:
            risk_score += 10
        elif avg_account_age < 180:
            risk_score += 5

        # 4. 设备和支付风险 (0-15分)
        mobile_rate = risk_indicators.get('mobile_device_rate', 0)
        bank_transfer_rate = risk_indicators.get('bank_transfer_rate', 0)
        if mobile_rate > 0.8 and bank_transfer_rate > 0.5:
            risk_score += 15
        elif mobile_rate > 0.6 or bank_transfer_rate > 0.3:
            risk_score += 8

        # 5. 地址一致性风险 (0-10分)
        address_mismatch_rate = risk_indicators.get('address_mismatch_rate', 0)
        if address_mismatch_rate > 0.5:
            risk_score += 10
        elif address_mismatch_rate > 0.2:
            risk_score += 6

        # 6. 商品类别风险 (0-10分)
        electronics_rate = risk_indicators.get('electronics_rate', 0)
        if electronics_rate > 0.7:
            risk_score += 10
        elif electronics_rate > 0.4:
            risk_score += 5

        return min(100, risk_score)

    def _calculate_individual_risk_scores(self, row: pd.Series, cluster_id: int,
                                        cluster_risk_mapping: Dict, data: pd.DataFrame,
                                        cluster_results: Dict) -> Dict:
        """计算个体样本的各维度风险分数"""
        score_breakdown = {}

        # 1. 聚类异常度评分 (0-100)
        cluster_info = cluster_risk_mapping.get(cluster_id, {})
        cluster_risk_score = cluster_info.get('risk_score', 50)
        score_breakdown['cluster_anomaly_score'] = cluster_risk_score

        # 2. 特征偏离度评分 (0-100)
        deviation_score = self._calculate_feature_deviation_score(row, cluster_id, data)
        score_breakdown['feature_deviation_score'] = deviation_score

        # 3. 业务规则评分 (0-100)
        rule_score = self._calculate_business_rule_score(row)
        score_breakdown['business_rule_score'] = rule_score

        # 4. 统计异常值评分 (0-100)
        outlier_score = self._calculate_statistical_outlier_score(row, data)
        score_breakdown['statistical_outlier_score'] = outlier_score

        # 5. 模式一致性评分 (0-100)
        consistency_score = self._calculate_pattern_consistency_score(row, cluster_id, cluster_results)
        score_breakdown['pattern_consistency_score'] = consistency_score

        # 确保所有值都是JSON安全的
        return make_json_safe(score_breakdown)

    def _calculate_feature_deviation_score(self, row: pd.Series, cluster_id: int,
                                         data: pd.DataFrame) -> float:
        """计算特征偏离度评分"""
        if cluster_id == -1:  # 噪声点
            return 90.0  # 提高噪声点的风险评分

        # 获取同聚类的数据 - 修复关键bug
        try:
            # 假设聚类标签存储在data中，需要从cluster_results获取
            # 这里先使用全局统计作为临时解决方案
            cluster_data = data

            deviation_score = 0.0
            feature_count = 0

            # 检查数值特征的偏离程度
            numeric_features = ['transaction_amount', 'customer_age', 'account_age_days', 'transaction_hour']

            for feature in numeric_features:
                if feature in row.index and feature in cluster_data.columns:
                    feature_value = row[feature]

                    # 使用全局统计计算偏离度
                    global_mean = cluster_data[feature].mean()
                    global_std = cluster_data[feature].std()

                    # 使用百分位数方法计算偏离度
                    q25 = cluster_data[feature].quantile(0.25)
                    q75 = cluster_data[feature].quantile(0.75)
                    iqr = q75 - q25

                    # 计算偏离度评分
                    if global_std > 0:
                        z_score = abs((feature_value - global_mean) / global_std)
                        # 调整评分策略，使分布更合理且更敏感
                        if z_score > 4:  # 极端异常
                            feature_deviation = 100
                        elif z_score > 3:  # 很高异常
                            feature_deviation = 85
                        elif z_score > 2:  # 高度异常
                            feature_deviation = 70
                        elif z_score > 1.5:  # 中度异常
                            feature_deviation = 55
                        elif z_score > 1:  # 轻度异常
                            feature_deviation = 35
                        else:  # 正常范围
                            feature_deviation = 10
                    else:
                        feature_deviation = 10  # 标准差为0时给低分

                    # 使用IQR方法补充评分
                    if feature_value < q25 - 1.5 * iqr or feature_value > q75 + 1.5 * iqr:
                        feature_deviation = max(feature_deviation, 60)

                    deviation_score += feature_deviation
                    feature_count += 1

            return float(deviation_score / max(1, feature_count))

        except Exception as e:
            logger.warning(f"计算特征偏离度时出错: {e}")
            return 30.0  # 返回中等风险评分

    def _calculate_business_rule_score(self, row: pd.Series) -> float:
        """计算业务规则评分 - 优化版本"""
        rule_score = 5.0  # 降低基础分数，确保低风险占大多数

        amount = row.get('transaction_amount', 0)
        hour = row.get('transaction_hour', 12)
        account_age = row.get('account_age_days', 365)
        age = row.get('customer_age', 30)
        device = row.get('device_used', '')
        payment = row.get('payment_method', '')

        # 规则1: 交易金额风险（渐进式评分，确保合理分布）
        if amount > 50000:  # 极高金额
            rule_score += 50
        elif amount > 20000:  # 很高金额
            rule_score += 35
        elif amount > 10000:  # 高金额
            rule_score += 25
        elif amount > 5000:  # 中高金额
            rule_score += 18
        elif amount > 2000:  # 中等金额
            rule_score += 12
        elif amount > 1000:  # 较高金额
            rule_score += 8
        elif amount > 500:   # 小额
            rule_score += 4

        # 规则2: 时间风险（适度评分）
        if hour <= 2 or hour >= 23:  # 深夜/凌晨
            rule_score += 20
        elif hour <= 5 or hour >= 22:  # 夜间
            rule_score += 12
        elif 9 <= hour <= 17:  # 工作时间
            rule_score += 2  # 轻微加分
        else:  # 其他时间
            rule_score += 6

        # 规则3: 账户年龄风险（适度评分）
        if account_age < 3:  # 极新账户
            rule_score += 25
        elif account_age < 14:  # 新账户
            rule_score += 15
        elif account_age < 60:  # 较新账户
            rule_score += 8
        else:  # 老账户
            rule_score += 2

        # 规则4: 客户年龄风险
        if age < 18 or age > 75:  # 极端年龄
            rule_score += 20
        elif age < 21 or age > 65:  # 边缘年龄
            rule_score += 10
        else:  # 正常年龄
            rule_score += 5

        # 规则5: 设备和支付方式组合风险
        if device == 'mobile':
            rule_score += 8  # 移动设备有一定风险
            if payment == 'bank transfer':
                rule_score += 15  # 移动+银行转账
            elif payment == 'digital wallet':
                rule_score += 10  # 移动+数字钱包
        elif device == 'desktop':
            rule_score += 3  # 桌面设备风险较低
        else:  # unknown设备
            rule_score += 20

        # 规则6: 支付方式风险
        if payment == 'bank transfer':
            rule_score += 12
        elif payment == 'digital wallet':
            rule_score += 8
        elif payment == 'credit card':
            rule_score += 5
        else:  # unknown支付方式
            rule_score += 15

        # 规则7: 地址一致性
        shipping = str(row.get('shipping_address', ''))
        billing = str(row.get('billing_address', ''))
        if shipping != billing:
            if amount > 500:
                rule_score += 20
            elif amount > 200:
                rule_score += 12
            else:
                rule_score += 8

        # 规则8: 组合风险（高风险组合）
        if amount > 1000 and account_age < 30 and (hour <= 5 or hour >= 22):
            rule_score += 25  # 大额+新账户+夜间

        if device == 'mobile' and payment == 'bank transfer' and amount > 500:
            rule_score += 20  # 移动+银行转账+大额

        return min(100, rule_score)

    def _calculate_statistical_outlier_score(self, row: pd.Series, data: pd.DataFrame) -> float:
        """计算统计异常值评分 - 优化版本"""
        outlier_score = 5.0  # 基础分数
        feature_count = 0

        numeric_features = ['transaction_amount', 'customer_age', 'account_age_days', 'transaction_hour']

        for feature in numeric_features:
            if feature in row.index and feature in data.columns:
                feature_value = row[feature]

                # 计算百分位数
                q1 = data[feature].quantile(0.25)
                q3 = data[feature].quantile(0.75)
                q5 = data[feature].quantile(0.05)
                q95 = data[feature].quantile(0.95)
                iqr = q3 - q1

                # 计算Z-score
                mean_val = data[feature].mean()
                std_val = data[feature].std()

                feature_outlier_score = 0

                if std_val > 0:
                    z_score = abs((feature_value - mean_val) / std_val)

                    # 基于Z-score的评分
                    if z_score > 3:  # 极端异常
                        feature_outlier_score += 35
                    elif z_score > 2.5:  # 高度异常
                        feature_outlier_score += 25
                    elif z_score > 2:  # 中度异常
                        feature_outlier_score += 18
                    elif z_score > 1.5:  # 轻度异常
                        feature_outlier_score += 12
                    elif z_score > 1:  # 边缘异常
                        feature_outlier_score += 6
                    else:  # 正常范围
                        feature_outlier_score += 2

                # 基于IQR的评分
                if feature_value < q1 - 3 * iqr or feature_value > q3 + 3 * iqr:
                    feature_outlier_score = max(feature_outlier_score, 40)
                elif feature_value < q1 - 1.5 * iqr or feature_value > q3 + 1.5 * iqr:
                    feature_outlier_score = max(feature_outlier_score, 25)
                elif feature_value < q5 or feature_value > q95:
                    feature_outlier_score = max(feature_outlier_score, 15)

                # 特征特定的异常检测
                if feature == 'transaction_amount':
                    if feature_value > 5000:  # 超大额交易
                        feature_outlier_score = max(feature_outlier_score, 30)
                    elif feature_value > 2000:  # 大额交易
                        feature_outlier_score = max(feature_outlier_score, 20)
                elif feature == 'customer_age':
                    if feature_value < 16 or feature_value > 80:  # 极端年龄
                        feature_outlier_score = max(feature_outlier_score, 25)
                elif feature == 'account_age_days':
                    if feature_value < 1:  # 当天开户
                        feature_outlier_score = max(feature_outlier_score, 35)
                    elif feature_value < 7:  # 一周内开户
                        feature_outlier_score = max(feature_outlier_score, 20)
                elif feature == 'transaction_hour':
                    if feature_value <= 3 or feature_value >= 23:  # 深夜交易
                        feature_outlier_score = max(feature_outlier_score, 20)

                outlier_score += feature_outlier_score
                feature_count += 1

        return float(min(100, outlier_score / max(1, feature_count)) if feature_count > 0 else 10)

    def _calculate_pattern_consistency_score(self, row: pd.Series, cluster_id: int,
                                           cluster_results: Dict) -> float:
        """计算模式一致性评分 - 优化版本"""
        if cluster_id == -1:
            return 75.0  # 噪声点给予较高风险

        # 基于聚类质量指标
        quality_metrics = cluster_results.get('quality_metrics', {})
        silhouette_score = quality_metrics.get('silhouette_score', 0)

        # 获取聚类详情
        cluster_details = cluster_results.get('cluster_details', [])
        cluster_info = None
        for detail in cluster_details:
            if detail.get('cluster_id') == cluster_id:
                cluster_info = detail
                break

        base_score = 15.0  # 基础分数

        # 基于轮廓系数的评分（调整评分策略）
        if silhouette_score < 0.1:  # 很差的聚类质量
            silhouette_risk = 50
        elif silhouette_score < 0.2:  # 较差的聚类质量
            silhouette_risk = 35
        elif silhouette_score < 0.3:  # 一般的聚类质量
            silhouette_risk = 25
        elif silhouette_score < 0.5:  # 较好的聚类质量
            silhouette_risk = 15
        else:  # 很好的聚类质量
            silhouette_risk = 10

        # 基于聚类大小的评分
        size_risk = 0
        if cluster_info:
            cluster_size = cluster_info.get('size', 0)
            cluster_percentage = cluster_info.get('percentage', 0)

            # 过小的聚类可能是异常
            if cluster_percentage < 2:
                size_risk += 25
            elif cluster_percentage < 5:
                size_risk += 15
            elif cluster_percentage < 10:
                size_risk += 8

            # 过大的聚类可能包含异常
            if cluster_percentage > 60:
                size_risk += 15
            elif cluster_percentage > 40:
                size_risk += 8

        # 基于聚类内部一致性的评分
        consistency_risk = 0
        if cluster_info:
            # 检查聚类内部的变异性
            amount_std = cluster_info.get('transaction_amount_std', 0)
            avg_amount = cluster_info.get('avg_transaction_amount', 1)

            if avg_amount > 0:
                cv = amount_std / avg_amount  # 变异系数
                if cv > 1.5:  # 高变异性
                    consistency_risk += 20
                elif cv > 1.0:  # 中等变异性
                    consistency_risk += 12
                elif cv > 0.5:  # 低变异性
                    consistency_risk += 5

        total_score = base_score + silhouette_risk + size_risk + consistency_risk
        return float(min(100, total_score))

    def _apply_risk_amplification(self, base_score: float, score_breakdown: Dict) -> float:
        """
        风险放大机制：当多个高风险因素同时出现时，给予额外的风险加分
        """
        amplification_bonus = 0.0

        # 定义高风险阈值
        high_risk_thresholds = {
            'cluster_anomaly_score': 20,      # 聚类异常度 > 20
            'feature_deviation_score': 70,   # 特征偏离度 > 70
            'business_rule_score': 80,       # 业务规则评分 > 80
            'statistical_outlier_score': 15, # 统计异常值 > 15
            'pattern_consistency_score': 60  # 模式一致性 > 60
        }

        # 计算高风险因素数量
        high_risk_factors = 0
        for factor, threshold in high_risk_thresholds.items():
            if factor in score_breakdown and score_breakdown[factor] > threshold:
                high_risk_factors += 1

        # 根据高风险因素数量给予放大加分（更保守）
        if high_risk_factors >= 4:  # 4个或以上高风险因素
            amplification_bonus = 20.0
        elif high_risk_factors >= 3:  # 3个高风险因素
            amplification_bonus = 12.0
        elif high_risk_factors >= 2:  # 2个高风险因素
            amplification_bonus = 6.0
        elif high_risk_factors >= 1:  # 1个高风险因素
            amplification_bonus = 2.0

        # 特殊组合加分（更保守）
        if (score_breakdown.get('feature_deviation_score', 0) > 85 and
            score_breakdown.get('business_rule_score', 0) > 80):
            amplification_bonus += 8.0  # 极端偏离 + 高业务风险

        if (score_breakdown.get('cluster_anomaly_score', 0) > 20 and
            score_breakdown.get('feature_deviation_score', 0) > 85):
            amplification_bonus += 6.0   # 噪声点 + 极端偏离

        final_score = base_score + amplification_bonus
        return float(min(100, final_score))

    def _identify_risk_factors(self, row: pd.Series, cluster_id: int,
                             cluster_risk_mapping: Dict) -> List[str]:
        """识别风险因素"""
        factors = []

        # 聚类相关风险
        cluster_info = cluster_risk_mapping.get(cluster_id, {})
        if cluster_info.get('risk_level') in ['high', 'critical']:
            factors.append(f'高风险聚类 (聚类{cluster_id})')

        # 交易金额风险
        amount = row.get('transaction_amount', 0)
        if amount > 1000:
            factors.append('大额交易')

        # 时间风险
        hour = row.get('transaction_hour', 12)
        if hour <= 5:
            factors.append('凌晨交易')
        elif hour >= 22:
            factors.append('深夜交易')

        # 账户风险
        account_age = row.get('account_age_days', 365)
        if account_age < 30:
            factors.append('新账户')

        # 设备风险
        if row.get('device_used') == 'mobile':
            factors.append('移动设备')

        # 支付方式风险
        if row.get('payment_method') == 'bank transfer':
            factors.append('银行转账')

        # 地址不一致
        shipping = str(row.get('shipping_address', ''))
        billing = str(row.get('billing_address', ''))
        if shipping != billing:
            factors.append('地址不一致')

        return factors

    def _determine_risk_level(self, risk_score: float) -> str:
        """确定风险等级"""
        if risk_score >= self.risk_thresholds['critical']:
            return 'critical'
        elif risk_score >= self.risk_thresholds['high']:
            return 'high'
        elif risk_score >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'

    def _generate_risk_summary(self, risk_scores: List[float], risk_details: List[Dict],
                             cluster_risk_mapping: Dict, dynamic_thresholds: Dict = None,
                             distribution_analysis: Dict = None) -> Dict[str, Any]:
        """生成风险评分汇总 - JSON安全版本"""
        # 确保所有数值都是Python原生类型，避免JSON序列化问题
        summary = {
            'total_transactions': int(len(risk_scores)),
            'average_risk_score': float(round(np.mean(risk_scores), 2)) if risk_scores else 0.0,
            'risk_distribution': self._get_risk_distribution(risk_details),
            'high_risk_count': int(len([r for r in risk_details if r['risk_level'] in ['high', 'critical']])),
            'cluster_risk_mapping': self._serialize_cluster_mapping(cluster_risk_mapping),
            'results': self._serialize_risk_details(risk_details),
            'score_statistics': {
                'min_score': float(round(min(risk_scores), 2)) if risk_scores else 0.0,
                'max_score': float(round(max(risk_scores), 2)) if risk_scores else 0.0,
                'std_score': float(round(np.std(risk_scores), 2)) if risk_scores else 0.0,
                'median_score': float(round(np.median(risk_scores), 2)) if risk_scores else 0.0
            }
        }

        # 添加动态阈值信息
        if dynamic_thresholds:
            summary['dynamic_thresholds'] = {
                'applied': True,
                'thresholds': dynamic_thresholds,
                'default_thresholds': self.default_risk_thresholds
            }

            # 添加分布分析
            if distribution_analysis:
                summary['distribution_analysis'] = distribution_analysis
                summary['threshold_optimization_quality'] = distribution_analysis.get('is_reasonable', False)
        else:
            summary['dynamic_thresholds'] = {
                'applied': False,
                'thresholds': self.risk_thresholds
            }

        return summary

    def _serialize_cluster_mapping(self, cluster_mapping: Dict) -> Dict:
        """序列化聚类映射，确保JSON兼容"""
        serialized = {}
        for key, value in cluster_mapping.items():
            if isinstance(value, dict):
                serialized_value = {}
                for k, v in value.items():
                    if isinstance(v, (np.integer, np.floating)):
                        serialized_value[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        serialized_value[k] = v.tolist()
                    else:
                        serialized_value[k] = v
                serialized[str(key)] = serialized_value
            else:
                serialized[str(key)] = value
        return serialized

    def _serialize_risk_details(self, risk_details: List[Dict]) -> List[Dict]:
        """序列化风险详情，确保JSON兼容"""
        serialized_details = []
        for detail in risk_details:
            serialized_detail = {}
            for key, value in detail.items():
                if isinstance(value, (np.integer, np.floating)):
                    serialized_detail[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serialized_detail[key] = value.tolist()
                elif isinstance(value, dict):
                    # 递归处理嵌套字典
                    serialized_dict = {}
                    for k, v in value.items():
                        if isinstance(v, (np.integer, np.floating)):
                            serialized_dict[k] = float(v)
                        elif isinstance(v, np.ndarray):
                            serialized_dict[k] = v.tolist()
                        else:
                            serialized_dict[k] = v
                    serialized_detail[key] = serialized_dict
                else:
                    serialized_detail[key] = value
            serialized_details.append(serialized_detail)
        return serialized_details

    def _get_risk_distribution(self, risk_details: List[Dict]) -> Dict[str, int]:
        """获取风险等级分布"""
        distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}

        for detail in risk_details:
            risk_level = detail['risk_level']
            if risk_level in distribution:
                distribution[risk_level] += 1

        return distribution

    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'total_transactions': 0,
            'average_risk_score': 0,
            'risk_distribution': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
            'high_risk_count': 0,
            'cluster_risk_mapping': {},
            'results': [],
            'score_statistics': {
                'min_score': 0, 'max_score': 0, 'std_score': 0, 'median_score': 0
            }
        }

    def calibrate_with_minimal_labels(self, data: pd.DataFrame,
                                    unsupervised_results: Dict[str, Any],
                                    sample_ratio: float = 0.1) -> Dict[str, Any]:
        """
        使用少量真实标签校准无监督风险评分

        Args:
            data: 包含真实标签的数据
            unsupervised_results: 无监督风险评分结果
            sample_ratio: 使用的真实标签比例 (默认10%)

        Returns:
            校准结果和验证指标
        """
        try:
            if 'is_fraudulent' not in data.columns:
                logger.warning("数据中没有真实标签，无法进行校准")
                return self._empty_calibration_result()

            logger.info(f"开始校准，使用 {sample_ratio*100:.1f}% 的真实标签")

            # 1. 随机采样真实标签
            sample_size = int(len(data) * sample_ratio)
            sample_indices = np.random.choice(data.index, size=sample_size, replace=False)
            sample_data = data.loc[sample_indices]

            # 2. 获取对应的无监督评分
            unsupervised_scores = []
            true_labels = []

            for idx in sample_indices:
                # 找到对应的无监督评分结果
                result_idx = idx if idx < len(unsupervised_results['results']) else 0
                if result_idx < len(unsupervised_results['results']):
                    score = unsupervised_results['results'][result_idx]['risk_score']
                    unsupervised_scores.append(score)
                    true_labels.append(data.loc[idx, 'is_fraudulent'])

            if not unsupervised_scores:
                logger.error("无法获取无监督评分数据")
                return self._empty_calibration_result()

            # 3. 分析评分与真实标签的关系
            calibration_analysis = self._analyze_score_label_relationship(
                unsupervised_scores, true_labels
            )

            # 4. 优化风险阈值
            optimized_thresholds = self._optimize_risk_thresholds(
                unsupervised_scores, true_labels
            )

            # 5. 计算校准后的性能指标
            calibrated_predictions = self._apply_optimized_thresholds(
                unsupervised_scores, optimized_thresholds
            )

            performance_metrics = self._calculate_performance_metrics(
                true_labels, calibrated_predictions
            )

            # 6. 生成校准结果
            calibration_results = {
                'sample_size': sample_size,
                'sample_ratio': sample_ratio,
                'original_thresholds': self.risk_thresholds.copy(),
                'optimized_thresholds': optimized_thresholds,
                'calibration_analysis': calibration_analysis,
                'performance_metrics': performance_metrics,
                'improvement_summary': self._calculate_improvement_summary(
                    unsupervised_scores, true_labels, optimized_thresholds
                )
            }

            # 7. 更新阈值（可选）
            self.calibration_results = calibration_results

            logger.info("校准完成，性能指标已计算")
            return calibration_results

        except Exception as e:
            logger.error(f"校准过程出错: {e}")
            return self._empty_calibration_result()

    def _analyze_score_label_relationship(self, scores: List[float],
                                        labels: List[int]) -> Dict[str, Any]:
        """分析评分与真实标签的关系"""
        analysis = {}

        # 转换为numpy数组
        scores_array = np.array(scores)
        labels_array = np.array(labels)

        # 欺诈和正常交易的评分分布
        fraud_scores = scores_array[labels_array == 1]
        normal_scores = scores_array[labels_array == 0]

        analysis['fraud_score_stats'] = {
            'mean': round(np.mean(fraud_scores), 2) if len(fraud_scores) > 0 else 0,
            'std': round(np.std(fraud_scores), 2) if len(fraud_scores) > 0 else 0,
            'min': round(np.min(fraud_scores), 2) if len(fraud_scores) > 0 else 0,
            'max': round(np.max(fraud_scores), 2) if len(fraud_scores) > 0 else 0,
            'count': len(fraud_scores)
        }

        analysis['normal_score_stats'] = {
            'mean': round(np.mean(normal_scores), 2) if len(normal_scores) > 0 else 0,
            'std': round(np.std(normal_scores), 2) if len(normal_scores) > 0 else 0,
            'min': round(np.min(normal_scores), 2) if len(normal_scores) > 0 else 0,
            'max': round(np.max(normal_scores), 2) if len(normal_scores) > 0 else 0,
            'count': len(normal_scores)
        }

        # 计算分离度
        if len(fraud_scores) > 0 and len(normal_scores) > 0:
            separation = abs(np.mean(fraud_scores) - np.mean(normal_scores))
            analysis['score_separation'] = round(separation, 2)
        else:
            analysis['score_separation'] = 0

        return analysis

    def _optimize_risk_thresholds(self, scores: List[float],
                                labels: List[int]) -> Dict[str, float]:
        """优化风险阈值"""
        scores_array = np.array(scores)
        labels_array = np.array(labels)

        # 使用ROC曲线找到最优阈值
        from sklearn.metrics import roc_curve

        try:
            # 计算ROC曲线
            fpr, tpr, thresholds = roc_curve(labels_array, scores_array)

            # 找到最优阈值（Youden's J statistic）
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]

            # 基于最优阈值调整风险等级阈值
            optimized_thresholds = {
                'low': max(10, optimal_threshold - 20),
                'medium': optimal_threshold,
                'high': min(90, optimal_threshold + 15),
                'critical': min(95, optimal_threshold + 25)
            }

        except Exception as e:
            logger.warning(f"ROC优化失败，使用默认调整: {e}")
            # 基于统计分析的简单调整
            fraud_mean = np.mean(scores_array[labels_array == 1]) if np.any(labels_array == 1) else 70
            normal_mean = np.mean(scores_array[labels_array == 0]) if np.any(labels_array == 0) else 30

            optimized_thresholds = {
                'low': max(10, normal_mean + 5),
                'medium': (normal_mean + fraud_mean) / 2,
                'high': min(90, fraud_mean - 5),
                'critical': min(95, fraud_mean + 10)
            }

        return optimized_thresholds

    def _apply_optimized_thresholds(self, scores: List[float],
                                  optimized_thresholds: Dict[str, float]) -> List[int]:
        """应用优化后的阈值进行预测"""
        predictions = []

        for score in scores:
            if score >= optimized_thresholds['high']:
                predictions.append(1)  # 高风险 -> 欺诈
            else:
                predictions.append(0)  # 低/中风险 -> 正常

        return predictions

    def _calculate_performance_metrics(self, true_labels: List[int],
                                     predictions: List[int]) -> Dict[str, float]:
        """计算性能指标"""
        try:
            metrics = {
                'accuracy': round(accuracy_score(true_labels, predictions), 3),
                'precision': round(precision_score(true_labels, predictions, zero_division=0), 3),
                'recall': round(recall_score(true_labels, predictions, zero_division=0), 3),
                'f1_score': round(f1_score(true_labels, predictions, zero_division=0), 3)
            }

            # 计算混淆矩阵
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_labels, predictions)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['true_positive'] = int(tp)
                metrics['false_positive'] = int(fp)
                metrics['true_negative'] = int(tn)
                metrics['false_negative'] = int(fn)

                # 计算特异性
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['specificity'] = round(specificity, 3)

            return metrics

        except Exception as e:
            logger.error(f"计算性能指标时出错: {e}")
            return {
                'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0,
                'specificity': 0, 'true_positive': 0, 'false_positive': 0,
                'true_negative': 0, 'false_negative': 0
            }

    def _calculate_improvement_summary(self, scores: List[float], labels: List[int],
                                     optimized_thresholds: Dict[str, float]) -> Dict[str, Any]:
        """计算改进总结"""
        # 使用原始阈值的性能
        original_predictions = []
        for score in scores:
            if score >= self.risk_thresholds['high']:
                original_predictions.append(1)
            else:
                original_predictions.append(0)

        original_metrics = self._calculate_performance_metrics(labels, original_predictions)

        # 使用优化阈值的性能
        optimized_predictions = self._apply_optimized_thresholds(scores, optimized_thresholds)
        optimized_metrics = self._calculate_performance_metrics(labels, optimized_predictions)

        # 计算改进
        improvement = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            original_value = original_metrics.get(metric, 0)
            optimized_value = optimized_metrics.get(metric, 0)
            improvement[f'{metric}_improvement'] = round(optimized_value - original_value, 3)

        return {
            'original_performance': original_metrics,
            'optimized_performance': optimized_metrics,
            'improvements': improvement,
            'threshold_changes': {
                'low_threshold_change': round(optimized_thresholds['low'] - self.risk_thresholds['low'], 2),
                'medium_threshold_change': round(optimized_thresholds['medium'] - self.risk_thresholds['medium'], 2),
                'high_threshold_change': round(optimized_thresholds['high'] - self.risk_thresholds['high'], 2),
                'critical_threshold_change': round(optimized_thresholds['critical'] - self.risk_thresholds['critical'], 2)
            }
        }

    def _empty_calibration_result(self) -> Dict[str, Any]:
        """返回空的校准结果"""
        return {
            'sample_size': 0,
            'sample_ratio': 0,
            'original_thresholds': self.risk_thresholds.copy(),
            'optimized_thresholds': self.risk_thresholds.copy(),
            'calibration_analysis': {},
            'performance_metrics': {},
            'improvement_summary': {}
        }

    def apply_calibration(self, calibration_results: Dict[str, Any]):
        """应用校准结果，更新阈值"""
        if calibration_results and 'optimized_thresholds' in calibration_results:
            self.risk_thresholds = calibration_results['optimized_thresholds'].copy()
            logger.info("已应用校准结果，风险阈值已更新")
        else:
            logger.warning("校准结果无效，未更新阈值")

    def get_calibration_status(self) -> Dict[str, Any]:
        """获取校准状态"""
        return {
            'is_calibrated': self.calibration_results is not None,
            'current_thresholds': self.risk_thresholds.copy(),
            'calibration_results': self.calibration_results
        }

    def perform_full_calibration(self, data: pd.DataFrame,
                               cluster_results: Dict[str, Any],
                               sample_ratio: float = 0.1) -> Dict[str, Any]:
        """
        执行完整的校准流程

        Args:
            data: 包含真实标签的数据
            cluster_results: 聚类结果
            sample_ratio: 校准样本比例

        Returns:
            完整的校准结果
        """
        try:
            logger.info("开始执行完整校准流程...")

            # 1. 计算无监督风险评分
            unsupervised_results = self.calculate_unsupervised_risk_score(data, cluster_results)

            # 2. 执行校准
            calibration_results = self.calibrator.calibrate_risk_scoring(
                data, unsupervised_results, sample_ratio
            )

            # 3. 如果校准成功，应用校准结果
            if calibration_results['calibration_summary']['recommended_action'] == 'apply_calibration':
                optimized_thresholds = calibration_results['threshold_optimization']['optimized_thresholds']
                self.risk_thresholds = optimized_thresholds
                self.calibration_results = calibration_results
                logger.info("校准成功，已应用优化阈值")
            elif calibration_results['calibration_summary']['recommended_action'] == 'apply_with_caution':
                optimized_thresholds = calibration_results['threshold_optimization']['optimized_thresholds']
                # 保守地应用校准结果
                for level in self.risk_thresholds:
                    old_threshold = self.risk_thresholds[level]
                    new_threshold = optimized_thresholds[level]
                    # 只进行小幅调整
                    self.risk_thresholds[level] = old_threshold + (new_threshold - old_threshold) * 0.5
                self.calibration_results = calibration_results
                logger.info("谨慎应用校准结果，进行了保守调整")
            else:
                logger.warning("校准效果不佳，建议改进特征工程")

            # 4. 使用校准后的阈值重新计算风险评分
            calibrated_results = self.calculate_unsupervised_risk_score(data, cluster_results)

            # 5. 生成完整报告
            full_report = {
                'original_results': unsupervised_results,
                'calibration_results': calibration_results,
                'calibrated_results': calibrated_results,
                'calibration_applied': calibration_results['calibration_summary']['recommended_action'] in ['apply_calibration', 'apply_with_caution'],
                'final_thresholds': self.risk_thresholds.copy(),
                'improvement_summary': self._calculate_calibration_improvement(
                    unsupervised_results, calibrated_results, calibration_results
                )
            }

            logger.info("完整校准流程执行完成")
            return full_report

        except Exception as e:
            logger.error(f"完整校准流程失败: {e}")
            return {
                'original_results': {},
                'calibration_results': self.calibrator._empty_calibration_result(),
                'calibrated_results': {},
                'calibration_applied': False,
                'final_thresholds': self.risk_thresholds.copy(),
                'improvement_summary': {}
            }

    def _calculate_calibration_improvement(self, original_results: Dict,
                                         calibrated_results: Dict,
                                         calibration_results: Dict) -> Dict[str, Any]:
        """计算校准改进效果"""
        try:
            # 比较风险分布变化
            original_dist = original_results.get('risk_distribution', {})
            calibrated_dist = calibrated_results.get('risk_distribution', {})

            distribution_changes = {}
            for level in ['low', 'medium', 'high', 'critical']:
                original_count = original_dist.get(level, 0)
                calibrated_count = calibrated_dist.get(level, 0)
                distribution_changes[f'{level}_change'] = calibrated_count - original_count

            # 获取验证性能指标
            validation_metrics = calibration_results.get('validation_results', {}).get('performance_metrics', {})

            return {
                'distribution_changes': distribution_changes,
                'validation_performance': validation_metrics,
                'calibration_quality': calibration_results.get('calibration_summary', {}).get('data_quality', 'unknown'),
                'roc_auc_improvement': calibration_results.get('threshold_optimization', {}).get('roc_auc', 0.5) - 0.5
            }

        except Exception as e:
            logger.error(f"计算校准改进效果失败: {e}")
            return {}