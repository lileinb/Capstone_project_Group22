"""
风险特征工程器
基于真实数据集字段生成风险相关特征
支持的数据集字段：
- Transaction ID, Customer ID, Transaction Amount, Transaction Date
- Payment Method, Product Category, Quantity, Customer Age
- Customer Location, Device Used, IP Address
- Shipping Address, Billing Address, Is Fraudulent
- Account Age Days, Transaction Hour
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime, timedelta
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskFeatureEngineer:
    """基于真实数据集字段的风险特征工程器"""
    
    def __init__(self):
        self.original_features = []
        self.risk_features = []
        self.feature_importance = {}
        self.feature_categories = {
            "时间风险特征": [],
            "金额风险特征": [],
            "设备地理特征": [],
            "账户行为特征": [],
            "支付行为特征": [],
            "地址一致性特征": []
        }
        
        # 数据集必需字段 (支持原始格式和清理后格式)
        self.required_fields_original = [
            'Transaction ID', 'Customer ID', 'Transaction Amount', 'Transaction Date',
            'Payment Method', 'Product Category', 'Quantity', 'Customer Age',
            'Customer Location', 'Device Used', 'IP Address',
            'Shipping Address', 'Billing Address', 'Is Fraudulent',
            'Account Age Days', 'Transaction Hour'
        ]

        self.required_fields_cleaned = [
            'transaction_id', 'customer_id', 'transaction_amount', 'transaction_date',
            'payment_method', 'product_category', 'quantity', 'customer_age',
            'customer_location', 'device_used', 'ip_address',
            'shipping_address', 'billing_address', 'is_fraudulent',
            'account_age_days', 'transaction_hour'
        ]
    
    def validate_data_fields(self, data: pd.DataFrame) -> bool:
        """验证数据字段是否符合要求 (支持原始格式和清理后格式)"""
        # 检查原始格式
        missing_original = [field for field in self.required_fields_original if field not in data.columns]
        if len(missing_original) == 0:
            return True

        # 检查清理后格式
        missing_cleaned = [field for field in self.required_fields_cleaned if field not in data.columns]
        if len(missing_cleaned) == 0:
            return True

        # 都不匹配
        logger.error(f"数据集字段格式不匹配。请确保包含以下字段之一:")
        logger.error(f"原始格式: {self.required_fields_original}")
        logger.error(f"清理后格式: {self.required_fields_cleaned}")
        return False

    def _get_field_name(self, data: pd.DataFrame, original_name: str, cleaned_name: str) -> str:
        """获取实际的字段名 (原始或清理后)"""
        if original_name in data.columns:
            return original_name
        elif cleaned_name in data.columns:
            return cleaned_name
        else:
            return None
        
    def create_time_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """基于Transaction Hour和Transaction Date创建时间风险特征"""
        if data is None or data.empty:
            return data

        df = data.copy()

        # 获取实际字段名
        hour_field = self._get_field_name(df, 'Transaction Hour', 'transaction_hour')
        date_field = self._get_field_name(df, 'Transaction Date', 'transaction_date')

        if hour_field is None:
            logger.warning("未找到交易小时字段，跳过时间风险特征创建")
            return df

        # 1. 基于Transaction Hour的时间风险评分
        # 深夜和凌晨时段(0-6点, 22-23点)风险较高
        df['time_risk_score'] = 1  # 默认低风险
        df.loc[df[hour_field].isin([0,1,2,3,4,5,6]), 'time_risk_score'] = 3  # 高风险
        df.loc[df[hour_field].isin([22,23]), 'time_risk_score'] = 2  # 中风险
        df.loc[df[hour_field].isin([7,8,9]), 'time_risk_score'] = 2  # 早高峰中风险
        self.feature_categories["时间风险特征"].append('time_risk_score')

        # 2. 深夜交易标记 (22点-6点)
        df['is_night_transaction'] = df[hour_field].apply(
            lambda x: 1 if x >= 22 or x <= 6 else 0
        )
        self.feature_categories["时间风险特征"].append('is_night_transaction')

        # 3. 凌晨交易标记 (0-5点)
        df['is_early_morning'] = df[hour_field].apply(
            lambda x: 1 if 0 <= x <= 5 else 0
        )
        self.feature_categories["时间风险特征"].append('is_early_morning')

        # 4. 基于Transaction Date的工作日/周末特征
        if date_field:
            try:
                df['transaction_date_parsed'] = pd.to_datetime(df[date_field])
                df['is_weekend'] = (df['transaction_date_parsed'].dt.weekday >= 5).astype(int)
                df['weekday'] = df['transaction_date_parsed'].dt.weekday
                self.feature_categories["时间风险特征"].extend(['is_weekend', 'weekday'])
            except Exception as e:
                logger.warning(f"日期解析失败，跳过工作日特征: {e}")

        # 5. 交易时间段分类
        df['time_period'] = 0  # 默认
        df.loc[df[hour_field].between(6, 11), 'time_period'] = 1  # 上午
        df.loc[df[hour_field].between(12, 17), 'time_period'] = 2  # 下午
        df.loc[df[hour_field].between(18, 21), 'time_period'] = 3  # 晚上
        df.loc[df[hour_field].isin([22,23,0,1,2,3,4,5]), 'time_period'] = 4  # 深夜
        self.feature_categories["时间风险特征"].append('time_period')

        logger.info(f"创建了{len(self.feature_categories['时间风险特征'])}个时间风险特征")
        return df
    
    def create_amount_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """基于Transaction Amount和Quantity创建金额风险特征"""
        if data is None or data.empty:
            return data

        df = data.copy()

        # 获取实际字段名
        amount_field = self._get_field_name(df, 'Transaction Amount', 'transaction_amount')
        quantity_field = self._get_field_name(df, 'Quantity', 'quantity')
        customer_field = self._get_field_name(df, 'Customer ID', 'customer_id')

        if amount_field is None:
            logger.warning("未找到交易金额字段，跳过金额风险特征创建")
            return df

        # 1. 交易金额标准化分数
        amount_mean = df[amount_field].mean()
        amount_std = df[amount_field].std()
        df['amount_z_score'] = (df[amount_field] - amount_mean) / amount_std
        df['amount_risk_score'] = np.abs(df['amount_z_score'])
        self.feature_categories["金额风险特征"].append('amount_risk_score')

        # 2. 大额交易标记 (基于95分位数)
        amount_95th = df[amount_field].quantile(0.95)
        df['is_large_amount'] = (df[amount_field] > amount_95th).astype(int)
        self.feature_categories["金额风险特征"].append('is_large_amount')

        # 3. 小额交易标记 (基于5分位数)
        amount_5th = df[amount_field].quantile(0.05)
        df['is_small_amount'] = (df[amount_field] < amount_5th).astype(int)
        self.feature_categories["金额风险特征"].append('is_small_amount')

        # 4. 金额分位数
        df['amount_percentile'] = df[amount_field].rank(pct=True)
        self.feature_categories["金额风险特征"].append('amount_percentile')

        # 5. 金额异常程度评分
        df['amount_anomaly_score'] = 1  # 默认正常
        df.loc[df['amount_z_score'].abs() > 2, 'amount_anomaly_score'] = 3  # 高异常
        df.loc[(df['amount_z_score'].abs() > 1) & (df['amount_z_score'].abs() <= 2), 'amount_anomaly_score'] = 2  # 中异常
        self.feature_categories["金额风险特征"].append('amount_anomaly_score')

        # 6. 基于Quantity的特征
        if quantity_field:
            df['unit_price'] = df[amount_field] / df[quantity_field]
            df['is_high_quantity'] = (df[quantity_field] >= 4).astype(int)  # 数量>=4为高数量
            df['is_single_item'] = (df[quantity_field] == 1).astype(int)
            self.feature_categories["金额风险特征"].extend(['unit_price', 'is_high_quantity', 'is_single_item'])

        # 7. 金额范围分类
        df['amount_range'] = 0  # 默认
        df.loc[df[amount_field] <= 50, 'amount_range'] = 1  # 小额
        df.loc[(df[amount_field] > 50) & (df[amount_field] <= 200), 'amount_range'] = 2  # 中额
        df.loc[(df[amount_field] > 200) & (df[amount_field] <= 500), 'amount_range'] = 3  # 较大额
        df.loc[df[amount_field] > 500, 'amount_range'] = 4  # 大额
        self.feature_categories["金额风险特征"].append('amount_range')

        # 8. 用户历史金额对比
        if customer_field:
            user_avg_amount = df.groupby(customer_field)[amount_field].transform('mean')
            user_std_amount = df.groupby(customer_field)[amount_field].transform('std').fillna(0)
            df['amount_vs_user_avg'] = df[amount_field] / user_avg_amount
            df['amount_deviation_from_user'] = np.where(
                user_std_amount > 0,
                np.abs(df[amount_field] - user_avg_amount) / user_std_amount,
                0
            )
            self.feature_categories["金额风险特征"].extend(['amount_vs_user_avg', 'amount_deviation_from_user'])

        logger.info(f"创建了{len(self.feature_categories['金额风险特征'])}个金额风险特征")
        return df
    
    def create_device_geographic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """基于Device Used, Customer Location, IP Address创建设备地理特征"""
        if data is None or data.empty:
            return data

        df = data.copy()

        # 获取实际字段名
        device_field = self._get_field_name(df, 'Device Used', 'device_used')
        location_field = self._get_field_name(df, 'Customer Location', 'customer_location')
        ip_field = self._get_field_name(df, 'IP Address', 'ip_address')

        # 1. 设备类型风险评分
        if device_field:
            device_risk_mapping = {
                'mobile': 2,    # 移动设备风险中等
                'tablet': 2,    # 平板设备风险中等
                'desktop': 1    # 桌面设备风险较低
            }
            df['device_risk_score'] = df[device_field].map(device_risk_mapping).fillna(3)
            self.feature_categories["设备地理特征"].append('device_risk_score')

            # 2. 设备类型编码
            df['is_mobile'] = (df[device_field] == 'mobile').astype(int)
            df['is_desktop'] = (df[device_field] == 'desktop').astype(int)
            df['is_tablet'] = (df[device_field] == 'tablet').astype(int)
            self.feature_categories["设备地理特征"].extend(['is_mobile', 'is_desktop', 'is_tablet'])

        # 3. IP地址风险分析
        if ip_field:
            df['ip_first_octet'] = df[ip_field].str.split('.').str[0].astype(float, errors='ignore')
            df['ip_risk_score'] = 1  # 默认低风险
            df.loc[df['ip_first_octet'].isin([10, 172, 192]), 'ip_risk_score'] = 2
            df.loc[df['ip_first_octet'] == 127, 'ip_risk_score'] = 3
            self.feature_categories["设备地理特征"].append('ip_risk_score')

        # 4. 客户位置特征
        if location_field:
            df['location_name_length'] = df[location_field].str.len()
            df['location_risk_score'] = 1  # 默认
            df.loc[df['location_name_length'] < 5, 'location_risk_score'] = 3
            df.loc[df['location_name_length'] > 20, 'location_risk_score'] = 2
            self.feature_categories["设备地理特征"].extend(['location_name_length', 'location_risk_score'])

        logger.info(f"创建了{len(self.feature_categories['设备地理特征'])}个设备地理特征")
        return df

    def create_account_behavior_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """基于Customer Age, Account Age Days创建账户行为特征"""
        if data is None or data.empty:
            return data

        df = data.copy()

        # 获取实际字段名
        age_field = self._get_field_name(df, 'Customer Age', 'customer_age')
        account_age_field = self._get_field_name(df, 'Account Age Days', 'account_age_days')
        customer_field = self._get_field_name(df, 'Customer ID', 'customer_id')

        # 1. 账户年龄风险评分
        if account_age_field:
            df['account_age_risk_score'] = 1  # 默认低风险
            df.loc[df[account_age_field] < 30, 'account_age_risk_score'] = 3  # 新账户高风险
            df.loc[(df[account_age_field] >= 30) & (df[account_age_field] < 90), 'account_age_risk_score'] = 2  # 较新账户中风险
            self.feature_categories["账户行为特征"].append('account_age_risk_score')

            # 2. 账户年龄分类
            df['is_new_account'] = (df[account_age_field] < 30).astype(int)
            df['is_very_new_account'] = (df[account_age_field] < 7).astype(int)
            df['is_mature_account'] = (df[account_age_field] > 180).astype(int)
            self.feature_categories["账户行为特征"].extend(['is_new_account', 'is_very_new_account', 'is_mature_account'])

        # 3. 客户年龄风险评分
        if age_field:
            df['customer_age_risk_score'] = 1  # 默认低风险
            df.loc[df[age_field] < 18, 'customer_age_risk_score'] = 3  # 未成年高风险
            df.loc[df[age_field] > 70, 'customer_age_risk_score'] = 2  # 高龄中风险
            df.loc[df[age_field] < 0, 'customer_age_risk_score'] = 3  # 异常年龄高风险
            self.feature_categories["账户行为特征"].append('customer_age_risk_score')

            # 4. 年龄分组
            df['age_group'] = 0  # 默认
            df.loc[df[age_field].between(18, 25), 'age_group'] = 1  # 青年
            df.loc[df[age_field].between(26, 35), 'age_group'] = 2  # 青壮年
            df.loc[df[age_field].between(36, 50), 'age_group'] = 3  # 中年
            df.loc[df[age_field].between(51, 65), 'age_group'] = 4  # 中老年
            df.loc[df[age_field] > 65, 'age_group'] = 5  # 老年
            self.feature_categories["账户行为特征"].append('age_group')

        # 5. 账户年龄与客户年龄的关系
        if account_age_field and age_field:
            df['account_age_ratio'] = df[account_age_field] / (df[age_field] * 365)  # 账户年龄占客户年龄比例
            df['account_age_ratio'] = df['account_age_ratio'].fillna(0)
            self.feature_categories["账户行为特征"].append('account_age_ratio')

        # 6. 用户交易频率特征
        if customer_field:
            user_transaction_count = df.groupby(customer_field).size()
            df['user_transaction_frequency'] = df[customer_field].map(user_transaction_count)
            df['is_frequent_user'] = (df['user_transaction_frequency'] > 1).astype(int)
            self.feature_categories["账户行为特征"].extend(['user_transaction_frequency', 'is_frequent_user'])

        logger.info(f"创建了{len(self.feature_categories['账户行为特征'])}个账户行为特征")
        return df

    def create_payment_behavior_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """基于Payment Method, Product Category创建支付行为特征"""
        if data is None or data.empty:
            return data

        df = data.copy()

        # 获取实际字段名
        payment_field = self._get_field_name(df, 'Payment Method', 'payment_method')
        category_field = self._get_field_name(df, 'Product Category', 'product_category')
        customer_field = self._get_field_name(df, 'Customer ID', 'customer_id')

        # 1. Payment method risk scoring
        if payment_field:
            payment_risk_mapping = {
                'credit card': 1,      # Credit card lower risk
                'debit card': 1,       # Debit card lower risk
                'bank transfer': 2,    # Bank transfer medium risk
                'PayPal': 2           # PayPal medium risk
            }
            df['payment_risk_score'] = df[payment_field].map(payment_risk_mapping).fillna(3)
            self.feature_categories["Payment Behavior Features"].append('payment_risk_score')

        # 2. 支付方式编码 (使用正确的列名)
        if payment_field:
            df['is_credit_card'] = (df[payment_field] == 'credit card').astype(int)
            df['is_debit_card'] = (df[payment_field] == 'debit card').astype(int)
            df['is_bank_transfer'] = (df[payment_field] == 'bank transfer').astype(int)
            df['is_paypal'] = (df[payment_field] == 'PayPal').astype(int)
            self.feature_categories["支付行为特征"].extend(['is_credit_card', 'is_debit_card', 'is_bank_transfer', 'is_paypal'])

        # 3. 产品类别风险评分 (使用正确的列名)
        if category_field:
            category_risk_mapping = {
                'electronics': 3,        # 电子产品风险较高
                'clothing': 1,          # 服装风险较低
                'home & garden': 1,     # 家居园艺风险较低
                'health & beauty': 2,   # 健康美容风险中等
                'toys & games': 2       # 玩具游戏风险中等
            }
            df['category_risk_score'] = df[category_field].map(category_risk_mapping).fillna(2)
            self.feature_categories["支付行为特征"].append('category_risk_score')

            # 4. 产品类别编码
            df['is_electronics'] = (df[category_field] == 'electronics').astype(int)
            df['is_clothing'] = (df[category_field] == 'clothing').astype(int)
            df['is_home_garden'] = (df[category_field] == 'home & garden').astype(int)
            df['is_health_beauty'] = (df[category_field] == 'health & beauty').astype(int)
            df['is_toys_games'] = (df[category_field] == 'toys & games').astype(int)
            self.feature_categories["支付行为特征"].extend(['is_electronics', 'is_clothing', 'is_home_garden', 'is_health_beauty', 'is_toys_games'])

        # 5. 用户支付方式多样性
        user_payment_diversity = df.groupby('Customer ID')['Payment Method'].nunique()
        df['payment_method_diversity'] = df['Customer ID'].map(user_payment_diversity)
        df['is_diverse_payment_user'] = (df['payment_method_diversity'] > 1).astype(int)
        self.feature_categories["支付行为特征"].extend(['payment_method_diversity', 'is_diverse_payment_user'])

        # 6. 用户产品类别多样性
        user_category_diversity = df.groupby('Customer ID')['Product Category'].nunique()
        df['category_diversity'] = df['Customer ID'].map(user_category_diversity)
        df['is_diverse_category_user'] = (df['category_diversity'] > 1).astype(int)
        self.feature_categories["支付行为特征"].extend(['category_diversity', 'is_diverse_category_user'])

        logger.info(f"创建了{len(self.feature_categories['支付行为特征'])}个支付行为特征")
        return df

    def create_address_consistency_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """基于Shipping Address, Billing Address创建地址一致性特征"""
        if data is None or data.empty:
            return data

        df = data.copy()

        # 获取实际字段名
        shipping_field = self._get_field_name(df, 'Shipping Address', 'shipping_address')
        billing_field = self._get_field_name(df, 'Billing Address', 'billing_address')

        if shipping_field and billing_field:
            # 1. 地址完全一致性检查
            df['address_exact_match'] = (df[shipping_field] == df[billing_field]).astype(int)
            df['address_mismatch_risk'] = (1 - df['address_exact_match']) * 3  # 不一致为高风险
            self.feature_categories["地址一致性特征"].extend(['address_exact_match', 'address_mismatch_risk'])

            # 2. 地址长度特征
            df['shipping_address_length'] = df[shipping_field].str.len()
            df['billing_address_length'] = df[billing_field].str.len()
            df['address_length_diff'] = np.abs(df['shipping_address_length'] - df['billing_address_length'])
            self.feature_categories["地址一致性特征"].extend(['shipping_address_length', 'billing_address_length', 'address_length_diff'])

            # 3. 地址相似性分析 (简化版)
            # 检查是否包含相同的关键词
            def extract_address_keywords(address):
                if pd.isna(address):
                    return set()
                # 提取数字和单词
                words = re.findall(r'\b\w+\b', str(address).lower())
                return set(words)

            df['shipping_keywords'] = df[shipping_field].apply(extract_address_keywords)
            df['billing_keywords'] = df[billing_field].apply(extract_address_keywords)

            # 计算关键词重叠度
            df['address_keyword_overlap'] = df.apply(
                lambda row: len(row['shipping_keywords'] & row['billing_keywords']) /
                           max(len(row['shipping_keywords'] | row['billing_keywords']), 1),
                axis=1
            )
            df['address_similarity_risk'] = df['address_keyword_overlap'].apply(
                lambda x: 1 if x > 0.8 else (2 if x > 0.5 else 3)
            )
            self.feature_categories["地址一致性特征"].extend(['address_keyword_overlap', 'address_similarity_risk'])

            # 4. 地址格式风险
            # 检查地址是否包含PO Box等高风险格式
            df['has_po_box'] = df[shipping_field].str.contains('Box|BOX|P.O.|PO', na=False).astype(int)
            df['has_dpo'] = df[shipping_field].str.contains('DPO|APO', na=False).astype(int)
            df['address_format_risk'] = (df['has_po_box'] + df['has_dpo']) * 2
            self.feature_categories["地址一致性特征"].extend(['has_po_box', 'has_dpo', 'address_format_risk'])

            # 清理临时列
            df = df.drop(['shipping_keywords', 'billing_keywords'], axis=1)

        logger.info(f"创建了{len(self.feature_categories['地址一致性特征'])}个地址一致性特征")
        return df

    def engineer_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """执行完整的特征工程流程"""
        if data is None or data.empty:
            logger.error("输入数据为空")
            return pd.DataFrame()

        # 验证数据字段
        if not self.validate_data_fields(data):
            logger.error("数据字段验证失败")
            return data

        logger.info("开始执行完整特征工程...")

        # 记录原始特征
        self.original_features = list(data.columns)

        # 清空特征分类记录
        for category in self.feature_categories:
            self.feature_categories[category] = []

        # Execute various feature engineering
        df = self.create_time_risk_features(data)
        df = self.create_amount_risk_features(df)
        df = self.create_device_geographic_features(df)
        df = self.create_account_behavior_features(df)

        # Simplified version, temporarily skip complex features
        # df = self.create_payment_behavior_features(df)
        # df = self.create_address_consistency_features(df)

        # Record newly created features
        self.risk_features = [col for col in df.columns if col not in self.original_features]

        logger.info(f"Feature engineering completed: {len(self.original_features)} original features, {len(self.risk_features)} new risk features added")
        logger.info(f"Feature category statistics: {[(k, len(v)) for k, v in self.feature_categories.items()]}")

        return df

    def get_feature_info(self) -> Dict[str, Any]:
        """获取特征工程信息"""
        return {
            'original_features': self.original_features,
            'risk_features': self.risk_features,
            'feature_categories': self.feature_categories,
            'total_features': len(self.original_features) + len(self.risk_features),
            'feature_importance': self.feature_importance
        }

    def calculate_feature_importance(self, data: pd.DataFrame, target_column: str = 'Is Fraudulent') -> Dict[str, float]:
        """Calculate feature importance"""
        if data is None or data.empty or target_column not in data.columns:
            return {}

        # Only calculate importance for numeric features
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != target_column]

        if len(numeric_columns) == 0:
            return {}

        # Calculate correlation with target variable
        correlations = data[numeric_columns].corrwith(data[target_column]).abs()
        self.feature_importance = correlations.to_dict()

        return self.feature_importance
