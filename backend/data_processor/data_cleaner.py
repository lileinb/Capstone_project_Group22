"""
数据清理器
基于真实电商欺诈数据集的清理和预处理
处理16个标准字段的数据质量问题
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
from datetime import datetime, timedelta
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """基于真实数据集的数据清理器"""

    def __init__(self):
        self.cleaning_log = []
        self.original_shape = None
        self.cleaned_shape = None

        # 标准字段映射（原始名称 -> 清理后名称）
        self.column_mapping = {
            'Transaction ID': 'transaction_id',
            'Customer ID': 'customer_id',
            'Transaction Amount': 'transaction_amount',
            'Transaction Date': 'transaction_date',
            'Payment Method': 'payment_method',
            'Product Category': 'product_category',
            'Quantity': 'quantity',
            'Customer Age': 'customer_age',
            'Customer Location': 'customer_location',
            'Device Used': 'device_used',
            'IP Address': 'ip_address',
            'Shipping Address': 'shipping_address',
            'Billing Address': 'billing_address',
            'Is Fraudulent': 'is_fraudulent',
            'Account Age Days': 'account_age_days',
            'Transaction Hour': 'transaction_hour'
        }

        # 数据类型映射
        self.dtype_mapping = {
            'transaction_id': 'object',
            'customer_id': 'object',
            'transaction_amount': 'float64',
            'transaction_date': 'object',
            'payment_method': 'category',
            'product_category': 'category',
            'quantity': 'int64',
            'customer_age': 'int64',
            'customer_location': 'object',
            'device_used': 'category',
            'ip_address': 'object',
            'shipping_address': 'object',
            'billing_address': 'object',
            'is_fraudulent': 'int64',
            'account_age_days': 'int64',
            'transaction_hour': 'int64'
        }

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        完整的数据清理流程

        Args:
            data: 原始DataFrame

        Returns:
            清洗后的DataFrame
        """
        if data is None or data.empty:
            logger.error("输入数据为空")
            return pd.DataFrame()

        self.original_shape = data.shape
        self.cleaning_log = []

        logger.info(f"开始数据清理，原始数据形状: {self.original_shape}")

        df = data.copy()

        # 1. 标准化列名
        df = self._standardize_column_names(df)

        # 2. 处理重复数据
        df = self._handle_duplicates(df)

        # 3. 处理缺失值
        df = self._handle_missing_values(df)

        # 4. 数据类型转换
        df = self._convert_data_types(df)

        # 5. 处理异常值
        df = self._handle_outliers(df)

        # 6. 数据验证
        df = self._validate_cleaned_data(df)

        self.cleaned_shape = df.shape
        logger.info(f"数据清理完成，清理后数据形状: {self.cleaned_shape}")
        logger.info(f"清理日志: {self.cleaning_log}")

        return df

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        # 使用映射表重命名列
        df_renamed = df.rename(columns=self.column_mapping)

        # 记录重命名的列
        renamed_cols = [col for col in df.columns if col in self.column_mapping]
        if renamed_cols:
            self.cleaning_log.append(f"重命名列: {renamed_cols}")

        return df_renamed

    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理重复数据"""
        initial_count = len(df)

        # 基于transaction_id去重（如果存在）
        if 'transaction_id' in df.columns:
            df = df.drop_duplicates(subset=['transaction_id'], keep='first')
        else:
            df = df.drop_duplicates()

        removed_count = initial_count - len(df)
        if removed_count > 0:
            self.cleaning_log.append(f"删除重复行: {removed_count}条")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]

        if len(missing_cols) > 0:
            self.cleaning_log.append(f"发现缺失值: {missing_cols.to_dict()}")

            # 对于关键字段，删除缺失值的行
            critical_fields = ['transaction_id', 'customer_id', 'transaction_amount', 'is_fraudulent']
            for field in critical_fields:
                if field in df.columns and df[field].isnull().any():
                    before_count = len(df)
                    df = df.dropna(subset=[field])
                    after_count = len(df)
                    self.cleaning_log.append(f"删除{field}缺失的行: {before_count - after_count}条")

            # 对于非关键字段，使用合理的默认值填充
            for col in df.columns:
                if df[col].isnull().any():
                    if col in ['customer_age']:
                        # 年龄用中位数填充
                        median_age = df[col].median()
                        df[col] = df[col].fillna(median_age)
                        self.cleaning_log.append(f"{col}用中位数{median_age}填充缺失值")
                    elif col in ['account_age_days']:
                        # 账户年龄用平均值填充
                        mean_days = df[col].mean()
                        df[col] = df[col].fillna(mean_days)
                        self.cleaning_log.append(f"{col}用平均值{mean_days:.0f}填充缺失值")
                    elif df[col].dtype == 'object':
                        # 文本字段用"unknown"填充
                        df[col] = df[col].fillna('unknown')
                        self.cleaning_log.append(f"{col}用'unknown'填充缺失值")
                    else:
                        # 数值字段用0填充
                        df[col] = df[col].fillna(0)
                        self.cleaning_log.append(f"{col}用0填充缺失值")

        return df

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型"""
        for col, target_dtype in self.dtype_mapping.items():
            if col in df.columns:
                try:
                    if target_dtype == 'category':
                        df[col] = df[col].astype('category')
                    elif target_dtype == 'int64':
                        # 确保没有小数点，然后转换为整数
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                    elif target_dtype == 'float64':
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                    elif target_dtype == 'object':
                        df[col] = df[col].astype('object')

                    self.cleaning_log.append(f"转换{col}为{target_dtype}类型")
                except Exception as e:
                    logger.warning(f"转换{col}类型失败: {e}")

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        # 处理客户年龄异常值
        if 'customer_age' in df.columns:
            # 年龄应该在合理范围内
            before_count = len(df)
            df = df[(df['customer_age'] >= 0) & (df['customer_age'] <= 100)]
            after_count = len(df)
            if before_count != after_count:
                self.cleaning_log.append(f"删除异常年龄记录: {before_count - after_count}条")

        # 处理交易金额异常值
        if 'transaction_amount' in df.columns:
            # 删除负数金额
            before_count = len(df)
            df = df[df['transaction_amount'] > 0]
            after_count = len(df)
            if before_count != after_count:
                self.cleaning_log.append(f"删除负数金额记录: {before_count - after_count}条")

            # 处理极端异常值（使用IQR方法）
            Q1 = df['transaction_amount'].quantile(0.25)
            Q3 = df['transaction_amount'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # 使用3倍IQR作为极端异常值阈值
            upper_bound = Q3 + 3 * IQR

            before_count = len(df)
            df = df[(df['transaction_amount'] >= max(0, lower_bound)) &
                   (df['transaction_amount'] <= upper_bound)]
            after_count = len(df)
            if before_count != after_count:
                self.cleaning_log.append(f"删除极端金额异常值: {before_count - after_count}条")

        # 处理数量异常值
        if 'quantity' in df.columns:
            before_count = len(df)
            df = df[(df['quantity'] >= 1) & (df['quantity'] <= 10)]
            after_count = len(df)
            if before_count != after_count:
                self.cleaning_log.append(f"删除异常数量记录: {before_count - after_count}条")

        return df

    def _validate_cleaned_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证清理后的数据"""
        # 检查关键字段是否存在
        required_fields = ['transaction_id', 'customer_id', 'transaction_amount', 'is_fraudulent']
        missing_required = [field for field in required_fields if field not in df.columns]
        if missing_required:
            logger.warning(f"清理后仍缺少关键字段: {missing_required}")

        # 检查数据完整性
        if len(df) == 0:
            logger.error("清理后数据为空")
        else:
            # 检查欺诈标签分布
            if 'is_fraudulent' in df.columns:
                fraud_rate = df['is_fraudulent'].mean()
                self.cleaning_log.append(f"欺诈率: {fraud_rate:.3f}")

        return df

    def get_cleaning_summary(self) -> Dict:
        """获取清理摘要"""
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.cleaned_shape,
            'cleaning_log': self.cleaning_log,
            'data_reduction': {
                'rows_removed': self.original_shape[0] - self.cleaned_shape[0] if self.original_shape and self.cleaned_shape else 0,
                'columns_changed': len(self.column_mapping)
            }
        }