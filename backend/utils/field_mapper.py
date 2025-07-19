"""
字段映射工具
统一处理原始数据集字段和清理后字段的映射关系
"""

import pandas as pd
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class FieldMapper:
    """字段映射器 - 统一处理数据集字段映射"""
    
    def __init__(self):
        # 原始字段到清理后字段的映射
        self.field_mapping = {
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
        
        # 反向映射
        self.reverse_mapping = {v: k for k, v in self.field_mapping.items()}
        
        # 数据类型映射
        self.data_types = {
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
        
        # 字段分类
        self.field_categories = {
            'id_fields': ['transaction_id', 'customer_id'],
            'amount_fields': ['transaction_amount', 'quantity'],
            'time_fields': ['transaction_date', 'transaction_hour'],
            'categorical_fields': ['payment_method', 'product_category', 'device_used'],
            'demographic_fields': ['customer_age', 'customer_location'],
            'address_fields': ['shipping_address', 'billing_address'],
            'technical_fields': ['ip_address'],
            'target_field': ['is_fraudulent'],
            'account_fields': ['account_age_days']
        }
    
    def detect_format(self, data: pd.DataFrame) -> str:
        """检测数据格式 (原始或清理后)"""
        if 'Transaction Amount' in data.columns:
            return 'original'
        elif 'transaction_amount' in data.columns:
            return 'cleaned'
        else:
            return 'unknown'
    
    def get_field_name(self, data: pd.DataFrame, original_name: str) -> Optional[str]:
        """获取实际的字段名"""
        format_type = self.detect_format(data)
        
        if format_type == 'original':
            return original_name if original_name in data.columns else None
        elif format_type == 'cleaned':
            cleaned_name = self.field_mapping.get(original_name)
            return cleaned_name if cleaned_name and cleaned_name in data.columns else None
        else:
            # 尝试两种格式
            if original_name in data.columns:
                return original_name
            cleaned_name = self.field_mapping.get(original_name)
            if cleaned_name and cleaned_name in data.columns:
                return cleaned_name
            return None
    
    def get_all_field_names(self, data: pd.DataFrame) -> Dict[str, str]:
        """获取所有字段的实际名称映射"""
        result = {}
        for original_name in self.field_mapping.keys():
            actual_name = self.get_field_name(data, original_name)
            if actual_name:
                result[original_name] = actual_name
        return result
    
    def validate_required_fields(self, data: pd.DataFrame, required_fields: List[str]) -> tuple[bool, List[str]]:
        """验证必需字段是否存在"""
        missing_fields = []
        for field in required_fields:
            if self.get_field_name(data, field) is None:
                missing_fields.append(field)
        
        return len(missing_fields) == 0, missing_fields
    
    def get_field_by_category(self, data: pd.DataFrame, category: str) -> List[str]:
        """根据分类获取字段名"""
        if category not in self.field_categories:
            return []
        
        result = []
        for cleaned_name in self.field_categories[category]:
            original_name = self.reverse_mapping.get(cleaned_name)
            if original_name:
                actual_name = self.get_field_name(data, original_name)
                if actual_name:
                    result.append(actual_name)
        return result
    
    def standardize_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化列名为清理后格式"""
        df = data.copy()
        
        # 如果是原始格式，转换为清理后格式
        if self.detect_format(df) == 'original':
            df = df.rename(columns=self.field_mapping)
        
        return df
    
    def get_data_info(self, data: pd.DataFrame) -> Dict:
        """获取数据集信息"""
        format_type = self.detect_format(data)
        field_mapping = self.get_all_field_names(data)
        
        return {
            'format': format_type,
            'shape': data.shape,
            'columns': list(data.columns),
            'field_mapping': field_mapping,
            'missing_fields': [f for f in self.field_mapping.keys() if f not in field_mapping],
            'available_categories': {
                cat: self.get_field_by_category(data, cat) 
                for cat in self.field_categories.keys()
            }
        }

# 全局实例
field_mapper = FieldMapper()
