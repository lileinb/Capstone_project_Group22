"""
数据加载器
基于真实电商欺诈数据集的加载和验证
支持的数据集字段：
- Transaction ID, Customer ID, Transaction Amount, Transaction Date
- Payment Method, Product Category, Quantity, Customer Age
- Customer Location, Device Used, IP Address
- Shipping Address, Billing Address, Is Fraudulent
- Account Age Days, Transaction Hour
"""

import pandas as pd
import os
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """基于真实数据集的数据加载器"""

    def __init__(self, data_dir: str = "Dataset"):
        """
        初始化数据加载器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir

        # 真实数据集的标准字段
        self.standard_columns = [
            'Transaction ID', 'Customer ID', 'Transaction Amount', 'Transaction Date',
            'Payment Method', 'Product Category', 'Quantity', 'Customer Age',
            'Customer Location', 'Device Used', 'IP Address',
            'Shipping Address', 'Billing Address', 'Is Fraudulent',
            'Account Age Days', 'Transaction Hour'
        ]

        # 字段数据类型映射
        self.column_dtypes = {
            'Transaction ID': 'object',
            'Customer ID': 'object',
            'Transaction Amount': 'float64',
            'Transaction Date': 'object',
            'Payment Method': 'category',
            'Product Category': 'category',
            'Quantity': 'int64',
            'Customer Age': 'int64',
            'Customer Location': 'object',
            'Device Used': 'category',
            'IP Address': 'object',
            'Shipping Address': 'object',
            'Billing Address': 'object',
            'Is Fraudulent': 'int64',
            'Account Age Days': 'int64',
            'Transaction Hour': 'int64'
        }

        # 预期的字段值范围
        self.field_constraints = {
            'Transaction Amount': {'min': 0, 'max': 10000},
            'Quantity': {'min': 1, 'max': 10},
            'Customer Age': {'min': 0, 'max': 100},
            'Is Fraudulent': {'values': [0, 1]},
            'Account Age Days': {'min': 1, 'max': 365},
            'Transaction Hour': {'min': 0, 'max': 23},
            'Payment Method': {'values': ['credit card', 'debit card', 'bank transfer', 'PayPal']},
            'Product Category': {'values': ['electronics', 'clothing', 'home & garden', 'health & beauty', 'toys & games']},
            'Device Used': {'values': ['mobile', 'tablet', 'desktop']}
        }
    
    def load_dataset(self, file_path: str, validate_schema: bool = True) -> pd.DataFrame:
        """
        加载数据集并验证字段结构

        Args:
            file_path: 文件路径
            validate_schema: 是否验证数据集结构

        Returns:
            加载的数据集DataFrame
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 根据文件扩展名选择加载方法
            if file_path.endswith('.csv'):
                # 尝试多种编码格式
                encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
                data = None
                for encoding in encodings:
                    try:
                        data = pd.read_csv(file_path, encoding=encoding)
                        logger.info(f"使用编码 {encoding} 成功加载CSV文件")
                        break
                    except UnicodeDecodeError:
                        continue
                if data is None:
                    raise ValueError("无法使用任何编码格式读取CSV文件")

            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                data = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                data = pd.read_json(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")

            logger.info(f"成功加载数据集: {file_path}")
            logger.info(f"数据集形状: {data.shape}")
            logger.info(f"列名: {list(data.columns)}")

            # 验证数据集结构
            if validate_schema:
                validation_result = self.validate_dataset_schema(data)
                if not validation_result['is_valid']:
                    logger.warning(f"数据集结构验证失败: {validation_result['issues']}")
                else:
                    logger.info("数据集结构验证通过")

            return data

        except Exception as e:
            logger.error(f"加载数据集时出错: {e}")
            raise
    
    def load_multiple_datasets(self, file_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """
        加载多个数据集
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            数据集字典，键为文件名，值为DataFrame
        """
        datasets = {}
        
        for file_path in file_paths:
            try:
                dataset_name = os.path.basename(file_path).split('.')[0]
                datasets[dataset_name] = self.load_dataset(file_path)
                logger.info(f"成功加载数据集: {dataset_name}")
            except Exception as e:
                logger.error(f"加载数据集 {file_path} 时出错: {e}")
        
        return datasets
    
    def get_available_datasets(self) -> List[str]:
        """
        获取可用的数据集列表
        
        Returns:
            可用数据集文件路径列表
        """
        available_datasets = []
        
        if os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                if file.endswith(('.csv', '.xlsx', '.xls', '.json')):
                    available_datasets.append(os.path.join(self.data_dir, file))
        
        return available_datasets
    
    def validate_dataset_schema(self, data: pd.DataFrame) -> Dict:
        """
        验证数据集是否符合标准结构

        Args:
            data: 数据集DataFrame

        Returns:
            验证结果字典
        """
        validation_result = {
            'is_valid': True,
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_columns': [],
            'extra_columns': [],
            'data_type_issues': [],
            'value_range_issues': [],
            'issues': []
        }

        # 检查必需字段
        missing_columns = [col for col in self.standard_columns if col not in data.columns]
        validation_result['missing_columns'] = missing_columns
        if missing_columns:
            validation_result['issues'].append(f"缺少必需字段: {missing_columns}")
            validation_result['is_valid'] = False

        # 检查额外字段
        extra_columns = [col for col in data.columns if col not in self.standard_columns]
        validation_result['extra_columns'] = extra_columns
        if extra_columns:
            validation_result['issues'].append(f"发现额外字段: {extra_columns}")

        # 检查数据类型和值范围（仅对存在的字段）
        for col in self.standard_columns:
            if col in data.columns:
                # 检查数值范围
                if col in self.field_constraints:
                    constraint = self.field_constraints[col]
                    if 'min' in constraint and 'max' in constraint:
                        out_of_range = data[(data[col] < constraint['min']) | (data[col] > constraint['max'])]
                        if len(out_of_range) > 0:
                            validation_result['value_range_issues'].append(
                                f"{col}: {len(out_of_range)}条记录超出范围 [{constraint['min']}, {constraint['max']}]"
                            )

                    if 'values' in constraint:
                        invalid_values = data[~data[col].isin(constraint['values'])]
                        if len(invalid_values) > 0:
                            unique_invalid = invalid_values[col].unique()
                            validation_result['value_range_issues'].append(
                                f"{col}: 发现无效值 {unique_invalid.tolist()}"
                            )

        # 汇总所有问题
        if validation_result['value_range_issues']:
            validation_result['issues'].extend(validation_result['value_range_issues'])
            validation_result['is_valid'] = False

        return validation_result

    def validate_dataset(self, data: pd.DataFrame) -> Dict:
        """
        验证数据集质量

        Args:
            data: 数据集DataFrame

        Returns:
            验证结果字典
        """
        validation_result = {
            'is_valid': True,
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_rows': data.duplicated().sum(),
            'data_types': {col: str(dtype) for col, dtype in data.dtypes.to_dict().items()},
            'issues': []
        }

        # 检查缺失值
        missing_cols = [col for col, count in validation_result['missing_values'].items() if count > 0]
        if missing_cols:
            validation_result['issues'].append(f"发现缺失值的列: {missing_cols}")

        # 检查重复行
        if validation_result['duplicate_rows'] > 0:
            validation_result['issues'].append(f"发现 {validation_result['duplicate_rows']} 行重复数据")

        # 检查数据类型
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns

        validation_result['numeric_columns'] = list(numeric_cols)
        validation_result['categorical_columns'] = list(categorical_cols)

        # 如果有问题，标记为无效
        if validation_result['issues']:
            validation_result['is_valid'] = False

        return validation_result
    
    def get_dataset_info(self, data: pd.DataFrame) -> Dict:
        """
        获取数据集信息
        
        Args:
            data: 数据集DataFrame
            
        Returns:
            数据集信息字典
        """
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'data_types': {col: str(dtype) for col, dtype in data.dtypes.to_dict().items()},
            'memory_usage': data.memory_usage(deep=True).sum(),
            'numeric_columns': list(data.select_dtypes(include=['number']).columns),
            'categorical_columns': list(data.select_dtypes(include=['object']).columns),
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_rows': data.duplicated().sum()
        }
        
        # 数值列统计
        if info['numeric_columns']:
            numeric_stats = data[info['numeric_columns']].describe()
            info['numeric_statistics'] = numeric_stats.to_dict()
        
        # 分类列统计
        if info['categorical_columns']:
            categorical_stats = {}
            for col in info['categorical_columns']:
                categorical_stats[col] = {
                    'unique_values': data[col].nunique(),
                    'most_common': data[col].mode().iloc[0] if not data[col].mode().empty else None
                }
            info['categorical_statistics'] = categorical_stats
        
        return info 