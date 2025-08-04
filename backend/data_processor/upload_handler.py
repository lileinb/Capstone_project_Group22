"""
数据上传处理器
负责数据上传、格式验证和基础处理
"""

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
from typing import Optional, Dict, Any, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataUploadHandler:
    """数据上传处理器"""
    
    def __init__(self):
        self.supported_formats = ['.csv']
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.uploaded_data = None
        self.data_info = {}
        
    def validate_file_format(self, file) -> bool:
        """验证文件格式"""
        if file is None:
            return False
            
        file_extension = Path(file.name).suffix.lower()
        return file_extension in self.supported_formats
    
    def validate_file_size(self, file) -> bool:
        """验证文件大小"""
        if file is None:
            return False
            
        return file.size <= self.max_file_size
    
    def load_csv_data(self, file) -> Optional[pd.DataFrame]:
        """Load CSV data"""
        try:
            if file is None:
                return None

            # Try different encoding methods
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            data = None

            for encoding in encodings:
                try:
                    file.seek(0)  # Reset file pointer
                    data = pd.read_csv(file, encoding=encoding)
                    logger.info(f"Successfully read file using {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Failed to read using {encoding} encoding: {e}")
                    continue

            if data is None:
                st.error("Unable to read file, please check file format and encoding")
                return None

            return data

        except Exception as e:
            logger.error(f"Error occurred while loading data: {e}")
            st.error(f"Error occurred while loading data: {e}")
            return None
    
    def get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取数据基本信息"""
        if data is None or data.empty:
            return {}
            
        info = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "null_counts": data.isnull().sum().to_dict(),
            "duplicate_count": data.duplicated().sum(),
            "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": data.select_dtypes(include=['datetime']).columns.tolist()
        }
        
        return info
    
    def validate_data_structure(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Validate data structure - check if it matches the system-supported dataset format"""
        if data is None or data.empty:
            return False, "Data is empty"

        # System-supported standard fields (based on existing datasets)
        standard_columns = [
            'Transaction ID', 'Customer ID', 'Transaction Amount', 'Transaction Date',
            'Payment Method', 'Product Category', 'Quantity', 'Customer Age',
            'Customer Location', 'Device Used', 'IP Address',
            'Shipping Address', 'Billing Address', 'Is Fraudulent',
            'Account Age Days', 'Transaction Hour'
        ]

        # Check if core fields are included (at least 80% of standard fields)
        matching_columns = [col for col in standard_columns if col in data.columns]
        match_ratio = len(matching_columns) / len(standard_columns)

        if match_ratio < 0.8:
            return False, f"Data format mismatch. Current match ratio: {match_ratio:.1%}. Please ensure dataset contains the following fields: {standard_columns}"

        return True, f"Data structure validation passed, field match ratio: {match_ratio:.1%}"
    
    def upload_data(self, file) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """Main function for uploading data"""
        try:
            # 验证文件格式
            if not self.validate_file_format(file):
                return False, "Unsupported file format, please upload CSV file", None

            # 验证文件大小
            if not self.validate_file_size(file):
                return False, f"File size exceeds limit ({self.max_file_size / 1024 / 1024:.1f}MB)", None

            # 加载数据
            data = self.load_csv_data(file)
            if data is None:
                return False, "Data loading failed", None

            # 验证数据结构
            is_valid, message = self.validate_data_structure(data)
            if not is_valid:
                return False, message, None

            # 获取数据信息
            self.data_info = self.get_data_info(data)
            self.uploaded_data = data

            logger.info(f"Data upload successful: {data.shape}")
            return True, "Data upload successful", data

        except Exception as e:
            logger.error(f"Error occurred during data upload: {e}")
            return False, f"Data upload failed: {e}", None
    
    def load_sample_data(self, dataset_name: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """加载示例数据"""
        try:
            from config.settings import DATASET_CONFIG
            
            if dataset_name not in DATASET_CONFIG:
                return False, f"未知的数据集: {dataset_name}", None
                
            dataset_path = DATASET_CONFIG[dataset_name]["path"]
            
            if not Path(dataset_path).exists():
                return False, f"数据集文件不存在: {dataset_path}", None
                
            # 加载数据
            data = pd.read_csv(dataset_path)
            
            # 获取数据信息
            self.data_info = self.get_data_info(data)
            self.uploaded_data = data
            
            logger.info(f"示例数据加载成功: {data.shape}")
            return True, f"示例数据加载成功: {dataset_name}", data
            
        except Exception as e:
            logger.error(f"加载示例数据时发生错误: {e}")
            return False, f"加载示例数据失败: {e}", None
    
    def get_data_preview(self, data: pd.DataFrame, n_rows: int = 10) -> pd.DataFrame:
        """获取数据预览"""
        if data is None or data.empty:
            return pd.DataFrame()
            
        return data.head(n_rows)
    
    def get_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取基础统计信息"""
        if data is None or data.empty:
            return {}
            
        stats = {
            "numeric_stats": data.describe().to_dict(),
            "null_percentage": (data.isnull().sum() / len(data) * 100).to_dict(),
            "unique_counts": data.nunique().to_dict(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return stats
    
    def export_data_info(self) -> Dict[str, Any]:
        """导出数据信息"""
        return {
            "data_info": self.data_info,
            "uploaded_data_shape": self.uploaded_data.shape if self.uploaded_data is not None else None,
            "file_size_mb": self.uploaded_data.memory_usage(deep=True).sum() / 1024 / 1024 if self.uploaded_data is not None else 0
        } 