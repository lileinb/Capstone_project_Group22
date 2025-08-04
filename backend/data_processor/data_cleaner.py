"""
Data Cleaner
Data cleaning and preprocessing based on real e-commerce fraud datasets
Handles data quality issues for 16 standard fields
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
from datetime import datetime, timedelta
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Data cleaner based on real datasets"""

    def __init__(self):
        self.cleaning_log = []
        self.original_shape = None
        self.cleaned_shape = None

        # Standard field mapping (original name -> cleaned name)
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

        # Data type mapping
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
        Complete data cleaning process

        Args:
            data: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        if data is None or data.empty:
            logger.error("Input data is empty")
            return pd.DataFrame()

        self.original_shape = data.shape
        self.cleaning_log = []

        logger.info(f"Starting data cleaning, original data shape: {self.original_shape}")

        df = data.copy()

        # 1. Standardize column names
        df = self._standardize_column_names(df)

        # 2. Handle duplicate data
        df = self._handle_duplicates(df)

        # 3. Handle missing values
        df = self._handle_missing_values(df)

        # 4. Data type conversion
        df = self._convert_data_types(df)

        # 5. Handle outliers
        df = self._handle_outliers(df)

        # 6. Data validation
        df = self._validate_cleaned_data(df)

        self.cleaned_shape = df.shape
        logger.info(f"Data cleaning completed, cleaned data shape: {self.cleaned_shape}")
        logger.info(f"Cleaning log: {self.cleaning_log}")

        return df

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        # Rename columns using mapping table
        df_renamed = df.rename(columns=self.column_mapping)

        # Record renamed columns
        renamed_cols = [col for col in df.columns if col in self.column_mapping]
        if renamed_cols:
            self.cleaning_log.append(f"Renamed columns: {renamed_cols}")

        return df_renamed

    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate data"""
        initial_count = len(df)

        # Remove duplicates based on transaction_id (if exists)
        if 'transaction_id' in df.columns:
            df = df.drop_duplicates(subset=['transaction_id'], keep='first')
        else:
            df = df.drop_duplicates()

        removed_count = initial_count - len(df)
        if removed_count > 0:
            self.cleaning_log.append(f"Removed duplicate rows: {removed_count} records")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]

        if len(missing_cols) > 0:
            self.cleaning_log.append(f"Found missing values: {missing_cols.to_dict()}")

            # For critical fields, remove rows with missing values
            critical_fields = ['transaction_id', 'customer_id', 'transaction_amount', 'is_fraudulent']
            for field in critical_fields:
                if field in df.columns and df[field].isnull().any():
                    before_count = len(df)
                    df = df.dropna(subset=[field])
                    after_count = len(df)
                    self.cleaning_log.append(f"Removed rows with missing {field}: {before_count - after_count} records")

            # For non-critical fields, use reasonable default values for filling
            for col in df.columns:
                if df[col].isnull().any():
                    if col in ['customer_age']:
                        # Fill age with median
                        median_age = df[col].median()
                        df[col] = df[col].fillna(median_age)
                        self.cleaning_log.append(f"Filled {col} missing values with median {median_age}")
                    elif col in ['account_age_days']:
                        # Fill account age with mean
                        mean_days = df[col].mean()
                        df[col] = df[col].fillna(mean_days)
                        self.cleaning_log.append(f"Filled {col} missing values with mean {mean_days:.0f}")
                    elif df[col].dtype == 'object':
                        # Fill text fields with "unknown"
                        df[col] = df[col].fillna('unknown')
                        self.cleaning_log.append(f"Filled {col} missing values with 'unknown'")
                    else:
                        # Fill numeric fields with 0
                        df[col] = df[col].fillna(0)
                        self.cleaning_log.append(f"Filled {col} missing values with 0")

        return df

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types"""
        for col, target_dtype in self.dtype_mapping.items():
            if col in df.columns:
                try:
                    if target_dtype == 'category':
                        df[col] = df[col].astype('category')
                    elif target_dtype == 'int64':
                        # Ensure no decimal points, then convert to integer
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                    elif target_dtype == 'float64':
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                    elif target_dtype == 'object':
                        df[col] = df[col].astype('object')

                    self.cleaning_log.append(f"Converted {col} to {target_dtype} type")
                except Exception as e:
                    logger.warning(f"Failed to convert {col} type: {e}")

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers"""
        # Handle customer age outliers
        if 'customer_age' in df.columns:
            # Age should be within reasonable range
            before_count = len(df)
            df = df[(df['customer_age'] >= 0) & (df['customer_age'] <= 100)]
            after_count = len(df)
            if before_count != after_count:
                self.cleaning_log.append(f"Removed abnormal age records: {before_count - after_count} records")

        # Handle transaction amount outliers
        if 'transaction_amount' in df.columns:
            # Remove negative amounts
            before_count = len(df)
            df = df[df['transaction_amount'] > 0]
            after_count = len(df)
            if before_count != after_count:
                self.cleaning_log.append(f"Removed negative amount records: {before_count - after_count} records")

            # Handle extreme outliers (using IQR method)
            Q1 = df['transaction_amount'].quantile(0.25)
            Q3 = df['transaction_amount'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Use 3x IQR as extreme outlier threshold
            upper_bound = Q3 + 3 * IQR

            before_count = len(df)
            df = df[(df['transaction_amount'] >= max(0, lower_bound)) &
                   (df['transaction_amount'] <= upper_bound)]
            after_count = len(df)
            if before_count != after_count:
                self.cleaning_log.append(f"Removed extreme amount outliers: {before_count - after_count} records")

        # Handle quantity outliers
        if 'quantity' in df.columns:
            before_count = len(df)
            df = df[(df['quantity'] >= 1) & (df['quantity'] <= 10)]
            after_count = len(df)
            if before_count != after_count:
                self.cleaning_log.append(f"Removed abnormal quantity records: {before_count - after_count} records")

        return df

    def _validate_cleaned_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate cleaned data"""
        # Check if critical fields exist
        required_fields = ['transaction_id', 'customer_id', 'transaction_amount', 'is_fraudulent']
        missing_required = [field for field in required_fields if field not in df.columns]
        if missing_required:
            logger.warning(f"Still missing critical fields after cleaning: {missing_required}")

        # Check data integrity
        if len(df) == 0:
            logger.error("Data is empty after cleaning")
        else:
            # Check fraud label distribution
            if 'is_fraudulent' in df.columns:
                fraud_rate = df['is_fraudulent'].mean()
                self.cleaning_log.append(f"Fraud rate: {fraud_rate:.3f}")

        return df

    def get_cleaning_summary(self) -> Dict:
        """Get cleaning summary"""
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.cleaned_shape,
            'cleaning_log': self.cleaning_log,
            'data_reduction': {
                'rows_removed': self.original_shape[0] - self.cleaned_shape[0] if self.original_shape and self.cleaned_shape else 0,
                'columns_changed': len(self.column_mapping)
            }
        }