"""
系统优化配置管理
统一管理聚类和风险评分的所有配置参数
"""

import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OptimizationConfig:
    """优化配置管理器"""
    
    def __init__(self, config_file: str = "config/optimization_settings.json"):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_default_config()
        self._load_config_file()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            "clustering": {
                "auto_k_optimization": True,
                "max_k": 10,
                "min_k": 2,
                "dbscan_auto_params": True,
                "feature_selection": {
                    "enabled": True,
                    "variance_threshold": 0.01,
                    "correlation_threshold": 0.95,
                    "min_features": 3
                },
                "data_quality": {
                    "outlier_method": "iqr",
                    "outlier_factor": 1.5,
                    "missing_value_threshold": 0.5,
                    "quality_threshold": 70.0
                }
            },
            "risk_scoring": {
                "dynamic_thresholds": {
                    "enabled": True,
                    "target_distribution": {
                        "low": 0.60,      # 60% 低风险
                        "medium": 0.25,   # 25% 中风险
                        "high": 0.12,     # 12% 高风险
                        "critical": 0.03  # 3% 极高风险
                    },
                    "max_iterations": 5,
                    "convergence_threshold": 0.05,
                    "default_thresholds": {
                        "low": 40,
                        "medium": 60,
                        "high": 80,
                        "critical": 100
                    }
                },
                "label_generation": {
                    "use_original_labels": True,
                    "original_label_weight": 0.6,
                    "clustering_weight": 0.25,
                    "rule_weight": 0.15
                },
                "model_architecture": {
                    "catboost_weight": 0.6,
                    "xgboost_weight": 0.4,
                    "dynamic_weights": True,
                    "class_weights": [0.6, 0.25, 0.12, 0.03]
                },
                "weights": {
                    "cluster_anomaly_score": 0.25,
                    "feature_deviation_score": 0.30,
                    "business_rule_score": 0.25,
                    "statistical_outlier_score": 0.15,
                    "pattern_consistency_score": 0.05
                },
                "default_thresholds": {
                    "low": 40,
                    "medium": 60,
                    "high": 80,
                    "critical": 100
                }
            },
            "performance": {
                "large_dataset_threshold": 10000,
                "enable_caching": True,
                "parallel_processing": False,
                "memory_optimization": True
            },
            "monitoring": {
                "enable_quality_monitoring": True,
                "enable_distribution_monitoring": True,
                "alert_thresholds": {
                    "data_quality_min": 60.0,
                    "distribution_deviation_max": 0.3,
                    "processing_time_max": 300.0
                }
            },
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat()
        }
    
    def _load_config_file(self) -> None:
        """从文件加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                
                # 合并配置（文件配置覆盖默认配置）
                self._merge_config(self.config, file_config)
                logger.info(f"配置文件加载成功: {self.config_file}")
            else:
                # 创建默认配置文件
                self.save_config()
                logger.info(f"创建默认配置文件: {self.config_file}")
                
        except Exception as e:
            logger.warning(f"配置文件加载失败: {e}，使用默认配置")
    
    def _merge_config(self, default: Dict, override: Dict) -> None:
        """递归合并配置"""
        for key, value in override.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key_path: 配置路径，如 'clustering.auto_k_optimization'
            default: 默认值
            
        Returns:
            配置值
        """
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                value = value[key]
            
            return value
            
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key_path: 配置路径
            value: 配置值
        """
        try:
            keys = key_path.split('.')
            config = self.config
            
            # 导航到父级
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            # 设置值
            config[keys[-1]] = value
            
            # 更新时间戳
            self.config['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"配置更新: {key_path} = {value}")
            
        except Exception as e:
            logger.error(f"配置设置失败: {e}")
    
    def save_config(self) -> bool:
        """保存配置到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置保存成功: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
            return False
    
    def get_clustering_config(self) -> Dict[str, Any]:
        """获取聚类配置"""
        return self.config.get('clustering', {})
    
    def get_risk_scoring_config(self) -> Dict[str, Any]:
        """获取风险评分配置"""
        return self.config.get('risk_scoring', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """获取性能配置"""
        return self.config.get('performance', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """获取监控配置"""
        return self.config.get('monitoring', {})
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """更新风险评分权重"""
        try:
            # 验证权重总和
            total_weight = sum(new_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"权重总和不为1: {total_weight}")
                # 归一化权重
                new_weights = {k: v/total_weight for k, v in new_weights.items()}
            
            self.config['risk_scoring']['weights'] = new_weights
            self.config['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"权重更新完成: {new_weights}")
            
        except Exception as e:
            logger.error(f"权重更新失败: {e}")
    
    def update_target_distribution(self, new_distribution: Dict[str, float]) -> None:
        """更新目标风险分布"""
        try:
            # 验证分布总和
            total_dist = sum(new_distribution.values())
            if abs(total_dist - 1.0) > 0.01:
                logger.warning(f"分布总和不为1: {total_dist}")
                # 归一化分布
                new_distribution = {k: v/total_dist for k, v in new_distribution.items()}
            
            self.config['risk_scoring']['dynamic_thresholds']['target_distribution'] = new_distribution
            self.config['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"目标分布更新完成: {new_distribution}")
            
        except Exception as e:
            logger.error(f"目标分布更新失败: {e}")
    
    def reset_to_defaults(self) -> None:
        """重置为默认配置"""
        self.config = self._load_default_config()
        logger.info("配置已重置为默认值")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'version': self.config.get('version', 'unknown'),
            'last_updated': self.config.get('last_updated', 'unknown'),
            'clustering_enabled_features': {
                'auto_k_optimization': self.get('clustering.auto_k_optimization', False),
                'dbscan_auto_params': self.get('clustering.dbscan_auto_params', False),
                'feature_selection': self.get('clustering.feature_selection.enabled', False)
            },
            'risk_scoring_enabled_features': {
                'dynamic_thresholds': self.get('risk_scoring.dynamic_thresholds.enabled', False),
                'quality_monitoring': self.get('monitoring.enable_quality_monitoring', False)
            },
            'performance_settings': {
                'large_dataset_threshold': self.get('performance.large_dataset_threshold', 10000),
                'caching_enabled': self.get('performance.enable_caching', False)
            }
        }


# 全局配置实例
optimization_config = OptimizationConfig()
