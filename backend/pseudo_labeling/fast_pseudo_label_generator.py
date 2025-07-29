"""
Fast Pseudo Label Generator
Use vectorized operations and caching mechanisms to significantly improve generation speed
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class FastPseudoLabelGenerator:
    """Fast Pseudo Label Generator"""

    def __init__(self):
        """Initialize fast pseudo label generator"""
        self.confidence_threshold = 0.7
        self.cache = {}  # Caching mechanism
        
    def generate_fast_pseudo_labels(self, data: pd.DataFrame,
                                  risk_results: Optional[Dict] = None,
                                  min_confidence: float = 0.8) -> Dict[str, Any]:
        """
        Fast pseudo label generation

        Args:
            data: Input data
            risk_results: Risk scoring results (if available)
            min_confidence: Minimum confidence threshold

        Returns:
            Pseudo label generation results
        """
        try:
            if data is None or data.empty:
                logger.error("Input data is empty")
                return self._empty_result()

            start_time = time.time()
            logger.info(f"Starting fast pseudo label generation, data size: {len(data)}")

            # 1. Use existing risk scoring results or fast calculation
            if risk_results is None:
                risk_results = self._get_cached_risk_results(data)

            # 2. Vectorized pseudo label generation
            labels, confidences = self._generate_labels_vectorized(data, risk_results)

            # 3. Filter high-quality labels
            high_quality_indices = np.where(np.array(confidences) >= min_confidence)[0].tolist()
            high_quality_labels = [labels[i] for i in high_quality_indices]
            high_quality_confidences = [confidences[i] for i in high_quality_indices]

            # 4. Generate simplified quality report
            quality_report = self._generate_fast_quality_report(
                labels, confidences, high_quality_indices
            )
            
            result = {
                'strategy': 'fast_generation',
                'all_labels': labels,
                'all_confidences': confidences,
                'high_quality_indices': high_quality_indices,
                'high_quality_labels': high_quality_labels,
                'high_quality_confidences': high_quality_confidences,
                'quality_report': quality_report,
                'min_confidence_threshold': min_confidence,
                'metadata': {
                    'total_samples': len(labels),
                    'high_quality_count': len(high_quality_labels),
                    'high_quality_rate': len(high_quality_labels) / len(labels) if labels else 0,
                    'avg_confidence_all': float(np.mean(confidences)) if confidences else 0,
                    'avg_confidence_hq': float(np.mean(high_quality_confidences)) if high_quality_confidences else 0,
                    'fraud_rate_all': float(np.mean(labels)) if labels else 0,
                    'fraud_rate_hq': float(np.mean(high_quality_labels)) if high_quality_labels else 0,
                    'generation_time': time.time() - start_time
                }
            }
            
            logger.info(f"Fast pseudo label generation completed, time taken: {time.time() - start_time:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Fast pseudo label generation failed: {e}")
            return self._empty_result()

    def _get_cached_risk_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get cached risk scoring results or fast calculation"""
        # 检查session state中是否有现成的风险评分结果
        try:
            import streamlit as st
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'unsupervised_risk_results'):
                risk_results = st.session_state.unsupervised_risk_results
                if risk_results and 'results' in risk_results:
                    logger.info("Using cached risk scoring results")
                    return risk_results
        except:
            pass

        # If no cache, use fast risk scoring
        logger.info("Using fast risk scoring")
        return self._calculate_fast_risk_scores(data)

    def _calculate_fast_risk_scores(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fast risk score calculation"""
        # 向量化计算风险评分
        amounts = data.get('transaction_amount', pd.Series([0] * len(data))).values
        hours = data.get('transaction_hour', pd.Series([12] * len(data))).values
        account_ages = data.get('account_age_days', pd.Series([365] * len(data))).values
        customer_ages = data.get('customer_age', pd.Series([30] * len(data))).values
        
        # 向量化评分计算
        amount_scores = self._calculate_amount_scores_vectorized(amounts)
        time_scores = self._calculate_time_scores_vectorized(hours)
        account_scores = self._calculate_account_scores_vectorized(account_ages)
        age_scores = self._calculate_age_scores_vectorized(customer_ages)
        
        # 综合评分
        risk_scores = (
            amount_scores * 0.4 +
            time_scores * 0.25 +
            account_scores * 0.25 +
            age_scores * 0.1
        )
        
        # 生成风险等级
        risk_levels = []
        for score in risk_scores:
            if score >= 75:
                risk_levels.append('critical')
            elif score >= 55:
                risk_levels.append('high')
            elif score >= 35:
                risk_levels.append('medium')
            else:
                risk_levels.append('low')
        
        # 构造结果格式
        results = []
        for i, (score, level) in enumerate(zip(risk_scores, risk_levels)):
            results.append({
                'transaction_id': data.iloc[i].get('transaction_id', f'tx_{i}'),
                'customer_id': data.iloc[i].get('customer_id', f'customer_{i}'),
                'risk_score': float(score),
                'risk_level': level,
                'risk_factors': []
            })
        
        return {
            'results': results,
            'total_transactions': len(results),
            'average_risk_score': float(np.mean(risk_scores)),
            'risk_distribution': {level: risk_levels.count(level) for level in ['low', 'medium', 'high', 'critical']}
        }
    
    def _calculate_amount_scores_vectorized(self, amounts: np.ndarray) -> np.ndarray:
        """Vectorized amount score calculation"""
        scores = np.zeros_like(amounts, dtype=float)
        scores[amounts > 50000] = 90
        scores[(amounts > 20000) & (amounts <= 50000)] = 75
        scores[(amounts > 5000) & (amounts <= 20000)] = 55
        scores[(amounts > 1000) & (amounts <= 5000)] = 35
        scores[(amounts > 100) & (amounts <= 1000)] = 20
        scores[(amounts > 0) & (amounts <= 100)] = 10
        scores[amounts < 10] = 70  # Abnormally small transactions
        return scores
    
    def _calculate_time_scores_vectorized(self, hours: np.ndarray) -> np.ndarray:
        """Vectorized time score calculation"""
        scores = np.full_like(hours, 20.0, dtype=float)  # Default score
        scores[(hours <= 2) | (hours >= 23)] = 80  # Late night
        scores[((hours <= 5) & (hours > 2)) | ((hours >= 22) & (hours < 23))] = 60  # Night
        scores[(hours >= 9) & (hours <= 17)] = 10  # Business hours
        return scores

    def _calculate_account_scores_vectorized(self, account_ages: np.ndarray) -> np.ndarray:
        """Vectorized account score calculation"""
        scores = np.zeros_like(account_ages, dtype=float)
        scores[account_ages < 1] = 80
        scores[(account_ages >= 1) & (account_ages < 7)] = 65
        scores[(account_ages >= 7) & (account_ages < 30)] = 45
        scores[(account_ages >= 30) & (account_ages < 90)] = 25
        scores[account_ages >= 90] = 10
        return scores

    def _calculate_age_scores_vectorized(self, customer_ages: np.ndarray) -> np.ndarray:
        """Vectorized customer age score calculation"""
        scores = np.full_like(customer_ages, 10.0, dtype=float)  # Default score
        scores[(customer_ages < 18) | (customer_ages > 80)] = 40  # Abnormal age
        scores[(customer_ages < 21) | (customer_ages > 70)] = 25  # Boundary age
        return scores
    
    def _generate_labels_vectorized(self, data: pd.DataFrame,
                                  risk_results: Dict[str, Any]) -> Tuple[List[int], List[float]]:
        """Vectorized pseudo label generation"""
        results = risk_results.get('results', [])
        if not results:
            logger.warning("Risk scoring results are empty")
            return [], []
        
        # 提取风险评分和等级
        risk_scores = np.array([r.get('risk_score', 0) for r in results])
        risk_levels = [r.get('risk_level', 'low') for r in results]
        
        # 向量化生成标签和置信度
        labels = np.zeros(len(risk_scores), dtype=int)
        confidences = np.zeros(len(risk_scores), dtype=float)
        
        for i, (score, level) in enumerate(zip(risk_scores, risk_levels)):
            if level == 'critical':
                labels[i] = 1
                confidences[i] = min(0.95, 0.85 + (score - 75) / 100)
            elif level == 'high':
                labels[i] = 1
                confidences[i] = min(0.90, 0.70 + (score - 55) / 100)
            elif level == 'medium':
                if score >= 50:
                    labels[i] = 1
                    confidences[i] = 0.60 + (score - 50) / 100
                else:
                    labels[i] = 0
                    confidences[i] = 0.55 + (50 - score) / 100
            else:  # low
                labels[i] = 0
                confidences[i] = min(0.85, 0.65 + (35 - score) / 100)
        
        # 添加一些随机性以避免过度确定性
        noise = np.random.normal(0, 0.02, len(confidences))
        confidences = np.clip(confidences + noise, 0.1, 0.95)
        
        return labels.tolist(), confidences.tolist()
    
    def _generate_fast_quality_report(self, labels: List[int], confidences: List[float],
                                    high_quality_indices: List[int]) -> Dict[str, Any]:
        """Generate fast quality report"""
        if not labels or not confidences:
            return {'quality_score': 0, 'quality_level': 'poor'}
        
        # 计算基础质量指标
        avg_confidence = np.mean(confidences)
        hq_ratio = len(high_quality_indices) / len(labels)
        fraud_rate = np.mean(labels)
        
        # 计算质量评分
        base_score = avg_confidence * 100
        hq_bonus = hq_ratio * 20
        
        # 标签平衡性评估
        if 0.05 <= fraud_rate <= 0.15:
            balance_bonus = 15
        elif 0.02 <= fraud_rate <= 0.25:
            balance_bonus = 8
        else:
            balance_bonus = 0
        
        quality_score = min(100, base_score + hq_bonus + balance_bonus)
        
        # 确定质量等级
        if quality_score >= 85:
            quality_level = 'excellent'
        elif quality_score >= 75:
            quality_level = 'good'
        elif quality_score >= 60:
            quality_level = 'fair'
        else:
            quality_level = 'poor'
        
        return {
            'quality_score': round(quality_score, 1),
            'quality_level': quality_level,
            'avg_confidence': round(avg_confidence, 3),
            'high_quality_ratio': round(hq_ratio, 3),
            'fraud_rate': round(fraud_rate, 3),
            'balance_score': balance_bonus
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            'strategy': 'fast_generation',
            'all_labels': [],
            'all_confidences': [],
            'high_quality_indices': [],
            'high_quality_labels': [],
            'high_quality_confidences': [],
            'quality_report': {'quality_score': 0, 'quality_level': 'poor'},
            'metadata': {
                'total_samples': 0,
                'high_quality_count': 0,
                'high_quality_rate': 0,
                'avg_confidence_all': 0,
                'avg_confidence_hq': 0,
                'fraud_rate_all': 0,
                'fraud_rate_hq': 0
            },
            'error': 'No data to process'
        }
