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

        # 基于风险评分生成标签的阈值 (降低阈值以增加欺诈检出率)
        self.risk_thresholds = {
            'critical': 70,    # 极高风险阈值 (从80降到70)
            'high': 45,        # 高风险阈值 (从60降到45)
            'medium': 30,      # 中风险阈值 (从40降到30)
            'low': 15          # 低风险阈值 (从20降到15)
        }
        
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

            # 调试信息
            max_conf = max(confidences) if confidences else 0
            avg_conf = np.mean(confidences) if confidences else 0
            logger.info(f"Confidence stats: max={max_conf:.3f}, avg={avg_conf:.3f}, threshold={min_confidence}")
            logger.info(f"High quality filtering: {len(high_quality_labels)}/{len(labels)} samples passed threshold")

            # 4. Generate simplified quality report
            quality_report = self._generate_fast_quality_report(
                labels, confidences, high_quality_indices
            )
            
            result = {
                'success': True,
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
            import traceback
            logger.error(f"Fast pseudo label generation failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._empty_result()

    def _get_cached_risk_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get cached risk scoring results with proper dependency checking"""
        # 检查session state中是否有现成的风险评分结果
        try:
            import streamlit as st

            # 首先检查是否有四分类风险评分结果（优先使用）
            if (hasattr(st, 'session_state') and
                hasattr(st.session_state, 'four_class_risk_results') and
                st.session_state.four_class_risk_results is not None):

                risk_results = st.session_state.four_class_risk_results
                if risk_results.get('success') and 'detailed_results' in risk_results:
                    logger.info("✅ Using cached four-class risk scoring results")
                    return self._convert_four_class_to_risk_format(risk_results)

            # 检查无监督风险评分结果
            if (hasattr(st.session_state, 'unsupervised_risk_results') and
                st.session_state.unsupervised_risk_results is not None):

                risk_results = st.session_state.unsupervised_risk_results
                if risk_results and 'results' in risk_results:
                    logger.info("✅ Using cached unsupervised risk scoring results")
                    return risk_results

            # 如果没有任何风险评分结果，抛出依赖错误
            logger.error("❌ 快速模式需要先完成风险评分步骤")
            raise ValueError("快速伪标签生成需要先完成风险评分。请先在'🎯 Risk Scoring'页面完成风险评分。")

        except ValueError:
            # 重新抛出依赖错误
            raise
        except Exception as e:
            logger.error(f"检查缓存风险结果时出错: {e}")
            raise ValueError("无法获取风险评分结果，请先完成风险评分步骤。")

    def _convert_four_class_to_risk_format(self, four_class_results: Dict) -> Dict[str, Any]:
        """将四分类风险结果转换为标准风险评分格式"""
        try:
            detailed_results = four_class_results.get('detailed_results', [])
            converted_results = []

            for result in detailed_results:
                # 将四分类风险等级转换为风险评分
                risk_level = result.get('risk_level', 'low')
                risk_score_mapping = {
                    'low': 25,
                    'medium': 50,
                    'high': 75,
                    'critical': 90
                }

                converted_result = {
                    'transaction_id': result.get('transaction_id', ''),
                    'risk_score': result.get('risk_score', risk_score_mapping.get(risk_level, 25)),
                    'risk_level': risk_level,
                    'confidence': result.get('confidence', 0.7)
                }
                converted_results.append(converted_result)

            return {
                'results': converted_results,
                'summary': {
                    'total_samples': len(converted_results),
                    'source': 'four_class_risk_scoring'
                }
            }

        except Exception as e:
            logger.error(f"四分类结果转换失败: {e}")
            raise ValueError("风险评分结果格式转换失败")

    # 移除了 _calculate_fast_risk_scores 方法
    # 快速模式现在强制依赖前置的风险评分步骤
    # 这确保了数据流程的一致性和质量
    
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
            # 确保score是标量值
            score = float(score) if hasattr(score, '__iter__') and not isinstance(score, str) else score

            if level == 'critical':
                # 极高风险：确定是欺诈
                labels[i] = 1
                confidences[i] = min(0.95, 0.88 + (score - 75) / 100)
            elif level == 'high':
                # 高风险：大部分标记为欺诈
                labels[i] = 1
                confidences[i] = min(0.92, 0.75 + (score - 55) / 80)  # 提高置信度
            elif level == 'medium':
                # 中风险：部分标记为欺诈，基于评分
                if score >= 45:  # 中风险中的高分
                    labels[i] = 1
                    confidences[i] = min(0.85, 0.70 + (score - 45) / 60)  # 提高置信度
                else:
                    labels[i] = 0
                    confidences[i] = min(0.88, 0.75 + (45 - score) / 80)  # 提高置信度
            else:  # low
                # 低风险：少量高分标记为欺诈
                if score >= 30:  # 低风险中的异常高分
                    labels[i] = 1
                    confidences[i] = min(0.82, 0.65 + (score - 30) / 100)  # 提高置信度
                else:
                    labels[i] = 0
                    confidences[i] = min(0.90, 0.80 + (30 - score) / 150)  # 提高置信度
        
        # 添加一些随机性以避免过度确定性
        noise = np.random.normal(0, 0.02, len(confidences))
        confidences = np.clip(confidences + noise, 0.1, 0.95)

        # 调试信息
        fraud_count = sum(labels)
        fraud_rate = fraud_count / len(labels) * 100 if len(labels) > 0 else 0
        logger.info(f"Generated {fraud_count} fraud labels out of {len(labels)} total ({fraud_rate:.2f}%)")

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
        
        # 标签平衡性评估 (调整期望的欺诈率范围)
        if 0.03 <= fraud_rate <= 0.20:  # 扩大合理范围
            balance_bonus = 15
        elif 0.01 <= fraud_rate <= 0.30:  # 更宽松的范围
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
            'success': False,
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
