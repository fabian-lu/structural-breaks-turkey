from typing import Optional, Dict, List
import json
import numpy as np
import pandas as pd
from ..timeseries.features import TimeSeriesFeatures
from .prompts import get_timeseries_explanation_prompt, get_specific_feature_prompt


class TimeSeriesExplainer:
    """Generate natural language explanations for time series characteristics."""
    
    def __init__(self, llm_provider):
        """
        Initialize explainer.
        
        Args:
            llm_provider: LLM provider instance
        """
        self.llm = llm_provider
        
    def explain_features(self,
                        features: TimeSeriesFeatures,
                        data_description: str = "time series",
                        context: str = "",
                        detail_level: str = "medium") -> str:
        """
        Generate comprehensive explanation of time series features.
        
        Args:
            features: Extracted time series features
            data_description: What the data represents
            context: Additional context
            detail_level: 'basic', 'medium', or 'detailed'
            
        Returns:
            Natural language explanation
        """
        # Build features summary and insights
        features_summary = self._build_features_summary(features)
        insights = self.generate_insights(features, data_description)
        
        # Get prompts from separated prompt module
        system_prompt, user_prompt = get_timeseries_explanation_prompt(
            features_summary=features_summary,
            insights=insights,
            data_description=data_description,
            context=context,
            detail_level=detail_level
        )
        
        response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=self._get_max_tokens(detail_level)
        )
        
        if response.error:
            return f"Unable to generate explanation: {response.error}"
            
        return response.content.strip()
    
    def explain_specific_feature(self,
                                features: TimeSeriesFeatures,
                                feature_name: str,
                                data_description: str = "time series",
                                context: str = "") -> str:
        """
        Explain a specific feature in detail.
        
        Args:
            features: Time series features
            feature_name: Which feature to explain (trend, seasonality, etc.)
            data_description: What the data represents
            context: Additional context
            
        Returns:
            Focused explanation of that feature
        """
        # Map feature names to feature data extraction
        feature_data_map = {
            'trend': {
                'strength': features.trend_strength,
                'slope': features.trend_slope,
                'direction': 'increasing' if features.trend_slope > 0 else 'decreasing',
                'acceleration': features.trend_acceleration
            },
            'seasonality': {
                'has_seasonality': features.has_seasonality,
                'primary_period': features.seasonal_period,
                'strength': features.seasonal_strength,
                'all_periods': features.seasonal_periods
            },
            'volatility': {
                'mean': features.volatility_mean,
                'std': features.volatility_std,
                'clustering': features.volatility_clustering,
                'persistence': features.volatility_persistence
            },
            'stationarity': {
                'is_stationary': features.is_stationary,
                'adf_pvalue': features.adf_pvalue,
                'differences_needed': features.num_differences_needed
            },
            'distribution': {
                'skewness': features.skewness,
                'kurtosis': features.kurtosis
            },
            'autocorrelation': {
                'significant_lags': features.significant_lags,
                'ljung_box_pvalue': features.ljung_box_pvalue,
                'acf_1': features.acf_1
            },
            'complexity': {
                'hurst_exponent': features.hurst_exponent,
                'entropy': features.entropy
            }
        }
        
        if feature_name not in feature_data_map:
            return f"Unknown feature: {feature_name}. Available: {', '.join(feature_data_map.keys())}"
        
        # Get prompts from separated prompt module
        system_prompt, user_prompt = get_specific_feature_prompt(
            feature_name=feature_name,
            feature_data=feature_data_map[feature_name],
            data_description=data_description,
            context=context
        )
        
        # Handle features that don't need LLM (simple stationarity/distribution have pre-built explanations)
        if feature_name in ['stationarity', 'distribution']:
            # These return complete explanations from the prompt module
            return user_prompt
        
        response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=200
        )
        
        if response.error:
            return f"Unable to explain {feature_name}: {response.error}"
            
        return response.content.strip()
    
    def generate_insights(self,
                         features: TimeSeriesFeatures,
                         data_description: str = "time series") -> List[str]:
        """
        Generate key insights and recommendations.
        
        Args:
            features: Time series features
            data_description: What the data represents
            
        Returns:
            List of key insights
        """
        insights = []
        
        # Trend insights
        if features.trend_strength > 0.7:
            direction = "upward" if features.trend_slope > 0 else "downward"
            insights.append(f"Strong {direction} trend detected (R²={features.trend_r2:.2f})")
        
        # Seasonality insights
        if features.has_seasonality:
            if len(features.seasonal_periods) > 1:
                insights.append(f"Multiple seasonal patterns: periods {features.seasonal_periods}")
            else:
                insights.append(f"Seasonal pattern with period {features.seasonal_period}")
        
        # Stationarity insights
        if not features.is_stationary:
            insights.append(f"Non-stationary series - requires {features.num_differences_needed} differences")
        
        # Volatility insights
        if features.volatility_clustering > 0.3:
            insights.append("Volatility clustering detected - periods of high/low volatility")
        
        # Memory insights
        if abs(features.hurst_exponent - 0.5) > 0.1:
            memory_type = "Long memory" if features.hurst_exponent > 0.5 else "Anti-persistent"
            insights.append(f"{memory_type} behavior (H={features.hurst_exponent:.2f})")
        
        # Autocorrelation insights
        if len(features.significant_lags) > 3:
            insights.append(f"Strong autocorrelation at {len(features.significant_lags)} lags")
        
        # Enhanced memory insights
        if abs(features.hurst_exponent - 0.5) > 0.1:
            memory_type = "Long memory" if features.hurst_exponent > 0.5 else "Anti-persistent"
            insights.append(f"{memory_type} behavior (H={features.hurst_exponent:.2f})")
        
        # Long memory analysis (multiple estimates)
        if features.hurst_rs != 0.5 or features.hurst_dfa != 0.5:
            estimates = []
            if features.hurst_rs != 0.5:
                estimates.append(f"R/S={features.hurst_rs:.2f}")
            if features.hurst_dfa != 0.5:
                estimates.append(f"DFA={features.hurst_dfa:.2f}")
            if estimates:
                insights.append(f"Multiple Hurst estimates: {', '.join(estimates)}")
        
        # Non-linearity test insights
        nonlin_tests = []
        if features.bds_test and features.bds_test.get('pvalue', 1) < 0.05:
            nonlin_tests.append('BDS')
        if features.mcleod_li_test and features.mcleod_li_test.get('pvalue', 1) < 0.05:
            nonlin_tests.append('McLeod-Li')
        if features.terasvirta_test and features.terasvirta_test.get('pvalue', 1) < 0.05:
            nonlin_tests.append('Terasvirta')
        if nonlin_tests:
            insights.append(f"Non-linearity detected: {', '.join(nonlin_tests)} tests significant")
        
        # Enhanced volatility clustering
        if features.volatility_clustering_summary:
            insights.append(f"Volatility: {features.volatility_clustering_summary}")
        
        # ARCH effects
        if features.arch_lm_pvalue and features.arch_lm_pvalue < 0.05:
            insights.append(f"ARCH effects present (p={features.arch_lm_pvalue:.3f})")
        
        # Nonlinear dependence insights
        if features.mutual_info_lags or features.distance_corr_lags:
            dep_parts = []
            if features.mutual_info_lags:
                dep_parts.append(f"MI lags {features.mutual_info_lags[:3]}")
            if features.distance_corr_lags:
                dep_parts.append(f"DistCorr lags {features.distance_corr_lags[:3]}")
            insights.append(f"Nonlinear dependence: {', '.join(dep_parts)}")
        
        # Calendar effects insights
        if features.weekend_vs_weekday_delta is not None and abs(features.weekend_vs_weekday_delta) > 0.1:
            direction = "higher" if features.weekend_vs_weekday_delta > 0 else "lower"
            insights.append(f"Weekend effect: {direction} values by {abs(features.weekend_vs_weekday_delta):.3f}")
        
        # Enhanced outlier insights
        total_outliers = 0
        if features.outlier_analysis and 'count' in features.outlier_analysis:
            total_outliers += features.outlier_analysis['count']
        if features.extreme_flags and 'count' in features.extreme_flags:
            total_outliers += features.extreme_flags['count']
        if total_outliers > 0:
            insights.append(f"Outliers detected: {total_outliers} points flagged")
        
        return insights
    
    def _assess_predictability(self, features: TimeSeriesFeatures) -> str:
        """Assess overall predictability of the series."""
        complexity_score = 0
        
        # High autocorrelation increases predictability
        if abs(features.acf_1) > 0.7:
            complexity_score -= 1
        
        # Strong trend increases predictability
        if features.trend_strength > 0.7:
            complexity_score -= 1
            
        # Strong seasonality increases predictability
        if features.seasonal_strength > 0.7:
            complexity_score -= 1
        
        # Non-stationarity decreases predictability
        if not features.is_stationary:
            complexity_score += 1
            
        # High volatility clustering decreases predictability
        if features.volatility_clustering > 0.5:
            complexity_score += 1
            
        # Non-linearity decreases predictability
        nonlin_count = 0
        for test in [features.bds_test, features.mcleod_li_test, features.terasvirta_test]:
            if test and test.get('pvalue', 1) < 0.05:
                nonlin_count += 1
        if nonlin_count >= 2:
            complexity_score += 2
        elif nonlin_count == 1:
            complexity_score += 1
        
        if complexity_score <= -2:
            return "Highly predictable"
        elif complexity_score == -1:
            return "Moderately predictable"
        elif complexity_score == 0:
            return "Mixed predictability"
        elif complexity_score == 1:
            return "Moderately complex"
        else:
            return "Highly complex/unpredictable"
    
    def _build_features_summary(self, features: TimeSeriesFeatures) -> Dict:
        """
        Build enhanced structured feature summary for prompt (with MVP features).
        
        Args:
            features: Time series features (enhanced with MVP analysis)
            
        Returns:
            Dictionary with structured feature information including new MVP features
        """
        
        # Use human-readable summary if available (from MVP)
        if features.analysis_summary_text:
            base_summary = {
                'mvp_analysis': True,
                'human_readable_summary': features.analysis_summary_text
            }
        else:
            base_summary = {'mvp_analysis': False}
        
        enhanced_summary = {
            **base_summary,
            'data_description': 'time series',
            'basic_statistics': {
                'mean': features.mean,
                'std': features.std,
                'skewness': features.skewness,
                'kurtosis': features.kurtosis,
                'coefficient_of_variation': features.std / abs(features.mean) if features.mean != 0 else float('inf'),
                'min': features.min,
                'max': features.max
            },
            'trend_analysis': {
                'has_trend': features.trend_strength > 0.3,
                'strength': features.trend_strength,
                'direction': 'increasing' if features.trend_slope > 0 else 'decreasing',
                'slope': features.trend_slope,
                'acceleration': features.trend_acceleration,
                'r_squared': features.trend_r2
            },
            'seasonality': {
                'detected': features.has_seasonality,
                'primary_period': features.seasonal_period,
                'strength': features.seasonal_strength,
                'multiple_periods': len(features.seasonal_periods) > 1,
                'all_periods': features.seasonal_periods[:5] if features.seasonal_periods else [],
                'dominant_frequencies': features.dominant_frequencies[:3] if features.dominant_frequencies else []
            },
            'volatility_profile': {
                'level': features.volatility_mean,
                'variability': features.volatility_std,
                'clustering_present': features.volatility_clustering > 0.1,
                'persistence': features.volatility_persistence,
                'classification': self._classify_volatility(features.volatility_mean),
                'clustering_summary': features.volatility_clustering_summary,
                'arch_lm_pvalue': features.arch_lm_pvalue
            },
            'stationarity_analysis': {
                'is_stationary': features.is_stationary,
                'adf_pvalue': features.adf_pvalue,
                'kpss_pvalue': features.kpss_pvalue,
                'differences_needed': features.num_differences_needed,
                'interpretation': 'stationary' if features.is_stationary else 'non-stationary'
            },
            'memory_and_dependence': {
                'hurst_exponent': features.hurst_exponent,
                'hurst_rs': features.hurst_rs,
                'hurst_dfa': features.hurst_dfa,
                'memory_type': self._classify_memory(features.hurst_exponent),
                'significant_autocorrelation_lags': len(features.significant_lags),
                'ljung_box_test_pvalue': features.ljung_box_pvalue,
                'serial_correlation_present': features.ljung_box_pvalue < 0.05 if features.ljung_box_pvalue is not None else False,
                'acf_1': features.acf_1,
                'pacf_1': features.pacf_1
            },
            'complexity_measures': {
                'sample_entropy': features.entropy,
                'lz_complexity': features.complexity,
                'predictability_assessment': self._assess_predictability(features)
            }
        }
        
        return enhanced_summary
    
    def _get_max_tokens(self, detail_level: str) -> int:
        """Get max tokens based on detail level."""
        tokens = {
            'basic': 150,
            'medium': 400,
            'detailed': 800
        }
        
        return tokens.get(detail_level, 400)
    
    def _classify_volatility(self, vol_level: float) -> str:
        """Classify volatility level."""
        if vol_level < 0.05:
            return 'low'
        elif vol_level < 0.15:
            return 'moderate'
        elif vol_level < 0.3:
            return 'high'
        else:
            return 'very high'
    
    def _classify_memory(self, hurst: float) -> str:
        """Classify memory type based on Hurst exponent."""
        if hurst < 0.4:
            return 'anti-persistent (mean-reverting)'
        elif hurst > 0.6:
            return 'persistent (trending)'
        else:
            return 'random walk (no memory)'