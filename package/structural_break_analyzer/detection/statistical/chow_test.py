from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from ..base import DetectionMethod


class ChowTestDetector(DetectionMethod):
    """
    Chow Test detector for structural breaks using statsmodels.
    
    The Chow test is used to test whether the coefficients in two linear
    regressions on different data sets are equal. In the context of structural
    breaks, it tests whether there's a significant change in the relationship
    at a specific point.
    """

    def __init__(self,
                 significance_level: float = 0.05,
                 trend: str = 'ct',
                 search_method: str = 'grid',
                 min_segment_size: int = None,
                 test_multiple_points: bool = True,
                 **kwargs):
        """
        Initialize Chow Test detector.

        Args:
            significance_level: Significance level for break detection
            trend: Trend specification ('n', 'c', 'ct' for none, constant, constant+trend)
            search_method: Method to search for breaks ('grid', 'sequential')
            min_segment_size: Minimum segment size (defaults to 15% of data)
            test_multiple_points: Whether to test multiple potential break points
        """
        super().__init__(**kwargs)
        self.significance_level = significance_level
        self.trend = trend
        self.search_method = search_method
        self.min_segment_size = min_segment_size
        self.test_multiple_points = test_multiple_points

    def _preprocess(self, data: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, Dict]:
        """Preprocess data for Chow Test."""
        values = data[value_col].values
        
        # Handle missing values
        mask = ~np.isnan(values)
        clean_values = values[mask]
        
        if len(clean_values) < 20:  # Chow test needs sufficient data for two regressions
            raise ValueError("Chow test requires at least 20 non-NaN values for detection")
        
        metadata = {
            'n_observations': len(values),
            'n_clean': len(clean_values),
            'mean': float(np.mean(clean_values)),
            'std': float(np.std(clean_values)),
            'missing_indices': np.where(~mask)[0].tolist() if not mask.all() else []
        }
        
        return clean_values, metadata

    def _detect_breaks(self, data: np.ndarray) -> List[int]:
        """Detect breaks using Chow Test."""
        n = len(data)
        
        # Set minimum segment size
        if self.min_segment_size is None:
            min_size = max(10, int(0.15 * n))  # 15% of data or at least 10
        else:
            min_size = self.min_segment_size
        
        if self.search_method == 'grid':
            return self._grid_search_chow_test(data, min_size)
        elif self.search_method == 'sequential':
            return self._sequential_chow_test(data, min_size)
        else:
            raise ValueError(f"Unknown search method: {self.search_method}")

    def _grid_search_chow_test(self, data: np.ndarray, min_size: int) -> List[int]:
        """Grid search approach for Chow test."""
        n = len(data)
        breaks = []
        
        # Test potential break points
        test_points = range(min_size, n - min_size)
        chow_stats = []
        
        for break_point in test_points:
            try:
                chow_stat, p_value = self._single_chow_test(data, break_point)
                chow_stats.append((break_point, chow_stat, p_value))
            except Exception:
                # Skip problematic break points
                continue
        
        if not chow_stats:
            return []
        
        if self.test_multiple_points:
            # Select all significant breaks
            significant_breaks = [
                (bp, stat, p) for bp, stat, p in chow_stats
                if p < self.significance_level
            ]
            
            # Sort by F-statistic (descending) and select non-overlapping breaks
            significant_breaks.sort(key=lambda x: x[1], reverse=True)
            
            for bp, stat, p_val in significant_breaks:
                # Check if this break is too close to existing breaks
                if not breaks or all(abs(bp - existing_bp) >= min_size for existing_bp in breaks):
                    breaks.append(bp)
        else:
            # Select single most significant break
            best_break = min(chow_stats, key=lambda x: x[2])  # Minimum p-value
            if best_break[2] < self.significance_level:
                breaks.append(best_break[0])
        
        return sorted(breaks)

    def _sequential_chow_test(self, data: np.ndarray, min_size: int) -> List[int]:
        """Sequential approach for Chow test."""
        breaks = []
        segments_to_test = [(data, 0)]
        
        while segments_to_test:
            segment, offset = segments_to_test.pop(0)
            n_seg = len(segment)
            
            if n_seg < 2 * min_size:
                continue
            
            # Find best break in this segment
            best_break = None
            best_stat = 0
            best_p = 1.0
            
            for i in range(min_size, n_seg - min_size):
                try:
                    chow_stat, p_value = self._single_chow_test(segment, i)
                    if p_value < self.significance_level and chow_stat > best_stat:
                        best_break = i
                        best_stat = chow_stat
                        best_p = p_value
                except Exception:
                    continue
            
            if best_break is not None:
                actual_break = offset + best_break
                breaks.append(actual_break)
                
                # Add subsegments for further testing
                left_segment = segment[:best_break]
                right_segment = segment[best_break:]
                
                if len(left_segment) >= 2 * min_size:
                    segments_to_test.append((left_segment, offset))
                if len(right_segment) >= 2 * min_size:
                    segments_to_test.append((right_segment, offset + best_break))
        
        return sorted(breaks)

    def _single_chow_test(self, data: np.ndarray, break_point: int) -> Tuple[float, float]:
        """Perform single Chow test at specified break point."""
        n = len(data)
        
        # Create time trend
        t = np.arange(n)
        
        # Prepare regressors for full model
        X_full = self._create_regressors(t, n)
        
        # Prepare regressors for subsamples
        X1 = X_full[:break_point]
        X2 = X_full[break_point:]
        y1 = data[:break_point]
        y2 = data[break_point:]
        
        # Fit models
        try:
            # Full model
            model_full = sm.OLS(data, X_full).fit()
            rss_full = np.sum(model_full.resid ** 2)
            
            # Subsample models
            model1 = sm.OLS(y1, X1).fit()
            model2 = sm.OLS(y2, X2).fit()
            rss1 = np.sum(model1.resid ** 2)
            rss2 = np.sum(model2.resid ** 2)
            
            rss_restricted = rss1 + rss2
            
            # Calculate Chow F-statistic
            k = X_full.shape[1]  # Number of parameters
            n1, n2 = len(y1), len(y2)
            
            numerator = (rss_full - rss_restricted) / k
            denominator = rss_restricted / (n1 + n2 - 2 * k)
            
            if denominator == 0:
                return 0.0, 1.0
            
            f_stat = numerator / denominator
            
            # Calculate p-value
            p_value = 1 - stats.f.cdf(f_stat, k, n1 + n2 - 2 * k)
            
            return float(f_stat), float(p_value)
            
        except Exception as e:
            # Return non-significant result if calculation fails
            return 0.0, 1.0

    def _create_regressors(self, t: np.ndarray, n: int) -> np.ndarray:
        """Create regressor matrix based on trend specification."""
        if self.trend == 'n':  # No trend
            X = np.ones((n, 1))
        elif self.trend == 'c':  # Constant only
            X = np.ones((n, 1))
        elif self.trend == 'ct':  # Constant and trend
            X = np.column_stack([np.ones(n), t])
        else:
            raise ValueError(f"Unknown trend specification: {self.trend}")
        
        return X

    def _calculate_statistics(self,
                              data: np.ndarray,
                              breaks: List[int],
                              preprocess_meta: Dict) -> Dict[str, Any]:
        """Calculate Chow Test specific statistics."""
        base_stats = super()._calculate_statistics(data, breaks, preprocess_meta)
        
        # Add method-specific statistics
        stats = {
            **base_stats,
            'significance_level': self.significance_level,
            'trend_specification': self.trend,
            'search_method': self.search_method,
            'min_segment_size': self.min_segment_size,
            'test_multiple_points': self.test_multiple_points
        }
        
        # Calculate test statistics for detected breaks
        if breaks:
            break_statistics = []
            for break_idx in breaks:
                try:
                    f_stat, p_val = self._single_chow_test(data, break_idx)
                    break_statistics.append({
                        'break_index': break_idx,
                        'f_statistic': float(f_stat),
                        'p_value': float(p_val)
                    })
                except Exception:
                    break_statistics.append({
                        'break_index': break_idx,
                        'f_statistic': None,
                        'p_value': None
                    })
            
            stats['break_test_statistics'] = break_statistics
            
            # Calculate segment statistics
            segments = self._get_segments(data, breaks)
            stats['segment_means'] = [float(np.mean(segment)) for segment in segments]
            stats['segment_vars'] = [float(np.var(segment)) for segment in segments]
            stats['segment_lengths'] = [len(segment) for segment in segments]
        
        return stats

    def _create_breakpoints(self, indices: List[int], original_data: pd.DataFrame, 
                           processed_values: np.ndarray, value_col: str, date_col: str):
        """Create breakpoints with Chow Test confidence scores."""
        breaks = super()._create_breakpoints(indices, original_data, processed_values, value_col, date_col)
        
        # Use the already calculated F-statistics and p-values for confidence
        if breaks:
            for break_point in breaks:
                idx = break_point.index
                try:
                    # Recalculate the Chow test for this specific break point
                    f_stat, p_value = self._single_chow_test(processed_values, idx)
                    
                    # Confidence based on p-value (lower p-value = higher confidence)
                    confidence = max(0.05, min(0.95, 1.0 - p_value))
                    
                    break_point.confidence = confidence
                    break_point.statistics = {
                        'chow_f_statistic': f_stat,
                        'chow_p_value': p_value,
                        'significance_level': self.significance_level
                    }
                except Exception:
                    break_point.confidence = 0.1
                    break_point.statistics = {'error': 'Could not calculate Chow test statistics'}
        
        return breaks

    def _get_segments(self, data: np.ndarray, breaks: List[int]) -> List[np.ndarray]:
        """Split data into segments based on break points."""
        if not breaks:
            return [data]
        
        segments = []
        start = 0
        
        for break_idx in breaks:
            if break_idx > start:
                segments.append(data[start:break_idx])
            start = break_idx
        
        # Add final segment
        if start < len(data):
            segments.append(data[start:])
        
        return segments