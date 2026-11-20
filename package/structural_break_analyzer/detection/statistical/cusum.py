from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from ..base import DetectionMethod


class CUSUMDetector(DetectionMethod):
    """
    CUSUM (Cumulative Sum) detector using statsmodels.
    
    CUSUM is a sequential analysis technique typically used for
    detecting changes in the mean of a time series. This implementation
    uses OLS residuals to detect structural breaks.
    """

    def __init__(self,
                 significance_level: float = 0.05,
                 trend: str = 'c',
                 use_recursive: bool = False,
                 **kwargs):
        """
        Initialize CUSUM detector.

        Args:
            significance_level: Significance level for break detection
            trend: Trend specification ('n', 'c', 'ct' for none, constant, constant+trend)
            use_recursive: Whether to use recursive CUSUM test
        """
        super().__init__(**kwargs)
        self.significance_level = significance_level
        self.trend = trend
        self.use_recursive = use_recursive

    def _preprocess(self, data: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, Dict]:
        """Preprocess data for CUSUM."""
        values = data[value_col].values
        
        # Handle missing values
        mask = ~np.isnan(values)
        clean_values = values[mask]
        
        if len(clean_values) < 15:  # CUSUM needs more points for reliable results
            raise ValueError("CUSUM requires at least 15 non-NaN values for detection")
        
        metadata = {
            'n_observations': len(values),
            'n_clean': len(clean_values),
            'mean': float(np.mean(clean_values)),
            'std': float(np.std(clean_values)),
            'missing_indices': np.where(~mask)[0].tolist() if not mask.all() else []
        }
        
        return clean_values, metadata

    def _detect_breaks(self, data: np.ndarray) -> List[int]:
        """Detect breaks using CUSUM test."""
        n = len(data)
        
        # Create time trend for regression
        t = np.arange(n)
        
        # Prepare regressors based on trend specification
        if self.trend == 'n':  # No trend
            X = np.ones((n, 1))
        elif self.trend == 'c':  # Constant only
            X = np.ones((n, 1))
        elif self.trend == 'ct':  # Constant and trend
            X = np.column_stack([np.ones(n), t])
        else:
            raise ValueError(f"Unknown trend specification: {self.trend}")
        
        # For CUSUM test, don't add arbitrary AR(1) structure
        # Use proper regression model based on trend specification only
        try:
            model = sm.OLS(data, X).fit()
            residuals = model.resid
        except Exception as e:
            # If OLS fails, use deviations from mean (simplest model)
            print(f"Warning: OLS failed ({e}), using CUSUM on deviations from mean")
            residuals = data - np.mean(data)
        
        # Apply CUSUM test
        breaks = self._cusum_test(residuals)
        
        return breaks

    def _cusum_test(self, residuals: np.ndarray) -> List[int]:
        """Apply CUSUM test to detect breaks."""
        n = len(residuals)
        breaks = []
        
        if self.use_recursive:
            # Use recursive CUSUM
            breaks = self._recursive_cusum(residuals)
        else:
            # Use standard CUSUM
            try:
                # Use statsmodels CUSUM test
                olsresid_test = breaks_cusumolsresid(residuals)
                stat, p_value = olsresid_test[:2]
                
                if p_value < self.significance_level:
                    # If test is significant, find approximate break location
                    # using cumulative sum
                    cusum = np.cumsum(residuals - np.mean(residuals))
                    
                    # Find point of maximum deviation
                    abs_cusum = np.abs(cusum)
                    break_idx = np.argmax(abs_cusum)
                    
                    # Validate break location
                    if break_idx > 5 and break_idx < n - 5:  # Avoid edge effects
                        breaks.append(break_idx)
                        
            except Exception as e:
                print(f"Warning: CUSUM test failed ({e}), using simple threshold method")
                breaks = self._simple_cusum_threshold(residuals)
        
        return breaks

    def _recursive_cusum(self, residuals: np.ndarray) -> List[int]:
        """Apply recursive CUSUM to find multiple breaks."""
        breaks = []
        
        def find_single_break(res):
            n = len(res)
            if n < 10:  # Too short for meaningful break
                return None
            
            # Calculate CUSUM
            cusum = np.cumsum(res - np.mean(res))
            
            # Calculate test statistic (scaled)
            sigma = np.std(res)
            if sigma == 0:
                return None
            
            scaled_cusum = cusum / (sigma * np.sqrt(n))
            
            # Find maximum absolute deviation
            max_idx = np.argmax(np.abs(scaled_cusum))
            max_stat = np.abs(scaled_cusum[max_idx])
            
            # Critical value based on sample size and significance level
            # Brown-Durbin-Evans critical values for CUSUM test
            if self.significance_level <= 0.01:
                critical_value = 1.63  # 1% significance
            elif self.significance_level <= 0.05:
                critical_value = 1.36  # 5% significance  
            else:
                critical_value = 1.14  # 10% significance
            
            # Adjust for sample size (smaller samples need larger critical values)
            if n < 50:
                critical_value *= 1.2
            elif n < 100:
                critical_value *= 1.1
            
            if max_stat > critical_value and 5 < max_idx < n - 5:
                return max_idx
            return None
        
        # Recursive application
        segments_to_check = [(residuals, 0)]
        
        while segments_to_check:
            segment, offset = segments_to_check.pop(0)
            break_idx = find_single_break(segment)
            
            if break_idx is not None:
                actual_idx = offset + break_idx
                breaks.append(actual_idx)
                
                # Split and add subsegments if they're large enough
                if break_idx > 10:
                    segments_to_check.append((segment[:break_idx], offset))
                if len(segment) - break_idx > 10:
                    segments_to_check.append((segment[break_idx:], offset + break_idx))
        
        return sorted(breaks)

    def _simple_cusum_threshold(self, residuals: np.ndarray) -> List[int]:
        """Simple threshold-based CUSUM detection."""
        n = len(residuals)
        cusum = np.cumsum(residuals - np.mean(residuals))
        
        # Use a threshold based on standard deviation
        threshold = 2.5 * np.std(cusum)
        
        breaks = []
        for i in range(5, n - 5):  # Avoid edges
            if abs(cusum[i]) > threshold:
                # Check if this is a local maximum
                is_local_max = True
                for j in range(max(0, i-3), min(n, i+4)):
                    if j != i and abs(cusum[j]) > abs(cusum[i]):
                        is_local_max = False
                        break
                
                if is_local_max and (not breaks or abs(i - breaks[-1]) > 10):
                    breaks.append(i)
        
        return breaks

    def _calculate_statistics(self,
                              data: np.ndarray,
                              breaks: List[int],
                              preprocess_meta: Dict) -> Dict[str, Any]:
        """Calculate CUSUM specific statistics."""
        base_stats = super()._calculate_statistics(data, breaks, preprocess_meta)
        
        # Add method-specific statistics
        stats = {
            **base_stats,
            'significance_level': self.significance_level,
            'trend_specification': self.trend,
            'recursive_cusum': self.use_recursive,
            'test_type': 'Recursive CUSUM' if self.use_recursive else 'Standard CUSUM'
        }
        
        # Calculate segment statistics
        if breaks:
            segments = self._get_segments(data, breaks)
            stats['segment_means'] = [float(np.mean(segment)) for segment in segments]
            stats['segment_vars'] = [float(np.var(segment)) for segment in segments]
            stats['segment_lengths'] = [len(segment) for segment in segments]
            
            # Calculate mean shift magnitudes
            mean_shifts = []
            for i in range(len(segments) - 1):
                shift = abs(np.mean(segments[i+1]) - np.mean(segments[i]))
                mean_shifts.append(float(shift))
            stats['mean_shifts'] = mean_shifts
        
        return stats

    def _create_breakpoints(self, indices: List[int], original_data: pd.DataFrame, 
                           processed_values: np.ndarray, value_col: str, date_col: str):
        """Create breakpoints with CUSUM-specific confidence scores."""
        breaks = super()._create_breakpoints(indices, original_data, processed_values, value_col, date_col)
        
        # Calculate CUSUM statistic confidence for each break
        if breaks:
            # Recalculate CUSUM for confidence scoring
            residuals = processed_values - np.mean(processed_values)
            cusum = np.cumsum(residuals)
            cusum_std = np.std(cusum)
            critical_value = 1.36  # 5% critical value
            
            for break_point in breaks:
                idx = break_point.index
                if idx < len(cusum):
                    # Calculate CUSUM statistic at break point
                    cusum_stat = abs(cusum[idx])
                    
                    # Confidence based on how much the statistic exceeds critical value
                    if cusum_std > 0:
                        normalized_stat = cusum_stat / cusum_std
                        # Convert to confidence: higher statistics = higher confidence
                        confidence = min(0.95, max(0.1, normalized_stat / critical_value))
                    else:
                        confidence = 0.1
                    
                    break_point.confidence = confidence
                    break_point.statistics = {
                        'cusum_statistic': cusum_stat,
                        'normalized_statistic': normalized_stat if cusum_std > 0 else 0.0,
                        'critical_value': critical_value
                    }
        
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