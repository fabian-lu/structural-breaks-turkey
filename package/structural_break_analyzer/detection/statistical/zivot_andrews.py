from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from ..base import DetectionMethod

try:
    from arch.unitroot import ZivotAndrews
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


class ZivotAndrewsDetector(DetectionMethod):
    """
    Zivot-Andrews structural break test using arch package.
    
    Tests for a single structural break at an unknown date in a time series.
    Popular in econometrics for detecting unit root breaks.
    """

    def __init__(self, trend: str = 'c', lags: int = None, **kwargs):
        """
        Initialize Zivot-Andrews detector.
        
        Args:
            trend: Trend specification ('c' for constant, 'ct' for constant+trend)
            lags: Number of lags to include (None for automatic selection)
        """
        if not HAS_ARCH:
            raise ImportError("arch package required for Zivot-Andrews test. Install with: pip install arch")
        
        super().__init__(**kwargs)
        self.trend = trend
        self.lags = lags

    def _preprocess(self, data: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, Dict]:
        """Minimal preprocessing for Zivot-Andrews."""
        values = data[value_col].values
        
        # Remove NaN values
        mask = ~np.isnan(values)
        clean_values = values[mask]
        
        if len(clean_values) < 20:
            raise ValueError("Zivot-Andrews test requires at least 20 observations")
        
        metadata = {
            'n_observations': len(values),
            'n_clean': len(clean_values),
            'nan_positions': np.where(~mask)[0].tolist()
        }
        
        return clean_values, metadata

    def _detect_breaks(self, data: np.ndarray) -> List[int]:
        """Detect structural break using Zivot-Andrews test."""
        try:
            # First run the ZA test to check if there's evidence of a break
            za_test = ZivotAndrews(data, trend=self.trend, lags=self.lags)
            
            # If not significant, return no breaks
            if za_test.pvalue >= 0.05:
                self._za_results = {
                    'statistic': za_test.stat,
                    'pvalue': za_test.pvalue,
                    'critical_values': za_test.critical_values,
                    'break_location': None,
                    'trend': self.trend,
                    'lags_used': za_test.lags
                }
                return []
            
            # If significant, search for the optimal break point using proper ZA procedure
            # ZA test finds break that makes series most stationary (minimizes unit root test statistic)
            n = len(data)
            min_size = max(int(0.15 * n), 10)
            
            best_break = None
            most_negative_tstat = 0  # More negative = more evidence against unit root
            
            # Proper ZA approach: find break that minimizes unit root test statistic
            for break_point in range(min_size, n - min_size):
                try:
                    # Create break dummy variables
                    t = np.arange(n)
                    break_dummy = np.zeros(n)
                    break_dummy[break_point:] = 1
                    
                    if self.trend == 'c':
                        # Constant break model: Δy_t = μ + δ*D_t + α*y_{t-1} + ε_t
                        y_diff = np.diff(data)
                        y_lag = data[:-1]
                        X = np.column_stack([np.ones(len(y_diff)), break_dummy[1:], y_lag])
                    elif self.trend == 'ct':
                        # Trend break model
                        y_diff = np.diff(data)
                        y_lag = data[:-1]
                        trend_break = break_dummy[1:] * t[1:]
                        X = np.column_stack([np.ones(len(y_diff)), t[1:], break_dummy[1:], trend_break, y_lag])
                    else:  # trend == 't'
                        # Trend only
                        y_diff = np.diff(data)
                        y_lag = data[:-1]
                        trend_break = break_dummy[1:] * t[1:]
                        X = np.column_stack([t[1:], trend_break, y_lag])
                    
                    # Run regression and get t-statistic for unit root coefficient (last column)
                    from scipy.linalg import lstsq
                    coef, residuals, rank, s = lstsq(X, y_diff)
                    
                    # Calculate t-statistic for unit root parameter (alpha)
                    if len(residuals) > 0 and residuals[0] > 0:
                        mse = residuals[0] / (len(y_diff) - X.shape[1])
                        cov_matrix = mse * np.linalg.inv(X.T @ X)
                        se_alpha = np.sqrt(cov_matrix[-1, -1])
                        
                        if se_alpha > 0:
                            t_stat = coef[-1] / se_alpha  # t-statistic for unit root
                            
                            # More negative = stronger evidence against unit root (better)
                            if t_stat < most_negative_tstat:
                                most_negative_tstat = t_stat
                                best_break = break_point
                                
                except Exception:
                    continue
            
            # If no valid break found, fall back to structural change approach
            if best_break is None:
                max_change = 0
                for break_point in range(min_size, n - min_size):
                    before_mean = np.mean(data[:break_point])
                    after_mean = np.mean(data[break_point:])
                    change = abs(after_mean - before_mean)
                    if change > max_change:
                        max_change = change
                        best_break = break_point
            
            # Store results
            self._za_results = {
                'statistic': za_test.stat,
                'pvalue': za_test.pvalue,
                'critical_values': za_test.critical_values,
                'break_location': best_break,
                'trend': self.trend,
                'lags_used': za_test.lags
            }
            
            return [int(best_break)] if best_break is not None else []
                
        except Exception as e:
            print(f"Warning: Zivot-Andrews test failed: {e}")
            return []

    def _calculate_statistics(self, data: np.ndarray, breaks: List[int], preprocess_meta: Dict) -> Dict[str, Any]:
        """Calculate Zivot-Andrews specific statistics."""
        base_stats = super()._calculate_statistics(data, breaks, preprocess_meta)
        
        # Add ZA-specific stats if available
        if hasattr(self, '_za_results'):
            za_stats = {
                'zivot_andrews_statistic': float(self._za_results['statistic']),
                'zivot_andrews_pvalue': float(self._za_results['pvalue']),
                'critical_values': {k: float(v) for k, v in self._za_results['critical_values'].items()},
                'trend_specification': self._za_results['trend'],
                'lags_used': int(self._za_results['lags_used']),
                'break_significant': len(breaks) > 0
            }
            base_stats.update(za_stats)
        
        return base_stats

    def _create_breakpoints(self, indices: List[int], original_data: pd.DataFrame, 
                           processed_values: np.ndarray, value_col: str, date_col: str):
        """Create breakpoints with ZA-specific confidence scores."""
        breaks = super()._create_breakpoints(indices, original_data, processed_values, value_col, date_col)
        
        # Set confidence based on ZA test significance
        if hasattr(self, '_za_results') and breaks:
            pvalue = self._za_results['pvalue']
            # Convert p-value to confidence (lower p-value = higher confidence)
            confidence = max(0.0, 1.0 - pvalue)
            
            for break_point in breaks:
                break_point.confidence = confidence
                break_point.statistics = {
                    'za_statistic': self._za_results['statistic'],
                    'za_pvalue': pvalue,
                    'significance_level': '1%' if pvalue < 0.01 else '5%' if pvalue < 0.05 else 'not significant'
                }
        
        return breaks