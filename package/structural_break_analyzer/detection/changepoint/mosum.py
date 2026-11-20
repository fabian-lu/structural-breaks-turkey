from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from ..base import DetectionMethod

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False


class MOSUMDetector(DetectionMethod):
    """
    MOSUM (Moving Sum) changepoint detection using ruptures package.
    
    MOSUM uses a moving window approach to detect changes in the mean
    of a time series. It's efficient and good for detecting multiple
    changepoints.
    """

    def __init__(self, window: int = None, n_bkps: int = None, penalty: float = None, **kwargs):
        """
        Initialize MOSUM detector.
        
        Args:
            window: Window size for moving sum (None for automatic)
            n_bkps: Number of breakpoints to detect (None for automatic)
            penalty: Penalty parameter for model selection (None for automatic)
        """
        if not HAS_RUPTURES:
            raise ImportError("ruptures package required for MOSUM. Install with: pip install ruptures")
        
        super().__init__(**kwargs)
        self.window = window
        self.n_bkps = n_bkps
        self.penalty = penalty

    def _preprocess(self, data: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, Dict]:
        """Preprocess data for MOSUM."""
        values = data[value_col].values
        
        # Handle missing values by interpolation
        if np.any(np.isnan(values)):
            mask = ~np.isnan(values)
            if np.sum(mask) < 10:
                raise ValueError("Too few non-NaN values for MOSUM detection")
            
            # Linear interpolation for missing values
            indices = np.arange(len(values))
            values = np.interp(indices, indices[mask], values[mask])
        
        if len(values) < 20:
            raise ValueError("MOSUM requires at least 20 observations")
        
        # Set default window size if not provided
        window = self.window or max(10, len(values) // 10)
        
        metadata = {
            'n_observations': len(values),
            'window_size': window,
            'interpolated': np.any(np.isnan(data[value_col].values))
        }
        
        return values, metadata

    def _detect_breaks(self, data: np.ndarray) -> List[int]:
        """Detect changepoints using proper MOSUM algorithm."""
        try:
            n_obs = len(data)
            window = self.window or max(10, n_obs // 10)
            
            # MOSUM (Moving Sum) algorithm implementation
            # Calculate moving sums for left and right windows
            mosum_stats = []
            
            for k in range(window, n_obs - window):
                # Left and right windows around potential break point k
                left_window = data[k-window:k]
                right_window = data[k:k+window]
                
                # Calculate standardized moving sum statistic
                left_mean = np.mean(left_window)
                right_mean = np.mean(right_window)
                
                # Combined variance estimate
                left_var = np.var(left_window, ddof=1) if len(left_window) > 1 else 0
                right_var = np.var(right_window, ddof=1) if len(right_window) > 1 else 0
                pooled_var = (left_var + right_var) / 2
                
                if pooled_var > 0:
                    # MOSUM statistic: standardized difference of means
                    mosum_stat = abs(right_mean - left_mean) * np.sqrt(window / (2 * pooled_var))
                    mosum_stats.append((k, mosum_stat))
                else:
                    mosum_stats.append((k, 0))
            
            if not mosum_stats:
                return []
            
            # Find significant breaks using proper MOSUM critical values
            # MOSUM critical values are much higher than CUSUM (typically 3-4)
            # Based on boundary crossing probability for Brownian motion
            critical_value = 3.5  # Approximately 5% significance level for MOSUM
            
            # Extract breaks that exceed threshold
            significant_breaks = []
            
            if self.n_bkps is not None:
                # Fixed number of breaks - select top n_bkps
                sorted_stats = sorted(mosum_stats, key=lambda x: x[1], reverse=True)
                for i in range(min(self.n_bkps, len(sorted_stats))):
                    if sorted_stats[i][1] > critical_value:
                        significant_breaks.append(sorted_stats[i][0])
            else:
                # Automatic selection - all breaks above threshold
                for k, stat in mosum_stats:
                    if stat > critical_value:
                        significant_breaks.append(k)
                
                # Apply penalty-based filtering if specified
                if self.penalty and len(significant_breaks) > 0:
                    penalty = self.penalty
                    # Simple penalty application: reduce breaks if too many
                    max_breaks = max(1, int(n_obs / penalty))
                    if len(significant_breaks) > max_breaks:
                        # Keep only the strongest breaks
                        break_stats = [(k, next(s for pos, s in mosum_stats if pos == k)) 
                                     for k in significant_breaks]
                        break_stats.sort(key=lambda x: x[1], reverse=True)
                        significant_breaks = [k for k, s in break_stats[:max_breaks]]
            
            breaks = sorted(significant_breaks)
            
            # Merge breaks that are too close together (within window/2)
            if len(breaks) > 1:
                merged_breaks = []
                current_group = [breaks[0]]
                
                for i in range(1, len(breaks)):
                    if breaks[i] - breaks[i-1] <= window // 2:
                        current_group.append(breaks[i])
                    else:
                        # Take the middle break from the group (or strongest)
                        merged_breaks.append(current_group[len(current_group)//2])
                        current_group = [breaks[i]]
                
                # Add the last group
                merged_breaks.append(current_group[len(current_group)//2])
                breaks = merged_breaks
            
            # Store results for statistics
            self._mosum_results = {
                'window_size': window,
                'model': 'MOSUM',
                'critical_value': critical_value,
                'penalty_used': self.penalty,
                'n_bkps_requested': self.n_bkps,
                'breaks_found': len(breaks)
            }
            
            return breaks
            
        except Exception as e:
            print(f"Warning: MOSUM detection failed: {e}")
            return []

    def _calculate_statistics(self, data: np.ndarray, breaks: List[int], preprocess_meta: Dict) -> Dict[str, Any]:
        """Calculate MOSUM-specific statistics."""
        base_stats = super()._calculate_statistics(data, breaks, preprocess_meta)
        
        # Add MOSUM-specific stats
        if hasattr(self, '_mosum_results'):
            mosum_stats = {
                'method': 'MOSUM',
                'window_size': self._mosum_results['window_size'],
                'model_type': self._mosum_results['model'],
                'penalty_parameter': self._mosum_results['penalty_used'],
                'requested_breaks': self._mosum_results['n_bkps_requested'],
                'detected_breaks': self._mosum_results['breaks_found']
            }
            base_stats.update(mosum_stats)
        
        return base_stats

    def _create_breakpoints(self, indices: List[int], original_data: pd.DataFrame, 
                           processed_values: np.ndarray, value_col: str, date_col: str):
        """Create breakpoints with MOSUM-specific confidence scores."""
        breaks = super()._create_breakpoints(indices, original_data, processed_values, value_col, date_col)
        
        # Set confidence based on window size and data length
        if breaks and hasattr(self, '_mosum_results'):
            window_size = self._mosum_results['window_size']
            n_obs = len(processed_values)
            
            for i, break_point in enumerate(breaks):
                # Higher confidence for larger window sizes relative to data size
                # and for breaks that are well-separated from boundaries
                boundary_distance = min(break_point.index, n_obs - break_point.index)
                window_factor = min(1.0, window_size / 20.0)  # Normalize window size
                boundary_factor = min(1.0, boundary_distance / window_size)
                
                confidence = 0.5 + 0.4 * window_factor + 0.1 * boundary_factor
                break_point.confidence = min(0.95, confidence)
                
                break_point.statistics = {
                    'window_size': window_size,
                    'boundary_distance': boundary_distance,
                    'detection_method': 'MOSUM'
                }
        
        return breaks