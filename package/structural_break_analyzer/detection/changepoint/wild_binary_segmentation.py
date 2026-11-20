from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from ..base import DetectionMethod

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False


class WildBinarySegmentationDetector(DetectionMethod):
    """
    Wild Binary Segmentation (WBS) changepoint detection using ruptures package.
    
    WBS is an extension of Binary Segmentation that uses random intervals
    instead of nested ones, making it more robust to noise and better at
    detecting multiple changepoints.
    """

    def __init__(self, width: float = 0.05, n_bkps: int = None, penalty: float = None, **kwargs):
        """
        Initialize Wild Binary Segmentation detector.
        
        Args:
            width: Relative width of random intervals (0.05 = 5% of data length)
            n_bkps: Number of breakpoints to detect (None for automatic)
            penalty: Penalty parameter for model selection (None for automatic)
        """
        if not HAS_RUPTURES:
            raise ImportError("ruptures package required for WBS. Install with: pip install ruptures")
        
        super().__init__(**kwargs)
        self.width = width
        self.n_bkps = n_bkps
        self.penalty = penalty

    def _preprocess(self, data: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, Dict]:
        """Preprocess data for Wild Binary Segmentation."""
        values = data[value_col].values
        
        # Handle missing values by interpolation
        if np.any(np.isnan(values)):
            mask = ~np.isnan(values)
            if np.sum(mask) < 15:
                raise ValueError("Too few non-NaN values for WBS detection")
            
            # Linear interpolation for missing values
            indices = np.arange(len(values))
            values = np.interp(indices, indices[mask], values[mask])
        
        if len(values) < 30:
            raise ValueError("Wild Binary Segmentation requires at least 30 observations")
        
        metadata = {
            'n_observations': len(values),
            'width_parameter': self.width,
            'interpolated': np.any(np.isnan(data[value_col].values))
        }
        
        return values, metadata

    def _detect_breaks(self, data: np.ndarray) -> List[int]:
        """Detect changepoints using proper Wild Binary Segmentation."""
        try:
            n_obs = len(data)
            width_samples = max(10, int(self.width * n_obs))
            
            # WBS parameters
            num_random_intervals = max(100, 2 * n_obs)  # Number of random intervals
            candidate_breaks = {}  # Dictionary to count break candidates
            
            # Generate random intervals and apply binary segmentation
            np.random.seed(42)  # For reproducibility
            
            for _ in range(num_random_intervals):
                # Generate random interval [start, end)
                start = np.random.randint(0, max(1, n_obs - width_samples))
                end = min(n_obs, start + np.random.randint(width_samples, 2 * width_samples))
                
                if end - start < 10:  # Skip too small intervals
                    continue
                
                # Extract data for this interval
                interval_data = data[start:end]
                
                # Apply binary segmentation to this interval
                try:
                    algo = rpt.Binseg(model="l2", min_size=2, jump=1)
                    algo.fit(interval_data)
                    
                    # Find one break in this interval
                    interval_breaks = algo.predict(n_bkps=1)
                    
                    # Convert back to global indices
                    for break_pos in interval_breaks:
                        if break_pos < len(interval_data):  # Exclude end marker
                            global_break = start + break_pos
                            if 5 < global_break < n_obs - 5:  # Avoid boundaries
                                candidate_breaks[global_break] = candidate_breaks.get(global_break, 0) + 1
                                
                except Exception:
                    continue
            
            # Select breaks based on frequency of detection
            if not candidate_breaks:
                return []
            
            # Sort by frequency of detection
            sorted_candidates = sorted(candidate_breaks.items(), key=lambda x: x[1], reverse=True)
            
            if self.n_bkps is not None:
                # Fixed number of breaks - take top n_bkps
                breaks = [pos for pos, count in sorted_candidates[:self.n_bkps]]
            else:
                # Automatic selection - use threshold based on detection frequency
                min_detections = max(1, num_random_intervals // 20)  # At least 5% detection rate
                breaks = [pos for pos, count in sorted_candidates if count >= min_detections]
                
                # Apply penalty-based filtering if specified  
                if self.penalty and len(breaks) > 0:
                    penalty = self.penalty
                    max_breaks = max(1, int(n_obs / penalty))
                    breaks = breaks[:max_breaks]
            
            breaks = sorted(breaks)
            if breaks and breaks[-1] == len(data):
                breaks = breaks[:-1]
            
            # Store results for statistics
            self._wbs_results = {
                'width_samples': width_samples,
                'model': 'Wild Binary Segmentation',
                'penalty_used': self.penalty,
                'n_bkps_requested': self.n_bkps,
                'breaks_found': len(breaks),
                'num_random_intervals': num_random_intervals
            }
            
            return breaks
            
        except Exception as e:
            print(f"Warning: Wild Binary Segmentation failed: {e}")
            return []

    def _calculate_statistics(self, data: np.ndarray, breaks: List[int], preprocess_meta: Dict) -> Dict[str, Any]:
        """Calculate WBS-specific statistics."""
        base_stats = super()._calculate_statistics(data, breaks, preprocess_meta)
        
        # Add WBS-specific stats
        if hasattr(self, '_wbs_results'):
            wbs_stats = {
                'method': 'Wild Binary Segmentation',
                'width_samples': self._wbs_results['width_samples'],
                'width_fraction': self.width,
                'model_type': self._wbs_results['model'],
                'penalty_parameter': self._wbs_results['penalty_used'],
                'requested_breaks': self._wbs_results['n_bkps_requested'],
                'detected_breaks': self._wbs_results['breaks_found']
            }
            base_stats.update(wbs_stats)
        
        return base_stats

    def _create_breakpoints(self, indices: List[int], original_data: pd.DataFrame, 
                           processed_values: np.ndarray, value_col: str, date_col: str):
        """Create breakpoints with WBS-specific confidence scores."""
        breaks = super()._create_breakpoints(indices, original_data, processed_values, value_col, date_col)
        
        # Set confidence based on detection robustness
        if breaks and hasattr(self, '_wbs_results'):
            n_obs = len(processed_values)
            width_samples = self._wbs_results['width_samples']
            
            for i, break_point in enumerate(breaks):
                # WBS is generally more robust than standard binary segmentation
                # Confidence based on distance from boundaries and width parameter
                boundary_distance = min(break_point.index, n_obs - break_point.index)
                
                # Higher confidence for well-separated breaks
                separation_factor = min(1.0, boundary_distance / width_samples)
                # WBS gets baseline confidence boost due to robustness
                base_confidence = 0.65
                
                confidence = base_confidence + 0.25 * separation_factor
                break_point.confidence = min(0.95, confidence)
                
                break_point.statistics = {
                    'width_samples': width_samples,
                    'boundary_distance': boundary_distance,
                    'detection_method': 'Wild Binary Segmentation',
                    'robustness_score': separation_factor
                }
        
        return breaks