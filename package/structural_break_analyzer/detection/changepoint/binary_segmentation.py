from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import ruptures as rpt
from ..base import DetectionMethod


class BinarySegmentationDetector(DetectionMethod):
    """
    Binary Segmentation detector using ruptures library.
    
    Binary segmentation is a sequential approach for change-point detection.
    It recursively applies a single change-point detection procedure to find
    multiple change-points.
    """

    def __init__(self,
                 model: str = 'l2',
                 n_bkps: int = None,
                 penalty: float = None,
                 jump: int = 5,
                 min_size: int = 2,
                 **kwargs):
        """
        Initialize Binary Segmentation detector.

        Args:
            model: Cost model ('l1', 'l2', 'rbf', 'linear', 'normal', 'ar')
            n_bkps: Number of breakpoints to find (if None, uses penalty)
            penalty: Penalty value (if None and n_bkps is None, uses BIC)
            jump: Subsample (one every jump points)
            min_size: Minimum segment length
        """
        super().__init__(**kwargs)
        self.model = model
        self.n_bkps = n_bkps
        self.penalty = penalty
        self.jump = jump
        self.min_size = min_size

    def _preprocess(self, data: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, Dict]:
        """Preprocess data for Binary Segmentation."""
        values = data[value_col].values
        
        # Handle missing values
        mask = ~np.isnan(values)
        clean_values = values[mask]
        
        if len(clean_values) < 10:
            raise ValueError("Too few non-NaN values for detection")
        
        # Normalize for better numerical stability
        mean_val = np.mean(clean_values)
        std_val = np.std(clean_values)
        if std_val > 0:
            normalized_values = (clean_values - mean_val) / std_val
        else:
            normalized_values = clean_values - mean_val
        
        metadata = {
            'n_observations': len(values),
            'n_clean': len(clean_values),
            'mean': mean_val,
            'std': std_val,
            'missing_indices': np.where(~mask)[0].tolist() if not mask.all() else []
        }
        
        return normalized_values, metadata

    def _detect_breaks(self, data: np.ndarray) -> List[int]:
        """Detect breaks using Binary Segmentation."""
        n = len(data)
        
        # Initialize detector
        algo = rpt.Binseg(model=self.model, jump=self.jump, min_size=self.min_size)
        algo.fit(data.reshape(-1, 1))
        
        # Detect breakpoints
        if self.n_bkps is not None:
            # Use fixed number of breakpoints
            bkps = algo.predict(n_bkps=self.n_bkps)
        else:
            # Use penalty-based detection
            if self.penalty is None:
                # Default to BIC penalty
                penalty = 2 * np.log(n)
            else:
                penalty = self.penalty
            bkps = algo.predict(penalty)
        
        # Remove the last breakpoint (end of series) and convert to indices
        if bkps and bkps[-1] == n:
            bkps = bkps[:-1]
        
        return bkps

    def _calculate_statistics(self,
                              data: np.ndarray,
                              breaks: List[int],
                              preprocess_meta: Dict) -> Dict[str, Any]:
        """Calculate Binary Segmentation specific statistics."""
        base_stats = super()._calculate_statistics(data, breaks, preprocess_meta)
        
        # Add method-specific statistics
        stats = {
            **base_stats,
            'model': self.model,
            'jump': self.jump,
            'min_size': self.min_size,
            'penalty_used': self.penalty if self.penalty is not None else f"BIC (2*log({len(data)}))",
            'n_bkps_requested': self.n_bkps
        }
        
        # Calculate segment statistics
        if breaks:
            segments = self._get_segments(data, breaks)
            stats['segment_means'] = [float(np.mean(segment)) for segment in segments]
            stats['segment_vars'] = [float(np.var(segment)) for segment in segments]
            stats['segment_lengths'] = [len(segment) for segment in segments]
        
        return stats

    def _create_breakpoints(self, indices: List[int], original_data: pd.DataFrame, 
                           processed_values: np.ndarray, value_col: str, date_col: str):
        """Create breakpoints with cost-based confidence scores."""
        breaks = super()._create_breakpoints(indices, original_data, processed_values, value_col, date_col)
        
        if breaks:
            # Calculate confidence based on cost reduction for each break
            segments = self._get_segments(processed_values, indices)
            
            # Calculate total variance (no breaks) vs segmented variance
            total_var = np.var(processed_values)
            segmented_var = sum(len(seg) * np.var(seg) for seg in segments if len(seg) > 1) / len(processed_values)
            
            # Overall cost reduction ratio
            if total_var > 0:
                cost_reduction = (total_var - segmented_var) / total_var
            else:
                cost_reduction = 0.0
            
            # Assign confidence to each break based on local cost reduction
            for i, break_point in enumerate(breaks):
                idx = break_point.index
                
                # Calculate local variance reduction around this break
                window_size = min(20, len(processed_values) // 4)
                start_idx = max(0, idx - window_size)
                end_idx = min(len(processed_values), idx + window_size)
                
                if end_idx - start_idx > 10:
                    local_data = processed_values[start_idx:end_idx]
                    local_break = idx - start_idx
                    
                    if 2 < local_break < len(local_data) - 2:
                        # Variance before and after vs total local variance
                        before_var = np.var(local_data[:local_break])
                        after_var = np.var(local_data[local_break:])
                        total_local_var = np.var(local_data)
                        
                        if total_local_var > 0:
                            weighted_var = (local_break * before_var + (len(local_data) - local_break) * after_var) / len(local_data)
                            local_reduction = (total_local_var - weighted_var) / total_local_var
                            confidence = min(0.95, max(0.1, local_reduction * 2))  # Scale to [0.1, 0.95]
                        else:
                            confidence = 0.1
                    else:
                        confidence = 0.1
                else:
                    confidence = 0.1
                
                break_point.confidence = confidence
                break_point.statistics = {
                    'cost_reduction_ratio': cost_reduction,
                    'local_variance_reduction': confidence / 2,  # Unscaled version
                    'detection_method': 'Binary Segmentation'
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