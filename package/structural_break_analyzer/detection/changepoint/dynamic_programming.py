from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import ruptures as rpt
from ..base import DetectionMethod


class DynamicProgrammingDetector(DetectionMethod):
    """
    Dynamic Programming detector using ruptures library.
    
    Dynamic programming (also known as optimal partitioning) is an exact
    algorithm for change-point detection that finds the optimal segmentation
    by minimizing a global cost function.
    """

    def __init__(self,
                 model: str = 'l2',
                 n_bkps: int = None,
                 penalty: float = None,
                 jump: int = 1,
                 min_size: int = 2,
                 **kwargs):
        """
        Initialize Dynamic Programming detector.

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
        """Preprocess data for Dynamic Programming."""
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
        """Detect breaks using Dynamic Programming."""
        n = len(data)
        
        # For very long series, dynamic programming can be slow
        # Use a reasonable limit and suggest alternatives
        if n > 2000:
            print(f"Warning: Dynamic Programming with {n} points may be slow. "
                  "Consider using Binary Segmentation for large datasets.")
        
        # Initialize detector
        algo = rpt.Dynp(model=self.model, jump=self.jump, min_size=self.min_size)
        algo.fit(data.reshape(-1, 1))
        
        # Detect breakpoints - Dynamic Programming only supports fixed n_bkps
        if self.n_bkps is not None:
            bkps = algo.predict(n_bkps=self.n_bkps)
        else:
            # Dynamic Programming requires fixed n_bkps, estimate reasonable number
            # Use penalty to estimate number of breaks (rough heuristic)
            if self.penalty is None:
                penalty = 2 * np.log(n)
            else:
                penalty = self.penalty
            
            # Convert penalty to estimated number of breaks (heuristic)
            # Higher penalty -> fewer breaks
            max_possible_breaks = min(10, n // 20)  # Conservative upper bound
            estimated_breaks = max(1, int(max_possible_breaks * np.exp(-penalty/n)))
            
            bkps = algo.predict(n_bkps=estimated_breaks)
        
        # Remove the last breakpoint (end of series) and convert to indices
        if bkps and bkps[-1] == n:
            bkps = bkps[:-1]
        
        return bkps

    def _calculate_statistics(self,
                              data: np.ndarray,
                              breaks: List[int],
                              preprocess_meta: Dict) -> Dict[str, Any]:
        """Calculate Dynamic Programming specific statistics."""
        base_stats = super()._calculate_statistics(data, breaks, preprocess_meta)
        
        # Add method-specific statistics
        stats = {
            **base_stats,
            'model': self.model,
            'jump': self.jump,
            'min_size': self.min_size,
            'penalty_used': self.penalty if self.penalty is not None else f"BIC (2*log({len(data)}))",
            'n_bkps_requested': self.n_bkps,
            'algorithm': 'Dynamic Programming (Optimal Partitioning)'
        }
        
        # Calculate segment statistics and optimal cost
        if breaks:
            segments = self._get_segments(data, breaks)
            stats['segment_means'] = [float(np.mean(segment)) for segment in segments]
            stats['segment_vars'] = [float(np.var(segment)) for segment in segments]
            stats['segment_lengths'] = [len(segment) for segment in segments]
            
            # Calculate total cost (approximation)
            total_cost = sum(len(seg) * np.var(seg) for seg in segments if len(seg) > 0)
            stats['total_cost'] = float(total_cost)
        
        return stats

    def _create_breakpoints(self, indices: List[int], original_data: pd.DataFrame, 
                           processed_values: np.ndarray, value_col: str, date_col: str):
        """Create breakpoints with Dynamic Programming cost-based confidence scores."""
        breaks = super()._create_breakpoints(indices, original_data, processed_values, value_col, date_col)
        
        if breaks:
            # Calculate confidence based on optimal cost reduction
            segments = self._get_segments(processed_values, indices)
            
            # Calculate total variance without breaks vs with optimal segmentation
            total_var = np.var(processed_values)
            if len(segments) > 0:
                weighted_var = sum(len(seg) * np.var(seg) for seg in segments if len(seg) > 1) / len(processed_values)
            else:
                weighted_var = total_var
            
            # Overall cost improvement
            if total_var > 0:
                overall_improvement = (total_var - weighted_var) / total_var
            else:
                overall_improvement = 0.0
            
            # Assign confidence based on local optimality of each break
            for break_point in breaks:
                idx = break_point.index
                
                # Calculate local cost improvement for this specific break
                window_size = min(30, len(processed_values) // 3)
                start_idx = max(0, idx - window_size)
                end_idx = min(len(processed_values), idx + window_size)
                
                if end_idx - start_idx > 6:
                    local_data = processed_values[start_idx:end_idx]
                    local_break = idx - start_idx
                    
                    if 2 < local_break < len(local_data) - 2:
                        # Cost with break vs without break (locally)
                        local_total_var = np.var(local_data)
                        before_cost = np.var(local_data[:local_break]) * local_break
                        after_cost = np.var(local_data[local_break:]) * (len(local_data) - local_break)
                        with_break_cost = (before_cost + after_cost) / len(local_data)
                        
                        if local_total_var > 0:
                            local_improvement = (local_total_var - with_break_cost) / local_total_var
                            # Scale to confidence range, bias upward for dynamic programming optimality
                            confidence = min(0.95, max(0.15, 0.3 + local_improvement * 0.6))
                        else:
                            confidence = 0.15
                    else:
                        confidence = 0.15
                else:
                    confidence = 0.15
                
                break_point.confidence = confidence
                break_point.statistics = {
                    'overall_cost_improvement': overall_improvement,
                    'local_cost_improvement': (confidence - 0.3) / 0.6 if confidence > 0.3 else 0.0,
                    'detection_method': 'Dynamic Programming'
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