from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from ..base import DetectionMethod, DetectionResult, BreakPoint


class VotingEnsembleDetector(DetectionMethod):
    """
    Advanced Ensemble detector that combines all 10 detection methods.
    
    Uses spatial clustering to group nearby breaks and requires consensus
    from multiple methods (5/10 minimum) to report a break.
    """

    def __init__(self, min_votes: int = 5, **kwargs):
        """
        Initialize Advanced Ensemble detector.

        Args:
            min_votes: Minimum number of methods that must agree (default: 5 out of 10)
        """
        super().__init__(**kwargs)
        self.min_votes = min_votes
        
        # All 10 detection methods
        self.all_methods = [
            'bai_perron', 'cusum', 'chow_test', 'zivot_andrews',
            'pelt', 'binary_segmentation', 'dynamic_programming', 
            'mosum', 'wild_binary_segmentation', 'prophet'
        ]

    def _calculate_tolerance(self, data_length: int) -> int:
        """Calculate adaptive tolerance based on series length (max 5)."""
        if data_length < 50:
            return max(2, min(5, data_length // 20))
        elif data_length < 200:
            return max(3, min(5, data_length // 40))
        else:
            return 5  # Maximum tolerance

    def _preprocess(self, data: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, Dict]:
        """Preprocess data for ensemble."""
        values = data[value_col].values
        
        # Basic validation
        mask = ~np.isnan(values)
        clean_values = values[mask]
        
        if len(clean_values) < 20:
            raise ValueError("Ensemble requires at least 20 non-NaN values")
        
        metadata = {
            'n_observations': len(values),
            'n_clean': len(clean_values),
            'n_methods': len(self.all_methods),
            'tolerance': self._calculate_tolerance(len(values))
        }
        
        return values, metadata

    def _detect_breaks(self, data: np.ndarray) -> List[int]:
        """Detect breaks using ensemble of all methods."""
        from structural_break_analyzer.analyzer import StructuralBreakAnalyzer
        
        analyzer = StructuralBreakAnalyzer()
        
        # Create temporary dataframe
        temp_data = pd.DataFrame({
            'date': pd.date_range('2000-01-01', periods=len(data)),
            'value': data
        })
        
        # Run all methods and collect results
        all_breaks = []
        method_results = {}
        
        for method_name in self.all_methods:
            try:
                result = analyzer.detect(temp_data, method=method_name)
                method_results[method_name] = result
                
                for bp in result.breaks:
                    all_breaks.append({
                        'position': bp.index,
                        'confidence': bp.confidence,
                        'method': method_name
                    })
                    
            except Exception as e:
                print(f"Warning: Method {method_name} failed: {e}")
                method_results[method_name] = None
        
        # Store individual results for statistics
        self._method_results = method_results
        
        if not all_breaks:
            return []
        
        # Apply spatial clustering and consensus voting
        tolerance = self._calculate_tolerance(len(data))
        ensemble_breaks = self._spatial_clustering_consensus(all_breaks, tolerance)
        
        return ensemble_breaks

    def _spatial_clustering_consensus(self, all_breaks: List[Dict], tolerance: int) -> List[int]:
        """Apply spatial clustering and consensus voting."""
        if not all_breaks:
            return []
        
        # Group breaks by proximity
        break_clusters = self._group_nearby_breaks(all_breaks, tolerance)
        
        # Filter clusters by minimum votes and compute final positions
        ensemble_breaks = []
        
        for cluster in break_clusters:
            if len(cluster) >= self.min_votes:  # At least 5/10 methods must agree
                # Confidence-weighted position averaging (ensure integer index)
                final_position = self._confidence_weighted_average(cluster)
                ensemble_breaks.append(int(round(final_position)))
        
        return sorted(ensemble_breaks)

    def _group_nearby_breaks(self, all_breaks: List[Dict], tolerance: int) -> List[List[Dict]]:
        """Group breaks that are within tolerance distance."""
        if not all_breaks:
            return []
        
        # Sort breaks by position
        sorted_breaks = sorted(all_breaks, key=lambda x: x['position'])
        
        clusters = []
        current_cluster = [sorted_breaks[0]]
        
        for i in range(1, len(sorted_breaks)):
            current_pos = sorted_breaks[i]['position']
            last_pos = current_cluster[-1]['position']
            
            if abs(current_pos - last_pos) <= tolerance:
                current_cluster.append(sorted_breaks[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_breaks[i]]
        
        clusters.append(current_cluster)
        return clusters

    def _confidence_weighted_average(self, cluster: List[Dict]) -> float:
        """Calculate confidence-weighted average position."""
        if not cluster:
            return 0.0
        
        # Handle zero confidence cases
        total_weight = 0
        weighted_sum = 0
        
        for break_info in cluster:
            confidence = max(0.1, break_info['confidence'])  # Minimum weight of 0.1
            weighted_sum += break_info['position'] * confidence
            total_weight += confidence
        
        return weighted_sum / total_weight if total_weight > 0 else np.mean([b['position'] for b in cluster])

    def _consensus_confidence(self, cluster: List[Dict], total_methods: int = 10) -> float:
        """Calculate consensus-based confidence."""
        if not cluster:
            return 0.0
        
        # Base confidence from individual methods
        confidences = [b['confidence'] for b in cluster]
        avg_confidence = np.mean(confidences)
        
        # Consensus bonus: more methods agreeing = higher confidence
        consensus_strength = len(cluster) / total_methods
        
        # Combine: base confidence boosted by consensus
        final_confidence = avg_confidence * (0.5 + 0.5 * consensus_strength)
        
        return min(0.95, max(0.1, final_confidence))  # Clamp to [0.1, 0.95]

    def _create_breakpoints(self, indices: List[int], original_data: pd.DataFrame, 
                           processed_values: np.ndarray, value_col: str, date_col: str):
        """Create breakpoints with ensemble-specific confidence scores."""
        breaks = super()._create_breakpoints(indices, original_data, processed_values, value_col, date_col)
        
        if breaks and hasattr(self, '_method_results'):
            # Calculate consensus confidence for each break
            all_breaks = []
            for method_name, result in self._method_results.items():
                if result is not None:
                    for bp in result.breaks:
                        all_breaks.append({
                            'position': bp.index,
                            'confidence': bp.confidence,
                            'method': method_name
                        })
            
            tolerance = self._calculate_tolerance(len(processed_values))
            break_clusters = self._group_nearby_breaks(all_breaks, tolerance)
            
            # Match ensemble breaks to clusters and assign confidence
            for break_point in breaks:
                # Find the cluster this break belongs to
                for cluster in break_clusters:
                    cluster_positions = [b['position'] for b in cluster]
                    if any(abs(break_point.index - pos) <= tolerance for pos in cluster_positions):
                        break_point.confidence = self._consensus_confidence(cluster)
                        break_point.statistics = {
                            'voting_methods': [b['method'] for b in cluster],
                            'votes_received': len(cluster),
                            'vote_ratio': len(cluster) / len(self.all_methods),
                            'detection_method': 'Ensemble'
                        }
                        break
        
        return breaks

    def _calculate_statistics(self, data: np.ndarray, breaks: List[int], preprocess_meta: Dict) -> Dict[str, Any]:
        """Calculate ensemble-specific statistics."""
        base_stats = super()._calculate_statistics(data, breaks, preprocess_meta)
        
        stats = {
            **base_stats,
            'ensemble_method': 'Spatial Clustering Consensus',
            'total_methods': len(self.all_methods),
            'min_votes_required': self.min_votes,
            'tolerance_used': preprocess_meta.get('tolerance', 5),
            'methods_used': self.all_methods
        }
        
        # Add individual method results
        if hasattr(self, '_method_results'):
            method_stats = []
            total_individual_breaks = 0
            
            for method_name, result in self._method_results.items():
                if result is not None:
                    n_breaks = result.n_breaks
                    total_individual_breaks += n_breaks
                    method_stats.append({
                        'method': method_name,
                        'breaks_found': n_breaks,
                        'break_positions': [bp.index for bp in result.breaks],
                        'succeeded': True
                    })
                else:
                    method_stats.append({
                        'method': method_name,
                        'breaks_found': 0,
                        'break_positions': [],
                        'succeeded': False
                    })
            
            stats['individual_method_results'] = method_stats
            stats['total_individual_breaks'] = total_individual_breaks
            stats['consensus_ratio'] = len(breaks) / total_individual_breaks if total_individual_breaks > 0 else 0
        
        return stats