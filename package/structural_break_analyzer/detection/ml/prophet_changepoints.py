from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from ..base import DetectionMethod

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


class ProphetChangepointDetector(DetectionMethod):
    """
    Prophet-based changepoint detection using Facebook's Prophet package.
    
    Uses Prophet's built-in changepoint detection which identifies trend changes
    using sparse priors. Good for business time series with trends and seasonality.
    """

    def __init__(self, n_changepoints: int = 25, changepoint_range: float = 0.8, 
                 changepoint_prior_scale: float = 0.02, **kwargs):
        """
        Initialize Prophet changepoint detector.
        
        Args:
            n_changepoints: Number of potential changepoints to include
            changepoint_range: Proportion of history in which trend changepoints are inferred  
            changepoint_prior_scale: Regularization parameter for changepoints
        """
        if not HAS_PROPHET:
            raise ImportError("prophet package required for Prophet changepoints. Install with: pip install prophet")
        
        super().__init__(**kwargs)
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale

    def _preprocess(self, data: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, Dict]:
        """Preprocess data for Prophet."""
        values = data[value_col].values
        
        if len(values) < 30:
            raise ValueError("Prophet changepoint detection requires at least 30 observations")
        
        # Prophet expects no NaN values
        if np.any(np.isnan(values)):
            mask = ~np.isnan(values)
            if np.sum(mask) < 20:
                raise ValueError("Too few non-NaN values for Prophet detection")
            
            # Linear interpolation for missing values
            indices = np.arange(len(values))
            values = np.interp(indices, indices[mask], values[mask])
        
        metadata = {
            'n_observations': len(values),
            'n_changepoints': self.n_changepoints,
            'changepoint_range': self.changepoint_range,
            'prior_scale': self.changepoint_prior_scale,
            'interpolated': np.any(np.isnan(data[value_col].values))
        }
        
        return values, metadata

    def _detect_breaks(self, data: np.ndarray) -> List[int]:
        """Detect changepoints using Prophet."""
        try:
            # Create DataFrame in Prophet format
            n_obs = len(data)
            df = pd.DataFrame({
                'ds': pd.date_range('2000-01-01', periods=n_obs, freq='D'),
                'y': data
            })
            
            # Initialize Prophet model with changepoint parameters
            model = Prophet(
                n_changepoints=self.n_changepoints,
                changepoint_range=self.changepoint_range,
                changepoint_prior_scale=self.changepoint_prior_scale,
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False
            )
            
            # Suppress Prophet's logging
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            
            # Fit the model
            model.fit(df)
            
            # Get changepoints
            changepoints = model.changepoints
            if len(changepoints) == 0:
                return []
            
            # Convert changepoint dates to indices
            changepoint_indices = []
            for cp_date in changepoints:
                # Find closest index to changepoint date
                idx = (df['ds'] - cp_date).abs().idxmin()
                changepoint_indices.append(idx)
            
            # Filter out changepoints with very small deltas (not significant)
            deltas = model.params['delta'][0]  # Changepoint deltas
            significant_changepoints = []
            
            delta_threshold = 0.01 * np.std(data)  # 1% of data std as threshold
            
            for i, idx in enumerate(changepoint_indices):
                if i < len(deltas) and abs(deltas[i]) > delta_threshold:
                    significant_changepoints.append(idx)
            
            # Store results for statistics
            self._prophet_results = {
                'total_changepoints': len(changepoint_indices),
                'significant_changepoints': len(significant_changepoints),
                'deltas': deltas.tolist() if len(deltas) > 0 else [],
                'changepoint_dates': [str(d) for d in changepoints],
                'delta_threshold': delta_threshold
            }
            
            return significant_changepoints
            
        except Exception as e:
            print(f"Warning: Prophet changepoint detection failed: {e}")
            return []

    def _calculate_statistics(self, data: np.ndarray, breaks: List[int], preprocess_meta: Dict) -> Dict[str, Any]:
        """Calculate Prophet-specific statistics."""
        base_stats = super()._calculate_statistics(data, breaks, preprocess_meta)
        
        # Add Prophet-specific stats
        if hasattr(self, '_prophet_results'):
            prophet_stats = {
                'method': 'Prophet Changepoints',
                'n_changepoints_configured': self.n_changepoints,
                'changepoint_range': self.changepoint_range,
                'prior_scale': self.changepoint_prior_scale,
                'total_changepoints_found': self._prophet_results['total_changepoints'],
                'significant_changepoints': self._prophet_results['significant_changepoints'],
                'delta_threshold': self._prophet_results['delta_threshold'],
                'changepoint_deltas': self._prophet_results['deltas']
            }
            base_stats.update(prophet_stats)
        
        return base_stats

    def _create_breakpoints(self, indices: List[int], original_data: pd.DataFrame, 
                           processed_values: np.ndarray, value_col: str, date_col: str):
        """Create breakpoints with Prophet-specific confidence scores."""
        breaks = super()._create_breakpoints(indices, original_data, processed_values, value_col, date_col)
        
        # Set confidence based on changepoint deltas
        if breaks and hasattr(self, '_prophet_results'):
            deltas = self._prophet_results['deltas']
            delta_threshold = self._prophet_results['delta_threshold']
            
            for i, break_point in enumerate(breaks):
                if i < len(deltas):
                    # Confidence based on relative size of the changepoint delta
                    delta_magnitude = abs(deltas[i])
                    max_delta = max(abs(d) for d in deltas) if deltas else 1.0
                    
                    # Normalize confidence: larger deltas get higher confidence
                    relative_strength = delta_magnitude / max_delta if max_delta > 0 else 0.5
                    confidence = 0.4 + 0.5 * relative_strength
                    
                    break_point.confidence = min(0.95, confidence)
                    break_point.statistics = {
                        'delta_magnitude': delta_magnitude,
                        'delta_threshold': delta_threshold,
                        'relative_strength': relative_strength,
                        'detection_method': 'Prophet'
                    }
                else:
                    break_point.confidence = 0.5
        
        return breaks