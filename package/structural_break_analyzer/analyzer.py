"""
Clean architecture for structural break analysis.

This module provides two main classes:
- TimeSeriesAnalyzer: Pure time series feature extraction and analysis
- StructuralBreakAnalyzer: Pure structural break detection
- LLMExplainer: Optional AI explanations (with RAG support only for breaks)

All result classes include built-in plotting capabilities.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

from .detection.base import DetectionResult, BreakPoint
from .detection.statistical.bai_perron import BaiPerronDetector
from .detection.changepoint.pelt import PeltDetector
from .detection.changepoint.binary_segmentation import BinarySegmentationDetector
from .detection.changepoint.dynamic_programming import DynamicProgrammingDetector
from .detection.statistical.cusum import CUSUMDetector
from .detection.statistical.chow_test import ChowTestDetector
from .detection.statistical.zivot_andrews import ZivotAndrewsDetector
from .detection.changepoint.mosum import MOSUMDetector
from .detection.changepoint.wild_binary_segmentation import WildBinarySegmentationDetector
from .detection.ml.prophet_changepoints import ProphetChangepointDetector
from .detection.ensemble.voting_ensemble import VotingEnsembleDetector
from .timeseries.features import TimeSeriesAnalyzer as TSFeatureAnalyzer, TimeSeriesFeatures


@dataclass
class TimeSeriesResult:
    """Clean result class for time series analysis with built-in plotting."""
    
    features: TimeSeriesFeatures
    data: pd.DataFrame
    value_col: str
    date_col: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def plot(self, figsize: Tuple[int, int] = (15, 12), 
             show_features: bool = True) -> Figure:
        """
        Plot comprehensive time series diagnostics with side-by-side layout.
        
        Args:
            figsize: Figure size (width, height)
            show_features: Whether to highlight key features
            
        Returns:
            matplotlib Figure
        """
        # Get enhanced data from MVP analysis if available
        reports = self.features.metadata.get('reports', {})
        
        dates = pd.to_datetime(self.data[self.date_col])
        values = self.data[self.value_col].values
        series = pd.Series(values, index=dates, name=self.value_col)
        
        # Create comprehensive plot layout (side-by-side)
        fig, axes = plt.subplots(6, 2, figsize=figsize)
        fig.suptitle('Comprehensive Time Series Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histogram (non-missing) + Missingness timeline
        ax1 = axes[0, 0]
        clean_values = series.dropna().values
        ax1.hist(clean_values, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(self.features.mean, color='red', linestyle='--', 
                   label=f'Mean: {self.features.mean:.2f}')
        ax1.axvline(self.features.mean + self.features.std, color='orange', 
                   linestyle=':', label=f'±1σ: {self.features.std:.2f}')
        ax1.axvline(self.features.mean - self.features.std, color='orange', linestyle=':')
        ax1.set_title('Distribution (non-missing)')
        ax1.set_xlabel(self.value_col)
        ax1.set_ylabel('Count')
        ax1.legend()
        
        ax2 = axes[0, 1]
        mflag = series.isna().astype(int)
        ax2.plot(mflag.index, mflag.values, drawstyle="steps-post", color='red')
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_title('Missingness timeline (1=missing)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Missing flag')
        
        # 2. Main time series + Enhanced ACF with confidence bands
        ax3 = axes[1, 0]
        ax3.plot(dates, values, 'b-', linewidth=1, alpha=0.8, label='Time Series')
        
        # Add trend line if significant
        if self.features.trend_strength > 0.3:
            x_num = np.arange(len(values))
            trend_line = np.polyval([self.features.trend_slope, 
                                   self.features.mean - self.features.trend_slope * len(values)/2], 
                                   x_num)
            ax3.plot(dates, trend_line, 'r--', linewidth=2, alpha=0.8, 
                    label=f'Trend (R²={self.features.trend_r2:.2f})')
            ax3.legend()
        ax3.set_title('Time Series with Trend')
        ax3.set_ylabel(self.value_col)
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        # Enhanced ACF with confidence bands
        lags = range(min(20, len(clean_values) // 4))
        acf_values = self._calculate_simple_acf(clean_values, max(lags)) if lags else [1.0]
        
        ax4.stem(lags, acf_values[:len(lags)], basefmt=" ")
        ax4.axhline(0, color='black', linewidth=0.5)
        conf_int = 1.96/np.sqrt(len(clean_values)) if len(clean_values) > 0 else 0
        ax4.axhline(conf_int, color='red', linestyle='--', alpha=0.7, label='95% CI')
        ax4.axhline(-conf_int, color='red', linestyle='--', alpha=0.7)
        ax4.set_title('ACF with Confidence Bands')
        ax4.set_xlabel('Lag')
        ax4.set_ylabel('ACF')
        ax4.legend()
        
        # 3. STL Decomposition (if available) + PACF with confidence bands
        ax5 = axes[2, 0]
        if self.features.decomposition and 'trend' in reports.get('seasonality', {}):
            stl_data = reports['seasonality']
            if isinstance(stl_data.get('trend'), pd.Series):
                trend_series = stl_data['trend']
                ax5.plot(trend_series.index, trend_series.values, 'g-', linewidth=1.5)
                ax5.set_title('STL Trend Component')
                ax5.set_ylabel('Trend')
            else:
                ax5.plot(dates, values, 'b-', alpha=0.3)
                ax5.set_title('Time Series (STL not available)')
        else:
            ax5.plot(dates, values, 'b-', alpha=0.7)
            ax5.set_title('Time Series (no decomposition)')
        ax5.grid(True, alpha=0.3)
        
        ax6 = axes[2, 1]
        # PACF with confidence bands (simplified)
        if len(clean_values) > 10:
            pacf_values = self._calculate_simple_acf(clean_values, min(10, len(lags)))  # Simplified PACF
            ax6.stem(range(len(pacf_values)), pacf_values, basefmt=" ")
            ax6.axhline(0, color='black', linewidth=0.5)
            ax6.axhline(conf_int, color='red', linestyle='--', alpha=0.7, label='95% CI')
            ax6.axhline(-conf_int, color='red', linestyle='--', alpha=0.7)
            ax6.set_title('PACF with Confidence Bands')
            ax6.set_xlabel('Lag')
            ax6.set_ylabel('PACF')
            ax6.legend()
        else:
            ax6.text(0.5, 0.5, 'Insufficient data\nfor PACF', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('PACF (insufficient data)')
        
        # 4. Calendar effects (weekday + month averages)
        ax7 = axes[3, 0]
        if isinstance(dates.index, pd.DatetimeIndex) and self.features.weekday_means:
            wd_means = pd.Series(self.features.weekday_means).sort_index()
            ax7.bar(range(len(wd_means)), wd_means.values, alpha=0.7, color='lightcoral')
            ax7.set_title('Weekday Averages (0=Mon)')
            ax7.set_xlabel('Weekday')
            ax7.set_ylabel(f'Mean({self.value_col})')
            ax7.set_xticks(range(len(wd_means)))
        else:
            ax7.text(0.5, 0.5, 'No calendar effects\navailable', ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Weekday Effects (not available)')
        
        ax8 = axes[3, 1]
        if isinstance(dates.index, pd.DatetimeIndex) and self.features.month_means:
            mo_means = pd.Series(self.features.month_means).sort_index()
            ax8.bar(range(len(mo_means)), mo_means.values, alpha=0.7, color='lightgreen')
            ax8.set_title('Month Averages (1=Jan)')
            ax8.set_xlabel('Month')
            ax8.set_ylabel(f'Mean({self.value_col})')
        else:
            ax8.text(0.5, 0.5, 'No monthly effects\navailable', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Monthly Effects (not available)')
        
        # 5. Volatility clustering plots
        ax9 = axes[4, 0]
        if self.features.acf_abs_residuals and len(self.features.acf_abs_residuals) > 1:
            acf_abs = self.features.acf_abs_residuals
            ax9.stem(range(len(acf_abs)), acf_abs, basefmt=" ")
            ax9.set_title('ACF of |residuals| (volatility clustering)')
            ax9.set_xlabel('Lag')
            ax9.set_ylabel('ACF')
        else:
            ax9.text(0.5, 0.5, 'Volatility clustering\nanalysis not available', ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('ACF |residuals| (not available)')
            
        ax10 = axes[4, 1]
        if self.features.acf_squared_residuals and len(self.features.acf_squared_residuals) > 1:
            acf_sq = self.features.acf_squared_residuals
            ax10.stem(range(len(acf_sq)), acf_sq, basefmt=" ")
            ax10.set_title('ACF of residuals² (volatility clustering)')
            ax10.set_xlabel('Lag')
            ax10.set_ylabel('ACF')
        else:
            ax10.text(0.5, 0.5, 'Squared residuals\nanalysis not available', ha='center', va='center', transform=ax10.transAxes)
            ax10.set_title('ACF residuals² (not available)')
        
        # 6. Nonlinear dependence + Enhanced summary
        ax11 = axes[5, 0]
        if self.features.mutual_info_values and len(self.features.mutual_info_values) > 1:
            mi_vals = self.features.mutual_info_values
            lags_mi = list(range(1, len(mi_vals) + 1))
            ax11.plot(lags_mi, mi_vals, 'o-', color='purple')
            
            # Mark significant lags
            if self.features.mutual_info_lags:
                sig_lags = self.features.mutual_info_lags
                sig_mi = [mi_vals[lag-1] for lag in sig_lags if lag <= len(mi_vals)]
                if sig_mi:
                    ax11.scatter(sig_lags, sig_mi, color='red', s=100, alpha=0.7, label='significant')
                    ax11.legend()
            
            ax11.set_title('Mutual Information vs Lag')
            ax11.set_xlabel('Lag')
            ax11.set_ylabel('Mutual Information')
        else:
            ax11.text(0.5, 0.5, 'Mutual information\nanalysis not available', ha='center', va='center', transform=ax11.transAxes)
            ax11.set_title('Mutual Information (not available)')
        
        # Enhanced feature summary
        ax12 = axes[5, 1]
        ax12.axis('off')
        
        # Enhanced summary with new features
        summary_parts = [
            f"Enhanced Time Series Analysis",
            f"─" * 35,
            f"Basic Stats:",
            f"• Mean: {self.features.mean:.3f}",
            f"• Std: {self.features.std:.3f}",
            f"• Skew: {self.features.skewness:.3f}",
            f"• Kurt: {self.features.kurtosis:.3f}",
            f"",
            f"Structural Properties:",
            f"• Trend: {'Strong' if self.features.trend_strength > 0.7 else 'Moderate' if self.features.trend_strength > 0.3 else 'Weak'}",
            f"• Seasonality: {'Yes' if self.features.has_seasonality else 'No'}",
            f"• Stationarity: {'Yes' if self.features.is_stationary else 'No'}",
            f"• Autocorr(1): {self.features.acf_1:.3f}",
            f"",
            f"Advanced Features:",
            f"• Hurst(R/S): {self.features.hurst_rs:.3f}",
            f"• Vol.Clustering: {self.features.volatility_clustering_summary[:25]}{'...' if len(self.features.volatility_clustering_summary) > 25 else ''}",
        ]
        
        # Add non-linearity test results
        if self.features.bds_test and 'pvalue' in self.features.bds_test:
            summary_parts.append(f"• BDS p-val: {self.features.bds_test['pvalue']:.3f}")
        
        # Add nonlinear dependence
        if self.features.mutual_info_lags:
            summary_parts.append(f"• MI lags: {self.features.mutual_info_lags[:3]}")
        
        summary_text = "\n".join(summary_parts)
        
        ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        
        # Try to display in Jupyter environments
        try:
            # Check if we're in a Jupyter environment
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None and hasattr(ipython, 'kernel'):
                # We're in Jupyter, the plot should display automatically
                pass
            else:
                # We're not in Jupyter, explicitly show the plot
                plt.show()
        except ImportError:
            # IPython not available, show the plot
            plt.show()
        
        return fig
    
    def _calculate_simple_acf(self, values: np.ndarray, max_lag: int) -> np.ndarray:
        """Simple ACF calculation for plotting."""
        n = len(values)
        acf_vals = np.zeros(max_lag + 1)
        acf_vals[0] = 1.0
        
        mean_val = np.mean(values)
        var_val = np.var(values)
        
        for lag in range(1, min(max_lag + 1, n)):
            covariance = np.mean((values[:-lag] - mean_val) * (values[lag:] - mean_val))
            acf_vals[lag] = covariance / (var_val + 1e-12)
        
        return acf_vals
    
    def summary(self) -> str:
        """Generate a readable summary of the analysis."""
        return self.features.summary()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'features': self.features.to_dict(),
            'metadata': self.metadata,
            'summary': self.summary()
        }


@dataclass 
class BreakDetectionResult:
    """Clean result class for structural break detection with built-in plotting."""
    
    breaks: List[BreakPoint]
    data: pd.DataFrame
    value_col: str
    date_col: str
    method: str
    confidence_threshold: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_breaks(self) -> int:
        """Number of breaks detected."""
        return len(self.breaks)
    
    def plot(self, figsize: Tuple[int, int] = (12, 6), 
             show_confidence: bool = True) -> Figure:
        """
        Plot the time series with detected breaks highlighted.
        
        Args:
            figsize: Figure size (width, height) 
            show_confidence: Whether to show confidence bands
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        dates = pd.to_datetime(self.data[self.date_col])
        values = self.data[self.value_col].values
        
        # Plot main time series
        ax.plot(dates, values, 'b-', linewidth=1, alpha=0.8, label='Time Series')
        
        # Highlight breaks
        colors = ['red', 'orange', 'purple', 'brown', 'pink']
        for i, break_point in enumerate(self.breaks):
            color = colors[i % len(colors)]
            
            # Convert break date to datetime if needed
            if hasattr(break_point.date, 'strftime'):
                break_date = break_point.date
            else:
                # Assume it's an index, get the corresponding date
                if break_point.index < len(dates):
                    break_date = dates.iloc[break_point.index]
                else:
                    continue
            
            # Vertical line at break
            ax.axvline(break_date, color=color, linestyle='--', linewidth=2,
                      alpha=0.8, label=f'Break {i+1} ({break_point.confidence:.1%})')
            
            # Add annotation
            ax.annotate(f'Break {i+1}\n{break_point.direction}\n{break_point.confidence:.1%}',
                       xy=(break_date, values[break_point.index] if break_point.index < len(values) else np.mean(values)),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                       fontsize=8)
        
        ax.set_title(f'Structural Break Detection ({self.method})\n{self.n_breaks} breaks detected')
        ax.set_xlabel('Date')
        ax.set_ylabel(self.value_col)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // 12)))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def summary(self) -> str:
        """Generate a readable summary of break detection."""
        if self.n_breaks == 0:
            return f"No structural breaks detected using {self.method}."
        
        lines = [
            f"Structural Break Detection Results ({self.method})",
            "=" * 50,
            f"Breaks detected: {self.n_breaks}",
            ""
        ]
        
        for i, break_point in enumerate(self.breaks):
            date_str = break_point.date.strftime('%Y-%m-%d') if hasattr(break_point.date, 'strftime') else str(break_point.date)
            lines.extend([
                f"Break {i+1}:",
                f"  Date: {date_str}",
                f"  Confidence: {break_point.confidence:.1%}", 
                f"  Direction: {break_point.direction}",
                f"  Magnitude: {break_point.magnitude:.3f}",
                ""
            ])
        
        # Add summary statistics
        avg_confidence = np.mean([b.confidence for b in self.breaks])
        directions = [b.direction for b in self.breaks]
        up_count = sum(1 for d in directions if d in ['up', 'increase'])
        down_count = sum(1 for d in directions if d in ['down', 'decrease'])
        
        lines.extend([
            "Summary:",
            f"  Average confidence: {avg_confidence:.1%}",
            f"  Upward breaks: {up_count}",
            f"  Downward breaks: {down_count}",
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            'n_breaks': self.n_breaks,
            'method': self.method,
            'breaks': [],
            'metadata': self.metadata,
            'summary': self.summary()
        }
        
        for break_point in self.breaks:
            break_dict = {
                'date': break_point.date.strftime('%Y-%m-%d') if hasattr(break_point.date, 'strftime') else str(break_point.date),
                'index': break_point.index,
                'confidence': float(break_point.confidence),
                'magnitude': float(break_point.magnitude),
                'direction': break_point.direction,
                'statistics': break_point.statistics
            }
            result['breaks'].append(break_dict)
        
        return result


class TimeSeriesAnalyzer:
    """
    Clean time series feature extraction and analysis.
    
    This class focuses solely on extracting and analyzing time series characteristics
    without any LLM or break detection functionality.
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize the time series analyzer.
        
        Args:
            use_cache: Whether to cache feature extraction results
        """
        self.feature_analyzer = TSFeatureAnalyzer(use_cache=use_cache)
    
    def analyze(self,
                data: Union[pd.DataFrame, pd.Series, np.ndarray],
                value_col: str = 'value',
                date_col: str = 'date',
                freq: Optional[str] = None) -> TimeSeriesResult:
        """
        Analyze time series for comprehensive feature extraction.
        
        Args:
            data: Time series data (DataFrame, Series, or ndarray)
            value_col: Column name for values
            date_col: Column name for dates
            freq: Data frequency ('D', 'M', 'H', etc.)
            
        Returns:
            TimeSeriesResult with features and plotting capabilities
        """
        # Ensure DataFrame format
        df = self._ensure_dataframe(data, value_col, date_col)
        self._validate_input(df, value_col, date_col)
        
        # Extract features using existing analyzer
        features = self.feature_analyzer.analyze(df, value_col, date_col, freq)
        
        # Create result with plotting capabilities
        result = TimeSeriesResult(
            features=features,
            data=df,
            value_col=value_col,
            date_col=date_col,
            metadata={
                'data_shape': df.shape,
                'date_range': f"{df[date_col].min()} to {df[date_col].max()}",
                'frequency': freq,
                'analyzer_version': '2.0'
            }
        )
        
        return result
    
    def _ensure_dataframe(self,
                          data: Union[pd.DataFrame, pd.Series, np.ndarray],
                          value_col: str,
                          date_col: str) -> pd.DataFrame:
        """Convert various input types to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        
        if isinstance(data, pd.Series):
            df = pd.DataFrame({value_col: data.values})
            # Use index as dates if it looks like datetimes
            if isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)) or np.issubdtype(data.index.dtype, np.datetime64):
                df[date_col] = pd.to_datetime(data.index)
            else:
                df[date_col] = np.arange(len(data))
            return df[[date_col, value_col]]
        
        if isinstance(data, np.ndarray):
            if data.ndim != 1:
                raise ValueError("NumPy array input must be 1-dimensional")
            df = pd.DataFrame({
                date_col: np.arange(len(data)),
                value_col: data.astype(float)
            })
            return df
        
        raise TypeError("Unsupported data type. Expected DataFrame, Series, or 1D ndarray.")
    
    def _validate_input(self, data: pd.DataFrame, value_col: str, date_col: str):
        """Validate input data."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        if value_col not in data.columns:
            raise ValueError(f"Column '{value_col}' not found in data")
        
        if date_col not in data.columns:
            raise ValueError(f"Column '{date_col}' not found in data")
        
        if len(data) < 10:
            raise ValueError(f"Time series too short ({len(data)} points). Need at least 10 points.")


class StructuralBreakAnalyzer:
    """
    Clean structural break detection.
    
    This class focuses solely on detecting structural breaks without any
    LLM functionality or time series feature extraction.
    """
    
    def __init__(self):
        """Initialize the structural break analyzer."""
        pass

    def detect(self,
               data: Union[pd.DataFrame, pd.Series, np.ndarray],
               value_col: str = 'value', 
               date_col: str = 'date',
               method: str = 'auto',
               **detector_params) -> BreakDetectionResult:
        """
        Detect structural breaks in time series.
        
        Args:
            data: Time series data (DataFrame, Series, or ndarray)
            value_col: Column name for values
            date_col: Column name for dates  
            method: Detection method ('bai_perron', 'pelt', 'auto', etc.)
            **detector_params: Additional parameters for the detector
            
        Returns:
            BreakDetectionResult with breaks and plotting capabilities
        """
        # Ensure DataFrame format
        df = self._ensure_dataframe(data, value_col, date_col)
        self._validate_input(df, value_col, date_col)
        
        # Run break detection
        detection_result = self._detect_breaks(df, value_col, date_col, method, **detector_params)
        
        # Create clean result
        result = BreakDetectionResult(
            breaks=detection_result.breaks,
            data=df,
            value_col=value_col,
            date_col=date_col,
            method=detection_result.method,
            metadata={
                'data_shape': df.shape,
                'date_range': f"{df[date_col].min()} to {df[date_col].max()}",
                'method_used': detection_result.method,
                'detector_params': detector_params,
                'analyzer_version': '2.0'
            }
        )
        
        return result

    def _ensure_dataframe(self,
                          data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                          value_col: str,
                          date_col: str) -> pd.DataFrame:
        """Convert various input types to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        
        if isinstance(data, pd.Series):
            df = pd.DataFrame({value_col: data.values})
            if isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)) or np.issubdtype(data.index.dtype, np.datetime64):
                df[date_col] = pd.to_datetime(data.index)
            else:
                df[date_col] = np.arange(len(data))
            return df[[date_col, value_col]]
        
        if isinstance(data, np.ndarray):
            if data.ndim != 1:
                raise ValueError("NumPy array input must be 1-dimensional")
            df = pd.DataFrame({
                date_col: np.arange(len(data)),
                value_col: data.astype(float)
            })
            return df
        
        raise TypeError("Unsupported data type. Expected DataFrame, Series, or 1D ndarray.")
    
    def _validate_input(self, data: pd.DataFrame, value_col: str, date_col: str):
        """Validate input data.""" 
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        if value_col not in data.columns:
            raise ValueError(f"Column '{value_col}' not found in data")
        
        if date_col not in data.columns:
            raise ValueError(f"Column '{date_col}' not found in data")
        
        if len(data) < 10:
            raise ValueError(f"Time series too short ({len(data)} points). Need at least 10 points.")
    
    def _detect_breaks(self,
                       data: pd.DataFrame,
                       value_col: str,
                       date_col: str, 
                       method: str,
                       **detector_params) -> DetectionResult:
        """Run break detection using specified method."""
        
        # Choose detector
        if method == 'bai_perron':
            detector = BaiPerronDetector(**detector_params)
        elif method == 'pelt':
            if 'model' not in detector_params:
                detector_params['model'] = 'l2'
            if 'penalty' not in detector_params and 'n_bkps' not in detector_params:
                detector_params['penalty'] = 3 * np.log(len(data))
            detector = PeltDetector(**detector_params)
        elif method == 'binary_segmentation':
            detector = BinarySegmentationDetector(**detector_params)
        elif method == 'dynamic_programming':
            detector = DynamicProgrammingDetector(**detector_params)
        elif method == 'cusum':
            detector = CUSUMDetector(**detector_params)
        elif method == 'chow_test':
            detector = ChowTestDetector(**detector_params)
        elif method == 'zivot_andrews':
            detector = ZivotAndrewsDetector(**detector_params)
        elif method == 'mosum':
            detector = MOSUMDetector(**detector_params)
        elif method == 'wild_binary_segmentation':
            detector = WildBinarySegmentationDetector(**detector_params)
        elif method == 'prophet':
            detector = ProphetChangepointDetector(**detector_params)
        elif method == 'ensemble':
            detector = VotingEnsembleDetector(**detector_params)
        elif method == 'auto':
            detector = self._auto_select_detector(data, value_col, **detector_params)
        else:
            available_methods = ['bai_perron', 'pelt', 'binary_segmentation', 'dynamic_programming',
                               'cusum', 'chow_test', 'zivot_andrews', 'mosum', 'wild_binary_segmentation', 
                               'prophet', 'ensemble', 'auto']
            raise ValueError(f"Unknown method: {method}. Choose from: {available_methods}")
        
        # Run detection
        return detector.detect(data, value_col, date_col)
    
    def _auto_select_detector(self, data: pd.DataFrame, value_col: str, **detector_params):
        """Auto-select best detector using Multi-Criteria Decision Matrix."""
        values = data[value_col].dropna().values
        n = len(values)
        
        if n < 10:
            raise ValueError("Not enough data for auto-selection")
        
        # Analyze data characteristics
        chars = self._analyze_data_characteristics(values)
        
        # All available methods
        all_methods = ['bai_perron', 'cusum', 'chow_test', 'zivot_andrews', 
                      'pelt', 'binary_segmentation', 'dynamic_programming', 
                      'mosum', 'wild_binary_segmentation', 'prophet']
        
        # Filter methods based on data size requirements
        valid_methods = []
        for method in all_methods:
            try:
                # Try creating detector to check if it can handle this data size
                temp_detector = self._create_detector_instance(method, {})
                # Prophet requires at least 30 observations
                if method == 'prophet' and n < 30:
                    continue
                # Zivot-Andrews requires enough points for regression
                if method == 'zivot_andrews' and n < 15:
                    continue
                # Other methods generally need at least 10 points (checked earlier)
                valid_methods.append(method)
            except Exception:
                # Skip methods that can't be created
                continue
        
        if not valid_methods:
            # Fallback to CUSUM if no methods pass validation
            valid_methods = ['cusum']
        
        # Score each valid method against data characteristics
        method_scores = {}
        for method in valid_methods:
            score = 0
            score += self._sample_size_score(method, chars['n_obs'])
            score += self._noise_tolerance_score(method, chars['noise_level'])
            score += self._trend_handling_score(method, chars['trend_strength'])
            score += self._seasonality_score(method, chars['seasonality'])
            score += self._computational_score(method, chars['n_obs'])
            score += self._stationarity_score(method, chars['stationarity'])
            score += self._outlier_robustness_score(method, chars['outlier_ratio'])
            
            method_scores[method] = score
        
        # Select method with highest score
        best_method = max(method_scores, key=method_scores.get)
        
        # Create detector instance based on selected method
        return self._create_detector_instance(best_method, detector_params)
    
    def _analyze_data_characteristics(self, values: np.ndarray) -> Dict[str, float]:
        """Analyze data characteristics for auto-selection."""
        from scipy import stats
        from statsmodels.tsa.stattools import adfuller
        
        n = len(values)
        chars = {'n_obs': n}
        
        # Noise level (coefficient of variation)
        mean_val = np.mean(values)
        std_val = np.std(values)
        chars['noise_level'] = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else 1.0
        
        # Trend strength (correlation with time)
        time_index = np.arange(n)
        trend_corr = abs(stats.pearsonr(time_index, values)[0]) if n > 2 else 0
        chars['trend_strength'] = trend_corr
        
        # Seasonality (simplified - check for periodic patterns)
        chars['seasonality'] = 0.0  # Simple placeholder
        if n >= 24:  # Need enough data
            # Check for weekly pattern (if daily data)
            if n >= 14:
                weekly_diff = np.mean([abs(values[i] - values[i+7]) for i in range(n-7)]) if n >= 14 else std_val
                chars['seasonality'] = max(0, 1 - weekly_diff / std_val) if std_val > 0 else 0
        
        # Stationarity (ADF test p-value, lower = more stationary)
        try:
            adf_stat, adf_pvalue = adfuller(values, autolag='AIC')[:2]
            chars['stationarity'] = adf_pvalue  # 0 = stationary, 1 = non-stationary
        except:
            chars['stationarity'] = 0.5  # Unknown
        
        # Outlier ratio
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25
        outliers = np.sum((values < q25 - 1.5*iqr) | (values > q75 + 1.5*iqr))
        chars['outlier_ratio'] = outliers / n
        
        return chars
    
    def _sample_size_score(self, method: str, n_obs: int) -> float:
        """Score method based on sample size suitability."""
        scores = {
            'bai_perron': 0.9 if 100 <= n_obs <= 1000 else 0.6 if 50 <= n_obs < 100 else 0.3,
            'cusum': 0.9 if n_obs >= 20 else 0.2,
            'chow_test': 0.8 if n_obs >= 40 else 0.4,
            'zivot_andrews': 0.8 if n_obs >= 30 else 0.3,
            'pelt': 0.9 if n_obs >= 50 else 0.6,
            'binary_segmentation': 0.8 if n_obs >= 30 else 0.5,
            'dynamic_programming': 0.7 if n_obs >= 50 else 0.4,
            'mosum': 0.8 if n_obs >= 40 else 0.3,
            'wild_binary_segmentation': 0.8 if n_obs >= 100 else 0.4,
            'prophet': 0.9 if n_obs >= 100 else 0.4 if n_obs >= 50 else 0.1
        }
        return scores.get(method, 0.5)
    
    def _noise_tolerance_score(self, method: str, noise_level: float) -> float:
        """Score method based on noise tolerance."""
        # Higher noise_level = more noise, need robust methods
        if noise_level < 0.2:  # Clean data
            scores = {
                'bai_perron': 0.9, 'chow_test': 0.8, 'zivot_andrews': 0.8,
                'cusum': 0.7, 'pelt': 0.8, 'binary_segmentation': 0.7,
                'dynamic_programming': 0.8, 'mosum': 0.6, 'wild_binary_segmentation': 0.5,
                'prophet': 0.6
            }
        elif noise_level < 0.5:  # Moderate noise
            scores = {
                'bai_perron': 0.6, 'chow_test': 0.7, 'zivot_andrews': 0.6,
                'cusum': 0.8, 'pelt': 0.9, 'binary_segmentation': 0.8,
                'dynamic_programming': 0.8, 'mosum': 0.7, 'wild_binary_segmentation': 0.8,
                'prophet': 0.8
            }
        else:  # High noise
            scores = {
                'bai_perron': 0.3, 'chow_test': 0.4, 'zivot_andrews': 0.4,
                'cusum': 0.6, 'pelt': 0.7, 'binary_segmentation': 0.7,
                'dynamic_programming': 0.6, 'mosum': 0.6, 'wild_binary_segmentation': 0.9,
                'prophet': 0.8
            }
        return scores.get(method, 0.5)
    
    def _trend_handling_score(self, method: str, trend_strength: float) -> float:
        """Score method based on trend handling capability."""
        if trend_strength < 0.2:  # No trend
            return 0.7  # All methods OK
        elif trend_strength < 0.6:  # Moderate trend
            scores = {
                'bai_perron': 0.7, 'chow_test': 0.8, 'zivot_andrews': 0.6,
                'cusum': 0.8, 'pelt': 0.7, 'binary_segmentation': 0.7,
                'dynamic_programming': 0.7, 'mosum': 0.7, 'wild_binary_segmentation': 0.6,
                'prophet': 0.9
            }
        else:  # Strong trend
            scores = {
                'bai_perron': 0.5, 'chow_test': 0.6, 'zivot_andrews': 0.4,
                'cusum': 0.6, 'pelt': 0.5, 'binary_segmentation': 0.5,
                'dynamic_programming': 0.5, 'mosum': 0.6, 'wild_binary_segmentation': 0.4,
                'prophet': 1.0
            }
        return scores.get(method, 0.5)
    
    def _seasonality_score(self, method: str, seasonality: float) -> float:
        """Score method based on seasonality handling."""
        if seasonality < 0.5:  # Low/no seasonality
            return 0.7  # All methods OK
        else:  # Strong seasonality
            scores = {
                'bai_perron': 0.4, 'chow_test': 0.5, 'zivot_andrews': 0.3,
                'cusum': 0.5, 'pelt': 0.6, 'binary_segmentation': 0.6,
                'dynamic_programming': 0.6, 'mosum': 0.5, 'wild_binary_segmentation': 0.5,
                'prophet': 0.9  # Prophet handles seasonality well (reduced from 1.0)
            }
        return scores.get(method, 0.5)
    
    def _computational_score(self, method: str, n_obs: int) -> float:
        """Score method based on computational efficiency."""
        if n_obs < 100:  # Small data - efficiency not critical
            return 0.7
        elif n_obs < 1000:  # Medium data
            scores = {
                'bai_perron': 0.6, 'chow_test': 0.7, 'zivot_andrews': 0.8,
                'cusum': 0.9, 'pelt': 0.9, 'binary_segmentation': 0.8,
                'dynamic_programming': 0.6, 'mosum': 0.7, 'wild_binary_segmentation': 0.5,
                'prophet': 0.7
            }
        else:  # Large data - need efficiency
            scores = {
                'bai_perron': 0.4, 'chow_test': 0.5, 'zivot_andrews': 0.6,
                'cusum': 0.8, 'pelt': 1.0, 'binary_segmentation': 0.9,
                'dynamic_programming': 0.4, 'mosum': 0.6, 'wild_binary_segmentation': 0.3,
                'prophet': 0.6
            }
        return scores.get(method, 0.5)
    
    def _stationarity_score(self, method: str, stationarity_pvalue: float) -> float:
        """Score method based on stationarity test results."""
        if stationarity_pvalue > 0.05:  # Non-stationary (unit root suspected)
            scores = {
                'bai_perron': 0.3, 'chow_test': 0.4, 'zivot_andrews': 1.0,  # ZA designed for unit roots
                'cusum': 0.5, 'pelt': 0.6, 'binary_segmentation': 0.6,
                'dynamic_programming': 0.6, 'mosum': 0.5, 'wild_binary_segmentation': 0.5,
                'prophet': 0.8  # Prophet handles non-stationarity well
            }
        else:  # Stationary
            scores = {
                'bai_perron': 0.9, 'chow_test': 0.8, 'zivot_andrews': 0.6,
                'cusum': 0.8, 'pelt': 0.8, 'binary_segmentation': 0.8,
                'dynamic_programming': 0.8, 'mosum': 0.8, 'wild_binary_segmentation': 0.7,
                'prophet': 0.7
            }
        return scores.get(method, 0.5)
    
    def _outlier_robustness_score(self, method: str, outlier_ratio: float) -> float:
        """Score method based on outlier robustness."""
        if outlier_ratio < 0.05:  # Few outliers
            return 0.7  # All methods OK
        else:  # Many outliers - need robustness
            scores = {
                'bai_perron': 0.3, 'chow_test': 0.4, 'zivot_andrews': 0.4,
                'cusum': 0.6, 'pelt': 0.7, 'binary_segmentation': 0.7,
                'dynamic_programming': 0.7, 'mosum': 0.6, 'wild_binary_segmentation': 0.9,
                'prophet': 0.8  # Prophet has outlier detection
            }
        return scores.get(method, 0.5)
    
    def _create_detector_instance(self, method: str, detector_params: Dict):
        """Create detector instance for the selected method."""
        if method == 'bai_perron':
            return BaiPerronDetector(**detector_params)
        elif method == 'cusum':
            return CUSUMDetector(**detector_params)
        elif method == 'chow_test':
            return ChowTestDetector(**detector_params)
        elif method == 'zivot_andrews':
            return ZivotAndrewsDetector(**detector_params)
        elif method == 'pelt':
            return PeltDetector(**detector_params)
        elif method == 'binary_segmentation':
            return BinarySegmentationDetector(**detector_params)
        elif method == 'dynamic_programming':
            return DynamicProgrammingDetector(**detector_params)
        elif method == 'mosum':
            return MOSUMDetector(**detector_params)
        elif method == 'wild_binary_segmentation':
            return WildBinarySegmentationDetector(**detector_params)
        elif method == 'prophet':
            return ProphetChangepointDetector(**detector_params)
        else:
            raise ValueError(f"Unknown method for auto-selection: {method}")
