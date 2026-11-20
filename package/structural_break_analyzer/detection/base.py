from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class BreakPoint:
    """Represents a single structural break."""
    date: datetime  # When the break occurred
    index: int  # Position in series
    confidence: float  # 0-1 confidence score
    magnitude: float  # Size of change
    direction: str  # 'up' or 'down'
    method: str  # Detection method used
    statistics: Dict[str, Any] = None  # Method-specific stats

    def __post_init__(self):
        if self.statistics is None:
            self.statistics = {}


@dataclass
class DetectionResult:
    """Container for detection results."""
    breaks: List[BreakPoint]
    method: str
    statistics: Dict[str, Any]
    preprocessed_data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def n_breaks(self) -> int:
        return len(self.breaks)

    def summary(self) -> str:
        """Generate a summary of detection results."""
        dates = [b.date.strftime('%Y-%m-%d') if hasattr(b.date, 'strftime') else str(b.date) 
                for b in self.breaks]
        return (f"Method: {self.method}\n"
                f"Breaks detected: {self.n_breaks}\n"
                f"Break dates: {dates}")


class DetectionMethod(ABC):
    """Abstract base class for all detection methods."""

    def __init__(self, **kwargs):
        """Initialize with method-specific parameters."""
        self.params = kwargs
        self.name = self.__class__.__name__

    @abstractmethod
    def _detect_breaks(self, data: np.ndarray) -> List[int]:
        """
        Core detection logic - works on preprocessed numpy array.
        Returns list of break indices.
        """
        pass

    @abstractmethod
    def _preprocess(self, data: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, Dict]:
        """
        Method-specific preprocessing.
        Returns preprocessed array and preprocessing metadata.
        """
        pass

    def detect(self,
               data,
               value_col: str = 'value',
               date_col: str = 'date') -> DetectionResult:
        """
        Main detection interface.

        Args:
            data: DataFrame with time series
            value_col: Column name for values
            date_col: Column name for dates

        Returns:
            DetectionResult with breaks and statistics
        """
        # Coerce input to DataFrame if needed
        if isinstance(data, np.ndarray):
            if data.ndim != 1:
                raise ValueError("NumPy array input must be 1-dimensional")
            data = pd.DataFrame({
                date_col: np.arange(len(data)),
                value_col: data.astype(float)
            })
        elif isinstance(data, pd.Series):
            df = pd.DataFrame({value_col: data.values})
            if isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)) or np.issubdtype(data.index.dtype, np.datetime64):
                df[date_col] = pd.to_datetime(data.index)
            else:
                df[date_col] = np.arange(len(data))
            data = df[[date_col, value_col]]

        # Validate input
        self._validate_input(data, value_col, date_col)

        # Store original for reference
        original_data = data.copy()

        # Preprocess data
        processed_values, preprocess_meta = self._preprocess(data, value_col)

        # Detect breaks on preprocessed data
        break_indices = self._detect_breaks(processed_values)

        # Convert indices to BreakPoint objects
        breaks = self._create_breakpoints(
            break_indices,
            original_data,
            processed_values,
            value_col,
            date_col
        )

        # Calculate method-specific statistics
        statistics = self._calculate_statistics(
            processed_values,
            break_indices,
            preprocess_meta
        )

        return DetectionResult(
            breaks=breaks,
            method=self.name,
            statistics=statistics,
            preprocessed_data=pd.DataFrame({
                date_col: data[date_col][:len(processed_values)],  # Match lengths
                f'{value_col}_processed': processed_values
            }),
            metadata=preprocess_meta
        )

    def _validate_input(self, data: pd.DataFrame, value_col: str, date_col: str):
        """Validate input data."""
        if value_col not in data.columns:
            raise ValueError(f"Column '{value_col}' not found in data")
        if date_col not in data.columns:
            raise ValueError(f"Column '{date_col}' not found in data")
        if data[value_col].isna().all():
            raise ValueError(f"Column '{value_col}' contains only NaN values")
        if len(data) < 10:  # Arbitrary minimum
            raise ValueError(f"Time series too short ({len(data)} points)")

    def _create_breakpoints(self,
                            indices: List[int],
                            original_data: pd.DataFrame,
                            processed_values: np.ndarray,
                            value_col: str,
                            date_col: str) -> List[BreakPoint]:
        """Convert break indices to BreakPoint objects."""
        breaks = []

        for idx in indices:
            if idx <= 0 or idx >= len(original_data):
                continue

            # Calculate magnitude and direction
            before_mean = np.mean(processed_values[max(0, idx - 5):idx])
            after_mean = np.mean(processed_values[idx:min(len(processed_values), idx + 5)])
            magnitude = abs(after_mean - before_mean)
            direction = 'up' if after_mean > before_mean else 'down'

            # Ensure date is a datetime object
            date_value = original_data.iloc[idx][date_col]
            if isinstance(date_value, str):
                try:
                    date_value = pd.to_datetime(date_value)
                except:
                    date_value = datetime.now()  # Fallback
            elif not isinstance(date_value, datetime):
                try:
                    date_value = pd.to_datetime(date_value)
                except:
                    date_value = datetime.now()  # Fallback
            
            break_point = BreakPoint(
                date=date_value,
                index=idx,
                confidence=0.0,  # To be set by specific method
                magnitude=magnitude,
                direction=direction,
                method=self.name
            )
            breaks.append(break_point)

        return breaks

    def _calculate_statistics(self,
                              data: np.ndarray,
                              breaks: List[int],
                              preprocess_meta: Dict) -> Dict[str, Any]:
        """Calculate general statistics. Override for method-specific stats."""
        return {
            'n_observations': len(data),
            'n_breaks': len(breaks),
            'preprocessing': preprocess_meta
        }
