import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import signal
from scipy.stats import zscore


class TimeSeriesPreprocessor:
    """Common preprocessing utilities for time series data."""

    @staticmethod
    def handle_missing(data: np.ndarray, method: str = 'interpolate') -> np.ndarray:
        """
        Handle missing values.

        Args:
            data: Input array
            method: 'interpolate', 'forward_fill', 'backward_fill', 'drop'
        """
        if not np.any(np.isnan(data)):
            return data

        if method == 'interpolate':
            # Linear interpolation
            nans = np.isnan(data)
            x = lambda z: z.nonzero()[0]
            valid_indices = x(~nans)
            if len(valid_indices) == 0:
                # All values are NaN - cannot interpolate
                return data  # Return as-is
            elif len(valid_indices) == 1:
                # Only one valid value - fill all NaN with this value
                data[nans] = data[valid_indices[0]]
            else:
                data[nans] = np.interp(x(nans), valid_indices, data[valid_indices])
        elif method == 'forward_fill':
            mask = np.isnan(data)
            idx = np.where(~mask, np.arange(mask.size), 0)
            np.maximum.accumulate(idx, out=idx)
            data = data[idx]
        elif method == 'backward_fill':
            # Reverse, forward fill, reverse back
            data = data[::-1]
            mask = np.isnan(data)
            idx = np.where(~mask, np.arange(mask.size), 0)
            np.maximum.accumulate(idx, out=idx)
            data = data[idx][::-1]
        elif method == 'drop':
            data = data[~np.isnan(data)]

        return data

    @staticmethod
    def remove_outliers(data: np.ndarray, method: str = 'zscore', threshold: float = 3.0) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Remove outliers from data.

        Args:
            data: Input array
            method: 'zscore', 'iqr', 'isolation'
            threshold: Threshold for outlier detection

        Returns:
            Cleaned data and mask of outliers
        """
        if method == 'zscore':
            if len(data) <= 1:
                # Can't calculate zscore for single value or empty array
                mask = np.ones(len(data), dtype=bool)
            else:
                z_scores = np.abs(zscore(data))
                mask = z_scores < threshold
        elif method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            mask = (data >= lower) & (data <= upper)
        else:
            mask = np.ones(len(data), dtype=bool)

        # Interpolate outliers instead of removing
        cleaned = data.copy()
        if not mask.all():
            outlier_indices = np.where(~mask)[0]
            for idx in outlier_indices:
                # Use neighboring values
                start = max(0, idx - 1)
                end = min(len(data), idx + 2)
                neighbors = [data[i] for i in range(start, end) if i != idx and mask[i]]
                if neighbors:
                    cleaned[idx] = np.mean(neighbors)

        return cleaned, mask

    @staticmethod
    def detrend(data: np.ndarray, method: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove trend from data.

        Args:
            data: Input array
            method: 'linear', 'polynomial', 'hp_filter', 'difference'

        Returns:
            Detrended data and trend component
        """
        if method == 'linear':
            # Linear detrending
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 1)
            trend = np.polyval(coeffs, x)
            detrended = data - trend
        elif method == 'polynomial':
            # Polynomial detrending (degree 2)
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 2)
            trend = np.polyval(coeffs, x)
            detrended = data - trend
        elif method == 'hp_filter':
            # Hodrick-Prescott filter
            # Simplified version - you might want statsmodels.api.tsa.filters.hpfilter
            trend = signal.savgol_filter(data, window_length=min(31, len(data) // 4 * 2 + 1), polyorder=3)
            detrended = data - trend
        elif method == 'difference':
            # First differencing
            detrended = np.diff(data, prepend=data[0])
            trend = np.zeros_like(data)
        else:
            detrended = data
            trend = np.zeros_like(data)

        return detrended, trend

    @staticmethod
    def normalize(data: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Dict]:
        """
        Normalize data.

        Args:
            data: Input array
            method: 'standard', 'minmax', 'robust'

        Returns:
            Normalized data and normalization parameters
        """
        if method == 'standard':
            mean = np.mean(data)
            std = np.std(data)
            normalized = (data - mean) / (std + 1e-8)
            params = {'mean': mean, 'std': std}
        elif method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            params = {'min': min_val, 'max': max_val}
        elif method == 'robust':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            normalized = (data - median) / (mad + 1e-8)
            params = {'median': median, 'mad': mad}
        else:
            normalized = data
            params = {}

        return normalized, params

    @staticmethod
    def test_stationarity(data: np.ndarray) -> Dict:
        """
        Test for stationarity using ADF and KPSS tests.

        Returns:
            Dictionary with test results
        """
        # Handle edge cases
        if len(data) == 0 or np.all(np.isnan(data)):
            return {
                'adf_statistic': np.nan,
                'adf_pvalue': np.nan,
                'adf_is_stationary': False,
                'kpss_statistic': np.nan,
                'kpss_pvalue': np.nan,
                'kpss_is_stationary': False,
                'is_stationary': False
            }
        
        # Remove NaN values for testing
        clean_data = data[~np.isnan(data)]
        if len(clean_data) < 10:  # Too few observations for reliable test
            return {
                'adf_statistic': np.nan,
                'adf_pvalue': np.nan,
                'adf_is_stationary': False,
                'kpss_statistic': np.nan,
                'kpss_pvalue': np.nan,
                'kpss_is_stationary': False,
                'is_stationary': False
            }
        
        try:
            # ADF test (null: non-stationary)
            adf_result = adfuller(clean_data, autolag='AIC')

            # KPSS test (null: stationary)
            kpss_result = kpss(clean_data, regression='c', nlags='auto')

            return {
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'adf_is_stationary': adf_result[1] < 0.05,
                'kpss_statistic': kpss_result[0],
                'kpss_pvalue': kpss_result[1],
                'kpss_is_stationary': kpss_result[1] > 0.05,
                'is_stationary': (adf_result[1] < 0.05) and (kpss_result[1] > 0.05)
            }
        except Exception:
            # If tests fail for any reason, assume non-stationary
            return {
                'adf_statistic': np.nan,
                'adf_pvalue': np.nan,
                'adf_is_stationary': False,
                'kpss_statistic': np.nan,
                'kpss_pvalue': np.nan,
                'kpss_is_stationary': False,
                'is_stationary': False
            }

    @staticmethod
    def make_stationary(data: np.ndarray, max_diff: int = 2) -> Tuple[np.ndarray, int]:
        """
        Transform data to stationary if needed.

        Args:
            data: Input array
            max_diff: Maximum differencing order

        Returns:
            Stationary data and differencing order used
        """
        diff_order = 0
        current_data = data.copy()

        for d in range(max_diff + 1):
            stat_test = TimeSeriesPreprocessor.test_stationarity(current_data)
            if stat_test['is_stationary']:
                return current_data, d
            if d < max_diff:
                current_data = np.diff(current_data)
                diff_order += 1

        return current_data, diff_order