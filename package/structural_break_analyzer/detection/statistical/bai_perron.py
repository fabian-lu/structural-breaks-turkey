import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy import stats
from ..base import DetectionMethod, BreakPoint
from ...timeseries.preprocessor import TimeSeriesPreprocessor


class BaiPerronDetector(DetectionMethod):
    """
    Bai-Perron structural break test implementation.

    This method tests for multiple structural breaks in linear regression models.
    It's particularly good for detecting changes in mean/trend relationships.
    """

    def __init__(self,
                 max_breaks: int = 5,
                 min_segment_size: Optional[int] = None,
                 significance_level: float = 0.05,
                 h: Optional[float] = None):
        """
        Initialize Bai-Perron detector.

        Args:
            max_breaks: Maximum number of breaks to search for
            min_segment_size: Minimum observations between breaks (default: 15% of series)
            significance_level: Significance level for F-tests
            h: Trimming parameter (0.15 means 15% trimming on each end)
        """
        super().__init__(
            max_breaks=max_breaks,
            min_segment_size=min_segment_size,
            significance_level=significance_level,
            h=h or 0.15
        )
        self.name = 'bai_perron'  # Override class name with user-friendly name
        self.preprocessor = TimeSeriesPreprocessor()

    def _preprocess(self, data: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess data for Bai-Perron test.

        Bai-Perron typically needs:
        1. No missing values
        2. Outlier handling
        3. Stationarity (optional but recommended)
        """
        values = data[value_col].values.astype(float)
        metadata = {'original_length': len(values)}

        # Handle missing values
        values = self.preprocessor.handle_missing(values, method='interpolate')
        metadata['missing_handled'] = True

        # Remove outliers (but keep track of them)
        values, outlier_mask = self.preprocessor.remove_outliers(values, method='zscore', threshold=3.5)
        metadata['outliers_removed'] = (~outlier_mask).sum()

        # Check stationarity
        stat_test = self.preprocessor.test_stationarity(values)
        metadata['stationarity'] = stat_test

        # If non-stationary, make it stationary
        if not stat_test['is_stationary']:
            values, diff_order = self.preprocessor.make_stationary(values, max_diff=1)
            metadata['differencing_order'] = diff_order
        else:
            metadata['differencing_order'] = 0

        # Normalize for numerical stability
        values, norm_params = self.preprocessor.normalize(values, method='standard')
        metadata['normalization'] = norm_params

        return values, metadata

    def _detect_breaks(self, data: np.ndarray) -> List[int]:
        """
        Core Bai-Perron detection algorithm.

        This is a simplified version. The full algorithm involves:
        1. Sequential F-test for l vs l+1 breaks
        2. Global optimization for each number of breaks
        3. Information criteria for model selection
        """
        n = len(data)

        # Set minimum segment size if not specified
        if self.params['min_segment_size'] is None:
            min_segment = int(n * self.params['h'])
        else:
            min_segment = self.params['min_segment_size']

        # Dynamic programming approach to find optimal breaks
        breaks = []

        for n_breaks in range(1, self.params['max_breaks'] + 1):
            candidate_breaks = self._find_n_breaks(data, n_breaks, min_segment)

            if candidate_breaks:
                # Test if these breaks are significant
                if self._test_breaks_significance(data, candidate_breaks):
                    breaks = candidate_breaks
                else:
                    break  # Stop if breaks are no longer significant

        return sorted(breaks)

    def _find_n_breaks(self, data: np.ndarray, n_breaks: int, min_segment: int) -> List[int]:
        """
        Find optimal placement of n breaks using dynamic programming.
        """
        n = len(data)

        if n_breaks == 1:
            # For single break, do exhaustive search
            return self._find_single_break(data, min_segment)

        # For multiple breaks, use recursive partitioning
        # This is a simplified version - full Bai-Perron uses global optimization
        breaks = []
        segments = [(0, n)]

        for _ in range(n_breaks):
            best_break = None
            best_improvement = -np.inf
            best_segment_idx = -1

            # Find the best segment to split
            for idx, (start, end) in enumerate(segments):
                if end - start < 2 * min_segment:
                    continue

                segment_data = data[start:end]
                break_point = self._find_single_break_in_segment(segment_data, min_segment)

                if break_point:
                    improvement = self._calculate_improvement(data, start, end, start + break_point)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_break = start + break_point
                        best_segment_idx = idx

            if best_break:
                breaks.append(best_break)
                # Split the segment
                start, end = segments[best_segment_idx]
                segments[best_segment_idx] = (start, best_break)
                segments.insert(best_segment_idx + 1, (best_break, end))
            else:
                break

        return breaks

    def _find_single_break(self, data: np.ndarray, min_segment: int) -> List[int]:
        """Find the optimal single break point."""
        n = len(data)
        best_break = None
        best_stat = -np.inf

        for i in range(min_segment, n - min_segment):
            stat = self._calculate_f_stat(data, [i])
            if stat > best_stat:
                best_stat = stat
                best_break = i

        if best_break and self._test_single_break_significance(data, best_break):
            return [best_break]
        return []

    def _find_single_break_in_segment(self, segment: np.ndarray, min_segment: int) -> Optional[int]:
        """Find single break within a segment."""
        n = len(segment)
        best_break = None
        best_stat = -np.inf

        for i in range(min_segment, n - min_segment):
            stat = self._calculate_f_stat(segment, [i])
            if stat > best_stat:
                best_stat = stat
                best_break = i

        return best_break

    def _calculate_f_stat(self, data: np.ndarray, breaks: List[int]) -> float:
        """
        Calculate F-statistic for given break points.

        Tests the null hypothesis of no breaks vs alternative of breaks at given points.
        """
        n = len(data)
        breaks = [0] + sorted(breaks) + [n]

        # Calculate SSR for model without breaks
        ssr_restricted = np.sum((data - np.mean(data)) ** 2)

        # Calculate SSR for model with breaks
        ssr_unrestricted = 0
        for i in range(len(breaks) - 1):
            segment = data[breaks[i]:breaks[i + 1]]
            if len(segment) > 0:
                ssr_unrestricted += np.sum((segment - np.mean(segment)) ** 2)

        # F-statistic
        k = len(breaks) - 2  # Number of breaks
        
        # Handle edge case where unrestricted SSR is zero (perfect fit)
        if ssr_unrestricted == 0:
            if ssr_restricted == 0:
                return 0.0  # No improvement possible
            else:
                return np.inf  # Perfect fit with breaks vs no fit
        
        # Correct Bai-Perron F-statistic formula
        # F = [(RSS_restricted - RSS_unrestricted) / k] / [RSS_unrestricted / (n - 2*k - 1)]
        # where k is number of breaks, n is sample size
        denominator = ssr_unrestricted / (n - 2*k - 1)
        if denominator == 0 or (n - 2*k - 1) <= 0:
            return 0.0
        
        f_stat = ((ssr_restricted - ssr_unrestricted) / k) / denominator

        return f_stat

    def _calculate_improvement(self, data: np.ndarray, start: int, end: int, break_point: int) -> float:
        """Calculate the improvement from adding a break."""
        segment = data[start:end]

        # SSR without break
        ssr_without = np.sum((segment - np.mean(segment)) ** 2)

        # SSR with break
        seg1 = segment[:break_point - start]
        seg2 = segment[break_point - start:]
        ssr_with = np.sum((seg1 - np.mean(seg1)) ** 2) + np.sum((seg2 - np.mean(seg2)) ** 2)

        return ssr_without - ssr_with

    def _test_breaks_significance(self, data: np.ndarray, breaks: List[int]) -> bool:
        """Test if the breaks are statistically significant."""
        if not breaks:
            return False

        f_stat = self._calculate_f_stat(data, breaks)

        # Critical value (simplified - should use Bai-Perron critical values)
        # This is approximation using F-distribution
        k = len(breaks)
        n = len(data)
        critical_value = stats.f.ppf(1 - self.params['significance_level'], k, n - k - 1)

        return bool(f_stat > critical_value)

    def _test_single_break_significance(self, data: np.ndarray, break_point: int) -> bool:
        """Test significance of a single break."""
        return self._test_breaks_significance(data, [break_point])

    def _calculate_statistics(self, data: np.ndarray, breaks: List[int], preprocess_meta: Dict) -> Dict:
        """Calculate Bai-Perron specific statistics."""
        stats = super()._calculate_statistics(data, breaks, preprocess_meta)

        if breaks:
            # Add F-statistics
            stats['f_statistic'] = self._calculate_f_stat(data, breaks)

            # Add segment statistics
            segments = []
            breaks_with_bounds = [0] + sorted(breaks) + [len(data)]

            for i in range(len(breaks_with_bounds) - 1):
                segment = data[breaks_with_bounds[i]:breaks_with_bounds[i + 1]]
                segments.append({
                    'start': breaks_with_bounds[i],
                    'end': breaks_with_bounds[i + 1],
                    'mean': float(np.mean(segment)),
                    'std': float(np.std(segment)),
                    'length': len(segment)
                })

            stats['segments'] = segments

        return stats

    def _create_breakpoints(self, indices: List[int], original_data: pd.DataFrame,
                            processed_values: np.ndarray, value_col: str, date_col: str) -> List[BreakPoint]:
        """Override to add confidence scores based on F-statistics."""
        breaks = super()._create_breakpoints(indices, original_data, processed_values, value_col, date_col)

        # Calculate confidence for each break
        for break_point in breaks:
            # Individual F-stat for this break
            f_stat = self._calculate_f_stat(processed_values, [break_point.index])

            # Convert to confidence score (0-1)
            # Using F-distribution CDF
            n = len(processed_values)
            p_value = 1 - stats.f.cdf(f_stat, 1, n - 2)
            confidence = 1 - p_value

            break_point.confidence = min(max(confidence, 0), 1)
            break_point.statistics = {
                'f_statistic': f_stat,
                'p_value': p_value
            }

        return breaks