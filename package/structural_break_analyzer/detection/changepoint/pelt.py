from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

from ..base import DetectionMethod, BreakPoint
from ...timeseries.preprocessor import TimeSeriesPreprocessor


class PeltDetector(DetectionMethod):
    """
    Wrapper around ruptures' PELT algorithm.

    Parameters
    ----------
    model : str
        Cost model for ruptures (e.g., "l2", "l1", "rbf", "linear", "normal", "ar").
    penalty : Optional[float]
        Penalty value controlling the number of change points. Use either `penalty` OR `n_bkps`.
    n_bkps : Optional[int]
        Number of breakpoints to detect. If provided, `penalty` is ignored.
    min_size : Optional[int]
        Minimum segment size (passed to ruptures). If None, ruptures default is used.
    jump : int
        Subsampling interval for speed/robustness (ruptures parameter). 1 = no subsampling.
    """

    def __init__(self,
                 model: str = "l2",
                 penalty: Optional[float] = None,
                 n_bkps: Optional[int] = None,
                 min_size: Optional[int] = None,
                 jump: int = 1):
        super().__init__(
            model=model,
            penalty=penalty,
            n_bkps=n_bkps,
            min_size=min_size,
            jump=jump
        )
        self.name = 'pelt'  # Override class name with user-friendly name
        self.preprocessor = TimeSeriesPreprocessor()

        # soft import so base package can be installed without ruptures
        try:
            import ruptures as rpt  # noqa: F401
        except Exception as e:
            raise ImportError(
                "PELT requires the 'ruptures' package. Install it with:\n"
                "    pip install ruptures\n"
                f"Underlying error: {e}"
            )

    # ---------- Preprocessing (reuse your existing pipeline) ----------
    def _preprocess(self, data: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, Dict]:
        values = data[value_col].values.astype(float)
        meta: Dict = {
            'original_length': len(values),
            'transforms': [],
            'notes': []
        }

        # 1) Handle missing (model-agnostic, but careful about long gaps)
        if np.any(np.isnan(values)):
            # Heuristic: if <= 5 consecutive NaNs -> interpolate, else forward-fill then back-fill
            isnan = np.isnan(values)
            # run-lengths of NaNs
            gaps = np.diff(np.where(np.concatenate(([isnan[0]],
                                                    isnan[:-1] != isnan[1:], [True])))[0])[::2] if isnan.any() else []
            long_gaps = any(g > 5 for g in gaps) if len(gaps) else False

            if long_gaps:
                # forward then backward fill
                meta['notes'].append('long_nan_gaps_ffill_bfill')
                mask = np.isnan(values)
                idx = np.where(~mask, np.arange(mask.size), 0)
                np.maximum.accumulate(idx, out=idx)
                ffilled = values[idx]
                # back-fill by reversing
                mask2 = np.isnan(ffilled[::-1])
                idx2 = np.where(~mask2, np.arange(mask2.size), 0)
                np.maximum.accumulate(idx2, out=idx2)
                values = ffilled[::-1][idx2][::-1]
                meta['transforms'].append({'missing': 'ffill_bfill'})
            else:
                # linear interpolation
                nans = np.isnan(values)
                x = lambda z: z.nonzero()[0]
                valid = x(~nans)
                if len(valid) == 0:
                    # all NaN, nothing to do
                    meta['notes'].append('all_nan')
                    return values, meta
                values[nans] = np.interp(x(nans), valid, values[valid])
                meta['transforms'].append({'missing': 'interpolate'})

        # 2) Outliers (model-aware)
        model = self.params.get('model', 'l2')

        def soft_clip(x, q=0.999):
            lo, hi = np.quantile(x, 1 - q), np.quantile(x, q)
            return np.clip(x, lo, hi), {'clip_lo': float(lo), 'clip_hi': float(hi)}

        if model in ('l2', 'linear', 'rbf', 'ar'):
            # light-touch: robust z or soft clip to avoid break inflation
            cleaned, mask = self.preprocessor.remove_outliers(values, method='zscore', threshold=4.0)
            # If many points flagged, fall back to soft clip (less likely to create artifacts)
            if (~mask).sum() > max(1, 0.01 * len(values)):
                cleaned, clip_meta = soft_clip(values, q=0.995)
                meta['transforms'].append({'outliers': {'method': 'soft_clip', **clip_meta}})
                values = cleaned
            else:
                values = cleaned
                meta['transforms'].append(
                    {'outliers': {'method': 'zscore', 'threshold': 4.0, 'n_altered': int((~mask).sum())}})
        elif model == 'normal':
            # preserve variance structure; only clip extreme spikes
            values, clip_meta = soft_clip(values, q=0.999)
            meta['transforms'].append({'outliers': {'method': 'soft_clip', **clip_meta}})
        elif model == 'l1':
            # very robust model; minimal outlier handling needed
            meta['transforms'].append({'outliers': 'none'})

        # 3) Stationarity / trend (strictly model-aware)
        # - l2/l1/normal/rbf: do NOT difference (we want level/variance/distribution changes)
        # - linear: do NOT detrend (the model handles trend changes)
        # - ar: no differencing; AR is modeled by the cost
        # Optional: mild detrend for l2 if there is a strong global trend AND you only care about mean shifts
        if model == 'l2':
            # quick trend check via R^2 of linear fit; detrend only if very strong (>0.9)
            x = np.arange(len(values))
            if len(values) > 2:
                coeffs = np.polyfit(x, values, 1)
                fit = np.polyval(coeffs, x)
                ss_tot = np.sum((values - values.mean()) ** 2) + 1e-12
                ss_res = np.sum((values - fit) ** 2)
                r2 = 1 - ss_res / ss_tot
                meta['trend_r2'] = float(r2)
                if r2 > 0.9:
                    values = values - fit
                    meta['transforms'].append({'detrend': 'linear', 'coeffs': [float(c) for c in coeffs]})
            # else: keep as-is

        # 4) Scaling (numeric stability)
        # - Avoid scaling that destroys the signal for variance/distribution models.
        if model in ('l2', 'l1', 'linear', 'ar'):
            values, norm = self.preprocessor.normalize(values, method='standard')
            meta['transforms'].append({'normalize': {'method': 'standard', **{k: float(v) for k, v in norm.items()}}})
        elif model in ('normal', 'rbf'):
            # standardization is still ok; it won’t remove relative variance differences across segments
            values, norm = self.preprocessor.normalize(values, method='standard')
            meta['transforms'].append({'normalize': {'method': 'standard', **{k: float(v) for k, v in norm.items()}}})

        meta['final_length'] = len(values)
        return values, meta

    # ---------- Core detection with ruptures.Pelt ----------
    def _detect_breaks(self, data: np.ndarray) -> List[int]:
        import ruptures as rpt

        X = data.reshape(-1, 1) if data.ndim == 1 else data
        n = X.shape[0]

        # robust default if user didn't set it
        user_min_size = self.params.get('min_size')
        if user_min_size is None:
            # pick a sensible default: at least 2, or ~2% of series length
            user_min_size = max(2, int(0.02 * n))

        algo = rpt.Pelt(
            model=self.params['model'],
            min_size=int(user_min_size),  # ensure int, not None
            jump=int(self.params.get('jump', 1))
        ).fit(X)

        penalty = self.params.get('penalty')
        n_bkps = self.params.get('n_bkps')

        if n_bkps is not None:
            bkps = algo.predict(n_bkps=int(n_bkps))
        else:
            if penalty is None:
                penalty = 3 * np.log(n)  # default
            bkps = algo.predict(pen=float(penalty))

        n = X.shape[0]
        bkps = [int(b) for b in bkps if 0 < int(b) < n]  # drop trailing n and invalids
        return sorted(bkps)

    # ---------- Stats + confidence (heuristic) ----------
    def _calculate_statistics(self, data: np.ndarray, breaks: List[int], preprocess_meta: Dict) -> Dict:
        stats: Dict = super()._calculate_statistics(data, breaks, preprocess_meta)
        stats.update({
            'algorithm': 'PELT',
            'model': self.params['model'],
            'penalty': self.params.get('penalty', None),
            'n_bkps_requested': self.params.get('n_bkps', None),
            'min_size': self.params.get('min_size', None),
            'jump': self.params.get('jump', 1),
        })

        # Optional: compute per-segment mean/std for convenience
        bounds = [0] + sorted(breaks) + [len(data)]
        segments = []
        for i in range(len(bounds) - 1):
            seg = data[bounds[i]:bounds[i + 1]]
            segments.append({
                'start': bounds[i],
                'end': bounds[i + 1],
                'mean': float(np.mean(seg)) if len(seg) else np.nan,
                'std': float(np.std(seg)) if len(seg) else np.nan,
                'length': int(len(seg)),
            })
        stats['segments'] = segments
        return stats

    def _create_breakpoints(self, indices: List[int], original_data: pd.DataFrame,
                            processed_values: np.ndarray, value_col: str, date_col: str) -> List[BreakPoint]:
        # Reuse base to compute direction & magnitude
        breaks = super()._create_breakpoints(indices, original_data, processed_values, value_col, date_col)

        # Optionally add a simple confidence heuristic:
        # Compare local window means around the break; larger standardized jump → higher confidence.
        win = 5
        n = len(processed_values)
        for bp in breaks:
            i = bp.index
            left = processed_values[max(0, i - win): i]
            right = processed_values[i: min(n, i + win)]
            if len(left) > 0 and len(right) > 0:
                jump = abs(np.mean(right) - np.mean(left))
                local_std = np.std(np.concatenate([left, right])) + 1e-8
                z = jump / local_std
                # squash to (0,1)
                conf = float(1 - np.exp(-z))
                bp.confidence = max(0.0, min(1.0, conf))
                bp.statistics = {'local_z': z}
            else:
                bp.confidence = 0.0
                bp.statistics = {'local_z': np.nan}

        return breaks
