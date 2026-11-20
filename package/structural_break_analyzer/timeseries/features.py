import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import warnings

# Try to import optional dependencies for enhanced features
try:
    from statsmodels.tsa.seasonal import STL, seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
    HAS_STATTOOLS = True
except ImportError:
    HAS_STATTOOLS = False

# Additional imports for new MVP features
try:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    mutual_info_regression = None
    LinearRegression = None

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import rankdata, norm, f
    HAS_SCIPY_DISTANCE = True
except ImportError:
    HAS_SCIPY_DISTANCE = False
    pdist = None
    squareform = None
    rankdata = None
    norm = None
    f = None

from .preprocessor import TimeSeriesPreprocessor
from datetime import datetime, timezone

warnings.filterwarnings('ignore')


@dataclass
class TimeSeriesFeatures:
    """Container for all time series features."""
    
    # Basic statistics (required fields first)
    mean: float
    std: float
    min: float
    max: float
    skewness: float
    kurtosis: float
    
    # Trend features (required fields)
    trend_strength: float
    trend_slope: float
    trend_r2: float
    
    # Seasonality features (required fields)
    has_seasonality: bool
    seasonal_period: Optional[int]
    seasonal_strength: float
    
    # Volatility features (required fields)
    volatility_mean: float
    volatility_std: float
    volatility_clustering: float
    
    # Autocorrelation features (required fields)
    acf_1: float  # First lag autocorrelation
    acf_5: float  # 5th lag
    pacf_1: float
    pacf_5: float
    significant_lags: List[int]
    
    # Stationarity (required fields)
    is_stationary: bool
    adf_pvalue: float
    kpss_pvalue: float
    
    # Optional fields with defaults (must come after all required fields)
    trend_acceleration: float = 0.0  # Second derivative
    seasonal_periods: List[int] = field(default_factory=list)  # Multiple periods
    dominant_frequencies: List[float] = field(default_factory=list)
    volatility_persistence: float = 0.0  # Long memory in volatility
    ljung_box_pvalue: float = 1.0  # Test for serial correlation
    num_differences_needed: int = 0
    hurst_exponent: float = 0.5  # Long memory parameter
    entropy: float = 0.0  # Sample entropy
    complexity: float = 0.0  # Lempel-Ziv complexity
    decomposition: Optional[Dict[str, np.ndarray]] = None
    
    # NEW ENHANCED FEATURES FROM MVP
    # Non-linearity tests
    bds_test: Dict[str, Any] = field(default_factory=dict)
    mcleod_li_test: Dict[str, Any] = field(default_factory=dict)
    terasvirta_test: Dict[str, Any] = field(default_factory=dict)
    
    # Long memory analysis (enhanced)
    hurst_rs: float = 0.5  # R/S method
    hurst_dfa: float = 0.5  # DFA method  
    rescaled_range: Dict[str, Any] = field(default_factory=dict)
    
    # Volatility clustering (enhanced)
    volatility_clustering_summary: str = ""
    arch_lm_pvalue: float = 1.0
    acf_abs_residuals: List[float] = field(default_factory=list)
    acf_squared_residuals: List[float] = field(default_factory=list)
    
    # Nonlinear dependence
    mutual_info_lags: List[int] = field(default_factory=list)
    distance_corr_lags: List[int] = field(default_factory=list)
    mutual_info_values: List[float] = field(default_factory=list)
    distance_corr_values: List[float] = field(default_factory=list)
    
    # Calendar effects
    calendar_effects: Dict[str, Any] = field(default_factory=dict)
    weekday_means: Dict[int, float] = field(default_factory=dict)
    month_means: Dict[int, float] = field(default_factory=dict)
    weekend_vs_weekday_delta: Optional[float] = None
    
    # Enhanced outlier analysis
    outlier_analysis: Dict[str, Any] = field(default_factory=dict)
    extreme_flags: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced residual diagnostics
    residual_diagnostics: Dict[str, Any] = field(default_factory=dict)
    gaps_audit: Dict[str, Any] = field(default_factory=dict)
    dst_audit: Dict[str, Any] = field(default_factory=dict)
    
    # Error metrics (no forecast)
    error_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Rolling behavior
    rolling_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Human-readable summary text (from MVP)
    analysis_summary_text: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                value = float(value)
            elif value is None:
                continue  # Skip None values
            result[field_name] = value
        return result
    
    def summary(self) -> str:
        """Generate enhanced text summary of features (from MVP analysis)."""
        # Use the comprehensive MVP analysis summary (now with fixed formatting)
        if self.analysis_summary_text:
            return self.analysis_summary_text
        
        # Fallback to basic summary if MVP analysis not available
        lines = [
            "TIME SERIES CHARACTERISTICS",
            "=" * 50,
            f"Basic: mean={self.mean:.2f}, std={self.std:.2f}",
            f"Range: [{self.min:.2f}, {self.max:.2f}]",
            f"Distribution: skew={self.skewness:.2f}, kurtosis={self.kurtosis:.2f}",
            "",
            f"Trend: strength={self.trend_strength:.2f}, slope={self.trend_slope:.4f}",
            ""
        ]
        
        if self.has_seasonality:
            periods_str = ', '.join(map(str, self.seasonal_periods[:3])) if self.seasonal_periods else 'None'
            period_display = self.seasonal_period if self.seasonal_period is not None else 'None'
            strength_display = self.seasonal_strength if self.seasonal_strength is not None else 0.0
            lines.extend([
                f"Seasonality: detected (strength={strength_display:.2f})",
                f"            primary period={period_display}",
                f"            all periods: {periods_str}{'...' if self.seasonal_periods and len(self.seasonal_periods) > 3 else ''}"
            ])
        else:
            lines.append("Seasonality: None detected")
        
        lines.extend([
            "",
            f"Stationarity: {'Yes' if self.is_stationary else 'No'} (ADF p={self.adf_pvalue:.3f})",
            f"Volatility: mean={self.volatility_mean:.3f}, clustering={self.volatility_clustering:.2f}"
        ])
        
        return "\n".join(lines)


# =============================================================================
# MVP ANALYSIS HELPER FUNCTIONS
# =============================================================================

def _to_float_series(y: pd.Series) -> pd.Series:
    """Return a float series aligned to y.index (no mutation to inputs); NaNs preserved."""
    s = pd.to_numeric(y, errors="coerce").astype(float)
    s.index = y.index
    return s


def _infer_frequency(idx: pd.DatetimeIndex) -> Optional[str]:
    if not isinstance(idx, pd.DatetimeIndex) or idx.size < 3:
        return None
    try:
        return pd.infer_freq(idx.unique().sort_values())
    except Exception:
        return None


def _gap_audit(idx: pd.DatetimeIndex, inferred_freq: Optional[str]) -> Dict[str, Any]:
    """Simple gap audit relative to inferred frequency."""
    res = {"has_gaps": False, "details": "Not applicable"}
    if not isinstance(idx, pd.DatetimeIndex) or idx.size < 2:
        return res
    vals = idx.unique().sort_values()
    diffs_ns = np.diff(vals.view("i8"))
    if inferred_freq:
        try:
            off = pd.tseries.frequencies.to_offset(inferred_freq)
            expected = off.nanos
            gap_pos = np.where(diffs_ns > expected)[0]
            if gap_pos.size > 0:
                res["has_gaps"] = True
                res["details"] = f"{int(gap_pos.size)} gaps larger than the expected {inferred_freq} cadence."
            else:
                res["has_gaps"] = False
                res["details"] = f"No gaps relative to the expected {inferred_freq} cadence."
        except:
            res["has_gaps"] = False
            res["details"] = "Could not validate frequency gaps."
    else:
        # Irregular spacing, report largest breaks
        order = np.argsort(diffs_ns)[::-1][:3]
        if order.size and diffs_ns[order[0]] > 0:
            largest = []
            for k in order:
                start = vals[k]
                end = vals[k + 1]
                largest.append(f"{start.date()} → {end.date()}")
            res["has_gaps"] = True
            res["details"] = "Largest time jumps: " + ", ".join(largest)
        else:
            res["has_gaps"] = False
            res["details"] = "No material time jumps detected."
    return res


def _extreme_flags(y: pd.Series) -> Dict[str, Any]:
    """Robust z via MAD; returns count and example timestamps (report-only)."""
    s = _to_float_series(y)
    med = float(np.nanmedian(s))
    mad = float(np.nanmedian(np.abs(s - med)))
    if mad == 0 or np.isnan(mad):
        return {"count": 0, "method": "MAD (robust z)", "threshold": 5.0, "examples": []}
    rz = 0.6745 * (s - med) / mad
    mask = np.abs(rz) > 5.0
    idxs = s.index[mask.fillna(False)]
    return {
        "count": int(mask.sum(skipna=True)),
        "method": "MAD (robust z)",
        "threshold": 5.0,
        "examples": [ts.isoformat() for ts in idxs[:5]],
    }


def _infer_season_period(idx: Optional[pd.DatetimeIndex], freq_hint: Optional[str]) -> Optional[int]:
    """Heuristic seasonal period based on frequency."""
    if not isinstance(idx, pd.DatetimeIndex):
        return None
    f = (freq_hint or "").upper() if freq_hint else None
    if f in ("H", "1H"):
        return 24
    if f in ("T", "MIN", "1T", "1MIN"):
        return 60
    if f in ("S", "1S"):
        return 60
    if f in ("D", "1D"):
        return 7 if idx.size < 180 else 365
    if f and f.startswith("W"):
        return 52
    if f and f.startswith("M"):
        return 12
    return None


def _spectral_candidates(y: pd.Series, k: int = 5) -> List[float]:
    """Return up to k candidate periods via periodogram (or FFT fallback)."""
    s = _to_float_series(y).dropna()
    if s.size < 10:
        return []
    x = s.values - np.nanmean(s.values)
    try:
        from scipy.signal import periodogram, find_peaks
        freqs, power = periodogram(x, fs=1.0, scaling="spectrum")
        mask = freqs > 0
        freqs, power = freqs[mask], power[mask]
        try:
            peaks, _ = find_peaks(power)
            top = peaks[np.argsort(power[peaks])[::-1][:k]]
        except:
            top = np.argsort(power)[::-1][:k]
        periods = []
        for i in top:
            f = float(freqs[i])
            if f > 0:
                periods.append(1.0 / f)
        return periods
    except:
        # FFT fallback
        fft = np.fft.rfft(x)
        power = (fft * np.conjugate(fft)).real
        freqs = np.fft.rfftfreq(x.size, d=1.0)
        mask = freqs > 0
        freqs, power = freqs[mask], power[mask]
        top = np.argsort(power)[::-1][:k]
        periods = []
        for i in top:
            f = float(freqs[i])
            if f > 0:
                periods.append(1.0 / f)
        return periods


def _stl_decompose(y: pd.Series, seasonal_period: Optional[int]) -> Dict[str, Any]:
    """Run STL if available and period is valid; return components and strength metrics."""
    out: Dict[str, Any] = {
        "seasonal_period": seasonal_period,
        "trend": None,
        "seasonal": None,
        "remainder": None,
        "trend_strength": None,
        "seasonal_strength": None,
        "error": None,
    }
    if not HAS_STATSMODELS or not seasonal_period or _to_float_series(y).dropna().size < (2 * seasonal_period + 5):
        out["error"] = "STL unavailable or insufficient data/period."
        return out
    try:
        s = _to_float_series(y)
        stl = STL(s, period=int(seasonal_period), robust=False)
        res = stl.fit()
        trend, seasonal, resid = res.trend, res.seasonal, res.resid

        def _strength(comp: pd.Series) -> Optional[float]:
            # Strength = max(0, 1 - var(rem) / var(rem + comp))
            num = np.nanvar(resid)
            den = np.nanvar(resid + comp)
            if den <= 0:
                return None
            return float(max(0.0, 1.0 - num / den))

        out["trend"] = trend
        out["seasonal"] = seasonal
        out["remainder"] = resid
        out["trend_strength"] = _strength(trend)
        out["seasonal_strength"] = _strength(seasonal)
        return out
    except Exception as e:
        out["error"] = repr(e)
        return out


def _stationarity_tests(y: pd.Series, seasonal_period: Optional[int]) -> Dict[str, Any]:
    """ADF & KPSS on raw, first diff, seasonal diff (if applicable)."""
    result: Dict[str, Any] = {
        "adf_raw_pvalue": None,
        "kpss_level_raw_pvalue": None,
        "kpss_trend_raw_pvalue": None,
        "adf_first_diff_pvalue": None,
        "kpss_level_first_diff_pvalue": None,
        "adf_seasonal_diff_pvalue": None,
        "kpss_level_seasonal_diff_pvalue": None,
        "note": None,
    }
    s = _to_float_series(y).dropna()
    if s.size < 20 or not HAS_STATTOOLS:
        result["note"] = "Stationarity tests unavailable or insufficient data."
        return result

    # raw
    try:
        result["adf_raw_pvalue"] = float(adfuller(s, autolag="AIC")[1])
    except Exception:
        pass
    try:
        stat_c, p_c, *_ = kpss(s, regression="c", nlags="auto")
        result["kpss_level_raw_pvalue"] = float(p_c)
    except Exception:
        pass
    try:
        stat_ct, p_ct, *_ = kpss(s, regression="ct", nlags="auto")
        result["kpss_trend_raw_pvalue"] = float(p_ct)
    except Exception:
        pass

    # first diff
    sd = s.diff().dropna()
    if sd.size >= 20:
        try:
            result["adf_first_diff_pvalue"] = float(adfuller(sd, autolag="AIC")[1])
        except Exception:
            pass
        try:
            stat_d, p_d, *_ = kpss(sd, regression="c", nlags="auto")
            result["kpss_level_first_diff_pvalue"] = float(p_d)
        except Exception:
            pass

    # seasonal diff
    if seasonal_period and seasonal_period > 1 and s.size > seasonal_period + 10:
        ssd = s.diff(int(seasonal_period)).dropna()
        if ssd.size >= 20:
            try:
                result["adf_seasonal_diff_pvalue"] = float(adfuller(ssd, autolag="AIC")[1])
            except Exception:
                pass
            try:
                stat_sd, p_sd, *_ = kpss(ssd, regression="c", nlags="auto")
                result["kpss_level_seasonal_diff_pvalue"] = float(p_sd)
            except Exception:
                pass

    return result


def _residual_proxy(y: pd.Series, which: str, stl: Optional[Dict[str, Any]], seasonal_period: Optional[int]) -> pd.Series:
    """Return a residual-like series without fitting a model."""
    s = _to_float_series(y)
    if which == "stl" and stl and isinstance(stl.get("remainder"), pd.Series):
        r = stl["remainder"].astype(float)
        r.index = s.index
        return r
    if which == "diff1":
        return s.diff()
    if which == "diffs" and seasonal_period and seasonal_period > 1:
        return s.diff(int(seasonal_period))
    return s - s.mean()  # fallback: demeaned


def _residual_diagnostics(y: pd.Series, stl: Optional[Dict[str, Any]], seasonal_period: Optional[int]) -> Dict[str, Any]:
    """Normality (Jarque–Bera), ARCH LM, ACF/PACF heads on a residual proxy (STL remainder preferred)."""
    r = _residual_proxy(y, "stl", stl, seasonal_period).dropna()
    out = {
        "size": int(r.size),
        "jarque_bera_pvalue": None,
        "arch_lm_pvalue": None,
        "acf_head": None,
        "pacf_head": None,
        "note": None,
    }
    if r.size < 8:
        out["note"] = "Residual proxy too short for diagnostics."
        return out
    
    try:
        from statsmodels.stats.stattools import jarque_bera
        _, p, _, _ = jarque_bera(r)
        out["jarque_bera_pvalue"] = float(p)
    except Exception:
        pass
    
    try:
        from statsmodels.stats.diagnostic import het_arch
        if r.size >= 20:
            _, p_lm, _, _ = het_arch(r, nlags=5)
            out["arch_lm_pvalue"] = float(p_lm)
    except Exception:
        pass
    
    if HAS_STATTOOLS:
        try:
            vals = acf(r, nlags=min(40, r.size - 1), fft=True)
            out["acf_head"] = [float(v) for v in vals[:10]]
        except Exception:
            pass
        try:
            if r.size > 41:
                vals = pacf(r, nlags=min(40, r.size - 1), method="ywunbiased")
                out["pacf_head"] = [float(v) for v in vals[:10]]
        except Exception:
            pass
    return out


def _calendar_effects(y: pd.Series) -> Dict[str, Any]:
    """Weekday/month averages and weekend vs weekday delta (no external holiday calendars)."""
    out = {"weekday_means": None, "month_means": None, "weekend_vs_weekday_delta": None, "note": None}
    if not isinstance(y.index, pd.DatetimeIndex):
        out["note"] = "Index is not a DatetimeIndex; calendar effects not applicable."
        return out
    s = _to_float_series(y)
    try:
        wd = s.groupby(s.index.dayofweek).mean()
        mo = s.groupby(s.index.month).mean()
        out["weekday_means"] = {int(k): float(v) for k, v in wd.items()}
        out["month_means"] = {int(k): float(v) for k, v in mo.items()}
        if set([0, 1, 2, 3, 4]).issubset(wd.index.tolist()) and wd.index.isin([5, 6]).any():
            weekday_mean = wd[[0, 1, 2, 3, 4]].mean()
            weekend_mean = wd[[i for i in wd.index if i in [5, 6]]].mean()
            out["weekend_vs_weekday_delta"] = float(weekend_mean - weekday_mean)
    except Exception as e:
        out["note"] = f"Calendar grouping error: {e}"
    return out


def _outliers(y: pd.Series, stl: Optional[Dict[str, Any]], seasonal_period: Optional[int]) -> Dict[str, Any]:
    """Robust z/MAD on residual proxy; label as additive outliers (AO)."""
    r = _residual_proxy(y, "stl", stl, seasonal_period)
    s = _to_float_series(r)
    med = float(np.nanmedian(s))
    mad = float(np.nanmedian(np.abs(s - med)))
    out = {"count": 0, "examples": [], "note": None}
    if mad == 0 or np.isnan(mad):
        out["note"] = "MAD is zero/NaN; cannot compute robust z."
        return out
    z = 0.6745 * (s - med) / mad
    mask = np.abs(z) > 5.0
    idxs = y.index[mask.fillna(False)]
    vals = y[mask.fillna(False)]
    out["count"] = int(mask.sum(skipna=True))
    out["examples"] = [
        {"timestamp": ts.isoformat(), "value": float(val)}
        for ts, val in list(zip(idxs, vals))[:5]
    ]
    return out


def _errors_no_forecast(y: pd.Series, seasonal_period: Optional[int]) -> Dict[str, Any]:
    """RMSE, sMAPE, MdAPE, MASE using naive baselines y_{t-1} and y_{t-s}."""
    def smape(a, b):
        denom = (np.abs(a) + np.abs(b))
        m = denom > 0
        if not m.any():
            return 0.0  # Return 0 instead of nan when no valid denominator values
        result = 200.0 * np.nanmean(np.abs(a[m] - b[m]) / denom[m])
        return result if np.isfinite(result) else 0.0

    s = _to_float_series(y)
    out = {
        "one_step": {"RMSE": None, "sMAPE": None, "MdAPE": None, "MASE": None,
                     "abs_error_percentiles": None},
        "seasonal_step": {"RMSE": None, "sMAPE": None, "MdAPE": None,
                          "abs_error_percentiles": None, "season_period": seasonal_period},
    }

    # one-step (t vs t-1)
    if s.size >= 2:
        y_true = s.iloc[1:].values
        y_pred = s.shift(1).iloc[1:].values
        e = y_true - y_pred
        out["one_step"]["RMSE"] = float(np.sqrt(np.nanmean(np.square(e))))
        out["one_step"]["sMAPE"] = float(smape(y_true, y_pred))
        mdape_values = 100.0 * np.abs(e / np.where(np.abs(y_true) > 1e-10, y_true, np.nan))
        mdape_result = np.nanmedian(mdape_values)
        out["one_step"]["MdAPE"] = float(mdape_result) if np.isfinite(mdape_result) else 0.0
        scale = np.nanmean(np.abs(np.diff(s.values)))
        out["one_step"]["MASE"] = (float(np.nanmean(np.abs(e)) / scale) if scale and np.isfinite(scale) else None)
        abs_e = np.abs(e)
        out["one_step"]["abs_error_percentiles"] = {
            "50th_percentile": float(np.nanpercentile(abs_e, 50)),
            "80th_percentile": float(np.nanpercentile(abs_e, 80)),
            "95th_percentile": float(np.nanpercentile(abs_e, 95)),
        }

    # seasonal-step (t vs t-s)
    if seasonal_period and seasonal_period > 1 and s.size > seasonal_period:
        ys_true = s.iloc[seasonal_period:].values
        ys_pred = s.shift(seasonal_period).iloc[seasonal_period:].values
        e = ys_true - ys_pred
        out["seasonal_step"]["RMSE"] = float(np.sqrt(np.nanmean(np.square(e))))
        out["seasonal_step"]["sMAPE"] = float(smape(ys_true, ys_pred))
        mdape_seasonal_values = 100.0 * np.abs(e / np.where(np.abs(ys_true) > 1e-10, ys_true, np.nan))
        mdape_seasonal_result = np.nanmedian(mdape_seasonal_values)
        out["seasonal_step"]["MdAPE"] = float(mdape_seasonal_result) if np.isfinite(mdape_seasonal_result) else 0.0
        abs_e = np.abs(e)
        out["seasonal_step"]["abs_error_percentiles"] = {
            "50th_percentile": float(np.nanpercentile(abs_e, 50)),
            "80th_percentile": float(np.nanpercentile(abs_e, 80)),
            "95th_percentile": float(np.nanpercentile(abs_e, 95)),
        }

    return out


def _rolling(y: pd.Series, seasonal_period: Optional[int], window: Optional[int] = None) -> Dict[str, Any]:
    """Rolling mean, variance, and ACF(1); optional rolling seasonal strength (via STL components)."""
    def acf1(x: pd.Series) -> float:
        x = x.dropna()
        if x.size < 2:
            return np.nan
        x0 = x - x.mean()
        denom = np.correlate(x0, x0)
        return float((np.correlate(x0[:-1], x0[1:]) / denom)[0]) if denom[0] != 0 else np.nan

    s = _to_float_series(y)
    w = window or max(20, int(seasonal_period) if seasonal_period else 30)
    mean_roll = s.rolling(w, min_periods=max(5, w // 3)).mean()
    var_roll = s.rolling(w, min_periods=max(5, w // 3)).var()
    acf1_roll = s.rolling(w, min_periods=max(5, w // 3)).apply(acf1, raw=False)

    seas_strength_roll = None
    if seasonal_period and seasonal_period > 1 and HAS_STATSMODELS and s.size >= (3 * seasonal_period):
        try:
            res = STL(s, period=int(seasonal_period), robust=False).fit()
            rem = res.resid
            seas = res.seasonal
            num = rem.rolling(w, min_periods=max(5, w // 3)).var()
            den = (rem + seas).rolling(w, min_periods=max(5, w // 3)).var()
            seas_strength_roll = (1.0 - (num / den)).clip(lower=0).replace([np.inf, -np.inf], np.nan)
        except Exception:
            seas_strength_roll = None

    return {
        "window": int(w),
        "rolling_mean": mean_roll,
        "rolling_variance": var_roll,
        "rolling_acf1": acf1_roll,
        "rolling_seasonal_strength": seas_strength_roll,
        "last_values": {
            "rolling_mean_last": float(mean_roll.dropna().iloc[-1]) if mean_roll.dropna().size else 0.0,
            "rolling_variance_last": float(var_roll.dropna().iloc[-1]) if var_roll.dropna().size else 0.0,
            "rolling_acf1_last": float(acf1_roll.dropna().iloc[-1]) if acf1_roll.dropna().size else 0.0,
            "rolling_seasonal_strength_last": float(seas_strength_roll.dropna().iloc[-1]) if isinstance(
                seas_strength_roll, pd.Series) and seas_strength_roll.dropna().size else 0.0
        }
    }


def _dst_audit(idx: pd.DatetimeIndex, inferred_freq: Optional[str]) -> Dict[str, Any]:
    """DST applicability and tz presence."""
    res = {
        "time_zone_present": False,
        "time_zone_name": None,
        "dst_audit_applicable": False,
        "note": None,
    }
    if not isinstance(idx, pd.DatetimeIndex):
        res["note"] = "Index is not a DatetimeIndex."
        return res
    tz = idx.tz
    res["time_zone_present"] = tz is not None
    res["time_zone_name"] = str(tz) if tz is not None else None
    intraday = False
    if inferred_freq:
        f = inferred_freq.upper()
        intraday = any(u in f for u in ["H", "T", "S", "MIN"])
    res["dst_audit_applicable"] = bool(intraday and tz is not None)
    if not res["dst_audit_applicable"]:
        res["note"] = "DST audit applies only to time zone–aware intraday data."
    return res


# =============================================================================
# NEW NON-LINEARITY TESTS
# =============================================================================

def _bds_test(y: pd.Series, m: int = 2, epsilon_factor: float = 1.0) -> Dict[str, Any]:
    """Simple BDS test implementation for non-linearity detection."""
    s = _to_float_series(y).dropna()
    out = {"statistic": None, "pvalue": None, "note": None}
    
    if s.size < 50:
        out["note"] = "Insufficient data for BDS test (need ≥50 points)."
        return out
    
    try:
        x = s.values
        n = len(x)
        
        # Use epsilon as fraction of standard deviation
        epsilon = epsilon_factor * np.std(x)
        
        # Count pairs within epsilon distance
        def count_pairs(data, eps, embed_dim):
            embedded = np.array([data[i:i+embed_dim] for i in range(len(data)-embed_dim+1)])
            n_embed = len(embedded)
            count = 0
            for i in range(n_embed):
                for j in range(i+1, n_embed):
                    if np.max(np.abs(embedded[i] - embedded[j])) < eps:
                        count += 1
            return count / (n_embed * (n_embed - 1) / 2)
        
        # Calculate correlation integrals
        c_m = count_pairs(x, epsilon, m)
        c_1 = count_pairs(x, epsilon, 1)
        
        if c_1 == 0 or c_m == 0:
            out["note"] = "Zero correlation integral; cannot compute BDS statistic."
            return out
        
        # BDS statistic (simplified)
        n_eff = n - m + 1
        bds_stat = np.sqrt(n_eff) * (c_m - c_1**m) / np.sqrt(c_1**(2*m))
        
        # Approximate p-value using normal distribution
        if norm:
            pval = 2 * (1 - norm.cdf(np.abs(bds_stat)))
            out["pvalue"] = float(pval)
        
        out["statistic"] = float(bds_stat)
        
    except Exception as e:
        out["note"] = f"BDS test error: {str(e)}"
    
    return out


def _mcleod_li_test(y: pd.Series, seasonal_period: Optional[int], max_lags: int = 10) -> Dict[str, Any]:
    """McLeod-Li test for non-linear dependence using squared residuals."""
    r = _residual_proxy(y, "stl", {}, seasonal_period).dropna()
    out = {"statistic": None, "pvalue": None, "note": None}
    
    if r.size < 30:
        out["note"] = "Insufficient data for McLeod-Li test."
        return out
    
    try:
        # Use squared residuals
        r_sq = r**2
        
        # Ljung-Box test on squared residuals
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(r_sq, lags=min(max_lags, r.size//4), return_df=True)
            
            # Use the overall test statistic
            out["statistic"] = float(lb_result["lb_stat"].iloc[-1])
            out["pvalue"] = float(lb_result["lb_pvalue"].iloc[-1])
        except:
            out["note"] = "statsmodels acorr_ljungbox unavailable"
        
    except Exception as e:
        out["note"] = f"McLeod-Li test error: {str(e)}"
    
    return out


def _terasvirta_test(y: pd.Series, seasonal_period: Optional[int]) -> Dict[str, Any]:
    """Simplified Terasvirta test for non-linearity."""
    r = _residual_proxy(y, "stl", {}, seasonal_period).dropna()
    out = {"statistic": None, "pvalue": None, "note": None}
    
    if r.size < 50 or not HAS_SKLEARN:
        out["note"] = "Insufficient data for Terasvirta test or sklearn unavailable."
        return out
    
    try:
        # Create lagged variables
        lags = min(4, r.size//10)
        X = []
        for lag in range(1, lags+1):
            X.append(r.shift(lag))
        
        if not X:
            out["note"] = "No valid lags for Terasvirta test."
            return out
        
        X_df = pd.concat(X, axis=1).dropna()
        y_reg = r.loc[X_df.index]
        
        if X_df.empty or y_reg.empty:
            out["note"] = "No valid data after creating lags."
            return out
        
        # Add polynomial terms (x^2, x^3, cross products)
        X_lin = X_df.values
        X_sq = X_lin**2
        X_cube = X_lin**3
        
        # Create design matrix with linear, squared, and cubic terms
        X_full = np.column_stack([X_lin, X_sq, X_cube])
        
        # Linear regression
        reg_lin = LinearRegression().fit(X_lin, y_reg.values)
        reg_full = LinearRegression().fit(X_full, y_reg.values)
        
        # F-test for additional terms
        rss_lin = np.sum((y_reg.values - reg_lin.predict(X_lin))**2)
        rss_full = np.sum((y_reg.values - reg_full.predict(X_full))**2)
        
        n = len(y_reg)
        p_lin = X_lin.shape[1]
        p_full = X_full.shape[1]
        
        f_stat = ((rss_lin - rss_full) / (p_full - p_lin)) / (rss_full / (n - p_full))
        
        if f:
            pval = 1 - f.cdf(f_stat, p_full - p_lin, n - p_full)
            out["pvalue"] = float(pval)
        
        out["statistic"] = float(f_stat)
        
    except Exception as e:
        out["note"] = f"Terasvirta test error: {str(e)}"
    
    return out


# =============================================================================
# LONG MEMORY ANALYSIS
# =============================================================================

def _hurst_exponent(y: pd.Series) -> Dict[str, Any]:
    """Hurst exponent using R/S statistic."""
    s = _to_float_series(y).dropna()
    out = {"hurst_rs": None, "hurst_dfa": None, "note": None}
    
    if s.size < 50:
        out["note"] = "Insufficient data for Hurst analysis."
        return out
    
    try:
        # R/S method
        x = s.values
        n = len(x)
        
        # Range of scales
        scales = np.logspace(1, np.log10(n//4), 10).astype(int)
        scales = np.unique(scales)
        rs_values = []
        
        for scale in scales:
            if scale >= n:
                continue
            
            # Partition into non-overlapping windows
            n_windows = n // scale
            rs_window = []
            
            for i in range(n_windows):
                window = x[i*scale:(i+1)*scale]
                if len(window) < scale:
                    continue
                
                # Cumulative deviations from mean
                mean_window = np.mean(window)
                cum_dev = np.cumsum(window - mean_window)
                
                # Range
                R = np.max(cum_dev) - np.min(cum_dev)
                
                # Standard deviation
                S = np.std(window)
                
                if S > 0:
                    rs_window.append(R / S)
            
            if rs_window:
                rs_values.append(np.mean(rs_window))
        
        if len(rs_values) >= 3:
            # Fit log(R/S) = H * log(n) + const
            log_scales = np.log(scales[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Linear regression
            coeffs = np.polyfit(log_scales, log_rs, 1)
            out["hurst_rs"] = float(coeffs[0])
        
        # DFA method (simplified)
        try:
            # Integrate the series
            y_int = np.cumsum(x - np.mean(x))
            
            scales_dfa = np.logspace(1, np.log10(n//4), 10).astype(int)
            scales_dfa = np.unique(scales_dfa)
            fluctuations = []
            
            for scale in scales_dfa:
                if scale >= n:
                    continue
                
                # Detrend using polynomial fit
                n_segments = n // scale
                F_sum = 0
                
                for i in range(n_segments):
                    segment = y_int[i*scale:(i+1)*scale]
                    t = np.arange(len(segment))
                    
                    # Linear detrending
                    coeffs = np.polyfit(t, segment, 1)
                    trend = np.polyval(coeffs, t)
                    
                    F_sum += np.sum((segment - trend)**2)
                
                if n_segments > 0:
                    fluctuations.append(np.sqrt(F_sum / (n_segments * scale)))
            
            if len(fluctuations) >= 3:
                log_scales_dfa = np.log(scales_dfa[:len(fluctuations)])
                log_fluct = np.log(fluctuations)
                coeffs_dfa = np.polyfit(log_scales_dfa, log_fluct, 1)
                out["hurst_dfa"] = float(coeffs_dfa[0])
        
        except Exception:
            pass
            
    except Exception as e:
        out["note"] = f"Hurst analysis error: {str(e)}"
    
    return out


def _rescaled_range(y: pd.Series) -> Dict[str, Any]:
    """Classical rescaled range analysis."""
    s = _to_float_series(y).dropna()
    out = {"rs_statistic": None, "expected_rs": None, "note": None}
    
    if s.size < 10:
        out["note"] = "Insufficient data for R/S analysis."
        return out
    
    try:
        x = s.values
        n = len(x)
        
        # Cumulative deviations from mean
        mean_x = np.mean(x)
        cum_dev = np.cumsum(x - mean_x)
        
        # Range and standard deviation
        R = np.max(cum_dev) - np.min(cum_dev)
        S = np.std(x)
        
        if S > 0:
            rs = R / S
            expected_rs = (np.pi * n / 2)**0.5  # Expected R/S for random walk
            
            out["rs_statistic"] = float(rs)
            out["expected_rs"] = float(expected_rs)
        else:
            out["note"] = "Zero standard deviation; cannot compute R/S."
    
    except Exception as e:
        out["note"] = f"R/S analysis error: {str(e)}"
    
    return out


# =============================================================================
# VOLATILITY CLUSTERING
# =============================================================================

def _volatility_clustering(y: pd.Series, stl: Optional[Dict[str, Any]], seasonal_period: Optional[int]) -> Dict[str, Any]:
    """ACF of |x| and x^2 with ARCH-LM test for time-varying variance."""
    r = _residual_proxy(y, "stl", stl, seasonal_period).dropna()
    out = {
        "arch_lm_pvalue": None,
        "acf_abs_residuals": None,
        "acf_squared_residuals": None,
        "summary": None,
        "note": None
    }
    
    if r.size < 20:
        out["note"] = "Insufficient data for volatility clustering analysis."
        return out
    
    try:
        # ARCH-LM test
        try:
            from statsmodels.stats.diagnostic import het_arch
            _, p_lm, _, _ = het_arch(r, nlags=min(5, r.size//4))
            out["arch_lm_pvalue"] = float(p_lm)
        except:
            pass
        
        # ACF of absolute residuals
        if HAS_STATTOOLS:
            try:
                acf_abs = acf(np.abs(r), nlags=min(20, r.size//2), fft=True)
                out["acf_abs_residuals"] = [float(v) for v in acf_abs]
                
                # ACF of squared residuals
                acf_sq = acf(r**2, nlags=min(20, r.size//2), fft=True)
                out["acf_squared_residuals"] = [float(v) for v in acf_sq]
                
                # Summary assessment
                significant_abs = np.sum(np.abs(acf_abs[1:6]) > 0.1)  # First 5 lags
                significant_sq = np.sum(np.abs(acf_sq[1:6]) > 0.1)
                
                if out["arch_lm_pvalue"] and out["arch_lm_pvalue"] < 0.05:
                    strength = "strong" if (significant_abs + significant_sq) >= 3 else "moderate"
                    out["summary"] = f"ARCH effects detected (p={out['arch_lm_pvalue']:.3f}); {strength} volatility clustering"
                else:
                    out["summary"] = "No significant ARCH effects; limited volatility clustering"
            except:
                pass
    
    except Exception as e:
        out["note"] = f"Volatility clustering error: {str(e)}"
    
    return out


# =============================================================================
# NONLINEAR DEPENDENCE
# =============================================================================

def _mutual_information_lags(y: pd.Series, max_lags: int = 10) -> Dict[str, Any]:
    """Mutual information vs lag to detect nonlinear dependencies."""
    s = _to_float_series(y).dropna()
    out = {"mi_values": None, "significant_lags": None, "note": None}
    
    if s.size < 50 or not HAS_SKLEARN:
        out["note"] = "Insufficient data or sklearn unavailable for MI analysis."
        return out
    
    try:
        x = s.values
        n = len(x)
        max_lag = min(max_lags, n//5)
        
        mi_values = []
        for lag in range(1, max_lag + 1):
            if n - lag < 10:
                break
            
            X = x[:-lag].reshape(-1, 1)
            y_lag = x[lag:]
            
            # Mutual information
            mi = mutual_info_regression(X, y_lag, random_state=42)[0]
            mi_values.append(float(mi))
        
        out["mi_values"] = mi_values
        
        # Find significant lags (simple threshold)
        if mi_values:
            threshold = np.mean(mi_values) + 2 * np.std(mi_values)
            significant = [i+1 for i, mi in enumerate(mi_values) if mi > threshold]
            out["significant_lags"] = significant
    
    except Exception as e:
        out["note"] = f"Mutual information error: {str(e)}"
    
    return out


def _distance_correlation_lags(y: pd.Series, max_lags: int = 10) -> Dict[str, Any]:
    """Distance correlation vs lag for nonlinear dependence detection."""
    s = _to_float_series(y).dropna()
    out = {"dcor_values": None, "significant_lags": None, "note": None}
    
    if s.size < 30 or not HAS_SCIPY_DISTANCE:
        out["note"] = "Insufficient data or scipy unavailable for distance correlation."
        return out
    
    try:
        x = s.values
        n = len(x)
        max_lag = min(max_lags, n//5)
        
        dcor_values = []
        for lag in range(1, max_lag + 1):
            if n - lag < 10:
                break
            
            X = x[:-lag]
            Y = x[lag:]
            
            # Distance correlation calculation
            def dcov(X, Y):
                n = len(X)
                a = squareform(pdist(X.reshape(-1, 1)))
                b = squareform(pdist(Y.reshape(-1, 1)))
                
                A = a - a.mean(axis=0) - a.mean(axis=1)[:, np.newaxis] + a.mean()
                B = b - b.mean(axis=0) - b.mean(axis=1)[:, np.newaxis] + b.mean()
                
                return np.sqrt(np.sum(A * B) / (n * n))
            
            def dvar(X):
                return dcov(X, X)
            
            dcov_xy = dcov(X, Y)
            dvar_x = dvar(X)
            dvar_y = dvar(Y)
            
            if dvar_x > 0 and dvar_y > 0:
                dcor = dcov_xy / np.sqrt(dvar_x * dvar_y)
                dcor_values.append(float(dcor))
            else:
                dcor_values.append(0.0)
        
        out["dcor_values"] = dcor_values
        
        # Find significant lags
        if dcor_values:
            threshold = np.mean(dcor_values) + 2 * np.std(dcor_values)
            significant = [i+1 for i, dc in enumerate(dcor_values) if dc > threshold]
            out["significant_lags"] = significant
    
    except Exception as e:
        out["note"] = f"Distance correlation error: {str(e)}"
    
    return out


# =============================================================================
# MAIN ANALYSIS FUNCTION (from MVP)
# =============================================================================

def analyze_series_mvp(y: pd.Series) -> Tuple[Dict[str, Any], str]:
    """
    Runs the enhanced analyzer (no data mutation) and returns:
    - reports: a dict of raw results for plotting,
    - summary_text: a single human-readable summary for enhanced features.
    """
    # ---- Health & profile ----
    idx = y.index if isinstance(y.index, pd.DatetimeIndex) else None
    inferred_freq = _infer_frequency(idx) if idx is not None else None
    gaps = _gap_audit(idx, inferred_freq) if idx is not None else {"has_gaps": False, "details": "Not applicable"}
    tz_info = _dst_audit(idx, inferred_freq) if idx is not None else {
        "time_zone_present": False, "time_zone_name": None, "dst_audit_applicable": False, "note": "Not applicable"
    }

    size = int(y.size)
    start = idx.min().isoformat() if idx is not None else None
    end = idx.max().isoformat() if idx is not None else None
    monotonic_increasing = bool(idx.is_monotonic_increasing) if idx is not None else None
    unique_timestamps = bool(idx.is_unique) if idx is not None else None
    missing_count = int(y.isna().sum())
    missing_pct = float(round(100 * y.isna().mean(), 3))
    dup_count = int(idx.duplicated(keep=False).sum()) if idx is not None else 0
    extremes = _extreme_flags(y)

    desc = y.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    dist = {
        "mean": float(desc.get("mean", np.nan)),
        "std": float(desc.get("std", np.nan)),
        "min": float(desc.get("min", np.nan)),
        "p05": float(desc.get("5%", np.nan)) if "5%" in desc else None,
        "p25": float(desc.get("25%", np.nan)) if "25%" in desc else None,
        "p50": float(desc.get("50%", np.nan)) if "50%" in desc else None,
        "p75": float(desc.get("75%", np.nan)) if "75%" in desc else None,
        "p95": float(desc.get("95%", np.nan)) if "95%" in desc else None,
        "max": float(desc.get("max", np.nan)),
        "skewness": float(y.skew(skipna=True)),
        "kurtosis": float(y.kurtosis(skipna=True)),
    }

    # ---- Seasonality & structure ----
    seasonal_period_guess = _infer_season_period(idx, inferred_freq)
    stl = _stl_decompose(y, seasonal_period_guess)
    spectral_periods = _spectral_candidates(y, k=5)

    # ---- Stationarity ----
    stationarity = _stationarity_tests(y, stl.get("seasonal_period"))

    # ---- Residual diagnostics ----
    residuals = _residual_diagnostics(y, stl, stl.get("seasonal_period"))

    # ---- Calendar effects ----
    calendar = _calendar_effects(y)

    # ---- Outliers ----
    outliers = _outliers(y, stl, stl.get("seasonal_period"))

    # ---- Error metrics (no forecasts) ----
    errors = _errors_no_forecast(y, stl.get("seasonal_period"))

    # ---- Rolling monitoring ----
    rolling = _rolling(y, stl.get("seasonal_period"))

    # ---- NEW: Non-linearity tests ----
    bds = _bds_test(y)
    mcleod_li = _mcleod_li_test(y, stl.get("seasonal_period"))
    terasvirta = _terasvirta_test(y, stl.get("seasonal_period"))

    # ---- NEW: Long memory analysis ----
    hurst = _hurst_exponent(y)
    rescaled_range = _rescaled_range(y)

    # ---- NEW: Volatility clustering ----
    volatility = _volatility_clustering(y, stl, stl.get("seasonal_period"))

    # ---- NEW: Nonlinear dependence ----
    mutual_info = _mutual_information_lags(y)
    distance_corr = _distance_correlation_lags(y)

    # Build human-readable summary (structured format)
    lines: List[str] = []
    
    # Helper function to safely format numbers
    def safe_format(value, format_spec=".3f", fallback="N/A"):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return fallback
        try:
            return f"{value:{format_spec}}"
        except (TypeError, ValueError):
            return fallback
    
    # Title
    lines.extend([
        "="*80,
        "TIME SERIES ANALYSIS REPORT",
        "="*80,
        ""
    ])

    # Data Overview Section
    lines.extend([
        "📊 DATA OVERVIEW",
        "-" * 40,
        f"• Length: {size} points",
        f"• Time Range: {start} → {end}",
        f"• Frequency: {inferred_freq or 'unknown'}",
        f"• Data Quality:",
        f"  - Timestamps unique: {unique_timestamps}",
        f"  - Monotonic increasing: {monotonic_increasing}",
        f"  - Missing values: {missing_count} ({missing_pct}%)",
        f"  - Duplicate timestamps: {dup_count}",
        f"  - Gap audit: {gaps.get('details')}",
        ""
    ])

    # Statistical Distribution Section
    lines.extend([
        "📈 STATISTICAL DISTRIBUTION",
        "-" * 40,
        f"• Central Tendency:",
        f"  - Mean: {safe_format(dist['mean'])}",
        f"  - Median: {safe_format(dist['p50'])}",
        f"  - Standard Deviation: {safe_format(dist['std'])}",
        f"• Range & Percentiles:",
        f"  - Min: {safe_format(dist['min'])}",
        f"  - 5th percentile: {safe_format(dist['p05'])}",
        f"  - 25th percentile: {safe_format(dist['p25'])}",
        f"  - 75th percentile: {safe_format(dist['p75'])}",
        f"  - 95th percentile: {safe_format(dist['p95'])}",
        f"  - Max: {safe_format(dist['max'])}",
        f"• Shape:",
        f"  - Skewness: {safe_format(dist['skewness'])}",
        f"  - Kurtosis: {safe_format(dist['kurtosis'])}",
        f"• Outliers: {extremes['count']} extreme values (threshold={extremes['threshold']})"
        + (f" - Examples: {', '.join(extremes['examples'])}" if extremes['examples'] else ""),
        ""
    ])
    
    # Timekeeping Section
    if tz_info['time_zone_present'] or tz_info['dst_audit_applicable']:
        lines.extend([
            "🕐 TIMEKEEPING",
            "-" * 40,
            f"• Time zone: {tz_info['time_zone_name'] if tz_info['time_zone_present'] else 'None'}",
            f"• DST audit applicable: {tz_info['dst_audit_applicable']}",
            "" if not tz_info.get("note") else f"• Note: {tz_info['note']}",
            ""
        ])
    else:
        lines.append("")

    # Seasonality & Trend Section
    lines.extend([
        "🔄 SEASONALITY & TREND",
        "-" * 40
    ])
    
    if stl.get("error"):
        lines.extend([
            f"• Seasonality: Not detected or insufficient data",
            f"  - Period guess: {seasonal_period_guess}",
            f"  - Status: STL decomposition failed",
        ])
    else:
        lines.extend([
            f"• Seasonality Analysis:",
            f"  - Primary period: {stl['seasonal_period'] or 'N/A'}",
            f"  - Seasonal strength: {safe_format(stl['seasonal_strength'])}",
            f"  - Trend strength: {safe_format(stl['trend_strength'])}",
        ])
    
    if spectral_periods:
        top = ", ".join(f"{p:.1f}" for p in spectral_periods[:3])
        lines.extend([
            f"• Spectral Analysis:",
            f"  - Top frequency candidates: {top}",
        ])
    
    lines.append("")

    # Stationarity Section
    adf_p = stationarity.get("adf_raw_pvalue")
    kpss_level_p = stationarity.get("kpss_level_raw_pvalue")
    
    lines.extend([
        "📊 STATIONARITY TESTS",
        "-" * 40,
    ])
    
    if adf_p is not None:
        adf_result = "Stationary" if adf_p < 0.05 else "Non-stationary"
        lines.append(f"• ADF Test: {adf_result} (p-value: {adf_p:.3g})")
    else:
        lines.append(f"• ADF Test: Not available")
        
    if kpss_level_p is not None:
        kpss_result = "Stationary" if kpss_level_p > 0.05 else "Non-stationary"
        lines.append(f"• KPSS Test: {kpss_result} (p-value: {kpss_level_p:.3g})")
    else:
        lines.append(f"• KPSS Test: Not available")
        
    lines.append("")

    # Diagnostic Tests Section
    jb_p = residuals.get("jarque_bera_pvalue")
    arch_p = residuals.get("arch_lm_pvalue")
    
    lines.extend([
        "🔍 DIAGNOSTIC TESTS",
        "-" * 40,
    ])
    
    if jb_p is not None:
        jb_result = "Normal" if jb_p > 0.05 else "Non-normal"
        lines.append(f"• Jarque-Bera Test: {jb_result} residuals (p-value: {jb_p:.3g})")
    else:
        lines.append(f"• Jarque-Bera Test: Not available")
        
    if arch_p is not None:
        arch_result = "No ARCH effects" if arch_p > 0.05 else "ARCH effects detected"
        lines.append(f"• ARCH LM Test: {arch_result} (p-value: {arch_p:.3g})")
    else:
        lines.append(f"• ARCH LM Test: Not available")
        
    lines.append("")

    # Advanced Analysis Section
    lines.extend([
        "🔬 ADVANCED ANALYSIS",
        "-" * 40,
    ])
    
    # Non-linearity tests
    lines.append("• Non-linearity Tests:")
    has_nonlinear = False
    if bds.get("pvalue") is not None:
        result = "Detected" if bds['pvalue'] < 0.05 else "Not detected"
        lines.append(f"  - BDS Test: {result} (p-value: {bds['pvalue']:.3g})")
        has_nonlinear = has_nonlinear or bds['pvalue'] < 0.05
    if mcleod_li.get("pvalue") is not None:
        result = "Detected" if mcleod_li['pvalue'] < 0.05 else "Not detected"
        lines.append(f"  - McLeod-Li Test: {result} (p-value: {mcleod_li['pvalue']:.3g})")
        has_nonlinear = has_nonlinear or mcleod_li['pvalue'] < 0.05
    if terasvirta.get("pvalue") is not None:
        result = "Detected" if terasvirta['pvalue'] < 0.05 else "Not detected"
        lines.append(f"  - Terasvirta Test: {result} (p-value: {terasvirta['pvalue']:.3g})")
        has_nonlinear = has_nonlinear or terasvirta['pvalue'] < 0.05
    
    if not any([bds.get("pvalue"), mcleod_li.get("pvalue"), terasvirta.get("pvalue")]):
        lines.append(f"  - Status: Tests unavailable or insufficient data")

    # Long memory analysis
    lines.append("• Long Memory Analysis:")
    if hurst.get("hurst_rs") is not None:
        memory_type = "Persistent" if hurst['hurst_rs'] > 0.5 else "Anti-persistent" if hurst['hurst_rs'] < 0.5 else "Random walk"
        lines.append(f"  - Hurst (R/S): {hurst['hurst_rs']:.3f} ({memory_type})")
    if hurst.get("hurst_dfa") is not None:
        memory_type = "Persistent" if hurst['hurst_dfa'] > 0.5 else "Anti-persistent" if hurst['hurst_dfa'] < 0.5 else "Random walk"
        lines.append(f"  - Hurst (DFA): {hurst['hurst_dfa']:.3f} ({memory_type})")
    if rescaled_range.get("rs_statistic") is not None:
        lines.append(f"  - R/S Statistic: {rescaled_range['rs_statistic']:.1f}")
    
    if not any([hurst.get("hurst_rs"), hurst.get("hurst_dfa"), rescaled_range.get("rs_statistic")]):
        lines.append(f"  - Status: Analysis unavailable or insufficient data")

    # Volatility clustering
    lines.append("• Volatility Analysis:")
    if volatility.get("summary"):
        lines.append(f"  - Clustering: {volatility['summary']}")
    else:
        lines.append(f"  - Status: Analysis unavailable or insufficient data")

    # Nonlinear dependence
    lines.append("• Nonlinear Dependence:")
    nonlin_dep_found = False
    if mutual_info.get("significant_lags"):
        lines.append(f"  - Mutual Information lags: {mutual_info['significant_lags']}")
        nonlin_dep_found = True
    if distance_corr.get("significant_lags"):
        lines.append(f"  - Distance Correlation lags: {distance_corr['significant_lags']}")
        nonlin_dep_found = True
    
    if not nonlin_dep_found:
        lines.append(f"  - Status: No significant nonlinear dependencies detected")
        
    lines.append("")

    # Calendar Effects & Outliers Section
    lines.extend([
        "📅 CALENDAR EFFECTS & OUTLIERS",
        "-" * 40,
    ])
    
    if calendar.get("weekday_means"):
        lines.append("• Calendar Effects:")
        if calendar.get("weekend_vs_weekday_delta") is not None:
            direction = "higher" if calendar['weekend_vs_weekday_delta'] > 0 else "lower"
            lines.append(f"  - Weekend vs weekday: {direction} by {safe_format(abs(calendar['weekend_vs_weekday_delta']))}")
        lines.append(f"  - Status: Weekday and monthly patterns analyzed")
    else:
        lines.append("• Calendar Effects: Not applicable (no datetime index)")
    
    lines.append(f"• Outlier Detection:")
    lines.append(f"  - Outliers found: {outliers['count']} additive outliers")
    if outliers.get("examples"):
        example_dates = [d['timestamp'] for d in outliers['examples'][:3]]
        lines.append(f"  - Examples: {', '.join(example_dates)}")
    
    lines.append("")

    # Performance Metrics Section  
    lines.extend([
        "📊 BASELINE PERFORMANCE",
        "-" * 40,
    ])
    
    one = errors["one_step"]
    seas = errors["seasonal_step"]
    
    lines.append("• One-step Naive Baseline:")
    if one.get("RMSE") is not None:
        lines.append(f"  - RMSE: {safe_format(one['RMSE'])}")
        lines.append(f"  - sMAPE: {safe_format(one['sMAPE'], '.2f')}%")
        lines.append(f"  - MdAPE: {safe_format(one['MdAPE'], '.2f')}%")
        if one.get("MASE") is not None:
            lines.append(f"  - MASE: {safe_format(one['MASE'])}")
    else:
        lines.append(f"  - Status: Metrics unavailable")
    
    if seas.get("season_period"):
        lines.append(f"• Seasonal Naive Baseline (period={seas['season_period']}):")
        if seas.get("RMSE") is not None:
            lines.append(f"  - RMSE: {safe_format(seas['RMSE'])}")
            lines.append(f"  - sMAPE: {safe_format(seas['sMAPE'], '.2f')}%") 
            lines.append(f"  - MdAPE: {safe_format(seas['MdAPE'], '.2f')}%")
            if seas.get("MASE") is not None:
                lines.append(f"  - MASE: {safe_format(seas['MASE'])}")
        else:
            lines.append(f"  - Status: Metrics unavailable")
    
    lines.append("")

    # Summary Monitoring Section
    lines.extend([
        "🔄 ROLLING WINDOW SUMMARY",
        "-" * 40,
    ])
    
    rv = rolling["last_values"]
    lines.append("• Recent Window Statistics:")
    lines.append(f"  - Mean: {safe_format(rv['rolling_mean_last'], '.4f')}")
    lines.append(f"  - Variance: {safe_format(rv['rolling_variance_last'], '.4f')}")
    lines.append(f"  - Autocorrelation (lag 1): {safe_format(rv['rolling_acf1_last'], '.4f')}")
    lines.append(f"  - Seasonal strength: {safe_format(rv['rolling_seasonal_strength_last'], '.4f')}")
    
    lines.append("")

    # Footer Section
    lines.extend([
        "ℹ️  ANALYSIS NOTES",
        "-" * 40,
        "• Data is not mutated during analysis",
        "• All diagnostics are model-free statistical tests",
        "• DST audit only applies to time zone-aware intraday data",
        "• Results are based on available data without imputation",
        "",
        "="*80
    ])

    # Filter out None values from lines before joining
    filtered_lines = [line for line in lines if line is not None]
    summary_text = "\n".join(filtered_lines)

    # Collect raw reports for plotting (NOT used by LLM)
    reports: Dict[str, Any] = {
        "health": {
            "size": size,
            "start": start,
            "end": end,
            "inferred_frequency": inferred_freq,
            "unique_timestamps": unique_timestamps,
            "monotonic_increasing": monotonic_increasing,
            "missing_count": missing_count,
            "missing_percent": missing_pct,
            "duplicate_timestamps": dup_count,
            "gaps": gaps,
            "distribution": dist,
            "extremes": extremes,
            "timekeeping": tz_info,
        },
        "seasonality": stl,
        "spectral_periods": spectral_periods,
        "stationarity": stationarity,
        "residuals": residuals,
        "calendar": calendar,
        "outliers": outliers,
        "errors": errors,
        "rolling": rolling,
        "nonlinearity": {
            "bds": bds,
            "mcleod_li": mcleod_li,
            "terasvirta": terasvirta,
        },
        "long_memory": {
            "hurst": hurst,
            "rescaled_range": rescaled_range,
        },
        "volatility": volatility,
        "nonlinear_dependence": {
            "mutual_info": mutual_info,
            "distance_correlation": distance_corr,
        },
    }

    return reports, summary_text


class TimeSeriesAnalyzer:
    """Comprehensive time series feature extraction and analysis."""
    
    def __init__(self, use_cache: bool = True):
        """Initialize the analyzer."""
        self.use_cache = use_cache
        self._feature_cache = {}
        self.preprocessor = TimeSeriesPreprocessor()
        
    def analyze(self, 
                data: pd.DataFrame,
                value_col: str = 'value',
                date_col: str = 'date',
                freq: Optional[str] = None) -> TimeSeriesFeatures:
        """
        Perform comprehensive time series analysis using enhanced MVP features.
        
        Args:
            data: Time series data
            value_col: Column with values
            date_col: Column with dates
            freq: Frequency of data ('D', 'M', 'H', etc.)
            
        Returns:
            TimeSeriesFeatures object with all extracted features
        """
        # Create cache key
        cache_key = f"{len(data)}_{data[value_col].sum():.6f}_{freq}"
        if self.use_cache and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        # Prepare data
        df = data.copy().sort_values(date_col)
        dates = pd.to_datetime(df[date_col])
        
        # Create a Series with datetime index for MVP analysis
        series = pd.Series(df[value_col].values, index=dates, name=value_col)
        
        # Run MVP comprehensive analysis
        reports, summary_text = analyze_series_mvp(series)
        
        # Extract basic statistics from reports
        dist = reports["health"]["distribution"]
        basic_stats = {
            'mean': dist['mean'],
            'std': dist['std'], 
            'min': dist['min'],
            'max': dist['max'],
            'skewness': dist['skewness'],
            'kurtosis': dist['kurtosis']
        }
        
        # Extract trend features (fallback to simple calculation)
        trend_features = self._extract_trend_features(series.values)
        
        # Extract seasonality from STL results
        stl_results = reports["seasonality"]
        seasonality_features = {
            'has_seasonality': bool((stl_results.get('seasonal_strength') or 0) > 0.1),
            'primary_period': stl_results.get('seasonal_period') or None,
            'strength': float(stl_results.get('seasonal_strength') or 0),
            'all_periods': reports.get("spectral_periods", [])[:5],  # Limit to top 5
            'dominant_frequencies': reports.get("spectral_periods", [])[:5]
        }
        
        # Extract volatility features (use existing method as fallback)
        volatility_features = self._extract_volatility_features(series.values)
        volatility_clustering_info = reports["volatility"]
        
        # Extract autocorrelation features (use existing method)
        autocorr_features = self._extract_autocorrelation_features(series.values)
        
        # Extract stationarity from MVP results
        stationarity_data = reports["stationarity"]
        adf_p = stationarity_data.get('adf_raw_pvalue')
        kpss_p = stationarity_data.get('kpss_level_raw_pvalue')
        is_stationary = ((adf_p or 1.0) < 0.05 and (kpss_p or 0.0) > 0.05)
        
        # Extract advanced features
        nonlinearity = reports["nonlinearity"]
        long_memory = reports["long_memory"]
        nonlinear_dep = reports["nonlinear_dependence"]
        
        # Get calendar effects
        calendar_data = reports["calendar"]
        
        # Extract outlier information
        outlier_data = reports["outliers"]
        extreme_data = reports["health"]["extremes"]
        
        # Extract error metrics and rolling stats
        error_data = reports["errors"]
        rolling_data = reports["rolling"]
        
        # Get residual diagnostics
        residual_data = reports["residuals"]
        
        # Create enhanced features object
        features = TimeSeriesFeatures(
            # Basic statistics
            mean=basic_stats['mean'],
            std=basic_stats['std'],
            min=basic_stats['min'],
            max=basic_stats['max'],
            skewness=basic_stats['skewness'],
            kurtosis=basic_stats['kurtosis'],
            
            # Trend features
            trend_strength=trend_features['strength'],
            trend_slope=trend_features['slope'],
            trend_r2=trend_features['r2'],
            trend_acceleration=trend_features.get('acceleration', 0.0),
            
            # Seasonality features
            has_seasonality=seasonality_features['has_seasonality'],
            seasonal_period=seasonality_features['primary_period'],
            seasonal_strength=seasonality_features['strength'],
            seasonal_periods=seasonality_features['all_periods'],
            dominant_frequencies=seasonality_features['dominant_frequencies'],
            
            # Volatility features
            volatility_mean=volatility_features['mean'],
            volatility_std=volatility_features['std'],
            volatility_clustering=volatility_features['clustering'],
            volatility_persistence=volatility_features['persistence'],
            
            # Autocorrelation features
            acf_1=autocorr_features['acf_1'],
            acf_5=autocorr_features['acf_5'],
            pacf_1=autocorr_features['pacf_1'],
            pacf_5=autocorr_features['pacf_5'],
            significant_lags=autocorr_features['significant_lags'],
            ljung_box_pvalue=autocorr_features['ljung_box_pvalue'],
            
            # Stationarity
            is_stationary=is_stationary,
            adf_pvalue=float(adf_p or 1.0),
            kpss_pvalue=float(kpss_p or 0.0),
            
            # Legacy fields
            hurst_exponent=long_memory["hurst"].get('hurst_rs', 0.5),
            entropy=0.0,  # Can be computed if needed
            complexity=0.0,  # Can be computed if needed
            decomposition=stl_results if stl_results.get('trend') is not None else None,
            num_differences_needed=0,  # Can be computed if needed
            
            # NEW ENHANCED FEATURES FROM MVP
            # Non-linearity tests
            bds_test=nonlinearity["bds"],
            mcleod_li_test=nonlinearity["mcleod_li"],
            terasvirta_test=nonlinearity["terasvirta"],
            
            # Long memory analysis (enhanced)
            hurst_rs=long_memory["hurst"].get('hurst_rs', 0.5),
            hurst_dfa=long_memory["hurst"].get('hurst_dfa', 0.5),
            rescaled_range=long_memory["rescaled_range"],
            
            # Volatility clustering (enhanced)
            volatility_clustering_summary=volatility_clustering_info.get('summary', ''),
            arch_lm_pvalue=volatility_clustering_info.get('arch_lm_pvalue', 1.0),
            acf_abs_residuals=volatility_clustering_info.get('acf_abs_residuals', []),
            acf_squared_residuals=volatility_clustering_info.get('acf_squared_residuals', []),
            
            # Nonlinear dependence
            mutual_info_lags=nonlinear_dep["mutual_info"].get('significant_lags', []),
            distance_corr_lags=nonlinear_dep["distance_correlation"].get('significant_lags', []),
            mutual_info_values=nonlinear_dep["mutual_info"].get('mi_values', []),
            distance_corr_values=nonlinear_dep["distance_correlation"].get('dcor_values', []),
            
            # Calendar effects
            calendar_effects=calendar_data,
            weekday_means=calendar_data.get('weekday_means', {}),
            month_means=calendar_data.get('month_means', {}),
            weekend_vs_weekday_delta=calendar_data.get('weekend_vs_weekday_delta'),
            
            # Enhanced outlier analysis
            outlier_analysis=outlier_data,
            extreme_flags=extreme_data,
            
            # Enhanced residual diagnostics
            residual_diagnostics=residual_data,
            gaps_audit=reports["health"]["gaps"],
            dst_audit=reports["health"]["timekeeping"],
            
            # Error metrics (no forecast)
            error_metrics=error_data,
            
            # Rolling behavior
            rolling_stats=rolling_data,
            
            # Human-readable summary text (from MVP)
            analysis_summary_text=summary_text,
            
            # Metadata
            metadata={
                'n_observations': len(series),
                'frequency': freq,
                'date_range': f"{dates.min()} to {dates.max()}",
                'has_missing': bool(series.isna().any()),
                'missing_rate': float(series.isna().sum() / len(series)),
                'mvp_analysis': True,  # Flag to indicate enhanced analysis
                'reports': reports  # Store full reports for plotting
            }
        )
        
        # Cache result
        if self.use_cache:
            self._feature_cache[cache_key] = features
            
        return features
    
    def _extract_basic_statistics(self, values: np.ndarray) -> Dict:
        """Extract basic statistical features."""
        clean_values = values[~np.isnan(values)]
        if len(clean_values) == 0:
            return {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'skewness': 0.0, 'kurtosis': 0.0
            }
            
        return {
            'mean': float(np.mean(clean_values)),
            'std': float(np.std(clean_values)),
            'min': float(np.min(clean_values)),
            'max': float(np.max(clean_values)),
            'skewness': float(stats.skew(clean_values)),
            'kurtosis': float(stats.kurtosis(clean_values))
        }
    
    def _extract_trend_features(self, values: np.ndarray) -> Dict:
        """Extract trend-related features."""
        n = len(values)
        clean_values = values[~np.isnan(values)]
        
        if len(clean_values) < 3:
            return {
                'slope': 0.0, 'intercept': 0.0, 'r2': 0.0, 
                'strength': 0.0, 'acceleration': 0.0
            }
        
        x = np.arange(len(clean_values))
        
        # Linear trend
        try:
            z1 = np.polyfit(x, clean_values, 1)
            trend_line = np.polyval(z1, x)
            
            # R² calculation
            ss_res = np.sum((clean_values - trend_line) ** 2)
            ss_tot = np.sum((clean_values - np.mean(clean_values)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-12))
            
            # Quadratic trend for acceleration
            acceleration = 0.0
            if len(clean_values) > 5:
                try:
                    z2 = np.polyfit(x, clean_values, 2)
                    acceleration = 2 * z2[0]  # Second derivative
                except:
                    pass
                    
        except:
            z1 = [0.0, np.mean(clean_values)]
            r2 = 0.0
            acceleration = 0.0
        
        return {
            'slope': float(z1[0]),
            'intercept': float(z1[1]),
            'r2': float(max(0, r2)),
            'strength': float(max(0, r2)),
            'acceleration': float(acceleration)
        }
    
    def _extract_seasonality_features(self, values: np.ndarray, freq: Optional[str]) -> Dict:
        """Extract enhanced seasonality features."""
        n = len(values)
        clean_values = values[~np.isnan(values)]
        
        if len(clean_values) < 12:
            return {
                'has_seasonality': False,
                'primary_period': None,
                'strength': 0.0,
                'all_periods': [],
                'dominant_frequencies': []
            }
        
        # Use STL decomposition if available
        if HAS_STATSMODELS and len(clean_values) >= 24:
            return self._stl_seasonality_detection(clean_values, freq)
        else:
            return self._fft_seasonality_detection(clean_values, freq)
    
    def _stl_seasonality_detection(self, values: np.ndarray, freq: Optional[str]) -> Dict:
        """Use STL decomposition for robust seasonality detection."""
        try:
            # Determine seasonal periods to test
            periods_to_test = self._get_seasonal_periods(len(values), freq)
            
            best_period = None
            best_strength = 0.0
            all_periods = []
            
            for period in periods_to_test:
                if len(values) < 2 * period:
                    continue
                    
                try:
                    stl = STL(values, seasonal=period, robust=True)
                    result = stl.fit()
                    
                    # Calculate seasonal strength
                    seasonal_var = np.var(result.seasonal)
                    remainder_var = np.var(result.resid)
                    strength = seasonal_var / (seasonal_var + remainder_var + 1e-12)
                    
                    if strength > 0.1:  # Threshold for significance
                        all_periods.append(period)
                        
                    if strength > best_strength:
                        best_strength = strength
                        best_period = period
                        
                except:
                    continue
            
            # FFT analysis for dominant frequencies
            dominant_freqs = self._extract_dominant_frequencies(values)
            
            return {
                'has_seasonality': best_strength > 0.1,
                'primary_period': best_period,
                'strength': float(best_strength),
                'all_periods': sorted(all_periods),
                'dominant_frequencies': dominant_freqs
            }
            
        except Exception:
            return self._fft_seasonality_detection(values, freq)
    
    def _fft_seasonality_detection(self, values: np.ndarray, freq: Optional[str]) -> Dict:
        """Fallback FFT-based seasonality detection."""
        n = len(values)
        
        # FFT analysis
        fft_values = fft(values - np.mean(values))
        power = np.abs(fft_values) ** 2
        frequencies = fftfreq(n)
        
        # Find peaks in power spectrum
        positive_freq_mask = frequencies > 0
        pos_frequencies = frequencies[positive_freq_mask]
        pos_power = power[positive_freq_mask]
        
        # Get dominant frequencies
        dominant_indices = np.argsort(pos_power)[-10:]  # Top 10
        dominant_freqs = pos_frequencies[dominant_indices]
        dominant_powers = pos_power[dominant_indices]
        
        # Convert to periods
        periods = []
        for i, freq_val in enumerate(dominant_freqs):
            if freq_val > 0:
                period = int(round(1.0 / freq_val))
                if period >= 2 and period < n // 3:
                    periods.append(period)
        
        # Find best period
        best_period = None
        best_strength = 0.0
        
        if periods:
            # Test each period
            for period in set(periods):
                if period < 2 or period > n // 3:
                    continue
                    
                # Calculate seasonal strength for this period
                strength = self._calculate_seasonal_strength(values, period)
                if strength > best_strength:
                    best_strength = strength
                    best_period = period
        
        significant_periods = [p for p in periods if self._calculate_seasonal_strength(values, p) > 0.1]
        
        return {
            'has_seasonality': best_strength > 0.1,
            'primary_period': best_period,
            'strength': float(best_strength),
            'all_periods': sorted(list(set(significant_periods))),
            'dominant_frequencies': dominant_freqs[-5:].tolist()
        }
    
    def _calculate_seasonal_strength(self, values: np.ndarray, period: int) -> float:
        """Calculate seasonal strength for a given period."""
        if len(values) < 2 * period:
            return 0.0
            
        try:
            # Simple seasonal decomposition
            n_complete_cycles = len(values) // period
            if n_complete_cycles < 2:
                return 0.0
                
            # Reshape to seasonal matrix
            seasonal_data = values[:n_complete_cycles * period].reshape(-1, period)
            seasonal_means = np.mean(seasonal_data, axis=0)
            
            # Calculate seasonal pattern
            seasonal_pattern = np.tile(seasonal_means, n_complete_cycles)
            
            # Seasonal strength
            ss_seasonal = np.sum((seasonal_pattern - np.mean(values)) ** 2)
            ss_total = np.sum((values[:len(seasonal_pattern)] - np.mean(values)) ** 2)
            
            return ss_seasonal / (ss_total + 1e-12)
            
        except:
            return 0.0
    
    def _extract_dominant_frequencies(self, values: np.ndarray) -> List[float]:
        """Extract dominant frequencies using FFT."""
        try:
            fft_values = fft(values - np.mean(values))
            power = np.abs(fft_values) ** 2
            frequencies = fftfreq(len(values))
            
            # Get positive frequencies only
            pos_mask = frequencies > 0
            pos_frequencies = frequencies[pos_mask]
            pos_power = power[pos_mask]
            
            # Find top frequencies
            top_indices = np.argsort(pos_power)[-5:]
            return pos_frequencies[top_indices].tolist()
            
        except:
            return []
    
    def _get_seasonal_periods(self, n: int, freq: Optional[str]) -> List[int]:
        """Get likely seasonal periods based on frequency."""
        periods = []
        
        # Default periods
        if freq:
            freq_upper = freq.upper()
            if 'D' in freq_upper or 'B' in freq_upper:  # Daily
                periods = [7, 30, 91, 365]  # Weekly, monthly, quarterly, yearly
            elif 'W' in freq_upper:  # Weekly
                periods = [4, 13, 52]  # Monthly, quarterly, yearly
            elif 'M' in freq_upper:  # Monthly
                periods = [3, 6, 12]  # Quarterly, semi-annual, yearly
            elif 'H' in freq_upper:  # Hourly
                periods = [24, 168, 8760]  # Daily, weekly, yearly
            elif 'T' in freq_upper or 'MIN' in freq_upper:  # Minute
                periods = [60, 1440]  # Hourly, daily
        
        # Fallback periods
        if not periods:
            periods = [7, 12, 24, 30, 52]
        
        # Filter by data length
        return [p for p in periods if p >= 2 and p <= n // 3]
    
    def _extract_volatility_features(self, values: np.ndarray) -> Dict:
        """Extract enhanced volatility features."""
        clean_values = values[~np.isnan(values)]
        
        if len(clean_values) < 2:
            return {
                'mean': 0.0, 'std': 0.0, 'clustering': 0.0, 'persistence': 0.0
            }
        
        # Calculate returns
        returns = np.diff(clean_values) / (clean_values[:-1] + 1e-12)
        
        # Rolling volatility
        window = min(20, len(returns) // 5)
        if window >= 2:
            rolling_vol = pd.Series(returns).rolling(window).std().dropna().values
            vol_mean = np.mean(rolling_vol)
            vol_std = np.std(rolling_vol)
        else:
            vol_mean = np.std(returns)
            vol_std = 0.0
        
        # Volatility clustering (ARCH effect)
        squared_returns = returns ** 2
        clustering = 0.0
        persistence = 0.0
        
        if len(squared_returns) > 1:
            # Simple ARCH test - correlation of squared returns
            try:
                clustering = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                clustering = 0.0 if np.isnan(clustering) else clustering
            except:
                clustering = 0.0
            
            # Persistence - sum of first 5 autocorrelations of squared returns
            try:
                if len(squared_returns) >= 10:
                    autocorrs = []
                    for lag in range(1, min(6, len(squared_returns) // 2)):
                        if len(squared_returns) > lag:
                            corr = np.corrcoef(squared_returns[:-lag], squared_returns[lag:])[0, 1]
                            if not np.isnan(corr):
                                autocorrs.append(abs(corr))
                    persistence = sum(autocorrs) if autocorrs else 0.0
            except:
                persistence = 0.0
        
        return {
            'mean': float(vol_mean),
            'std': float(vol_std),
            'clustering': float(abs(clustering)),
            'persistence': float(persistence)
        }
    
    def _extract_autocorrelation_features(self, values: np.ndarray) -> Dict:
        """Extract autocorrelation features."""
        clean_values = values[~np.isnan(values)]
        n = len(clean_values)
        
        if n < 3:
            return {
                'acf_1': 0.0, 'acf_5': 0.0, 'pacf_1': 0.0, 'pacf_5': 0.0,
                'significant_lags': [], 'ljung_box_pvalue': 1.0
            }
        
        max_lag = min(40, n // 4)
        
        if HAS_STATTOOLS:
            try:
                # Use statsmodels for better ACF/PACF
                acf_values = acf(clean_values, nlags=max_lag, fft=True)
                pacf_values = pacf(clean_values, nlags=max_lag)
                
                # Ljung-Box test for serial correlation
                from statsmodels.stats.diagnostic import acorr_ljungbox
                ljung_result = acorr_ljungbox(clean_values, lags=min(10, n//4), return_df=False)
                ljung_pvalue = ljung_result['lb_pvalue'].iloc[-1] if hasattr(ljung_result, 'iloc') else ljung_result[1][-1]
                
            except Exception:
                acf_values = self._manual_acf(clean_values, max_lag)
                pacf_values = self._manual_pacf(clean_values, max_lag)
                ljung_pvalue = 1.0
        else:
            acf_values = self._manual_acf(clean_values, max_lag)
            pacf_values = self._manual_pacf(clean_values, max_lag)
            ljung_pvalue = 1.0
        
        # Confidence interval (95%)
        ci = 1.96 / np.sqrt(n)
        
        # Find significant lags
        significant_lags = [i for i in range(1, len(acf_values)) 
                          if abs(acf_values[i]) > ci]
        
        return {
            'acf_1': float(acf_values[1]) if len(acf_values) > 1 else 0.0,
            'acf_5': float(acf_values[5]) if len(acf_values) > 5 else 0.0,
            'pacf_1': float(pacf_values[1]) if len(pacf_values) > 1 else 0.0,
            'pacf_5': float(pacf_values[5]) if len(pacf_values) > 5 else 0.0,
            'significant_lags': significant_lags[:20],  # Limit output
            'ljung_box_pvalue': float(ljung_pvalue)
        }
    
    def _manual_acf(self, values: np.ndarray, max_lag: int) -> np.ndarray:
        """Manual ACF calculation fallback."""
        n = len(values)
        acf_values = np.zeros(max_lag + 1)
        acf_values[0] = 1.0
        
        mean_val = np.mean(values)
        var_val = np.var(values)
        
        for lag in range(1, max_lag + 1):
            if n > lag:
                covariance = np.mean((values[:-lag] - mean_val) * (values[lag:] - mean_val))
                acf_values[lag] = covariance / (var_val + 1e-12)
        
        return acf_values
    
    def _manual_pacf(self, values: np.ndarray, max_lag: int) -> np.ndarray:
        """Manual PACF calculation fallback (simplified)."""
        # Simplified PACF using Yule-Walker equations
        acf_vals = self._manual_acf(values, max_lag)
        pacf_values = np.zeros(max_lag + 1)
        pacf_values[0] = 1.0
        
        if len(acf_vals) > 1:
            pacf_values[1] = acf_vals[1]
            
        # For higher orders, use approximation
        for k in range(2, min(max_lag + 1, len(acf_vals))):
            pacf_values[k] = acf_vals[k]  # Simplified approximation
            
        return pacf_values
    
    def _test_stationarity(self, values: np.ndarray) -> Dict:
        """Enhanced stationarity testing."""
        clean_values = values[~np.isnan(values)]
        
        if len(clean_values) < 10:
            return {
                'is_stationary': False,
                'adf_pvalue': 1.0,
                'kpss_pvalue': 0.0,
                'differences_needed': 0
            }
        
        # Use existing preprocessor method
        stationarity_result = self.preprocessor.test_stationarity(clean_values)
        
        # Test how many differences are needed
        differences_needed = 0
        current_data = clean_values.copy()
        
        for d in range(3):  # Test up to 3 differences
            test_result = self.preprocessor.test_stationarity(current_data)
            if test_result['is_stationary']:
                break
            current_data = np.diff(current_data)
            differences_needed += 1
            if len(current_data) < 10:  # Too short
                break
        
        return {
            'is_stationary': stationarity_result['is_stationary'],
            'adf_pvalue': stationarity_result['adf_pvalue'],
            'kpss_pvalue': stationarity_result['kpss_pvalue'],
            'differences_needed': differences_needed
        }
    
    def _extract_advanced_features(self, values: np.ndarray) -> Dict:
        """Extract advanced features like Hurst exponent, entropy, etc."""
        clean_values = values[~np.isnan(values)]
        
        if len(clean_values) < 10:
            return {'hurst': 0.5, 'entropy': 0.0, 'complexity': 0.0}
        
        hurst = self._calculate_hurst_exponent(clean_values)
        entropy = self._calculate_sample_entropy(clean_values)
        complexity = self._calculate_lz_complexity(clean_values)
        
        return {
            'hurst': float(hurst),
            'entropy': float(entropy),
            'complexity': float(complexity)
        }
    
    def _calculate_hurst_exponent(self, values: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        try:
            n = len(values)
            if n < 20:
                return 0.5
            
            # R/S analysis
            lags = range(2, min(n // 4, 100))
            rs_values = []
            
            for lag in lags:
                # Split into blocks
                blocks = n // lag
                if blocks < 2:
                    continue
                
                rs_block = []
                for i in range(blocks):
                    block = values[i * lag:(i + 1) * lag]
                    if len(block) < lag:
                        continue
                    
                    # Mean-adjusted series
                    y = block - np.mean(block)
                    
                    # Cumulative sum
                    z = np.cumsum(y)
                    
                    # Range
                    R = np.max(z) - np.min(z)
                    
                    # Standard deviation
                    S = np.std(block) + 1e-12
                    
                    rs_block.append(R / S)
                
                if rs_block:
                    rs_values.append(np.mean(rs_block))
            
            if len(rs_values) < 3:
                return 0.5
            
            # Fit power law: R/S ~ lag^H
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            hurst = np.polyfit(log_lags, log_rs, 1)[0]
            return max(0.0, min(1.0, hurst))  # Bound between 0 and 1
            
        except:
            return 0.5
    
    def _calculate_sample_entropy(self, values: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate sample entropy."""
        try:
            n = len(values)
            if n < 10:
                return 0.0
            
            if r is None:
                r = 0.2 * np.std(values)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([values[i:i + m] for i in range(n - m + 1)])
                C = np.zeros(n - m + 1)
                
                for i in range(n - m + 1):
                    template = patterns[i]
                    matches = [1 for j in range(n - m + 1) if _maxdist(template, patterns[j], m) <= r]
                    C[i] = sum(matches) / float(n - m + 1)
                
                return np.mean(np.log(C + 1e-12))
            
            return _phi(m) - _phi(m + 1)
            
        except:
            return 0.0
    
    def _calculate_lz_complexity(self, values: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity (simplified)."""
        try:
            # Binarize the signal
            binary = (values > np.median(values)).astype(int)
            
            # LZ complexity calculation
            n = len(binary)
            if n < 2:
                return 0.0
            
            i, k, l = 0, 1, 1
            c = 1
            k_max = 1
            
            while True:
                if binary[i + k - 1] == binary[l + k - 1]:
                    k += 1
                    if l + k >= n:
                        c += 1
                        break
                else:
                    if k > k_max:
                        k_max = k
                    i += 1
                    if i == l:
                        c += 1
                        l += k_max
                        if l + 1 > n:
                            break
                        else:
                            i = 0
                            k = 1
                            k_max = 1
                    else:
                        k = 1
            
            # Normalize
            return c * np.log2(n) / n if n > 0 else 0.0
            
        except:
            return 0.0
    
    def _decompose_series(self, values: np.ndarray, freq: Optional[str]) -> Optional[Dict]:
        """Enhanced series decomposition."""
        if not HAS_STATSMODELS or len(values) < 24:
            return None
        
        try:
            period = self._infer_period(freq)
            if len(values) < 2 * period:
                return None
            
            # Try STL decomposition first (more robust)
            try:
                stl = STL(values, seasonal=period, robust=True)
                result = stl.fit()
                
                return {
                    'trend': result.trend,
                    'seasonal': result.seasonal,
                    'residual': result.resid,
                    'period': period,
                    'method': 'STL'
                }
            except:
                # Fallback to classical decomposition
                decomposition = seasonal_decompose(
                    values, 
                    model='additive',
                    period=period,
                    extrapolate_trend='freq'
                )
                
                return {
                    'trend': decomposition.trend,
                    'seasonal': decomposition.seasonal,
                    'residual': decomposition.resid,
                    'period': period,
                    'method': 'classical'
                }
                
        except Exception:
            return None
    
    def _infer_period(self, freq: Optional[str]) -> int:
        """Infer period from frequency."""
        if not freq:
            return 12
            
        freq_upper = freq.upper()
        period_map = {
            'D': 7, 'B': 5, 'W': 52, 'M': 12, 'Q': 4,
            'H': 24, 'T': 60, 'MIN': 60, 'S': 60
        }
        
        for key, period in period_map.items():
            if key in freq_upper:
                return period
                
        return 12
    
    def clear_cache(self):
        """Clear feature cache."""
        self._feature_cache.clear()