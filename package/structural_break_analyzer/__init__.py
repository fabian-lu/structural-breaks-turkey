"""
Structural Break Analyzer v2.0 - Clean Architecture

A Python package for analyzing structural breaks and time series features.

New clean architecture with two main classes:
- TimeSeriesAnalyzer: Pure time series feature extraction
- StructuralBreakAnalyzer: Pure structural break detection  
- LLMExplainer: Optional AI explanations (with RAG support only for breaks)
"""

__version__ = "2.0.0"
__author__ = "Fabian Lukassen"

# Core detection and time series components
from .detection.base import DetectionMethod, BreakPoint, DetectionResult
from .detection.statistical.bai_perron import BaiPerronDetector
from .detection.changepoint.pelt import PeltDetector
from .detection.changepoint.binary_segmentation import BinarySegmentationDetector
from .detection.changepoint.dynamic_programming import DynamicProgrammingDetector
from .detection.statistical.cusum import CUSUMDetector
from .detection.statistical.chow_test import ChowTestDetector
from .detection.ensemble.voting_ensemble import VotingEnsembleDetector
from .timeseries.preprocessor import TimeSeriesPreprocessor
from .timeseries.features import TimeSeriesFeatures

# NEW CLEAN API - Main classes and results
from .analyzer import (
    TimeSeriesAnalyzer,
    StructuralBreakAnalyzer, 
    TimeSeriesResult,
    BreakDetectionResult
)

# Optional LLM components
try:
    from .llm_explainer import (
        LLMExplainer,
        ExplainedTimeSeriesResult,
        ExplainedBreakDetectionResult
    )
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

# Optional Report Generator
try:
    from .report_generator import ReportGenerator
    _REPORT_AVAILABLE = True
except ImportError:
    _REPORT_AVAILABLE = False

# Optional Analysis Widget
try:
    from .analysis_widget import AnalysisWidget
    _WIDGET_AVAILABLE = True
except ImportError:
    _WIDGET_AVAILABLE = False

# Core API
__all__ = [
    # Main user interface (NEW CLEAN API)
    "TimeSeriesAnalyzer",
    "StructuralBreakAnalyzer", 
    "TimeSeriesResult",
    "BreakDetectionResult",
    
    # Core detection components
    "DetectionMethod",
    "BreakPoint", 
    "DetectionResult",
    "BaiPerronDetector",
    "PeltDetector",
    "BinarySegmentationDetector",
    "DynamicProgrammingDetector", 
    "CUSUMDetector",
    "ChowTestDetector",
    "VotingEnsembleDetector",
    
    # Time series components
    "TimeSeriesPreprocessor",
    "TimeSeriesFeatures",
]

# Add LLM components if available
if _LLM_AVAILABLE:
    __all__.extend([
        "LLMExplainer",
        "ExplainedTimeSeriesResult", 
        "ExplainedBreakDetectionResult",
    ])

# Add Report Generator if available
if _REPORT_AVAILABLE:
    __all__.extend([
        "ReportGenerator",
    ])

# Add Analysis Widget if available  
if _WIDGET_AVAILABLE:
    __all__.extend([
        "AnalysisWidget",
    ])