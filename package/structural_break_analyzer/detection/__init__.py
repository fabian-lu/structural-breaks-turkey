"""Detection methods for structural breaks."""

from .base import DetectionMethod, BreakPoint, DetectionResult

# Import from categorized modules
from .changepoint import PeltDetector, BinarySegmentationDetector, DynamicProgrammingDetector
from .statistical import BaiPerronDetector, CUSUMDetector, ChowTestDetector
from .ensemble import VotingEnsembleDetector

__all__ = [
    # Base classes
    "DetectionMethod",
    "BreakPoint", 
    "DetectionResult",
    
    # Change-point methods
    "PeltDetector",
    "BinarySegmentationDetector", 
    "DynamicProgrammingDetector",
    
    # Statistical methods
    "BaiPerronDetector",
    "CUSUMDetector",
    "ChowTestDetector",
    
    # Ensemble methods
    "VotingEnsembleDetector"
]