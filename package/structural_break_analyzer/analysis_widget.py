"""
Interactive Analysis Widget for Structural Break Analysis.

This module provides the top-level API for the comprehensive Jupyter widget.
The actual implementation is in narrative/widgets/interactive_widget.py

Usage:
    from structural_break_analyzer import AnalysisWidget
    
    widget = AnalysisWidget()
    widget.show()  # Display the interactive interface
"""

from .narrative.widgets.interactive_widget import InteractiveAnalysisWidget


class AnalysisWidget:
    """
    Top-level API for the interactive analysis widget.
    
    This is a clean wrapper around the actual implementation,
    following the same pattern as the other top-level modules.
    """
    
    def __init__(self):
        """Initialize the analysis widget."""
        # Delegate to the actual implementation
        self._widget = InteractiveAnalysisWidget()
    
    def show(self):
        """Display the widget interface."""
        return self._widget.show()
    
    def servable(self):
        """Make the widget servable with Panel serve."""
        return self._widget.servable()
    
    # Provide access to the underlying widget for advanced usage
    @property
    def widget(self):
        """Access the underlying widget implementation."""
        return self._widget


# For easy importing
def create_analysis_widget():
    """Create and return an AnalysisWidget instance."""
    return AnalysisWidget()


