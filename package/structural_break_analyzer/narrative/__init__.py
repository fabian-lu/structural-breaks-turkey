"""
Narrative module for report generation and data storytelling.

This module contains the infrastructure for creating beautiful reports
and narratives from structural break analysis results.
"""

from .formatters.html_formatter import HTMLFormatter
from .formatters.pdf_formatter import PDFFormatter
from .formatters.markdown_formatter import MarkdownFormatter

__all__ = [
    'HTMLFormatter',
    'PDFFormatter', 
    'MarkdownFormatter'
]