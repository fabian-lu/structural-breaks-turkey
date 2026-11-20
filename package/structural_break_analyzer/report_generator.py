"""
Report Generator for Structural Break Analysis.

This module provides the main ReportGenerator class for creating comprehensive
reports from analysis results in multiple formats (HTML, PDF, Markdown).
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import os
from datetime import datetime

from .analyzer import TimeSeriesResult, BreakDetectionResult
from .llm_explainer import ExplainedTimeSeriesResult, ExplainedBreakDetectionResult
from .narrative.formatters.html_formatter import HTMLFormatter
from .narrative.formatters.pdf_formatter import PDFFormatter
from .narrative.formatters.markdown_formatter import MarkdownFormatter


class ReportGenerator:
    """
    Generate comprehensive reports from structural break analysis results.
    
    Features:
    - Supports both raw and LLM-enhanced results
    - Multiple output formats: HTML, PDF, Markdown
    - Beautiful styling with embedded plots
    - Flexible template system
    """
    
    def __init__(self):
        """Initialize the report generator."""
        self.formatters = {
            'html': HTMLFormatter(),
            'pdf': PDFFormatter(),
            'markdown': MarkdownFormatter()
        }
    
    def generate_report(self,
                       timeseries_result: Optional[Union[TimeSeriesResult, ExplainedTimeSeriesResult]] = None,
                       break_result: Optional[Union[BreakDetectionResult, ExplainedBreakDetectionResult]] = None,
                       format: Union[str, List[str]] = 'pdf',
                       title: str = "Structural Break Analysis Report",
                       output_path: Union[str, Path] = ".",
                       filename: Optional[str] = None,
                       fast_pdf: bool = True,
                       **template_params) -> Dict[str, str]:
        """
        Generate formatted reports from analysis results.
        
        Args:
            timeseries_result: Time series analysis result (raw or LLM-enhanced)
            break_result: Break detection result (raw or LLM-enhanced)  
            format: Output format(s) - 'html', 'pdf', 'markdown', or list of formats
            title: Report title
            output_path: Directory to save reports (default: current directory)
            filename: Base filename (default: auto-generated from timestamp)
            fast_pdf: Use faster PDF generation (skips weasyprint, uses reportlab)
            **template_params: Additional parameters for template customization
            
        Returns:
            Dict mapping format -> file_path for generated reports
            
        Raises:
            ValueError: If no results provided or invalid format specified
        """
        # Validation
        if not timeseries_result and not break_result:
            raise ValueError("At least one of timeseries_result or break_result must be provided")
        
        # Normalize format to list
        if isinstance(format, str):
            format = [format]
        
        # Validate formats
        for fmt in format:
            if fmt not in self.formatters:
                available_formats = list(self.formatters.keys())
                raise ValueError(f"Unknown format: {fmt}. Choose from: {available_formats}")
        
        # Setup output path
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"structural_break_report_{timestamp}"
        
        # Prepare report context
        context = self._prepare_report_context(
            timeseries_result=timeseries_result,
            break_result=break_result,
            title=title,
            **template_params
        )
        
        # Generate reports in each format
        generated_files = {}
        
        for fmt in format:
            try:
                formatter = self.formatters[fmt]
                
                # Pass fast_pdf option to PDF formatter
                if fmt == 'pdf':
                    formatter.fast_mode = fast_pdf
                
                file_path = formatter.generate_report(
                    context=context,
                    output_path=output_path,
                    filename=filename
                )
                generated_files[fmt] = str(file_path)
                print(f"✅ Generated {fmt.upper()} report: {file_path}")
                
            except Exception as e:
                print(f"❌ Failed to generate {fmt.upper()} report: {e}")
                generated_files[fmt] = f"Error: {str(e)}"
        
        return generated_files
    
    def _prepare_report_context(self,
                               timeseries_result: Optional[Union[TimeSeriesResult, ExplainedTimeSeriesResult]],
                               break_result: Optional[Union[BreakDetectionResult, ExplainedBreakDetectionResult]],
                               title: str,
                               **template_params) -> Dict[str, Any]:
        """Prepare context data for report templates."""
        
        context = {
            'title': title,
            'generated_at': datetime.now(),
            'has_timeseries': timeseries_result is not None,
            'has_breaks': break_result is not None,
            'template_params': template_params
        }
        
        # Add timeseries data
        if timeseries_result:
            context['timeseries'] = {
                'result': timeseries_result,
                'is_explained': isinstance(timeseries_result, ExplainedTimeSeriesResult),
                'data_shape': timeseries_result.data.shape if hasattr(timeseries_result, 'data') else timeseries_result.original_result.data.shape,
                'value_col': timeseries_result.value_col if hasattr(timeseries_result, 'value_col') else timeseries_result.original_result.value_col,
                'date_col': timeseries_result.date_col if hasattr(timeseries_result, 'date_col') else timeseries_result.original_result.date_col
            }
        
        # Add break detection data
        if break_result:
            context['breaks'] = {
                'result': break_result,
                'is_explained': isinstance(break_result, ExplainedBreakDetectionResult),
                'n_breaks': break_result.n_breaks if hasattr(break_result, 'n_breaks') else break_result.original_result.n_breaks,
                'method': break_result.method if hasattr(break_result, 'method') else break_result.original_result.method,
                'rag_used': getattr(break_result, 'rag_used', False)
            }
        
        return context
    
    def list_available_formats(self) -> List[str]:
        """Get list of available output formats."""
        return list(self.formatters.keys())
    
    def get_formatter(self, format: str):
        """Get specific formatter instance."""
        if format not in self.formatters:
            available_formats = list(self.formatters.keys())
            raise ValueError(f"Unknown format: {format}. Choose from: {available_formats}")
        return self.formatters[format]