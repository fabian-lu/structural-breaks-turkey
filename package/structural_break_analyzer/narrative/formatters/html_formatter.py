"""
HTML Report Formatter.

This module generates beautiful HTML reports with embedded plots
and modern styling from analysis results.
"""

import io
import base64
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ...analyzer import TimeSeriesResult, BreakDetectionResult
from ...llm_explainer import ExplainedTimeSeriesResult, ExplainedBreakDetectionResult


class HTMLFormatter:
    """Format analysis results as beautiful HTML reports."""
    
    def __init__(self):
        """Initialize the HTML formatter."""
        # Setup Jinja2 template environment
        template_dir = Path(__file__).parent.parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Configure matplotlib for better plots
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'available') and 'seaborn-v0_8' in plt.style.available else 'default')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'savefig.dpi': 150,
            'figure.dpi': 150,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 10
        })
    
    def generate_report(self, 
                       context: Dict[str, Any],
                       output_path: Path,
                       filename: str) -> Path:
        """
        Generate HTML report from context data.
        
        Args:
            context: Report context with analysis data
            output_path: Directory to save the report
            filename: Base filename (without extension)
            
        Returns:
            Path to the generated HTML file
        """
        # Generate plots as base64 encoded images
        plot_data = self._generate_plots(context)
        context.update(plot_data)
        
        # Load and render template
        template = self.env.get_template('report_template.html')
        html_content = template.render(**context)
        
        # Write to file
        output_file = output_path / f"{filename}.html"
        output_file.write_text(html_content, encoding='utf-8')
        
        return output_file
    
    def _generate_plots(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Generate matplotlib plots and convert to base64 HTML."""
        plots = {}
        
        try:
            # Generate time series plot
            if context.get('has_timeseries'):
                plots['timeseries_plot_html'] = self._create_timeseries_plot(context['timeseries'])
            
            # Generate breaks plot
            if context.get('has_breaks'):
                plots['breaks_plot_html'] = self._create_breaks_plot(context['breaks'])
                
        except Exception as e:
            print(f"Warning: Error generating plots: {e}")
            plots['timeseries_plot_html'] = f"<p>Error generating time series plot: {e}</p>"
            plots['breaks_plot_html'] = f"<p>Error generating breaks plot: {e}</p>"
        
        return plots
    
    def _create_timeseries_plot(self, timeseries_context: Dict[str, Any]) -> str:
        """Create time series plot as base64 encoded HTML."""
        try:
            result = timeseries_context['result']
            
            # Get the actual result (handle both explained and raw results)
            if timeseries_context['is_explained']:
                actual_result = result.original_result
            else:
                actual_result = result
            
            # Use the built-in plot method
            fig = actual_result.plot(figsize=(12, 8))
            
            # Convert to base64
            img_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            
            return f'<img src="data:image/png;base64,{img_base64}" alt="Time Series Plot">'
            
        except Exception as e:
            print(f"Error creating time series plot: {e}")
            return f"<p>Error generating time series plot: {e}</p>"
    
    def _create_breaks_plot(self, breaks_context: Dict[str, Any]) -> str:
        """Create structural breaks plot as base64 encoded HTML."""
        try:
            result = breaks_context['result']
            
            # Get the actual result (handle both explained and raw results)
            if breaks_context['is_explained']:
                actual_result = result.original_result
            else:
                actual_result = result
            
            # Use the built-in plot method
            fig = actual_result.plot(figsize=(14, 8))
            
            # Convert to base64
            img_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            
            return f'<img src="data:image/png;base64,{img_base64}" alt="Structural Breaks Plot">'
            
        except Exception as e:
            print(f"Error creating breaks plot: {e}")
            return f"<p>Error generating breaks plot: {e}</p>"
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 encoded string."""
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        img_buffer.close()
        return img_base64