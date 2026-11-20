"""
Markdown Report Formatter.

This module generates Markdown reports that can be easily converted
to other formats or used in documentation systems.
"""

import io
import base64
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime


class MarkdownFormatter:
    """Format analysis results as Markdown reports."""
    
    def __init__(self):
        """Initialize the Markdown formatter."""
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
        Generate Markdown report from context data.
        
        Args:
            context: Report context with analysis data
            output_path: Directory to save the report
            filename: Base filename (without extension)
            
        Returns:
            Path to the generated Markdown file
        """
        # Generate plots and save as separate files
        plot_files = self._save_plots(context, output_path, filename)
        
        # Generate markdown content
        markdown_content = self._build_markdown_report(context, plot_files)
        
        # Write to file
        output_file = output_path / f"{filename}.md"
        output_file.write_text(markdown_content, encoding='utf-8')
        
        return output_file
    
    def _save_plots(self, context: Dict[str, Any], output_path: Path, filename: str) -> Dict[str, str]:
        """Save plots as separate PNG files and return their paths."""
        plot_files = {}
        
        try:
            # Save time series plot
            if context.get('has_timeseries'):
                ts_plot_file = self._save_timeseries_plot(context['timeseries'], output_path, filename)
                if ts_plot_file:
                    plot_files['timeseries'] = ts_plot_file.name
            
            # Save breaks plot
            if context.get('has_breaks'):
                breaks_plot_file = self._save_breaks_plot(context['breaks'], output_path, filename)
                if breaks_plot_file:
                    plot_files['breaks'] = breaks_plot_file.name
                    
        except Exception as e:
            print(f"Warning: Error saving plots: {e}")
        
        return plot_files
    
    def _save_timeseries_plot(self, timeseries_context: Dict[str, Any], output_path: Path, filename: str) -> Optional[Path]:
        """Save time series plot as PNG file."""
        try:
            result = timeseries_context['result']
            
            # Get the actual result (handle both explained and raw results)
            if timeseries_context['is_explained']:
                actual_result = result.original_result
            else:
                actual_result = result
            
            # Use the built-in plot method
            fig = actual_result.plot(figsize=(12, 8))
            
            # Save plot
            plot_file = output_path / f"{filename}_timeseries.png"
            fig.savefig(plot_file, bbox_inches='tight', facecolor='white', dpi=150)
            plt.close(fig)
            
            return plot_file
            
        except Exception as e:
            print(f"Error saving time series plot: {e}")
            return None
    
    def _save_breaks_plot(self, breaks_context: Dict[str, Any], output_path: Path, filename: str) -> Optional[Path]:
        """Save structural breaks plot as PNG file."""
        try:
            result = breaks_context['result']
            
            # Get the actual result (handle both explained and raw results)
            if breaks_context['is_explained']:
                actual_result = result.original_result
            else:
                actual_result = result
            
            # Use the built-in plot method
            fig = actual_result.plot(figsize=(14, 8))
            
            # Save plot
            plot_file = output_path / f"{filename}_breaks.png"
            fig.savefig(plot_file, bbox_inches='tight', facecolor='white', dpi=150)
            plt.close(fig)
            
            return plot_file
            
        except Exception as e:
            print(f"Error saving breaks plot: {e}")
            return None
    
    def _build_markdown_report(self, context: Dict[str, Any], plot_files: Dict[str, str]) -> str:
        """Build the complete Markdown report."""
        lines = []
        
        # Title and metadata
        lines.extend([
            f"# {context['title']}",
            "",
            "**Comprehensive Structural Break Analysis**",
            "",
            f"*Generated on: {context['generated_at'].strftime('%B %d, %Y at %I:%M %p')}*",
            "",
            "---",
            ""
        ])
        
        # Table of Contents
        lines.extend([
            "## Table of Contents",
            ""
        ])
        
        if context.get('has_timeseries') or context.get('has_breaks'):
            lines.append("- [Executive Summary](#executive-summary)")
            lines.append("- [Key Statistics](#key-statistics)")
        
        if context.get('has_timeseries'):
            lines.append("- [Time Series Analysis](#time-series-analysis)")
        
        if context.get('has_breaks'):
            lines.append("- [Structural Break Analysis](#structural-break-analysis)")
        
        lines.extend([
            "- [Technical Details](#technical-details)",
            "",
            "---",
            ""
        ])
        
        # Executive Summary
        lines.extend([
            "## Executive Summary",
            "",
            self._build_executive_summary(context),
            "",
            "---",
            ""
        ])
        
        # Key Statistics
        if context.get('has_timeseries') or context.get('has_breaks'):
            lines.extend([
                "## Key Statistics",
                ""
            ])
            
            # Statistics table
            stats_table = self._build_stats_table(context)
            lines.extend(stats_table)
            lines.extend(["", "---", ""])
        
        # Time Series Analysis
        if context.get('has_timeseries'):
            lines.extend(self._build_timeseries_section(context, plot_files))
        
        # Break Analysis
        if context.get('has_breaks'):
            lines.extend(self._build_breaks_section(context, plot_files))
        
        # Technical Details
        lines.extend(self._build_technical_section(context))
        
        # Footer
        lines.extend([
            "---",
            "",
            f"*Report generated by [Structural Break Analyzer](https://github.com/your-repo/structural-break-analyzer) • {context['generated_at'].year}*"
        ])
        
        return "\n".join(lines)
    
    def _build_executive_summary(self, context: Dict[str, Any]) -> str:
        """Build executive summary text."""
        summary_parts = [
            "This report provides a comprehensive analysis of"
        ]
        
        if context.get('has_timeseries') and context.get('has_breaks'):
            summary_parts.append("time series characteristics and structural breaks")
        elif context.get('has_timeseries'):
            summary_parts.append("time series characteristics")  
        else:
            summary_parts.append("structural breaks")
        
        summary_parts.append("in the provided dataset.")
        
        if context.get('has_breaks'):
            n_breaks = context['breaks']['n_breaks']
            method = context['breaks']['method']
            summary_parts.append(f" The analysis identified **{n_breaks}** structural break{'s' if n_breaks != 1 else ''} using the **{method}** method.")
        
        if context.get('has_breaks') and context['breaks'].get('is_explained') and context['breaks'].get('rag_used'):
            summary_parts.append(" Context-aware explanations were generated using external documentation to provide deeper insights into the underlying causes of detected patterns.")
        
        return " ".join(summary_parts)
    
    def _build_stats_table(self, context: Dict[str, Any]) -> list:
        """Build key statistics table."""
        lines = ["| Statistic | Value |", "|-----------|--------|"]
        
        if context.get('has_timeseries'):
            ts_result = context['timeseries']['result']
            if context['timeseries']['is_explained']:
                features = ts_result.original_result.features
            else:
                features = ts_result.features
            
            lines.extend([
                f"| Mean Value | {features.mean:.4f} |",
                f"| Standard Deviation | {features.std:.4f} |",
                f"| Stationary | {'Yes' if features.is_stationary else 'No'} |"
            ])
        
        if context.get('has_breaks'):
            lines.append(f"| Breaks Detected | {context['breaks']['n_breaks']} |")
            lines.append(f"| Detection Method | {context['breaks']['method'].title()} |")
        
        return lines
    
    def _build_timeseries_section(self, context: Dict[str, Any], plot_files: Dict[str, str]) -> list:
        """Build time series analysis section."""
        lines = [
            "## Time Series Analysis",
            ""
        ]
        
        # Add LLM enhancement badge if applicable
        if context['timeseries']['is_explained']:
            lines.append("🤖 *Enhanced with AI explanations*")
            lines.append("")
        
        # Add plot
        if 'timeseries' in plot_files:
            lines.extend([
                "### Time Series Overview",
                "",
                f"![Time Series Plot](./{plot_files['timeseries']})",
                ""
            ])
        
        # Add features table
        lines.extend([
            "### Statistical Features",
            ""
        ])
        
        ts_result = context['timeseries']['result']
        if context['timeseries']['is_explained']:
            features = ts_result.original_result.features
        else:
            features = ts_result.features
        
        features_table = [
            "| Feature | Value | Description |",
            "|---------|--------|-------------|",
            f"| Mean | {features.mean:.4f} | Average value of the time series |",
            f"| Standard Deviation | {features.std:.4f} | Measure of variability |",
            f"| Skewness | {features.skewness:.4f} | Asymmetry of the distribution |",
            f"| Kurtosis | {features.kurtosis:.4f} | Tail heaviness of the distribution |",
            f"| Trend Strength | {features.trend_strength:.4f} | Strength of linear trend component |",
            f"| Stationarity | {'Yes' if features.is_stationary else 'No'} | Whether the series is stationary |",
            f"| Hurst Exponent | {features.hurst_exponent:.4f} | Long-range dependence measure |"
        ]
        
        lines.extend(features_table)
        lines.append("")
        
        # Add LLM explanation if available
        if context['timeseries']['is_explained']:
            lines.extend([
                "### AI Analysis",
                "",
                ts_result.explanation,
                ""
            ])
            
            if ts_result.insights:
                lines.extend([
                    "### Key Insights",
                    ""
                ])
                for insight in ts_result.insights:
                    lines.append(f"- 💡 {insight}")
                lines.append("")
        
        lines.extend(["", "---", ""])
        return lines
    
    def _build_breaks_section(self, context: Dict[str, Any], plot_files: Dict[str, str]) -> list:
        """Build structural breaks analysis section."""
        lines = [
            "## Structural Break Analysis",
            ""
        ]
        
        # Add enhancement badges if applicable
        badges = []
        if context['breaks']['is_explained']:
            badges.append("🤖 *Enhanced with AI explanations*")
        if context['breaks'].get('rag_used'):
            badges.append("🔍 *Context-aware using external documents*")
        
        if badges:
            lines.extend(badges)
            lines.append("")
        
        # Add plot
        if 'breaks' in plot_files:
            lines.extend([
                "### Detected Structural Breaks",
                "",
                f"![Structural Breaks Plot](./{plot_files['breaks']})",
                ""
            ])
        
        # Add break details
        result = context['breaks']['result']
        if context['breaks']['is_explained']:
            break_result = result
            break_list = result.original_result.breaks
        else:
            break_result = result
            break_list = result.breaks
        
        lines.extend([
            "### Break Details",
            ""
        ])
        
        for i, break_point in enumerate(break_list):
            date_str = break_point.date.strftime('%Y-%m-%d') if hasattr(break_point.date, 'strftime') else str(break_point.date)
            
            # Direction emoji
            direction_emoji = "📈" if break_point.direction in ['up', 'increase'] else "📉"
            
            lines.extend([
                f"#### Break {i+1}: {date_str} {direction_emoji}",
                "",
                f"- **Confidence:** {break_point.confidence:.1%}",
                f"- **Direction:** {break_point.direction.title()}",
                f"- **Magnitude:** {break_point.magnitude:.3f}",
                f"- **Index:** {break_point.index}",
                ""
            ])
            
            # Add LLM explanation if available
            if context['breaks']['is_explained'] and hasattr(break_result, 'break_explanations') and i in break_result.break_explanations:
                lines.extend([
                    "**Analysis:**",
                    "",
                    break_result.break_explanations[i],
                    ""
                ])
        
        # Overall summary if available
        if context['breaks']['is_explained'] and hasattr(break_result, 'overall_summary') and break_result.overall_summary:
            lines.extend([
                "### Overall Analysis",
                "",
                break_result.overall_summary,
                ""
            ])
        
        lines.extend(["", "---", ""])
        return lines
    
    def _build_technical_section(self, context: Dict[str, Any]) -> list:
        """Build technical details section."""
        lines = [
            "## Technical Details",
            "",
            "### Analysis Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|--------|"
        ]
        
        if context.get('has_timeseries'):
            lines.extend([
                f"| Data Points | {context['timeseries']['data_shape'][0]} |",
                f"| Value Column | `{context['timeseries']['value_col']}` |",
                f"| Date Column | `{context['timeseries']['date_col']}` |"
            ])
        
        if context.get('has_breaks'):
            lines.extend([
                f"| Detection Method | {context['breaks']['method'].title()} |",
                f"| Breaks Detected | {context['breaks']['n_breaks']} |"
            ])
            
            if context['breaks']['is_explained']:
                lines.append(f"| RAG Context Used | {'Yes' if context['breaks'].get('rag_used') else 'No'} |")
        
        lines.append("")
        
        return lines