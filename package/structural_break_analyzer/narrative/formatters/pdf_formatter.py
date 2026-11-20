"""
PDF Report Formatter.

This module generates PDF reports from HTML using weasyprint or
falls back to basic reportlab generation.
"""

import io
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

from .html_formatter import HTMLFormatter


class PDFFormatter:
    """Format analysis results as PDF reports."""
    
    def __init__(self):
        """Initialize the PDF formatter."""
        self.html_formatter = HTMLFormatter()
        self.fast_mode = False  # Can be set to skip weasyprint
        
        # Try to import weasyprint for better PDF generation
        self.weasyprint_available = False
        try:
            import weasyprint
            self.weasyprint_available = True
        except ImportError:
            print("Note: weasyprint not available. PDF generation will use basic fallback.")
            print("For better PDF reports, install: pip install weasyprint")
    
    def generate_report(self, 
                       context: Dict[str, Any],
                       output_path: Path,
                       filename: str) -> Path:
        """
        Generate PDF report from context data.
        
        Args:
            context: Report context with analysis data
            output_path: Directory to save the report
            filename: Base filename (without extension)
            
        Returns:
            Path to the generated PDF file
        """
        output_file = output_path / f"{filename}.pdf"
        
        # Use fast mode or fallback if requested, otherwise try weasyprint
        if self.fast_mode or not self.weasyprint_available:
            return self._generate_pdf_fallback(context, output_file)
        else:
            return self._generate_pdf_with_weasyprint(context, output_file)
    
    def _generate_pdf_with_weasyprint(self, context: Dict[str, Any], output_file: Path) -> Path:
        """Generate PDF using weasyprint (HTML to PDF conversion)."""
        try:
            import weasyprint
            from weasyprint import HTML, CSS
            
            # First generate HTML content
            html_formatter = HTMLFormatter()
            plot_data = html_formatter._generate_plots(context)
            context.update(plot_data)
            
            # Load and render template
            template = html_formatter.env.get_template('report_template.html')
            html_content = template.render(**context)
            
            # Additional CSS for PDF (optimized for speed)
            pdf_css = CSS(string="""
                @page {
                    margin: 2cm;
                    size: A4;
                }
                
                /* Use system fonts for faster rendering */
                body {
                    font-family: Arial, sans-serif !important;
                }
                
                .container {
                    box-shadow: none !important;
                    border-radius: 0 !important;
                }
                
                .header {
                    background: #667eea !important;
                    -webkit-print-color-adjust: exact;
                }
                
                .section {
                    page-break-inside: avoid;
                }
                
                .break-item {
                    page-break-inside: avoid;
                }
                
                /* Simplify complex styles for PDF */
                .stat-card {
                    box-shadow: none !important;
                    transform: none !important;
                }
            """)
            
            # Convert HTML to PDF with progress indication
            print("🔄 Converting HTML to PDF (this may take a moment)...")
            html_doc = HTML(string=html_content)
            html_doc.write_pdf(str(output_file), stylesheets=[pdf_css])
            
            return output_file
            
        except Exception as e:
            print(f"Error generating PDF with weasyprint: {e}")
            print("Falling back to basic PDF generation...")
            return self._generate_pdf_fallback(context, output_file)
    
    def _generate_pdf_fallback(self, context: Dict[str, Any], output_file: Path) -> Path:
        """Generate basic PDF using reportlab."""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
            
            # Create PDF document
            doc = SimpleDocTemplate(str(output_file), pagesize=A4, 
                                  rightMargin=72, leftMargin=72, 
                                  topMargin=72, bottomMargin=18)
            
            # Get styles
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#667eea')
            )
            
            subtitle_style = ParagraphStyle(
                'CustomSubtitle', 
                parent=styles['Normal'],
                fontSize=14,
                spaceAfter=20,
                alignment=TA_CENTER,
                textColor=colors.grey
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceBefore=20,
                spaceAfter=12,
                textColor=colors.HexColor('#2c3e50')
            )
            
            # Build document content
            story = []
            
            # Title page
            story.append(Spacer(1, 2*inch))
            story.append(Paragraph(context['title'], title_style))
            story.append(Paragraph("Comprehensive Structural Break Analysis", subtitle_style))
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph(f"Generated on {context['generated_at'].strftime('%B %d, %Y at %I:%M %p')}", subtitle_style))
            story.append(PageBreak())
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", heading_style))
            summary_text = self._build_executive_summary(context)
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # Time Series Analysis
            if context.get('has_timeseries'):
                story.append(Paragraph("Time Series Analysis", heading_style))
                
                # Add time series plot if available
                ts_plot_img = self._create_timeseries_plot_for_pdf(context['timeseries'])
                if ts_plot_img:
                    story.append(ts_plot_img)
                    story.append(Spacer(1, 0.2*inch))
                
                # Add features table
                features_table = self._create_features_table(context['timeseries'])
                if features_table:
                    story.append(features_table)
                    story.append(Spacer(1, 0.3*inch))
                
                # Add LLM explanation if available
                if context['timeseries']['is_explained']:
                    ts_result = context['timeseries']['result']
                    story.append(Paragraph("AI Analysis", heading_style))
                    story.append(Paragraph(ts_result.explanation, styles['Normal']))
                    story.append(Spacer(1, 0.3*inch))
            
            # Break Analysis
            if context.get('has_breaks'):
                story.append(Paragraph("Structural Break Analysis", heading_style))
                
                # Add breaks plot if available
                breaks_plot_img = self._create_breaks_plot_for_pdf(context['breaks'])
                if breaks_plot_img:
                    story.append(breaks_plot_img)
                    story.append(Spacer(1, 0.2*inch))
                
                # Add break details
                break_details = self._create_breaks_content(context['breaks'], styles)
                story.extend(break_details)
            
            # Build PDF
            doc.build(story)
            return output_file
            
        except ImportError as e:
            print(f"Error: reportlab not available for PDF generation: {e}")
            print("To enable PDF generation, install: pip install reportlab")
            # Create a simple text file as fallback
            output_file = output_file.with_suffix('.txt')
            output_file.write_text(self._create_text_fallback(context))
            return output_file
        except Exception as e:
            print(f"Error generating PDF: {e}")
            # Create a simple text file as fallback
            output_file = output_file.with_suffix('.txt')
            output_file.write_text(self._create_text_fallback(context))
            return output_file
    
    def _build_executive_summary(self, context: Dict[str, Any]) -> str:
        """Build executive summary text."""
        summary_parts = [
            f"This report provides a comprehensive analysis of"
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
            summary_parts.append(f" The analysis identified {n_breaks} structural break{'s' if n_breaks != 1 else ''} using the {method} method.")
        
        if context.get('has_breaks') and context['breaks'].get('is_explained') and context['breaks'].get('rag_used'):
            summary_parts.append(" Context-aware explanations were generated using external documentation to provide deeper insights into the underlying causes of detected patterns.")
        
        return " ".join(summary_parts)
    
    def _create_timeseries_plot_for_pdf(self, timeseries_context: Dict[str, Any]) -> Optional[Any]:
        """Create time series plot for PDF."""
        try:
            from reportlab.platypus import Image
            from reportlab.lib.units import inch
            
            result = timeseries_context['result']
            if timeseries_context['is_explained']:
                actual_result = result.original_result
            else:
                actual_result = result
            
            # Create plot
            fig = actual_result.plot(figsize=(8, 5))
            
            # Save to memory buffer
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight', 
                       facecolor='white', dpi=150)
            img_buffer.seek(0)
            plt.close(fig)
            
            # Create ReportLab Image
            return Image(img_buffer, width=6*inch, height=3.75*inch)
            
        except Exception as e:
            print(f"Error creating time series plot for PDF: {e}")
            return None
    
    def _create_breaks_plot_for_pdf(self, breaks_context: Dict[str, Any]) -> Optional[Any]:
        """Create breaks plot for PDF."""
        try:
            from reportlab.platypus import Image
            from reportlab.lib.units import inch
            
            result = breaks_context['result']
            if breaks_context['is_explained']:
                actual_result = result.original_result
            else:
                actual_result = result
            
            # Create plot
            fig = actual_result.plot(figsize=(8, 5))
            
            # Save to memory buffer
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight', 
                       facecolor='white', dpi=150)
            img_buffer.seek(0)
            plt.close(fig)
            
            # Create ReportLab Image
            return Image(img_buffer, width=6*inch, height=3.75*inch)
            
        except Exception as e:
            print(f"Error creating breaks plot for PDF: {e}")
            return None
    
    def _create_features_table(self, timeseries_context: Dict[str, Any]) -> Optional[Any]:
        """Create features table for PDF."""
        try:
            from reportlab.platypus import Table, TableStyle
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            
            result = timeseries_context['result']
            if timeseries_context['is_explained']:
                features = result.original_result.features
            else:
                features = result.features
            
            # Create table data
            data = [
                ['Feature', 'Value', 'Description'],
                ['Mean', f"{features.mean:.4f}", 'Average value of the time series'],
                ['Standard Deviation', f"{features.std:.4f}", 'Measure of variability'],
                ['Skewness', f"{features.skewness:.4f}", 'Asymmetry of the distribution'],
                ['Kurtosis', f"{features.kurtosis:.4f}", 'Tail heaviness of the distribution'],
                ['Trend Strength', f"{features.trend_strength:.4f}", 'Strength of linear trend component'],
                ['Stationarity', 'Yes' if features.is_stationary else 'No', 'Whether the series is stationary'],
                ['Hurst Exponent', f"{features.hurst_exponent:.4f}", 'Long-range dependence measure']
            ]
            
            # Create table
            table = Table(data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            return table
            
        except Exception as e:
            print(f"Error creating features table: {e}")
            return None
    
    def _create_breaks_content(self, breaks_context: Dict[str, Any], styles) -> list:
        """Create break analysis content for PDF."""
        content = []
        
        try:
            from reportlab.platypus import Paragraph, Spacer
            from reportlab.lib.units import inch
            
            result = breaks_context['result']
            if breaks_context['is_explained']:
                break_result = result
                break_list = result.original_result.breaks
            else:
                break_result = result
                break_list = result.breaks
            
            for i, break_point in enumerate(break_list):
                # Break header
                date_str = break_point.date.strftime('%Y-%m-%d') if hasattr(break_point.date, 'strftime') else str(break_point.date)
                header_text = f"Break {i+1}: {date_str} ({break_point.confidence:.1%} confidence)"
                content.append(Paragraph(header_text, styles['Heading3']))
                
                # Break details
                details_text = f"""
                <b>Direction:</b> {break_point.direction.title()}<br/>
                <b>Magnitude:</b> {break_point.magnitude:.3f}<br/>
                <b>Index:</b> {break_point.index}<br/>
                """
                content.append(Paragraph(details_text, styles['Normal']))
                
                # LLM explanation if available
                if breaks_context['is_explained'] and hasattr(break_result, 'break_explanations') and i in break_result.break_explanations:
                    content.append(Paragraph("<b>Analysis:</b>", styles['Normal']))
                    explanation_text = break_result.break_explanations[i].replace('\n', '<br/>')
                    content.append(Paragraph(explanation_text, styles['Normal']))
                
                content.append(Spacer(1, 0.2*inch))
            
            # Overall summary if available
            if breaks_context['is_explained'] and hasattr(break_result, 'overall_summary') and break_result.overall_summary:
                content.append(Paragraph("Overall Analysis", styles['Heading3']))
                overall_text = break_result.overall_summary.replace('\n', '<br/>')
                content.append(Paragraph(overall_text, styles['Normal']))
                
        except Exception as e:
            print(f"Error creating breaks content: {e}")
        
        return content
    
    def _create_text_fallback(self, context: Dict[str, Any]) -> str:
        """Create simple text fallback when PDF libraries are not available."""
        lines = [
            "=" * 60,
            context['title'].upper(),
            "=" * 60,
            f"Generated on: {context['generated_at'].strftime('%B %d, %Y at %I:%M %p')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 20,
            self._build_executive_summary(context),
            ""
        ]
        
        if context.get('has_timeseries'):
            lines.extend([
                "TIME SERIES ANALYSIS",
                "-" * 25,
                f"Data points: {context['timeseries']['data_shape'][0]}",
                ""
            ])
        
        if context.get('has_breaks'):
            lines.extend([
                "STRUCTURAL BREAK ANALYSIS", 
                "-" * 30,
                f"Breaks detected: {context['breaks']['n_breaks']}",
                f"Method: {context['breaks']['method']}",
                ""
            ])
        
        lines.extend([
            "=" * 60,
            "Note: This is a simplified text version.",
            "For full reports with plots, install: pip install weasyprint reportlab",
            "=" * 60
        ])
        
        return "\n".join(lines)