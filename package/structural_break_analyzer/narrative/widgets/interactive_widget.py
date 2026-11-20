"""
Core interactive widget implementation for structural break analysis.

This module contains the main widget logic, keeping the top-level API clean.
"""

import pandas as pd
import numpy as np
import io
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Widget framework  
try:
    import panel as pn
    try:
        pn.extension('tabulator', template='material')
    except:
        pn.extension()  # Fallback to basic extension
    _PANEL_AVAILABLE = True
except ImportError:
    _PANEL_AVAILABLE = False

# Plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

# Core analysis components
from ...analyzer import TimeSeriesAnalyzer, StructuralBreakAnalyzer
from ...llm_explainer import LLMExplainer
from ...report_generator import ReportGenerator


class InteractiveAnalysisWidget:
    """
    Core implementation of the interactive analysis widget.
    
    This contains all the logic while keeping the top-level API simple.
    """
    
    def __init__(self):
        """Initialize the widget implementation."""
        if not _PANEL_AVAILABLE:
            raise ImportError("Panel is required. Install with: pip install panel")
        
        # Core analyzers
        self.ts_analyzer = TimeSeriesAnalyzer()
        self.break_analyzer = StructuralBreakAnalyzer() 
        self.llm_explainer = None
        self.report_generator = ReportGenerator()
        
        # Data storage
        self.data = None
        self.ts_results = None
        self.break_results = None
        self.explained_ts_results = None
        self.explained_break_results = None
        
        # Initialize UI components
        self._init_components()
        self._init_layout()
        
    def _init_components(self):
        """Initialize all UI components."""
        
        # === DATA SECTION ===
        self.file_input = pn.widgets.FileInput(
            accept='.csv,.xlsx,.json',
            multiple=False,
            name="📁 Upload Data File",
            height=80
        )
        
        self.data_preview = pn.pane.DataFrame(
            pd.DataFrame(), 
            width=600,
            height=300
        )
        
        self.column_selectors = pn.Column(
            pn.widgets.Select(name="📅 Date Column", options=[], value=None),
            pn.widgets.Select(name="📊 Value Column", options=[], value=None),
            pn.widgets.Select(name="⏰ Frequency", 
                            options=['Auto', 'D', 'W', 'M', 'Q', 'Y'], 
                            value='Auto'),
            width=300
        )
        
        # === ANALYSIS CONFIGURATION ===
        # Store config widgets for easy access
        self.ts_config = self._create_ts_config()
        self.break_config = self._create_break_config() 
        self.llm_config = self._create_llm_config()
        self.report_config = self._create_report_config()
        
        self.analysis_tabs = pn.Tabs(
            ("🔍 Time Series Analysis", self.ts_config),
            ("⚡ Break Detection", self.break_config),
            ("🤖 LLM Enhancement", self.llm_config),
            ("📄 Report Generation", self.report_config),
            dynamic=True
        )
        
        # === RESULTS SECTION ===
        self.results_tabs = pn.Tabs(
            ("📈 Time Series Results", pn.pane.Markdown("Run analysis to see results")),
            ("⚡ Break Detection Results", pn.pane.Markdown("Run analysis to see results")),
            ("📊 Plots", pn.pane.Markdown("Run analysis to see plots")),
            dynamic=True
        )
        
        # === CONTROL BUTTONS ===
        self.run_button = pn.widgets.Button(
            name="🚀 Run Analysis", 
            button_type='primary',
            height=50,
            width=200
        )
        
        self.status_indicator = pn.pane.Markdown("**Status:** Ready", margin=(10, 0))
        
        # === SETUP CALLBACKS ===
        self.file_input.param.watch(self._on_file_upload, 'value')
        self.run_button.on_click(self._run_analysis)
        
    def _create_ts_config(self):
        """Create time series analysis configuration panel."""
        return pn.Column(
            pn.pane.Markdown("### Time Series Analysis Configuration"),
            pn.widgets.Checkbox(name="Enable Time Series Analysis", value=True),
            pn.widgets.Checkbox(name="Generate Plots", value=True),
            pn.pane.Markdown("*Extracts comprehensive time series features including trend, seasonality, stationarity, volatility, and complexity measures.*"),
            width=400
        )
    
    def _create_break_config(self):
        """Create break detection configuration panel."""
        return pn.Column(
            pn.pane.Markdown("### Break Detection Configuration"),
            pn.widgets.Checkbox(name="Enable Break Detection", value=True),
            pn.widgets.Select(
                name="Detection Method",
                options=['auto', 'pelt', 'bai_perron', 'binary_segmentation', 'cusum', 'ensemble'],
                value='auto'
            ),
            pn.widgets.IntSlider(name="Max Breaks (if applicable)", start=1, end=10, value=5),
            pn.widgets.FloatSlider(name="Confidence Threshold", start=0.8, end=0.99, value=0.95, step=0.01),
            pn.pane.Markdown("*Detects structural breaks in the time series using advanced statistical methods.*"),
            width=400
        )
    
    def _create_llm_config(self):
        """Create LLM enhancement configuration panel."""
        return pn.Column(
            pn.pane.Markdown("### LLM Enhancement Configuration"),
            pn.widgets.Checkbox(name="Enable LLM Explanations", value=False),
            pn.widgets.Select(
                name="LLM Provider",
                options=['azure', 'openai', 'anthropic'],
                value='azure'
            ),
            pn.widgets.PasswordInput(name="API Key", placeholder="Enter your API key"),
            pn.widgets.TextInput(name="Endpoint (Azure)", placeholder="https://your-endpoint.openai.azure.com/"),
            pn.widgets.TextInput(name="Deployment Name (Azure)", placeholder="gpt-4"),
            pn.widgets.Select(
                name="Detail Level",
                options=['basic', 'medium', 'detailed'],
                value='medium'
            ),
            pn.widgets.Checkbox(name="Enable RAG Context", value=False),
            pn.widgets.FileInput(name="📚 Upload Context Documents", accept='.txt,.md,.pdf', multiple=True),
            pn.pane.Markdown("*Add AI-powered explanations to understand the meaning behind detected patterns.*"),
            width=400
        )
    
    def _create_report_config(self):
        """Create report generation configuration panel."""
        self.generate_report_button = pn.widgets.Button(
            name="📄 Generate Report", 
            button_type='primary',
            width=200,
            height=40
        )
        self.generate_report_button.on_click(self._generate_report)
        
        return pn.Column(
            pn.pane.Markdown("### Report Generation Configuration"),
            pn.widgets.CheckBoxGroup(
                name="Output Formats",
                options=['html', 'pdf', 'markdown'],
                value=['html'],
                inline=True
            ),
            pn.widgets.TextInput(name="Report Title", value="Structural Break Analysis Report"),
            pn.widgets.TextInput(name="Filename", placeholder="Auto-generated if empty"),
            pn.widgets.Checkbox(name="Fast PDF Mode", value=True),
            self.generate_report_button,
            pn.pane.HTML("<div id='report-download-links'></div>"),
            pn.pane.Markdown("*Generate beautiful reports in multiple formats for sharing and documentation.*"),
            width=400
        )
        
    def _init_layout(self):
        """Initialize the main layout."""
        # Header
        header = pn.pane.Markdown(
            "# 🔬 Interactive Structural Break Analysis\n"
            "Upload your data and configure analysis parameters to get started.",
            sizing_mode='stretch_width'
        )
        
        # Data section
        data_section = pn.Card(
            pn.Row(
                pn.Column(
                    self.file_input,
                    self.column_selectors,
                    width=350
                ),
                pn.Column(
                    self.data_preview,
                    min_width=600
                )
            ),
            title="📊 Data Input & Preview",
            collapsed=False
        )
        
        # Configuration section
        config_section = pn.Card(
            self.analysis_tabs,
            title="⚙️ Analysis Configuration", 
            collapsed=False
        )
        
        # Controls section
        controls_section = pn.Row(
            self.run_button,
            self.status_indicator,
            sizing_mode='stretch_width'
        )
        
        # Results section
        results_section = pn.Card(
            self.results_tabs,
            title="📈 Analysis Results",
            collapsed=False
        )
        
        # Main layout
        self.layout = pn.Column(
            header,
            data_section,
            config_section, 
            controls_section,
            results_section,
            sizing_mode='stretch_width',
            margin=20
        )
    
    def _on_file_upload(self, event):
        """Handle file upload."""
        if event.new is None or not self.file_input.value:
            return
            
        try:
            self.status_indicator.object = "**Status:** 📥 Loading data..."
            
            # Read the file
            filename = self.file_input.filename
            file_content = self.file_input.value
            
            # Handle different file formats
            if filename.endswith('.csv'):
                # Try different encodings for CSV files
                try:
                    self.data = pd.read_csv(io.BytesIO(file_content))
                except UnicodeDecodeError:
                    self.data = pd.read_csv(io.BytesIO(file_content), encoding='latin1')
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                self.data = pd.read_excel(io.BytesIO(file_content))
            else:
                raise ValueError(f"Unsupported file format: {filename}. Please use CSV or Excel files.")
            
            # Validate data
            if self.data.empty:
                raise ValueError("The uploaded file is empty")
            
            if len(self.data.columns) < 2:
                raise ValueError("Data must have at least 2 columns (date and value)")
            
            # Update preview
            self.data_preview.object = self.data.head(100)
            
            # Update column selectors
            columns = list(self.data.columns)
            date_col_widget = self.column_selectors[0]
            value_col_widget = self.column_selectors[1]
            
            # Clear previous selections first
            date_col_widget.value = None
            value_col_widget.value = None
            
            # Update options
            date_col_widget.options = [''] + columns
            value_col_widget.options = [''] + columns
            
            # Auto-detect columns with better patterns
            date_cols = [col for col in columns if any(word in col.lower() for word in ['date', 'time', 'day', 'timestamp', 'period'])]
            value_cols = [col for col in columns if any(word in col.lower() for word in ['value', 'price', 'amount', 'rate', 'close', 'level', 'data'])]
            
            # Also check for numeric columns as potential value columns
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(self.data[col])]
            
            # Set auto-detected values
            if date_cols:
                date_col_widget.value = date_cols[0]
            elif columns:
                # If no clear date column, suggest the first column
                date_col_widget.value = columns[0]
                
            if value_cols:
                value_col_widget.value = value_cols[0]
            elif numeric_cols:
                # If no clear value column, suggest the first numeric column
                value_col_widget.value = numeric_cols[0]
            elif len(columns) > 1:
                # Otherwise suggest the second column
                value_col_widget.value = columns[1]
            
            self.status_indicator.object = f"**Status:** ✅ Data loaded ({len(self.data)} rows, {len(self.data.columns)} columns)"
            
        except Exception as e:
            self.status_indicator.object = f"**Status:** ❌ Error loading data: {str(e)}"
            self.data = None
            self.data_preview.object = pd.DataFrame()  # Clear preview on error
    
    def _run_analysis(self, event):
        """Run the complete analysis pipeline."""
        if self.data is None:
            self.status_indicator.object = "**Status:** ❌ Please upload data first"
            return
            
        try:
            self.status_indicator.object = "**Status:** 🔄 Running analysis..."
            
            # Get configuration
            date_col = self.column_selectors[0].value
            value_col = self.column_selectors[1].value
            freq = self.column_selectors[2].value
            freq = None if freq == 'Auto' else freq
            
            if not date_col or not value_col:
                raise ValueError("Please select date and value columns")
            
            # Ensure date column is properly parsed as datetime
            if date_col in self.data.columns:
                try:
                    self.data[date_col] = pd.to_datetime(self.data[date_col])
                    self.status_indicator.object = "**Status:** 🔄 Date column converted to datetime..."
                except Exception as e:
                    self.status_indicator.object = f"**Status:** ⚠️ Warning: Could not parse dates in '{date_col}': {e}"
            
            # Get analysis settings
            config = self._get_analysis_config()
            
            # Run time series analysis
            if config['ts_enabled']:
                self.status_indicator.object = "**Status:** 🔄 Running time series analysis..."
                self.ts_results = self.ts_analyzer.analyze(
                    self.data, 
                    value_col=value_col, 
                    date_col=date_col, 
                    freq=freq
                )
            
            # Run break detection
            if config['break_enabled']:
                self.status_indicator.object = "**Status:** 🔄 Running break detection..."
                self.break_results = self.break_analyzer.detect(
                    self.data,
                    value_col=value_col,
                    date_col=date_col,
                    method=config['break_method']
                )
            
            # Run LLM enhancement if enabled
            if config['llm_enabled']:
                self._setup_llm_explainer(config)
                
                if self.ts_results and self.llm_explainer:
                    self.status_indicator.object = "**Status:** 🔄 Generating time series explanations..."
                    self.explained_ts_results = self.llm_explainer.explain_timeseries(
                        self.ts_results,
                        detail_level=config['detail_level']
                    )
                
                if self.break_results and self.llm_explainer:
                    self.status_indicator.object = "**Status:** 🔄 Generating break explanations..."
                    self.explained_break_results = self.llm_explainer.explain_breaks(
                        self.break_results,
                        use_rag=config['rag_enabled']
                    )
            
            # Update results display
            self._update_results_display()
            
            self.status_indicator.object = "**Status:** ✅ Analysis complete!"
            
        except Exception as e:
            self.status_indicator.object = f"**Status:** ❌ Error: {str(e)}"
    
    def _get_analysis_config(self):
        """Extract configuration from UI components."""
        try:
            return {
                'ts_enabled': self.ts_config[1].value,  # "Enable Time Series Analysis" checkbox
                'break_enabled': self.break_config[1].value,  # "Enable Break Detection" checkbox
                'break_method': self.break_config[2].value,  # Method selection
                'llm_enabled': self.llm_config[1].value,  # "Enable LLM Explanations" checkbox
                'llm_provider': self.llm_config[2].value,  # Provider selection
                'api_key': self.llm_config[3].value,  # API Key
                'endpoint': self.llm_config[4].value,  # Azure endpoint
                'deployment_name': self.llm_config[5].value,  # Azure deployment
                'detail_level': self.llm_config[6].value,  # Detail level selection
                'rag_enabled': self.llm_config[7].value,  # "Enable RAG Context" checkbox
            }
        except Exception as e:
            print(f"Error extracting config: {e}")
            # Fallback to safe defaults
            return {
                'ts_enabled': True,
                'break_enabled': True, 
                'break_method': 'auto',
                'llm_enabled': False,
                'llm_provider': 'azure',
                'api_key': '',
                'endpoint': '',
                'deployment_name': '',
                'detail_level': 'medium',
                'rag_enabled': False
            }
    
    def _setup_llm_explainer(self, config):
        """Setup LLM explainer based on configuration."""
        try:
            if not config['llm_enabled'] or not config['api_key']:
                self.llm_explainer = None
                return
            
            # Create LLM explainer with proper configuration
            if config['llm_provider'] == 'azure':
                if config['endpoint'] and config['deployment_name']:
                    self.llm_explainer = LLMExplainer(
                        provider='azure',
                        api_key=config['api_key'],
                        azure_endpoint=config['endpoint'],
                        deployment_name=config['deployment_name']
                    )
                else:
                    self.status_indicator.object = "**Status:** ⚠️ Azure LLM requires endpoint and deployment name"
                    self.llm_explainer = None
            
            elif config['llm_provider'] == 'openai':
                self.llm_explainer = LLMExplainer(
                    provider='openai',
                    api_key=config['api_key']
                )
            
            elif config['llm_provider'] == 'anthropic':
                self.llm_explainer = LLMExplainer(
                    provider='anthropic',
                    api_key=config['api_key']
                )
                
        except Exception as e:
            print(f"Error setting up LLM explainer: {e}")
            self.llm_explainer = None
    
    def _update_results_display(self):
        """Update the results tabs with analysis results."""
        tabs_content = []
        
        # Time series results
        if self.ts_results or self.explained_ts_results:
            result = self.explained_ts_results or self.ts_results
            ts_content = pn.Column(
                pn.pane.Markdown(f"```\n{result.summary()}\n```"),
                name="Time Series Results"
            )
            tabs_content.append(("📈 Time Series Results", ts_content))
        
        # Break detection results  
        if self.break_results or self.explained_break_results:
            result = self.explained_break_results or self.break_results
            break_content = pn.Column(
                pn.pane.Markdown(f"```\n{result.summary()}\n```"),
                name="Break Detection Results"
            )
            tabs_content.append(("⚡ Break Detection Results", break_content))
        
        # Add interactive plots
        if self.ts_results or self.break_results:
            plot_content = self._create_interactive_plots()
            tabs_content.append(("📊 Plots", plot_content))
        
        # Update the results tabs
        if tabs_content:
            self.results_tabs[:] = tabs_content
    
    def _create_interactive_plots(self):
        """Create interactive plots using Panel/Matplotlib."""
        try:
            plots = []
            
            # Time series plot
            if self.ts_results:
                result = self.explained_ts_results or self.ts_results
                actual_result = result.original_result if hasattr(result, 'original_result') else result
                
                fig = actual_result.plot(figsize=(12, 6))
                plots.append(pn.pane.Matplotlib(fig, tight=True, name="Time Series Analysis"))
                plt.close(fig)
            
            # Break detection plot
            if self.break_results:
                result = self.explained_break_results or self.break_results
                actual_result = result.original_result if hasattr(result, 'original_result') else result
                
                fig = actual_result.plot(figsize=(12, 6))
                plots.append(pn.pane.Matplotlib(fig, tight=True, name="Break Detection"))
                plt.close(fig)
            
            if plots:
                return pn.Column(*plots)
            else:
                return pn.pane.Markdown("📊 No plots available")
                
        except Exception as e:
            return pn.pane.Markdown(f"❌ Error generating plots: {str(e)}")
    
    def _generate_report(self, event):
        """Generate reports based on current results."""
        if not (self.ts_results or self.break_results):
            self.status_indicator.object = "**Status:** ❌ Please run analysis first"
            return
            
        try:
            self.status_indicator.object = "**Status:** 📄 Generating report..."
            
            # Get report configuration
            formats = self.report_config[1].value  # Output formats
            title = self.report_config[2].value  # Report title
            filename = self.report_config[3].value  # Filename
            fast_pdf = self.report_config[4].value  # Fast PDF mode
            
            # Use explained results if available, otherwise raw results
            ts_result = self.explained_ts_results or self.ts_results
            break_result = self.explained_break_results or self.break_results
            
            # Generate reports
            reports = self.report_generator.generate_report(
                timeseries_result=ts_result,
                break_result=break_result,
                format=formats,
                title=title,
                filename=filename or None,
                fast_pdf=fast_pdf
            )
            
            # Create download links
            download_html = "<h4>📥 Download Reports:</h4><ul>"
            for fmt, path in reports.items():
                if not path.startswith("Error"):
                    download_html += f'<li><a href="{path}" target="_blank">{fmt.upper()} Report</a></li>'
            download_html += "</ul>"
            
            # Update the download links section
            self.report_config[5].object = download_html
            
            self.status_indicator.object = f"**Status:** ✅ Reports generated successfully!"
            
        except Exception as e:
            self.status_indicator.object = f"**Status:** ❌ Report generation error: {str(e)}"
    
    def show(self):
        """Display the widget interface."""
        return self.layout
    
    def servable(self):
        """Make the widget servable with Panel serve."""
        return self.layout.servable("Structural Break Analysis")