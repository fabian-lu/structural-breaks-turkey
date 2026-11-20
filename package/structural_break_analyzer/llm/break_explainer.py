from typing import Dict, Optional
import numpy as np
import pandas as pd

from .prompts import get_break_explanation_prompt, get_break_explanation_with_context, get_overall_summary_prompt
from .providers.base import LLMProvider, LLMResponse
from ..detection.base import BreakPoint


class BreakExplainer:
    """Generates natural language explanations for individual structural breaks."""

    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize the explainer.

        Args:
            llm_provider: LLM provider instance
        """
        self.llm = llm_provider

    def explain_break(self,
                      break_point: BreakPoint,
                      data: pd.DataFrame,
                      value_col: str = 'value',
                      date_col: str = 'date',
                      data_description: str = "time series data",
                      additional_context: str = "",
                      window_size: int = 30) -> str:
        """
        Generate explanation for a single break.

        Args:
            break_point: The break to explain
            data: Original time series data
            value_col: Name of value column
            date_col: Name of date column
            data_description: What the data represents
            additional_context: Any additional context
            window_size: Days before/after break to analyze

        Returns:
            String explanation of the break
        """
        # Calculate statistics before and after break
        before_stats = self._calculate_stats(
            data, value_col, break_point.index,
            window_size, before=True
        )
        after_stats = self._calculate_stats(
            data, value_col, break_point.index,
            window_size, before=False
        )

        # Generate prompt
        system_prompt, user_prompt = get_break_explanation_prompt(
            break_date=break_point.date.strftime('%Y-%m-%d'),
            confidence=break_point.confidence,
            magnitude=break_point.magnitude,
            direction=break_point.direction,
            before_stats=before_stats,
            after_stats=after_stats,
            data_description=data_description,
            additional_context=additional_context
        )

        # Get explanation from LLM
        response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
        )

        if response.error:
            return f"Unable to generate explanation: {response.error}"

        return response.content.strip()

    def explain_break_with_documents(self,
                                     break_point: BreakPoint,
                                     data: pd.DataFrame,
                                     document_context: str,
                                     value_col: str = 'value',
                                     window_size: int = 30) -> str:
        """
        Generate explanation with document context (for future RAG integration).

        Args:
            break_point: The break to explain
            data: Original time series data
            document_context: Relevant documents as string
            value_col: Name of value column
            window_size: Days before/after break to analyze

        Returns:
            String explanation of the break
        """
        # Calculate statistics
        before_stats = self._calculate_stats(
            data, value_col, break_point.index, window_size, before=True
        )
        after_stats = self._calculate_stats(
            data, value_col, break_point.index, window_size, before=False
        )

        # Prepare break info
        break_info = {
            'date': break_point.date.strftime('%Y-%m-%d'),
            'confidence': break_point.confidence,
            'magnitude': break_point.magnitude,
            'direction': break_point.direction,
            'method': break_point.method
        }

        statistical_context = {
            'before': before_stats,
            'after': after_stats
        }

        # Generate prompt
        system_prompt, user_prompt = get_break_explanation_with_context(
            break_info=break_info,
            statistical_context=statistical_context,
            document_context=document_context
        )

        # Get explanation
        response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
        )

        if response.error:
            return f"Unable to generate explanation: {response.error}"

        return response.content.strip()

    def _calculate_stats(self,
                         data: pd.DataFrame,
                         value_col: str,
                         break_index: int,
                         window_size: int,
                         before: bool = True) -> Dict:
        """Calculate statistics for a window before or after break."""

        if before:
            start_idx = max(0, break_index - window_size)
            end_idx = break_index
        else:
            start_idx = break_index
            end_idx = min(len(data), break_index + window_size)

        window_data = data[value_col].iloc[start_idx:end_idx].values

        if len(window_data) == 0:
            return {'mean': 0, 'std': 0, 'trend': 'insufficient data'}

        # Calculate basic stats
        stats = {
            'mean': float(np.mean(window_data)),
            'std': float(np.std(window_data)),
            'trend': self._detect_trend(window_data)
        }

        return stats

    def _detect_trend(self, data: np.ndarray) -> str:
        """Simple trend detection."""
        if len(data) < 3:
            return 'insufficient data'

        # Fit linear trend
        x = np.arange(len(data))
        coef = np.polyfit(x, data, 1)[0]

        # Determine trend based on slope
        if abs(coef) < 0.01 * np.std(data):
            return 'stable'
        elif coef > 0:
            return 'increasing'
        else:
            return 'decreasing'

    def generate_overall_summary(self, 
                               breaks: list, 
                               data: pd.DataFrame,
                               data_description: str = "time series") -> str:
        """
        Generate an overall summary when multiple breaks are detected.
        
        Args:
            breaks: List of BreakPoint objects
            data: Original time series data
            data_description: Description of what the data represents
            
        Returns:
            Overall summary string explaining the pattern of breaks
        """
        if len(breaks) == 0:
            return "No structural breaks detected."
        
        if len(breaks) == 1:
            return f"A single structural break was detected in the {data_description}."
        
        # Analyze break patterns
        break_dates = [bp.date for bp in breaks]
        directions = [bp.direction for bp in breaks]
        magnitudes = [bp.magnitude for bp in breaks]
        confidences = [bp.confidence for bp in breaks]
        
        # Count directions
        up_count = sum(1 for d in directions if d in ['up', 'increase'])
        down_count = sum(1 for d in directions if d in ['down', 'decrease'])
        
        # Calculate average metrics
        avg_confidence = np.mean(confidences)
        avg_magnitude = np.mean(np.abs(magnitudes))
        
        # Time span analysis
        first_date = min(break_dates)
        last_date = max(break_dates)
        
        # Create context for LLM
        context = {
            'n_breaks': len(breaks),
            'up_breaks': up_count,
            'down_breaks': down_count,
            'avg_confidence': avg_confidence,
            'avg_magnitude': avg_magnitude,
            'time_span': f"{first_date} to {last_date}",
            'data_description': data_description,
            'break_pattern': 'volatile' if up_count > 0 and down_count > 0 else 'trending'
        }
        
        # Generate prompt using separated prompts module
        system_prompt, user_prompt = get_overall_summary_prompt(context, data_description)
        
        try:
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
            )
            
            if response.error:
                # Fall through to rule-based fallback
                raise Exception(f"LLM error: {response.error}")
                
            return response.content.strip()
        except Exception as e:
            # Fallback to rule-based summary if LLM fails
            if context['break_pattern'] == 'volatile':
                return f"The {data_description} shows a volatile pattern with {context['n_breaks']} structural breaks ({context['up_breaks']} upward, {context['down_breaks']} downward) between {first_date.strftime('%Y-%m-%d') if hasattr(first_date, 'strftime') else first_date} and {last_date.strftime('%Y-%m-%d') if hasattr(last_date, 'strftime') else last_date}. This suggests periods of significant regime changes with mixed directional shifts."
            else:
                direction = "upward" if up_count > down_count else "downward"
                return f"The {data_description} demonstrates a {direction} trending pattern with {context['n_breaks']} structural breaks over the period from {first_date.strftime('%Y-%m-%d') if hasattr(first_date, 'strftime') else first_date} to {last_date.strftime('%Y-%m-%d') if hasattr(last_date, 'strftime') else last_date}. This indicates sustained directional momentum with clear regime shifts."