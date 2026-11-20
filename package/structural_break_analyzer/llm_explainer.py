"""
LLM Explainer for adding AI-powered explanations to analysis results.

This module provides the LLMExplainer class that can add explanations to:
- TimeSeriesResult: General feature explanations (no RAG needed)
- BreakDetectionResult: Contextual break explanations (with optional RAG support)

RAG (Retrieval Augmented Generation) is ONLY used for break explanations
where external context (news, reports, events) helps explain WHY breaks occurred.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
import pandas as pd
from datetime import datetime

from .llm.break_explainer import BreakExplainer
from .llm.timeseries_explainer import TimeSeriesExplainer
from .llm.providers.base import LLMProvider
from .llm.providers.azure import AzureOpenAIProvider
from .llm.providers.openai_provider import OpenAIProvider
from .llm.providers.anthropic_provider import AnthropicProvider
from .llm.providers.openrouter_provider import OpenRouterProvider
from .rag.rag_system import RAGSystem
from .analyzer import TimeSeriesResult, BreakDetectionResult


@dataclass
class ExplainedTimeSeriesResult:
    """Time series result enhanced with LLM explanations."""
    original_result: TimeSeriesResult
    explanation: str
    insights: List[str] = field(default_factory=list)
    feature_explanations: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def plot(self, **kwargs):
        """Delegate plotting to original result."""
        return self.original_result.plot(**kwargs)
    
    def summary(self) -> str:
        """Enhanced summary with explanations."""
        base_summary = self.original_result.summary()
        
        enhanced_summary = [
            base_summary,
            "",
            "=" * 50,
            "LLM EXPLANATION",
            "=" * 50,
            self.explanation
        ]
        
        if self.insights:
            enhanced_summary.extend([
                "",
                "KEY INSIGHTS",
                "-" * 30
            ])
            for i, insight in enumerate(self.insights, 1):
                enhanced_summary.append(f"{i}. {insight}")
        
        return "\n".join(enhanced_summary)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary including explanations."""
        result = self.original_result.to_dict()
        result['explanation'] = self.explanation
        result['insights'] = self.insights
        if self.feature_explanations:
            result['feature_explanations'] = self.feature_explanations
        result['explained_metadata'] = self.metadata
        return result


@dataclass
class ExplainedBreakDetectionResult:
    """Break detection result enhanced with LLM explanations (with optional RAG)."""
    original_result: BreakDetectionResult
    break_explanations: Dict[int, str] = field(default_factory=dict)  # break_index -> explanation
    overall_summary: str = ""
    rag_used: bool = False
    context_documents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def plot(self, **kwargs):
        """Delegate plotting to original result.""" 
        return self.original_result.plot(**kwargs)
    
    def summary(self) -> str:
        """Enhanced summary with break explanations."""
        base_summary = self.original_result.summary()
        
        enhanced_summary = [
            base_summary,
            "",
            "=" * 50,
            "BREAK EXPLANATIONS" + (" (with RAG context)" if self.rag_used else ""),
            "=" * 50
        ]
        
        # Add individual break explanations
        for i, break_point in enumerate(self.original_result.breaks):
            if i in self.break_explanations:
                date_str = break_point.date.strftime('%Y-%m-%d') if hasattr(break_point.date, 'strftime') else str(break_point.date)
                enhanced_summary.extend([
                    f"",
                    f"Break {i+1} ({date_str}):",
                    f"{self.break_explanations[i]}"
                ])
        
        if self.overall_summary:
            enhanced_summary.extend([
                "",
                "OVERALL ANALYSIS",
                "-" * 30,
                self.overall_summary
            ])
        
        if self.rag_used and self.context_documents:
            enhanced_summary.extend([
                "",
                "CONTEXT SOURCES",
                "-" * 30,
                f"Used {len(self.context_documents)} relevant documents for context"
            ])
        
        return "\n".join(enhanced_summary)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary including explanations."""
        result = self.original_result.to_dict()
        
        # Add explanations to individual breaks
        for i, break_dict in enumerate(result['breaks']):
            if i in self.break_explanations:
                break_dict['explanation'] = self.break_explanations[i]
        
        result['overall_summary'] = self.overall_summary
        result['rag_used'] = self.rag_used
        result['context_documents'] = self.context_documents
        result['explained_metadata'] = self.metadata
        return result


class LLMExplainer:
    """
    Add AI-powered explanations to analysis results.
    
    Features:
    - TimeSeriesResult: General feature explanations (no RAG)
    - BreakDetectionResult: Contextual explanations (with optional RAG)
    - RAG support ONLY for break explanations where context matters
    """
    
    def __init__(self, 
                 provider: Optional[str] = None,
                 **provider_params):
        """
        Initialize LLM explainer.
        
        Args:
            provider: LLM provider ('azure', 'openai', 'anthropic') or None to disable LLM
            **provider_params: Provider-specific parameters
                For Azure: azure_endpoint, api_key, deployment_name, api_version
                For OpenAI: api_key, model
                For Anthropic: api_key, model
        """
        self.provider = provider
        self.llm_provider_instance = None
        self.break_explainer = None
        self.timeseries_explainer = None
        
        if provider:
            try:
                self.llm_provider_instance = self._create_provider(provider, **provider_params)
                self.break_explainer = BreakExplainer(self.llm_provider_instance)
                self.timeseries_explainer = TimeSeriesExplainer(self.llm_provider_instance)
            except Exception as e:
                print(f"Warning: Failed to initialize LLM provider '{provider}': {e}")
                self.provider = None
        
        # RAG system (only for breaks)
        self.rag_system: Optional[RAGSystem] = None
        self.rag_enabled = False
    
    def _create_provider(self, provider: str, **params) -> Optional[LLMProvider]:
        """Create appropriate LLM provider."""
        if not provider:
            return None
            
        if provider.lower() == 'azure':
            required_params = ['azure_endpoint', 'api_key', 'deployment_name']
            for param in required_params:
                if param not in params:
                    raise ValueError(f"Azure provider requires {param}")
            
            return AzureOpenAIProvider(
                azure_endpoint=params['azure_endpoint'],
                api_key=params['api_key'],
                deployment_name=params['deployment_name'],
                api_version=params.get('api_version', '2024-10-21'),
                cache_enabled=params.get('cache_enabled', True)
            )
        
        elif provider.lower() == 'openai':
            if 'api_key' not in params:
                raise ValueError("OpenAI provider requires api_key")
            
            return OpenAIProvider(
                api_key=params['api_key'],
                model=params.get('model', 'gpt-4'),
                cache_enabled=params.get('cache_enabled', True)
            )
        
        elif provider.lower() == 'anthropic':
            if 'api_key' not in params:
                raise ValueError("Anthropic provider requires api_key")

            return AnthropicProvider(
                api_key=params['api_key'],
                model=params.get('model', 'claude-3-sonnet-20240229'),
                cache_enabled=params.get('cache_enabled', True)
            )

        elif provider.lower() == 'openrouter':
            if 'api_key' not in params:
                raise ValueError("OpenRouter provider requires api_key")

            return OpenRouterProvider(
                api_key=params['api_key'],
                model=params.get('model', 'openai/gpt-4o'),
                cache_enabled=params.get('cache_enabled', True)
            )

        else:
            available_providers = ['azure', 'openai', 'anthropic', 'openrouter']
            raise ValueError(f"Unknown provider: {provider}. Choose from: {available_providers}")
    
    def enable_rag(self, 
                   cache_dir: str = ".rag_cache",
                   embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Enable RAG system for context-aware break explanations.
        
        Args:
            cache_dir: Directory to store RAG embeddings and cache
            embedding_model: Embedding model name for RAG
        """
        try:
            self.rag_system = RAGSystem(
                embedding_model=embedding_model,
                cache_dir=cache_dir
            )
            self.rag_enabled = True
            print(f"RAG system enabled with embedding model: {embedding_model}")
        except ImportError as e:
            print(f"Warning: RAG dependencies not installed. RAG disabled. Error: {e}")
            self.rag_system = None
            self.rag_enabled = False
    
    def add_documents(self, document_paths: List[str]) -> bool:
        """
        Add documents to RAG system for break explanation context.
        
        Args:
            document_paths: List of file or directory paths to add
            
        Returns:
            True if successful, False otherwise
        """
        if not self.rag_enabled or not self.rag_system:
            print("RAG system not enabled. Call enable_rag() first.")
            return False
        
        try:
            self.rag_system.add_documents(document_paths)
            print(f"Added {len(document_paths)} document paths to RAG system")
            return True
        except Exception as e:
            print(f"Error adding documents to RAG: {e}")
            return False
    
    def explain_timeseries(self, 
                          result: TimeSeriesResult,
                          data_description: str = "time series",
                          detail_level: str = "medium") -> ExplainedTimeSeriesResult:
        """
        Add LLM explanations to time series analysis (NO RAG - features are self-explanatory).
        
        Args:
            result: TimeSeriesResult to explain
            data_description: Description of what the data represents
            detail_level: 'basic', 'medium', or 'detailed'
            
        Returns:
            ExplainedTimeSeriesResult with AI explanations
        """
        if detail_level not in ['basic', 'medium', 'detailed']:
            raise ValueError(f"Invalid detail_level '{detail_level}'. Must be 'basic', 'medium', or 'detailed'.")
            
        try:
            # Generate comprehensive explanation
            explanation = self.timeseries_explainer.explain_features(
                result.features,
                data_description,
                detail_level=detail_level
            )
            
            # Generate insights
            insights = self.timeseries_explainer.generate_insights(
                result.features, 
                data_description
            )
            
            # Generate feature-specific explanations for detailed analysis
            feature_explanations = None
            if detail_level == 'detailed':
                feature_explanations = {}
                feature_types = ['trend', 'seasonality', 'volatility', 'stationarity', 'distribution']
                
                for feature_name in feature_types:
                    try:
                        feat_explanation = self.timeseries_explainer.explain_specific_feature(
                            result.features, feature_name, data_description
                        )
                        feature_explanations[feature_name] = feat_explanation
                    except Exception as e:
                        feature_explanations[feature_name] = f"Error explaining {feature_name}: {str(e)}"
            
            return ExplainedTimeSeriesResult(
                original_result=result,
                explanation=explanation,
                insights=insights,
                feature_explanations=feature_explanations,
                metadata={
                    'llm_provider': self.provider,
                    'detail_level': detail_level,
                    'rag_used': False  # Never used for time series
                }
            )
            
        except Exception as e:
            # Fallback with error message
            return ExplainedTimeSeriesResult(
                original_result=result,
                explanation=f"Error generating explanation: {str(e)}",
                insights=[f"LLM explanation failed: {str(e)}"],
                metadata={'error': str(e)}
            )
    
    def explain_breaks(self,
                      result: BreakDetectionResult,
                      data_description: str = "time series",
                      use_rag: bool = None) -> ExplainedBreakDetectionResult:
        """
        Add LLM explanations to break detection (WITH optional RAG for context).
        
        Args:
            result: BreakDetectionResult to explain
            data_description: Description of what the data represents
            use_rag: Whether to use RAG context (defaults to True if RAG enabled)
            
        Returns:
            ExplainedBreakDetectionResult with AI explanations and optional context
        """
        if use_rag is None:
            use_rag = self.rag_enabled
        
        if use_rag and not self.rag_enabled:
            print("Warning: RAG requested but not enabled. Using standard explanations.")
            use_rag = False
        
        break_explanations = {}
        context_documents = []
        
        # Explain each break
        for i, break_point in enumerate(result.breaks):
            try:
                if use_rag and self.rag_system:
                    # Get contextual documents for this break
                    document_context = self.rag_system.get_context_for_break(
                        break_date=break_point.date,
                        description=data_description,
                        window_days=30,
                        top_k=3
                    )
                    
                    # Check if we got useful context
                    if document_context and "No relevant documents" not in document_context:
                        # RAG-enhanced explanation
                        explanation = self.break_explainer.explain_break_with_documents(
                            break_point=break_point,
                            data=result.data,
                            document_context=document_context,
                            value_col=result.value_col
                        )
                        
                        # Extract source names from context string for tracking
                        import re
                        sources = re.findall(r'Document \d+: ([^\n]+)', document_context)
                        context_documents.extend(sources)
                    else:
                        # Fall back to standard explanation if no RAG context
                        explanation = self.break_explainer.explain_break(
                            break_point, 
                            result.data, 
                            result.value_col, 
                            result.date_col, 
                            data_description
                        )
                else:
                    # Standard explanation without RAG
                    explanation = self.break_explainer.explain_break(
                        break_point, 
                        result.data, 
                        result.value_col, 
                        result.date_col, 
                        data_description
                    )
                
                break_explanations[i] = explanation
                
            except Exception as e:
                break_explanations[i] = f"Error generating explanation: {str(e)}"
        
        # Generate overall summary if multiple breaks
        overall_summary = ""
        if len(result.breaks) > 1:
            try:
                overall_summary = self.break_explainer.generate_overall_summary(
                    result.breaks, result.data, data_description
                )
            except Exception as e:
                overall_summary = f"Error generating overall summary: {str(e)}"
        
        return ExplainedBreakDetectionResult(
            original_result=result,
            break_explanations=break_explanations,
            overall_summary=overall_summary,
            rag_used=use_rag,
            context_documents=list(set(context_documents)),  # Remove duplicates
            metadata={
                'llm_provider': self.provider,
                'rag_enabled': self.rag_enabled,
                'rag_used': use_rag,
                'n_context_docs': len(set(context_documents)) if use_rag else 0
            }
        )
    
    # RAG management methods (only affect break explanations)
    def clear_rag_documents(self):
        """Clear all documents from RAG system."""
        if self.rag_system:
            self.rag_system.clear()
        else:
            print("RAG system not enabled.")
    
    def clear_rag_embedding_cache(self):
        """Clear only the embedding cache (keeps documents)."""
        if self.rag_system:
            self.rag_system.clear_embedding_cache()
        else:
            print("RAG system not enabled.")
    
    def get_rag_stats(self) -> Optional[Dict]:
        """Get RAG system statistics."""
        if self.rag_system:
            try:
                return self.rag_system.get_stats()
            except Exception as e:
                print(f"Warning: Could not get RAG stats: {e}")
                return {
                    'vector_store': {'total_documents': 0, 'collection_name': 'unknown'},
                    'embedder': {'model': 'unknown', 'dimension': 0, 'cache_size': 0}
                }
        return None
    
    def delete_rag_documents(self, document_ids: List[str]) -> bool:
        """Delete specific documents from RAG system by IDs."""
        if self.rag_system:
            return self.rag_system.delete_documents(document_ids)
        return False
    
    def delete_rag_documents_by_date(self, start_date: datetime, end_date: datetime) -> bool:
        """Delete RAG documents within a date range."""
        if self.rag_system:
            return self.rag_system.delete_documents_by_date_range(start_date, end_date)
        return False
    
    def cleanup_all_rag_files(self, cache_dir: str = None):
        """
        Complete cleanup - removes all RAG data and cache directories.
        
        Args:
            cache_dir: Specific cache directory to remove, or None to use system default
        """
        import shutil
        from pathlib import Path
        
        # Clear RAG system data first
        if self.rag_system:
            try:
                self.rag_system.clear()
                print("✅ RAG documents and embeddings cleared")
            except Exception as e:
                print(f"⚠️  Error clearing RAG system: {e}")
        
        # Remove cache directories
        cache_dirs_to_check = []
        
        if cache_dir:
            cache_dirs_to_check.append(Path(cache_dir))
        else:
            # Check common cache directory names
            cache_dirs_to_check.extend([
                Path(".rag_cache"),
                Path(".demo_rag_cache"), 
                Path(".example_rag_cache"),
                Path("rag_cache")
            ])
        
        for cache_path in cache_dirs_to_check:
            if cache_path.exists():
                try:
                    shutil.rmtree(cache_path)
                    print(f"🗂️  Removed cache directory: {cache_path}")
                except Exception as e:
                    print(f"⚠️  Could not remove {cache_path}: {e}")
        
        print("🧹 Complete RAG cleanup finished")