from typing import Dict, List
import json

# ================================================================================
# BREAK DETECTION PROMPTS
# ================================================================================

def get_break_explanation_prompt(
        break_date: str,
        confidence: float,
        magnitude: float,
        direction: str,
        before_stats: Dict,
        after_stats: Dict,
        data_description: str = "time series data",
        additional_context: str = "") -> tuple[str, str]:
    """
    Generate prompt for explaining a single structural break.

    Returns:
        (system_prompt, user_prompt) tuple
    """

    system_prompt = """You are a data analyst expert in time series analysis. 
Your task is to explain structural breaks - significant, persistent changes in time series data.

Provide clear, concise explanations that:
1. Describe what changed (magnitude and direction)
2. Suggest possible causes based on the timing and statistical evidence
3. Think of possible external events near the break date (e.g. macro, policy, compan
y news..), flagging speculation if unsure, and providing dates of these events if included
4. Assess the significance of the change
5. Avoid speculation beyond what the data supports

Be specific and professional."""

    user_prompt = f"""Analyze this structural break in {data_description}:

**Break Details:**
- Date: {break_date}
- Confidence: {confidence:.1%}
- Magnitude: {magnitude:.2f} ({direction} shift)

**Before Break (30-day window):**
- Mean: {before_stats.get('mean', 0):.2f}
- Std Dev: {before_stats.get('std', 0):.2f}
- Trend: {before_stats.get('trend', 'stable')}

**After Break (30-day window):**
- Mean: {after_stats.get('mean', 0):.2f}
- Std Dev: {after_stats.get('std', 0):.2f}
- Trend: {after_stats.get('trend', 'stable')}

{f'Additional Context: {additional_context}' if additional_context else ''}

Provide a brief explanation of this structural break, focusing on what changed and potential reasons."""

    return system_prompt, user_prompt


def get_break_explanation_with_context(
        break_info: Dict,
        statistical_context: Dict,
        document_context: str) -> tuple[str, str]:
    """
    Generate prompt for explaining a break with additional document context.

    Returns:
        (system_prompt, user_prompt) tuple
    """

    system_prompt = """You are a data analyst expert in time series analysis.
You have access to relevant documents that may explain the structural break.

When explaining:
1. Connect the statistical evidence to events in the documents
2. Be specific about which information supports your explanation
3. Distinguish between correlation and likely causation
4. Keep explanations concise and actionable"""

    user_prompt = f"""Analyze this structural break with additional context:

**Break Information:**
{json.dumps(break_info, indent=2)}

**Statistical Context:**
{json.dumps(statistical_context, indent=2)}

**Relevant Documents:**
{document_context}

Explain this break using both the statistical evidence and document context. Be specific about how the documents relate to the observed change."""

    return system_prompt, user_prompt


# ================================================================================
# TIME SERIES EXPLANATION PROMPTS
# ================================================================================


def get_timeseries_explanation_prompt(
        features_summary: Dict,
        insights: List[str],
        data_description: str = "time series",
        context: str = "",
        detail_level: str = "medium") -> tuple[str, str]:
    """
    Generate prompt for comprehensive time series analysis explanation.
    
    Args:
        features_summary: Dictionary containing all extracted features
        insights: List of key insights generated from the analysis
        data_description: Description of what the data represents
        context: Additional context about the data
        detail_level: 'basic', 'medium', or 'detailed'
        
    Returns:
        (system_prompt, user_prompt) tuple
    """
    
    system_prompt = """You are a time series analysis expert. 
Your task is to explain time series characteristics in clear, accessible language.

Focus on:
- Key patterns and behaviors in the data
- What these patterns suggest about the underlying process
- Implications for modeling and forecasting
- Any notable characteristics or concerns

Keep explanations professional and data-driven."""
    
    # Adjust instructions based on detail level
    if detail_level == "basic":
        detail_instruction = "Provide a high-level summary in 2-3 sentences. Focus only on the most important patterns."
    elif detail_level == "detailed": 
        detail_instruction = "Provide an in-depth analysis with technical details, statistical significance, and actionable insights."
    else:  # medium
        detail_instruction = "Provide a comprehensive but concise explanation. Cover main patterns and their implications."
    
    system_prompt += f"\n\n{detail_instruction}"
    
    # Build user prompt based on detail level
    if detail_level == "basic":
        user_prompt = f"""Analyze this {data_description} briefly:

Key characteristics: {json.dumps(features_summary, indent=2)}

Key insights: {'; '.join(insights[:3])}

Context: {context if context else 'None provided'}

Provide a 2-3 sentence summary of the most important patterns and what they mean."""

    elif detail_level == "detailed":
        user_prompt = f"""Perform a detailed analysis of this {data_description}:

Complete feature analysis: {json.dumps(features_summary, indent=2)}

Key insights: {'; '.join(insights)}

Additional context: {context if context else 'None provided'}

Provide a comprehensive analysis covering:
1. Statistical properties and distribution characteristics
2. Trend analysis and implications  
3. Seasonal patterns and their significance
4. Volatility behavior and clustering
5. Stationarity properties and modeling implications
6. Memory characteristics and dependencies
7. Complexity and predictability assessment
8. Domain-specific insights and recommendations

Be specific about statistical significance and provide actionable insights."""

    else:  # medium
        user_prompt = f"""Analyze this {data_description}:

Time series characteristics: {json.dumps(features_summary, indent=2)}

Key insights: {'; '.join(insights)}

Additional context: {context if context else 'None provided'}

Provide a comprehensive but concise explanation covering:
1. Main characteristics and patterns
2. What these patterns suggest about the underlying process
3. Implications for modeling and forecasting  
4. Any concerns or notable features
5. Domain-specific insights

Keep the explanation under 300 words and focus on actionable insights."""

    return system_prompt, user_prompt


def get_specific_feature_prompt(
        feature_name: str,
        feature_data: Dict,
        data_description: str = "time series",
        context: str = "") -> tuple[str, str]:
    """
    Generate prompt for explaining a specific time series feature.
    
    Args:
        feature_name: Name of the feature to explain (trend, seasonality, etc.)
        feature_data: Dictionary containing relevant data for this feature
        data_description: Description of what the data represents  
        context: Additional context about the data
        
    Returns:
        (system_prompt, user_prompt) tuple
    """
    
    system_prompt = f"You are a time series expert. Explain {feature_name} characteristics clearly and concisely."
    
    # Feature-specific prompts
    if feature_name == "trend":
        user_prompt = f"""Explain this trend pattern in {data_description}:

Trend characteristics:
- Strength (R²): {feature_data.get('strength', 0):.3f}
- Slope: {feature_data.get('slope', 0):.6f}
- Direction: {feature_data.get('direction', 'stable')}
- Acceleration: {feature_data.get('acceleration', 0):.6f}

Context: {context}

What does this trend pattern mean for this type of data? Keep it under 150 words."""

    elif feature_name == "seasonality":
        if not feature_data.get('has_seasonality', False):
            user_prompt = f"No significant seasonality detected in the {data_description}."
        else:
            multiple_info = ""
            if len(feature_data.get('all_periods', [])) > 1:
                periods_str = ', '.join(map(str, feature_data.get('all_periods', [])[:3]))
                multiple_info = f" Multiple seasonal periods detected: {periods_str}."
                
            user_prompt = f"""Explain this seasonal pattern in {data_description}:

Seasonality characteristics:
- Primary period: {feature_data.get('primary_period', 'None')} observations
- Strength: {feature_data.get('strength', 0):.3f}
- Multiple periods: {len(feature_data.get('all_periods', [])) > 1}
- All periods: {feature_data.get('all_periods', [])}

Context: {context}

What does this seasonal pattern suggest?{multiple_info} Keep it under 150 words."""

    elif feature_name == "volatility":
        vol_level = _classify_volatility(feature_data.get('mean', 0))
        
        clustering_info = ""
        if feature_data.get('clustering', 0) > 0.2:
            clustering_info = f" Strong volatility clustering detected (correlation: {feature_data.get('clustering', 0):.2f})."
        
        persistence_info = ""
        if feature_data.get('persistence', 0) > 0.3:
            persistence_info = f" Volatility shows persistence (score: {feature_data.get('persistence', 0):.2f})."
        
        user_prompt = f"""Explain this volatility pattern in {data_description}:

Volatility characteristics:
- Level: {vol_level} ({feature_data.get('mean', 0):.4f})
- Variability: {feature_data.get('std', 0):.4f}
- Clustering: {feature_data.get('clustering', 0):.3f}
- Persistence: {feature_data.get('persistence', 0):.3f}

Context: {context}

What does this mean for risk and predictability?{clustering_info}{persistence_info} Keep it under 150 words."""

    elif feature_name == "stationarity":
        if feature_data.get('is_stationary', False):
            user_prompt = f"The {data_description} is stationary (ADF p-value: {feature_data.get('adf_pvalue', 0):.3f}), meaning its statistical properties are consistent over time. This is ideal for many forecasting methods and statistical models."
        else:
            diff_info = f" The series requires {feature_data.get('differences_needed', 0)} difference(s) to achieve stationarity." if feature_data.get('differences_needed', 0) > 0 else ""
            user_prompt = f"The {data_description} is non-stationary (ADF p-value: {feature_data.get('adf_pvalue', 0):.3f}), meaning its statistical properties change over time.{diff_info} Consider differencing, detrending, or using methods designed for non-stationary data."

    elif feature_name == "distribution":
        skewness = feature_data.get('skewness', 0)
        kurtosis = feature_data.get('kurtosis', 0)
        
        # Interpret skewness
        if abs(skewness) < 0.5:
            skew_desc = "approximately symmetric"
        elif skewness > 0:
            skew_desc = f"right-skewed (skewness: {skewness:.2f})"
        else:
            skew_desc = f"left-skewed (skewness: {skewness:.2f})"
            
        # Interpret kurtosis  
        if abs(kurtosis) < 1:
            kurt_desc = "normal tail weight"
        elif kurtosis > 0:
            kurt_desc = f"heavy tails (excess kurtosis: {kurtosis:.2f})"
        else:
            kurt_desc = f"light tails (excess kurtosis: {kurtosis:.2f})"
            
        outlier_info = "outliers are common" if kurtosis > 1 else "relatively few extreme values"
        
        user_prompt = f"The {data_description} distribution is {skew_desc} with {kurt_desc}. This suggests {outlier_info}, which has implications for risk assessment and model selection."

    elif feature_name == "autocorrelation":
        sig_lags = feature_data.get('significant_lags', [])
        if len(sig_lags) == 0:
            user_prompt = f"The {data_description} shows no significant autocorrelation, suggesting observations are independent. This is good for many statistical models but may indicate low predictability."
        else:
            ljung_info = ""
            ljung_pvalue = feature_data.get('ljung_box_pvalue', 1.0)
            if ljung_pvalue < 0.05:
                ljung_info = f" The Ljung-Box test confirms significant serial correlation (p={ljung_pvalue:.3f})."
            
            acf_1 = feature_data.get('acf_1', 0)
            user_prompt = f"The {data_description} shows significant autocorrelation at {len(sig_lags)} lag(s): {sig_lags[:5]}. First lag correlation is {acf_1:.2f}.{ljung_info} This suggests predictable patterns that could be leveraged for forecasting."

    elif feature_name == "complexity":
        hurst = feature_data.get('hurst_exponent', 0.5)
        entropy = feature_data.get('entropy', 1.0)
        
        memory_type = _classify_memory(hurst)
        regularity = 'highly regular' if entropy < 0.5 else 'moderately regular' if entropy < 1.0 else 'highly irregular'
        
        user_prompt = f"The {data_description} shows {memory_type} behavior (Hurst exponent: {hurst:.2f}) and is {regularity} (entropy: {entropy:.2f}). This affects long-term predictability and the choice of forecasting methods."

    else:
        user_prompt = f"Unknown feature: {feature_name}. Available features: trend, seasonality, volatility, stationarity, distribution, autocorrelation, complexity."

    return system_prompt, user_prompt


def _classify_volatility(vol_level: float) -> str:
    """Classify volatility level."""
    if vol_level < 0.05:
        return 'low'
    elif vol_level < 0.15:
        return 'moderate'  
    elif vol_level < 0.3:
        return 'high'
    else:
        return 'very high'


def get_overall_summary_prompt(
        breaks_context: Dict,
        data_description: str = "time series") -> tuple[str, str]:
    """
    Generate prompt for overall summary when multiple breaks are detected.
    
    Args:
        breaks_context: Dictionary containing break pattern analysis
        data_description: Description of what the data represents
        
    Returns:
        (system_prompt, user_prompt) tuple
    """
    
    system_prompt = """You are a data analyst expert in time series analysis.
Your task is to provide concise overall analysis of structural break patterns.

Focus on:
1. The overall trend or pattern of breaks
2. Market/economic implications if applicable  
3. What this might indicate about underlying factors

Keep responses professional, analytical, and actionable (2-3 sentences max)."""

    user_prompt = f"""Analyze this pattern of structural breaks in {data_description}:

Break Summary:
- Total breaks detected: {breaks_context['n_breaks']}
- Upward breaks: {breaks_context['up_breaks']}
- Downward breaks: {breaks_context['down_breaks']}
- Average confidence: {breaks_context['avg_confidence']:.1%}
- Average magnitude: {breaks_context['avg_magnitude']:.3f}
- Time period: {breaks_context['time_span']}

Pattern type: {breaks_context['break_pattern']}

Please provide a concise overall analysis (2-3 sentences) of what this pattern of structural breaks suggests about the {data_description}. Focus on:
1. The overall trend or pattern
2. Market/economic implications if applicable
3. What this might indicate about underlying factors

Keep the response professional and analytical."""

    return system_prompt, user_prompt


def _classify_memory(hurst: float) -> str:
    """Classify memory type based on Hurst exponent."""
    if hurst < 0.4:
        return 'anti-persistent (mean-reverting)'
    elif hurst > 0.6:
        return 'persistent (trending)'
    else:
        return 'random walk (no memory)'