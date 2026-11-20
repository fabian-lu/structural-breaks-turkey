from typing import List, Dict, Optional
import google.generativeai as genai
from .base import LLMProvider, LLMResponse


class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation."""

    def __init__(self,
                 api_key: str,
                 model: str = "gemini-1.5-flash",
                 cache_enabled: bool = True,
                 rate_limit_per_minute: Optional[int] = None,
                 max_retries: int = 3,
                 adaptive_rate_limiting: bool = False):
        """Initialize Gemini provider."""
        super().__init__(
            cache_enabled=cache_enabled,
            rate_limit_per_minute=rate_limit_per_minute,
            max_retries=max_retries,
            adaptive_rate_limiting=adaptive_rate_limiting
        )
        
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    def _call_api(self,
                  messages: List[Dict[str, str]],
                  temperature: float = 0.3,
                  max_tokens: Optional[int] = None) -> LLMResponse:
        """Make API call to Google Gemini."""
        try:
            # Convert messages to Gemini format
            # Gemini expects a simple prompt for basic usage
            prompt_parts = []

            for msg in messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"System: {msg['content']}\n\n")
                elif msg["role"] == "user":
                    prompt_parts.append(f"User: {msg['content']}\n\n")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"Assistant: {msg['content']}\n\n")

            prompt = "".join(prompt_parts)

            # Configure generation settings
            config_kwargs = dict(temperature=temperature)
            if max_tokens is not None:
                config_kwargs["max_output_tokens"] = max_tokens
            generation_config = genai.types.GenerationConfig(**config_kwargs)
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            content = response.text if response.text else ""
            
            # Gemini doesn't provide token usage in the same way
            # We can estimate or leave it as None
            usage = None
            if hasattr(response, 'usage_metadata'):
                usage = {
                    'prompt_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
                    'completion_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0),
                    'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            return LLMResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                cached=False,
                error=None
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model_name,
                error=str(e)
            )