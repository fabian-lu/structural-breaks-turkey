from typing import List, Dict, Optional
from openai import OpenAI
from .base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""

    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4o-mini",
                 organization: Optional[str] = None,
                 base_url: Optional[str] = None,
                 cache_enabled: bool = True,
                 rate_limit_per_minute: Optional[int] = None,
                 max_retries: int = 3,
                 adaptive_rate_limiting: bool = False):
        """Initialize OpenAI provider."""
        super().__init__(
            cache_enabled=cache_enabled,
            rate_limit_per_minute=rate_limit_per_minute,
            max_retries=max_retries,
            adaptive_rate_limiting=adaptive_rate_limiting
        )
        
        self.client = OpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url
        )
        self.model = model

    def _call_api(self,
                  messages: List[Dict[str, str]],
                  temperature: float = 0.3,
                  max_tokens: Optional[int] = None) -> LLMResponse:
        """Make API call to OpenAI."""
        try:
            kwargs = dict(model=self.model, messages=messages, temperature=temperature)
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            response = self.client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content
            
            usage = None
            if response.usage:
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                cached=False,
                error=None
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                error=str(e)
            )