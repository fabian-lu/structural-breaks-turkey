from typing import List, Dict, Optional
from anthropic import Anthropic
from .base import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self,
                 api_key: str,
                 model: str = "claude-3-haiku-20240307",
                 base_url: Optional[str] = None,
                 cache_enabled: bool = True,
                 rate_limit_per_minute: Optional[int] = None,
                 max_retries: int = 3,
                 adaptive_rate_limiting: bool = False):
        """Initialize Anthropic provider."""
        super().__init__(
            cache_enabled=cache_enabled,
            rate_limit_per_minute=rate_limit_per_minute,
            max_retries=max_retries,
            adaptive_rate_limiting=adaptive_rate_limiting
        )
        
        self.client = Anthropic(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def _call_api(self,
                  messages: List[Dict[str, str]],
                  temperature: float = 0.3,
                  max_tokens: Optional[int] = None) -> LLMResponse:
        """Make API call to Anthropic Claude."""
        try:
            # Extract system message if present
            system_message = None
            user_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)

            # Anthropic API requires max_tokens, default to 4096 if not set
            response = self.client.messages.create(
                model=self.model,
                messages=user_messages,
                system=system_message if system_message else None,
                temperature=temperature,
                max_tokens=max_tokens or 4096
            )
            
            content = response.content[0].text if response.content else ""
            
            usage = None
            if response.usage:
                usage = {
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
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