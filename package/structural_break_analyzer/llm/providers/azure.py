from typing import List, Dict, Optional
from openai import AzureOpenAI
from .base import LLMProvider, LLMResponse


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI provider implementation."""

    def __init__(self,
                 azure_endpoint: str,
                 api_key: str,
                 deployment_name: str,
                 api_version: str = "2024-10-21",
                 cache_enabled: bool = True,
                 rate_limit_per_minute: Optional[int] = None,
                 max_retries: int = 3,
                 adaptive_rate_limiting: bool = False):
        """Initialize Azure OpenAI provider."""
        super().__init__(
            cache_enabled=cache_enabled,
            rate_limit_per_minute=rate_limit_per_minute,
            max_retries=max_retries,
            adaptive_rate_limiting=adaptive_rate_limiting
        )

        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name

    def _call_api(self,
                  messages: List[Dict[str, str]],
                  temperature: float = 0.3,
                  max_tokens: Optional[int] = None) -> LLMResponse:
        """Make API call to Azure OpenAI."""
        try:
            kwargs = dict(model=self.deployment_name, messages=messages, temperature=temperature)
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
                model=self.deployment_name,
                usage=usage,
                cached=False,
                error=None
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=self.deployment_name,
                error=str(e)
            )