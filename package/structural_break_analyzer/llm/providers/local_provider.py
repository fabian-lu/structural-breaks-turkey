from typing import List, Dict, Optional
import requests
import json
from .base import LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    """Ollama provider for local models."""

    def __init__(self,
                 model: str = "llama2",
                 base_url: str = "http://localhost:11434",
                 cache_enabled: bool = True,
                 rate_limit_per_minute: Optional[int] = None,
                 max_retries: int = 3,
                 adaptive_rate_limiting: bool = False):
        """Initialize Ollama provider."""
        super().__init__(
            cache_enabled=cache_enabled,
            rate_limit_per_minute=rate_limit_per_minute,
            max_retries=max_retries,
            adaptive_rate_limiting=adaptive_rate_limiting
        )
        
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_endpoint = f"{self.base_url}/api/generate"
        self.chat_endpoint = f"{self.base_url}/api/chat"

    def _call_api(self,
                  messages: List[Dict[str, str]],
                  temperature: float = 0.3,
                  max_tokens: int = 500) -> LLMResponse:
        """Make API call to Ollama."""
        try:
            # Use chat endpoint for conversation
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            content = data.get("message", {}).get("content", "")
            
            # Ollama provides different metrics
            usage = None
            if "eval_count" in data and "prompt_eval_count" in data:
                usage = {
                    'prompt_tokens': data.get("prompt_eval_count", 0),
                    'completion_tokens': data.get("eval_count", 0),
                    'total_tokens': data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
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


class LMStudioProvider(LLMProvider):
    """LM Studio provider for local models (OpenAI-compatible)."""

    def __init__(self,
                 model: str = "local-model",
                 base_url: str = "http://localhost:1234/v1",
                 cache_enabled: bool = True,
                 rate_limit_per_minute: Optional[int] = None,
                 max_retries: int = 3,
                 adaptive_rate_limiting: bool = False):
        """Initialize LM Studio provider."""
        super().__init__(
            cache_enabled=cache_enabled,
            rate_limit_per_minute=rate_limit_per_minute,
            max_retries=max_retries,
            adaptive_rate_limiting=adaptive_rate_limiting
        )
        
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_endpoint = f"{self.base_url}/chat/completions"

    def _call_api(self,
                  messages: List[Dict[str, str]],
                  temperature: float = 0.3,
                  max_tokens: int = 500) -> LLMResponse:
        """Make API call to LM Studio (OpenAI-compatible)."""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            usage = None
            if "usage" in data:
                usage = {
                    'prompt_tokens': data["usage"].get("prompt_tokens", 0),
                    'completion_tokens': data["usage"].get("completion_tokens", 0),
                    'total_tokens': data["usage"].get("total_tokens", 0)
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


class MistralLocalProvider(LLMProvider):
    """Mistral provider for locally hosted Mistral models."""

    def __init__(self,
                 model: str = "mistral-7b",
                 base_url: str = "http://localhost:8000",
                 cache_enabled: bool = True,
                 rate_limit_per_minute: Optional[int] = None,
                 max_retries: int = 3,
                 adaptive_rate_limiting: bool = False):
        """Initialize Mistral local provider."""
        super().__init__(
            cache_enabled=cache_enabled,
            rate_limit_per_minute=rate_limit_per_minute,
            max_retries=max_retries,
            adaptive_rate_limiting=adaptive_rate_limiting
        )
        
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_endpoint = f"{self.base_url}/v1/chat/completions"

    def _call_api(self,
                  messages: List[Dict[str, str]],
                  temperature: float = 0.3,
                  max_tokens: int = 500) -> LLMResponse:
        """Make API call to locally hosted Mistral."""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            usage = None
            if "usage" in data:
                usage = {
                    'prompt_tokens': data["usage"].get("prompt_tokens", 0),
                    'completion_tokens': data["usage"].get("completion_tokens", 0),
                    'total_tokens': data["usage"].get("total_tokens", 0)
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