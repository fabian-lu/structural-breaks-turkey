from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import hashlib
import time
import random
from functools import wraps
from threading import Lock
from collections import deque
from datetime import datetime, timedelta


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    cached: bool = False
    error: Optional[str] = None
    retry_count: int = 0
    response_time: float = 0.0


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, tokens_per_minute: int = 60, burst_size: Optional[int] = None):
        """Initialize rate limiter."""
        self.tokens_per_minute = tokens_per_minute
        self.burst_size = burst_size or tokens_per_minute
        self.tokens = self.burst_size
        self.last_refill = time.time()
        self.lock = Lock()
        
    def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, blocking if necessary. Returns wait time."""
        wait_time = 0.0
        
        with self.lock:
            now = time.time()
            # Refill tokens based on time passed
            time_passed = now - self.last_refill
            new_tokens = time_passed * (self.tokens_per_minute / 60.0)
            self.tokens = min(self.burst_size, self.tokens + new_tokens)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
            else:
                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = (tokens_needed / self.tokens_per_minute) * 60.0
                time.sleep(wait_time)
                self.tokens = 0
                
        return wait_time


class RequestTracker:
    """Track request history for adaptive rate limiting."""
    
    def __init__(self, window_minutes: int = 5):
        """Initialize request tracker."""
        self.window_minutes = window_minutes
        self.requests = deque()
        self.errors = deque()
        self.lock = Lock()
        
    def add_request(self, success: bool):
        """Add a request to the tracker."""
        with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=self.window_minutes)
            
            # Clean old requests
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            while self.errors and self.errors[0] < cutoff:
                self.errors.popleft()
                
            # Add new request
            self.requests.append(now)
            if not success:
                self.errors.append(now)
                
    def get_error_rate(self) -> float:
        """Get error rate in the window."""
        with self.lock:
            if not self.requests:
                return 0.0
            return len(self.errors) / len(self.requests)
            
    def get_request_rate(self) -> float:
        """Get requests per minute."""
        with self.lock:
            if not self.requests:
                return 0.0
            time_span = (self.requests[-1] - self.requests[0]).total_seconds() / 60.0
            if time_span == 0:
                return float(len(self.requests))
            return len(self.requests) / time_span


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    # Check if result indicates an error that should be retried
                    if hasattr(result, 'error') and result.error:
                        if attempt < max_retries:
                            # Check if it's a retryable error
                            error_lower = result.error.lower()
                            retryable_errors = [
                                'rate limit', 'timeout', 'connection',
                                'temporary', '429', '503', '502', '500'
                            ]
                            if any(err in error_lower for err in retryable_errors):
                                if jitter:
                                    actual_delay = delay * (1 + random.random() * 0.1)
                                else:
                                    actual_delay = delay
                                    
                                time.sleep(actual_delay)
                                delay = min(delay * exponential_base, max_delay)
                                result.retry_count = attempt + 1
                                continue
                    return result
                    
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        if jitter:
                            actual_delay = delay * (1 + random.random() * 0.1)
                        else:
                            actual_delay = delay
                            
                        time.sleep(actual_delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        raise
                        
            # If we get here, all retries failed
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class LLMProvider(ABC):
    """Abstract base class for LLM providers with robust features."""
    
    def __init__(self,
                 cache_enabled: bool = True,
                 rate_limit_per_minute: Optional[int] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 adaptive_rate_limiting: bool = False):
        """
        Initialize provider with optional robust features.
        
        Args:
            cache_enabled: Whether to cache responses
            rate_limit_per_minute: Optional rate limit (requests per minute)
            max_retries: Maximum number of retries for failed requests
            retry_delay: Initial delay between retries
            adaptive_rate_limiting: Whether to adapt rate based on error rate
        """
        self._cache = {} if cache_enabled else None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Rate limiting
        self.rate_limiter = None
        if rate_limit_per_minute:
            self.rate_limiter = RateLimiter(rate_limit_per_minute)
            
        # Request tracking for adaptive behavior
        self.request_tracker = RequestTracker() if adaptive_rate_limiting else None
        
    @abstractmethod
    def _call_api(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Make actual API call to the LLM provider."""
        pass
        
    def _call_api_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Call API with retry logic."""
        if self.max_retries > 0:
            decorator = retry_with_backoff(
                max_retries=self.max_retries,
                initial_delay=self.retry_delay
            )
            return decorator(self._call_api)(messages, **kwargs)
        else:
            return self._call_api(messages, **kwargs)
        
    def generate(self,
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: Optional[int] = None) -> LLMResponse:
        """Generate response from LLM with optional rate limiting and retry."""
        # Check cache
        cache_key = self._get_cache_key(prompt, system_prompt)
        if self._cache is not None and cache_key in self._cache:
            cached_response = self._cache[cache_key]
            cached_response.cached = True
            return cached_response
            
        # Apply rate limiting
        wait_time = 0.0
        if self.rate_limiter:
            wait_time = self.rate_limiter.acquire()
            
        # Adaptive rate limiting based on error rate
        if self.request_tracker:
            error_rate = self.request_tracker.get_error_rate()
            if error_rate > 0.3:  # If more than 30% errors
                time.sleep(2.0)  # Add extra delay
            elif error_rate > 0.1:  # If more than 10% errors
                time.sleep(0.5)  # Add small delay
                
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Call API with retry
        start_time = time.time()
        response = self._call_api_with_retry(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        response.response_time = time.time() - start_time
        
        # Track request
        if self.request_tracker:
            self.request_tracker.add_request(response.error is None)
            
        # Cache successful response
        if self._cache is not None and response.error is None:
            self._cache[cache_key] = response
            
        return response
        
    def _get_cache_key(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate cache key for the request."""
        key_string = f"{system_prompt or ''}|{prompt}"
        return hashlib.md5(key_string.encode()).hexdigest()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        stats = {
            'cache_size': len(self._cache) if self._cache else 0,
        }
        
        if self.request_tracker:
            stats['error_rate'] = self.request_tracker.get_error_rate()
            stats['request_rate'] = self.request_tracker.get_request_rate()
            
        return stats
        
    def clear_cache(self):
        """Clear the cache."""
        if self._cache:
            self._cache.clear()