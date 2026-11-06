"""Universal Asynchronous Model Gateway.

Connects to OpenAI-compatible cloud providers:
- OpenRouter (openrouter.ai) with free models (Llama 3.3 70B, DeepSeek R1, Qwen 2.5)
- NVIDIA NIM API (integrate.api.nvidia.com)
- Groq Cloud (api.groq.com)
- Local vLLM / Ollama

Features:
- Asynchronous HTTP connection pooling via httpx.AsyncClient.
- Exponential backoff with jitter on rate limits (429) and transient errors (500, 502, 503).
- Automated fallback routing from primary to secondary provider.
- Token usage accounting.
"""

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from src.data.redaction import redact_data
from src.utils.config import Settings, get_settings

logger = logging.getLogger(__name__)


class GatewayResponse(BaseModel):
    """Standardized response from the model gateway."""

    content: str = ""
    raw_response: Dict[str, Any] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    provider: str = ""
    model: str = ""


class ModelGateway:
    """Asynchronous client managing connections and failover to cloud LLM APIs."""

    PROVIDER_URLS = {
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
        "groq": "https://api.groq.com/openai/v1/chat/completions",
        "local": "http://localhost:8000/v1/chat/completions",
    }

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        """Retrieve or create an async HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.model_timeout_seconds, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Resolve API key for the requested provider."""
        if provider == "openrouter":
            return self.settings.openrouter_api_key
        elif provider == "nvidia":
            return self.settings.nvidia_api_key
        elif provider == "groq":
            return self.settings.groq_api_key
        return None

    def _prepare_headers(self, provider: str) -> Dict[str, str]:
        """Construct authentication and routing headers."""
        headers = {
            "Content-Type": "application/json",
        }
        api_key = self._get_api_key(provider)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://lexsi.ai"
            headers["X-Title"] = "Lexsi Track 3 Agent"

        return headers

    async def _execute_single_call(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> GatewayResponse:
        """Execute a single HTTP request to the designated provider."""
        client = await self.get_client()
        url = self.PROVIDER_URLS.get(provider, self.PROVIDER_URLS["openrouter"])
        headers = self._prepare_headers(provider)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_time = time.time()
        response = await client.post(url, headers=headers, json=payload)
        latency = time.time() - start_time

        if response.status_code != 200:
            error_text = response.text[:500]
            raise httpx.HTTPStatusError(
                f"Provider {provider} returned status {response.status_code}: {error_text}",
                request=response.request,
                response=response,
            )

        data = response.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        return GatewayResponse(
            content=content,
            raw_response=data,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency,
            provider=provider,
            model=model,
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GatewayResponse:
        """Generate response with retry, backoff, and fallback routing."""
        primary_provider = provider or self.settings.default_model_provider
        primary_model = model or self.settings.default_model_name
        temp = temperature if temperature is not None else self.settings.model_temperature
        tokens = max_tokens or self.settings.model_max_tokens

        # Attempt primary provider with retries
        for attempt in range(1, self.settings.model_max_retries + 1):
            try:
                logger.debug("Calling %s (%s) attempt %d", primary_provider, primary_model, attempt)
                return await self._execute_single_call(
                    provider=primary_provider,
                    model=primary_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                )
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                is_rate_limit = isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429
                logger.warning(
                    "Primary provider %s failed (attempt %d/%d): %s",
                    primary_provider, attempt, self.settings.model_max_retries, e
                )
                if attempt < self.settings.model_max_retries:
                    # Exponential backoff with jitter
                    sleep_time = (self.settings.model_retry_backoff_factor ** attempt) + random.uniform(0.1, 0.5)
                    await asyncio.sleep(sleep_time)
                else:
                    break

        # Fallback to secondary provider if available and different
        fallback_provider = self.settings.fallback_model_provider
        fallback_model = self.settings.fallback_model_name
        if fallback_provider and (fallback_provider != primary_provider or fallback_model != primary_model):
            logger.info("Failing over to fallback provider: %s (%s)", fallback_provider, fallback_model)
            try:
                return await self._execute_single_call(
                    provider=fallback_provider,
                    model=fallback_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                )
            except Exception as fb_err:
                logger.error("Fallback provider %s failed: %s", fallback_provider, fb_err)
                raise fb_err

        raise RuntimeError(f"All model providers failed for model {primary_model}")

