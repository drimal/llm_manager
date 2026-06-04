from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from ..base import BaseLLMClient, GenerationParams
from ..exceptions import LLMProviderError, classify_error
from ..rate_limit import RateLimiter
from ..utils import LLMResponse, normalize_usage

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when SDK is absent
    OpenAI = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nemotron-mini"


class OllamaClient(BaseLLMClient):
    """Ollama provider client.

    Ollama exposes an OpenAI-compatible API, so this reuses the ``openai`` SDK
    pointed at a local (or remote) Ollama base URL.
    """

    _provider = "ollama"

    def __init__(self, base_url: str, system_prompt: str = "You are a helpful assistant"):
        """Initialize the Ollama client.

        Args:
            base_url: URL of the Ollama instance (e.g. ``http://localhost:11434/v1``).
            system_prompt: System message prepended to all requests.
        """
        super().__init__(system_prompt=system_prompt)
        self._base_url = base_url
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            if OpenAI is None:
                raise LLMProviderError("openai library is required for OllamaClient")
            self._client = OpenAI(base_url=self._base_url, api_key="ollama")
        return self._client

    def _build_request(self, prompt: str, params: GenerationParams, *, stream: bool) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        request: dict[str, Any] = {
            "messages": messages,
            "model": params.model or DEFAULT_MODEL,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "top_p": params.top_p,
            "stop": params.stop,
            "stream": stream,
        }
        if params.tools:
            request["tools"] = params.tools
        return request

    def _complete(self, prompt: str, params: GenerationParams) -> LLMResponse:
        client = self._get_client()
        request = self._build_request(prompt, params, stream=False)
        logger.debug("Ollama request: %s", request)
        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            raise classify_error(self._provider, exc) from exc

        text = (response.choices[0].message.content or "").strip()
        usage_raw: Any = getattr(response, "usage", None) or {}
        usage_dict = usage_raw.model_dump() if hasattr(usage_raw, "model_dump") else usage_raw
        usage = normalize_usage(usage_dict, provider=self._provider)
        return LLMResponse(
            text=text,
            usage=usage,
            stop_reason=response.choices[0].finish_reason,
        )

    def _stream(
        self, prompt: str, params: GenerationParams, limiter: RateLimiter | None
    ) -> Iterator[str]:
        client = self._get_client()
        request = self._build_request(prompt, params, stream=True)
        if limiter is not None:
            limiter.acquire()
        try:
            stream_resp = client.chat.completions.create(**request)
        except Exception as exc:
            raise classify_error(self._provider, exc) from exc
        for chunk in stream_resp:
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                yield content
