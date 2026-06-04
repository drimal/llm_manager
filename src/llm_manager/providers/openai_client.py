from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from ..base import BaseLLMClient, GenerationParams
from ..exceptions import LLMProviderError, classify_error
from ..rate_limit import RateLimiter
from ..utils import LLMResponse, normalize_usage

try:
    import openai
except ImportError:  # pragma: no cover - exercised only when SDK is absent
    openai = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-3.5-turbo"


class OpenAIClient(BaseLLMClient):
    """OpenAI chat-completions provider client."""

    _provider = "openai"

    def __init__(self, api_key: str, system_prompt: str = "You are a helpful assistant"):
        """Initialize the OpenAI client.

        The underlying ``openai`` SDK is imported lazily and the client is
        constructed on first use, so instances can be created in environments
        where the SDK is not installed (e.g. unit tests).
        """
        super().__init__(system_prompt=system_prompt)
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            if openai is None:
                raise LLMProviderError("openai library is not installed")
            self._client = openai.OpenAI(api_key=self._api_key)
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
        logger.debug("OpenAI request: %s", request)
        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            raise classify_error(self._provider, exc) from exc

        text = (response.choices[0].message.content or "").strip()
        usage_raw: Any = getattr(response, "usage", None) or {}
        # OpenAI usage is a pydantic model; fall back to dict-like access otherwise.
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
