from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from ..base import BaseLLMClient, GenerationParams
from ..exceptions import LLMProviderError, classify_error
from ..rate_limit import RateLimiter
from ..utils import LLMResponse, normalize_usage

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised only when SDK is absent
    anthropic = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-3-5-sonnet-20241022"


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude provider client."""

    _provider = "anthropic"

    def __init__(self, api_key: str, system_prompt: str = "You are a helpful assistant"):
        """Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key.
            system_prompt: System message sent with every request.
        """
        super().__init__(system_prompt=system_prompt)
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            if anthropic is None:
                raise LLMProviderError("anthropic library is not installed")
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _build_request(self, prompt: str, params: GenerationParams) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": params.model or DEFAULT_MODEL,
            "max_tokens": params.max_tokens,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.temperature,
            "top_p": params.top_p,
        }
        if params.tools:
            request["tools"] = params.tools
        return request

    def _complete(self, prompt: str, params: GenerationParams) -> LLMResponse:
        client = self._get_client()
        request = self._build_request(prompt, params)
        logger.debug("Anthropic request: %s", request)
        try:
            response = client.messages.create(**request)
        except Exception as exc:
            raise classify_error(self._provider, exc) from exc

        text = response.content[0].text if response.content else ""
        usage = normalize_usage(
            {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            provider=self._provider,
        )
        return LLMResponse(text=text, usage=usage, stop_reason=response.stop_reason)

    def _stream(
        self, prompt: str, params: GenerationParams, limiter: RateLimiter | None
    ) -> Iterator[str]:
        client = self._get_client()
        request = self._build_request(prompt, params)
        request["stream"] = True
        if limiter is not None:
            limiter.acquire()
        try:
            stream_resp = client.messages.create(**request)
        except Exception as exc:
            raise classify_error(self._provider, exc) from exc
        for event in stream_resp:
            # Text deltas arrive on content_block_delta events.
            delta = getattr(event, "delta", None)
            text = getattr(delta, "text", None) if delta is not None else None
            if text:
                yield text
