from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from ..base import BaseLLMClient, GenerationParams
from ..exceptions import LLMProviderError, classify_error
from ..rate_limit import RateLimiter
from ..utils import LLMResponse, normalize_usage

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-1.5-flash"


class GeminiClient(BaseLLMClient):
    """Google Gemini provider client using the modern ``google-genai`` SDK."""

    _provider = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        system_prompt: str = "You are a helpful assistant",
        **kwargs: Any,
    ):
        super().__init__(system_prompt=system_prompt)
        self._api_key = api_key
        self._client: Any = None
        # Default config extras applied to every request (filtered at call time).
        self._init_kwargs = kwargs

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from google import genai
        except ImportError as exc:
            raise LLMProviderError(
                "google-genai SDK not installed. Run: pip install 'llm-manager[gemini]'"
            ) from exc
        try:
            self._client = genai.Client(api_key=self._api_key)
        except Exception as exc:
            raise classify_error(self._provider, exc) from exc

    def _build_config(self, params: GenerationParams) -> Any:
        """Build a GenerateContentConfig, filtering extras to valid SDK fields."""
        from google.genai import types

        merged = {**self._init_kwargs, **params.extra}
        valid_keys = types.GenerateContentConfig.model_fields.keys()
        filtered = {k: v for k, v in merged.items() if k in valid_keys}
        if params.stop is not None and "stop_sequences" not in filtered:
            filtered["stop_sequences"] = (
                params.stop if isinstance(params.stop, list) else [params.stop]
            )
        return types.GenerateContentConfig(
            max_output_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            system_instruction=self.system_prompt,
            **filtered,
        )

    def _complete(self, prompt: str, params: GenerationParams) -> LLMResponse:
        self._ensure_client()
        config = self._build_config(params)
        model = params.model or DEFAULT_MODEL
        try:
            response = self._client.models.generate_content(
                model=model, contents=prompt, config=config
            )
        except Exception as exc:
            raise classify_error(self._provider, exc) from exc

        text = response.text or ""
        usage_raw: dict[str, Any] = {}
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is not None:
            usage_raw = {
                "prompt_tokens": usage_metadata.prompt_token_count,
                "completion_tokens": usage_metadata.candidates_token_count,
                "total_tokens": usage_metadata.total_token_count,
            }
        usage = normalize_usage(usage_raw, provider=self._provider)
        return LLMResponse(text=text, usage=usage, stop_reason=None)

    def _stream(
        self, prompt: str, params: GenerationParams, limiter: RateLimiter | None
    ) -> Iterator[str]:
        self._ensure_client()
        if limiter is not None:
            limiter.acquire()
        config = self._build_config(params)
        model = params.model or DEFAULT_MODEL
        try:
            stream = self._client.models.generate_content_stream(
                model=model, contents=prompt, config=config
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            raise classify_error(self._provider, exc) from exc
