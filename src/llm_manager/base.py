from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .rate_limit import RateLimiter
from .retry import retry_call
from .utils import LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class GenerationParams:
    """Normalized request parameters shared by every provider.

    Providers translate these into their own SDK-specific request shape inside
    :meth:`BaseLLMClient._complete` / :meth:`BaseLLMClient._stream`.
    """

    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    stop: Any | None = None
    tools: list[Any] = field(default_factory=list)
    # Provider-specific extras that don't map onto the common fields above.
    extra: dict[str, Any] = field(default_factory=dict)


class BaseLLMClient:
    """Base class enforcing a consistent interface for all providers.

    Subclasses implement the provider-specific :meth:`_complete` (and optionally
    :meth:`_stream`) hooks. The public :meth:`generate` method is a template that
    handles rate limiting, retries, and streaming dispatch uniformly so that
    individual providers stay small and focused.
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant",
        default_model: str | None = None,
    ):
        """Initialize the LLM client.

        Args:
            system_prompt: System message used for all generations.
            default_model: Model used by :meth:`generate` when ``model`` is not
                passed explicitly. Falls back to the provider's own default.
        """
        self.system_prompt = system_prompt
        self.default_model = default_model

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        stop: Any | None = None,
        tools: list[Any] | None = None,
        stream: bool = False,
        rate_limit: dict[str, Any] | RateLimiter | None = None,
        retries: int = 3,
        backoff: float = 1.0,
        **kwargs: Any,
    ) -> LLMResponse | Iterator[str]:
        """Generate a response for ``prompt``.

        Args:
            prompt: The user prompt to send to the model.
            model: Model name/ID. Each provider supplies a default if omitted.
            temperature: Sampling temperature.
            max_tokens: Maximum number of output tokens.
            top_p: Nucleus sampling parameter.
            stop: Optional stop sequence(s).
            tools: Optional tool/function definitions.
            stream: If True, return an iterator of text chunks instead of an
                :class:`~llm_manager.utils.LLMResponse`.
            rate_limit: Either a configured :class:`RateLimiter` or a dict like
                ``{"calls": 60, "period": 60}``.
            retries: Number of attempts for non-streaming calls.
            backoff: Initial backoff (seconds) between retries; doubles each time.
            **kwargs: Provider-specific extras forwarded via ``params.extra``.

        Returns:
            An :class:`LLMResponse` for non-streaming calls, or an iterator of
            string chunks when ``stream=True``.

        Raises:
            LLMProviderError: (or a subclass) if the provider call fails.
        """
        params = GenerationParams(
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            tools=list(tools) if tools else [],
            extra=dict(kwargs),
        )
        limiter = self._resolve_rate_limiter(rate_limit)

        if stream:
            return self._stream(prompt, params, limiter)

        def _call() -> LLMResponse:
            if limiter is not None:
                limiter.acquire()
            return self._complete(prompt, params)

        return retry_call(_call, retries=retries, backoff=backoff)

    @staticmethod
    def _resolve_rate_limiter(
        rate_limit: dict[str, Any] | RateLimiter | None,
    ) -> RateLimiter | None:
        """Build a :class:`RateLimiter` from a dict config, or pass one through."""
        if rate_limit is None:
            return None
        if isinstance(rate_limit, RateLimiter):
            return rate_limit
        if isinstance(rate_limit, dict) and rate_limit:
            return RateLimiter(
                calls=rate_limit.get("calls", 60),
                period=rate_limit.get("period", 60),
            )
        return None

    def _complete(self, prompt: str, params: GenerationParams) -> LLMResponse:
        """Provider-specific non-streaming generation. Override in subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement non-streaming generation"
        )

    def _stream(
        self,
        prompt: str,
        params: GenerationParams,
        limiter: RateLimiter | None,
    ) -> Iterator[str]:
        """Provider-specific streaming generation. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")
