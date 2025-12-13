from __future__ import annotations

from typing import Any, Dict, Generator, Optional

from ..base import BaseLLMClient
from ..utils import LLMResponse, normalize_usage
from ..exceptions import LLMProviderError
from ..retry import retry_call
from ..rate_limit import RateLimiter
import logging

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """Minimal Gemini client wrapper.

    This tries to lazy-import Google's Generative AI library (`google.generativeai`).
    If the SDK is missing, operations will raise ImportError with an explanatory message.

    The implementation follows the project's provider patterns: lazy SDK import,
    `generate(..., stream=...)` support (best-effort), `rate_limit` and `retry` integration,
    and normalization of usage into `LLMResponse`.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5", system_prompt: str = "You are a helpful assistant", **kwargs: Any):
        super().__init__(system_prompt=system_prompt)
        self._api_key = api_key
        self.model = model
        self._client = None
        self._init_kwargs = kwargs

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on SDK availability
            # Use project exception for consistency across providers
            raise LLMProviderError("google-generativeai SDK not installed (pip install google-generativeai)") from exc

        if self._api_key:
            try:
                genai.configure(api_key=self._api_key)
            except Exception:
                # Some SDK versions may not expose configure; ignore if so
                logger.debug("genai.configure failed or not present; continuing")

        self._client = genai

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stream: bool = False,
        retry: int = 2,
        rate_limit: Optional[RateLimiter] = None,
        **kwargs: Any,
    ) -> Generator[LLMResponse, None, LLMResponse] | LLMResponse:
        """Generate text from Gemini model.

        - If `stream=True` and the SDK supports streaming, yields incremental `LLMResponse` chunks.
        - Otherwise returns a single `LLMResponse`.
        """

        def _call_once() -> LLMResponse:
            self._ensure_client()
            # Use best-effort call shape. The `google.generativeai` module exposes
            # `genai.chat.completions.create` or `genai.text.completions.create` depending on usage.
            client = self._client

            request_kwargs: Dict[str, Any] = dict(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            request_kwargs.update(self._init_kwargs)
            request_kwargs.update(kwargs)

            # Prefer chat completions if available
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                resp = client.chat.completions.create(**request_kwargs)
                # Attempt to extract text and usage
                text = ""
                if hasattr(resp, "candidates") and resp.candidates:
                    text = resp.candidates[0].content if hasattr(resp.candidates[0], "content") else str(resp.candidates[0])
                elif hasattr(resp, "message"):
                    text = resp.message.get("content", "")
                usage = getattr(resp, "usage", None)
            else:
                # Fallback to text completions API shape
                resp = client.text.completions.create(**request_kwargs)
                text = "".join([c.output for c in getattr(resp, "candidates", [])])
                usage = getattr(resp, "usage", None)

            usage_raw = usage or {}
            usage_dict = normalize_usage(usage_raw, provider="gemini")
            return LLMResponse(text=text or "", usage=usage_dict, stop_reason=None)

        if stream:
            def _stream_gen():
                try:
                    self._ensure_client()
                    client = self._client
                    if rate_limit is not None:
                        rate_limit.acquire()

                    # If SDK offers a streaming iterator, yield text chunks (strings)
                    if hasattr(client, "chat") and hasattr(client.chat.completions, "stream"):
                        for chunk in client.chat.completions.stream(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            max_output_tokens=max_tokens,
                            temperature=temperature,
                            **self._init_kwargs,
                            **kwargs,
                        ):
                            try:
                                # delta-based content
                                yield chunk.candidates[0].delta.get("content", "")
                            except Exception:
                                try:
                                    yield getattr(chunk, "text", "")
                                except Exception:
                                    continue
                        return

                    # Fallback: no streaming support; yield single string
                    resp = retry_call(_call_once, retries=retry)
                    yield resp.text
                    return
                except LLMProviderError:
                    raise
                except Exception as e:
                    logger.error(f"Gemini streaming error: {e}")
                    resp = retry_call(_call_once, retries=retry)
                    yield resp.text
                    return

            return _stream_gen()

        # non-streaming
        if rate_limit is not None:
            with rate_limit:
                return retry_call(_call_once, retries=retry)
        return retry_call(_call_once, retries=retry)

