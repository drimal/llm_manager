from __future__ import annotations

from typing import Any, Dict, Generator, Optional

from ..utils import LLMResponse, normalize_usage
from ..retry import retry_call
from ..rate_limit import RateLimiter


class GeminiClient:
    """Minimal Gemini client wrapper.

    This tries to lazy-import Google's Generative AI library (`google.generativeai`).
    If the SDK is missing, operations will raise ImportError with an explanatory message.

    The implementation follows the project's provider patterns: lazy SDK import,
    `generate(..., stream=...)` support (best-effort), `rate_limit` and `retry` integration,
    and normalization of usage into `LLMResponse`.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5", **kwargs: Any):
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
            raise ImportError(
                "google.generativeai is required for GeminiClient. Install with `pip install google-generativeai`"
            ) from exc

        if self._api_key:
            genai.configure(api_key=self._api_key)

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
            # Return a generator created by a nested function so this outer
            # function is not a generator (allowing non-stream returns).
            def _stream_gen():
                try:
                    self._ensure_client()
                    client = self._client
                    rate_limiter_local = rate_limit
                    if rate_limiter_local is not None:
                        rate_limiter_local.acquire()

                    if hasattr(client, "chat") and hasattr(client.chat.completions, "stream"):
                        for chunk in client.chat.completions.stream(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            max_output_tokens=max_tokens,
                            temperature=temperature,
                            **self._init_kwargs,
                            **kwargs,
                        ):
                            text = getattr(chunk, "delta", getattr(chunk, "content", ""))
                            usage_raw = getattr(chunk, "usage", {}) or {}
                            usage = normalize_usage(usage_raw, provider="gemini")
                            yield LLMResponse(text=str(text or ""), usage=usage, stop_reason=None)
                        # After stream completes, yield final combined response
                        final = _call_once()
                        yield final
                        return
                    else:
                        # No streaming support; fall back to single response but yield it once for compatibility
                        resp = retry_call(_call_once, retries=retry)
                        yield resp
                        return
                except Exception:
                    # On streaming errors, fall back to non-streaming response
                    resp = retry_call(_call_once, retries=retry)
                    yield resp
                    return

            return _stream_gen()

        # non-streaming
        if rate_limit is not None:
            with rate_limit:
                return retry_call(_call_once, retries=retry)
        return retry_call(_call_once, retries=retry)

