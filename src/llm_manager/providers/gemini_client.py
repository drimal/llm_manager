from __future__ import annotations

from typing import Any, Dict, List, Generator, Optional

from ..base import BaseLLMClient
from ..utils import LLMResponse, normalize_usage
from ..exceptions import LLMProviderError
from ..retry import retry_call
from ..rate_limit import RateLimiter
import logging

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """Minimal Gemini client wrapper using the modern 'google-genai' SDK."""

    def __init__(self, api_key: Optional[str] = None, system_prompt: str = "You are a helpful assistant", **kwargs: Any):
        super().__init__(system_prompt=system_prompt)
        self._api_key = api_key
        self._client = None
        # Store extras, but we must filter them later
        self._init_kwargs = kwargs

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            # Lazy import is fine, but we need the types for validation
            from google import genai
        except ImportError as exc:
            raise LLMProviderError("google-genai SDK not installed. Run: pip install google-genai") from exc
        
        try:
            self._client = genai.Client(api_key=self._api_key)
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Gemini Client: {e}")

    def _create_safe_config(self, max_tokens: int, temperature: float, **kwargs) -> genai.types.GenerateContentConfig:
        """Helper to filter kwargs against valid Pydantic fields to prevent crashes."""
        from google import genai
        from google.genai import types

        # 1. Merge init-time kwargs with request-time kwargs
        all_kwargs = {**self._init_kwargs, **kwargs}

        # 2. Introspect the Pydantic model to find valid field names
        #    (e.g., top_p, top_k, candidate_count, stop_sequences, response_mime_type)
        valid_keys = types.GenerateContentConfig.model_fields.keys()

        # 3. Filter out anything not supported by the SDK
        filtered_kwargs = {k: v for k, v in all_kwargs.items() if k in valid_keys}

        # 4. Return the strict config object
        return types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=self.system_prompt,
            **filtered_kwargs
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stream: bool = False,
        retry: int = 2,
        rate_limit: Optional[Any] = None,
        **kwargs: Any,
    ) -> Generator[LLMResponse, None, None] | LLMResponse:
        
        def _call_once() -> LLMResponse:
            self._ensure_client()
            
            # Use the safe config creator
            config = self._create_safe_config(max_tokens, temperature, **kwargs)

            try:
                response = self._client.models.generate_content(
                    model=kwargs.get["model"],
                    contents=prompt,
                    config=config
                )
            except Exception as e:
                raise LLMProviderError(f"Gemini generation failed: {e}") from e

            text_content = response.text if response.text else ""

            # Extract Usage
            usage_raw = {}
            if hasattr(response, "usage_metadata"):
                usage_raw = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
            
            # Use your existing normalize_usage function
            usage_dict = normalize_usage(usage_raw, provider="gemini")
            
            return LLMResponse(text=text_content, usage=usage_dict, stop_reason=None)

        if stream:
            def _stream_gen():
                self._ensure_client()
                
                if rate_limit is not None:
                    rate_limit.acquire()
                
                # Use the safe config creator
                config = self._create_safe_config(max_tokens, temperature, **kwargs)

                try:
                    response_stream = self._client.models.generate_content_stream(
                        model=kwargs.get["model"],
                        contents=prompt,
                        config=config
                    )
                    
                    for chunk in response_stream:
                        if chunk.text:
                            yield chunk.text
                            
                except Exception as e:
                    # Log error, then fallback to non-streaming retry logic
                    # logger.error(f"Gemini streaming error: {e}") 
                    resp = retry_call(_call_once, retries=retry)
                    yield resp.text

            return _stream_gen()

        if rate_limit is not None:
            with rate_limit:
                return retry_call(_call_once, retries=retry)
        return retry_call(_call_once, retries=retry)