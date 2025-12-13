from typing import Any
from ..base import BaseLLMClient
from ..utils import LLMResponse, normalize_usage
from ..exceptions import LLMProviderError
from ..retry import retry_call

try:
    import openai  # type: ignore
except Exception:
    openai = None
import logging

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM provider client.
    
    Implements the BaseLLMClient interface for OpenAI's API,
    supporting chat completions with customizable parameters.
    """
    
    def __init__(
        self, api_key: str, system_prompt: str = "You are a helpful assistant"
    ):
        """Initialize OpenAI client.

        Note: the underlying `openai` library is imported lazily when `generate`
        is called. This allows creating client instances in environments where
        the `openai` package is not installed (e.g., unit tests).
        """
        super().__init__(system_prompt=system_prompt)
        self._api_key = api_key
        self._client = None

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a response using OpenAI API.
        
        Args:
            prompt: User prompt to send to the model
            **kwargs: Additional parameters including:
                - model: Model name (default: gpt-3.5-turbo)
                - temperature: Sampling temperature (default: 0.0)
                - max_tokens: Max output tokens (default: 512)
                - top_p: Top-p sampling parameter (default: 1.0)
                - stream: Whether to stream response (default: False)
                - stop: Stop sequences
                - n: Number of completions
                - tools: Tool definitions
                
        Returns:
            LLMResponse: Standardized response with text, usage, and stop_reason
            
        Raises:
            LLMProviderError: If API call fails
        """
        model = kwargs.get("model", "gpt-3.5-turbo")
        tools = kwargs.get("tools", [])

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        new_kwargs = {
            "messages": messages,
            "model": model,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 512),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": kwargs.get("stream", False),
            "stop": kwargs.get("stop", None),
            "n": kwargs.get("n", 1),
        }
        if tools:
            new_kwargs["tools"] = tools

        # Handle optional rate limiting configuration
        rate_conf = kwargs.get("rate_limit") or {}
        from ..rate_limit import RateLimiter

        rate_limiter = None
        if rate_conf:
            calls = rate_conf.get("calls", 60)
            period = rate_conf.get("period", 60)
            rate_limiter = RateLimiter(calls=calls, period=period)

        try:
            logger.debug(f"LLM Request: {messages}")
            # Lazily instantiate the underlying OpenAI client if needed
            if self._client is None:
                if openai is None:
                    raise LLMProviderError("openai library is not installed")
                self._client = openai.OpenAI(api_key=self._api_key)

            # If streaming is requested, return a generator that yields chunks
            if new_kwargs.get("stream"):
                def _stream_generator():
                    if rate_limiter:
                        rate_limiter.acquire()
                    stream_resp = self._client.chat.completions.create(**new_kwargs)
                    for chunk in stream_resp:
                        # SDK chunk shape may vary; yield text content when available
                        try:
                            # for delta-based streaming
                            yield chunk.choices[0].delta.get("content", "")
                        except Exception:
                            try:
                                yield getattr(chunk, "text", "")
                            except Exception:
                                continue

                return _stream_generator()

            if rate_limiter:
                rate_limiter.acquire()

            response = retry_call(lambda: self._client.chat.completions.create(**new_kwargs), retries=3, backoff=1.0)
            logger.debug(f"LLM Response: {response}")
            text = response.choices[0].message.content.strip()
            # normalize usage for both pydantic and dict-like objects
            usage_raw = getattr(response, "usage", None) or {}
            try:
                usage_dict = usage_raw.model_dump()
            except Exception:
                usage_dict = usage_raw
            usage = normalize_usage(usage_dict, provider="openai")
            stop_reason = response.choices[0].finish_reason
            llm_response = LLMResponse(
                text=text, usage=usage, stop_reason=stop_reason
            )
            return llm_response
        except Exception as e:
            logger.error(f"OpenAI API Error: {e}")
            raise LLMProviderError(f"OpenAI API Error: {e}")