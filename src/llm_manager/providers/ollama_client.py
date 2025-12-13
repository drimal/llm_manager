from typing import Any
from ..base import BaseLLMClient
from ..utils import LLMResponse, normalize_usage
from ..exceptions import LLMProviderError
from ..retry import retry_call
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None
import logging

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    """Ollama LLM provider client.
    
    Ollama provides an OpenAI-compatible API, so we use the OpenAI client library
    with a custom base URL pointing to the local Ollama instance.
    """

    def __init__(
        self,
        base_url: str,
        system_prompt: str = "You are a helpful assistant",
    ):
        """Initialize Ollama client.
        
        Args:
            base_url: URL of the Ollama instance (e.g., http://localhost:11434/v1)
            system_prompt: System message to prepend to all requests
        """
        super().__init__(system_prompt=system_prompt)
        self._base_url = base_url
        self._client = None

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a response using Ollama.
        
        Args:
            prompt: User prompt to send to the model
            **kwargs: Additional parameters including:
                - model: Model name in Ollama (default: nemotron-mini)
                - temperature: Sampling temperature (default: 0.0)
                - max_tokens: Max output tokens (default: 512)
                - top_p: Top-p sampling (default: 1.0)
                - stream: Whether to stream response (default: False)
                - stop: Stop sequences
                - n: Number of completions
                - tools: Tool definitions
                
        Returns:
            LLMResponse: Standardized response with text, usage, and stop_reason
            
        Raises:
            LLMProviderError: If API call fails
        """
        content = [{"type": "text", "text": prompt}]
        model = kwargs.get("model", "nemotron-mini")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {"role": "user", "content": content},
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
        tools = kwargs.get("tools", [])
        if tools:
            new_kwargs["tools"] = tools
        logger.debug(f"LLM Request: {messages}")            
        try:
            if self._client is None:
                if OpenAI is None:
                    raise LLMProviderError("openai library is not available for OllamaClient")
                self._client = OpenAI(base_url=self._base_url, api_key="ollama")

            # Rate limiting support
            rate_conf = kwargs.get("rate_limit") or {}
            from ..rate_limit import RateLimiter

            rate_limiter = None
            if rate_conf:
                calls = rate_conf.get("calls", 60)
                period = rate_conf.get("period", 60)
                rate_limiter = RateLimiter(calls=calls, period=period)

            # Streaming support
            if new_kwargs.get("stream"):
                def _stream_generator():
                    if rate_limiter:
                        rate_limiter.acquire()
                    stream_resp = self._client.chat.completions.create(**new_kwargs)
                    for chunk in stream_resp:
                        try:
                            yield chunk.choices[0].delta.get("content", "")
                        except Exception:
                            try:
                                yield getattr(chunk, "text", "")
                            except Exception:
                                continue

                return _stream_generator()

            response = retry_call(lambda: self._client.chat.completions.create(**new_kwargs), retries=3, backoff=1.0)
            logger.debug(f"LLM Response: {response}")
            text = response.choices[0].message.content.strip()
            # support both dict-like and pydantic-like usage objects
            usage_raw = getattr(response, "usage", None) or getattr(response, "usage", {})
            try:
                usage_dict = usage_raw.model_dump()  # pydantic style
            except Exception:
                usage_dict = usage_raw
            usage = normalize_usage(usage_dict, provider="ollama")
            stop_reason = response.choices[0].finish_reason
            llm_response = LLMResponse(
                text=text, usage=usage, stop_reason=stop_reason
            )
            
            return llm_response
        except Exception as e:
            logger.error(f"Ollama API Error: {e}")
            raise LLMProviderError(f"Ollama API Error: {e}")

    