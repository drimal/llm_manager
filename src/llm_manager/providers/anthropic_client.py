from typing import Any
from ..base import BaseLLMClient
from ..utils import LLMResponse, normalize_usage
from ..exceptions import LLMProviderError
import logging

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude LLM provider client.
    
    Implements the BaseLLMClient interface for Anthropic's Claude API,
    supporting the latest Claude models with customizable parameters.
    """
    
    def __init__(
        self,
        api_key: str,
        system_prompt: str = "You are a helpful assistant",
    ):
        """Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key
            system_prompt: System message to prepend to all requests
        """
        super().__init__(system_prompt=system_prompt)
        self._api_key = api_key
        self._client = None

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a response using Anthropic Claude API.
        
        Args:
            prompt: User prompt to send to the model
            **kwargs: Additional parameters including:
                - model: Model name (default: claude-3-5-sonnet-20241022)
                - temperature: Sampling temperature (default: 0.0)
                - max_tokens: Max output tokens (default: 512)
                - top_p: Top-p sampling (default: 1.0)
                - tools: Tool definitions
                
        Returns:
            LLMResponse: Standardized response with text, usage, and stop_reason
            
        Raises:
            LLMProviderError: If API call fails
        """
        model = kwargs.get("model", "claude-3-5-sonnet-20241022")
        tools = kwargs.get("tools", [])

        try:
            logger.debug(f"LLM Request - Prompt: {prompt}, Model: {model}")
            
            from ..retry import retry_call

            if self._client is None:
                if anthropic is None:
                    raise LLMProviderError("anthropic library is not installed")
                self._client = anthropic.Anthropic(api_key=self._api_key)

            # Rate limiting support
            rate_conf = kwargs.get("rate_limit") or {}
            from ..rate_limit import RateLimiter

            rate_limiter = None
            if rate_conf:
                calls = rate_conf.get("calls", 60)
                period = rate_conf.get("period", 60)
                rate_limiter = RateLimiter(calls=calls, period=period)

            if self._client is None:
                if anthropic is None:
                    raise LLMProviderError("anthropic library is not installed")
                self._client = anthropic.Anthropic(api_key=self._api_key)

            # Anthropic supports streaming via incremental responses; if stream requested, yield chunks
            if kwargs.get("stream"):
                def _stream_generator():
                    if rate_limiter:
                        rate_limiter.acquire()
                    stream_resp = self._client.messages.create(
                        model=model,
                        max_tokens=kwargs.get("max_tokens", 512),
                        system=self.system_prompt,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=kwargs.get("temperature", 0.0),
                        top_p=kwargs.get("top_p", 1.0),
                        tools=tools if tools else None,
                        stream=True,
                    )
                    for chunk in stream_resp:
                        try:
                            yield getattr(chunk, "text", "")
                        except Exception:
                            continue

                return _stream_generator()

            response = retry_call(
                lambda: self._client.messages.create(
                    model=model,
                    max_tokens=kwargs.get("max_tokens", 512),
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.0),
                    top_p=kwargs.get("top_p", 1.0),
                    tools=tools if tools else None,
                ),
                retries=3,
                backoff=1.0,
            )
            
            logger.debug(f"LLM Response: {response}")
            
            text = response.content[0].text if response.content else ""
            usage = normalize_usage({
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }, provider="anthropic")
            stop_reason = response.stop_reason
            
            llm_response = LLMResponse(
                text=text, usage=usage, stop_reason=stop_reason
            )
            return llm_response
            
        except Exception as e:
            logger.error(f"Anthropic API Error: {e}")
            raise LLMProviderError(f"Anthropic API Error: {e}")
