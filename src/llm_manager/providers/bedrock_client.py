from typing import Any
from ..base import BaseLLMClient
from ..utils import LLMResponse, normalize_usage
from ..exceptions import LLMProviderError
from ..retry import retry_call
import logging

try:
    import boto3  # type: ignore
except Exception:
    boto3 = None

logger = logging.getLogger(__name__)


class BedrockClient(BaseLLMClient):
    """AWS Bedrock LLM provider client.
    
    Implements the BaseLLMClient interface for AWS Bedrock's Converse API,
    supporting Claude and other models available through Bedrock.
    """
    
    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str,
        system_prompt: str = "You are a helpful assistant",
    ):
        """Initialize Bedrock client.
        
        Args:
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            region_name: AWS region
            system_prompt: System message to prepend to all requests
        """
        super().__init__(system_prompt=system_prompt)
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._region_name = region_name
        self._client = None

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a response using AWS Bedrock API.
        
        Args:
            prompt: User prompt to send to the model
            **kwargs: Additional parameters including:
                - model: Model ID (default: anthropic.claude-3-sonnet-20240229-v1:0)
                - temperature: Sampling temperature (default: 0.0)
                - max_tokens: Max output tokens (default: 512)
                - top_p: Top-p sampling (default: 1.0)
                - top_k: Top-k sampling (default: 100)
                - tools: Tool definitions
                
        Returns:
            LLMResponse: Standardized response with text, usage, and stop_reason
            
        Raises:
            LLMProviderError: If API call fails
        """
        system_message = [{"text": self.system_prompt}]
        messages = [
            {"role": "user", "content": [{"text": prompt}]},
        ]
        modelId = kwargs.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
        inference_params = {
            "temperature": kwargs.get("temperature", 0.0),
            "topP": kwargs.get("top_p", 1.0),
            "maxTokens": kwargs.get("max_tokens", 512),
        }
        additional_model_fields = {"top_k": kwargs.get("top_k", 100)}
        tool_config = {"tools": kwargs.get("tools", [])}

        new_kwargs = {
            "system": system_message,
            "messages": messages,
            "modelId": modelId,
            "inferenceConfig": inference_params,
            "additionalModelRequestFields": additional_model_fields,
        }
        # Include toolConfig only when it's not None
        if tool_config is not None:
            new_kwargs["toolConfig"] = tool_config
        logger.debug(f"LLM Request: {new_kwargs}")
        try:
            if self._client is None:
                if boto3 is None:
                    raise LLMProviderError("boto3 is not available for BedrockClient")
                self._client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=self._region_name,
                    aws_access_key_id=self._aws_access_key_id,
                    aws_secret_access_key=self._aws_secret_access_key,
                )

            # Rate limiting support
            rate_conf = kwargs.get("rate_limit") or {}
            from ..rate_limit import RateLimiter

            rate_limiter = None
            if rate_conf:
                calls = rate_conf.get("calls", 60)
                period = rate_conf.get("period", 60)
                rate_limiter = RateLimiter(calls=calls, period=period)

            # Bedrock doesn't reliably support streaming in the same way; perform a single call
            if rate_limiter:
                rate_limiter.acquire()
            response = retry_call(lambda: self._client.converse(**new_kwargs), retries=3, backoff=1.0)
            logger.debug(f"LLM Response: {response}")
            text = response["output"]["message"]["content"][0]["text"]
            usage = normalize_usage(response["usage"], provider="bedrock")
            stop_reason = response["stopReason"]
            llm_response = LLMResponse(text=text, usage=usage, stop_reason=stop_reason)
            return llm_response
        except Exception as e:
            logger.error(f"Bedrock API Error: {e}")
            raise LLMProviderError(f"Bedrock API Error: {e}")