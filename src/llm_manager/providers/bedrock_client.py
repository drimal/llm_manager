from __future__ import annotations

import logging
from typing import Any

from ..base import BaseLLMClient, GenerationParams
from ..exceptions import LLMProviderError, classify_error
from ..utils import LLMResponse, normalize_usage

try:
    import boto3
except ImportError:  # pragma: no cover - exercised only when SDK is absent
    boto3 = None

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"


class BedrockClient(BaseLLMClient):
    """AWS Bedrock provider client using the Converse API."""

    _provider = "bedrock"

    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str,
        system_prompt: str = "You are a helpful assistant",
    ):
        """Initialize the Bedrock client.

        Args:
            aws_access_key_id: AWS access key ID.
            aws_secret_access_key: AWS secret access key.
            region_name: AWS region.
            system_prompt: System message sent with every request.
        """
        super().__init__(system_prompt=system_prompt)
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._region_name = region_name
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            if boto3 is None:
                raise LLMProviderError("boto3 is required for BedrockClient")
            self._client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self._region_name,
                aws_access_key_id=self._aws_access_key_id,
                aws_secret_access_key=self._aws_secret_access_key,
            )
        return self._client

    def _build_request(self, prompt: str, params: GenerationParams) -> dict[str, Any]:
        request: dict[str, Any] = {
            "system": [{"text": self.system_prompt}],
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "modelId": params.model or DEFAULT_MODEL,
            "inferenceConfig": {
                "temperature": params.temperature,
                "topP": params.top_p,
                "maxTokens": params.max_tokens,
            },
            "additionalModelRequestFields": {"top_k": params.extra.get("top_k", 100)},
        }
        # Bedrock rejects an empty toolConfig; only include it when tools exist.
        if params.tools:
            request["toolConfig"] = {"tools": params.tools}
        return request

    def _complete(self, prompt: str, params: GenerationParams) -> LLMResponse:
        client = self._get_client()
        request = self._build_request(prompt, params)
        logger.debug("Bedrock request: %s", request)
        try:
            response = client.converse(**request)
        except Exception as exc:
            raise classify_error(self._provider, exc) from exc

        text = response["output"]["message"]["content"][0]["text"]
        usage = normalize_usage(response["usage"], provider=self._provider)
        return LLMResponse(text=text, usage=usage, stop_reason=response.get("stopReason"))
