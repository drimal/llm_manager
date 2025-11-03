from typing import Any, Dict, List
from ..base import BaseLLMClient
import boto3
from ..utils import LLMResponse
from ..exceptions import LLMProviderError
import logging

logger = logging.getLogger(__name__)



class BedrockClient(BaseLLMClient):
    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.system_prompt = system_prompt
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        system_message = [{"text": self.system_prompt}]
        messages = [
            {"role": "user", "content": [{"text": prompt}]},
        ]
        modelId = kwargs.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
        inference_params = {
            "temperature": kwargs.get("temperature", 0.0),
            "topP": kwargs.get("top_p", 0.0),
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
            kwargs["toolConfig"] = tool_config
        logger.debug(f"LLM Request: {new_kwargs}")
        try:
            response = self.client.converse(**new_kwargs)
            logger.debug(f"LLM Response: {response}")
            text = response["output"]["message"]["content"][0]["text"]
            usage = response["usage"]
            stop_reason = response["stopReason"]
            llm_response = LLMResponse(text=text, usage=usage, stop_reason=stop_reason)
            return llm_response
        except LLMProviderError as e:
            logger.error(f"Error: {e}")
            raise e