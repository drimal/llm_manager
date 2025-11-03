from typing import Any, Dict, List
from ..base import BaseLLMClient
from ..utils import LLMResponse
from ..exceptions import LLMProviderError
import openai
import logging

logger = logging.getLogger(__name__)



class OpenAIClient(BaseLLMClient):
    def __init__(
        self, api_key: str, system_prompt: str = "You are a helpful assistant."
    ):
        self.system_prompt = system_prompt
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        model = kwargs.get("model", "gpt-3.5-turbo")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        try:
            logger.debug(f"LLM Request: {messages}")
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.5),
                max_tokens=kwargs.get("max_tokens", 256),
                top_p=kwargs.get("top_p", 1),
            )
            logger.debug(f"LLM Response: {response}")
            text = response.choices[0].message.content.strip()
            usage = response.usage
            usage_stats = {
                "inputTokens": usage.prompt_tokens,
                "outputTokens": usage.completion_tokens,
                "totalTokens": usage.total_tokens,
            }
            stop_reason = response.choices[0].finish_reason
            llm_response = LLMResponse(
                text=text, usage=usage_stats, stop_reason=stop_reason
            )
            return llm_response
        except LLMProviderError as e:
            logger.error(f"Error: {e}")
            raise LLMProviderError(f"Error: {e}")