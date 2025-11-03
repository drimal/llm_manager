from typing import Any, Dict, List
from ..base import BaseLLMClient
from ..utils import LLMResponse
from ..exceptions import LLMProviderError
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    """Client for Ollama, which mirrors OpenAI's API structure."""

    def __init__(
        self,
        base_url: str,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.system_prompt = system_prompt
        self.client = OpenAI(base_url=base_url, api_key="ollama")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        content = [{"type": "text", "text": prompt}]
        model = kwargs.get("model", "nemotron-mini")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {"role": "user", "content": content},
        ]
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 256),
                top_p=kwargs.get("top_p", 0.0),
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
