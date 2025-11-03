from typing import Any, Dict
from .providers.openai_client import OpenAIClient

# from .providers.anthropic_client import AnthropicClient
from .providers.bedrock_client import BedrockClient
from .providers.ollama_client import OllamaClient
from .exceptions import UnknownProviderError


class LLMFactory:
    """Factory to initialize LLM clients based on provider name."""

    @staticmethod
    def get_client(provider_name: str, **kwargs: Any):
        providers = {
            "openai": OpenAIClient,
            # "anthropic": AnthropicClient,
            "bedrock": BedrockClient,
            "ollama": OllamaClient,
        }
        if provider_name not in providers:
            raise UnknownProviderError(f"Provider '{provider_name}' is not supported.")
        return providers[provider_name](**kwargs)
