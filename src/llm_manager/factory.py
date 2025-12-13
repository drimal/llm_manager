from typing import Any
from .exceptions import UnknownProviderError


class LLMFactory:
    """Factory to initialize LLM clients based on provider name.

    Provider classes are imported lazily to avoid importing heavy third-party
    libraries at module import time (which helps when running tests).
    """

    @staticmethod
    def get_client(provider_name: str, **kwargs: Any):
        """Get an LLM client for the specified provider.

        Lazy-imports provider implementations to prevent top-level import side-effects.
        """
        provider_name = provider_name.lower()
        if provider_name == "openai":
            from .providers.openai_client import OpenAIClient

            return OpenAIClient(**kwargs)
        if provider_name == "anthropic":
            from .providers.anthropic_client import AnthropicClient

            return AnthropicClient(**kwargs)
        if provider_name == "bedrock":
            from .providers.bedrock_client import BedrockClient

            return BedrockClient(**kwargs)
        if provider_name == "ollama":
            from .providers.ollama_client import OllamaClient

            return OllamaClient(**kwargs)
        if provider_name == "gemini":
            from .providers.gemini_client import GeminiClient

            return GeminiClient(**kwargs)

        raise UnknownProviderError(
            f"Provider '{provider_name}' is not supported. Available providers: openai, anthropic, bedrock, ollama, gemini"
        )
