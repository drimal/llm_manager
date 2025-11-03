class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""


class UnknownProviderError(LLMProviderError):
    """Thrown when an unknown provider is requested."""
