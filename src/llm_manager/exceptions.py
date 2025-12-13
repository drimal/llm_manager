"""Exception classes for the LLM Manager package."""


class LLMProviderError(Exception):
    """Base exception for LLM provider errors.
    
    All provider-related exceptions should inherit from this class.
    """


class UnknownProviderError(LLMProviderError):
    """Raised when an unknown or unsupported provider is requested."""


class APIConnectionError(LLMProviderError):
    """Raised when there's an error connecting to the LLM provider's API."""


class AuthenticationError(LLMProviderError):
    """Raised when authentication fails (invalid API key, etc.)."""


class RateLimitError(LLMProviderError):
    """Raised when the API rate limit is exceeded."""


class TokenLimitError(LLMProviderError):
    """Raised when the token limit for a request is exceeded."""


class InvalidRequestError(LLMProviderError):
    """Raised when the request to the provider is invalid."""


class ProviderUnavailableError(LLMProviderError):
    """Raised when the provider is temporarily unavailable."""
