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


def classify_error(provider: str, exc: Exception) -> LLMProviderError:
    """Map a raw provider/SDK exception to a specific LLMProviderError subclass.

    Provider SDKs raise their own exception types with inconsistent names, so we
    classify heuristically using the exception's class name and message. This
    lets callers catch semantically meaningful errors (e.g. ``RateLimitError``)
    regardless of which provider produced them.

    Args:
        provider: Name of the provider that raised the error (for the message).
        exc: The original exception raised by the SDK.

    Returns:
        An ``LLMProviderError`` subclass instance. Callers should ``raise`` it
        ``from`` the original exception to preserve the traceback chain.
    """
    if isinstance(exc, LLMProviderError):
        return exc

    haystack = f"{type(exc).__name__} {exc}".lower()
    message = f"{provider} API error: {exc}"

    def _has(*needles: str) -> bool:
        return any(n in haystack for n in needles)

    if _has("authenticat", "unauthorized", "api key", "apikey", "permissiondenied", "401", "403"):
        return AuthenticationError(message)
    if _has("rate limit", "ratelimit", "too many requests", "429", "resourceexhausted", "quota"):
        return RateLimitError(message)
    if _has(
        "context length", "maximum context", "token limit", "too many tokens", "context window"
    ):
        return TokenLimitError(message)
    if _has("connection", "timeout", "timed out", "network", "unreachable"):
        return APIConnectionError(message)
    if _has(
        "unavailable", "service_unavailable", "internal server", "500", "502", "503", "overloaded"
    ):
        return ProviderUnavailableError(message)
    if _has("invalid", "bad request", "not found", "400", "404", "validation"):
        return InvalidRequestError(message)
    return LLMProviderError(message)
