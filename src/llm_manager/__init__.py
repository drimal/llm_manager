"""llm_manager: a unified, provider-agnostic interface for LLM providers.

The public API is resolved lazily via module ``__getattr__`` so that importing
``llm_manager`` does not pull in heavy provider SDKs. Typical usage::

    from llm_manager import LLMFactory

    client = LLMFactory.get_client(provider_name="openai", api_key="...")
    response = client.generate("Why is the sky blue?")
    print(response.text)
"""

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LLMFactory",
    "BaseLLMClient",
    "GenerationParams",
    "LLMResponse",
    "RateLimiter",
    "ReflectiveLLMManager",
    "ReflectionStrategy",
    "ReflectionResult",
    "LLMProviderError",
    "UnknownProviderError",
    "APIConnectionError",
    "AuthenticationError",
    "RateLimitError",
    "TokenLimitError",
    "InvalidRequestError",
    "ProviderUnavailableError",
]

# Maps each public symbol to the submodule that defines it (for lazy import).
_EXPORTS = {
    "LLMFactory": "llm_manager.factory",
    "BaseLLMClient": "llm_manager.base",
    "GenerationParams": "llm_manager.base",
    "LLMResponse": "llm_manager.utils",
    "RateLimiter": "llm_manager.rate_limit",
    "ReflectiveLLMManager": "llm_manager.reflection",
    "ReflectionStrategy": "llm_manager.reflection",
    "ReflectionResult": "llm_manager.reflection",
    "LLMProviderError": "llm_manager.exceptions",
    "UnknownProviderError": "llm_manager.exceptions",
    "APIConnectionError": "llm_manager.exceptions",
    "AuthenticationError": "llm_manager.exceptions",
    "RateLimitError": "llm_manager.exceptions",
    "TokenLimitError": "llm_manager.exceptions",
    "InvalidRequestError": "llm_manager.exceptions",
    "ProviderUnavailableError": "llm_manager.exceptions",
}

if TYPE_CHECKING:  # pragma: no cover - import hints for type checkers only
    from .base import BaseLLMClient, GenerationParams
    from .exceptions import (
        APIConnectionError,
        AuthenticationError,
        InvalidRequestError,
        LLMProviderError,
        ProviderUnavailableError,
        RateLimitError,
        TokenLimitError,
        UnknownProviderError,
    )
    from .factory import LLMFactory
    from .rate_limit import RateLimiter
    from .reflection import ReflectionResult, ReflectionStrategy, ReflectiveLLMManager
    from .utils import LLMResponse


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
