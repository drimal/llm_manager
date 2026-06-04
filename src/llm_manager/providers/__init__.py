"""LLM provider client implementations.

Client classes are resolved lazily via module ``__getattr__`` so that importing
this subpackage does not eagerly import every vendor SDK. ``from
llm_manager.providers import OpenAIClient`` works as expected.
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    "OpenAIClient",
    "AnthropicClient",
    "BedrockClient",
    "OllamaClient",
    "GeminiClient",
]

_MODULE_BY_NAME = {
    "OpenAIClient": "openai_client",
    "AnthropicClient": "anthropic_client",
    "BedrockClient": "bedrock_client",
    "OllamaClient": "ollama_client",
    "GeminiClient": "gemini_client",
}

if TYPE_CHECKING:  # pragma: no cover - import hints for type checkers only
    from .anthropic_client import AnthropicClient
    from .bedrock_client import BedrockClient
    from .gemini_client import GeminiClient
    from .ollama_client import OllamaClient
    from .openai_client import OpenAIClient


def __getattr__(name: str) -> Any:
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
