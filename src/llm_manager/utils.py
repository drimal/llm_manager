from typing import Any

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Pydantic-based standardized response format.

    Pydantic is a required dependency for the project: this model provides
    validation and convenient serialization methods used across the codebase.
    """

    text: str = Field(..., description="The generated text response.")
    usage: dict[str, int] = Field(default_factory=dict, description="Normalized token usage info")
    stop_reason: str | None = Field(None, description="Stop reason")


def normalize_usage(usage_dict: dict[str, Any], provider: str = "generic") -> dict[str, int]:
    """Normalize usage information across different providers.

    Different providers return token counts with different key names.
    This function normalizes them to a standard format:
    - input_tokens
    - output_tokens
    - total_tokens

    Args:
        usage_dict: Raw usage dict from the provider
        provider: Name of the provider (openai, bedrock, ollama)

    Returns:
        Dict with standardized keys: input_tokens, output_tokens, total_tokens
    """
    if provider == "bedrock":
        # Bedrock returns inputTokens, outputTokens
        return {
            "input_tokens": usage_dict.get("inputTokens", 0),
            "output_tokens": usage_dict.get("outputTokens", 0),
            "total_tokens": usage_dict.get("inputTokens", 0) + usage_dict.get("outputTokens", 0),
        }
    elif provider in ("openai", "ollama", "gemini"):
        # OpenAI, Ollama and (mapped) Gemini use prompt/completion/total tokens.
        return {
            "input_tokens": usage_dict.get("prompt_tokens", usage_dict.get("inputTokens", 0)),
            "output_tokens": usage_dict.get("completion_tokens", usage_dict.get("outputTokens", 0)),
            "total_tokens": usage_dict.get("total_tokens", 0),
        }
    else:
        # Fallback: try common patterns (covers anthropic and unknown providers).
        input_tokens = usage_dict.get(
            "input_tokens", usage_dict.get("inputTokens", usage_dict.get("prompt_tokens", 0))
        )
        output_tokens = usage_dict.get(
            "output_tokens", usage_dict.get("outputTokens", usage_dict.get("completion_tokens", 0))
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            # Fall back to input + output when the provider omits a total.
            "total_tokens": usage_dict.get("total_tokens", input_tokens + output_tokens),
        }
