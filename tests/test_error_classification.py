"""Tests for provider-agnostic error classification."""

import pytest

from llm_manager.exceptions import (
    APIConnectionError,
    AuthenticationError,
    InvalidRequestError,
    LLMProviderError,
    ProviderUnavailableError,
    RateLimitError,
    TokenLimitError,
    classify_error,
)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Error code: 401 - invalid api key", AuthenticationError),
        ("Unauthorized request", AuthenticationError),
        ("Rate limit exceeded, code 429", RateLimitError),
        ("RESOURCE_EXHAUSTED: quota", RateLimitError),
        ("maximum context length is 8192 tokens", TokenLimitError),
        ("Connection timed out", APIConnectionError),
        ("503 Service Unavailable", ProviderUnavailableError),
        ("400 Bad Request: validation failed", InvalidRequestError),
        ("something totally unexpected", LLMProviderError),
    ],
)
def test_classify_error_maps_messages(message, expected):
    result = classify_error("openai", Exception(message))
    assert isinstance(result, expected)
    assert isinstance(result, LLMProviderError)


def test_classify_error_passes_through_existing():
    original = RateLimitError("already classified")
    assert classify_error("openai", original) is original
