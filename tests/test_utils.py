"""Unit tests for utilities and response handling."""

import pytest
from llm_manager.utils import LLMResponse, normalize_usage


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_response_creation(self):
        """Test creating an LLMResponse."""
        usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        response = LLMResponse(
            text="Hello, world!",
            usage=usage,
            stop_reason="stop"
        )
        assert response.text == "Hello, world!"
        assert response.usage == usage
        assert response.stop_reason == "stop"

    def test_response_without_stop_reason(self):
        """Test creating response without stop_reason."""
        usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        response = LLMResponse(
            text="Hello!",
            usage=usage
        )
        assert response.stop_reason is None

    def test_response_json_serialization(self):
        """Test that response can be serialized to JSON."""
        usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        response = LLMResponse(
            text="Test",
            usage=usage,
            stop_reason="stop"
        )
        json_str = response.model_dump_json()
        assert "Test" in json_str
        assert "stop" in json_str


class TestNormalizeUsage:
    """Tests for usage normalization across providers."""

    def test_openai_usage_normalization(self):
        """Test normalizing OpenAI usage format."""
        openai_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
        normalized = normalize_usage(openai_usage, provider="openai")
        assert normalized["input_tokens"] == 10
        assert normalized["output_tokens"] == 20
        assert normalized["total_tokens"] == 30

    def test_bedrock_usage_normalization(self):
        """Test normalizing Bedrock usage format."""
        bedrock_usage = {
            "inputTokens": 10,
            "outputTokens": 20
        }
        normalized = normalize_usage(bedrock_usage, provider="bedrock")
        assert normalized["input_tokens"] == 10
        assert normalized["output_tokens"] == 20
        assert normalized["total_tokens"] == 30

    def test_ollama_usage_normalization(self):
        """Test normalizing Ollama usage format."""
        ollama_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
        normalized = normalize_usage(ollama_usage, provider="ollama")
        assert normalized["input_tokens"] == 10
        assert normalized["output_tokens"] == 20
        assert normalized["total_tokens"] == 30

    def test_generic_usage_normalization(self):
        """Test fallback normalization for unknown provider."""
        generic_usage = {
            "input_tokens": 5,
            "output_tokens": 15,
            "total_tokens": 20
        }
        normalized = normalize_usage(generic_usage)
        assert normalized["input_tokens"] == 5
        assert normalized["output_tokens"] == 15
        assert normalized["total_tokens"] == 20

    def test_missing_tokens_default_to_zero(self):
        """Test that missing token counts default to zero."""
        incomplete_usage = {}
        normalized = normalize_usage(incomplete_usage, provider="openai")
        assert normalized["input_tokens"] == 0
        assert normalized["output_tokens"] == 0
        assert normalized["total_tokens"] == 0
