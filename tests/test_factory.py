"""Unit tests for the LLM Factory."""

import pytest
from llm_manager.factory import LLMFactory
from llm_manager.exceptions import UnknownProviderError
from llm_manager.providers import (
    OpenAIClient,
    AnthropicClient,
    BedrockClient,
    OllamaClient,
)


class TestLLMFactory:
    """Tests for LLMFactory.get_client method."""

    def test_openai_client_creation(self):
        """Test creating an OpenAI client."""
        client = LLMFactory.get_client(
            provider_name="openai",
            api_key="test-key"
        )
        assert isinstance(client, OpenAIClient)
        assert client.system_prompt == "You are a helpful assistant"

    def test_anthropic_client_creation(self):
        """Test creating an Anthropic client."""
        client = LLMFactory.get_client(
            provider_name="anthropic",
            api_key="test-key"
        )
        assert isinstance(client, AnthropicClient)

    def test_bedrock_client_creation(self):
        """Test creating a Bedrock client."""
        client = LLMFactory.get_client(
            provider_name="bedrock",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1"
        )
        assert isinstance(client, BedrockClient)

    def test_ollama_client_creation(self):
        """Test creating an Ollama client."""
        client = LLMFactory.get_client(
            provider_name="ollama",
            base_url="http://localhost:11434/v1"
        )
        assert isinstance(client, OllamaClient)

    def test_custom_system_prompt(self):
        """Test creating a client with custom system prompt."""
        custom_prompt = "You are an expert programmer"
        client = LLMFactory.get_client(
            provider_name="openai",
            api_key="test-key",
            system_prompt=custom_prompt
        )
        assert client.system_prompt == custom_prompt

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises UnknownProviderError."""
        with pytest.raises(UnknownProviderError):
            LLMFactory.get_client(
                provider_name="unknown_provider",
                api_key="test-key"
            )

    def test_unknown_provider_error_message(self):
        """Test that error message lists available providers."""
        with pytest.raises(UnknownProviderError) as exc_info:
            LLMFactory.get_client(
                provider_name="invalid",
                api_key="test-key"
            )
        error_msg = str(exc_info.value)
        assert "openai" in error_msg
        assert "anthropic" in error_msg
        assert "bedrock" in error_msg
        assert "ollama" in error_msg
