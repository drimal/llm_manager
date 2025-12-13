"""Conftest for pytest configuration and fixtures."""

import pytest


@pytest.fixture
def mock_llm_response():
    """Fixture providing a mock LLM response."""
    from llm_manager.utils import LLMResponse
    
    return LLMResponse(
        text="This is a test response.",
        usage={
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30
        },
        stop_reason="stop"
    )


@pytest.fixture
def mock_llm_client(mock_llm_response):
    """Fixture providing a mock LLM client."""
    from unittest.mock import Mock
    from llm_manager.base import BaseLLMClient
    
    client = Mock(spec=BaseLLMClient)
    client.system_prompt = "You are helpful"
    client.generate.return_value = mock_llm_response
    return client
