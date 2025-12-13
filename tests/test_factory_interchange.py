import pytest

from llm_manager.factory import LLMFactory
from llm_manager.base import BaseLLMClient
from llm_manager.exceptions import UnknownProviderError


def test_factory_returns_clients_and_has_generate():
    providers = [
        ("openai", {"api_key": "x"}),
        ("anthropic", {"api_key": "x"}),
        ("ollama", {"base_url": "http://localhost:11434"}),
        ("bedrock", {"region_name": "us-east-1", "aws_access_key_id": "x", "aws_secret_access_key": "y"}),
        ("gemini", {"api_key": "x"}),
    ]

    for name, kwargs in providers:
        client = LLMFactory.get_client(provider_name=name, **kwargs)
        assert isinstance(client, BaseLLMClient)
        assert callable(getattr(client, "generate", None))


def test_unknown_provider_raises():
    with pytest.raises(UnknownProviderError):
        LLMFactory.get_client("this-provider-does-not-exist")
