"""Tests for the lazy top-level public API and registry-driven factory."""

import sys

import llm_manager
from llm_manager import LLMFactory


def test_top_level_exports_resolve():
    for name in (
        "LLMFactory",
        "LLMResponse",
        "BaseLLMClient",
        "GenerationParams",
        "RateLimiter",
        "ReflectiveLLMManager",
        "ReflectionStrategy",
        "LLMProviderError",
        "RateLimitError",
    ):
        assert getattr(llm_manager, name) is not None


def test_unknown_top_level_attribute_raises():
    import pytest

    with pytest.raises(AttributeError):
        llm_manager.DoesNotExist  # noqa: B018


def test_available_providers():
    assert set(LLMFactory.available_providers()) == {
        "openai",
        "anthropic",
        "bedrock",
        "ollama",
        "gemini",
    }


def test_get_client_does_not_import_other_sdks():
    # Building an openai client must not import boto3/anthropic/google.
    sys.modules.pop("boto3", None)
    LLMFactory.get_client("openai", api_key="x")
    assert "boto3" not in sys.modules


def test_from_model_id_resolves_default_model(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
providers:
  openai:
    env_vars:
      api_key: TEST_OPENAI_KEY
models:
  fast:
    provider: openai
    model_name: gpt-4o-mini
    tags: [fast]
"""
    )
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")
    client = LLMFactory.from_model_id("fast", config, system_prompt="sys")
    assert client.default_model == "gpt-4o-mini"
    assert client.system_prompt == "sys"


def test_from_model_id_unknown_raises(tmp_path):
    import pytest

    from llm_manager.exceptions import UnknownProviderError

    config = tmp_path / "models.yaml"
    config.write_text("providers: {}\nmodels: {}\n")
    with pytest.raises(UnknownProviderError):
        LLMFactory.from_model_id("nope", config)
