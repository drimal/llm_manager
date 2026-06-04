"""Tests for the Gemini provider client.

These mock the modern ``google-genai`` SDK by replacing ``genai.Client`` with a
fake. The real ``google.genai.types.GenerateContentConfig`` is used so that the
client's config-filtering logic is exercised against the actual SDK schema.
"""

import sys
import types as pytypes

import pytest

from llm_manager.providers.gemini_client import GeminiClient
from llm_manager.utils import LLMResponse
from llm_manager.exceptions import LLMProviderError

genai = pytest.importorskip("google.genai")


class _FakeUsage:
    prompt_token_count = 10
    candidates_token_count = 20
    total_token_count = 30


class _FakeResponse:
    text = "Hello from Gemini"
    usage_metadata = _FakeUsage()


class _FakeModels:
    def generate_content(self, model, contents, config):
        return _FakeResponse()

    def generate_content_stream(self, model, contents, config):
        for part in ("Hello ", "world"):
            yield pytypes.SimpleNamespace(text=part)


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.models = _FakeModels()


def test_generate_non_stream(monkeypatch):
    monkeypatch.setattr(genai, "Client", _FakeClient)

    client = GeminiClient(api_key="x")
    resp = client.generate("hi there", model="gemini-1.5-flash", stream=False)

    assert isinstance(resp, LLMResponse)
    assert resp.text == "Hello from Gemini"
    assert resp.usage["input_tokens"] == 10
    assert resp.usage["output_tokens"] == 20
    assert resp.usage["total_tokens"] == 30


def test_generate_stream(monkeypatch):
    monkeypatch.setattr(genai, "Client", _FakeClient)

    client = GeminiClient(api_key="x")
    gen = client.generate("streaming test", model="gemini-1.5-flash", stream=True)

    outputs = list(gen)
    assert outputs == ["Hello ", "world"]
    assert all(isinstance(o, str) for o in outputs)


def test_missing_sdk_raises(monkeypatch):
    # Simulate the SDK being absent so the import inside _ensure_client fails.
    monkeypatch.setitem(sys.modules, "google.genai", None)

    client = GeminiClient(api_key=None)
    with pytest.raises(LLMProviderError):
        client.generate("hi", model="gemini-1.5-flash", stream=False)
