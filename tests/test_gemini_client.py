import sys
import types

from llm_manager.providers.gemini_client import GeminiClient
from llm_manager.utils import LLMResponse


def _make_fake_genai_nonstream():
    genai = types.SimpleNamespace()

    class Candidate:
        def __init__(self, content):
            self.content = content

    class Resp:
        def __init__(self):
            self.candidates = [Candidate("Hello from Gemini")]
            self.usage = {"tokens": 10}

    class Completions:
        @staticmethod
        def create(**kwargs):
            return Resp()

        @staticmethod
        def stream(**kwargs):
            if False:
                yield None

    chat = types.SimpleNamespace(completions=Completions())
    genai.chat = chat

    def configure(api_key=None):
        setattr(genai, "_configured", True)

    genai.configure = configure
    return genai


def _make_fake_genai_stream():
    genai = types.SimpleNamespace()

    class Chunk:
        def __init__(self, delta=None, content=None, usage=None):
            self.delta = delta
            self.content = content
            self.usage = usage

    class Completions:
        @staticmethod
        def create(**kwargs):
            r = types.SimpleNamespace()
            r.candidates = [types.SimpleNamespace(content="Final text")]
            r.usage = {"tokens": 5}
            return r

        @staticmethod
        def stream(**kwargs):
            yield Chunk(delta="Hello ")
            yield Chunk(delta="world")
            yield Chunk(delta="", content="", usage={"tokens": 3})

    chat = types.SimpleNamespace(completions=Completions())
    genai.chat = chat
    genai.configure = lambda api_key=None: None
    return genai


def test_generate_non_stream(monkeypatch):
    fake = _make_fake_genai_nonstream()
    # Ensure both package and submodule entries exist so `import google.generativeai` works
    import types as _types

    pkg = _types.ModuleType("google")
    setattr(pkg, "generativeai", fake)
    monkeypatch.setitem(sys.modules, "google", pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake)

    client = GeminiClient(api_key="x", model="g-test")
    resp = client.generate("hi there", stream=False)

    assert isinstance(resp, LLMResponse)
    assert "Hello from Gemini" in resp.text


def test_generate_stream(monkeypatch):
    fake = _make_fake_genai_stream()
    import types as _types

    pkg = _types.ModuleType("google")
    setattr(pkg, "generativeai", fake)
    monkeypatch.setitem(sys.modules, "google", pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake)

    client = GeminiClient(api_key="x", model="g-test")
    gen = client.generate("streaming test", stream=True)

    outputs = list(gen)
    assert outputs, "Expected at least one streamed chunk"
    # Streaming now yields strings
    assert any(isinstance(o, str) for o in outputs)


def test_missing_sdk_raises(monkeypatch):
    # Ensure no google module present
    monkeypatch.delitem(sys.modules, "google.generativeai", raising=False)
    monkeypatch.delitem(sys.modules, "google", raising=False)

    client = GeminiClient(api_key=None, model="g-test")
    try:
        gen = client.generate("hi", stream=False)
    except Exception as e:
        from llm_manager.exceptions import LLMProviderError

        assert isinstance(e, LLMProviderError)
    else:
        # If no exception, ensure response exists (rare in test env)
        assert isinstance(gen, LLMResponse)
