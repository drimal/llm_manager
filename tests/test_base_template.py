"""Tests for the BaseLLMClient template method (rate-limit, retry, streaming)."""

from collections.abc import Iterator

import pytest

from llm_manager.base import BaseLLMClient, GenerationParams
from llm_manager.rate_limit import RateLimiter
from llm_manager.utils import LLMResponse


class RecordingClient(BaseLLMClient):
    """Provider stub that records the params it receives."""

    def __init__(self, fail_times: int = 0):
        super().__init__()
        self.calls: list[GenerationParams] = []
        self.fail_times = fail_times
        self.attempts = 0

    def _complete(self, prompt: str, params: GenerationParams) -> LLMResponse:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError("transient")
        self.calls.append(params)
        return LLMResponse(text=f"ok:{prompt}", usage={}, stop_reason=None)

    def _stream(
        self, prompt: str, params: GenerationParams, limiter: RateLimiter | None
    ) -> Iterator[str]:
        yield from ("a", "b", "c")


def test_generate_returns_response_and_passes_params():
    client = RecordingClient()
    resp = client.generate("hello", model="m", temperature=0.5, max_tokens=10)
    assert isinstance(resp, LLMResponse)
    assert resp.text == "ok:hello"
    assert client.calls[0].model == "m"
    assert client.calls[0].temperature == 0.5
    assert client.calls[0].max_tokens == 10


def test_default_model_is_used_when_not_specified():
    client = RecordingClient()
    client.default_model = "fallback-model"
    client.generate("hi")
    assert client.calls[0].model == "fallback-model"


def test_extra_kwargs_flow_into_params_extra():
    client = RecordingClient()
    client.generate("hi", top_k=42)
    assert client.calls[0].extra["top_k"] == 42


def test_retry_on_transient_failure():
    client = RecordingClient(fail_times=2)
    resp = client.generate("hi", retries=3, backoff=0)
    assert resp.text == "ok:hi"
    assert client.attempts == 3


def test_exhausted_retries_raise():
    client = RecordingClient(fail_times=5)
    with pytest.raises(RuntimeError):
        client.generate("hi", retries=2, backoff=0)


def test_stream_returns_iterator_of_strings():
    client = RecordingClient()
    out = list(client.generate("hi", stream=True))
    assert out == ["a", "b", "c"]


def test_rate_limit_dict_is_converted():
    client = RecordingClient()
    # Should not raise; a generous limit lets the single call through.
    client.generate("hi", rate_limit={"calls": 5, "period": 1})
    assert client.calls


def test_default_complete_not_implemented():
    class Bare(BaseLLMClient):
        pass

    with pytest.raises(NotImplementedError):
        Bare().generate("hi")
