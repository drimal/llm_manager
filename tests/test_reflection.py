from llm_manager.reflection import ReflectiveLLMManager, ReflectionResult
from llm_manager.base import BaseLLMClient
from llm_manager.utils import LLMResponse


class DummyClient(BaseLLMClient):
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        super().__init__(system_prompt=system_prompt)
        self.counter = 0

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Return predictable responses that change each call
        self.counter += 1
        text = f"response-{self.counter}: {prompt[:30]}"
        usage = {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        return LLMResponse(text=text, usage=usage, stop_reason=None)


def test_reflective_manager_basic():
    client = DummyClient()
    mgr = ReflectiveLLMManager(llm_client=client)
    result: ReflectionResult = mgr.reflect(
        user_query="Explain test",
        reflection_strategy="self_critique",
        num_iterations=2,
    )

    assert result.original_query == "Explain test"
    assert len(result.iterations) == 2
    assert result.final_response.startswith("response-")
    assert result.total_tokens >= 0
"""Unit tests for reflection module."""

import pytest
from llm_manager.reflection import (
    ReflectionStrategy,
    ReflectionResult,
    ReflectionPromptBuilder,
)


class TestReflectionStrategy:
    """Tests for ReflectionStrategy enum."""

    def test_all_strategies_exist(self):
        """Test that all expected strategies are defined."""
        expected = [
            "self_critique",
            "alternative_generation",
            "confidence_assessment",
            "verification",
            "adversarial"
        ]
        actual = [s.value for s in ReflectionStrategy]
        for strategy in expected:
            assert strategy in actual

    def test_strategy_from_string(self):
        """Test creating strategy from string value."""
        strategy = ReflectionStrategy("self_critique")
        assert strategy == ReflectionStrategy.SELF_CRITIQUE

    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError):
            ReflectionStrategy("invalid_strategy")


class TestReflectionResult:
    """Tests for ReflectionResult model."""

    def test_result_creation(self):
        """Test creating a ReflectionResult."""
        iterations = [
            {"iteration": 1, "prompt": "test", "response": "response"}
        ]
        result = ReflectionResult(
            original_query="What is AI?",
            iterations=iterations,
            final_response="AI is...",
            strategy_used=ReflectionStrategy.SELF_CRITIQUE,
            total_tokens=100
        )
        assert result.original_query == "What is AI?"
        assert result.final_response == "AI is..."
        assert result.strategy_used == ReflectionStrategy.SELF_CRITIQUE
        assert result.total_tokens == 100

    def test_result_json_serialization(self):
        """Test that result can be serialized to JSON."""
        result = ReflectionResult(
            original_query="Test",
            iterations=[],
            final_response="Result",
            strategy_used=ReflectionStrategy.SELF_CRITIQUE,
            total_tokens=50
        )
        json_str = result.model_dump_json()
        assert "Test" in json_str
        assert "Result" in json_str
        assert "self_critique" in json_str


class TestReflectionPromptBuilder:
    """Tests for ReflectionPromptBuilder."""

    def test_self_critique_prompt_building(self):
        """Test building a self-critique prompt."""
        builder = ReflectionPromptBuilder(ReflectionStrategy.SELF_CRITIQUE)
        prompt = builder.build_prompt(
            original_query="What is AI?",
            previous_response="AI is artificial intelligence..."
        )
        assert "What is AI?" in prompt
        assert "AI is artificial intelligence..." in prompt
        assert "critique" in prompt.lower()

    def test_alternative_generation_prompt_building(self):
        """Test building an alternative generation prompt."""
        builder = ReflectionPromptBuilder(ReflectionStrategy.ALTERNATIVE_GENERATION)
        prompt = builder.build_prompt(
            original_query="Explain ML",
            previous_response="ML is machine learning..."
        )
        assert "Explain ML" in prompt
        assert "ML is machine learning..." in prompt

    def test_confidence_assessment_prompt_building(self):
        """Test building a confidence assessment prompt."""
        builder = ReflectionPromptBuilder(ReflectionStrategy.CONFIDENCE_ASSESSMENT)
        prompt = builder.build_prompt(
            original_query="What is DL?",
            previous_response="DL is deep learning..."
        )
        assert "What is DL?" in prompt
        assert "confidence" in prompt.lower()

    def test_verification_prompt_building(self):
        """Test building a verification prompt."""
        builder = ReflectionPromptBuilder(ReflectionStrategy.VERIFICATION)
        prompt = builder.build_prompt(
            original_query="Verify statement",
            previous_response="The statement is true..."
        )
        assert "Verify statement" in prompt
        assert "verify" in prompt.lower()

    def test_adversarial_prompt_building(self):
        """Test building an adversarial prompt."""
        builder = ReflectionPromptBuilder(ReflectionStrategy.ADVERSARIAL)
        prompt = builder.build_prompt(
            original_query="Argue a point",
            previous_response="Point is valid because..."
        )
        assert "Argue a point" in prompt
        assert "challenge" in prompt.lower() or "argue" in prompt.lower()
