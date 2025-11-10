from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLMClient(ABC):
    """Abstract base class to enforce a consistent interface for all LLM providers."""

    def __init__(self, system_prompt: str = "You are a helpful assistant"):
        """Initialize the LLM client with a system prompt."""
        self.system_prompt = system_prompt

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Given a prompt, generate a text response."""
        raise NotImplementedError


