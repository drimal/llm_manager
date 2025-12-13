from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from llm_manager.utils import LLMResponse


class BaseLLMClient(ABC):
    """Abstract base class to enforce a consistent interface for all LLM providers.
    
    This class defines the interface that all LLM provider implementations must follow,
    ensuring consistent behavior across different providers (OpenAI, Bedrock, Ollama, etc.).
    """

    def __init__(self, system_prompt: str = "You are a helpful assistant"):
        """Initialize the LLM client with a system prompt.
        
        Args:
            system_prompt: The system message to use for all generations. Defaults to a generic helper prompt.
        """
        self.system_prompt = system_prompt

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate a response for the given prompt.
        
        Args:
            prompt: The user prompt/query to generate a response for.
            **kwargs: Additional arguments like model, temperature, max_tokens, etc.
            
        Returns:
            LLMResponse: An LLMResponse object containing the text, usage info, and stop reason.
            
        Raises:
            LLMProviderError: If there's an error communicating with the provider.
        """
        raise NotImplementedError


