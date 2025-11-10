from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from llm_manager.prompts.prompt_library import (
    self_critique_prompt_template,
    alternative_generation_prompt_template,
    confidence_assessment_prompt_template,
    verification_prompt_template,
    adversarial_prompt_template,
)
from .base import BaseLLMClient


class ReflectionStrategy(Enum):
    """Enumeration of reflection strategies."""

    SELF_CRITIQUE = "self_critique"
    ALTERNATIVE_GENERATION = "alternative_generation"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"
    VERIFICATION = "verification"
    ADVERSARIAL = "adversarial"



class ReflectionResult(BaseModel):
    """Represents the result of a reflection process."""

    original_query: str = Field(description="The original user query.")
    iterations: List[Dict[str, str]] = Field(description="List of iteration details.")
    final_response: str = Field(description="The final refined response.")
    strategy_used: ReflectionStrategy = Field(description="The reflection strategy used.")
    total_tokens: int = Field(description="Total tokens used in the reflection process.")


class ReflectionPromptBuilder:
    """Builds reflection prompts based on chosen strategy."""

    def __init__(self, strategy: ReflectionStrategy):
        self.strategy = strategy

    def build_prompt(
        self, original_query: str, previous_response: str) -> str:
        """Constructs a reflection prompt based on the chosen strategy.
        Args:
            original_query: Initial user query
            previous_response: LLM's previous response
        Returns:
            str: The constructed reflection prompt
        """
        strategy_map = {
            ReflectionStrategy.SELF_CRITIQUE: self._build_self_critique_prompt,
            ReflectionStrategy.ALTERNATIVE_GENERATION: self._build_alternative_generation_prompt,
            ReflectionStrategy.CONFIDENCE_ASSESSMENT: self._build_confidence_assessment_prompt,
            ReflectionStrategy.VERIFICATION: self._build_verification_prompt,
            ReflectionStrategy.ADVERSARIAL: self._build_adversarial_prompt,
            # Add more strategies as needed
        }
        return strategy_map[self.strategy](original_query, previous_response)

    def _build_self_critique_prompt(
        self, original_query: str, previous_response: str) -> str:
        """Builds a self-critique prompt."""
        return self_critique_prompt_template.format(
            query=original_query, response=previous_response
        )

    def _build_alternative_generation_prompt(
        self, original_query: str, previous_response: str) -> str:
        """Builds an alternative generation prompt."""  
        return alternative_generation_prompt_template.format(
            query=original_query, response=previous_response
        )

    def _build_confidence_assessment_prompt(
        self, original_query: str, previous_response: str) -> str:
        """Builds a confidence assessment prompt."""
        return confidence_assessment_prompt_template.format(
            query=original_query, response=previous_response
        )

    def _build_verification_prompt(
        self, original_query: str, previous_response: str) -> str:
        """Builds a verification prompt."""
        return verification_prompt_template.format(
            query=original_query, response=previous_response
        )

    def _build_adversarial_prompt(
        self, original_query: str, previous_response: str
    ) -> str:
        """Builds an adversarial prompt."""
        return adversarial_prompt_template.format(
            query=original_query, response=previous_response
        )

class ReflectiveLLMManager:
    """Generates reflection prompts based on the selected strategy."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client  
    
    def reflect(
        self,
        user_query: str,
        reflection_strategy: str,
        num_iterations: int,
        **kwargs: Dict[str, str]
    ) -> str:
        """Performs reflection on a user query using the specified strategy.
        Args:
            user_query: The initial user query
            reflection_strategy: The strategy to use for reflection
            num_iterations: Number of iterations to perform
        Returns:
            str: The final response after reflection
        """
        # Convert the reflection strategy string to an enum
        strategy_enum = ReflectionStrategy(reflection_strategy)
    
        # Start with the original query
        current_query = user_query
        previous_response = self.llm_client.generate(current_query, **kwargs)

        # Perform iterations of reflection
        total_output_tokens = previous_response.usage.get("output_tokens", 0)
        total_input_tokens = previous_response.usage.get("input_tokens", 0)
        iteration_responses = []
        for _ in range(num_iterations):
            # Build the reflection prompt
            prompt_builder = ReflectionPromptBuilder(strategy_enum)
            reflection_prompt = prompt_builder.build_prompt(original_query=user_query, previous_response=previous_response.text)
            # Generate a response using the LLM client
            reflection_response = self.llm_client.generate(reflection_prompt, **kwargs)
            total_output_tokens += reflection_response.usage.get("output_tokens", 0)
            total_input_tokens += reflection_response.usage.get("input_tokens", 0)
            iteration_responses.append({
                "prompt": reflection_prompt,
                "response": reflection_response.text
            })
            # Update the previous response for next iteration
            previous_response = reflection_response  # Store the generated text as previous response
        reflection_result = ReflectionResult(
            original_query=user_query,
            iterations=iteration_responses,  # Could be populated with detailed iteration info
            final_response=previous_response.text,
            strategy_used=strategy_enum,
            total_tokens=total_output_tokens + total_input_tokens  # Token counting can be implemented as needed
        )
        return reflection_result  # Return the final refined response after all iterations
