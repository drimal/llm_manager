from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import logging
from llm_manager.prompts.prompt_library import (
    self_critique_prompt_template,
    alternative_generation_prompt_template,
    confidence_assessment_prompt_template,
    verification_prompt_template,
    adversarial_prompt_template,
)
from .base import BaseLLMClient
from .exceptions import LLMProviderError

logger = logging.getLogger(__name__)


class ReflectionStrategy(Enum):
    """Enumeration of reflection strategies for iterative output refinement.
    
    Each strategy represents a different approach to critiquing and improving
    the model's responses through multiple iterations.
    """

    SELF_CRITIQUE = "self_critique"
    ALTERNATIVE_GENERATION = "alternative_generation"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"
    VERIFICATION = "verification"
    ADVERSARIAL = "adversarial"



class ReflectionResult(BaseModel):
    """Pydantic model representing the result of a reflection process."""
    original_query: str = Field(..., description="The original user query.")
    iterations: List[Dict[str, Any]] = Field(..., description="List of iteration details.")
    final_response: str = Field(..., description="The final refined response.")
    strategy_used: ReflectionStrategy = Field(..., description="The reflection strategy used.")
    total_tokens: int = Field(..., description="Total tokens used in the reflection process.")


class ReflectionPromptBuilder:
    """Builds reflection prompts based on chosen strategy.
    
    This class encapsulates the logic for constructing reflection prompts
    that guide the model to critique and improve its own responses.
    """

    def __init__(self, strategy: ReflectionStrategy):
        """Initialize the prompt builder with a strategy.
        
        Args:
            strategy: The reflection strategy to use
        """
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
    """Generates reflection prompts and manages iterative refinement of LLM responses.
    
    This manager uses the reflection pattern to iteratively improve model outputs
    by having the model critique and refine its own responses across multiple iterations.
    """

    def __init__(self, llm_client: BaseLLMClient):
        """Initialize the reflection manager with an LLM client.
        
        Args:
            llm_client: The LLM client to use for generation and reflection
        """
        self.llm_client = llm_client  
    
    def reflect(
        self,
        user_query: str,
        reflection_strategy: str,
        num_iterations: int,
        **kwargs
    ) -> ReflectionResult:
        """Performs reflection on a user query using the specified strategy.
        
        Iteratively refines the model's response by having it critique and improve
        its own output according to the specified reflection strategy.
        
        Args:
            user_query: The initial user query
            reflection_strategy: The strategy to use for reflection (one of ReflectionStrategy values)
            num_iterations: Number of reflection iterations to perform
            **kwargs: Additional arguments passed to the LLM client's generate method
            
        Returns:
            ReflectionResult: Result object containing original query, iterations, final response, 
                            strategy, and total tokens consumed
                            
        Raises:
            LLMProviderError: If there's an error generating responses
            ValueError: If reflection_strategy is not valid
        """
        try:
            # Convert the reflection strategy string to an enum
            strategy_enum = ReflectionStrategy(reflection_strategy)
        except ValueError:
            raise ValueError(
                f"Invalid reflection strategy '{reflection_strategy}'. "
                f"Valid options: {[s.value for s in ReflectionStrategy]}"
            )
    
        # Start with the original query
        previous_response = self.llm_client.generate(user_query, **kwargs)

        # Perform iterations of reflection
        total_output_tokens = previous_response.usage.get("output_tokens", 0)
        total_input_tokens = previous_response.usage.get("input_tokens", 0)
        iteration_responses = []
        
        for iteration_num in range(num_iterations):
            logger.info(f"Reflection iteration {iteration_num + 1}/{num_iterations}")
            
            # Build the reflection prompt
            prompt_builder = ReflectionPromptBuilder(strategy_enum)
            reflection_prompt = prompt_builder.build_prompt(
                original_query=user_query, 
                previous_response=previous_response.text
            )
            
            # Generate a response using the LLM client
            try:
                reflection_response = self.llm_client.generate(reflection_prompt, **kwargs)
            except LLMProviderError as e:
                logger.error(f"Error during reflection iteration {iteration_num + 1}: {e}")
                raise
            
            total_output_tokens += reflection_response.usage.get("output_tokens", 0)
            total_input_tokens += reflection_response.usage.get("input_tokens", 0)
            
            iteration_responses.append({
                "iteration": iteration_num + 1,
                "prompt": reflection_prompt,
                "response": reflection_response.text
            })
            
            # Update the previous response for next iteration
            previous_response = reflection_response
        
        reflection_result = ReflectionResult(
            original_query=user_query,
            iterations=iteration_responses,
            final_response=previous_response.text,
            strategy_used=strategy_enum,
            total_tokens=total_output_tokens + total_input_tokens
        )
        
        logger.info(f"Reflection complete. Total tokens used: {reflection_result.total_tokens}")
        return reflection_result
