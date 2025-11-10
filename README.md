# LLM Manager

A unified interface for interacting with multiple Large Language Model (LLM) providers using a consistent and scalable API.

This package abstracts differences between model providers such as OpenAI, AWS Bedrock, and Ollama. It lets you write application code that is **provider-agnostic**, with support for both text and chat-style prompt formats.


---

## Features

- Consistent interface across LLM providers
- Plug-and-play provider architecture
- Factory-based initialization
- System-level prompt support
- Built-in clients for:
  - OpenAI
  - AWS Bedrock
  - Ollama
  - Easy extensibility for new/custom providers

## Conceptual Overview

LLM Manager provides a factory-based initialization that abstracts the creation of provider-specific clients, enabling a unified interface to interact with various LLM providers seamlessly. Users instantiate clients through the factory by specifying provider details, after which they can generate text or chat completions without worrying about provider-specific APIs. For advanced use cases, the reflection loop allows iterative refinement of model outputs through configurable strategies, enhancing response quality by enabling the model to critique and improve its own answers.

## Reflection Prompting

The reflection module enables iterative refinement of model outputs. After an initial generation, the model can "reflect" on its own response across multiple iterations using configurable reflection strategies (e.g., critique, self-improve, summarize). This helps produce more coherent, accurate, and self-corrected answers.

### Reflection Flow Diagram

```
User Query
     ↓
Initial Generation → Reflection Step 1 → Reflection Step 2 → ... → Final Output
                      ↑                ↑
          Reflection Strategy Applied  Reflection Strategy Applied
```

### Available Reflection

| Strategy Type       | Options                  | Description                                      |
|---------------------|--------------------------|------------------------------------------------|
| Reflection Strategy  | critique, self_critique, self_improve, summarize | Different methods for iterative output refinement |


## Installation

```bash
pip install llm-manager
```
⸻


# Quick Start

```python
from llm_manager.factory import LLMFactory

# Create an OpenAI client
client = LLMFactory.get_client(
    provider_name="openai",
    api_key="your-openai-api-key",
    model="gpt-4"
)
response = client.generate(prompt="What is reinforcement learning?")
print(response["text")
```

## Using Reflection Prompting

The reflection module allows you to iteratively refine model outputs through multiple reflection steps.

```python
from llm_manager.reflection import ReflectiveLLMManager
from llm_manager.providers import OpenAIProvider

# Initialize provider
provider = OpenAIProvider(api_key="your-openai-api-key", model="gpt-4")

# Create reflection manager
manager = ReflectiveLLMManager(provider)

# Run reflection
response = manager.reflect(
    user_query="Explain quantum entanglement in simple terms.",
    reflection_strategy="self_critique",
    num_iterations=3,
    context_strategy="recent",
)

print(response)
```

## Comparing Providers

You can easily compare outputs from multiple providers using the factory interface. Here's an example that runs the same query across OpenAI and Ollama providers:

```python
from llm_manager.factory import LLMFactory

query = "What are the benefits of renewable energy?"

providers = [
    {"provider_name": "openai", "api_key": "your-openai-api-key", "model": "gpt-4"},
    {"provider_name": "ollama", "base_url": "http://localhost:11434", "model": "nemotron_mini"},
]

for config in providers:
    client = LLMFactory.get_client(**config)
    response = client.generate(prompt=query)
    print(f"Response from {config['provider_name']}:\n{response['text']}\n")
```
⸻

# Using System Prompts

System prompts allow you to specify consistent persona or context for all model interactions.
```python
from llm_manager.factory import LLMFactory
client = LLMFactory.get_client(
    provider_name="ollama",
    base_url="http://localhost:11434",
    model="nemotron_mini",
    system_prompt="You are an expert Python software engineer."
)

print(client.generate("How do I write an async function?"))

```

⸻

# Supported Providers

|Provider | Class Name|Notes|
|:---:|:---:|:---:|
|OpenAI|OpenAIClient|Supports chat & text, uses openai SDK|
|AWS Bedrock|BedrockClient|Integrates w/ boto3; flexible across models|
|Ollama|OllamaClient|Works with locally served models|

## Logging and Debugging

To facilitate troubleshooting and improve transparency, LLM Manager supports logging and debugging features. You can inspect the full reflection history to understand how outputs evolved over iterations, or enable verbose output for detailed request/response logs.

Example to inspect reflection history:

```python
response, history = manager.reflect(
    user_query="Explain recursion.",
    reflection_strategy="self_improve",
    num_iterations=2,
    return_history=True
)

print("Reflection History:")
for step, output in enumerate(history):
    print(f"Iteration {step+1}: {output}")
```

To enable verbose logging globally, set the environment variable or configure the logger in your application:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will output detailed information about API calls, prompts, and responses.

# Adding a New Provider

To add another LLM provider:
	1.	Create a class that inherits from BaseLLMClient
	2.	Implement generate()
	3.	Register it in factory.py

## Example skeleton:
```python
class MyProviderClient(BaseLLMClient):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(system_prompt=kwargs.get("system_prompt"))
        # Init here

    def generate(self, prompt: str, **kwargs) -> str:
        # Call provider API
        pass
```
---

# Roadmap
    * Add async support
    * Structured output parsing (Pydantic models)
    * Support for streaming/real-time tokens
    * Expanded Bedrock model support detection
	* CLI utility for testing providers
    * Add support for context strategies (How much prior context is included during reflection? e.g. recent, full, none)


⸻


# Development
```bash
# Install
git clone https://github.com/yourname/llm-manager.git
cd llm-manager
pip install -e .


# Run tests:

pytest

```
⸻

License

MIT License. See LICENSE file for more details.

Contributing

Contributions are welcome! Please create a pull request or open an issue for bug reports and feature requests.

⸻

## Version Compatibility and Dependencies

- **Python:** Supported on Python 3.9 and above.
- **Major Dependencies:**
  - `openai` for OpenAI provider integration
  - `boto3` for AWS Bedrock support
  - `requests` for HTTP communication with providers like Ollama

Ensure these packages are installed and compatible with your environment for smooth operation.

Acknowledgments
This project was built to reduce duplicate effort across LLM integration projects and streamline experimentation with different models.