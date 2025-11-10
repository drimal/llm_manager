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

## Reflection Prompting

The reflection module enables iterative refinement of model outputs. After an initial generation, the model can "reflect" on its own response across multiple iterations using configurable reflection strategies (e.g., critique, self-improve, summarize) . This helps produce more coherent, accurate, and self-corrected answers.

---

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

Acknowledgments
This project was built to reduce duplicate effort across LLM integration projects and streamline experimentation with different models.

---</file>