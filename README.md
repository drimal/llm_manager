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
|:---------------------:|:--------------------------:|:------------------------------------------------:|
| Reflection Strategy  | self_critique, altenative_genertion, confidence_assessment, verification, adversarial| Different methods for iterative output refinement |


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

Streaming and Rate-limiting Example

```python
from llm_manager.factory import LLMFactory

# Create client (OpenAI example)
client = LLMFactory.get_client(
    provider_name="openai",
    api_key="your-openai-api-key",
    system_prompt="You are a helpful assistant"
)

# Streaming example: iterate chunks
stream = client.generate("Tell me a short story.", stream=True)
for chunk in stream:
    print(chunk, end="", flush=True)

# Rate limiting example: allow 120 calls per 60 seconds
resp = client.generate(
    "Summarize the plot of Dune",
    rate_limit={"calls": 120, "period": 60}
)
print(resp.text)
```

## Google Gemini (optional)

If you have Google's Generative AI SDK installed (`google-generativeai`), you can use the Gemini provider via the factory. The SDK is optional — the package exposes `GeminiClient` lazily and will raise a clear ImportError if the dependency is missing.

```python
from llm_manager.factory import LLMFactory

# Create a Gemini client (optional dependency: google-generativeai)
client = LLMFactory.get_client(
    provider_name="gemini",
    api_key="YOUR_GOOGLE_API_KEY",
    model="gemini-1.5"
)

# Non-streaming:
resp = client.generate("Write a two-sentence sci-fi microstory.")
print(resp.text)

# Streaming example (yields LLMResponse chunks):
stream = client.generate("Stream a short poem.", stream=True)
for chunk in stream:
    # Each `chunk` is an `LLMResponse` Pydantic model; use `.text`
    print(chunk.text, end="", flush=True)
```


## Using Reflection Prompting

The reflection module allows you to iteratively refine model outputs through multiple reflection steps.

```python
from llm_manager.reflection import ReflectiveLLMManager
from llm_manager.providers import OpenAIProvider

# Initialize provider

provider_name = "ollama" #openai, bedrock
params = {"provider_name": provider_name}
model = "nemotron-mini"
query = "Why is the sky blue during the day?"
if provider_name == "openai":
    params["api_key"] = os.getenv("OPENAI_API_KEY")
    model = "gpt-4o-mini" # model to be used. 
elif provider_name == "anthropic":
    params["api_key"] = os.getenv("ANTHROPIC_API_KEY")
elif provider_name == "bedrock":
    params["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
    params["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    params["region_name"] = os.getenv("AWS_REGION")
    model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
elif provider_name == "ollama":
    params["base_url"] = os.getenv("OLLAMA_BASE_URL")
    model = "nemotron-mini"
else:
    raise UnknownProviderError(f"Unsupported provider: {provider_name}")

params["system_prompt"] = system_prompt
client = LLMFactory.get_client(**params)

## Without reflection
response_without_reflection = client.generate(query)
print(json.dumps(response_without_reflection), indent=4)


# For reflection
reflecton_manager = ReflectiveLLMManager(llm_client=client)
llm_config = {"model": model, 'max_tokens': 2048, "temperature" : 0.5}


reflection_response = reflecton_manager.reflect(
    user_query=query,
    reflection_strategy="adversarial",
    num_iterations=3,
    kwargs

response_dictionary = reflection_response.model_dump()
iterations = response_dictionary.get('iterations')

for i, iteration in enumerate(iterations):
    print(f"Step: {i+1}\n")
    print(f"Prompt: {iteration.get('prompt')}\n\n")
    print(f"Response: {iteration.get('response')}\n\n")
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

System prompts allow you to specify a consistent persona or context for all model interactions.
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
reflection_response = manager.reflect(
    user_query="Explain recursion.",
    reflection_strategy="self_improve",
    num_iterations=2,
    return_history=True
)
reflection_response_dict = response.model_dump()
iterations = reflection_response_dict.get('iterations')
for i, iteration in enumerate(iterations):
    print(f"Step: {i+1}\n")
    print(f"Prompt: {iteration.get('prompt')}\n\n")
    print(f"Response: {iteration.get('response')}\n\n")
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

# Installing optional Gemini SDK

If you want to use the optional Google Gemini provider, install the package with the `gemini` extra:

```bash
# editable install with Gemini extras
pip install -e '.[gemini]'

# or install just the SDK
pip install google-generativeai
```



## Version Compatibility and Dependencies

- **Python:** Supported on Python 3.9 and above.
- **Major Dependencies:**
  - `openai` for OpenAI provider integration
  - `boto3` for AWS Bedrock support
  - `requests` for HTTP communication with providers like Ollama

Ensure these packages are installed and compatible with your environment for smooth operation.

## Current Project Status

This project is actively maintained and tested. Below is a short summary of the current state so you can get started quickly.

- **Providers:** `OpenAIClient`, `AnthropicClient`, `BedrockClient`, `OllamaClient`, and `GeminiClient` (Gemini is optional and lazy-imports `google-generativeai`).
- **Validation:** `pydantic` is a required runtime dependency used for `LLMResponse` and other data models.
- **Streaming:** Providers support streaming where the upstream SDK exposes it. Streaming yields `LLMResponse` chunks.
- **Rate limiting:** Built-in `RateLimiter` utility supports token-bucket style throttling per-call via the `rate_limit` argument.
- **Retry:** Providers use a `retry_call` helper with exponential/backoff support for transient errors.
- **Testing:** A test suite (pytest) exists under `tests/`; run tests with your project venv Python:

```bash
# from project root
/path/to/venv/bin/python -m pytest -q
```

- **Optional Gemini SDK:** The `gemini` dependency group is available in `pyproject.toml` (name: `google-generativeai`). Install with extras or individually:

```bash
pip install -e .[gemini]
# or
pip install google-generativeai
```

- **How to get a client:** Use the factory: see [src/llm_manager/factory.py](src/llm_manager/factory.py) and provider implementations in [src/llm_manager/providers](src/llm_manager/providers/__init__.py).

If you'd like, I can also:
- Add an explicit `extras_require`-style entry in `pyproject.toml` to document optional installs.
- Add a short integration example for using Gemini with credentials and env var hints.


Acknowledgments
This project was built to reduce duplicate effort across LLM integration projects and streamline experimentation with different models.

⸻

## License

MIT License. See LICENSE file for more details.

## Contributing

Contributions are welcome! Please create a pull request or open an issue for bug reports and feature requests.

⸻
