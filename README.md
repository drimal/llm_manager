# LLM Manager

A unified, **provider-agnostic** interface for multiple Large Language Model providers.

Write your application against one consistent API and switch between **OpenAI**,
**Anthropic**, **AWS Bedrock**, **Ollama**, and **Google Gemini** without changing
your code. Provider SDKs are optional extras, so you only install what you use.

## Features

- One consistent interface (`generate`) across every provider
- Factory-based, lazy client creation — no provider SDK is imported until needed
- Optional per-provider dependencies (`pip install 'llm-manager[openai]'`)
- Standardized `LLMResponse` (text + normalized token usage) via Pydantic
- Uniform **streaming** contract: streaming yields plain `str` chunks
- Built-in **retry** with exponential backoff and **rate limiting** (token bucket)
- Provider-agnostic **exception hierarchy** (auth, rate-limit, token-limit, …)
- A **reflection** loop for iterative self-critique and refinement
- A YAML **model registry** to map friendly model ids to providers/credentials
- Fully type-hinted (`py.typed`)

## Installation

```bash
pip install llm-manager                 # core only (no provider SDKs)
pip install 'llm-manager[openai]'       # + OpenAI
pip install 'llm-manager[anthropic]'    # + Anthropic
pip install 'llm-manager[bedrock]'      # + AWS Bedrock (boto3)
pip install 'llm-manager[gemini]'       # + Google Gemini (google-genai)
pip install 'llm-manager[ollama]'       # + Ollama (uses the OpenAI SDK)
pip install 'llm-manager[all]'          # everything
```

Requires Python 3.11+.

## Quick start

```python
from llm_manager import LLMFactory

client = LLMFactory.get_client(provider_name="openai", api_key="sk-...")

response = client.generate("What is reinforcement learning?", model="gpt-4o-mini")
print(response.text)
print(response.usage)   # {"input_tokens": ..., "output_tokens": ..., "total_tokens": ...}
```

Switching providers is just different constructor arguments:

```python
anthropic = LLMFactory.get_client(provider_name="anthropic", api_key="sk-ant-...")
ollama    = LLMFactory.get_client(provider_name="ollama", base_url="http://localhost:11434/v1")
gemini    = LLMFactory.get_client(provider_name="gemini", api_key="...")
bedrock   = LLMFactory.get_client(
    provider_name="bedrock",
    aws_access_key_id="...", aws_secret_access_key="...", region_name="us-east-1",
)
```

## The `generate` API

```python
client.generate(
    prompt,
    model=None,            # provider default used if omitted
    temperature=0.0,
    max_tokens=512,
    top_p=1.0,
    stop=None,
    tools=None,
    stream=False,          # True -> returns an iterator of str chunks
    rate_limit=None,       # {"calls": 60, "period": 60} or a RateLimiter
    retries=3,
    backoff=1.0,
    **provider_specific,   # forwarded to the provider (e.g. top_k for Bedrock/Gemini)
)
```

Non-streaming calls return an `LLMResponse`; streaming calls return an iterator of
text fragments.

### Streaming

All providers yield plain string chunks, so reassembly is identical everywhere:

```python
parts = []
for chunk in client.generate("Write a short poem.", model="gpt-4o-mini", stream=True):
    print(chunk, end="", flush=True)
    parts.append(chunk)
full_text = "".join(parts)
```

### Rate limiting

```python
resp = client.generate(
    "Summarize the plot of Dune.",
    model="gpt-4o-mini",
    rate_limit={"calls": 120, "period": 60},  # 120 calls per 60 seconds
)
```

## Model registry (`from_model_id`)

Define friendly model ids in a YAML file (see [`examples/config.yaml`](examples/config.yaml)):

```yaml
providers:
  anthropic:
    env_vars:
      api_key: ANTHROPIC_API_KEY
models:
  claude-haiku:
    provider: anthropic
    model_name: claude-3-5-haiku-20241022
    tags: [fast, cheap]
```

Then resolve a fully-configured client (credentials pulled from the environment,
resolved model name set as the client default):

```python
from llm_manager import LLMFactory

client = LLMFactory.from_model_id("claude-haiku", "examples/config.yaml")
print(client.generate("Summarize reflection prompting in one sentence.").text)
```

## Reflection prompting

Iteratively critique and improve a response.

```
User Query → Initial Generation → Reflection 1 → Reflection 2 → … → Final Output
```

```python
from llm_manager import LLMFactory, ReflectiveLLMManager

client = LLMFactory.get_client(provider_name="openai", api_key="sk-...")
manager = ReflectiveLLMManager(llm_client=client)

result = manager.reflect(
    user_query="Why is the sky blue during the day?",
    reflection_strategy="self_critique",
    num_iterations=3,
    model="gpt-4o-mini",
)

print(result.final_response)
for step in result.iterations:
    print(step["iteration"], step["response"])
```

Available strategies: `self_critique`, `alternative_generation`,
`confidence_assessment`, `verification`, `adversarial`.

## Error handling

Raw SDK errors are mapped to a provider-agnostic hierarchy, all subclassing
`LLMProviderError`:

```python
from llm_manager import LLMProviderError, RateLimitError, AuthenticationError

try:
    client.generate("hello", model="gpt-4o-mini")
except RateLimitError:
    ...        # back off
except AuthenticationError:
    ...        # bad/missing credentials
except LLMProviderError:
    ...        # any other provider failure
```

Also available: `APIConnectionError`, `TokenLimitError`, `InvalidRequestError`,
`ProviderUnavailableError`, `UnknownProviderError`.

## Supported providers

| Provider     | Class            | Extra        | Notes                                   |
|:-------------|:-----------------|:-------------|:----------------------------------------|
| OpenAI       | `OpenAIClient`   | `openai`     | Chat completions                        |
| Anthropic    | `AnthropicClient`| `anthropic`  | Claude Messages API                     |
| AWS Bedrock  | `BedrockClient`  | `bedrock`    | Converse API via `boto3`                |
| Ollama       | `OllamaClient`   | `ollama`     | Local/remote, OpenAI-compatible API     |
| Google Gemini| `GeminiClient`   | `gemini`     | Uses the modern `google-genai` SDK      |

## Adding a new provider

1. Subclass `BaseLLMClient` and implement `_complete` (and optionally `_stream`).
2. Build the request from the shared `GenerationParams`.
3. Register it in `LLMFactory._PROVIDERS`.

```python
from llm_manager.base import BaseLLMClient, GenerationParams
from llm_manager.utils import LLMResponse, normalize_usage
from llm_manager.exceptions import classify_error

class MyProviderClient(BaseLLMClient):
    _provider = "myprovider"

    def __init__(self, api_key: str, system_prompt: str = "You are a helpful assistant"):
        super().__init__(system_prompt=system_prompt)
        self._api_key = api_key

    def _complete(self, prompt: str, params: GenerationParams) -> LLMResponse:
        try:
            raw = ...  # call the SDK using params.model, params.temperature, ...
        except Exception as exc:
            raise classify_error(self._provider, exc) from exc
        return LLMResponse(text=..., usage=normalize_usage(..., provider=self._provider))
```

Rate limiting, retries, and streaming dispatch are handled for you by
`BaseLLMClient.generate`.

## Development

```bash
git clone https://github.com/drimal/llm_manager.git
cd llm_manager
uv sync                     # or: pip install -e '.[all]' and the dev tools

uv run pytest               # tests
uv run ruff check .         # lint
uv run mypy                 # type-check
```

Runnable examples live in [`examples/`](examples/).

## Roadmap

- Async (`agenerate`) support
- Per-chunk streaming usage metadata
- Expanded model registry tooling / CLI

## Contributing

Contributions are welcome — please open an issue or pull request. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
