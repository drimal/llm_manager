# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Optional per-provider dependency extras: `openai`, `anthropic`, `bedrock`,
  `gemini`, `ollama`, and `all`. The core install is now lightweight.
- Lazy top-level public API: `from llm_manager import LLMFactory, LLMResponse, ...`.
- `LLMFactory.from_model_id()` to build a configured client from a YAML model
  registry, and `LLMFactory.available_providers()`.
- `BaseLLMClient.generate()` template method centralizing rate limiting, retries,
  and streaming dispatch; providers implement `_complete`/`_stream` hooks over a
  shared `GenerationParams` contract, plus a `default_model`.
- `classify_error()` mapping raw SDK errors to the existing exception hierarchy.
- Runnable examples under `examples/`.
- CI now runs ruff and mypy across Python 3.11–3.13; a PyPI publish workflow runs
  on `v*` tags via Trusted Publishing.

### Fixed
- Gemini client crashed on every call (`kwargs.get["model"]`); fixed and updated
  to the modern `google-genai` SDK.
- Removed a duplicated client-init block in the Anthropic client.
- Bedrock no longer sends an empty `toolConfig`, which the Converse API rejects.
- `normalize_usage` now computes `total_tokens` as input + output when a provider
  omits it (e.g. Anthropic).

### Changed
- Single-source the package version from `src/llm_manager/__init__.py`.
- Consolidated tooling config into `pyproject.toml`; removed `pytest.ini`.

## [0.1.0]

- Initial provider clients (OpenAI, Anthropic, Bedrock, Ollama, Gemini),
  factory, reflection loop, retry, and rate-limit utilities.
