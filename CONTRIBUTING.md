# Contributing to llm-manager

Thanks for your interest in improving llm-manager!

## Development setup

This project uses [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/drimal/llm_manager.git
cd llm_manager
uv sync            # creates .venv and installs the project + dev dependencies
```

## Checks

All of these must pass before a PR is merged (CI enforces them):

```bash
uv run ruff check .     # lint
uv run ruff format --check .
uv run mypy             # type-check
uv run pytest           # tests
```

To auto-fix lint and formatting:

```bash
uv run ruff check --fix .
uv run ruff format .
```

## Guidelines

- Keep the public API consistent across providers. New providers subclass
  `BaseLLMClient` and implement `_complete` (and optionally `_stream`); they get
  rate limiting, retries, and streaming dispatch for free.
- Map raw SDK exceptions to the shared hierarchy via
  `llm_manager.exceptions.classify_error`.
- Add tests for new behavior. Provider SDK calls should be mocked so the suite
  runs without network access or credentials.
- Provider SDKs are **optional extras** — never import them at module top level
  in a way that breaks `import llm_manager` when the SDK is absent.
- Update `CHANGELOG.md` under the `Unreleased` section.

## Releasing

Releases are published to PyPI automatically when a `v*` tag is pushed (see
`.github/workflows/publish.yml`). To cut a release:

1. Bump `__version__` in `src/llm_manager/__init__.py`.
2. Move the `Unreleased` notes in `CHANGELOG.md` under the new version.
3. Tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
