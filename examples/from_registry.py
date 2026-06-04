"""Create a client from a friendly model id defined in a YAML registry.

The registry (see ``examples/config.yaml``) maps a model id like ``claude-haiku``
to a provider, a concrete model name, and the environment variables that supply
that provider's credentials. ``from_model_id`` resolves all of that for you.

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/from_registry.py
"""

from __future__ import annotations

from pathlib import Path

from llm_manager import LLMFactory

CONFIG = Path(__file__).with_name("config.yaml")


def main() -> None:
    # The resolved model name becomes the client's default, so generate() needs
    # no explicit model.
    client = LLMFactory.from_model_id("claude-haiku", CONFIG)
    response = client.generate("Give me a one-sentence summary of reflection prompting.")
    print(response.text)
    print("usage:", response.usage)


if __name__ == "__main__":
    main()
