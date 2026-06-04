"""Provider-agnostic quick start for llm_manager.

Run with, e.g.::

    export OPENAI_API_KEY=sk-...
    python examples/quickstart.py --provider openai --question "Why is the sky blue?"

Install the provider extra you need first, e.g. ``pip install 'llm-manager[openai]'``.
"""

from __future__ import annotations

import argparse
import logging
import os

from llm_manager import LLMFactory
from llm_manager.exceptions import UnknownProviderError

logging.basicConfig(level=logging.INFO)

# Per-provider constructor arguments sourced from the environment, plus a
# sensible default model for each.
PROVIDER_SETUP = {
    "openai": (lambda: {"api_key": os.environ["OPENAI_API_KEY"]}, "gpt-4o-mini"),
    "anthropic": (
        lambda: {"api_key": os.environ["ANTHROPIC_API_KEY"]},
        "claude-3-5-haiku-20241022",
    ),
    "ollama": (
        lambda: {"base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")},
        "llama3",
    ),
    "gemini": (lambda: {"api_key": os.environ["GOOGLE_API_KEY"]}, "gemini-1.5-flash"),
    "bedrock": (
        lambda: {
            "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
            "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
            "region_name": os.environ.get("AWS_REGION", "us-east-1"),
        },
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--provider", default="openai", choices=sorted(PROVIDER_SETUP))
    parser.add_argument("-q", "--question", default="Why is the sky blue?")
    parser.add_argument("-m", "--model", default=None, help="Override the default model.")
    args = parser.parse_args()

    if args.provider not in PROVIDER_SETUP:
        raise UnknownProviderError(args.provider)

    build_kwargs, default_model = PROVIDER_SETUP[args.provider]
    client = LLMFactory.get_client(provider_name=args.provider, **build_kwargs())

    response = client.generate(args.question, model=args.model or default_model)
    print(response.text)
    print("usage:", response.usage)


if __name__ == "__main__":
    main()
