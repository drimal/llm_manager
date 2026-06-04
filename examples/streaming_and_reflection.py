"""Streaming, rate-limiting, and the reflection loop.

    export OPENAI_API_KEY=sk-...
    python examples/streaming_and_reflection.py
"""

from __future__ import annotations

import os

from llm_manager import LLMFactory, ReflectiveLLMManager


def main() -> None:
    client = LLMFactory.get_client(
        provider_name="openai",
        api_key=os.environ["OPENAI_API_KEY"],
        system_prompt="You are a concise assistant.",
    )

    # Streaming: generate(stream=True) returns an iterator of text chunks.
    # rate_limit accepts a {"calls", "period"} dict or a RateLimiter instance.
    print("--- streaming ---")
    for chunk in client.generate(
        "List three uses of vector databases.",
        model="gpt-4o-mini",
        stream=True,
        rate_limit={"calls": 30, "period": 60},
    ):
        print(chunk, end="", flush=True)
    print()

    # Reflection: iteratively critique and improve an answer.
    print("\n--- reflection ---")
    manager = ReflectiveLLMManager(llm_client=client)
    result = manager.reflect(
        user_query="What is the bias-variance tradeoff?",
        reflection_strategy="self_critique",
        num_iterations=2,
        model="gpt-4o-mini",
    )
    print(result.final_response)
    print("total tokens:", result.total_tokens)


if __name__ == "__main__":
    main()
